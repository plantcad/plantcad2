#!/usr/bin/env python3
"""
Zero-shot evaluation utilities for causal/auto-regressive models.
"""

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

MANIFEST_SCHEMA_VERSION = 1

import fire
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


logger = logging.getLogger(__name__)

NUCLEOTIDES = ("A", "C", "G", "T")
NUCLEOTIDES_LOWER = tuple(n for n in NUCLEOTIDES)
NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def _require_cuda(device: str) -> str:
    if not (torch.cuda.is_available() and device.startswith("cuda")):
        raise RuntimeError(
            "CUDA is required for zero-shot evaluation; CPU is not supported. "
            "Set -device cuda:0 and ensure a CUDA GPU is available."
        )
    return device


def _optimal_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major >= 8:
        return torch.bfloat16
    if major >= 6:
        return torch.float16
    return torch.float32


def _is_evo2_model(model_name: str) -> bool:
    """Detect Evo2 checkpoints (e.g. 'evo2_20b' or 'arcinstitute/evo2_20b')."""
    return model_name.split("/")[-1].lower().startswith("evo2")


class _Evo2TokenizerShim:
    """Adapt Evo2's byte-level CharLevelTokenizer to the HF-tokenizer calls used here.

    Evo2 tokenizes each character to its raw byte value (ASCII), so 'A','C','G','T'
    map to 65,67,71,84 -- the same indices used to slice the model's logits.
    """

    def __init__(self, evo2_tokenizer):
        self._tok = evo2_tokenizer

    def get_vocab(self) -> Dict[str, int]:
        return {chr(i): i for i in range(256)}

    def _tokenize_one(self, seq: str) -> List[int]:
        return list(self._tok.tokenize(seq))

    def __call__(self, seq, return_tensors=None, **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(seq, (list, tuple, pd.Series)):
            ids = [self._tokenize_one(s) for s in seq]
            lengths = {len(x) for x in ids}
            if len(lengths) != 1:
                raise ValueError(
                    f"Evo2 tokenizer shim requires equal-length sequences per batch; got {sorted(lengths)}"
                )
            arr = torch.tensor(ids, dtype=torch.long)
        else:
            arr = torch.tensor(self._tokenize_one(seq), dtype=torch.long).unsqueeze(0)
        return {"input_ids": arr}


class _Evo2ForCausalLM:
    """Wrap an Evo2 model so it exposes the minimal HF interface this script uses:
    `.to()`, `.eval()`, and `model(input_ids=...).logits`.
    """

    def __init__(self, evo2_model, device: str):
        self.evo2 = evo2_model
        self._device = device

    def to(self, *args, **kwargs):  # Evo2 already lives on GPU; ignore dtype/device moves
        return self

    def eval(self):
        return self

    def __call__(self, input_ids: torch.Tensor = None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        input_ids = input_ids.to(self._device).to(torch.int)
        # Evo2.forward returns (out, embeddings), where `out` is whatever vortex's
        # StripedHyena.forward returned -- currently the tuple (logits, inference_params).
        # This is the `outputs, _ = model(ids); logits = outputs[0]` idiom from the Evo2
        # README; unwrap defensively so a vortex that returns the tensor directly still works.
        out = self.evo2(input_ids)[0]
        if isinstance(out, (tuple, list)):
            out = out[0]
        # logits shape [B, L, vocab]
        return SimpleNamespace(logits=out)


def _load_evo2_model(model_name: str, device: str, use_kernels: bool = False):
    from evo2 import Evo2  # lazy: only required when actually evaluating Evo2
    from evo2.utils import MODEL_NAMES

    name = model_name.split("/")[-1]  # accept 'arcinstitute/evo2_20b' or 'evo2_20b'
    if name not in MODEL_NAMES:
        # The installed evo2 package predates the requested checkpoint. Fail here with
        # the fix rather than letting Evo2.__init__ report only the names it knows.
        raise ValueError(
            f"Evo2 checkpoint {name!r} is not in the installed evo2 package, which offers: "
            f"{', '.join(MODEL_NAMES)}.\n"
            "Upgrade it:  pip install -U 'evo2 @ git+https://github.com/ArcInstitute/evo2.git'\n"
            "The 20b/40b checkpoints additionally need transformer_engine (FP8 input "
            "projections); only the 7b models run without it."
        )
    logger.warning(
        "Loading Evo2 model %r via the evo2 package (HF AutoModelForCausalLM path bypassed).",
        name,
    )
    evo2_model = Evo2(name, use_kernels=use_kernels)
    return _Evo2ForCausalLM(evo2_model, device), _Evo2TokenizerShim(evo2_model.tokenizer)


def _load_model(
    model_name: str, device: str, revision: str = "main", use_kernels: bool = False
):
    if _is_evo2_model(model_name):
        return _load_evo2_model(model_name, device, use_kernels=use_kernels)
    dtype = _optimal_dtype()
    logger.warning(
        "Loading causal model with dtype %s (no auto-fallback). Incompatible weights or kernels may error.",
        dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype, revision=revision
    )
    model.to(dtype)
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, revision=revision
    )
    model.to(device)
    model.eval()
    return model, tok


class SequenceDataset(Dataset):
    def __init__(self, sequences: pd.Series, tokenizer):
        self.sequences = sequences.reset_index(drop=True)
        self.tok = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i: int):
        seq = self.sequences.iloc[i]
        enc = self.tok(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"input_ids": enc["input_ids"]}


def _causal_probs_at_positions(
    model,
    tokenizer,
    loader: DataLoader,
    device: str,
    positions: List[int],
    desc: str,
) -> np.ndarray:
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    if any(p <= 0 for p in positions):
        raise ValueError(
            "Causal scoring requires target positions > 0 (position 0 has no left context)."
        )

    all_probs = []
    for batch in tqdm(loader, desc=desc):
        input_ids = batch["input_ids"].to(device).squeeze(1)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        # logits[:, t-1] predicts token at position t
        next_logits = logits[:, :-1, :][:, :, idxs]
        next_probs = torch.softmax(next_logits.float(), dim=-1).cpu().numpy()
        for b in range(next_probs.shape[0]):
            for pos in positions:
                all_probs.append(next_probs[b, pos - 1, :])
    return np.asarray(all_probs)


def _sequence_loglik(
    model,
    tokenizer,
    loader,
    device,
    positions,
    desc,
) -> torch.Tensor:
    """
    Return per-sequence autoregressive log-likelihood:
      sum_{t=1..L-1} log p(x_t | x_<t).
    input_ids shape: [B, L]
    """
    all_probs = []
    for batch in tqdm(loader, desc=desc):
        input_ids = batch["input_ids"].to(device).squeeze(1)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # [B, L-1, V]
        target = input_ids[:, 1:].unsqueeze(-1)  # [B, L-1]
        token_logp = torch.gather(log_probs, dim=-1, index=target).squeeze(
            -1
        )  # [B, L-1]
        batch_scores = token_logp.mean(
            dim=1
        ).cpu()  # NOTE: mean to match up with evo, but should be sum
        all_probs.append(batch_scores)
    all_probs = (
        torch.cat(all_probs, dim=0).unsqueeze(-1).repeat(len(positions), 4).numpy()
    )
    return np.asarray(all_probs)


def _reverse_complement(seq: str) -> str:
    seq_u = seq.upper()
    invalid = sorted({b for b in seq_u if b not in COMPLEMENT})
    if invalid:
        raise ValueError(f"Unexpected token(s) for complement transform: {invalid}")
    return "".join(COMPLEMENT[b] for b in reversed(seq_u))


def _complement_same_order(seq: str) -> str:
    seq_u = seq.upper()
    invalid = sorted({b for b in seq_u if b not in COMPLEMENT})
    if invalid:
        raise ValueError(f"Unexpected token(s) for complement transform: {invalid}")
    return "".join(COMPLEMENT[b] for b in seq_u)


def _transform_sequences_and_positions(
    sequences: pd.Series,
    positions: List[int],
    context_mode: str,
):
    seqs = sequences.astype(str).reset_index(drop=True)
    if context_mode == "left":
        return seqs, positions
    if context_mode == "left_complement":
        return seqs.apply(_complement_same_order), positions

    lengths = seqs.str.len().to_numpy()
    if len(lengths) == 0:
        return seqs, positions
    if len(set(lengths.tolist())) != 1:
        raise ValueError(
            "All sequences must have equal length for right-context transforms."
        )
    seq_len = int(lengths[0])

    mapped = [seq_len - 1 - p for p in positions]
    mapped = sorted(mapped)

    if context_mode == "right_reverse":
        return seqs.str[::-1], mapped
    if context_mode == "right_reverse_complement":
        return seqs.apply(_reverse_complement), mapped
    raise ValueError(
        "context_mode must be one of: left, left_complement, right_reverse, right_reverse_complement"
    )


def _beam_decode_positions_chunk(
    sequences: pd.Series,
    tokenizer,
    model,
    device: str,
    positions: List[int],
    beam_width: int,
) -> np.ndarray:
    """
    Decode highest-probability motif jointly with beam search over exact positions.
    Returns array of predicted motif tokens with shape [N, motif_len].
    """
    if not positions:
        raise ValueError("positions cannot be empty")
    if any(p <= 0 for p in positions):
        raise ValueError(
            "Beam motif decoding requires all positions > 0 for causal scoring."
        )
    if sorted(positions) != positions:
        raise ValueError("positions must be sorted ascending for beam decoding.")

    vocab = tokenizer.get_vocab()
    nuc_ids = [vocab[n] for n in NUCLEOTIDES_LOWER]
    id_to_nuc = {tok_id: nuc for tok_id, nuc in zip(nuc_ids, NUCLEOTIDES)}
    seqs = sequences.astype(str).tolist()
    preds = []

    for seq in tqdm(seqs, desc=f"Beam motif decode (beam={beam_width})"):
        enc = tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        base_ids = enc["input_ids"].to(device)
        beams = [(0.0, {}, [])]  # (logp, replacements, chosen_token_ids)

        for pos in positions:
            new_beams = []
            for cur_logp, repl, chosen in beams:
                cur_ids = base_ids.clone()
                for rp, rid in repl.items():
                    cur_ids[0, rp] = rid

                with torch.inference_mode():
                    logits = model(input_ids=cur_ids).logits
                dist = torch.log_softmax(logits[0, pos - 1, nuc_ids].float(), dim=-1)

                for j, tok_id in enumerate(nuc_ids):
                    next_logp = cur_logp + float(dist[j].item())
                    next_repl = dict(repl)
                    next_repl[pos] = tok_id
                    new_beams.append((next_logp, next_repl, chosen + [tok_id]))

            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        best = beams[0][2]
        preds.append([id_to_nuc[tok] for tok in best])
    return np.asarray(preds)


def _unmasked_causal_probs(
    sequences: pd.Series,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    desc: str = "Inference (causal)",
) -> np.ndarray:
    """
    Per-position probabilities over A,C,G,T aligned to token positions.
    Output shape: [N, L, 4]. Position 0 is set to 0 because it has no left context.
    """
    idxs = [tokenizer.get_vocab()[n] for n in NUCLEOTIDES_LOWER]
    seqs = sequences.astype(str).tolist()
    first_len = None
    all_probs = None
    for i in tqdm(range(0, len(seqs), batch_size), desc=desc):
        batch = seqs[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=False,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"].to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids)
        logits = out.logits[..., idxs]
        next_probs = torch.softmax(logits[:, :-1, :].float(), dim=-1).cpu().numpy()
        cur_len = next_probs.shape[1] + 1

        if first_len is None:
            first_len = cur_len
            all_probs = np.zeros((len(seqs), first_len, 4), dtype=np.float32)
        elif cur_len != first_len:
            raise ValueError(
                f"All sequences must have same length; got {cur_len} vs {first_len}"
            )

        all_probs[i : i + len(batch), 1:, :] = next_probs
    return all_probs


def _compute_true_tokens_from_seq(
    sequences: pd.Series, positions: List[int]
) -> np.ndarray:
    tokens = []
    for seq in sequences.tolist():
        for i in positions:
            tokens.append(seq[i].upper())
    return np.array(tokens)


def _metric_token_accuracy(probs: np.ndarray, true_tokens: np.ndarray) -> float:
    pred_idx = probs.argmax(axis=1)
    nuc = np.array(NUCLEOTIDES)
    pred = nuc[pred_idx]
    valid = np.isin(true_tokens, nuc)
    if not valid.any():
        return 0.0
    return float((pred[valid] == true_tokens[valid]).mean())


def _metric_motif_accuracy(
    probs: np.ndarray, true_tokens: np.ndarray, motif_len: int
) -> float:
    nuc = np.array(NUCLEOTIDES)
    pred = nuc[probs.argmax(axis=1)]
    n = len(true_tokens)
    assert n % motif_len == 0, "total target positions not divisible by motif_len"
    pred_groups = pred.reshape(-1, motif_len)
    true_groups = true_tokens.reshape(-1, motif_len)
    valid_groups = np.all(np.isin(true_groups, nuc), axis=1)
    if not valid_groups.any():
        return 0.0
    return float(
        np.all(pred_groups[valid_groups] == true_groups[valid_groups], axis=1).mean()
    )


def _compute_auroc(
    df: pd.DataFrame, probs: np.ndarray, token_idx: int, seq_col: str
) -> float:
    ref_series = df[seq_col].str[token_idx].str.upper()
    y_true = df["label"].astype(int)
    nuc = list(NUCLEOTIDES)
    ref_map = {b: i for i, b in enumerate(nuc)}
    scores = np.zeros(len(df), dtype=float)
    valid = ref_series.isin(nuc)
    idx = valid[valid].index
    scores[idx] = probs.reshape(len(df), -1)[
        valid.values, [ref_map[b] for b in ref_series[valid]]
    ]
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(auc(fpr, tpr))


def _refprob_scores(
    df: pd.DataFrame, probs: np.ndarray, token_idx: int, seq_col: str
) -> np.ndarray:
    ref_series = df[seq_col].str[token_idx].str.upper()
    nuc = list(NUCLEOTIDES)
    ref_map = {b: i for i, b in enumerate(nuc)}
    scores = np.zeros(len(df), dtype=float)
    valid = ref_series.isin(nuc)
    idx = valid[valid].index
    scores[idx] = probs.reshape(len(df), -1)[
        valid.values, [ref_map[b] for b in ref_series[valid]]
    ]
    return scores


def _avg_trueprob_scores(
    probs: np.ndarray, true_tokens: np.ndarray, motif_len: int
) -> np.ndarray:
    nuc = np.array(NUCLEOTIDES)
    n = len(true_tokens)
    assert n % motif_len == 0, "total target positions not divisible by motif_len"
    idx_map = {b: i for i, b in enumerate(nuc)}
    idxs = np.array([idx_map.get(t, -1) for t in true_tokens], dtype=int)
    row_idx = np.arange(probs.shape[0])
    token_probs = np.zeros(probs.shape[0], dtype=float)
    valid = idxs >= 0
    token_probs[valid] = probs[row_idx[valid], idxs[valid]]
    return token_probs.reshape(-1, motif_len).mean(axis=1)


def _sv_llr_boundary(
    df: pd.DataFrame,
    ref_probs: np.ndarray,
    mut_probs: np.ndarray,
    flanking: int,
) -> np.ndarray:
    base_idx = NUCLEOTIDE_TO_INDEX
    lseq = ref_probs.shape[1]
    center0 = lseq // 2
    mut_left0 = list(range(center0 - flanking, center0))
    mut_right0 = list(range(center0, center0 + flanking))

    scores = np.zeros(len(df), dtype=float)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scoring (SV effect)"):
        left1 = int(row["left"])
        right1 = int(row["right"])

        left_end = left1 - 1
        left_ref = list(range(left_end - (flanking - 1), left_end + 1))
        right_start = right1 + 1
        right_ref = list(range(right_start, right_start + flanking))

        vals = []
        mut_full = row["MutSeq"]
        mut_start0 = mut_left0[0]
        center_seq = mut_full[mut_start0 : mut_start0 + 2 * flanking]

        for k in range(flanking):
            p_ref1 = left_ref[k]
            p_mut0 = mut_left0[k]
            b = center_seq[k].upper()
            if b in base_idx and p_ref1 - 1 > 0 and p_mut0 > 0:
                j = base_idx[b]
                r = ref_probs[i, p_ref1 - 1, j]
                m = mut_probs[i, p_mut0, j]
                vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
            else:
                vals.append(0.0)

            p_ref1 = right_ref[k]
            p_mut0 = mut_right0[k]
            b = center_seq[flanking + k].upper()
            if b in base_idx and p_ref1 - 1 > 0 and p_mut0 > 0:
                j = base_idx[b]
                r = ref_probs[i, p_ref1 - 1, j]
                m = mut_probs[i, p_mut0, j]
                vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
            else:
                vals.append(0.0)
        scores[i] = float(np.mean(vals)) * -1
    return scores


def _write_run_manifest(path: str, payload: Dict[str, Any]) -> None:
    """Write a single-run manifest JSON for later max-over-context aggregation."""
    body = {"schema_version": MANIFEST_SCHEMA_VERSION, **payload}
    with open(path, "w") as f:
        json.dump(body, f, indent=2)


def _collect_manifest_paths(manifest_list_file: str, manifest_json: str) -> List[str]:
    paths: List[str] = []
    if manifest_list_file:
        with open(manifest_list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(line)
    if manifest_json:
        for part in manifest_json.split(","):
            p = part.strip()
            if p:
                paths.append(p)
    if not paths:
        raise ValueError(
            "Provide manifest_list_file (one path per line) and/or manifest_json (comma-separated paths)."
        )
    return paths


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def _manifest_required_keys(m: Dict[str, Any], path: str) -> None:
    required = (
        "schema_version",
        "subcommand",
        "model",
        "revision",
        "context_mode",
        "metrics",
    )
    missing = [k for k in required if k not in m]
    if missing:
        raise KeyError(f"Manifest {path} missing required keys: {missing}")
    if m["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest {path} has schema_version={m['schema_version']!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )


def _primary_metric_for_subcommand(
    subcommand: str, metrics: Dict[str, Any]
) -> Tuple[str, float]:
    """Return (metric_name, value) used for max-over-context selection."""
    if subcommand == "evo_cons":
        if "auroc" not in metrics:
            raise KeyError("evo_cons manifest metrics must include 'auroc'")
        return "auroc", float(metrics["auroc"])
    if subcommand == "motif_acc":
        if "beam_motif_accuracy" in metrics:
            return "beam_motif_accuracy", float(metrics["beam_motif_accuracy"])
        if "motif_accuracy" not in metrics:
            raise KeyError(
                "motif_acc manifest metrics must include 'motif_accuracy' or 'beam_motif_accuracy'"
            )
        return "motif_accuracy", float(metrics["motif_accuracy"])
    if subcommand == "core_noncore":
        if "auroc" not in metrics:
            raise KeyError("core_noncore manifest metrics must include 'auroc'")
        return "auroc", float(metrics["auroc"])
    if subcommand == "sv_effect":
        if "auprc" not in metrics:
            raise KeyError("sv_effect manifest metrics must include 'auprc'")
        return "auprc", float(metrics["auprc"])
    raise ValueError(f"Unknown subcommand for aggregation: {subcommand!r}")


def _aggregation_compare_keys(m: Dict[str, Any]) -> Tuple:
    """Fields that must match across manifests in one aggregation group.

    Omits work_token_idx / work_mask_idx so the same logical run can be compared
    across context_mode (those indices are derived from context transform).
    """
    flags = dict(m.get("flags") or {})
    for omit in ("work_token_idx", "work_mask_idx"):
        flags.pop(omit, None)
    return (
        m.get("subcommand"),
        m.get("repo_id"),
        m.get("task"),
        m.get("split"),
        m.get("model"),
        m.get("revision"),
        json.dumps(flags, sort_keys=True),
    )


def _write_predictions(
    path: str,
    work_df: pd.DataFrame,
    seq_column: str,
    positions: List[int],
    probs: np.ndarray,
    label_column: Optional[str] = None,
) -> None:
    """Long-format predictions: one row per (example, position).

    Columns: example_idx, position, true_token, pred_token, correct, p_A..p_T (and
    label when available). Group by example_idx to recover motif-level calls.
    """
    nuc = np.array(NUCLEOTIDES)
    seqs = work_df[seq_column].astype(str).tolist()
    true_tokens = np.array([seqs[i][p].upper() for i in range(len(seqs)) for p in positions])
    pred_tokens = nuc[probs.argmax(axis=1)]
    out = pd.DataFrame(
        {
            "example_idx": np.repeat(np.arange(len(seqs)), len(positions)),
            "position": np.tile(np.asarray(positions), len(seqs)),
            "true_token": true_tokens,
            "pred_token": pred_tokens,
            "correct": (true_tokens == pred_tokens).astype(int),
        }
    )
    for j, base in enumerate(NUCLEOTIDES):
        out[f"p_{base}"] = probs[:, j]
    if label_column and label_column in work_df.columns:
        out["label"] = work_df[label_column].to_numpy()[out["example_idx"].to_numpy()]
    out.to_csv(path, sep="\t", index=False)
    logger.info("Wrote %d prediction rows to %s", len(out), path)


def _slurm_shard() -> Tuple[int, int]:
    """(rank, world_size) from Slurm/PMI env; (0, 1) when not launched under srun."""

    def _first(names: Sequence[str], default: int) -> int:
        for name in names:
            if name in os.environ:
                return int(os.environ[name])
        return default

    return (
        _first(["PMIX_RANK", "SLURM_PROCID", "PMI_RANK"], 0),
        _first(["PMIX_SIZE", "SLURM_NTASKS", "PMI_SIZE"], 1),
    )


def _partition_indices(n: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Contiguous row range for this rank; remainder spread over the low ranks."""
    chunk, remainder = divmod(n, world_size)
    start = rank * chunk + min(rank, remainder)
    return start, start + chunk + (1 if rank < remainder else 0)


def _shard_path(prefix: str, rank: int) -> str:
    return f"{prefix}.rank{rank}.npy"


def _write_shard(prefix: str, rank: int, arr: np.ndarray) -> str:
    """Write via a temp file + rename so the root rank never sees a partial shard."""
    path = _shard_path(prefix, rank)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
    return path


def _gather_shards(
    prefix: str, world_size: int, timeout_s: float = 7200.0
) -> np.ndarray:
    """Root-rank collect: block until every shard lands, then concat in rank order."""
    paths = [_shard_path(prefix, r) for r in range(world_size)]
    deadline = time.time() + timeout_s
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"Waited {timeout_s:.0f}s for {len(missing)} of {world_size} shard(s); "
                f"first missing: {missing[0]}"
            )
        logger.info("[rank 0] waiting on %d/%d shard(s)", len(missing), world_size)
        time.sleep(10)
    return np.concatenate([np.load(p) for p in paths], axis=0)


def _causal_probs_sharded(
    work_df: pd.DataFrame,
    seq_column: str,
    positions: List[int],
    model_name: str,
    device: str,
    batch_size: int,
    revision: str,
    desc: str,
    shard_prefix: Optional[str],
) -> Optional[np.ndarray]:
    """Score this rank's slice of the rows, then gather on rank 0.

    Returns the full [N * len(positions), 4] array on rank 0 (or on a single-process
    run), and None on the other ranks -- they have written their shard and should
    exit without computing metrics, which need globally pooled scores.
    """
    rank, world_size = _slurm_shard()
    if world_size > 1 and not shard_prefix:
        raise ValueError(
            f"Running under srun with world_size={world_size} but no --shard_prefix; "
            "ranks need a shared path to exchange partial results."
        )

    start, end = _partition_indices(len(work_df), rank, world_size)
    logger.info(
        "[rank %d/%d] scoring rows %d:%d of %d", rank, world_size, start, end, len(work_df)
    )

    if end > start:
        dev = _require_cuda(device)
        model_, tok = _load_model(model_name, dev, revision=revision)
        dataset = SequenceDataset(work_df[seq_column].iloc[start:end], tok)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        probs = _causal_probs_at_positions(
            model_, tok, loader, dev, positions, desc=f"{desc} [rank {rank}]"
        )
    else:
        # More ranks than rows: contribute an empty slice rather than loading the model.
        probs = np.zeros((0, len(NUCLEOTIDES)), dtype=np.float32)

    if world_size == 1:
        return probs

    _write_shard(shard_prefix, rank, probs)
    if rank != 0:
        logger.info("[rank %d] shard written; rank 0 aggregates", rank)
        return None
    return _gather_shards(shard_prefix, world_size)


class ZeroShotEvalCausal:
    def evo_cons(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "",
        device: str = "cuda:0",
        token_idx: int = 255,
        batch_size: int = 128,
        seq_column: str = "sequence",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        manifest_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
        context_mode: str = "left",
        revision: str = "main",
        shard_prefix: Optional[str] = None,
        save_predictions: Optional[str] = None,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        if token_idx <= 0:
            raise ValueError("token_idx must be > 0 for causal scoring.")

        if input_tsv:
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError(
                    "Either input_tsv or both repo_id and task must be provided"
                )
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()

        work_seq, work_pos = _transform_sequences_and_positions(
            df[seq_column], [token_idx], context_mode=context_mode
        )
        work_token_idx = work_pos[0]
        work_df = df.copy()
        work_df[seq_column] = work_seq

        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
        else:
            probs = _causal_probs_sharded(
                work_df,
                seq_column,
                [work_token_idx],
                model,
                device,
                batch_size,
                revision,
                desc=f"Causal probs @ {work_token_idx}",
                shard_prefix=shard_prefix,
            )
            if probs is None:
                return  # non-root shard rank: shard written, rank 0 aggregates
            if save_logits:
                pd.DataFrame(probs, columns=list(NUCLEOTIDES)).to_csv(
                    save_logits, sep="\t", index=False
                )

        if save_predictions:
            _write_predictions(
                save_predictions, work_df, seq_column, [work_token_idx], probs, "label"
            )

        assert probs.shape[0] == len(df), (
            f"Row mismatch: probs={probs.shape[0]} examples={len(df)}"
        )
        roc_auc = _compute_auroc(work_df, probs, work_token_idx, seq_column)
        y_true = df["label"].astype(int).to_numpy()
        pr_scores = _refprob_scores(work_df, probs, work_token_idx, seq_column)
        auprc = float(average_precision_score(y_true, pr_scores))
        print(f"AUROC\t{roc_auc:.6f}")
        print(f"AUPRC\t{auprc:.6f}")
        if metrics_json:
            with open(metrics_json, "w") as f:
                json.dump(
                    {
                        "auroc": roc_auc,
                        "auprc": auprc,
                        "token_idx": token_idx,
                        "context_mode": context_mode,
                    },
                    f,
                    indent=2,
                )

        if manifest_json:
            _write_run_manifest(
                manifest_json,
                {
                    "subcommand": "evo_cons",
                    "repo_id": repo_id or None,
                    "task": task or None,
                    "split": split,
                    "model": model,
                    "revision": revision,
                    "context_mode": context_mode,
                    "data": {"input_tsv": input_tsv},
                    "flags": {
                        "token_idx": token_idx,
                        "work_token_idx": work_token_idx,
                        "seq_column": seq_column,
                        "batch_size": batch_size,
                        "logits_path": logits_path,
                    },
                    "metrics": {"auroc": roc_auc, "auprc": auprc},
                    "paths": {"metrics_json": metrics_json, "save_logits": save_logits},
                },
            )
            logger.info("Wrote run manifest to %s", manifest_json)

    def motif_acc(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "",
        device: str = "cuda:0",
        mask_idx: Sequence[int] = (255, 256, 257),
        motif_len: int = 3,
        batch_size: int = 128,
        seq_column: str = "sequence",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        manifest_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
        decode_mode: str = "independent",
        beam_width: int = 4,
        context_mode: str = "left",
        revision: str = "main",
        shard_prefix: Optional[str] = None,
        save_predictions: Optional[str] = None,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        # Accept both a Fire-parsed sequence (e.g. --mask_idx=4094,4095,4096) and a
        # raw string (e.g. --mask_idx="[4094,4095,4096]"); reference uses [int(x) for x in mask_idx].
        positions = [
            int(str(x).strip("]["))
            for x in (mask_idx.split(",") if isinstance(mask_idx, str) else mask_idx)
        ]
        if any(p <= 0 for p in positions):
            raise ValueError("All mask_idx positions must be > 0 for causal scoring.")
        assert len(positions) == motif_len, "mask_idx count must equal motif_len"

        if input_tsv:
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError(
                    "Either input_tsv or both repo_id and task must be provided"
                )
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()

        work_seq, work_positions = _transform_sequences_and_positions(
            df[seq_column], positions, context_mode=context_mode
        )
        work_df = df.copy()
        work_df[seq_column] = work_seq

        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
        else:
            probs = _causal_probs_sharded(
                work_df,
                seq_column,
                work_positions,
                model,
                device,
                batch_size,
                revision,
                desc=f"Causal probs motif_len={motif_len}",
                shard_prefix=shard_prefix,
            )
            if probs is None:
                return  # non-root shard rank: shard written, rank 0 aggregates
            if save_logits:
                pd.DataFrame(probs, columns=list(NUCLEOTIDES)).to_csv(
                    save_logits, sep="\t", index=False
                )

        if save_predictions:
            _write_predictions(
                save_predictions, work_df, seq_column, work_positions, probs
            )

        expected = len(df) * len(work_positions)
        assert probs.shape[0] == expected, (
            f"Row mismatch: probs={probs.shape[0]} expected={expected}"
        )
        true_tokens = _compute_true_tokens_from_seq(work_df[seq_column], work_positions)
        token_acc = _metric_token_accuracy(probs, true_tokens)
        motif_acc = _metric_motif_accuracy(probs, true_tokens, motif_len)

        if decode_mode not in {"independent", "beam"}:
            raise ValueError("decode_mode must be one of: independent, beam")
        beam_motif_acc: Optional[float] = None
        if decode_mode == "beam":
            dev = _require_cuda(device)
            model_, tok = _load_model(model, dev, revision=revision)
            pred_motifs = _beam_decode_positions_chunk(
                work_df[seq_column],
                tok,
                model_,
                dev,
                positions=work_positions,
                beam_width=beam_width,
            )
            true_motifs = []
            for seq in work_df[seq_column].astype(str).tolist():
                true_motifs.append([seq[p].upper() for p in work_positions])
            true_motifs = np.asarray(true_motifs)
            valid = np.all(np.isin(true_motifs, np.array(NUCLEOTIDES)), axis=1)
            beam_motif_acc = (
                float(np.all(pred_motifs[valid] == true_motifs[valid], axis=1).mean())
                if valid.any()
                else 0.0
            )
            print(f"beam_motif_accuracy\t{beam_motif_acc:.6f}")
        print(f"token_accuracy\t{token_acc:.6f}")
        print(f"motif_accuracy\t{motif_acc:.6f}")
        metrics_for_manifest: Dict[str, Any] = {
            "token_accuracy": token_acc,
            "motif_accuracy": motif_acc,
        }
        if beam_motif_acc is not None:
            metrics_for_manifest["beam_motif_accuracy"] = beam_motif_acc
        if metrics_json:
            with open(metrics_json, "w") as f:
                payload = dict(metrics_for_manifest)
                if decode_mode == "beam":
                    payload["beam_width"] = beam_width
                payload["context_mode"] = context_mode
                json.dump(payload, f, indent=2)
        if manifest_json:
            mf_flags: Dict[str, Any] = {
                "mask_idx": list(positions),
                "work_mask_idx": list(work_positions),
                "motif_len": motif_len,
                "decode_mode": decode_mode,
                "beam_width": beam_width,
                "seq_column": seq_column,
                "batch_size": batch_size,
                "logits_path": logits_path,
            }
            _write_run_manifest(
                manifest_json,
                {
                    "subcommand": "motif_acc",
                    "repo_id": repo_id or None,
                    "task": task or None,
                    "split": split,
                    "model": model,
                    "revision": revision,
                    "context_mode": context_mode,
                    "data": {"input_tsv": input_tsv},
                    "flags": mf_flags,
                    "metrics": metrics_for_manifest,
                    "paths": {"metrics_json": metrics_json, "save_logits": save_logits},
                },
            )
            logger.info("Wrote run manifest to %s", manifest_json)

    def core_noncore(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "",
        device: str = "cuda:0",
        mask_idx: Sequence[int] = (255, 256, 257),
        motif_len: int = 3,
        batch_size: int = 128,
        seq_column: str = "sequence",
        label_column: str = "label",
        save_logits: Optional[str] = None,
        logits_path: Optional[str] = None,
        metrics_json: Optional[str] = None,
        manifest_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
        use_joint_logprob: bool = False,
        context_mode: str = "left",
        revision: str = "main",
        shard_prefix: Optional[str] = None,
        save_predictions: Optional[str] = None,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        # Accept both a Fire-parsed sequence (e.g. --mask_idx=4094,4095,4096) and a
        # raw string (e.g. --mask_idx="[4094,4095,4096]"); reference uses [int(x) for x in mask_idx].
        positions = [
            int(str(x).strip("]["))
            for x in (mask_idx.split(",") if isinstance(mask_idx, str) else mask_idx)
        ]
        if any(p <= 0 for p in positions):
            raise ValueError("All mask_idx positions must be > 0 for causal scoring.")
        assert len(positions) == motif_len, "mask_idx count must equal motif_len"

        if input_tsv:
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError(
                    "Either input_tsv or both repo_id and task must be provided"
                )
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()

        work_seq, work_positions = _transform_sequences_and_positions(
            df[seq_column], positions, context_mode=context_mode
        )
        work_df = df.copy()
        work_df[seq_column] = work_seq

        if logits_path is not None:
            probs = pd.read_csv(logits_path, sep="\t").values
        else:
            probs = _causal_probs_sharded(
                work_df,
                seq_column,
                work_positions,
                model,
                device,
                batch_size,
                revision,
                desc=f"Causal probs (core/non-core) motif_len={motif_len}",
                shard_prefix=shard_prefix,
            )
            if probs is None:
                return  # non-root shard rank: shard written, rank 0 aggregates
            if save_logits:
                pd.DataFrame(probs, columns=["A", "C", "G", "T"]).to_csv(
                    save_logits, sep="\t", index=False
                )

        if save_predictions:
            _write_predictions(
                save_predictions, work_df, seq_column, work_positions, probs, label_column
            )

        expected = len(df) * len(work_positions)
        assert probs.shape[0] == expected, (
            f"Row mismatch: probs={probs.shape[0]} expected={expected}"
        )
        true_tokens = _compute_true_tokens_from_seq(work_df[seq_column], work_positions)
        if use_joint_logprob:
            # Joint motif score under chain rule: sum log p(x_t | x_<t) across motif positions.
            safe = np.clip(probs, 1e-12, 1.0)
            idx_map = {b: i for i, b in enumerate(NUCLEOTIDES)}
            idxs = np.array([idx_map.get(t, -1) for t in true_tokens], dtype=int)
            token_logp = np.zeros(safe.shape[0], dtype=float)
            valid = idxs >= 0
            ridx = np.arange(safe.shape[0])
            token_logp[valid] = np.log(safe[ridx[valid], idxs[valid]])
            scores = token_logp.reshape(-1, motif_len).sum(axis=1)
        else:
            scores = _avg_trueprob_scores(probs, true_tokens, motif_len)
        y_true = df[label_column].astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))
        auprc = float(average_precision_score(y_true, scores))
        print(f"AUROC\t{roc_auc:.6f}")
        print(f"AUPRC\t{auprc:.6f}")
        if metrics_json:
            with open(metrics_json, "w") as f:
                json.dump(
                    {
                        "auroc": roc_auc,
                        "auprc": auprc,
                        "use_joint_logprob": use_joint_logprob,
                        "context_mode": context_mode,
                    },
                    f,
                    indent=2,
                )

        if manifest_json:
            _write_run_manifest(
                manifest_json,
                {
                    "subcommand": "core_noncore",
                    "repo_id": repo_id or None,
                    "task": task or None,
                    "split": split,
                    "model": model,
                    "revision": revision,
                    "context_mode": context_mode,
                    "data": {"input_tsv": input_tsv},
                    "flags": {
                        "mask_idx": list(positions),
                        "work_mask_idx": list(work_positions),
                        "motif_len": motif_len,
                        "label_column": label_column,
                        "seq_column": seq_column,
                        "batch_size": batch_size,
                        "use_joint_logprob": use_joint_logprob,
                        "logits_path": logits_path,
                    },
                    "metrics": {"auroc": roc_auc, "auprc": auprc},
                    "paths": {"metrics_json": metrics_json, "save_logits": save_logits},
                },
            )
            logger.info("Wrote run manifest to %s", manifest_json)

    def sv_effect(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "",
        device: str = "cuda:0",
        batch_size: int = 64,
        flanking: int = 5,
        output: Optional[str] = None,
        save_ref_logits: Optional[str] = None,
        save_mut_logits: Optional[str] = None,
        input_tsv: Optional[str] = None,
        manifest_json: Optional[str] = None,
        context_mode: str = "left",
        revision: str = "main",
    ) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        if input_tsv:
            df = pd.read_csv(input_tsv, sep="\t")
        else:
            if not repo_id or not task:
                raise ValueError(
                    "Either input_tsv or both repo_id and task must be provided"
                )
            ds = load_dataset(repo_id, task)
            df = ds[split].to_pandas()
        required = ["RefSeq", "MutSeq", "left", "right", "label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        dev = _require_cuda(device)
        model_, tok = _load_model(model, dev, revision=revision)
        ref_probs = _unmasked_causal_probs(
            df["RefSeq"], tok, model_, dev, batch_size, desc="Ref (causal)"
        )
        mut_probs = _unmasked_causal_probs(
            df["MutSeq"], tok, model_, dev, batch_size, desc="Mut (causal)"
        )

        if save_ref_logits:
            np.savez_compressed(save_ref_logits, logits=ref_probs)
        if save_mut_logits:
            np.savez_compressed(save_mut_logits, logits=mut_probs)

        scores = _sv_llr_boundary(df, ref_probs, mut_probs, flanking)
        y_true = df["label"].astype(int).to_numpy()
        auprc = float(average_precision_score(y_true, scores))
        print(f"AUPRC\t{auprc:.6f}")

        if output:
            out_df = df.copy()
            out_df["score"] = scores
            out_df = out_df.drop(
                columns=["Left5_Positions", "Right5_Positions"], errors="ignore"
            )
            out_df.to_csv(output, sep="\t", index=False)

        if manifest_json:
            _write_run_manifest(
                manifest_json,
                {
                    "subcommand": "sv_effect",
                    "repo_id": repo_id or None,
                    "task": task or None,
                    "split": split,
                    "model": model,
                    "revision": revision,
                    "context_mode": context_mode,
                    "data": {"input_tsv": input_tsv},
                    "flags": {
                        "batch_size": batch_size,
                        "flanking": flanking,
                        "save_ref_logits": save_ref_logits,
                        "save_mut_logits": save_mut_logits,
                    },
                    "metrics": {"auprc": auprc},
                    "paths": {
                        "output": output,
                        "save_ref_logits": save_ref_logits,
                        "save_mut_logits": save_mut_logits,
                    },
                },
            )
            logger.info("Wrote run manifest to %s", manifest_json)

    def aggregate_context_max(
        self,
        manifest_list_file: str = "",
        manifest_json: str = "",
        expected_context_modes: str = "",
        output_json: str = "",
        strict_expected_modes: bool = False,
    ) -> None:
        """
        Load per-run manifests from separate context-mode runs and select the best
        by primary metric (evo_cons/core_noncore: auroc; motif_acc: beam_motif_accuracy
        if present else motif_accuracy; sv_effect: auprc).

        Provide manifest paths via manifest_list_file (one path per line) and/or
        manifest_json (comma-separated paths). Write summary to output_json when set.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        paths = _collect_manifest_paths(manifest_list_file, manifest_json)
        loaded: List[Tuple[str, Dict[str, Any]]] = []
        for p in paths:
            m = _load_manifest(p)
            _manifest_required_keys(m, p)
            loaded.append((p, m))

        first_key = _aggregation_compare_keys(loaded[0][1])
        for p, m in loaded[1:]:
            if _aggregation_compare_keys(m) != first_key:
                raise ValueError(
                    "Incompatible manifests for aggregation (subcommand/repo/task/split/model/revision/flags must match):\n"
                    f"  {loaded[0][0]!r} vs {p!r}"
                )

        expected_modes: Optional[List[str]] = None
        if expected_context_modes.strip():
            expected_modes = [
                x.strip() for x in expected_context_modes.split(",") if x.strip()
            ]

        seen_modes = {m["context_mode"] for _, m in loaded}
        if len(seen_modes) < len(loaded):
            raise ValueError(
                f"Duplicate context_mode in manifests: {len(loaded)} files but only {len(seen_modes)} distinct modes."
            )
        if expected_modes is not None:
            missing = [x for x in expected_modes if x not in seen_modes]
            if missing:
                msg = f"Missing context_mode(s) in manifests: {missing}; have {sorted(seen_modes)}"
                if strict_expected_modes:
                    raise ValueError(msg)
                logger.warning(msg)

        best_path: Optional[str] = None
        best_mode: Optional[str] = None
        best_metric_name: Optional[str] = None
        best_value: Optional[float] = None
        subcommand = loaded[0][1]["subcommand"]

        per_run: List[Dict[str, Any]] = []
        for p, m in loaded:
            metric_name, value = _primary_metric_for_subcommand(
                subcommand, m["metrics"]
            )
            per_run.append(
                {
                    "manifest_path": p,
                    "context_mode": m["context_mode"],
                    "primary_metric": metric_name,
                    "primary_value": value,
                    "metrics": m["metrics"],
                }
            )
            if best_value is None or value > best_value:
                best_value = value
                best_metric_name = metric_name
                best_mode = m["context_mode"]
                best_path = p

        summary: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "aggregation": "max_over_context",
            "subcommand": subcommand,
            "compare_key": {
                "repo_id": loaded[0][1].get("repo_id"),
                "task": loaded[0][1].get("task"),
                "split": loaded[0][1].get("split"),
                "model": loaded[0][1].get("model"),
                "revision": loaded[0][1].get("revision"),
                "flags": loaded[0][1].get("flags"),
            },
            "primary_metric_rule": {
                "evo_cons": "auroc",
                "motif_acc": "beam_motif_accuracy if present else motif_accuracy",
                "core_noncore": "auroc",
                "sv_effect": "auprc",
            }.get(subcommand, "unknown"),
            "best": {
                "context_mode": best_mode,
                "primary_metric": best_metric_name,
                "primary_value": best_value,
                "manifest_path": best_path,
            },
            "per_context": sorted(per_run, key=lambda x: x["context_mode"]),
            "manifest_paths": paths,
        }

        print(f"BEST_CONTEXT_MODE\t{best_mode}")
        print(f"BEST_PRIMARY_METRIC\t{best_metric_name}\t{best_value:.6f}")
        print(f"BEST_MANIFEST\t{best_path}")

        if output_json:
            with open(output_json, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Wrote aggregation summary to %s", output_json)


def main():
    fire.Fire(ZeroShotEvalCausal)


if __name__ == "__main__":
    main()
