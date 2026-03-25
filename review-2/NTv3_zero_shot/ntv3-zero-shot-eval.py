#!/usr/bin/env python3
"""
Zero-shot evaluation for InstaDeepAI NTv3-series models (6-mer tokenization).

NTv3 uses non-overlapping 6-mer tokenization: nucleotide position P maps to
token index P // 6, at intra-token offset P % 6.

All position arguments (token_idx, mask_idx) accept NUCLEOTIDE positions
(0-based), matching the interface of zero-shot-eval.py, and are converted
to token positions internally.

Metrics computed are the same as zero-shot-eval.py for fair comparison.
"""

import json
import logging
from typing import List, Optional, Sequence

import fire
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

NUCLEOTIDES = ("A", "C", "G", "T")
NUC_TO_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}
KMER_SIZE = 6


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _require_cuda(device: str) -> str:
    if not (torch.cuda.is_available() and device.startswith("cuda")):
        raise RuntimeError(
            "CUDA is required. Set --device cuda:0 and ensure a GPU is available."
        )
    return device


def _load_model(model_name: str, device: str):
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tok


def _build_kmer_nuc_matrix(tokenizer) -> np.ndarray:
    """
    Build a float32 matrix of shape [vocab_size, KMER_SIZE, 4].

    kmer_nuc_mat[token_id, offset, nuc_idx] = 1.0 if the token's k-mer string
    has nucleotide NUCLEOTIDES[nuc_idx] at position `offset`, else 0.0.
    Special (non-k-mer) tokens are all-zero rows.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    mat = np.zeros((vocab_size, KMER_SIZE, 4), dtype=np.float32)
    n_kmers = 0
    for token, idx in vocab.items():
        upper = token.upper()
        if len(upper) == KMER_SIZE and all(c in "ACGT" for c in upper):
            for pos, nuc in enumerate(upper):
                mat[idx, pos, NUC_TO_IDX[nuc]] = 1.0
            n_kmers += 1
    logger.info(f"k-mer vocabulary: {n_kmers} tokens of length {KMER_SIZE} (total vocab {vocab_size})")
    if n_kmers < 100:
        raise RuntimeError(
            f"Only {n_kmers} k-mer tokens found in vocabulary. "
            "This model may not use 6-mer tokenization; use zero-shot-eval.py instead."
        )
    return mat


def _nuc_to_token(nuc_pos: int) -> int:
    """0-based nucleotide position → 0-based token index (assumes no CLS prepended)."""
    return nuc_pos // KMER_SIZE


def _nuc_to_offset(nuc_pos: int) -> int:
    """0-based nucleotide position → offset within its k-mer token."""
    return nuc_pos % KMER_SIZE


def _marginalize_nuc(kmer_probs: np.ndarray, kmer_nuc_mat: np.ndarray, offset: int) -> np.ndarray:
    """
    Marginalize k-mer probabilities to per-nucleotide probabilities at a given offset.

    kmer_probs  : [N, vocab_size]
    kmer_nuc_mat: [vocab_size, KMER_SIZE, 4]
    offset      : position within k-mer (0 .. KMER_SIZE-1)
    Returns     : [N, 4]  (probabilities for A, C, G, T)
    """
    # kmer_nuc_mat[:, offset, :] is [vocab_size, 4]
    return kmer_probs @ kmer_nuc_mat[:, offset, :]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class MaskTokenDataset(Dataset):
    """Mask a fixed set of token positions in every sequence."""

    def __init__(self, sequences: pd.Series, tokenizer, token_positions: List[int]):
        self.sequences = sequences.reset_index(drop=True)
        self.tok = tokenizer
        self.token_positions = sorted(set(token_positions))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i: int):
        seq = self.sequences.iloc[i]
        enc = self.tok(seq, return_tensors="pt", padding=False, add_special_tokens=False)
        input_ids = enc["input_ids"]  # [1, L]
        assert input_ids.size(1) > max(self.token_positions), (
            f"token position {max(self.token_positions)} out of range "
            f"(sequence has {input_ids.size(1)} tokens)"
        )
        input_ids[0, self.token_positions] = self.tok.mask_token_id
        return {"masked_ids": input_ids}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _masked_kmer_probs(
    model, tokenizer, loader: DataLoader, device: str, n_masked: int, desc: str
) -> np.ndarray:
    """
    Forward pass with masked tokens; return k-mer probabilities at masked positions.

    Returns np.ndarray of shape [N * n_masked, vocab_size], where rows are ordered
    as (example_0_mask_0, example_0_mask_1, ..., example_1_mask_0, ...).
    When n_masked == 1, shape is [N, vocab_size].
    """
    vocab_size = len(tokenizer.get_vocab())
    all_probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc):
        cur_ids = batch["masked_ids"].to(device).squeeze(1)  # [B, L]
        with torch.inference_mode():
            logits = model(input_ids=cur_ids).logits  # [B, L, V]
        mask_bool = cur_ids == tokenizer.mask_token_id  # [B, L]
        # Boolean indexing: extracts [B * n_masked, V] in row-major order
        masked_logits = logits[mask_bool]
        probs = torch.softmax(masked_logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)  # [N * n_masked, V]


def _unmasked_kmer_probs(
    sequences: pd.Series,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """
    Unmasked forward pass; return k-mer probabilities at every token position.

    Returns np.ndarray of shape [N, L_tokens, vocab_size].
    All sequences must produce the same token length.
    """
    vocab_size = len(tokenizer.get_vocab())
    seqs = sequences.astype(str).tolist()
    first_len: Optional[int] = None
    all_probs: Optional[np.ndarray] = None
    for i in tqdm(range(0, len(seqs), batch_size), desc=desc):
        batch = seqs[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids)  # [B, L, V]
        probs = torch.softmax(logits.logits.float(), dim=-1).cpu().numpy()
        if first_len is None:
            first_len = probs.shape[1]
            all_probs = np.zeros((len(seqs), first_len, vocab_size), dtype=np.float32)
        elif probs.shape[1] != first_len:
            raise ValueError(
                f"Token length mismatch: expected {first_len}, got {probs.shape[1]}. "
                "All sequences must produce the same number of tokens."
            )
        all_probs[i : i + len(batch)] = probs
    return all_probs  # [N, L_tokens, V]


def _kmer_probs_to_nuc_probs(
    kmer_probs_3d: np.ndarray,
    kmer_nuc_mat: np.ndarray,
) -> np.ndarray:
    """
    Convert [N, L_tokens, vocab_size] k-mer probs to [N, L_tokens * KMER_SIZE, 4]
    nucleotide-level probs by marginalizing each token at each of its 6 offsets.
    """
    N, L_tok, _ = kmer_probs_3d.shape
    nuc_probs = np.zeros((N, L_tok * KMER_SIZE, 4), dtype=np.float32)
    for o in range(KMER_SIZE):
        # kmer_probs_3d @ kmer_nuc_mat[:, o, :] → [N, L_tok, 4]
        nuc_probs[:, o::KMER_SIZE, :] = kmer_probs_3d @ kmer_nuc_mat[:, o, :]
    return nuc_probs


# ---------------------------------------------------------------------------
# SV effect boundary scoring (same logic as zero-shot-eval.py)
# ---------------------------------------------------------------------------

NUCLEOTIDE_TO_INDEX = {b: i for i, b in enumerate(NUCLEOTIDES)}


def _sv_llr_boundary(
    df: pd.DataFrame,
    ref_nuc_probs: np.ndarray,
    mut_nuc_probs: np.ndarray,
    flanking: int,
) -> np.ndarray:
    """
    Mean log(mut/ref) using boundary windows. Works on per-nucleotide probability
    arrays of shape [N, L_nuc, 4] (same formula as in zero-shot-eval.py).
    """
    base_idx = NUCLEOTIDE_TO_INDEX
    L = ref_nuc_probs.shape[1]
    center0 = L // 2
    mut_left0 = list(range(center0 - flanking, center0))
    mut_right0 = list(range(center0, center0 + flanking))

    scores = np.zeros(len(df), dtype=float)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scoring (SV effect)"):
        left1 = int(row["left"])
        right1 = int(row["right"])
        left_ref = list(range(left1 - flanking, left1))
        right_ref = list(range(right1 + 1, right1 + 1 + flanking))

        vals = []
        mut_full = row["MutSeq"]
        center_seq = mut_full[mut_left0[0] : mut_left0[0] + 2 * flanking]

        for k in range(flanking):
            for p_ref1, p_mut0, b in [
                (left_ref[k], mut_left0[k], center_seq[k].upper()),
                (right_ref[k], mut_right0[k], center_seq[flanking + k].upper()),
            ]:
                if b in base_idx:
                    j = base_idx[b]
                    r = ref_nuc_probs[i, p_ref1 - 1, j]
                    m = mut_nuc_probs[i, p_mut0, j]
                    vals.append(float(np.log(max(m, 1e-12) / max(r, 1e-12))))
                else:
                    vals.append(0.0)
        scores[i] = float(np.mean(vals)) * -1
    return scores


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def _load_df(repo_id, task, split, input_tsv):
    if input_tsv:
        logger.info(f"Loading data from local TSV: {input_tsv}")
        return pd.read_csv(input_tsv, sep="\t")
    if not repo_id or not task:
        raise ValueError("Either --input_tsv or both --repo_id and --task must be provided")
    ds = load_dataset(repo_id, task)
    return ds[split].to_pandas()


class ZeroShotEvalNTv3:

    def evo_cons(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "InstaDeepAI/NTv3_650M_pre",
        device: str = "cuda:0",
        token_idx: int = 4095,
        batch_size: int = 16,
        seq_column: str = "sequence",
        metrics_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """
        Evolutionary conservation: mask the k-mer token covering token_idx
        (nucleotide position, 0-based), marginalize to single-nucleotide probs,
        then compute AUROC / AUPRC vs. labels.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        df = _load_df(repo_id, task, split, input_tsv)

        dev = _require_cuda(device)
        model_, tok = _load_model(model, dev)
        kmer_nuc_mat = _build_kmer_nuc_matrix(tok)

        t_pos = _nuc_to_token(token_idx)
        t_off = _nuc_to_offset(token_idx)
        logger.info(f"Nucleotide pos {token_idx} → token {t_pos}, offset {t_off}")

        dataset = MaskTokenDataset(df[seq_column], tok, [t_pos])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        kmer_probs = _masked_kmer_probs(
            model_, tok, loader, dev, n_masked=1, desc=f"Evo cons nuc={token_idx}"
        )  # [N, vocab_size]

        nuc_probs = _marginalize_nuc(kmer_probs, kmer_nuc_mat, t_off)  # [N, 4]

        ref_bases = df[seq_column].str[token_idx].str.upper()
        scores = np.array(
            [nuc_probs[i, NUC_TO_IDX[b]] if b in NUC_TO_IDX else 0.0
             for i, b in enumerate(ref_bases)],
            dtype=float,
        )
        y_true = df["label"].astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = float(auc(fpr, tpr))
        auprc = float(average_precision_score(y_true, scores))
        print(f"AUROC\t{auroc:.6f}")
        print(f"AUPRC\t{auprc:.6f}")
        if metrics_json:
            with open(metrics_json, "w") as f:
                json.dump({"auroc": auroc, "auprc": auprc, "token_idx": token_idx}, f, indent=2)

    def motif_acc(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "InstaDeepAI/NTv3_650M_pre",
        device: str = "cuda:0",
        mask_idx: Sequence[int] = (4094, 4095, 4096),
        motif_len: int = 3,
        batch_size: int = 16,
        seq_column: str = "sequence",
        metrics_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """
        Motif recovery: mask token(s) covering mask_idx (nucleotide positions),
        marginalize to per-nucleotide predictions, report token and motif accuracy.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        df = _load_df(repo_id, task, split, input_tsv)

        nuc_positions = [int(x) for x in mask_idx]
        assert len(nuc_positions) == motif_len, "mask_idx count must equal motif_len"

        # Deduplicated token positions (all motif nucs may share one token)
        token_positions = sorted(set(_nuc_to_token(p) for p in nuc_positions))
        offsets = [_nuc_to_offset(p) for p in nuc_positions]
        t_list_idx = [token_positions.index(_nuc_to_token(p)) for p in nuc_positions]
        logger.info(
            f"Nuc positions {nuc_positions} → token positions {token_positions}, offsets {offsets}"
        )

        dev = _require_cuda(device)
        model_, tok = _load_model(model, dev)
        kmer_nuc_mat = _build_kmer_nuc_matrix(tok)

        n_masked = len(token_positions)
        dataset = MaskTokenDataset(df[seq_column], tok, token_positions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        raw = _masked_kmer_probs(
            model_, tok, loader, dev, n_masked=n_masked, desc="Motif recovery"
        )  # [N * n_masked, V]
        N = len(df)
        kmer_probs = raw.reshape(N, n_masked, -1)  # [N, n_masked, V]

        # Per-nucleotide probs at each motif position
        nuc_probs_per_pos = [
            _marginalize_nuc(kmer_probs[:, ti, :], kmer_nuc_mat, off)
            for ti, off in zip(t_list_idx, offsets)
        ]  # list of [N, 4]

        nuc = np.array(NUCLEOTIDES)
        true_nucs = np.array(
            [[df[seq_column].iloc[n][p].upper() for p in nuc_positions] for n in range(N)]
        )  # [N, motif_len]

        # Token accuracy
        token_hits, token_total = 0, 0
        for p_idx, nuc_p in enumerate(nuc_probs_per_pos):
            pred = nuc[nuc_p.argmax(axis=1)]
            valid = np.isin(true_nucs[:, p_idx], nuc)
            token_hits += int((pred[valid] == true_nucs[valid, p_idx]).sum())
            token_total += int(valid.sum())
        token_accuracy = token_hits / token_total if token_total > 0 else 0.0

        # Motif accuracy
        valid_mask = np.all(np.isin(true_nucs, nuc), axis=1)
        if valid_mask.any():
            preds = np.stack(
                [nuc[nuc_probs_per_pos[p].argmax(axis=1)] for p in range(motif_len)], axis=1
            )  # [N, motif_len]
            motif_accuracy = float(
                np.all(preds[valid_mask] == true_nucs[valid_mask], axis=1).mean()
            )
        else:
            motif_accuracy = 0.0

        print(f"token_accuracy\t{token_accuracy:.6f}")
        print(f"motif_accuracy\t{motif_accuracy:.6f}")
        if metrics_json:
            with open(metrics_json, "w") as f:
                json.dump(
                    {"token_accuracy": token_accuracy, "motif_accuracy": motif_accuracy}, f, indent=2
                )

    def core_noncore(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "InstaDeepAI/NTv3_650M_pre",
        device: str = "cuda:0",
        mask_idx: Sequence[int] = (4094, 4095, 4096),
        motif_len: int = 3,
        batch_size: int = 16,
        seq_column: str = "sequence",
        label_column: str = "label",
        metrics_json: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """
        Core vs non-core classification: score = average probability assigned to
        true nucleotide at each motif position, computed from masked k-mer probs.
        Reports AUROC and AUPRC.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        df = _load_df(repo_id, task, split, input_tsv)

        nuc_positions = [int(x) for x in mask_idx]
        assert len(nuc_positions) == motif_len, "mask_idx count must equal motif_len"

        token_positions = sorted(set(_nuc_to_token(p) for p in nuc_positions))
        offsets = [_nuc_to_offset(p) for p in nuc_positions]
        t_list_idx = [token_positions.index(_nuc_to_token(p)) for p in nuc_positions]

        dev = _require_cuda(device)
        model_, tok = _load_model(model, dev)
        kmer_nuc_mat = _build_kmer_nuc_matrix(tok)

        n_masked = len(token_positions)
        dataset = MaskTokenDataset(df[seq_column], tok, token_positions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        raw = _masked_kmer_probs(
            model_, tok, loader, dev, n_masked=n_masked, desc="Core/non-core"
        )
        N = len(df)
        kmer_probs = raw.reshape(N, n_masked, -1)  # [N, n_masked, V]

        scores = np.zeros(N, dtype=float)
        for p_idx, (ti, off) in enumerate(zip(t_list_idx, offsets)):
            nuc_p = _marginalize_nuc(kmer_probs[:, ti, :], kmer_nuc_mat, off)  # [N, 4]
            for n_idx in range(N):
                b = df[seq_column].iloc[n_idx][nuc_positions[p_idx]].upper()
                if b in NUC_TO_IDX:
                    scores[n_idx] += nuc_p[n_idx, NUC_TO_IDX[b]]
        scores /= motif_len

        y_true = df[label_column].astype(int).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = float(auc(fpr, tpr))
        auprc = float(average_precision_score(y_true, scores))
        print(f"AUROC\t{auroc:.6f}")
        print(f"AUPRC\t{auprc:.6f}")
        if metrics_json:
            with open(metrics_json, "w") as f:
                json.dump({"auroc": auroc, "auprc": auprc}, f, indent=2)

    def sv_effect(
        self,
        repo_id: str = "",
        task: str = "",
        split: str = "valid",
        model: str = "InstaDeepAI/NTv3_650M_pre",
        device: str = "cuda:0",
        batch_size: int = 16,
        flanking: int = 5,
        output: Optional[str] = None,
        input_tsv: Optional[str] = None,
    ) -> None:
        """
        SV effect prediction using boundary-based LLR scoring.
        Uses unmasked inference (pseudo-log-likelihood) on reference and mutant
        sequences, marginalizes k-mer probs to per-nucleotide probs, then applies
        the same boundary LLR formula as zero-shot-eval.py.
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        df = _load_df(repo_id, task, split, input_tsv)
        required = ["RefSeq", "MutSeq", "left", "right", "label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        dev = _require_cuda(device)
        model_, tok = _load_model(model, dev)
        kmer_nuc_mat = _build_kmer_nuc_matrix(tok)

        ref_kmer = _unmasked_kmer_probs(df["RefSeq"], tok, model_, dev, batch_size, "Ref (unmasked)")
        mut_kmer = _unmasked_kmer_probs(df["MutSeq"], tok, model_, dev, batch_size, "Mut (unmasked)")

        ref_nuc = _kmer_probs_to_nuc_probs(ref_kmer, kmer_nuc_mat)
        mut_nuc = _kmer_probs_to_nuc_probs(mut_kmer, kmer_nuc_mat)

        scores = _sv_llr_boundary(df, ref_nuc, mut_nuc, flanking)
        y_true = df["label"].astype(int).to_numpy()
        auprc = float(average_precision_score(y_true, scores))
        print(f"AUPRC\t{auprc:.6f}")

        if output:
            out_df = df.copy()
            out_df["score"] = scores
            out_df = out_df.drop(columns=["Left5_Positions", "Right5_Positions"], errors="ignore")
            out_df.to_csv(output, sep="\t", index=False)


def main():
    fire.Fire(ZeroShotEvalNTv3)


if __name__ == "__main__":
    main()
