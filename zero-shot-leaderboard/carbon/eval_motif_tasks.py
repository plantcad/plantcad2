#!/usr/bin/env python3
"""Carbon-native zero-shot scoring for PlantCAD2 motif recovery/core tasks."""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET = "plantcad/PlantCAD2_zero_shot_tasks"
BASES = ("A", "C", "G", "T")
COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

TASK_SPECS = {
    "tis": {"positions": [4094, 4095, 4096], "motif_len": 3},
    "tts": {"positions": [4094, 4095, 4096], "motif_len": 3},
    "donor": {"positions": [4095, 4096], "motif_len": 2},
    "acceptor": {"positions": [4095, 4096], "motif_len": 2},
}


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1].upper()


def candidates(motif_len: int) -> list[str]:
    return ["".join(parts) for parts in itertools.product(BASES, repeat=motif_len)]


def infer_spec(task: str) -> dict:
    for prefix, spec in TASK_SPECS.items():
        if task.startswith(prefix + "_"):
            return spec
    raise ValueError(f"Cannot infer motif positions for task: {task}")


def replace_motif(seq: str, positions: list[int], motif: str) -> str:
    chars = list(seq.upper())
    for pos, base in zip(positions, motif):
        chars[pos] = base
    return "".join(chars)


def anchored_prefix(seq: str, motif_start: int, motif_len: int, context_bp: int, k: int) -> tuple[str, str, int]:
    seq = seq.upper()
    context_bp = (context_bp // k) * k
    prefix_start = max(0, motif_start - context_bp)
    prefix_start = motif_start - ((motif_start - prefix_start) // k) * k
    target_end = motif_start + k
    if target_end > len(seq):
        raise ValueError(f"target token [{motif_start}, {target_end}) exceeds sequence length {len(seq)}")
    return seq[prefix_start:target_end], seq[motif_start : motif_start + motif_len], prefix_start


def anchored_context(seq: str, motif_start: int, context_bp: int, k: int) -> str:
    seq = seq.upper()
    context_bp = (context_bp // k) * k
    prefix_start = max(0, motif_start - context_bp)
    prefix_start = motif_start - ((motif_start - prefix_start) // k) * k
    return seq[prefix_start:motif_start]


@torch.inference_mode()
def score_prefix_batch(model, tokenizer, prefixes: list[str], device: str) -> np.ndarray:
    texts = [f"<dna>{seq}" for seq in prefixes]
    enc = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    body = getattr(model, "model", None) or getattr(model, "transformer", None)
    if body is None:
        raise RuntimeError("Model body not found at .model or .transformer")

    hidden = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    lengths = attention_mask.sum(dim=1)
    batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
    prev_hidden = hidden[batch_idx, lengths - 2, :]
    logits = model.lm_head(prev_hidden)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    target_ids = input_ids[batch_idx, lengths - 1].to(log_probs.device)
    return log_probs.gather(1, target_ids[:, None]).squeeze(1).float().cpu().numpy()


@torch.inference_mode()
def next_token_log_probs_batch(model, tokenizer, contexts: list[str], device: str) -> np.ndarray:
    texts = [f"<dna>{seq}" for seq in contexts]
    enc = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    body = getattr(model, "model", None) or getattr(model, "transformer", None)
    if body is None:
        raise RuntimeError("Model body not found at .model or .transformer")

    hidden = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    lengths = attention_mask.sum(dim=1)
    batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
    prev_hidden = hidden[batch_idx, lengths - 1, :]
    logits = model.lm_head(prev_hidden)
    return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()


def score_many_prefixes(model, tokenizer, prefixes: list[str], micro_batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    out = np.empty(len(prefixes), dtype=np.float32)
    for start in range(0, len(prefixes), micro_batch_size):
        end = min(start + micro_batch_size, len(prefixes))
        out[start:end] = score_prefix_batch(model, tokenizer, prefixes[start:end], device)
    return out


def next_token_log_probs_many(model, tokenizer, contexts: list[str], micro_batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(contexts), micro_batch_size):
        end = min(start + micro_batch_size, len(contexts))
        chunks.append(next_token_log_probs_batch(model, tokenizer, contexts[start:end], device))
    return np.vstack(chunks)


def motif_score_inputs(seq: str, positions: list[int], motif: str, context_bp: int, k: int) -> tuple[str, str]:
    motif_start = positions[0]
    motif_len = len(positions)
    mutated = replace_motif(seq, positions, motif)
    fwd_prefix, _, _ = anchored_prefix(mutated, motif_start, motif_len, context_bp, k)

    rc_mutated = revcomp(mutated)
    rc_positions = [len(mutated) - 1 - pos for pos in reversed(positions)]
    rc_start = min(rc_positions)
    rc_prefix, _, _ = anchored_prefix(rc_mutated, rc_start, motif_len, context_bp, k)
    return fwd_prefix, rc_prefix


def motif_context_inputs(seq: str, positions: list[int], context_bp: int, k: int) -> tuple[str, str]:
    motif_start = positions[0]
    fwd_context = anchored_context(seq, motif_start, context_bp, k)

    rc_seq = revcomp(seq)
    rc_positions = [len(seq) - 1 - pos for pos in reversed(positions)]
    rc_start = min(rc_positions)
    rc_context = anchored_context(rc_seq, rc_start, context_bp, k)
    return fwd_context, rc_context


def motif_token_ids(tokenizer, motif_len: int, k: int) -> dict[str, np.ndarray]:
    ids_by_motif = {}
    suffix_len = k - motif_len
    for motif in candidates(motif_len):
        ids = []
        for suffix in candidates(suffix_len):
            kmer = motif + suffix
            token_ids = tokenizer(f"<dna>{kmer}", add_special_tokens=False).input_ids
            if len(token_ids) != 2:
                raise RuntimeError(f"Expected DNA tag plus one token for {kmer!r}, got {token_ids}")
            ids.append(token_ids[-1])
        ids_by_motif[motif] = np.array(sorted(set(ids)), dtype=np.int64)
    return ids_by_motif


def score_true_motifs(model, tokenizer, sequences: list[str], positions: list[int], args) -> dict:
    k = getattr(tokenizer, "k", 6)
    scores = np.empty(len(sequences), dtype=np.float32)
    fwd_scores = np.empty(len(sequences), dtype=np.float32)
    rc_scores = np.empty(len(sequences), dtype=np.float32)
    motifs = []

    for start in tqdm(
        range(0, len(sequences), args.batch_size),
        desc="true-motif",
        unit="batch",
        mininterval=args.progress_interval,
    ):
        end = min(start + args.batch_size, len(sequences))
        fwd_prefixes = []
        rc_prefixes = []
        for seq in sequences[start:end]:
            motif = "".join(seq[pos].upper() for pos in positions)
            fwd_prefix, rc_prefix = motif_score_inputs(seq, positions, motif, args.context_bp, k)
            fwd_prefixes.append(fwd_prefix)
            rc_prefixes.append(rc_prefix)
            motifs.append(motif)
        fwd = score_many_prefixes(model, tokenizer, fwd_prefixes, args.micro_batch_size)
        rc = score_many_prefixes(model, tokenizer, rc_prefixes, args.micro_batch_size)
        fwd_scores[start:end] = fwd
        rc_scores[start:end] = rc
        scores[start:end] = (fwd + rc) / 2.0

    return {
        "score": scores,
        "forward_logp": fwd_scores,
        "revcomp_logp": rc_scores,
        "true_motif": motifs,
    }


def score_recovery(model, tokenizer, sequences: list[str], positions: list[int], args) -> dict:
    k = getattr(tokenizer, "k", 6)
    motif_len = len(positions)
    cand = candidates(motif_len)
    ids_by_motif = motif_token_ids(tokenizer, motif_len, k)
    pred = []
    true = []
    best_scores = np.empty(len(sequences), dtype=np.float32)
    true_scores = np.empty(len(sequences), dtype=np.float32)

    for start in tqdm(
        range(0, len(sequences), args.batch_size),
        desc="recovery",
        unit="batch",
        mininterval=args.progress_interval,
    ):
        end = min(start + args.batch_size, len(sequences))
        batch = sequences[start:end]
        fwd_contexts = []
        rc_contexts = []
        for seq in batch:
            true.append("".join(seq[pos].upper() for pos in positions))
            fwd_context, rc_context = motif_context_inputs(seq, positions, args.context_bp, k)
            fwd_contexts.append(fwd_context)
            rc_contexts.append(rc_context)

        fwd_lp = next_token_log_probs_many(model, tokenizer, fwd_contexts, args.micro_batch_size)
        rc_lp = next_token_log_probs_many(model, tokenizer, rc_contexts, args.micro_batch_size)
        scores = np.empty((len(batch), len(cand)), dtype=np.float32)
        for motif_idx, motif in enumerate(cand):
            fwd = np.logaddexp.reduce(fwd_lp[:, ids_by_motif[motif]], axis=1)
            rc_motif = revcomp(motif)
            rc = np.logaddexp.reduce(rc_lp[:, ids_by_motif[rc_motif]], axis=1)
            scores[:, motif_idx] = (fwd + rc) / 2.0

        best_idx = scores.argmax(axis=1)
        best_scores[start:end] = scores[np.arange(len(batch)), best_idx]
        batch_true = true[start:end]
        for row_idx, motif in enumerate(batch_true):
            true_scores[start + row_idx] = scores[row_idx, cand.index(motif)] if motif in cand else np.nan
        pred.extend(cand[i] for i in best_idx)

    pred_arr = np.array(pred)
    true_arr = np.array(true)
    valid = np.array([all(base in BASES for base in motif) for motif in true_arr], dtype=bool)
    motif_accuracy = float((pred_arr[valid] == true_arr[valid]).mean()) if valid.any() else 0.0
    token_correct = 0
    token_total = 0
    for p, t, ok in zip(pred_arr, true_arr, valid):
        if not ok:
            continue
        token_correct += sum(a == b for a, b in zip(p, t))
        token_total += motif_len
    token_accuracy = float(token_correct / token_total) if token_total else 0.0
    return {
        "best_score": best_scores,
        "true_score": true_scores,
        "pred_motif": pred,
        "true_motif": true,
        "token_accuracy": token_accuracy,
        "motif_accuracy": motif_accuracy,
    }


def load_model(args):
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    print(f"device_name={torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    ).to("cuda").eval()
    return model, tokenizer


def eval_core_task(model, tokenizer, task: str, split: str, args) -> dict:
    spec = infer_spec(task)
    ds = load_dataset(args.dataset, task, split=split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    df = ds.to_pandas()
    sequences = df[args.sequence_column].astype(str).tolist()
    labels = df[args.label_column].astype(int).to_numpy()
    t0 = time.time()
    score_data = score_true_motifs(model, tokenizer, sequences, spec["positions"], args)
    elapsed = time.time() - t0
    scores = score_data["score"]
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    out = pd.DataFrame(
        {
            "label": labels,
            "score": scores,
            "forward_logp": score_data["forward_logp"],
            "revcomp_logp": score_data["revcomp_logp"],
            "true_motif": score_data["true_motif"],
        }
    )
    stem = f"{task}_{split}_carbon_core_noncore"
    out_path = args.output_dir / f"{stem}_scores.parquet"
    out.to_parquet(out_path, index=False)
    summary = {
        "task": task,
        "split": split,
        "mode": "core_noncore",
        "model": args.model,
        "num_examples": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df[args.label_column].value_counts().sort_index().items()},
        "positions": spec["positions"],
        "motif_len": spec["motif_len"],
        "score_definition": "0.5 * (forward true-motif 6-mer logp + reverse-complement true-motif 6-mer logp)",
        "context_bp": args.context_bp,
        "auroc": auroc,
        "auprc": auprc,
        "elapsed_seconds": elapsed,
        "output": str(out_path),
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{task}/{split} AUROC={auroc:.6f} AUPRC={auprc:.6f} n={len(df)}")
    return summary


def eval_recovery_task(model, tokenizer, task: str, split: str, args) -> dict:
    spec = infer_spec(task)
    ds = load_dataset(args.dataset, task, split=split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    df = ds.to_pandas()
    sequences = df[args.sequence_column].astype(str).tolist()
    t0 = time.time()
    score_data = score_recovery(model, tokenizer, sequences, spec["positions"], args)
    elapsed = time.time() - t0
    out = pd.DataFrame(
        {
            "best_score": score_data["best_score"],
            "true_score": score_data["true_score"],
            "pred_motif": score_data["pred_motif"],
            "true_motif": score_data["true_motif"],
        }
    )
    stem = f"{task}_{split}_carbon_recovery"
    out_path = args.output_dir / f"{stem}_scores.parquet"
    out.to_parquet(out_path, index=False)
    summary = {
        "task": task,
        "split": split,
        "mode": "recovery",
        "model": args.model,
        "num_examples": int(len(df)),
        "positions": spec["positions"],
        "motif_len": spec["motif_len"],
        "score_definition": "prompt ends before motif; argmax over motif candidates by 0.5 * (forward next-6mer logsumexp over candidate suffixes + reverse-complement next-6mer logsumexp over candidate suffixes)",
        "context_bp": args.context_bp,
        "token_accuracy": score_data["token_accuracy"],
        "motif_accuracy": score_data["motif_accuracy"],
        "elapsed_seconds": elapsed,
        "output": str(out_path),
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{task}/{split} token_accuracy={score_data['token_accuracy']:.6f} "
        f"motif_accuracy={score_data['motif_accuracy']:.6f} n={len(df)}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--mode", choices=["core_noncore", "recovery"], required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["test_maize", "test_tomato"])
    parser.add_argument("--model", default="HuggingFaceBio/Carbon-3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sequence_column", default="sequence")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--context_bp", type=int, default=1020)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=128)
    parser.add_argument("--progress_interval", type=float, default=600.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args)
    summaries = []
    for task in args.tasks:
        for split in args.splits:
            print(f"\n=== {args.mode}: {task}/{split} ===")
            if args.mode == "core_noncore":
                summaries.append(eval_core_task(model, tokenizer, task, split, args))
            else:
                summaries.append(eval_recovery_task(model, tokenizer, task, split, args))
    combined = args.output_dir / f"carbon_{args.mode}_summary.json"
    combined.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nSaved combined summary: {combined}")


if __name__ == "__main__":
    main()
