#!/usr/bin/env python3
"""Carbon-native zero-shot scoring for PlantCAD2 conservation tasks."""

import argparse
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
DEFAULT_TASKS = (
    "conservation_within_poaceae_non_tis",
    "conservation_within_andropogoneae",
    "conservation_within_poaceae_tis",
)
COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1].upper()


def target_prefix(
    seq: str,
    k: int,
    target_base_idx: int | None,
    context_bp: int | None,
) -> tuple[str, int, int, int, str]:
    seq = seq.upper()
    if target_base_idx is None:
        target_base_idx = len(seq) // 2
    target_token_idx = target_base_idx // k
    target_start = target_token_idx * k
    target_end = target_start + k
    if target_end > len(seq):
        raise ValueError(f"target token [{target_start}, {target_end}) exceeds sequence length {len(seq)}")
    if context_bp is None:
        prefix_start = 0
    else:
        context_bp = (context_bp // k) * k
        prefix_start = max(0, target_start - context_bp)
        # Keep the supplied substring in the same non-overlapping 6-mer frame.
        prefix_start = target_start - ((target_start - prefix_start) // k) * k
    local_target_token_idx = (target_start - prefix_start) // k
    return seq[prefix_start:target_end], local_target_token_idx, prefix_start, target_start, seq[target_start:target_end]


def dna_trim(seq: str, k: int) -> str:
    seq = seq.upper()
    return seq[: (len(seq) // k) * k]


@torch.inference_mode()
def score_prefix_batch(model, tokenizer, prefixes: list[str], device: str) -> np.ndarray:
    texts = [f"<dna>{seq}" for seq in prefixes]
    enc = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False)
    enc = enc.to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    body = getattr(model, "model", None) or getattr(model, "transformer", None)
    if body is None:
        raise RuntimeError("Model body not found at .model or .transformer")

    hidden = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    lengths = attention_mask.sum(dim=1)
    prev_positions = lengths - 2
    batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
    prev_hidden = hidden[batch_idx, prev_positions, :]
    logits = model.lm_head(prev_hidden)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    target_ids = input_ids[batch_idx, lengths - 1].to(log_probs.device)
    return log_probs.gather(1, target_ids[:, None]).squeeze(1).float().cpu().numpy()


@torch.inference_mode()
def score_full_ll_batch(
    model,
    tokenizer,
    sequences: list[str],
    device: str,
    logit_chunk_size: int,
) -> np.ndarray:
    texts = [f"<dna>{seq}" for seq in sequences]
    enc = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False)
    enc = enc.to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    body = getattr(model, "model", None) or getattr(model, "transformer", None)
    if body is None:
        raise RuntimeError("Model body not found at .model or .transformer")

    hidden = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].bool()
    prev_hidden = hidden[:, :-1, :]
    sums = torch.zeros(input_ids.shape[0], dtype=torch.float32, device=device)
    counts = target_mask.sum(dim=1).clamp(min=1).to(torch.float32)

    for start in range(0, targets.shape[1], logit_chunk_size):
        end = min(start + logit_chunk_size, targets.shape[1])
        chunk_hidden = prev_hidden[:, start:end, :]
        chunk_logits = model.lm_head(chunk_hidden)
        chunk_log_probs = torch.log_softmax(chunk_logits.float(), dim=-1)
        chunk_targets = targets[:, start:end].to(chunk_log_probs.device)
        chunk_lp = chunk_log_probs.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
        chunk_mask = target_mask[:, start:end].to(chunk_lp.device)
        sums += (chunk_lp * chunk_mask).sum(dim=1)

    return (sums / counts).float().cpu().numpy()


def score_sequences(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    target_base_idx: int | None,
    context_bp: int | None,
    progress_interval: float,
) -> dict:
    device = next(model.parameters()).device
    k = getattr(tokenizer, "k", 6)
    scores = np.empty(len(sequences), dtype=np.float32)
    fwd_scores = np.empty(len(sequences), dtype=np.float32)
    rc_scores = np.empty(len(sequences), dtype=np.float32)
    context_starts = np.empty(len(sequences), dtype=np.int32)
    target_starts = np.empty(len(sequences), dtype=np.int32)
    target_kmers = []
    rc_target_kmers = []

    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc="center6mer",
        unit="batch",
        mininterval=progress_interval,
    ):
        end = min(start + batch_size, len(sequences))
        batch = sequences[start:end]
        fwd_prefixes = []
        rc_prefixes = []
        for seq in batch:
            fwd_prefix, _, context_start, target_start, target_kmer = target_prefix(
                seq, k, target_base_idx, context_bp
            )
            rc_prefix, _, _, _, rc_target_kmer = target_prefix(
                revcomp(seq), k, target_base_idx, context_bp
            )
            fwd_prefixes.append(fwd_prefix)
            rc_prefixes.append(rc_prefix)
            row_idx = start + len(fwd_prefixes) - 1
            context_starts[row_idx] = context_start
            target_starts[row_idx] = target_start
            target_kmers.append(target_kmer)
            rc_target_kmers.append(rc_target_kmer)

        fwd = score_prefix_batch(model, tokenizer, fwd_prefixes, device)
        rc = score_prefix_batch(model, tokenizer, rc_prefixes, device)
        fwd_scores[start:end] = fwd
        rc_scores[start:end] = rc
        scores[start:end] = (fwd + rc) / 2.0

    return {
        "score": scores,
        "forward_logp": fwd_scores,
        "revcomp_logp": rc_scores,
        "context_start": context_starts,
        "target_start": target_starts,
        "target_kmer": target_kmers,
        "revcomp_target_kmer": rc_target_kmers,
    }


def score_full_sequences(
    model,
    tokenizer,
    sequences: list[str],
    batch_size: int,
    logit_chunk_size: int,
    progress_interval: float,
) -> dict:
    device = next(model.parameters()).device
    k = getattr(tokenizer, "k", 6)
    scores = np.empty(len(sequences), dtype=np.float32)
    fwd_scores = np.empty(len(sequences), dtype=np.float32)
    rc_scores = np.empty(len(sequences), dtype=np.float32)
    scored_lengths = np.empty(len(sequences), dtype=np.int32)

    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc="full-ll",
        unit="batch",
        mininterval=progress_interval,
    ):
        end = min(start + batch_size, len(sequences))
        batch = [dna_trim(seq, k) for seq in sequences[start:end]]
        batch_rc = [revcomp(seq) for seq in batch]
        fwd = score_full_ll_batch(model, tokenizer, batch, device, logit_chunk_size)
        rc = score_full_ll_batch(model, tokenizer, batch_rc, device, logit_chunk_size)
        fwd_scores[start:end] = fwd
        rc_scores[start:end] = rc
        scores[start:end] = (fwd + rc) / 2.0
        scored_lengths[start:end] = [len(seq) for seq in batch]

    return {
        "full_mean_ll": scores,
        "full_forward_mean_ll": fwd_scores,
        "full_revcomp_mean_ll": rc_scores,
        "full_scored_bp": scored_lengths,
    }


def metric_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def eval_task(args, model, tokenizer, task: str) -> dict:
    ds = load_dataset(args.dataset, task, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    df = ds.to_pandas()
    sequences = df[args.sequence_column].astype(str).tolist()
    labels = df[args.label_column].astype(int).to_numpy()

    t0 = time.time()
    context_bp = None if args.full_left_context else args.context_bp
    score_data = {}
    score_metrics = {}
    if "center6mer" in args.score_modes:
        center_data = score_sequences(
            model,
            tokenizer,
            sequences,
            args.batch_size,
            args.target_base_idx,
            context_bp,
            args.progress_interval,
        )
        score_data.update(center_data)
        score_metrics["center6mer"] = metric_summary(labels, center_data["score"])
    if "full_mean_ll" in args.score_modes:
        full_data = score_full_sequences(
            model,
            tokenizer,
            sequences,
            args.full_batch_size,
            args.logit_chunk_size,
            args.progress_interval,
        )
        score_data.update(full_data)
        score_metrics["full_mean_ll"] = metric_summary(labels, full_data["full_mean_ll"])
    elapsed = time.time() - t0

    label_counts = {str(k): int(v) for k, v in df[args.label_column].value_counts().sort_index().items()}

    out_columns = {"label": labels}
    out_columns.update(score_data)
    out_df = pd.DataFrame(out_columns)
    out_path = args.output_dir / f"{task}_carbon_scores.parquet"
    out_df.to_parquet(out_path, index=False)

    summary = {
        "task": task,
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "num_examples": int(len(df)),
        "label_counts": label_counts,
        "score_modes": args.score_modes,
        "score_definitions": {
            "center6mer": "0.5 * (forward center-6mer logp + reverse-complement center-6mer logp)",
            "full_mean_ll": "0.5 * (forward mean teacher-forced logp over full sequence + reverse-complement mean teacher-forced logp over full sequence)",
        },
        "context_bp": context_bp,
        "target_base_idx": args.target_base_idx,
        "target_start_mode": "floor(center_base / 6) * 6 when target_base_idx is null",
        "metrics": score_metrics,
        "elapsed_seconds": elapsed,
        "output": str(out_path),
    }
    summary_path = args.output_dir / f"{task}_carbon_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="HuggingFaceBio/Carbon-3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results/plantcad2_carbon_conservation"))
    parser.add_argument("--score_modes", nargs="+", choices=["center6mer", "full_mean_ll"], default=["center6mer", "full_mean_ll"])
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for center6mer scoring")
    parser.add_argument("--full_batch_size", type=int, default=2, help="Batch size for full_mean_ll scoring")
    parser.add_argument("--logit_chunk_size", type=int, default=64)
    parser.add_argument("--progress_interval", type=float, default=300.0)
    parser.add_argument("--context_bp", type=int, default=1020)
    parser.add_argument("--full_left_context", action="store_true")
    parser.add_argument("--target_base_idx", type=int, default=None)
    parser.add_argument("--sequence_column", default="sequence")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    print(f"device_name={torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=dtype,
    ).to("cuda").eval()

    summaries = []
    for task in args.tasks:
        print(f"\n=== {task} ===")
        summary = eval_task(args, model, tokenizer, task)
        summaries.append(summary)
        metric_bits = [
            f"{mode}: AUROC={values['auroc']:.6f} AUPRC={values['auprc']:.6f}"
            for mode, values in summary["metrics"].items()
        ]
        print(f"{' | '.join(metric_bits)} n={summary['num_examples']}")

    combined_path = args.output_dir / "carbon_conservation_summary.json"
    combined_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nSaved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
