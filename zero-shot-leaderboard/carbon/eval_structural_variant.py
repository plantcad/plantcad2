#!/usr/bin/env python3
"""Carbon-native zero-shot scoring for PlantCAD2 structural variant effect prediction."""

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
TASK = "structural_variant_effect_prediction"
COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1].upper()


def dna_trim(seq: str, k: int) -> str:
    seq = seq.upper()
    return seq[: (len(seq) // k) * k]


@torch.inference_mode()
def score_full_ll_batch(model, tokenizer, sequences: list[str], device: str, logit_chunk_size: int) -> np.ndarray:
    texts = [f"<dna>{seq}" for seq in sequences]
    enc = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=False).to(device)
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
        chunk_logits = model.lm_head(prev_hidden[:, start:end, :])
        chunk_log_probs = torch.log_softmax(chunk_logits.float(), dim=-1)
        chunk_targets = targets[:, start:end].to(chunk_log_probs.device)
        chunk_lp = chunk_log_probs.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
        chunk_mask = target_mask[:, start:end].to(chunk_lp.device)
        sums += (chunk_lp * chunk_mask).sum(dim=1)

    return (sums / counts).float().cpu().numpy()


def score_pair_delta(model, tokenizer, ref_sequences: list[str], mut_sequences: list[str], args) -> dict:
    device = next(model.parameters()).device
    k = getattr(tokenizer, "k", 6)
    n = len(ref_sequences)
    ref_fwd = np.empty(n, dtype=np.float32)
    mut_fwd = np.empty(n, dtype=np.float32)
    ref_rc = np.empty(n, dtype=np.float32)
    mut_rc = np.empty(n, dtype=np.float32)
    scored_ref_bp = np.empty(n, dtype=np.int32)
    scored_mut_bp = np.empty(n, dtype=np.int32)

    for start in tqdm(
        range(0, n, args.batch_size),
        desc="structural-vep",
        unit="batch",
        mininterval=args.progress_interval,
    ):
        end = min(start + args.batch_size, n)
        ref_batch = [dna_trim(seq, k) for seq in ref_sequences[start:end]]
        mut_batch = [dna_trim(seq, k) for seq in mut_sequences[start:end]]
        ref_fwd[start:end] = score_full_ll_batch(model, tokenizer, ref_batch, device, args.logit_chunk_size)
        mut_fwd[start:end] = score_full_ll_batch(model, tokenizer, mut_batch, device, args.logit_chunk_size)
        ref_rc[start:end] = score_full_ll_batch(
            model, tokenizer, [revcomp(seq) for seq in ref_batch], device, args.logit_chunk_size
        )
        mut_rc[start:end] = score_full_ll_batch(
            model, tokenizer, [revcomp(seq) for seq in mut_batch], device, args.logit_chunk_size
        )
        scored_ref_bp[start:end] = [len(seq) for seq in ref_batch]
        scored_mut_bp[start:end] = [len(seq) for seq in mut_batch]

    ref_mean_ll = (ref_fwd + ref_rc) / 2.0
    mut_mean_ll = (mut_fwd + mut_rc) / 2.0
    score = ref_mean_ll - mut_mean_ll
    return {
        "score": score,
        "ref_mean_ll": ref_mean_ll,
        "mut_mean_ll": mut_mean_ll,
        "ref_forward_mean_ll": ref_fwd,
        "mut_forward_mean_ll": mut_fwd,
        "ref_revcomp_mean_ll": ref_rc,
        "mut_revcomp_mean_ll": mut_rc,
        "scored_ref_bp": scored_ref_bp,
        "scored_mut_bp": scored_mut_bp,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="HuggingFaceBio/Carbon-3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ref_column", default="RefSeq")
    parser.add_argument("--mut_column", default="MutSeq")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--logit_chunk_size", type=int, default=64)
    parser.add_argument("--progress_interval", type=float, default=600.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args)

    ds = load_dataset(args.dataset, args.task, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    df = ds.to_pandas()
    ref_sequences = df[args.ref_column].astype(str).tolist()
    mut_sequences = df[args.mut_column].astype(str).tolist()
    labels = df[args.label_column].astype(int).to_numpy()

    t0 = time.time()
    score_data = score_pair_delta(model, tokenizer, ref_sequences, mut_sequences, args)
    elapsed = time.time() - t0

    auroc = float(roc_auc_score(labels, score_data["score"]))
    auprc = float(average_precision_score(labels, score_data["score"]))
    out = pd.DataFrame({"label": labels})
    for key, value in score_data.items():
        out[key] = value

    stem = f"{args.task}_{args.split}_carbon_structural_variant"
    out_path = args.output_dir / f"{stem}_scores.parquet"
    out.to_parquet(out_path, index=False)

    summary = {
        "task": args.task,
        "split": args.split,
        "model": args.model,
        "num_examples": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df[args.label_column].value_counts().sort_index().items()},
        "score_definition": "0.5 * ((ref forward mean LL - mut forward mean LL) + (ref reverse-complement mean LL - mut reverse-complement mean LL)); higher means mutant is less likely than reference",
        "scored_bp_note": "Sequences are trimmed to a multiple of the Carbon tokenizer k-mer size before scoring.",
        "auroc": auroc,
        "auprc": auprc,
        "elapsed_seconds": elapsed,
        "output": str(out_path),
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    combined_path = args.output_dir / "carbon_structural_variant_summary.json"
    combined_path.write_text(json.dumps([summary], indent=2) + "\n")
    print(f"{args.task}/{args.split} AUROC={auroc:.6f} AUPRC={auprc:.6f} n={len(df)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
