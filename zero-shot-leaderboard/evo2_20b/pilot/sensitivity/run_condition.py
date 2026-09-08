#!/usr/bin/env python3
"""Run one fixed-sample numerical/runtime sensitivity condition on an HF model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
EVALUATOR_PATH = HERE.parents[1] / "zero-shot-eval-causal.py"
SPEC = importlib.util.spec_from_file_location("plantcad2_causal_eval", EVALUATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load evaluator at {EVALUATOR_PATH}")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)

TASKS = (
    {
        "key": "conservation_poaceae_non_tis",
        "kind": "conservation",
        "file": "conservation_within_poaceae_non_tis__test.tsv",
        "positions": [4095],
        "contexts": ["left", "right_reverse_complement"],
    },
    {
        "key": "motif_tomato_acceptor",
        "kind": "motif",
        "file": "acceptor_recovery__test_tomato.tsv",
        "positions": [4095, 4096],
        "contexts": ["left", "right_reverse_complement"],
    },
    {
        "key": "core_maize_tis",
        "kind": "core",
        "file": "tis_core_noncore_classification__test_maize.tsv",
        "positions": [4094, 4095, 4096],
        "contexts": ["left", "right_reverse_complement"],
    },
    {
        "key": "sv_impact",
        "kind": "sv",
        "file": "structural_variant_effect_prediction__test.tsv",
        "contexts": ["left"],
    },
)


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _sync() -> None:
    torch.cuda.synchronize()


def _environment(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    try:
        import flash_attn

        flash_attn_version = flash_attn.__version__
    except Exception as exc:  # pragma: no cover - environment diagnostic
        flash_attn_version = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn_version,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "requested_dtype": args.dtype,
        "model_dtype": str(model.dtype),
        "requested_attention": args.attention,
        "resolved_attention": getattr(model.config, "_attn_implementation", None),
        "tf32": args.tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "softmax_dtype": args.softmax_dtype,
        "use_cache": args.use_cache,
        "batch_size": args.batch_size,
        "sv_batch_size": args.sv_batch_size,
    }


def _profile_attention(model: Any, tokenizer: Any, sequence: str, require_flash: bool) -> dict[str, Any]:
    ids = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"].to("cuda:0")
    with torch.inference_mode():
        model(input_ids=ids)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.inference_mode():
            model(input_ids=ids)
        _sync()
    keys = sorted({event.key for event in prof.key_averages()})
    flash_keys = [key for key in keys if "flash" in key.lower()]
    verified_flash_keys = [
        key
        for key in flash_keys
        if any(
            signature in key.lower()
            for signature in (
                "flash_fwd_kernel",
                "flash_attn::_flash_attn_forward",
                "aten::_flash_attention_forward",
                "aten::_scaled_dot_product_flash_attention",
            )
        )
    ]
    if require_flash and not verified_flash_keys:
        raise RuntimeError(
            "FlashAttention was required but profiler dispatch contained no known flash "
            "forward operation or CUDA kernel: " + ", ".join(flash_keys)
        )
    attention_classes = sorted(
        {
            f"{module.__class__.__module__}.{module.__class__.__name__}"
            for name, module in model.named_modules()
            if name.endswith("self_attn")
        }
    )
    return {
        "verified": bool(verified_flash_keys),
        "matching_kernel_events": flash_keys,
        "verified_kernel_events": verified_flash_keys,
        "attention_classes": attention_classes,
        "profile_event_count": len(keys),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _softmax_acgt(logits: torch.Tensor, ids: list[int], mode: str) -> torch.Tensor:
    selected = logits[..., ids]
    if mode == "fp32":
        return torch.softmax(selected.float(), dim=-1)
    return torch.softmax(selected, dim=-1).float()


def _score_positions(
    model: Any,
    tokenizer: Any,
    sequences: pd.Series,
    positions: list[int],
    batch_size: int,
    softmax_dtype: str,
) -> tuple[np.ndarray, dict[str, float]]:
    nucleotide_ids = EVAL._nucleotide_token_ids(tokenizer)
    seqs = sequences.astype(str).tolist()
    outputs: list[np.ndarray] = []
    gpu_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    _sync()
    started = time.perf_counter()
    for offset in range(0, len(seqs), batch_size):
        batch = seqs[offset : offset + batch_size]
        ids = tokenizer(
            batch,
            truncation=False,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"].to("cuda:0")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            logits = model(input_ids=ids).logits
        end.record()
        gpu_events.append((start, end))
        probs = _softmax_acgt(logits[:, :-1, :], nucleotide_ids, softmax_dtype)
        picked = torch.stack([probs[:, position - 1, :] for position in positions], dim=1)
        outputs.append(picked.cpu().numpy())
    _sync()
    wall = time.perf_counter() - started
    gpu = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000
    arr = np.concatenate(outputs, axis=0)
    tokens = sum(len(seq) for seq in seqs)
    return arr, {
        "sequences": len(seqs),
        "tokens": tokens,
        "wall_seconds": wall,
        "gpu_seconds": gpu,
        "sequences_per_second": len(seqs) / wall,
        "tokens_per_second": tokens / wall,
    }


def _score_full(
    model: Any,
    tokenizer: Any,
    sequences: pd.Series,
    batch_size: int,
    softmax_dtype: str,
) -> tuple[np.ndarray, dict[str, float]]:
    nucleotide_ids = EVAL._nucleotide_token_ids(tokenizer)
    seqs = sequences.astype(str).tolist()
    result: np.ndarray | None = None
    gpu_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    _sync()
    started = time.perf_counter()
    for offset in range(0, len(seqs), batch_size):
        batch = seqs[offset : offset + batch_size]
        ids = tokenizer(
            batch,
            truncation=False,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"].to("cuda:0")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            logits = model(input_ids=ids).logits
        end.record()
        gpu_events.append((start, end))
        probs = _softmax_acgt(logits[:, :-1, :], nucleotide_ids, softmax_dtype).cpu().numpy()
        if result is None:
            result = np.zeros((len(seqs), probs.shape[1] + 1, 4), dtype=np.float32)
        result[offset : offset + len(batch), 1:, :] = probs
    _sync()
    wall = time.perf_counter() - started
    gpu = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000
    assert result is not None
    tokens = sum(len(seq) for seq in seqs)
    return result, {
        "sequences": len(seqs),
        "tokens": tokens,
        "wall_seconds": wall,
        "gpu_seconds": gpu,
        "sequences_per_second": len(seqs) / wall,
        "tokens_per_second": tokens / wall,
    }


def _merge_timing(parts: list[dict[str, float]]) -> dict[str, float]:
    sequences = int(sum(p["sequences"] for p in parts))
    tokens = int(sum(p["tokens"] for p in parts))
    wall = sum(p["wall_seconds"] for p in parts)
    gpu = sum(p["gpu_seconds"] for p in parts)
    return {
        "sequences": sequences,
        "tokens": tokens,
        "wall_seconds": wall,
        "gpu_seconds": gpu,
        "sequences_per_second": sequences / wall,
        "tokens_per_second": tokens / wall,
    }


def _score_standard_task(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    context_results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for context in spec["contexts"]:
        work_sequences, work_positions = EVAL._transform_sequences_and_positions(
            frame["sequence"], spec["positions"], context
        )
        torch.cuda.reset_peak_memory_stats()
        probs3, timing = _score_positions(
            model,
            tokenizer,
            work_sequences,
            work_positions,
            args.batch_size,
            args.softmax_dtype,
        )
        probs = probs3.reshape(-1, 4)
        work_frame = frame.copy()
        work_frame["sequence"] = work_sequences
        true_tokens = EVAL._compute_true_tokens_from_seq(work_sequences, work_positions)
        if spec["kind"] == "conservation":
            scores = EVAL._refprob_scores(work_frame, probs, work_positions[0], "sequence")
            metrics = {
                "auroc": float(roc_auc_score(frame["label"].astype(int), scores)),
                "auprc": float(average_precision_score(frame["label"].astype(int), scores)),
            }
        elif spec["kind"] == "motif":
            pred = np.asarray(EVAL.NUCLEOTIDES)[probs.argmax(axis=1)].reshape(len(frame), -1)
            true = true_tokens.reshape(len(frame), -1)
            scores = np.all(pred == true, axis=1).astype(np.float32)
            metrics = {
                "token_accuracy": EVAL._metric_token_accuracy(probs, true_tokens),
                "motif_accuracy": EVAL._metric_motif_accuracy(
                    probs, true_tokens, len(work_positions)
                ),
            }
        else:
            scores = EVAL._avg_trueprob_scores(probs, true_tokens, len(work_positions))
            metrics = {
                "auroc": float(roc_auc_score(frame["label"].astype(int), scores)),
                "auprc": float(average_precision_score(frame["label"].astype(int), scores)),
            }
        arrays[f"{context}__probs"] = probs3
        arrays[f"{context}__scores"] = scores
        context_results[context] = {
            "metrics": metrics,
            "timing": timing,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
    primary = "motif_accuracy" if spec["kind"] == "motif" else "auroc"
    best = max(spec["contexts"], key=lambda context: context_results[context]["metrics"][primary])
    return {
        "kind": spec["kind"],
        "samples": len(frame),
        "primary_metric": primary,
        "best_context": best,
        "best_value": context_results[best]["metrics"][primary],
        "contexts": context_results,
    }, arrays


def _score_sv(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    torch.cuda.reset_peak_memory_stats()
    ref_probs, ref_timing = _score_full(
        model, tokenizer, frame["RefSeq"], args.sv_batch_size, args.softmax_dtype
    )
    mut_probs, mut_timing = _score_full(
        model, tokenizer, frame["MutSeq"], args.sv_batch_size, args.softmax_dtype
    )
    scores = EVAL._sv_llr_boundary(frame, ref_probs, mut_probs, flanking=5)
    auprc = float(average_precision_score(frame["label"].astype(int), scores))
    return {
        "kind": "sv",
        "samples": len(frame),
        "primary_metric": "auprc",
        "best_context": "left",
        "best_value": auprc,
        "contexts": {
            "left": {
                "metrics": {"auprc": auprc},
                "timing": _merge_timing([ref_timing, mut_timing]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        },
    }, {"left__scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--attention", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sv-batch-size", type=int, default=2)
    parser.add_argument("--softmax-dtype", choices=("fp32", "native"), default="fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-flash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.manual_seed(0)
    np.random.seed(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=_dtype(args.dtype),
        attn_implementation=args.attention,
    ).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.config.use_cache = args.use_cache
    model.eval()
    _sync()
    load_seconds = time.perf_counter() - load_started

    first_frame = pd.read_csv(args.sample_dir / TASKS[0]["file"], sep="\t")
    if args.max_samples:
        first_frame = first_frame.iloc[: args.max_samples].copy()
    profile = _profile_attention(
        model, tokenizer, str(first_frame["sequence"].iloc[0]), args.require_flash
    )

    started = time.perf_counter()
    task_results: dict[str, Any] = {}
    archive: dict[str, np.ndarray] = {}
    for spec in TASKS:
        frame = pd.read_csv(args.sample_dir / spec["file"], sep="\t")
        if args.max_samples:
            frame = frame.iloc[: args.max_samples].copy()
        if spec["kind"] == "sv":
            result, arrays = _score_sv(spec, frame, model, tokenizer, args)
        else:
            result, arrays = _score_standard_task(spec, frame, model, tokenizer, args)
        task_results[spec["key"]] = result
        archive.update({f"{spec['key']}__{name}": value for name, value in arrays.items()})
    run_seconds = time.perf_counter() - started
    all_timings = [
        context["timing"]
        for task in task_results.values()
        for context in task["contexts"].values()
    ]
    aggregate_timing = _merge_timing(all_timings)
    payload = {
        "condition": args.condition,
        "model": args.model,
        "sample_dir": str(args.sample_dir),
        "max_samples": args.max_samples or None,
        "load_seconds": load_seconds,
        "run_wall_seconds": run_seconds,
        "environment": _environment(model, args),
        "flash_verification": profile,
        "aggregate_timing": aggregate_timing,
        "tasks": task_results,
    }
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "scores.npz", **archive)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
