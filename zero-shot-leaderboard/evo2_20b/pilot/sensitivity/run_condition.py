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
from typing import Any, Callable

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

REPRESENTATIVE_TASKS = (
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


def _leaderboard_tasks() -> tuple[dict[str, Any], ...]:
    tasks: list[dict[str, Any]] = [
        {
            "key": "conservation_andropogoneae",
            "kind": "conservation",
            "file": "conservation_within_andropogoneae__test.tsv",
            "positions": [4095],
            "contexts": ["left", "right_reverse_complement"],
        },
        {
            "key": "conservation_poaceae_non_tis",
            "kind": "conservation",
            "file": "conservation_within_poaceae_non_tis__test.tsv",
            "positions": [4095],
            "contexts": ["left", "right_reverse_complement"],
        },
        {
            "key": "conservation_poaceae_tis",
            "kind": "conservation",
            "file": "conservation_within_poaceae_tis__test.tsv",
            "positions": [4095],
            "contexts": ["left", "right_reverse_complement"],
        },
    ]
    for species in ("maize", "tomato"):
        for motif, positions in (
            ("tis", [4094, 4095, 4096]),
            ("tts", [4094, 4095, 4096]),
            ("donor", [4095, 4096]),
            ("acceptor", [4095, 4096]),
        ):
            tasks.append(
                {
                    "key": f"motif_{species}_{motif}",
                    "kind": "motif",
                    "file": f"{motif}_recovery__test_{species}.tsv",
                    "positions": positions,
                    "contexts": ["left", "right_reverse_complement"],
                }
            )
    for species in ("maize", "tomato"):
        for motif, positions in (
            ("tis", [4094, 4095, 4096]),
            ("tts", [4094, 4095, 4096]),
            ("donor", [4095, 4096]),
            ("acceptor", [4095, 4096]),
        ):
            tasks.append(
                {
                    "key": f"core_{species}_{motif}",
                    "kind": "core",
                    "file": f"{motif}_core_noncore_classification__test_{species}.tsv",
                    "positions": positions,
                    "contexts": ["left", "right_reverse_complement"],
                }
            )
    tasks.append(
        {
            "key": "sv_impact",
            "kind": "sv",
            "file": "structural_variant_effect_prediction__test.tsv",
            "contexts": ["left"],
        }
    )
    return tuple(tasks)


LEADERBOARD_TASKS = _leaderboard_tasks()
TASK_SETS = {"representative": REPRESENTATIVE_TASKS, "leaderboard": LEADERBOARD_TASKS}


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


def _profile_attention(
    model: Any, tokenizer: Any, sequence: str, require_flash: bool, use_cache: bool
) -> dict[str, Any]:
    ids = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"].to("cuda:0")
    with torch.inference_mode():
        model(input_ids=ids, use_cache=use_cache)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=use_cache)
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
    use_cache: bool,
    progress: Callable[[int, int], None] | None = None,
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
            logits = model(input_ids=ids, use_cache=use_cache).logits
        end.record()
        gpu_events.append((start, end))
        probs = _softmax_acgt(logits[:, :-1, :], nucleotide_ids, softmax_dtype)
        picked = torch.stack([probs[:, position - 1, :] for position in positions], dim=1)
        outputs.append(picked.cpu().numpy())
        completed = offset + len(batch)
        if progress is not None and (completed == len(seqs) or completed % (batch_size * 25) == 0):
            progress(completed, len(seqs))
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
    use_cache: bool,
    progress: Callable[[int, int], None] | None = None,
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
            logits = model(input_ids=ids, use_cache=use_cache).logits
        end.record()
        gpu_events.append((start, end))
        probs = _softmax_acgt(logits[:, :-1, :], nucleotide_ids, softmax_dtype).cpu().numpy()
        if result is None:
            result = np.zeros((len(seqs), probs.shape[1] + 1, 4), dtype=np.float32)
        result[offset : offset + len(batch), 1:, :] = probs
        completed = offset + len(batch)
        if progress is not None and (completed == len(seqs) or completed % (batch_size * 25) == 0):
            progress(completed, len(seqs))
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
    progress: Callable[[str, int, int, int, int], None] | None = None,
    compute_metrics: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    context_results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for context_index, context in enumerate(spec["contexts"]):
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
            args.use_cache,
            None
            if progress is None
            else lambda completed, total, context=context, context_index=context_index: progress(
                context, context_index, len(spec["contexts"]), completed, total
            ),
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
            } if compute_metrics else {}
        elif spec["kind"] == "motif":
            pred = np.asarray(EVAL.NUCLEOTIDES)[probs.argmax(axis=1)].reshape(len(frame), -1)
            true = true_tokens.reshape(len(frame), -1)
            scores = np.all(pred == true, axis=1).astype(np.float32)
            metrics = {
                "token_accuracy": EVAL._metric_token_accuracy(probs, true_tokens),
                "motif_accuracy": EVAL._metric_motif_accuracy(
                    probs, true_tokens, len(work_positions)
                ),
            } if compute_metrics else {}
        else:
            scores = EVAL._avg_trueprob_scores(probs, true_tokens, len(work_positions))
            metrics = {
                "auroc": float(roc_auc_score(frame["label"].astype(int), scores)),
                "auprc": float(average_precision_score(frame["label"].astype(int), scores)),
            } if compute_metrics else {}
        arrays[f"{context}__probs"] = probs3
        arrays[f"{context}__scores"] = scores
        context_results[context] = {
            "metrics": metrics,
            "timing": timing,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
    primary = "motif_accuracy" if spec["kind"] == "motif" else "auroc"
    # Distributed full-split chunks may have one label class. In that mode only
    # predictions/timing are returned; the reducer computes all metrics globally.
    best = max(spec["contexts"], key=lambda context: context_results[context]["metrics"][primary]) if compute_metrics else None
    return {
        "kind": spec["kind"],
        "samples": len(frame),
        "primary_metric": primary,
        "best_context": best,
        "best_value": context_results[best]["metrics"][primary] if compute_metrics else None,
        "contexts": context_results,
    }, arrays


def _score_sv(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    progress: Callable[[str, int, int, int, int], None] | None = None,
    compute_metrics: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    torch.cuda.reset_peak_memory_stats()
    ref_probs, ref_timing = _score_full(
        model,
        tokenizer,
        frame["RefSeq"],
        args.sv_batch_size,
        args.softmax_dtype,
        args.use_cache,
        None if progress is None else lambda completed, total: progress("left_ref", 0, 2, completed, total),
    )
    mut_probs, mut_timing = _score_full(
        model,
        tokenizer,
        frame["MutSeq"],
        args.sv_batch_size,
        args.softmax_dtype,
        args.use_cache,
        None if progress is None else lambda completed, total: progress("left_mut", 1, 2, completed, total),
    )
    scores = EVAL._sv_llr_boundary(frame, ref_probs, mut_probs, flanking=5)
    auprc = float(average_precision_score(frame["label"].astype(int), scores)) if compute_metrics else None
    return {
        "kind": "sv",
        "samples": len(frame),
        "primary_metric": "auprc",
        "best_context": "left",
        "best_value": auprc,
        "contexts": {
            "left": {
                "metrics": {"auprc": auprc} if compute_metrics else {},
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
    parser.add_argument("--task-set", choices=tuple(TASK_SETS), default="representative")
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--task-shard-index", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.manual_seed(0)
    np.random.seed(0)
    if args.task_shard_count < 1:
        raise ValueError("task-shard-count must be at least 1")
    if not 0 <= args.task_shard_index < args.task_shard_count:
        raise ValueError("task-shard-index must be in [0, task-shard-count)")
    all_tasks = TASK_SETS[args.task_set]
    tasks = tuple(
        task for index, task in enumerate(all_tasks) if index % args.task_shard_count == args.task_shard_index
    )
    if not tasks:
        raise ValueError("The selected task shard is empty")

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

    first_frame = pd.read_csv(args.sample_dir / tasks[0]["file"], sep="\t")
    if args.max_samples:
        first_frame = first_frame.iloc[: args.max_samples].copy()
    profile = _profile_attention(
        model,
        tokenizer,
        str(first_frame["sequence"].iloc[0] if "sequence" in first_frame else first_frame["RefSeq"].iloc[0]),
        args.require_flash,
        args.use_cache,
    )
    resolved_attention = getattr(model.config, "_attn_implementation", None)
    if args.attention == "flash_attention_2":
        external_fa2_events = [
            key
            for key in profile["matching_kernel_events"]
            if "flash_attn" in key.lower() or "flash::" in key.lower()
        ]
        profile["external_flash_attention_2_events"] = external_fa2_events
        profile["external_flash_attention_2_verified"] = bool(external_fa2_events)
        if resolved_attention != "flash_attention_2" or (args.require_flash and not external_fa2_events):
            raise RuntimeError(
                "External FlashAttention-2 was required but not verified: "
                f"resolved_attention={resolved_attention!r}, matching_events={profile['matching_kernel_events']}"
            )

    started = time.perf_counter()
    task_results: dict[str, Any] = {}
    archive: dict[str, np.ndarray] = {}
    environment = _environment(model, args)
    progress_path = args.output_dir / "progress.json"

    def write_progress(
        current_task_key: str | None,
        current_phase: str | None,
        phase_index: int = 0,
        phase_count: int = 0,
        phase_completed: int = 0,
        phase_total: int = 0,
    ) -> None:
        current_task_completed = phase_index * phase_total + phase_completed if phase_total else 0
        current_task_total = phase_count * phase_total if phase_total else 0
        current_fraction = current_task_completed / current_task_total if current_task_total else 0.0
        progress = {
            "condition": args.condition,
            "model": args.model,
            "task_set": args.task_set,
            "task_shard_count": args.task_shard_count,
            "task_shard_index": args.task_shard_index,
            "completed_task_keys": list(task_results),
            "total_task_keys": [task["key"] for task in tasks],
            "current_task_key": current_task_key,
            "current_phase": current_phase,
            "current_phase_completed": phase_completed,
            "current_phase_total": phase_total,
            "current_task_completed_forwards": current_task_completed,
            "current_task_total_forwards": current_task_total,
            "estimated_fraction": (len(task_results) + current_fraction) / len(tasks),
            "elapsed_seconds": time.perf_counter() - started,
            "environment": environment,
            "flash_verification": profile,
            "tasks": task_results,
        }
        progress_tmp = progress_path.with_suffix(".json.tmp")
        progress_tmp.write_text(json.dumps(progress, indent=2) + "\n")
        progress_tmp.replace(progress_path)

    for spec in tasks:
        frame = pd.read_csv(args.sample_dir / spec["file"], sep="\t")
        if args.max_samples:
            frame = frame.iloc[: args.max_samples].copy()
        initial_phase = "left_ref" if spec["kind"] == "sv" else spec["contexts"][0]
        write_progress(spec["key"], initial_phase, 0, 2, 0, len(frame))

        def report_progress(
            phase: str, phase_index: int, phase_count: int, completed: int, total: int
        ) -> None:
            write_progress(spec["key"], phase, phase_index, phase_count, completed, total)

        if spec["kind"] == "sv":
            result, arrays = _score_sv(spec, frame, model, tokenizer, args, report_progress)
        else:
            result, arrays = _score_standard_task(spec, frame, model, tokenizer, args, report_progress)
        task_results[spec["key"]] = result
        archive.update({f"{spec['key']}__{name}": value for name, value in arrays.items()})
        write_progress(None, None)
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
        "task_set": args.task_set,
        "task_shard_count": args.task_shard_count,
        "task_shard_index": args.task_shard_index,
        "task_keys": [task["key"] for task in tasks],
        "load_seconds": load_seconds,
        "run_wall_seconds": run_seconds,
        "environment": environment,
        "flash_verification": profile,
        "aggregate_timing": aggregate_timing,
        "tasks": task_results,
    }
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "scores.npz", **archive)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
