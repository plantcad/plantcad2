#!/usr/bin/env python3
"""Run paired sensitivity comparisons and estimate full-leaderboard runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# https://datasets-server.huggingface.co/size?dataset=plantcad%2FPlantCAD2_zero_shot_tasks
# at dataset revision d340debe0c8402c84f0696cd2002f87c2f7ba6db.
FULL_LEADERBOARD_ROWS = 1_727_943
FULL_SV_ROWS = 18_075
FULL_STANDARD_ROWS = FULL_LEADERBOARD_ROWS - FULL_SV_ROWS
FULL_LEADERBOARD_FORWARD_SEQUENCES = 2 * FULL_LEADERBOARD_ROWS
H100_DOLLARS_PER_HOUR = 4.29
TASKS = {
    "conservation_poaceae_non_tis": {
        "kind": "conservation",
        "file": "conservation_within_poaceae_non_tis__test.tsv",
    },
    "motif_tomato_acceptor": {
        "kind": "motif",
        "file": "acceptor_recovery__test_tomato.tsv",
    },
    "core_maize_tis": {
        "kind": "core",
        "file": "tis_core_noncore_classification__test_maize.tsv",
    },
    "sv_impact": {
        "kind": "sv",
        "file": "structural_variant_effect_prediction__test.tsv",
    },
}

REFERENCE_CONDITIONS = {
    "bf16-eager-b1": "bf16-fa2-b1",
    "fp32-eager-tf32-b1": "bf16-eager-b1",
    "fp32-eager-no-tf32-b1": "fp32-eager-tf32-b1",
}


def _metric(kind: str, scores: np.ndarray, labels: np.ndarray | None) -> float:
    if kind == "motif":
        return float(scores.mean())
    if kind == "sv":
        assert labels is not None
        return float(average_precision_score(labels, scores))
    assert labels is not None
    return float(roc_auc_score(labels, scores))


def _peak(result: dict[str, Any]) -> float:
    return max(
        context["peak_reserved_gib"]
        for task in result["tasks"].values()
        for context in task["contexts"].values()
    )


def _sequence_rate(result: dict[str, Any], *, sv: bool) -> float:
    timings = [
        context["timing"]
        for task in result["tasks"].values()
        if (task["kind"] == "sv") is sv
        for context in task["contexts"].values()
    ]
    sequences = sum(timing["sequences"] for timing in timings)
    wall_seconds = sum(timing["wall_seconds"] for timing in timings)
    return float(sequences / wall_seconds)


def _shared_arrays(
    reference_scores: Any, condition_scores: Any, n: int
) -> tuple[float, float, float, float, int, int]:
    max_score_diff = 0.0
    max_prob_diff = 0.0
    changed_calls = 0
    compared_calls = 0
    probability_diffs: list[np.ndarray] = []
    for key in condition_scores.files:
        if key not in reference_scores.files:
            continue
        current = condition_scores[key]
        reference = reference_scores[key][:n]
        if current.shape != reference.shape:
            raise ValueError(f"Shape mismatch for {key}: {current.shape} vs {reference.shape}")
        diff = float(np.max(np.abs(current - reference))) if current.size else 0.0
        if key.endswith("__scores"):
            max_score_diff = max(max_score_diff, diff)
        elif key.endswith("__probs"):
            max_prob_diff = max(max_prob_diff, diff)
            probability_diffs.append(np.abs(current - reference).reshape(-1))
            ref_calls = reference.argmax(axis=-1)
            cur_calls = current.argmax(axis=-1)
            changed_calls += int(np.count_nonzero(ref_calls != cur_calls))
            compared_calls += int(ref_calls.size)
    all_probability_diffs = (
        np.concatenate(probability_diffs) if probability_diffs else np.zeros(1)
    )
    return (
        max_score_diff,
        max_prob_diff,
        float(all_probability_diffs.mean()),
        float(np.quantile(all_probability_diffs, 0.99)),
        changed_calls,
        compared_calls,
    )


def _reference_task_metrics(
    key: str,
    reference_scores: Any,
    contexts: list[str],
    n: int,
    labels: np.ndarray | None,
) -> tuple[str, float, dict[str, float]]:
    kind = TASKS[key]["kind"]
    values = {
        context: _metric(
            kind,
            reference_scores[f"{key}__{context}__scores"][:n],
            labels,
        )
        for context in contexts
    }
    best_context = max(contexts, key=values.__getitem__)
    return best_context, values[best_context], values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="bf16-fa2-b4")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    context_records: list[dict[str, Any]] = []

    for condition_dir in sorted(args.matrix_dir.iterdir()):
        if not (condition_dir / "result.json").exists():
            continue
        result = json.loads((condition_dir / "result.json").read_text())
        scores = np.load(condition_dir / "scores.npz")
        reference_name = REFERENCE_CONDITIONS.get(result["condition"], args.baseline)
        reference_dir = args.matrix_dir / reference_name
        if not (reference_dir / "scores.npz").exists():
            raise FileNotFoundError(
                f"Reference condition {reference_name!r} is missing for {result['condition']!r}"
            )
        reference_scores = np.load(reference_dir / "scores.npz")
        n = int(next(iter(result["tasks"].values()))["samples"])
        (
            max_score_diff,
            max_prob_diff,
            mean_prob_diff,
            p99_prob_diff,
            changed_calls,
            compared_calls,
        ) = _shared_arrays(reference_scores, scores, n)
        max_metric_delta = 0.0
        max_context_metric_delta = 0.0
        context_changes = 0
        for task_key, task in result["tasks"].items():
            frame = pd.read_csv(args.sample_dir / TASKS[task_key]["file"], sep="\t", nrows=n)
            labels = frame["label"].astype(int).to_numpy() if "label" in frame else None
            contexts = list(task["contexts"])
            (
                reference_context,
                reference_value,
                reference_context_values,
            ) = _reference_task_metrics(task_key, reference_scores, contexts, n, labels)
            current_context_values = {
                context: _metric(
                    TASKS[task_key]["kind"],
                    scores[f"{task_key}__{context}__scores"],
                    labels,
                )
                for context in contexts
            }
            current_context = max(contexts, key=current_context_values.__getitem__)
            current_value = current_context_values[current_context]
            delta = float(current_value - reference_value)
            max_metric_delta = max(max_metric_delta, abs(delta))
            context_changed = current_context != reference_context
            context_changes += int(context_changed)
            for context in contexts:
                context_value = current_context_values[context]
                context_delta = context_value - reference_context_values[context]
                max_context_metric_delta = max(
                    max_context_metric_delta, abs(context_delta)
                )
                context_records.append(
                    {
                        "condition": result["condition"],
                        "reference_condition": reference_name,
                        "samples": n,
                        "task": task_key,
                        "context": context,
                        "reference_value": reference_context_values[context],
                        "value": context_value,
                        "delta": context_delta,
                    }
                )
            task_records.append(
                {
                    "condition": result["condition"],
                    "reference_condition": reference_name,
                    "samples": n,
                    "task": task_key,
                    "reference_value": reference_value,
                    "value": current_value,
                    "delta": delta,
                    "reference_context": reference_context,
                    "context": current_context,
                    "context_changed": context_changed,
                }
            )
        timing = result["aggregate_timing"]
        sequences_per_second = float(timing["sequences_per_second"])
        standard_sequences_per_second = _sequence_rate(result, sv=False)
        sv_sequences_per_second = _sequence_rate(result, sv=True)
        full_eval_forward_hours = (
            (2 * FULL_STANDARD_ROWS) / standard_sequences_per_second
            + (2 * FULL_SV_ROWS) / sv_sequences_per_second
        ) / 3600
        overhead_multiplier = result["run_wall_seconds"] / timing["wall_seconds"]
        full_eval_wall_hours = full_eval_forward_hours * overhead_multiplier
        records.append(
            {
                "condition": result["condition"],
                "reference_condition": reference_name,
                "samples_per_task": n,
                "dtype": result["environment"]["requested_dtype"],
                "attention": result["environment"]["requested_attention"],
                "transformers": result["environment"]["transformers"],
                "batch_size": result["environment"]["batch_size"],
                "sv_batch_size": result["environment"]["sv_batch_size"],
                "use_cache": result["environment"]["use_cache"],
                "tf32": result["environment"]["tf32"],
                "softmax_dtype": result["environment"]["softmax_dtype"],
                "flash_verified": result["flash_verification"]["verified"],
                "flash_kernel_events": result["flash_verification"]["matching_kernel_events"],
                "sequences_per_second": sequences_per_second,
                "standard_sequences_per_second": standard_sequences_per_second,
                "sv_sequences_per_second": sv_sequences_per_second,
                "tokens_per_second": float(timing["tokens_per_second"]),
                "peak_reserved_gib": _peak(result),
                "measured_overhead_multiplier": overhead_multiplier,
                "full_eval_forward_hours": full_eval_forward_hours,
                "full_eval_wall_hours": full_eval_wall_hours,
                "full_eval_cost_dollars": full_eval_wall_hours * H100_DOLLARS_PER_HOUR,
                "max_abs_primary_metric_delta": max_metric_delta,
                "max_abs_context_metric_delta": max_context_metric_delta,
                "max_abs_probability_delta": max_prob_diff,
                "mean_abs_probability_delta": mean_prob_diff,
                "p99_abs_probability_delta": p99_prob_diff,
                "max_abs_score_delta": max_score_diff,
                "changed_base_calls": changed_calls,
                "compared_base_calls": compared_calls,
                "context_changes": context_changes,
            }
        )

    payload = {
        "baseline": args.baseline,
        "full_leaderboard_rows": FULL_LEADERBOARD_ROWS,
        "full_standard_rows": FULL_STANDARD_ROWS,
        "full_sv_rows": FULL_SV_ROWS,
        "full_leaderboard_forward_sequences": FULL_LEADERBOARD_FORWARD_SEQUENCES,
        "h100_dollars_per_hour": H100_DOLLARS_PER_HOUR,
        "conditions": records,
        "tasks": task_records,
        "contexts": context_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
