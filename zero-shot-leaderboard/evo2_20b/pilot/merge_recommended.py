#!/usr/bin/env python3
"""Merge two task-parallel recommended-setting leaderboard shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FULL_LEADERBOARD_ROWS = 1_727_943
FULL_SV_ROWS = 18_075
FULL_STANDARD_ROWS = FULL_LEADERBOARD_ROWS - FULL_SV_ROWS

LABELS = {
    "conservation_andropogoneae": ("Conservation", "Andropogoneae, genome-wide"),
    "conservation_poaceae_non_tis": ("Conservation", "Poaceae, non-TIS CDS"),
    "conservation_poaceae_tis": ("Conservation", "Poaceae, TIS CDS"),
    "motif_maize_tis": ("Masked motif", "Maize TIS (start)"),
    "motif_maize_tts": ("Masked motif", "Maize TTS (stop)"),
    "motif_maize_donor": ("Masked motif", "Maize splice donor"),
    "motif_maize_acceptor": ("Masked motif", "Maize splice acceptor"),
    "motif_tomato_tis": ("Masked motif", "Tomato TIS (start)"),
    "motif_tomato_tts": ("Masked motif", "Tomato TTS (stop)"),
    "motif_tomato_donor": ("Masked motif", "Tomato splice donor"),
    "motif_tomato_acceptor": ("Masked motif", "Tomato splice acceptor"),
    "core_maize_tis": ("Core/non-core", "Maize TIS (start)"),
    "core_maize_tts": ("Core/non-core", "Maize TTS (stop)"),
    "core_maize_donor": ("Core/non-core", "Maize splice donor"),
    "core_maize_acceptor": ("Core/non-core", "Maize splice acceptor"),
    "core_tomato_tis": ("Core/non-core", "Tomato TIS (start)"),
    "core_tomato_tts": ("Core/non-core", "Tomato TTS (stop)"),
    "core_tomato_donor": ("Core/non-core", "Tomato splice donor"),
    "core_tomato_acceptor": ("Core/non-core", "Tomato splice acceptor"),
    "sv_impact": ("Structural variant", "Impact prediction"),
}


def _rate(tasks: dict[str, Any], *, sv: bool) -> float:
    timings = [
        context["timing"]
        for task in tasks.values()
        if (task["kind"] == "sv") is sv
        for context in task["contexts"].values()
    ]
    return sum(timing["sequences"] for timing in timings) / sum(timing["wall_seconds"] for timing in timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shards = [json.loads(path.read_text()) for path in args.shard]
    tasks: dict[str, Any] = {}
    for shard in shards:
        if shard["task_set"] != "leaderboard":
            raise ValueError(f"Expected leaderboard task set, got {shard['task_set']!r}")
        if not shard["flash_verification"]["verified"]:
            raise RuntimeError(f"FlashAttention was not profiler-verified for shard {shard['task_shard_index']}")
        if not shard["flash_verification"].get("external_flash_attention_2_verified"):
            raise RuntimeError(f"External FlashAttention-2 was not profiler-verified for shard {shard['task_shard_index']}")
        overlap = set(tasks) & set(shard["tasks"])
        if overlap:
            raise ValueError(f"Task overlap between shards: {sorted(overlap)}")
        tasks.update(shard["tasks"])
    if set(tasks) != set(LABELS):
        raise ValueError(f"Task mismatch: missing={sorted(set(LABELS) - set(tasks))}, extra={sorted(set(tasks) - set(LABELS))}")

    sample_manifest = json.loads(args.sample_manifest.read_text())
    if len(sample_manifest) != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} sample records, got {len(sample_manifest)}")
    sample_counts = {record["samples"] for record in sample_manifest}
    seeds = {record["seed"] for record in sample_manifest}
    sampling = {record["sampling"] for record in sample_manifest}
    revisions = {record["resolved_revision"] for record in sample_manifest}
    if len(sample_counts) != 1 or len(seeds) != 1 or sampling != {"random_unstratified"} or len(revisions) != 1:
        raise ValueError(f"Unexpected sampling manifest: counts={sample_counts}, seeds={seeds}, sampling={sampling}, revisions={revisions}")

    total_sequences = sum(context["timing"]["sequences"] for task in tasks.values() for context in task["contexts"].values())
    total_tokens = sum(context["timing"]["tokens"] for task in tasks.values() for context in task["contexts"].values())
    worker_run_seconds = [shard["run_wall_seconds"] for shard in shards]
    worker_timing_seconds = [shard["aggregate_timing"]["wall_seconds"] for shard in shards]
    measured_wall_seconds = max(worker_run_seconds)
    overhead_multiplier = sum(worker_run_seconds) / sum(worker_timing_seconds)
    standard_rate = _rate(tasks, sv=False)
    sv_rate = _rate(tasks, sv=True)
    full_forward_h100_hours = ((2 * FULL_STANDARD_ROWS) / standard_rate + (2 * FULL_SV_ROWS) / sv_rate) / 3600
    full_estimated_h100_hours = full_forward_h100_hours * overhead_multiplier
    full_estimated_2gpu_hours = full_estimated_h100_hours / 2
    peak_reserved_gib = max(context["peak_reserved_gib"] for task in tasks.values() for context in task["contexts"].values())

    rows = []
    for key in LABELS:
        category, label = LABELS[key]
        task = tasks[key]
        rows.append(
            {
                "key": key,
                "category": category,
                "task": label,
                "metric": task["primary_metric"],
                "samples": task["samples"],
                "selected_context": task["best_context"],
                "selected_value": task["best_value"],
                "left": task["contexts"]["left"]["metrics"][task["primary_metric"]],
                "reverse_complement": task["contexts"].get("right_reverse_complement", {}).get("metrics", {}).get(task["primary_metric"]),
            }
        )

    result = {
        "model": shards[0]["model"],
        "condition": "bf16-fa2-fp32-acgt-softmax-no-cache-b32",
        "sampling": {
            "samples_per_task": next(iter(sample_counts)),
            "seed": next(iter(seeds)),
            "method": next(iter(sampling)),
            "dataset_revision": next(iter(revisions)),
            "manifest": str(args.sample_manifest),
        },
        "runtime": {
            "gpus": len(shards),
            "worker_run_seconds": worker_run_seconds,
            "measured_parallel_wall_seconds": measured_wall_seconds,
            "scored_sequences": total_sequences,
            "scored_tokens": total_tokens,
            "cluster_sequences_per_second": total_sequences / measured_wall_seconds,
            "h100_sequences_per_second": total_sequences / sum(worker_run_seconds),
            "standard_h100_sequences_per_second": standard_rate,
            "sv_h100_sequences_per_second": sv_rate,
            "peak_reserved_gib": peak_reserved_gib,
            "overhead_multiplier": overhead_multiplier,
            "full_leaderboard_rows": FULL_LEADERBOARD_ROWS,
            "full_estimated_h100_hours": full_estimated_h100_hours,
            "full_estimated_2gpu_hours": full_estimated_2gpu_hours,
        },
        "environments": [shard["environment"] for shard in shards],
        "flash_verification": [shard["flash_verification"] for shard in shards],
        "rows": rows,
        "tasks": tasks,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ("key", "category", "task", "metric", "samples", "selected_context", "selected_value", "left", "reverse_complement")
    with (args.output_dir / "leaderboard.tsv").open("w") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in rows:
            stream.write("\t".join("" if row[column] is None else str(row[column]) for column in columns) + "\n")
    summary = [
        "# Recommended-setting sampled leaderboard",
        "",
        f"- Model: `{result['model']}`",
        f"- Sampling: {result['sampling']['samples_per_task']:,} random unstratified examples per task, shared seed {result['sampling']['seed']}, dataset revision `{result['sampling']['dataset_revision']}`",
        f"- Runtime: {measured_wall_seconds / 3600:.3f} parallel evaluation hours on {len(shards)} H100s; {total_sequences:,} sequence forwards; {result['runtime']['cluster_sequences_per_second']:.2f} aggregate sequences/s",
        f"- Full-scale estimate: {full_estimated_h100_hours:.2f} H100-hours or {full_estimated_2gpu_hours:.2f} hours on 2 H100s",
        f"- FlashAttention: profiler-verified on all {len(shards)} workers",
        "",
        "| Category | Task | Metric | Selected | Context | Left | Reverse complement |",
        "| :--- | :--- | :--- | ---: | :--- | ---: | ---: |",
    ]
    for row in rows:
        reverse = "—" if row["reverse_complement"] is None else f"{row['reverse_complement']:.6f}"
        summary.append(f"| {row['category']} | {row['task']} | {row['metric']} | {row['selected_value']:.6f} | {row['selected_context']} | {row['left']:.6f} | {reverse} |")
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(json.dumps(result["runtime"], indent=2))


if __name__ == "__main__":
    main()
