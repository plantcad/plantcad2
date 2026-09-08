#!/usr/bin/env python3
"""Summarize atomic progress files from a two-worker recommended evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--compact", action="store_true", help="Print one human-readable line for polling")
    args = parser.parse_args()

    workers = []
    remaining = []
    for index in range(2):
        path = args.output_dir / "workers" / f"gpu{index}" / "progress.json"
        if not path.exists():
            workers.append({"gpu": index, "state": "starting"})
            continue
        progress = json.loads(path.read_text())
        fraction = float(progress["estimated_fraction"])
        elapsed = float(progress["elapsed_seconds"])
        if fraction > 0:
            remaining.append(elapsed * (1 - fraction) / fraction)
        completed = []
        for key in progress["completed_task_keys"]:
            task = progress["tasks"][key]
            completed.append(
                {
                    "key": key,
                    "metric": task["primary_metric"],
                    "value": task["best_value"],
                    "context": task["best_context"],
                }
            )
        workers.append(
            {
                "gpu": index,
                "fraction": fraction,
                "elapsed_seconds": elapsed,
                "completed": completed,
                "current_task": progress["current_task_key"],
                "current_phase": progress["current_phase"],
                "current_phase_completed": progress["current_phase_completed"],
                "current_phase_total": progress["current_phase_total"],
            }
        )
    summary = {
        "workers": workers,
        "mean_fraction": sum(worker.get("fraction", 0.0) for worker in workers) / 2,
        "estimated_remaining_seconds": max(remaining) if remaining else None,
    }
    if args.compact:
        eta = "?" if summary["estimated_remaining_seconds"] is None else str(timedelta(seconds=round(summary["estimated_remaining_seconds"])))
        worker_text = []
        for worker in workers:
            if worker.get("state") == "starting":
                worker_text.append(f"gpu{worker['gpu']}=starting")
            else:
                worker_text.append(
                    f"gpu{worker['gpu']}={worker['current_task']}:{worker['current_phase']}:{worker['current_phase_completed']}/{worker['current_phase_total']}:done{len(worker['completed'])}"
                )
        print(f"{summary['mean_fraction']:.1%} eta={eta} {' '.join(worker_text)}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
