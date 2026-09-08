#!/usr/bin/env python3
"""Plot production-relevant sensitivity results from analyze.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DISPLAY = (
    ("bf16-fa2-b4", "BF16 · FA2 · b4", "#4C78A8"),
    ("transformers4-bf16-fa2-b4", "BF16 · FA2 · b4 · HF 4", "#72B7B2"),
    ("bf16-sdpa-flash-b4", "BF16 · SDPA-flash · b4", "#B279A2"),
    ("fp16-fa2-b4", "FP16 · FA2 · b4", "#F58518"),
    ("bf16-fa2-b1", "BF16 · FA2 · b1", "#9D755D"),
    ("bf16-fa2-native-softmax-b4", "BF16 · FA2 · native softmax", "#E45756"),
    ("bf16-fa2-no-tf32-b4", "BF16 · FA2 · TF32 off · n1k", "#BAB0AC"),
    ("bf16-fa2-no-cache-b32", "BF16 · FA2 · no cache · b32", "#54A24B"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text())
    by_name = {row["condition"]: row for row in payload["conditions"]}
    selected = [(name, label, color, by_name[name]) for name, label, color in DISPLAY]
    unverified = [name for name, _, _, row in selected if not row["flash_verified"]]
    if unverified:
        raise RuntimeError(f"Plot refuses unverified flash conditions: {unverified}")

    selected.reverse()
    labels = [label for _, label, _, _ in selected]
    colors = [color for _, _, color, _ in selected]
    hours = [row["full_eval_wall_hours"] for _, _, _, row in selected]
    deltas_pp = [100 * row["max_abs_context_metric_delta"] for _, _, _, row in selected]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, (speed_ax, delta_ax) = plt.subplots(
        1, 2, figsize=(13.5, 5.7), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.suptitle("exp472 step 75,046 sensitivity: runtime and metric movement", fontsize=15)

    speed_ax.barh(labels, hours, color=colors)
    speed_ax.set_title("Estimated full PlantCAD2 evaluation")
    speed_ax.set_xlabel("H100 wall hours (lower is better)")
    speed_ax.grid(axis="x", alpha=0.2)
    speed_ax.set_axisbelow(True)
    for index, value in enumerate(hours):
        speed_ax.text(value + 0.35, index, f"{value:.1f} h", va="center", fontsize=9)
    speed_ax.set_xlim(0, max(hours) * 1.15)

    delta_ax.barh(labels, deltas_pp, color=colors)
    delta_ax.set_title("Largest strand/context metric change")
    delta_ax.set_xlabel("Absolute change vs paired control (percentage points)")
    delta_ax.grid(axis="x", alpha=0.2)
    delta_ax.set_axisbelow(True)
    for index, value in enumerate(deltas_pp):
        delta_ax.text(value + 0.002, index, f"{value:.3f}", va="center", fontsize=9)
    delta_ax.set_xlim(0, max(deltas_pp) * 1.22 if max(deltas_pp) else 0.1)

    fig.text(
        0.5,
        0.015,
        "Same sealed seed-0 samples; all plotted runs profiler-verified a flash forward kernel. "
        "FP32/eager diagnostics are intentionally omitted.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
