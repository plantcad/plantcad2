#!/usr/bin/env python3
"""Plot category means for an exp472 sampled result against published leaderboard rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_MODELS = {
    "PlantCAD2.5-Large": ("PlantCAD2.5-L", "8192"),
    "PlantCAD2-Large": ("PlantCAD2-L", "8192"),
    "PlantCAD2-Medium": ("PlantCAD2-M", "8192"),
    "PlantCAD2-Small": ("PlantCAD2-S", "8192"),
    "PlantCAD (512 bp)": ("PlantCAD", "512"),
    "evo2_20b": ("evo2_20b", "8192"),
}
TASK_IDS = [
    "cons_andropogoneae",
    "cons_poaceae_nontis",
    "cons_poaceae_tis",
    "motif_maize_start_sites",
    "motif_maize_stop_sites",
    "motif_maize_donor",
    "motif_maize_acceptor",
    "motif_tomato_start_sites",
    "motif_tomato_stop_sites",
    "motif_tomato_donor",
    "motif_tomato_acceptor",
    "coreclass_maize_tis",
    "coreclass_maize_tts",
    "coreclass_maize_donor",
    "coreclass_maize_acceptor",
    "coreclass_tomato_tis",
    "coreclass_tomato_tts",
    "coreclass_tomato_donor",
    "coreclass_tomato_acceptor",
    "sv_impact_auprc",
]
CATEGORY_SLICES = {"Conservation": slice(0, 3), "Masked motif": slice(3, 11), "Core/non-core": slice(11, 19), "SV": slice(19, 20)}
PLOT_CATEGORIES = (*CATEGORY_SLICES, "Composite")


def category_summary(values: list[float]) -> list[float]:
    array = np.asarray(values)
    group_means = [float(np.mean(array[span])) for span in CATEGORY_SLICES.values()]
    return [*group_means, float(np.mean(group_means))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--leaderboard-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exp-label", default="exp472 step 206,144")
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    values = [row["selected_value"] for row in result["rows"]]
    if len(values) != 20:
        raise ValueError(f"Expected 20 result rows, got {len(values)}")
    published = list(csv.DictReader(args.leaderboard_results.open()))
    baseline_values = {}
    for display, (model, context) in BASELINE_MODELS.items():
        by_task = {row["task_id"]: float(row["value"]) for row in published if row["model"] == model and row["context_bp"] == context}
        missing = [task_id for task_id in TASK_IDS if task_id not in by_task]
        if missing:
            raise ValueError(f"Missing published rows for {model} at {context} bp: {missing}")
        baseline_values[display] = [by_task[task_id] for task_id in TASK_IDS]
    models = {args.exp_label: values, **baseline_values}
    categories = PLOT_CATEGORIES
    means = {model: category_summary(model_values) for model, model_values in models.items()}

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    x = np.arange(len(categories))
    width = 0.135
    colors = ["#E45756", "#4C78A8", "#72B7B2", "#59A14F", "#F2CF5B", "#B279A2"]
    for index, (model, model_means) in enumerate(means.items()):
        positions = x + (index - (len(means) - 1) / 2) * width
        sampled = index == 0
        bars = ax.bar(positions, model_means, width, label=f"{model} (sampled)" if sampled else model, color=colors[index], hatch="//" if sampled else None, edgecolor="#6E2524" if sampled else colors[index], linewidth=0.7)
        if index == 0:
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, fontweight="bold")
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Group score / composite")
    ax.set_title("exp472 final checkpoint: sampled PlantCAD2 comparison")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.text(0.5, 0.018, "Group scores: Conservation = mean of 3 AUROCs; Masked motif = mean of 8 accuracies; Core/non-core = mean of 8 AUROCs; SV = 1 AUPRC.\nComposite = unweighted mean of those four group scores. Hatched exp472 bars use 10,000 seed-0 examples/task; published baselines use full splits.", ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
