#!/usr/bin/env python3
"""Plot category means for multiple MarinDNA checkpoints and published model families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_recommended_comparison import PLOT_CATEGORIES, TASK_IDS, category_summary


PUBLISHED_MODELS = (
    ("PlantCAD2.5-Large", "PlantCAD2.5-L", "8192", "#174A7E"),
    ("PlantCAD2-Large", "PlantCAD2-L", "8192", "#2F78A8"),
    ("PlantCAD2-Medium", "PlantCAD2-M", "8192", "#4E90B6"),
    ("PlantCAD2-Small", "PlantCAD2-S", "8192", "#66A6C9"),
    ("PlantCAD (512 bp)", "PlantCAD", "512", "#A6CEE3"),
    ("evo2_20b", "evo2_20b", "8192", "#7A5195"),
)
MARINDNA_COLORS = ("#C43C39", "#F28E2B", "#FFBE6F", "#8C2D27")
MARINDNA_CHECKPOINT_COLORS = {
    "MarinDNA 1B 0.56T (LR 1e-4, WD 0.2)": "#8C2D27",
    "MarinDNA 1B 0.56T (LR 5e-4, WD 0.1)": "#FFBE6F",
    "MarinDNA 1B 0.56T": "#8C2D27",
    "MarinDNA 1B 0.39T": "#C43C39",
    "MarinDNA 1B 0.22T": "#F28E2B",
    "MarinDNA 1B 20E": "#C43C39",
    "MarinDNA 1B 10E": "#F28E2B",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marindna-result", action="append", nargs=2, metavar=("LABEL", "RESULT_JSON"), required=True)
    parser.add_argument("--leaderboard-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="MarinDNA post-cooldown checkpoint: sampled PlantCAD2 comparison")
    parser.add_argument("--legend-label", action="append", nargs=2, metavar=("LABEL", "DISPLAY_LABEL"), default=[], help="Override a legend label without changing checkpoint identity or color")
    parser.add_argument("--caption-note", default="", help="Append a line beneath the shared comparison caption")
    args = parser.parse_args()
    legend_labels = dict(args.legend_label)

    series: list[tuple[str, list[float], str, str]] = []
    sampled_labels = set()
    for index, (label, result_path) in enumerate(args.marindna_result):
        result = json.loads(Path(result_path).read_text())
        if result.get("sampling", {}).get("method") != "full":
            sampled_labels.add(label)
        values = [row["selected_value"] for row in result["rows"]]
        if len(values) != len(TASK_IDS):
            raise ValueError(f"Expected {len(TASK_IDS)} rows for {label}, got {len(values)}")
        color = MARINDNA_CHECKPOINT_COLORS.get(label, MARINDNA_COLORS[index % len(MARINDNA_COLORS)])
        series.append((label, values, "MarinDNA", color))

    published = list(csv.DictReader(args.leaderboard_results.open()))
    for display, model, context, color in PUBLISHED_MODELS:
        by_task = {row["task_id"]: float(row["value"]) for row in published if row["model"] == model and row["context_bp"] == context}
        missing = [task_id for task_id in TASK_IDS if task_id not in by_task]
        if missing:
            raise ValueError(f"Missing published rows for {model} at {context} bp: {missing}")
        family = "Evo" if model.startswith("evo") else "PlantCAD"
        series.append((display, [by_task[task_id] for task_id in TASK_IDS], family, color))

    categories = PLOT_CATEGORIES
    means = [(label, category_summary(values), family, color) for label, values, family, color in series]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(15, 6.8))
    x = np.arange(len(categories))
    width = 0.78 / len(means)
    for index, (label, model_means, family, color) in enumerate(means):
        positions = x + (index - (len(means) - 1) / 2) * width
        sampled = label in sampled_labels
        display_label = legend_labels.get(label, label)
        bars = ax.bar(positions, model_means, width * 0.94, label=f"{display_label} (sampled)" if sampled else display_label, color=color, hatch="//" if sampled else None, edgecolor="#6E2524" if sampled else color, linewidth=0.7)
        if index == 0 or family == "Evo":
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, fontweight="bold")
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Group score / composite")
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(ncols=3 if len(means) > 8 else 4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9.5)
    caption = "Group scores: Conservation = mean of 3 AUROCs; Masked motif = mean of 8 accuracies; Core/non-core = mean of 8 AUROCs; SV = 1 AUPRC.\nComposite = unweighted mean of those four group scores. Hatched MarinDNA bars use the same 10,000 seed-0 examples/task; published baselines use full splits."
    if not sampled_labels:
        caption = "Conservation: mean of 3 AUROCs · Motif: mean of 8 accuracies · Core/non-core: mean of 8 AUROCs · SV: 1 AUPRC.\nComposite: unweighted mean of the four group scores (25% each). All results use full splits; no downsampling."
    if args.caption_note:
        caption += "\n" + args.caption_note
    fig.text(0.5, 0.018, caption, ha="center", fontsize=11 if sampled_labels else 13, color="#444444")
    fig.tight_layout(rect=(0, (0.15 if not sampled_labels else 0.13) if args.caption_note else 0.10, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
