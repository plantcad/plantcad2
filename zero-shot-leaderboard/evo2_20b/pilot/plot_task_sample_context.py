#!/usr/bin/env python3
"""Plot per-task full-versus-sampled scores in the context of published models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from plot_recommended_comparison import BASELINE_MODELS, TASK_IDS


RESULT_KEYS = [
    "conservation_andropogoneae",
    "conservation_poaceae_non_tis",
    "conservation_poaceae_tis",
    "motif_maize_tis",
    "motif_maize_tts",
    "motif_maize_donor",
    "motif_maize_acceptor",
    "motif_tomato_tis",
    "motif_tomato_tts",
    "motif_tomato_donor",
    "motif_tomato_acceptor",
    "core_maize_tis",
    "core_maize_tts",
    "core_maize_donor",
    "core_maize_acceptor",
    "core_tomato_tis",
    "core_tomato_tts",
    "core_tomato_donor",
    "core_tomato_acceptor",
    "sv_impact",
]
TASK_LABELS = [
    "Conservation · Andropogoneae genome",
    "Conservation · Poaceae non-TIS",
    "Conservation · Poaceae TIS",
    "Motif · Maize start",
    "Motif · Maize stop",
    "Motif · Maize donor",
    "Motif · Maize acceptor",
    "Motif · Tomato start",
    "Motif · Tomato stop",
    "Motif · Tomato donor",
    "Motif · Tomato acceptor",
    "Core · Maize start",
    "Core · Maize stop",
    "Core · Maize donor",
    "Core · Maize acceptor",
    "Core · Tomato start",
    "Core · Tomato stop",
    "Core · Tomato donor",
    "Core · Tomato acceptor",
    "SV · Impact",
]
GROUPS = ((0, 3), (3, 11), (11, 19), (19, 20))


def result_values(path: Path) -> np.ndarray:
    result = json.loads(path.read_text())
    rows = {row["key"]: float(row["selected_value"]) for row in result["rows"]}
    missing = [key for key in RESULT_KEYS if key not in rows]
    extra = sorted(set(rows) - set(RESULT_KEYS))
    if missing or extra:
        raise ValueError(f"Unexpected result rows in {path}: missing={missing}, extra={extra}")
    return np.asarray([rows[key] for key in RESULT_KEYS])


def published_values(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open()))
    series = []
    for model, context in BASELINE_MODELS.values():
        by_task = {row["task_id"]: float(row["value"]) for row in rows if row["model"] == model and row["context_bp"] == context}
        missing = [task_id for task_id in TASK_IDS if task_id not in by_task]
        if missing:
            raise ValueError(f"Missing published rows for {model} at {context} bp: {missing}")
        series.append([by_task[task_id] for task_id in TASK_IDS])
    return np.asarray(series)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--sampled-result", type=Path, required=True)
    parser.add_argument("--leaderboard-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full = result_values(args.full_result)
    sampled = result_values(args.sampled_result)
    published = published_values(args.leaderboard_results)
    y = np.arange(len(TASK_LABELS))
    full_ranks = 1 + np.sum(published > full, axis=0)
    sampled_ranks = 1 + np.sum(published > sampled, axis=0)
    changed_ranks = int(np.count_nonzero(full_ranks != sampled_ranks))
    rank_shift = np.abs(full_ranks - sampled_ranks)
    rank_shift_note = "all changes were one place" if np.max(rank_shift) == 1 else f"largest change was {int(np.max(rank_shift))} places"

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(13.2, 10.6))

    for group_index, (start, stop) in enumerate(GROUPS):
        if group_index % 2 == 0:
            ax.axhspan(start - 0.5, stop - 0.5, color="#F5F5F5", zorder=0)
        if start:
            ax.axhline(start - 0.5, color="#CFCFCF", linewidth=0.9, zorder=1)

    published_min = published.min(axis=0)
    published_max = published.max(axis=0)
    ax.hlines(y, published_min, published_max, color="#A7A7A7", linewidth=2.2, zorder=2)
    for values in published:
        ax.scatter(values, y, s=24, color="#A7A7A7", alpha=0.72, linewidths=0, zorder=3)

    ax.hlines(y, np.minimum(full, sampled), np.maximum(full, sampled), color="#D98372", linewidth=2.0, zorder=4)
    ax.scatter(full, y, s=74, color="#8C2D27", edgecolor="white", linewidth=0.8, zorder=6)
    ax.scatter(sampled, y, s=72, marker="D", facecolor="#F28E2B", edgecolor="white", linewidth=0.8, zorder=5)

    deltas = full - sampled
    for row, delta in enumerate(deltas):
        ax.text(1.015, row, f"{delta:+.3f}", va="center", ha="left", fontsize=9.2, color="#555555", clip_on=False)
    ax.text(1.015, -0.92, "Δ full−10k", va="bottom", ha="left", fontsize=9.2, fontweight="bold", color="#555555", clip_on=False)

    ax.set_yticks(y, TASK_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0.13, 1.0)
    ax.set_xticks(np.arange(0.2, 1.01, 0.1))
    ax.set_xlabel("Task score")
    ax.set_title("MarinDNA 1B 0.56T — task-level sensitivity to sample size", pad=30)
    ax.text(0.5, 1.005, f"Rank changed on {changed_ranks}/{len(TASK_LABELS)} tasks ({len(TASK_LABELS) - changed_ranks} unchanged; {rank_shift_note})", transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color="#555555")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)

    legend = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="#8C2D27", markeredgecolor="white", label="MarinDNA full split"),
        mlines.Line2D([], [], marker="D", linestyle="none", markersize=7.5, markerfacecolor="#F28E2B", markeredgecolor="white", label="MarinDNA 10k/task, seed 0"),
        mlines.Line2D([], [], marker="o", color="#A7A7A7", linewidth=2.2, markersize=5, label=f"{published.shape[0]} published models (full split)"),
    ]
    ax.legend(handles=legend, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.075), frameon=False)
    fig.text(0.5, 0.015, "Gray dots and ranges show PlantCAD2.5-Large, PlantCAD2-Large, PlantCAD2-Medium, PlantCAD2-Small, PlantCAD, and evo2_20b; task rank compares MarinDNA with those six models. Δ = full − 10k.\nMetrics are AUROC except motif accuracy and SV AUPRC. PlantCAD uses 512 bp; MarinDNA and the other published models use 8,192 bp.", ha="center", fontsize=10.5, color="#444444")
    fig.tight_layout(rect=(0, 0.09, 0.96, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")

    print(f"max_abs_delta={np.max(np.abs(deltas)):.12f}")
    print(f"mean_abs_delta={np.mean(np.abs(deltas)):.12f}")
    print(f"changed_ranks={changed_ranks}/{len(TASK_LABELS)}")
    print(f"max_abs_rank_shift={int(np.max(rank_shift))}")


if __name__ == "__main__":
    main()
