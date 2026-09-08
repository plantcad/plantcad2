#!/usr/bin/env python3
"""Plot AF correlation versus requested sample size for the three checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import HISTORICAL_10K, SAMPLE_SIZES


CHECKPOINTS = (
    ("022t", "MarinDNA 1B 0.22T", "#b7a0d2"),
    ("039t", "MarinDNA 1B 0.39T", "#8b63ad"),
    ("056t", "MarinDNA 1B 0.56T", "#5b2a86"),
)
HISTORICAL = (
    ("PlantCaduceus_l32", "PlantCaduceus L32", "o"),
    ("PlantCAD2-Small", "PlantCAD2-Small", "s"),
    ("PlantCAD2-Medium", "PlantCAD2-Medium", "D"),
    ("PlantCAD2-Large", "PlantCAD2-Large", "P"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="NAME=RESULT.JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(value.split("=", 1) for value in args.result)
    results = {name: json.loads(Path(path).read_text()) for name, path in paths.items()}
    if set(results) != {name for name, _, _ in CHECKPOINTS}:
        raise ValueError(f"Expected results for {[name for name, _, _ in CHECKPOINTS]}")
    actual_rows = [results["056t"]["metrics"][str(size)]["actual_rows"] for size in SAMPLE_SIZES]
    if any([results[name]["metrics"][str(size)]["actual_rows"] for size in SAMPLE_SIZES] != actual_rows for name, _, _ in CHECKPOINTS):
        raise ValueError("Sample inventories differ across checkpoints")

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True)
    x = list(range(1, len(SAMPLE_SIZES) + 1))
    for axis, metric, title in zip(axes, ("spearman", "pearson"), ("Spearman correlation", "Pearson correlation"), strict=True):
        for name, label, color in CHECKPOINTS:
            values = [results[name]["metrics"][str(size)]["llr_acgt_avg"][metric] for size in SAMPLE_SIZES]
            axis.plot(x, values, color=color, marker="o", linewidth=2.4, markersize=6, label=label, zorder=3)
        for key, label, marker in HISTORICAL:
            axis.scatter(0, HISTORICAL_10K[key][metric], color="#31866b", marker=marker, s=58, label=f"{label} (historical 10k)", zorder=4)
        axis.axvline(0.5, color="#c8c8c8", linewidth=1)
        axis.grid(axis="y", color="#e8e8e8", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Correlation with allele frequency")
        tick_labels = ["Historical 10k\nn=9,999", *[f"{size // 1000}k target\nn={actual:,}" for size, actual in zip(SAMPLE_SIZES, actual_rows, strict=True)]]
        axis.set_xticks([0, *x], tick_labels)
        axis.set_xlim(-0.35, 4.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.3, handletextpad=0.5)
    figure.suptitle("Maize allele-frequency correlation across sample sizes", fontsize=17, fontweight="bold", y=0.995)
    figure.text(0.5, 0.915, "MarinDNA: full-sequence FP32 ACGT LLR, forward/reverse-complement averaged · green points: published legacy 10k results", ha="center", color="#555555", fontsize=11.5)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.28, wspace=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
