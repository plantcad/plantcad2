#!/usr/bin/env python3
"""Plot maize-AF Spearman correlation across centered context lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import CONTEXT_LENGTHS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="CONTEXT=RESULT.JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {int(context): Path(path) for context, path in (value.split("=", 1) for value in args.result)}
    if set(paths) != set(CONTEXT_LENGTHS):
        raise ValueError(f"Expected one result for each context: {CONTEXT_LENGTHS}")
    results = {context: json.loads(path.read_text()) for context, path in paths.items()}
    metrics = [results[context]["metrics"]["100000"] for context in CONTEXT_LENGTHS]
    actual_rows = {metric["actual_rows"] for metric in metrics}
    if actual_rows != {94_075}:
        raise ValueError(f"Unexpected sample inventories: {actual_rows}")
    spearman = [metric["llr_acgt_avg"]["spearman"] for metric in metrics]

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(9.6, 4.5))
    x = list(range(len(CONTEXT_LENGTHS)))
    axis.plot(x, spearman, color="#5b2a86", marker="o", linewidth=2.8, markersize=8, zorder=3)
    axis.fill_between(x, spearman, min(spearman) - 0.006, color="#5b2a86", alpha=0.07)
    for index, value in enumerate(spearman):
        axis.annotate(f"{value:.4f}", (index, value), xytext=(0, 10), textcoords="offset points", ha="center", color="#452064", fontweight="bold")
    axis.set_xticks(x, [f"{context:,}" for context in CONTEXT_LENGTHS])
    axis.set_xlabel("Centered context length (bp)", labelpad=10)
    axis.set_ylabel("Spearman correlation with allele frequency")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.set_ylim(min(spearman) - 0.006, max(spearman) + 0.011)
    figure.suptitle("Maize allele-frequency correlation by context length", fontsize=17, fontweight="bold", y=0.98)
    figure.text(0.5, 0.87, "MarinDNA 1B 0.56T · same 94,075 variants at every context · exact centered crops", ha="center", color="#555555", fontsize=12)
    figure.text(0.5, 0.015, "Score: Spearman ρ(AF, mean forward/reverse-complement full-suffix LLR)", ha="center", color="#555555", fontsize=11.5)
    figure.subplots_adjust(left=0.12, right=0.98, top=0.76, bottom=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
