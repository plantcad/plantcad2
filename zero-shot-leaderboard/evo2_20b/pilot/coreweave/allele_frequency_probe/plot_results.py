#!/usr/bin/env python3
"""Plot held-out zero-shot and frozen-embedding maize-AF results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


CHECKPOINTS = (
    ("022t", "0.22T"),
    ("039t", "0.39T"),
    ("056t", "0.56T"),
)
REPRESENTATIONS = (
    ("whole_window", "Whole-window probe", "#5b2a86", "o"),
    ("variant_token", "Variant-token probe", "#d46b27", "s"),
)
CODING_CONSEQUENCES = {
    "grouped_start_stop",
    "missense_variant",
    "synonymous_variant",
}
CONSEQUENCE_GROUPS = (
    ("CODING", "#b45f06"),
    ("NONCODING", "#267a78"),
)
LABELS = {
    "3_prime_UTR_variant": "3′ UTR",
    "5_prime_UTR_variant": "5′ UTR",
    "downstream_gene_variant": "Downstream",
    "grouped_splice_region": "Splice region",
    "grouped_start_stop": "Start/stop",
    "intergenic_variant": "Intergenic",
    "intron_variant": "Intron",
    "missense_variant": "Missense",
    "non_coding_transcript_exon_variant": "Non-coding exon",
    "synonymous_variant": "Synonymous",
    "upstream_gene_variant": "Upstream",
}


def _ci(metric: dict) -> tuple[float, float, float]:
    value = metric["value"]
    low, high = metric["ci95"]
    return value, value - low, high - value


def _mark_consequence_groups(axis: plt.Axes, consequences: list[str], *, show_rail: bool = False) -> None:
    coding_count = sum(consequence in CODING_CONSEQUENCES for consequence in consequences)
    if any(consequence not in CODING_CONSEQUENCES for consequence in consequences[:coding_count]):
        raise ValueError("Coding consequences must precede noncoding consequences")
    axis.axhline(coding_count - 0.5, color="white", linewidth=3.2)
    axis.axhline(coding_count - 0.5, color="#454545", linewidth=0.8)
    if not show_rail:
        return
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    ranges = ((-0.5, coding_count - 0.5), (coding_count - 0.5, len(consequences) - 0.5))
    for (label, color), (start, stop) in zip(CONSEQUENCE_GROUPS, ranges, strict=True):
        axis.add_patch(Rectangle((-0.45, start + 0.05), 0.018, stop - start - 0.1, transform=transform, clip_on=False, facecolor=color, edgecolor="none"))
        axis.text(-0.49, (start + stop) / 2, label, transform=transform, rotation=90, ha="center", va="center", color=color, fontsize=7.2, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="NAME=RESULT.JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(item.split("=", 1) for item in args.result)
    results = {name: json.loads(Path(path).read_text()) for name, path in paths.items()}
    if set(results) != {name for name, _ in CHECKPOINTS}:
        raise ValueError("Expected .22T, .39T, and .56T result files")
    split_hashes = {result["split"]["sha256"] for result in results.values()}
    if len(split_hashes) != 1 or {result["split"]["rows"]["test"] for result in results.values()} != {28_378}:
        raise ValueError("Results do not share the expected held-out split")

    all_consequences = list(results["022t"]["metrics"]["whole_window"]["by_consequence"])

    def zero_shot_mean(consequence: str) -> float:
        return float(np.mean([
            results[name]["metrics"]["whole_window"]["by_consequence"][consequence]["zero_shot"]["value"]
            for name, _ in CHECKPOINTS
        ]))

    consequences = sorted((name for name in all_consequences if name in CODING_CONSEQUENCES), key=zero_shot_mean, reverse=True)
    consequences += sorted((name for name in all_consequences if name not in CODING_CONSEQUENCES), key=zero_shot_mean, reverse=True)
    deltas = {
        representation: np.asarray([
            [results[name]["metrics"][representation]["by_consequence"][consequence]["delta"]["value"] for name, _ in CHECKPOINTS]
            for consequence in consequences
        ])
        for representation, *_ in REPRESENTATIONS
    }
    limit = max(abs(float(matrix.min())) for matrix in deltas.values())
    limit = max(limit, max(abs(float(matrix.max())) for matrix in deltas.values()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(12.8, 5.25))
    outer = figure.add_gridspec(1, 2, width_ratios=(1.32, 2), wspace=0.38)
    trend = figure.add_subplot(outer[0, 0])
    heat_grid = outer[0, 1].subgridspec(1, 2, wspace=0.16)
    heat_whole = figure.add_subplot(heat_grid[0, 0])
    heat_variant = figure.add_subplot(heat_grid[0, 1], sharey=heat_whole)
    x = np.arange(len(CHECKPOINTS))

    zero_metrics = [results[name]["metrics"]["whole_window"]["overall"]["zero_shot"] for name, _ in CHECKPOINTS]
    values = np.asarray([_ci(metric)[0] for metric in zero_metrics])
    errors = np.asarray([_ci(metric)[1:] for metric in zero_metrics]).T
    trend.errorbar(x, values, yerr=errors, label="Zero-shot LLR", color="#666666", marker="D", linestyle="--", linewidth=2.2, capsize=3, zorder=3)
    for representation, label, color, marker in REPRESENTATIONS:
        metrics = [results[name]["metrics"][representation]["overall"]["probe"] for name, _ in CHECKPOINTS]
        values = np.asarray([_ci(metric)[0] for metric in metrics])
        errors = np.asarray([_ci(metric)[1:] for metric in metrics]).T
        trend.errorbar(x, values, yerr=errors, label=label, color=color, marker=marker, linewidth=2.4, capsize=3, zorder=4)
    trend.set_xticks(x, [label for _, label in CHECKPOINTS])
    trend.set_xlabel("Post-cooldown training tokens")
    trend.set_ylabel("Held-out Spearman correlation with AF")
    trend.set_title("Pooled held-out score", fontweight="bold", fontsize=12.5, pad=8)
    trend.grid(axis="y", color="#e8e8e8", linewidth=0.8, zorder=0)
    trend.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.73, 0.01), fontsize=8.4, handlelength=1.6, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.4)

    for axis, matrix, title in ((heat_whole, deltas["whole_window"], "Whole-window Δρ"), (heat_variant, deltas["variant_token"], "Variant-token Δρ")):
        axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        axis.set_title(title, fontweight="bold", fontsize=11.5, pad=8)
        axis.set_xticks(x, [label for _, label in CHECKPOINTS])
        axis.set_xlabel("Training tokens")
        axis.tick_params(length=0)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if norm(value) < 0.22 or norm(value) > 0.78 else "#111111"
                axis.text(column, row, f"{value:+.3f}", ha="center", va="center", color=color, fontsize=8.7, fontweight="bold")
    heat_whole.set_yticks(np.arange(len(consequences)), [LABELS[name] for name in consequences])
    heat_whole.tick_params(axis="y", labelsize=8.6, pad=4)
    heat_variant.tick_params(axis="y", labelleft=False)
    _mark_consequence_groups(heat_whole, consequences, show_rail=True)
    _mark_consequence_groups(heat_variant, consequences)

    figure.suptitle("Maize allele-frequency prediction: zero-shot vs frozen-embedding probes", fontsize=15.2, fontweight="bold", y=0.985)
    figure.text(0.5, 0.918, "MarinDNA 1B primary lineage · 8,192-bp context · fixed seed-0 genomic-block split", ha="center", color="#555555", fontsize=10.2)
    figure.text(0.5, 0.012, "Pooled n=28,378; error bars: paired 1 Mb block-bootstrap 95% CIs · heatmaps: within-consequence Δρ vs zero-shot (blue improves, red declines)", ha="center", color="#555555", fontsize=9.4)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    raw = {
        "zero_shot": np.asarray([
            [results[name]["metrics"]["whole_window"]["by_consequence"][consequence]["zero_shot"]["value"] for name, _ in CHECKPOINTS]
            for consequence in consequences
        ]),
        "whole_window": np.asarray([
            [results[name]["metrics"]["whole_window"]["by_consequence"][consequence]["probe"]["value"] for name, _ in CHECKPOINTS]
            for consequence in consequences
        ]),
        "variant_token": np.asarray([
            [results[name]["metrics"]["variant_token"]["by_consequence"][consequence]["probe"]["value"] for name, _ in CHECKPOINTS]
            for consequence in consequences
        ]),
    }
    raw_norm = Normalize(vmin=0, vmax=max(0.35, max(float(matrix.max()) for matrix in raw.values())))
    raw_figure = plt.figure(figsize=(12.4, 5.15))
    raw_grid = raw_figure.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.045), wspace=0.16)
    raw_axes = [raw_figure.add_subplot(raw_grid[0, 0])]
    raw_axes += [raw_figure.add_subplot(raw_grid[0, index], sharey=raw_axes[0]) for index in (1, 2)]
    colorbar_axis = raw_figure.add_subplot(raw_grid[0, 3])
    raw_image = None
    for axis, (key, title) in zip(raw_axes, (("zero_shot", "Zero-shot LLR"), ("whole_window", "Whole-window probe"), ("variant_token", "Variant-token probe")), strict=True):
        matrix = raw[key]
        raw_image = axis.imshow(matrix, cmap="YlGnBu", norm=raw_norm, aspect="auto")
        axis.set_title(title, fontweight="bold", fontsize=12, pad=8)
        axis.set_xticks(x, [label for _, label in CHECKPOINTS])
        axis.set_xlabel("Training tokens")
        axis.tick_params(length=0)
        _mark_consequence_groups(axis, consequences, show_rail=axis is raw_axes[0])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if raw_norm(value) > 0.58 else "#111111"
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8.6, fontweight="bold")
    raw_axes[0].set_yticks(np.arange(len(consequences)), [LABELS[name] for name in consequences])
    raw_axes[0].tick_params(axis="y", labelsize=8.8, pad=4, labelleft=True)
    for axis in raw_axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    assert raw_image is not None
    colorbar = raw_figure.colorbar(raw_image, cax=colorbar_axis)
    colorbar.set_label("Spearman ρ", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.5)
    raw_figure.suptitle("Maize allele-frequency prediction by consequence and training scale", fontsize=15.2, fontweight="bold", y=0.985)
    raw_figure.text(0.5, 0.918, "Raw held-out Spearman ρ · MarinDNA 1B primary lineage · 8,192-bp context", ha="center", color="#555555", fontsize=10.2)
    raw_figure.text(0.5, 0.012, "Same fixed 28,378-variant test split; one global probe per checkpoint · shared color scale across all panels", ha="center", color="#555555", fontsize=9.4)
    raw_figure.subplots_adjust(left=0.19, right=0.96, top=0.82, bottom=0.15)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_figure.savefig(args.raw_output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(raw_figure)


if __name__ == "__main__":
    main()
