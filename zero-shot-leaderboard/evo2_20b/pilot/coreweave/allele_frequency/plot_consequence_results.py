#!/usr/bin/env python3
"""Plot within-consequence maize-AF correlations from retained predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import CONSEQUENCES, EXPECTED_SAMPLE_ROWS


LABELS = {
    "3_prime_UTR_variant": "3′ UTR",
    "5_prime_UTR_variant": "5′ UTR",
    "grouped_splice_region": "Splice region (grouped)",
    "intergenic_variant": "Intergenic",
    "intron_variant": "Intron",
    "missense_variant": "Missense",
    "non_coding_transcript_exon_variant": "Non-coding transcript exon",
    "synonymous_variant": "Synonymous",
    "upstream_gene_variant": "Upstream gene",
    "downstream_gene_variant": "Downstream gene",
    "grouped_start_stop": "Start/stop (grouped)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.predictions)
    frame = frame.loc[frame["in_sample_100000"]].copy()
    required = ["row_id", "chrom", "pos", "ref", "alt", "AF", "consequence", "llr_acgt_avg"]
    if len(frame) != EXPECTED_SAMPLE_ROWS[100_000] or frame[required].isna().any().any() or frame["row_id"].duplicated().any() or frame[["chrom", "pos", "ref", "alt"]].duplicated().any():
        raise ValueError("Predictions do not contain the expected unique, complete 100k-target sample")
    if not np.isfinite(frame[["AF", "llr_acgt_avg"]].to_numpy()).all() or set(frame["consequence"]) != set(CONSEQUENCES):
        raise ValueError("Predictions contain unexpected values or consequence groups")
    rows = []
    for consequence, group in frame.groupby("consequence"):
        rows.append({"consequence": consequence, "n": len(group), "spearman": float(spearmanr(group["AF"], group["llr_acgt_avg"]).statistic), "pearson": float(pearsonr(group["AF"], group["llr_acgt_avg"]).statistic)})
    rows.sort(key=lambda row: row["spearman"], reverse=True)
    pooled = float(spearmanr(frame["AF"], frame["llr_acgt_avg"]).statistic)

    plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    figure, axis = plt.subplots(figsize=(9.6, 5.0))
    y = np.arange(len(rows))
    values = [row["spearman"] for row in rows]
    axis.barh(y, values, color="#6a3294", height=0.62, zorder=3)
    axis.axvline(pooled, color="#666666", linestyle="--", linewidth=1.7, zorder=2)
    for index, row in enumerate(rows):
        axis.text(row["spearman"] + 0.004, index, f'{row["spearman"]:.4f}  n={row["n"]:,}', va="center", color="#3f2352", fontweight="bold", fontsize=10.5)
    axis.set_yticks(y, [LABELS[row["consequence"]] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 0.33)
    axis.set_xlabel("Spearman correlation with allele frequency", labelpad=6)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.grid(axis="x", color="#e6e6e6", linewidth=0.8, zorder=0)
    figure.suptitle("Maize allele-frequency correlation by consequence", fontsize=17, fontweight="bold", y=0.985)
    figure.text(0.5, 0.91, "MarinDNA 1B 0.56T · 8,192-bp context · fixed 100k-target sample (n=94,075)", ha="center", color="#555555", fontsize=11.5)
    figure.text(0.5, 0.015, f"Within-consequence Spearman ρ · dashed line: pooled whole-sample ρ={pooled:.4f} (not the mean of the bars)", ha="center", color="#555555", fontsize=10.5)
    figure.subplots_adjust(left=0.29, right=0.98, top=0.84, bottom=0.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
