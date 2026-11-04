"""Plot completed DNA validation losses; no fitted or projected values."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "validation-loss.json").read_text())["models"]
rows = sorted(rows, key=lambda row: row["validation_loss"])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#A9B1BA",
        "text.color": "#20262E",
        "axes.labelcolor": "#404B58",
        "xtick.color": "#52606E",
        "svg.fonttype": "none",
    }
)
fig = plt.figure(figsize=(12.8, 6.7), facecolor="white")
grid = fig.add_gridspec(
    1, 3, width_ratios=[5.9, 4.65, 1.15],
    left=0.025, right=0.975, bottom=0.145, top=0.81, wspace=0.015,
)
labels, chart, values = [fig.add_subplot(grid[0, i]) for i in range(3)]

for ax in [labels, chart, values]:
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0)
    ax.axhspan(-0.46, 0.46, color="#E8F4EF", zorder=-5)
    for y in range(len(rows)):
        ax.axhline(y + 0.5, color="#EFF1F4", lw=0.8, zorder=-4)

for ax in [labels, values]:
    ax.set_xlim(0, 1)
    ax.axis("off")

for i, row in enumerate(rows):
    highlight = row["code"] == "M61-L-1T"
    color = "#087F5B" if highlight else "#355F91"
    weight = "bold" if highlight else "normal"
    labels.text(
        0.015, i, row["label"], va="center", fontsize=11.6,
        fontweight=weight, color="#087F5B" if highlight else "#20262E",
    )
    chart.scatter(
        row["validation_loss"], i, s=105 if highlight else 68,
        color=color, edgecolors="white", linewidths=0.8, zorder=5,
    )
    values.text(
        0.96, i, f'{row["validation_loss"]:.6f}', ha="right", va="center",
        fontsize=12, fontweight=weight,
        color="#087F5B" if highlight else "#20262E",
    )

chart.set_xlim(0.875, 1.055)
chart.set_xticks(np.arange(0.88, 1.041, 0.04))
chart.grid(axis="x", color="#E3E7EB", lw=0.8)
chart.tick_params(axis="x", pad=9, labelsize=11)
chart.set_xlabel("Validation cross-entropy (nats / base) · lower is better", labelpad=13, fontsize=11)
labels.text(0.015, 1.035, "MODEL · DNA TRAINING TOKENS · INITIALIZATION", transform=labels.transAxes, fontsize=10, color="#65717E", weight="bold")
values.text(0.96, 1.035, "LOSS", transform=values.transAxes, ha="right", fontsize=10, color="#65717E", weight="bold")

fig.text(0.032, 0.94, "DNA validation loss at completed checkpoints", fontsize=21, weight="bold")
fig.text(0.032, 0.885, "Same 65,536 held-out windows of 8,192 bases for every checkpoint", fontsize=13, color="#52606E")
for extension in ["png", "svg"]:
    fig.savefig(ROOT / f"validation-loss-final.{extension}", dpi=150, facecolor="white")
plt.close(fig)
