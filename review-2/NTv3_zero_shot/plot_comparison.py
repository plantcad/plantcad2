import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = [
    "NTv3-650M",
    "PlantCAD2\nSmall",
    "PlantCAD2\nMedium",
    "PlantCAD2\nLarge",
]

model_colors = [
    "#888888",  # NTv3: gray
    "#A8C8E8",  # Small: light blue
    "#4472C4",  # Medium: medium blue
    "#1A3A6B",  # Large: dark blue
]

tasks = [
    "conservation\nandropogoneae\n(AUROC)",
    "conservation\npoaceae TIS\n(AUROC)",
    "conservation\npoaceae non-TIS\n(AUROC)",
    "motif recovery\nmaize TIS\n(motif acc.)",
    "motif recovery\nmaize TTS\n(motif acc.)",
    "motif recovery\nmaize acceptor\n(motif acc.)",
    "motif recovery\nmaize donor\n(motif acc.)",
    "core maize TIS\n(AUROC)",
    "core maize TTS\n(AUROC)",
    "core maize acceptor\n(AUROC)",
    "core maize donor\n(AUROC)",
    "SV effect\n(AUPRC)",
]

nan = float("nan")

# NTv3-650M values from NTv3_650M_performance.md (zero-shot, test_maize split)
# PlantCAD2 values from plantcad2_performance.md (excluding baseline)
data = [
    # NTv3-650M  Small   Medium   Large
    [0.612,      0.656,  0.708,   0.725],   # conservation andropogoneae (AUROC)
    [0.613,      0.632,  0.665,   0.670],   # conservation poaceae TIS (AUROC)
    [0.684,      0.646,  0.687,   0.713],   # conservation poaceae non-TIS (AUROC)
    [0.288,      0.545,  0.633,   0.657],   # motif recovery maize TIS (motif acc.)
    [0.114,      0.230,  0.360,   0.410],   # motif recovery maize TTS (motif acc.)
    [0.560,      0.854,  0.890,   0.900],   # motif recovery maize acceptor (motif acc.)
    [0.686,      0.875,  0.903,   0.910],   # motif recovery maize donor (motif acc.)
    [0.656,      0.707,  0.710,   0.696],   # core maize TIS (AUROC)
    [0.577,      0.598,  0.618,   0.608],   # core maize TTS (AUROC)
    [0.627,      0.763,  0.829,   0.836],   # core maize acceptor (AUROC)
    [0.651,      0.764,  0.804,   0.808],   # core maize donor (AUROC)
    [0.642,      0.795,  0.833,   0.841],   # SV effect prediction (AUPRC)
]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

x = np.arange(len(models))
bar_width = 0.6

for i, (task, values) in enumerate(zip(tasks, data)):
    ax = axes[i]
    for j, (val, color) in enumerate(zip(values, model_colors)):
        if not np.isnan(val):
            ax.bar(j, val, width=bar_width, color=color)
            ax.text(
                j, val + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold"
            )
    ax.set_title(task, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7, rotation=30, ha="right")
    valid = [v for v in values if not np.isnan(v)]
    if valid:
        ymin = min(valid) * 0.97
        ymax = max(valid) * 1.08
        ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=c, label=m.replace("\n", " "))
    for m, c in zip(models, model_colors)
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    fontsize=9,
    title="Model",
    bbox_to_anchor=(0.5, 0.01),
)

fig.suptitle(
    "NTv3-650M (zero-shot) vs PlantCAD2 Benchmark Results",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight")
plt.savefig("comparison_plot.pdf", bbox_inches="tight")
print("Saved comparison_plot.png and comparison_plot.pdf")