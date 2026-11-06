"""Compare alternatives for dense MarinDNA without changing the draft palette."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
OPTIONS = [("Coral", "#e76f51"), ("Charcoal", "#505050"), ("Light rose", "#e690a8")]


def main() -> None:
    palette = json.loads((HERE / "figure-palette.json").read_text())["marin"]
    baseline = [model["color"] for model in json.loads((HERE / "benchmark-comparison-data.json").read_text())["models"][3:]]
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(-1.5, 8.25)
    ax.set_ylim(-.35, 3.5)
    ax.axis("off")
    columns = [(0, "Dense MarinDNA"), (1.7, "MoE 216B"), (3.4, "MoE 1T"), (6.2, "Baselines (unchanged)")]
    for x, name in columns:
        ax.text(x + .60, 3.10, name, ha="center", va="center")
    for index, (name, color) in enumerate(OPTIONS):
        y = 2.1 - index
        ax.text(-.20, y + .27, name, ha="right", va="center")
        for x, fill in [(0, color), (1.7, palette["216b"]), (3.4, palette["1t"])]:
            ax.add_patch(Rectangle((x, y), 1.2, .55, facecolor=fill))
        ax.text(.60, y - .10, color.upper(), ha="center", va="top", fontsize=10, color="#555555")
        for position, fill in enumerate(baseline):
            ax.add_patch(Rectangle((5.35 + position * .61, y), .55, .55, facecolor=fill))
    fig.subplots_adjust(left=.03, right=.99, top=.96, bottom=.08)
    for extension in ["svg", "png"]:
        fig.savefig(HERE / "figures" / f"palette-options.{extension}", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
