"""Render six paired zero-shot/probe metrics against tokens and W&B progress."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


HERE = Path(__file__).resolve().parent
RUNS = ["216B-token run", "1T-token run"]
METRICS = [
    ("Composite +SV", "Composite", "Mean category score"),
    ("Conservation", "Conservation", "AUROC"),
    ("Motifs", "Motifs", "Accuracy"),
    ("Core/non-core", "Core/non-core", "AUROC"),
    ("SV impact", "SV impact", "Average precision"),
    ("Allele frequency", "Allele frequency", "Spearman correlation"),
]


def render(rows: list[dict], progress: dict, mode: str) -> None:
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": "medium",
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(13.0, 15.0))
    outer = fig.add_gridspec(2, 3, left=.075, right=.99, bottom=.045, top=.85,
                            wspace=.42, hspace=.33)
    axes = np.empty((4, 3), dtype=object)
    for pair_row in range(2):
        for column in range(3):
            inner = outer[pair_row, column].subgridspec(2, 1, hspace=.50)
            for method_index in range(2):
                axes[2 * pair_row + method_index, column] = fig.add_subplot(inner[method_index])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    seen = set()
    for index, (metric, _, unit) in enumerate(METRICS):
        pair_row, column = divmod(index, 3)
        for method_index, readout in enumerate(["Zero-shot likelihood", "Linear probe"]):
            ax = axes[2 * pair_row + method_index, column]
            for run, color in zip(RUNS, colors, strict=True):
                points = sorted((row for row in rows if row["run"] == run
                                 and row["metric"] == metric and row["readout"] == readout),
                                key=lambda row: float(row["DNA_tokens_B"]))
                assert len(points) == 5
                for point in points:
                    key = (run, point["checkpoint"], metric, readout)
                    assert key not in seen
                    seen.add(key)
                    assert float(point["DNA_tokens_B"]) == float(progress[run, point["checkpoint"]]["DNA_tokens_B"])
                x = [float(point["DNA_tokens_B"]) if mode == "tokens"
                     else float(progress[run, point["checkpoint"]]["run_progress_percent"])
                     for point in points]
                ax.plot(x, [float(point["score"]) for point in points],
                        marker="o", markersize=4.5, linewidth=1.8, color=color, label=run)
            if mode == "tokens":
                ax.set_xscale("symlog", linthresh=10, linscale=.55)
                ax.set_xlim(-.7, 1250)
                ax.xaxis.set_major_locator(FixedLocator([0, 10, 100, 1000]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
                ax.set_xlabel("DNA tokens (billions)")
            else:
                ax.set_xlim(-3, 103)
                ax.xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
                ax.set_xlabel("Run progress")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylabel(unit)
            ax.set_title("Zero-shot" if method_index == 0 else "Linear probe")
            ax.set_box_aspect(1)
            ax.grid(axis="y", color="#e4e4e4", linewidth=.8)
            ax.margins(y=.12)
    assert len(seen) == 120
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Zero-shot scores and frozen linear probes", y=.995)
    fig.legend(handles, labels, title="Training horizon", ncols=2, frameon=False,
               loc="upper center", bbox_to_anchor=(.5, .97))
    # Nested grids reserve more separation between task pairs than within them.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [[ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
              for ax in row] for row in axes]
    left_pad = max(axes[row, col].get_position().x0 - boxes[row][col].x0
                   for row in range(4) for col in range(3))
    right_pad = max(boxes[row][col].x1 - axes[row, col].get_position().x1
                    for row in range(4) for col in range(3))
    # Work out one common card height from text extents, then align each row.
    extents = []
    for index, (_, heading, _) in enumerate(METRICS):
        pair_row, col = divmod(index, 3)
        top = boxes[2 * pair_row][col].y1 + .027
        bottom = boxes[2 * pair_row + 1][col].y0 - .009
        extents.append((top, bottom))
    height = max(top - bottom for top, bottom in extents)
    row_tops = [max(extents[index][0] for index in range(row * 3, row * 3 + 3))
                for row in range(2)]
    cards = []
    for index, (_, heading, _) in enumerate(METRICS):
        pair_row, col = divmod(index, 3)
        position = axes[2 * pair_row, col].get_position()
        left, right = position.x0 - left_pad - .006, position.x1 + right_pad + .006
        top, bottom = row_tops[pair_row], row_tops[pair_row] - height
        fig.add_artist(FancyBboxPatch((left, bottom), right-left, height,
                                     transform=fig.transFigure, boxstyle="round,pad=0",
                                     facecolor="#f4f6f8", edgecolor="#cbd2d9", linewidth=1,
                                     zorder=-1, clip_on=False))
        fig.text((left+right)/2, top-.012, heading, ha="center", va="center",
                 weight="bold", fontsize="large")
        cards.append({"metric": heading, "left": left, "right": right,
                      "top": top, "bottom": bottom})
    gaps = [cards[row*3+col+1]["left"] - cards[row*3+col]["right"]
            for row in range(2) for col in range(2)]
    assert min(gaps) > .005 and max(gaps)-min(gaps) < 1e-8, gaps
    assert cards[3]["top"] < cards[0]["bottom"], cards
    name = "checkpoint-tokens" if mode == "tokens" else "checkpoint-progress"
    for extension in ("svg", "png"):
        export = {"pil_kwargs": {"optimize": True}} if extension == "png" else {}
        fig.savefig(HERE / "figures" / f"{name}.{extension}", dpi=160,
                    facecolor="white", bbox_inches="tight", pad_inches=.12, **export)
    plt.close(fig)
    (HERE / ("line-figure-provenance.json" if mode == "tokens" else "progress-figure-provenance.json")).write_text(json.dumps({
        "source_sha256": hashlib.sha256((HERE/"all-category-scores.csv").read_bytes()).hexdigest(),
        "progress_sha256": hashlib.sha256((HERE/"run-progress.csv").read_bytes()).hexdigest(),
        "panels": 12, "score_points": len(seen), "card_geometry": cards,
        "horizontal_gaps": gaps,
        "layout": "Top: Composite, Conservation, Motifs; bottom: Core/non-core, SV, AF; zero-shot above probe in each pair",
        "composite": "Equal mean of conservation, motifs, core/non-core and SV category scores, independently for each readout; AF excluded",
        "x_axis": mode, "smoothing": False, "y_scales": "Independent per panel",
    }, indent=2) + "\n")


def main() -> None:
    with (HERE / "all-category-scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (HERE / "run-progress.csv").open() as stream:
        progress = {(row["run"], row["checkpoint"]): row for row in csv.DictReader(stream)}
    assert len(rows) == 140 and len(progress) == 10
    for mode in ("tokens", "progress"):
        render(rows, progress, mode)


if __name__ == "__main__":
    main()
