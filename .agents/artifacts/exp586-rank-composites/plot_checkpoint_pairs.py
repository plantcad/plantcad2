"""Render six paired zero-shot/probe metrics against tokens and W&B progress."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
from artifact_io import normalize_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
PALETTE_PATH = HERE / "figure-palette.json"
PALETTE = json.loads(PALETTE_PATH.read_text())
RUNS = ["216B-token run", "1T-token run"]
METRICS = [
    ("Avg normalized score", "Avg normalized score", "Normalized score"),
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
        "axes.facecolor": "white", "figure.facecolor": "white",
        "svg.fonttype": "none", "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(13.0, 14.0))
    axes = np.empty((4, 3), dtype=object)
    for pair_row in range(2):
        for column in range(3):
            for method_index in range(2):
                row = 2 * pair_row + method_index
                axes[row, column] = fig.add_subplot(4, 3, row * 3 + column + 1)
    colors = [PALETTE["marin"][key] for key in ["216b", "1t"]]
    seen = set()
    for index, (metric, _, unit) in enumerate(METRICS):
        pair_row, column = divmod(index, 3)
        for method_index, readout in enumerate(["Zero-shot", "Linear probe"]):
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
            ax.grid(axis="y", color="#e4e4e4", linewidth=.8)
            ax.margins(y=.12)
            if metric == "Avg normalized score":
                ax.set_ylim(-.035, 1.035)
                ax.set_yticks([0, .25, .5, .75, 1])
    assert len(seen) == 120
    title = fig.suptitle("Zero-shot scores and frozen linear probes", va="top")
    legend = fig.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label)
                 for color, label in zip(colors, ["216B tokens", "1T tokens"], strict=True)],
        title="MoE training horizon", title_fontsize=9.5, prop={"size": 12, "weight": "normal"},
        ncols=2, loc="upper center", borderaxespad=0,
        frameon=True, facecolor="white", edgecolor="#b8c0c9", framealpha=1,
        borderpad=.65, labelspacing=.45, columnspacing=2.0,
        handlelength=1.8, handleheight=.9, handletextpad=.65,
    )
    # Measure text in inches, then give every card the same narrow physical gap.
    # Wider plot boxes fill the cards instead of leaving square-axis side gutters.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [[ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
              for ax in row] for row in axes]
    positions = [[ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                  for ax in row] for row in axes]
    left_pad = max(positions[r][c].x0 - boxes[r][c].x0
                   for r in range(4) for c in range(3)) + .10
    right_pad = max(boxes[r][c].x1 - positions[r][c].x1
                    for r in range(4) for c in range(3)) + .10
    bottom_pad = max(positions[r][c].y0 - boxes[r][c].y0
                     for r in range(4) for c in range(3)) + .10
    title_pad = max(boxes[r][c].y1 - positions[r][c].y1
                    for r in range(4) for c in range(3))
    title_height = title.get_window_extent(renderer).height / fig.dpi
    legend_height = legend.get_window_extent(renderer).height / fig.dpi
    width, margin, gap, plot_height, heading_band = 13.0, .12, .22, 2.20, .40
    title_legend_gap, legend_cards_gap = .08, .14
    header = margin + title_height + title_legend_gap + legend_height + legend_cards_gap
    inner_gap = bottom_pad + title_pad + .10
    card_width = (width - 2 * margin - 2 * gap) / 3
    card_height = heading_band + title_pad + 2 * plot_height + inner_gap + bottom_pad
    height = header + 2 * card_height + gap + margin
    plot_width = card_width - left_pad - right_pad
    assert plot_width > plot_height
    fig.set_size_inches(width, height)
    title.set_position((.5, 1 - margin / height))
    legend.set_bbox_to_anchor((.5, 1 - (margin + title_height + title_legend_gap) / height))
    cards = []
    for index, (_, heading, _) in enumerate(METRICS):
        pair_row, col = divmod(index, 3)
        left = margin + col * (card_width + gap)
        top = height - header - pair_row * (card_height + gap)
        bottom = top - card_height
        for method_index in range(2):
            ax_bottom = bottom + bottom_pad + (1 - method_index) * (plot_height + inner_gap)
            axes[2 * pair_row + method_index, col].set_position(
                [(left + left_pad) / width, ax_bottom / height,
                 plot_width / width, plot_height / height])
        left, right, top, bottom = left / width, (left + card_width) / width, top / height, bottom / height
        fig.add_artist(FancyBboxPatch((left, bottom), right-left, top-bottom,
                                     transform=fig.transFigure, boxstyle="round,pad=0",
                                     facecolor="white", edgecolor="#cbd2d9", linewidth=1,
                                     zorder=-1, clip_on=False))
        fig.text((left+right)/2, top-.09/height, heading, ha="center", va="top",
                 weight="bold", fontsize="large")
        cards.append({"metric": heading, "left": left, "right": right,
                      "top": top, "bottom": bottom})
    gaps = [cards[row*3+col+1]["left"] - cards[row*3+col]["right"]
            for row in range(2) for col in range(2)]
    assert min(gaps) > .005 and max(gaps)-min(gaps) < 1e-8, gaps
    assert cards[3]["top"] < cards[0]["bottom"], cards
    vertical_gap = (cards[0]["bottom"] - cards[3]["top"]) * height
    assert np.allclose([g * width for g in gaps], vertical_gap, atol=1e-10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for index, card in enumerate(cards):
        pair_row, col = divmod(index, 3)
        panel_boxes = []
        for method_index in range(2):
            box = axes[2 * pair_row + method_index, col].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            assert card["left"] < box.x0 < box.x1 < card["right"], (mode, card, box)
            assert card["bottom"] < box.y0 < box.y1 < card["top"] - .35 / height, (mode, card, box)
            panel_boxes.append(box)
        assert panel_boxes[0].y0 > panel_boxes[1].y1, (mode, card, panel_boxes)
    title_box = title.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    assert title_box.y0 > legend_box.y1
    assert legend_box.y0 / fig.dpi > cards[0]["top"] * height
    name = "checkpoint-tokens" if mode == "tokens" else "checkpoint-progress"
    for extension in ("svg", "png"):
        export = {"pil_kwargs": {"optimize": True}} if extension == "png" else {}
        fig.savefig(HERE / "figures" / f"{name}.{extension}", dpi=156,
                    facecolor="white", bbox_inches="tight", pad_inches=.12, **export)
    normalize_svg(HERE / "figures" / f"{name}.svg")
    plt.close(fig)
    (HERE / ("line-figure-provenance.json" if mode == "tokens" else "progress-figure-provenance.json")).write_text(json.dumps({
        "source_sha256": hashlib.sha256((HERE/"normalized-category-scores.csv").read_bytes()).hexdigest(),
        "palette_sha256": hashlib.sha256(PALETTE_PATH.read_bytes()).hexdigest(),
        "run_colors": dict(zip(RUNS, colors, strict=True)),
        "progress_sha256": hashlib.sha256((HERE/"run-progress.csv").read_bytes()).hexdigest(),
        "panels": 12, "score_points": len(seen), "card_geometry": cards,
        "card_style": {"facecolor": "white", "edgecolor": "#cbd2d9", "linewidth": 1},
        "horizontal_gaps": gaps,
        "facet_gap_inches": {"horizontal": [g * width for g in gaps], "vertical": vertical_gap},
        "plot_size_inches": [plot_width, plot_height],
        "figure_size_inches": [width, height],
        "header_gaps_inches": {"title_to_legend": title_legend_gap, "legend_to_facets": legend_cards_gap},
        "model_legend": {"title": "MoE training horizon", "labels": ["216B tokens", "1T tokens"],
                         "handles": "Filled color swatches", "label_fontsize": 12, "label_weight": "normal",
                         "frame": "White with a thin gray border"},
        "layout": "Top: Avg normalized score, Conservation, Motifs; bottom: Core/non-core, SV, AF; zero-shot above probe in each pair",
        "aggregate": "Min–max scale each saved family mean across both runs’ checkpoints, separately for each readout, then average the five normalized families equally, including AF",
        "x_axis": mode, "smoothing": False, "y_scales": "Independent per panel",
    }, indent=2) + "\n")


def main() -> None:
    with (HERE / "normalized-category-scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (HERE / "run-progress.csv").open() as stream:
        progress = {(row["run"], row["checkpoint"]): row for row in csv.DictReader(stream)}
    assert len(rows) == 120 and len(progress) == 10
    for mode in ("tokens", "progress"):
        render(rows, progress, mode)


if __name__ == "__main__":
    main()
