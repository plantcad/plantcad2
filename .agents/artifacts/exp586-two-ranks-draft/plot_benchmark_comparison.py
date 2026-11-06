"""Draft six benchmark readouts with both average-rank definitions."""

import json
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent.with_name("exp586-rank-composites")))
from artifact_io import normalize_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
RANGES = {"Conservation": (.62, .86), "SV impact": (.72, .91),
          "Unweighted Avg Rank": (45, 102), "Weighted Avg Rank": (45, 102)}
TICKS = {"Conservation": [.65, .70, .75, .80, .85], "SV impact": [.75, .80, .85, .90],
         "Unweighted Avg Rank": [50, 60, 70, 80, 90, 100], "Weighted Avg Rank": [50, 60, 70, 80, 90, 100]}


def breaks(ax: plt.Axes, edge: float, horizontal: bool) -> None:
    """Mark the omitted interval at both ends in fixed physical dimensions."""
    box = ax.get_position()
    dx = .035 / (box.width * ax.figure.get_figwidth())
    dy = .035 / (box.height * ax.figure.get_figheight())
    for end in [0, 1]:
        x, y = (end, edge) if horizontal else (edge, end)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], transform=ax.transAxes,
                color="#333333", linewidth=1.1, clip_on=False, zorder=10)


def main() -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    data = json.loads((HERE / "benchmark-comparison-data.json").read_text())
    models, categories = data["models"], data["categories"]
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "axes.axisbelow": True})
    fig = plt.figure(figsize=(21.6, 12.1))
    # AUROC, accuracy and AP are multiplied by 100 for display only.
    # Avg Rank is already a percentile. The common break omits 5–45.
    top = fig.add_axes((.055, .666, .935, .28))
    stub = fig.add_axes((.055, .603, .935, .047), sharex=top)
    centers, width = np.arange(6), .105
    labels = []
    top_labels = []
    display_multipliers = np.array([100, 100, 100, 100, 1, 1])
    for i, model in enumerate(models):
        x = centers + (i - (len(models) - 1) / 2) * width
        scores = np.array(model["scores"]) * display_multipliers
        assert np.all((0 <= scores) & (scores <= 100))
        for axis in [top, stub]:
            bars = axis.bar(x, scores, width=width, color=model["color"], edgecolor="white", linewidth=.4, label=model["label"])
            np.testing.assert_array_equal([bar.get_height() for bar in bars], scores)
        for xpos, score in zip(x, scores, strict=True):
            text = top.annotate(f"{score:.1f}", (xpos, score), xytext=(0, 5), textcoords="offset points",
                                rotation=90, ha="center", va="bottom", fontsize=10.5)
            labels.append(text)
            top_labels.append(text)
    displayed_max = max(max(np.array(model["scores"]) * display_multipliers) for model in models)
    top.set_ylim(45, max(95, 5 * np.ceil(displayed_max / 5) + 5))
    top.set_yticks([50, 60, 70, 80, 90])
    top.set_ylabel("Metric score (×100) or percentile rank")
    title = fig.text(.5225, .989, "Zero-shot PlantCAD2 benchmark", weight="bold",
                     fontsize="large", ha="center", va="top")
    top.spines[["bottom", "top", "right"]].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    top.grid(axis="y", color="#e4e4e4", linewidth=.7)
    stub.set_ylim(0, 5)
    stub.set_yticks([0, 5])
    stub.set_xlim(-.54, 5.54)
    stub.set_xticks(centers, [label.replace(" Avg Rank", "\nAvg Rank") for label in categories])
    stub.spines[["top", "right"]].set_visible(False)
    breaks(top, 0, True)
    breaks(stub, 1, True)
    legend = fig.legend(*top.get_legend_handles_labels(), loc="center", ncols=3,
                        bbox_to_anchor=(.51, .523), frameon=False, columnspacing=2.3, handletextpad=.7)
    lower_axes, zeros = [], []
    for metric, left_inches, unit in [("Conservation", 2.02, "AUROC"),
                                       ("SV impact", 7.42, "Average precision"),
                                       ("Unweighted Avg Rank", 12.82, "Percentile rank ↑"),
                                       ("Weighted Avg Rank", 18.22, "Percentile rank ↑")]:
        idx = categories.index(metric)
        ranked = sorted(models, key=lambda m: m["scores"][idx], reverse=True)
        left, bottom, height = left_inches / 21.6, .075, .345
        zero = fig.add_axes((left, bottom, .16 / 21.6, height))
        axis = fig.add_axes((left + .28 / 21.6, bottom, 2.85 / 21.6, height), sharey=zero)
        zeros.append(zero)
        lower_axes.append(axis)
        values = [model["scores"][idx] for model in ranked]
        is_rank = "Avg Rank" in metric
        low, high = RANGES[metric]
        assert low < min(values) < max(values) < high
        for part in [zero, axis]:
            part.barh(np.arange(8), values, height=.72, color=[m["color"] for m in ranked])
        zero.set_yticks(np.arange(8), [m["label"].replace("1.5B–60M ", "1.5B–60M\n") for m in ranked], fontsize=10.5)
        zero.set_ylim(7.5, -.5)
        zero.set_xlim(0, 5 if is_rank else .05)
        zero.set_xticks([0], ["0"])
        zero.spines[["top", "right"]].set_visible(False)
        zero.tick_params(axis="y", length=0, pad=6)
        axis.set_xlim(low, high)
        axis.set_xticks(TICKS[metric])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _, is_rank=is_rank: f"{v:.0f}" if is_rank else f"{v:.2f}"))
        axis.tick_params(axis="x", labelsize=10.5)
        axis.set_title(metric, weight="bold", pad=10, fontsize=12)
        axis.set_xlabel(unit)
        axis.grid(axis="x", color="#e4e4e4", linewidth=.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", left=False, labelleft=False)
        breaks(zero, 1, False)
        breaks(axis, 0, False)
        for row, value in enumerate(values):
            value_label = f"{value:.1f}" if is_rank else f"{value:.4f}"
            labels.append(axis.annotate(value_label,
                                        (value, row), xytext=(4, 0), textcoords="offset points", rotation=0,
                                        ha="left", va="center", fontsize=10.5))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = [label.get_window_extent(renderer).padded(1) for label in labels]
    overlaps = [(labels[i].get_text(), labels[j].get_text(), a.bounds, b.bounds)
                for i, a in enumerate(label_boxes) for j, b in enumerate(label_boxes) if i < j and a.overlaps(b)]
    assert not overlaps, overlaps
    assert all(top.bbox.contains(box.x0, box.y0) and top.bbox.contains(box.x1, box.y1)
               for box in [text.get_window_extent(renderer) for text in top_labels])
    legend_box = legend.get_window_extent(renderer)
    panels = [Bbox.union([top.get_tightbbox(renderer), stub.get_tightbbox(renderer)]),
              *[Bbox.union([z.get_tightbbox(renderer), ax.get_tightbbox(renderer)]) for z, ax in zip(zeros, lower_axes, strict=True)]]
    assert not any(a.overlaps(b) for i, a in enumerate(panels) for b in panels[i + 1:])
    assert not any(box.overlaps(legend_box) for box in panels)
    assert len(top_labels) == 48
    assert sum(len(ax.patches) for ax in [top, *lower_axes]) == 80
    assert title.get_window_extent(renderer).y0 > top.bbox.y1
    for ext in ["png", "svg"]:
        fig.savefig(HERE / "figures" / f"benchmark-comparison.{ext}", dpi=156,
                    bbox_inches="tight", pad_inches=.12, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
    normalize_svg(HERE / "figures" / "benchmark-comparison.svg")
    plt.close(fig)
    (HERE / "bar-layout-validation.json").write_text(json.dumps({
        "native_top_bar_labels": 32, "rank_top_bar_labels": 16,
        "all_annotated_labels": len(labels), "overlapping_value_labels": 0,
        "top_axis_intervals": [[0, 5], list(top.get_ylim())], "breaks_marked": True,
        "underlying_native_values_unchanged": True,
        "primary_display_multipliers": dict(zip(categories, display_multipliers.tolist(), strict=True)),
        "primary_y_label": "Metric score (×100) or percentile rank",
        "rank_axes": "Unweighted and Weighted Avg Rank are the fifth and sixth primary groups, with separate sorted horizontal panels",
        "horizontal_panels": 4, "horizontal_bar_value_labels": 32,
        "vertical_bar_value_label_rotation_degrees": 90, "horizontal_bar_value_label_rotation_degrees": 0,
        "native_horizontal_label_decimals": 4, "native_horizontal_labels_omit_leading_zero": False,
        "horizontal_value_labels": "Outside each bar, horizontal, with compact model labels and enough numeric headroom",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
