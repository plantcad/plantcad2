"""Native family metrics and a separately scaled frozen-reference Avg Rank."""

import json
from pathlib import Path

import matplotlib
from artifact_io import normalize_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
RANGES = {"Conservation": (.62, .82), "SV impact": (.72, .89), "Avg Rank": (45, 91)}
TICKS = {"Conservation": [.62, .66, .70, .74, .78, .82], "SV impact": [.72, .76, .80, .84, .88], "Avg Rank": [50, 60, 70, 80, 90]}


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
    data = json.loads((HERE / "benchmark-comparison-data.json").read_text())
    models, categories = data["models"], data["categories"]
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "axes.axisbelow": True})
    fig = plt.figure(figsize=(26.0, 12.1))
    # The primary axis retains a zero stub; .05–.50 is explicitly omitted.
    top = fig.add_axes((.055, .666, .72, .28))
    stub = fig.add_axes((.055, .603, .72, .047), sharex=top)
    rank_top = fig.add_axes((.835, .666, .155, .28))
    rank_stub = fig.add_axes((.835, .603, .155, .047), sharex=rank_top)
    centers, width = np.arange(4), .105
    labels = []
    top_labels = []
    rank_labels = []
    for i, model in enumerate(models):
        x = centers + (i - (len(models) - 1) / 2) * width
        for axis in [top, stub]:
            axis.bar(x, model["scores"][:4], width=width, color=model["color"], edgecolor="white", linewidth=.4, label=model["label"])
        for xpos, score in zip(x, model["scores"][:4], strict=True):
            text = top.annotate(f"{score:.3f}", (xpos, score), xytext=(0, 5), textcoords="offset points",
                                rotation=0, ha="center", va="bottom", fontsize=11)
            labels.append(text)
            top_labels.append(text)
        rank_x = (i - (len(models) - 1) / 2) * width
        for axis in [rank_top, rank_stub]:
            axis.bar(rank_x, model["scores"][4], width=width, color=model["color"],
                     edgecolor="white", linewidth=.4)
        text = rank_top.annotate(f"{model['scores'][4]:.1f}", (rank_x, model["scores"][4]),
                                 xytext=(0, 5), textcoords="offset points", ha="center",
                                 va="bottom", fontsize=11)
        labels.append(text)
        rank_labels.append(text)
    top.set_ylim(.50, .95)
    top.set_yticks([.5, .6, .7, .8, .9])
    top.set_ylabel("Native task-family score")
    title = fig.text(.5225, .989, "Zero-shot PlantCAD2 benchmark", weight="bold",
                     fontsize="large", ha="center", va="top")
    top.spines[["bottom", "top", "right"]].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    top.grid(axis="y", color="#e4e4e4", linewidth=.7)
    stub.set_ylim(0, .05)
    stub.set_yticks([0, .05], ["0", "0.05"])
    stub.set_xlim(-.54, 3.54)
    stub.set_xticks(centers, categories[:4])
    stub.spines[["top", "right"]].set_visible(False)
    breaks(top, 0, True)
    breaks(stub, 1, True)
    rank_top.set_ylim(45, 95)
    rank_top.set_yticks([50, 60, 70, 80, 90])
    rank_top.set_ylabel("Percentile rank ↑", labelpad=9)
    rank_top.spines[["bottom", "top", "right"]].set_visible(False)
    rank_top.tick_params(axis="x", bottom=False, labelbottom=False)
    rank_top.grid(axis="y", color="#e4e4e4", linewidth=.7)
    rank_stub.set_ylim(0, 5)
    rank_stub.set_yticks([0, 5])
    rank_stub.set_xlim(-.54, .54)
    rank_stub.set_xticks([0], ["Avg Rank"])
    rank_stub.spines[["top", "right"]].set_visible(False)
    for axis in [rank_top, rank_stub]:
        axis.set_facecolor("#f6f8fa")
    breaks(rank_top, 0, True)
    breaks(rank_stub, 1, True)
    legend = fig.legend(*top.get_legend_handles_labels(), loc="center", ncols=3,
                        bbox_to_anchor=(.51, .523), frameon=False, columnspacing=2.3, handletextpad=.7)
    lower_axes, zeros = [], []
    for metric, left_inches, unit in [("Conservation", 2.42, "AUROC"),
                                       ("SV impact", 9.35, "Average precision"),
                                       ("Avg Rank", 16.28, "Percentile rank ↑")]:
        idx = categories.index(metric)
        ranked = sorted(models, key=lambda m: m["scores"][idx], reverse=True)
        left, bottom, height = left_inches / 20.8, .075, .345
        zero = fig.add_axes((left, bottom, .22 / 20.8, height))
        axis = fig.add_axes((left + .34 / 20.8, bottom, 2.95 / 20.8, height), sharey=zero)
        zeros.append(zero)
        lower_axes.append(axis)
        values = [model["scores"][idx] for model in ranked]
        low, high = RANGES[metric]
        assert low < min(values) < max(values) < high
        for part in [zero, axis]:
            part.barh(np.arange(8), values, height=.72, color=[m["color"] for m in ranked])
        zero.set_yticks(np.arange(8), [m["label"].replace("1.5B–60M ", "1.5B–60M\n") for m in ranked])
        zero.set_ylim(7.5, -.5)
        zero.set_xlim(0, 5 if metric == "Avg Rank" else .05)
        zero.set_xticks([0], ["0"])
        zero.spines[["top", "right"]].set_visible(False)
        zero.tick_params(axis="y", length=0, pad=8)
        axis.set_xlim(low, high)
        axis.set_xticks(TICKS[metric])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _, metric=metric: f"{v:.0f}" if metric == "Avg Rank" else f"{v:.2f}"))
        axis.set_title(metric, weight="bold", pad=10)
        axis.set_xlabel(unit)
        axis.grid(axis="x", color="#e4e4e4", linewidth=.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", left=False, labelleft=False)
        breaks(zero, 1, False)
        breaks(axis, 0, False)
        for row, value in enumerate(values):
            labels.append(axis.annotate(f"{value:.1f}" if metric == "Avg Rank" else f"{value:.4f}",
                                        (value, row), xytext=(5, 0), textcoords="offset points", va="center", fontsize=11))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = [label.get_window_extent(renderer).padded(1) for label in labels]
    overlaps = [(labels[i].get_text(), labels[j].get_text(), a.bounds, b.bounds)
                for i, a in enumerate(label_boxes) for j, b in enumerate(label_boxes) if i < j and a.overlaps(b)]
    assert not overlaps, overlaps
    assert all(top.bbox.contains(box.x0, box.y0) and top.bbox.contains(box.x1, box.y1)
               for box in [text.get_window_extent(renderer) for text in top_labels])
    assert all(rank_top.bbox.contains(box.x0, box.y0) and rank_top.bbox.contains(box.x1, box.y1)
               for box in [text.get_window_extent(renderer) for text in rank_labels])
    legend_box = legend.get_window_extent(renderer)
    panels = [Bbox.union([top.get_tightbbox(renderer), stub.get_tightbbox(renderer)]),
              Bbox.union([rank_top.get_tightbbox(renderer), rank_stub.get_tightbbox(renderer)]),
              *[Bbox.union([z.get_tightbbox(renderer), ax.get_tightbbox(renderer)]) for z, ax in zip(zeros, lower_axes, strict=True)]]
    assert not any(a.overlaps(b) for i, a in enumerate(panels) for b in panels[i + 1:])
    assert not any(box.overlaps(legend_box) for box in panels)
    assert len(top_labels) == 32 and len(rank_labels) == 8
    assert sum(len(ax.patches) for ax in [top, rank_top, *lower_axes]) == 64
    assert title.get_window_extent(renderer).y0 > max(top.bbox.y1, rank_top.bbox.y1)
    for ext in ["png", "svg"]:
        fig.savefig(HERE / "figures" / f"benchmark-comparison.{ext}", dpi=156,
                    bbox_inches="tight", pad_inches=.12, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
    normalize_svg(HERE / "figures" / "benchmark-comparison.svg")
    plt.close(fig)
    (HERE / "bar-layout-validation.json").write_text(json.dumps({
        "native_top_bar_labels": len(top_labels), "rank_top_bar_labels": len(rank_labels),
        "all_annotated_labels": len(labels), "overlapping_value_labels": 0,
        "top_axis_intervals": [[0, .05], [.50, .95]], "breaks_marked": True,
        "rank_top_axis_intervals": [[0, 5], [45, 95]],
        "native_values_unchanged": True,
        "Avg_Rank_axis": "Fifth group in primary vertical chart with its own percentile axis; also retained in sorted horizontal chart",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
