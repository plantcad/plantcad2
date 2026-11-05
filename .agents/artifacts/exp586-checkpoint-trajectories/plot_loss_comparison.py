"""Render complete W&B loss histories and two final benchmark readouts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
PALETTE_PATH = HERE / "figure-palette.json"
START_TOKENS = 1_000_000_000
HALF_LIFE_TOKENS = 1_000_000_000
MODELS = {"216b": "MarinMoE 1.5B–60M 216B", "1t": "MarinMoE 1.5B–60M 1T"}
REFERENCE_CODES = {"216b": "M61-L", "1t": "M61-L-1T"}
MATCHED_PATH = HERE.parents[2] / "zero-shot-leaderboard/marindna_exp586/runs/loss-correlation/analysis.json"
MATCHED_SHA256 = "b4b15aa88bf470481eceda6a582dc0b0977d9c1f27a4764623a81216f7e3c2e7"
REFERENCE_COMMENT = "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6604041302"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(path: Path, expected: dict) -> tuple[np.ndarray, dict]:
    """Keep the latest W&B record at each token coordinate after resumes."""
    assert digest(path) == expected["sha256"]
    with gzip.open(path, "rt") as stream:
        values = np.array([
            (row["input_tokens"], row["loss"], row["wandb_step"])
            for line in stream if (row := json.loads(line))
        ])
    assert len(values) == expected["rows"] and np.isfinite(values).all()
    assert (np.diff(values[:, 2]) > 0).all()
    if "training" in path.name:
        # Both runs log training loss at every nonempty W&B history step.
        # Check the logged-step coverage independently of training-token replay.
        assert values[0, 2] == 1 and (np.diff(values[:, 2]) == 1).all()
    values = values[np.lexsort((values[:, 2], values[:, 0]))]
    keep = np.r_[np.diff(values[:, 0]) > 0, True]
    duplicates = int((~keep).sum())
    values = values[keep]
    assert (np.diff(values[:, 0]) > 0).all()
    assert int(values[-1, 0]) == expected["last"]["input_tokens"]
    assert values[-1, 1] == expected["last"]["loss"]
    selected = values[values[:, 0] >= START_TOKENS, :2]
    return selected, {
        "source_rows": expected["rows"], "replayed_coordinates_removed": duplicates,
        "unique_rows_before_start_filter": len(values), "plotted_raw_rows": len(selected),
        "unique_token_increment_range": [int(np.diff(values[:, 0]).min()), int(np.diff(values[:, 0]).max())],
        "first_plotted_tokens": int(selected[0, 0]), "last_plotted_tokens": int(selected[-1, 0]),
        "raw_loss_range": [float(selected[:, 1].min()), float(selected[:, 1].max())],
    }


def smooth(values: np.ndarray) -> np.ndarray:
    """Exponential smoothing with decay measured in input tokens, not rows."""
    result = np.empty(len(values))
    result[0] = values[0, 1]
    decay = np.exp2(-np.diff(values[:, 0]) / HALF_LIFE_TOKENS)
    for index, weight in enumerate(decay, start=1):
        result[index] = weight * result[index - 1] + (1 - weight) * values[index, 1]
    assert np.isfinite(result).all()
    return result


def token_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, 1_140)
    ax.set_xticks([1, 10, 100, 1_000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("DNA tokens (billions, log scale)")
    ax.set_ylabel("Next-token cross-entropy loss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, required=True)
    args = parser.parse_args()
    benchmark_path = HERE / "benchmark-comparison-data.json"
    benchmark = json.loads(benchmark_path.read_text())
    models = {key: next(model for model in benchmark["models"] if model["label"] == label)
              for key, label in MODELS.items()}
    palette = json.loads(PALETTE_PATH.read_text())
    assert all(model["color"] == palette["marin"][key] for key, model in models.items())
    assert digest(MATCHED_PATH) == MATCHED_SHA256, "Matched-loss reference changed; review before plotting"
    reference = {row["code"]: row for row in json.loads(MATCHED_PATH.read_text())["rows"]}
    dense = [reference[code] for code in ["D216", "D389", "D562"]]
    assert len({row["total_parameters_approx"] for row in dense}) == 1
    first_tokens, last_tokens = dense[0]["dna_tokens"], dense[-1]["dna_tokens"]
    rugs = [{"model": row["code"], "DNA_tokens": row["dna_tokens"],
             "validation_loss": row["loss"],
             "alpha": .35 + .60 * (row["dna_tokens"] - first_tokens) / (last_tokens - first_tokens)}
            for row in dense]
    assert all(row["loss"] == row["original_loss"] for row in dense)
    assert all(row["original_sampling"]["examples"] == 65_536 for row in dense)
    common_hashes = set()
    for key, model in models.items():
        saved = reference[REFERENCE_CODES[key]]
        common = saved["validation"]["samples"]["common-65536"]
        assert common["examples"] == 65_536 and common["targets"] == 536_805_376
        assert common["dropped_assignments"] == 0 and common["native_loss"] == saved["loss"]
        assert digest(MATCHED_PATH.parent / saved["validation_result_file"]) == saved["validation_result_sha256"]
        assert model["checkpoint"] == saved["checkpoint"] and model["DNA_tokens"] == saved["dna_tokens"]
        for metric in ["Composite", "Conservation"]:
            assert model["scores"][benchmark["categories"].index(metric)] == saved["scores"][metric]
        common_hashes.add(common["input_sha256"])
    assert common_hashes == {reference["D216"]["validation"]["samples"]["common-65536"]["input_sha256"]}
    provenance = {
        "benchmark_source_sha256": digest(benchmark_path),
        "palette_sha256": digest(PALETTE_PATH),
        "run_colors": {key: model["color"] for key, model in models.items()},
        "sources": {}, "normalization": "Latest W&B step at each identical input-token count",
        "curve_start_tokens": START_TOKENS,
        "training_smoothing": {"method": "Exponential smoothing", "half_life_tokens": HALF_LIFE_TOKENS,
                               "initialization": "First raw observation at or above 1B tokens"},
        "validation_smoothing": "None; every unique logged evaluation at or above 1B tokens",
        "uncertainty": "None estimated: two individual training runs; connecting segments are descriptive",
        "scatter_loss_source": {
            "comment": REFERENCE_COMMENT, "commit": "b6e2962fe6d2ac27dd97992b54781ded4dd8427d",
            "file": str(MATCHED_PATH.relative_to(HERE.parents[2])), "sha256": MATCHED_SHA256,
            "validation_windows": 65_536, "bases_per_window": 8192,
            "targets": 536_805_376, "sample_sha256": next(iter(common_hashes)),
            "loss": "Full-native-vocabulary cross-entropy, in nats, exactly as in the reference plot",
            "curves": "W&B histories retain their original 512-window evaluation values",
        },
        "dense_rugs": {"color": palette["marin"]["dense"], "rows": rugs,
                       "opacity": "Linear in DNA training tokens: 0.35 at 216B, 0.95 at 562B; equal rug heights",
                       "labels": "Direct DNA-token labels; shared annotation identifies the dense model as 1B parameters",
                       "total_parameters_each": dense[0]["total_parameters_approx"]},
        "scatter": [],
    }
    plot_data = {"training": {}, "validation": {}, "scatter": [], "dense_rugs": rugs}
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "axes.axisbelow": True})
    fig = plt.figure(figsize=(19.2, 5.8))
    training_ax = fig.add_axes((.054, .18, .247, .66))
    validation_ax = fig.add_axes((.382, .18, .247, .66))
    scatter_ax = fig.add_axes((.710, .18, .237, .66))
    conservation_ax = scatter_ax.twinx()
    training_ranges = []
    validation_ranges = []
    smooth_lines = []
    for key, model in models.items():
        metadata_path = args.history_dir / f"{key}-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert model["DNA_tokens"] == metadata["expected_final_tokens"]
        provenance["sources"][key] = {"metadata": metadata, "metadata_sha256": digest(metadata_path),
                                      "normalization_counts": {}}
        for kind, axis in [("training", training_ax), ("validation", validation_ax)]:
            path = args.history_dir / metadata["histories"][kind]["file"]
            values, counts = history(path, metadata["histories"][kind])
            provenance["sources"][key]["normalization_counts"][kind] = counts
            if kind == "training":
                updates = int(metadata["summary"]["train/step"])
                tokens_per_update = metadata["expected_final_tokens"] // updates
                assert tokens_per_update * updates == metadata["expected_final_tokens"]
                assert counts["unique_rows_before_start_filter"] <= updates
                assert counts["unique_token_increment_range"][0] == tokens_per_update
                counts["updates_without_unique_loss_record"] = updates - counts["unique_rows_before_start_filter"]
                axis.plot(values[:, 0] / 1e9, values[:, 1], color=model["color"], alpha=.19,
                          linewidth=.45, rasterized=True, zorder=1)
                smoothed = smooth(values)
                line, = axis.plot(values[:, 0] / 1e9, smoothed, color=model["color"], linewidth=2.4, zorder=3)
                smooth_lines.append(line)
                training_ranges.append(counts["raw_loss_range"])
                # Store a compact, log-spaced view of the exact full-resolution smoother.
                indices = np.unique(np.r_[0, np.searchsorted(values[:, 0], np.geomspace(
                    values[0, 0], values[-1, 0], 2_000)), len(values) - 1])
                indices = indices[indices < len(values)]
                plot_data[kind][key] = [{"input_tokens": int(values[i, 0]), "smoothed_loss": float(smoothed[i])}
                                        for i in indices]
            else:
                axis.plot(values[:, 0] / 1e9, values[:, 1], color=model["color"], linewidth=2.2)
                axis.scatter(values[-1, 0] / 1e9, values[-1, 1], color=model["color"], s=34, zorder=4)
                validation_ranges.append(counts["raw_loss_range"])
                plot_data[kind][key] = [{"input_tokens": int(tokens), "loss": float(loss)} for tokens, loss in values]
                assert values[-1, 1] == metadata["summary"]["eval/loss"]
        matched = reference[REFERENCE_CODES[key]]
        assert metadata["summary"]["eval/loss"] == matched["original_loss"]
        row = {"model": key, "DNA_tokens": model["DNA_tokens"],
               "final_validation_loss": matched["loss"],
               "logged_final_validation_loss": metadata["summary"]["eval/loss"],
               "matched_reference_code": REFERENCE_CODES[key],
               **{name: model["scores"][benchmark["categories"].index(name)]
                  for name in ["Composite", "Conservation"]}}
        provenance["scatter"].append(row)
        plot_data["scatter"].append(row)
    for axis in [training_ax, validation_ax, scatter_ax]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e4e4e4", linewidth=.7)
    conservation_ax.spines["top"].set_visible(False)
    for axis, ranges in [(training_ax, training_ranges), (validation_ax, validation_ranges)]:
        token_axis(axis)
        low = np.floor((min(values[0] for values in ranges) - .025) * 20) / 20
        high = np.ceil((max(values[1] for values in ranges) + .015) * 20) / 20
        axis.set_ylim(low, high)
    training_ax.set_title("Training loss", weight="bold", pad=12)
    validation_ax.set_title("Validation loss", weight="bold", pad=12)
    scatter_ax.set_title("Final loss vs zero-shot scores", weight="bold", pad=12)
    smoothing_legend = training_ax.legend(
        [Line2D([], [], color="#666666", alpha=.25, linewidth=.8),
         Line2D([], [], color="#666666", linewidth=2.4)],
        ["Raw", "Smoothed · 1B-token half-life"], loc="lower left", frameon=False,
        handlelength=1.5, fontsize=10)
    rows = sorted(provenance["scatter"], key=lambda row: row["final_validation_loss"])
    metric_handles = []
    dot_count = 0
    for metric, axis, marker, style in [
        ("Composite", scatter_ax, "o", "-"), ("Conservation", conservation_ax, "D", "--"),
    ]:
        xs = [row["final_validation_loss"] for row in rows]
        ys = [row[metric] for row in rows]
        line, = axis.plot(xs, ys, color="#626262", linestyle=style, linewidth=1.5, zorder=2)
        for row, x, y in zip(rows, xs, ys, strict=True):
            axis.scatter(x, y, marker=marker, color=models[row["model"]]["color"], s=78,
                         edgecolor="white", linewidth=.8, zorder=4)
            dot_count += 1
        metric_handles.append(Line2D([], [], color="#626262", marker=marker, linestyle=style,
                                     label=metric, markersize=7))
    rug_labels = []
    offsets = {"D216": (0, 12, "center"), "D389": (0, 32, "center"), "D562": (-4, 12, "right")}
    for row in rugs:
        rug = scatter_ax.vlines(row["validation_loss"], 0, .065,
                                transform=scatter_ax.get_xaxis_transform(),
                                color=palette["marin"]["dense"], alpha=row["alpha"], linewidth=3, zorder=5)
        rug.set_gid(f"dense-rug-{row['model'].lower()}")
        dx, dy, alignment = offsets[row["model"]]
        label = scatter_ax.annotate(
            f"{row['DNA_tokens']/1e9:.0f}B", (row["validation_loss"], .065),
            xycoords=scatter_ax.get_xaxis_transform(), xytext=(dx, dy), textcoords="offset points",
            ha=alignment, va="bottom", fontsize=10, color="#505050", zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": .2},
            arrowprops={"arrowstyle": "-", "color": palette["marin"]["dense"],
                        "alpha": row["alpha"], "linewidth": 1, "shrinkA": 2, "shrinkB": 0},
        )
        label.set_gid(f"dense-rug-label-{row['model'].lower()}")
        rug_labels.append(label)
    dense_note = scatter_ax.text(
        .035, .10, "Dense MarinDNA (1B parameters)\nDNA training tokens",
        transform=scatter_ax.transAxes, ha="left", va="bottom", fontsize=9.5, color="#505050",
    )
    scatter_ax.set_xlim(.875, .99)
    scatter_ax.set_xticks([.89, .91, .93, .95, .97, .99])
    # Identical numerical limits avoid implying arbitrary slopes from independent scales.
    for axis in [scatter_ax, conservation_ax]:
        axis.set_ylim(.68, .715)
        axis.yaxis.set_major_locator(MultipleLocator(.01))
    scatter_ax.set_ylabel("Composite")
    conservation_ax.set_ylabel("Conservation (AUROC)", labelpad=10)
    scatter_ax.set_xlabel("Matched validation loss")
    metric_legend = scatter_ax.legend(handles=metric_handles, loc="upper center", ncols=2,
                                      frameon=False, fontsize=10, columnspacing=1.0, handlelength=1.6)
    model_legend = fig.legend(smooth_lines, ["216B-token run", "1T-token run"],
                              title="DNA training horizon", loc="upper center", bbox_to_anchor=(.5, 1.01),
                              ncols=2, frameon=False, columnspacing=2.2)
    assert dot_count == 4 and len(rugs) == 3 and len(scatter_ax.lines) == len(conservation_ax.lines) == 1
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_boxes = [axis.title.get_window_extent(renderer) for axis in [training_ax, validation_ax, scatter_ax]]
    model_box = model_legend.get_window_extent(renderer)
    assert not any(box.overlaps(model_box) for box in title_boxes)
    for label in [*title_boxes, model_box, smoothing_legend.get_window_extent(renderer),
                  metric_legend.get_window_extent(renderer), dense_note.get_window_extent(renderer)]:
        assert fig.bbox.contains(label.x0, label.y0) and fig.bbox.contains(label.x1, label.y1)
    note_boxes = [dense_note.get_window_extent(renderer)] + [label.get_bbox_patch().get_window_extent(renderer) for label in rug_labels]
    for index, box in enumerate(note_boxes):
        assert scatter_ax.bbox.contains(box.x0, box.y0) and scatter_ax.bbox.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in note_boxes[index + 1:])
        for row in rows:
            for metric in ["Composite", "Conservation"]:
                dot = scatter_ax.transData.transform((row["final_validation_loss"], row[metric]))
                assert not box.expanded(1.05, 1.1).contains(*dot)
    assert all(scatter_ax.get_xlim()[0] < row["validation_loss"] < scatter_ax.get_xlim()[1] for row in rugs)
    target = HERE / "figures" / "loss-comparison"
    fig.savefig(target.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(target.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    plt.close(fig)
    (HERE / "loss-figure-data.json").write_text(json.dumps(plot_data, indent=2) + "\n")
    provenance["plot_data_sha256"] = digest(HERE / "loss-figure-data.json")
    provenance["figure_sha256"] = {extension: digest(target.with_suffix("." + extension)) for extension in ["png", "svg"]}
    (HERE / "loss-figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"scatter": provenance["scatter"], "figure": str(target), "dots": dot_count}, indent=2))


if __name__ == "__main__":
    main()
