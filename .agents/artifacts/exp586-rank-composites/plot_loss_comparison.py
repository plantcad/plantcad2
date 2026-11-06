"""Compare final linear probes above the unchanged W&B loss histories."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
from artifact_io import normalize_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from plot_probe_scorecard import load_rows as load_probe_rows

HERE = Path(__file__).resolve().parent
PALETTE_PATH = HERE / "figure-palette.json"
START_TOKENS = 1_000_000_000
HALF_LIFE_TOKENS = 1_000_000_000
MODELS = {"216b": "MarinMoE 1.5B–60M 216B", "1t": "MarinMoE 1.5B–60M 1T"}
REFERENCE_CODES = {"216b": "M61-L", "1t": "M61-L-1T"}
MATCHED_PATH = HERE.parents[2] / "zero-shot-leaderboard/marindna_exp586/runs/loss-correlation/analysis.json"
MATCHED_SHA256 = "0f3fea347f311b7af85bfa50ae9a0443ac69be3a9c0d9f41b9bb5a856625cf38"
REFERENCE_COMMENT = "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6604041302"
PROBE_METRICS = ["Conservation", "Motifs", "Core/non-core", "SV impact", "Allele frequency", "Win Rate"]
FINAL_CHECKPOINTS = {"216b": ("216B-token run", "dna-216b"), "1t": ("1T-token run", "dna-1000b")}


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


def task_winner_map(ax: plt.Axes, rows: list[dict], models: dict) -> dict:
    """Use species headers and one categorical tile for each measured task."""
    ax.set_xlim(0, 10)
    ax.set_ylim(15.1, 0)
    ax.axis("off")
    ax.set_title("Higher-scoring model by task", weight="bold", pad=12)
    texts = []
    cells = []
    tile_texts = []
    source = {row["task_id"]: row for row in rows}

    def label(x: float, y: float, text: str, **kwargs) -> None:
        style = {"ha": "center", "va": "center", "fontsize": 10.5} | kwargs
        texts.append(ax.text(x, y, text, **style))

    def tile(task: str, x: float, y: float, species: str, text: str = "", height: float = 1.0) -> None:
        row = source[task]
        winner = row["winner"]
        color = "#e4e7eb" if winner == "tie" else models[winner]["color"]
        patch = Rectangle((x, y), 1.65, height, facecolor=color, edgecolor="white", linewidth=1)
        patch.set_gid(f"task-winner-{task}")
        ax.add_patch(patch)
        if text:
            label(x + .825, y + height / 2, text,
                  color="white" if winner == "1t" else "#1f2328", linespacing=1.15, fontsize=10)
            tile_texts.append((patch, texts[-1]))
        cells.append({"task_id": task, "family": row["family"], "species": species,
                      "task": row["task"], "winner": winner,
                      "216b": row["216b"], "1t": row["1t"],
                      "delta_1t_minus_216b": row["delta_1t_minus_216b"],
                      "xy": [x, y], "width": 1.65, "height": height,
                      "tile_text": text})

    # Conservation clades are distinct from the species supplying the sequence.
    label(.02, 1.85, "Conservation", weight="bold", ha="left")
    label(3.625, .55, "Sorghum", weight="bold")
    label(6.325, .55, "Pharus", weight="bold")
    ax.plot([2.8, 4.45], [.91, .91], color="#cbd2d9", linewidth=.8)
    ax.plot([4.6, 8.05], [.91, .91], color="#cbd2d9", linewidth=.8)
    tile("conservation_andropogoneae", 2.8, 1.1, "sorghum", "Andropogoneae\ngenome-wide", height=1.5)
    tile("conservation_poaceae_non_tis", 4.6, 1.1, "pharus", "Poaceae\nnon-TIS CDS", height=1.5)
    tile("conservation_poaceae_tis", 6.4, 1.1, "pharus", "Poaceae\nTIS CDS", height=1.5)

    # Shared species rows and target-type columns name all sixteen tasks exactly.
    for family, prefix, header_y in [("Motifs", "motif", 3.7), ("Core/non-core", "core", 7.6)]:
        label(.02, header_y, family, weight="bold", ha="left")
        for column, (suffix, target) in enumerate([
            ("tis", "Start\ncodon"), ("tts", "Stop\ncodon"),
            ("donor", "Splice\ndonor"), ("acceptor", "Splice\nacceptor"),
        ]):
            x = 2.8 + column * 1.8
            label(x + .825, header_y, target)
            for row_index, species in enumerate(["maize", "tomato"]):
                y = header_y + .55 + row_index * 1.25
                tile(f"{prefix}_{species}_{suffix}", x, y, species)
        for row_index, species in enumerate(["Maize", "Tomato"]):
            label(2.55, header_y + 1.05 + row_index * 1.25, species, ha="right")

    label(.02, 11.7, "SV impact", weight="bold", ha="left")
    tile("sv_impact", 2.8, 11.2, "arabidopsis", "Arabidopsis")
    label(.02, 13.1, "Allele frequency", weight="bold", ha="left")
    tile("af", 2.8, 12.6, "maize", "Maize")

    label(5, 14.5, "One tile per task; color shows the higher-scoring model.", color="#505050")
    assert len(cells) == len({cell["task_id"] for cell in cells}) == len(source) == 21
    return {"cells": cells, "texts": texts, "tile_texts": tile_texts,
            "color": "Higher unrounded native task score; no margin encoding; gray for exact ties",
            "species_source": "zero-shot-leaderboard/marindna_exp586/src/exp586_evals/classification_data.py:task_species; maize AF protocol",
            "summary": "Task-level breakdown of the separately plotted family-balanced and pooled win rates"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, required=True)
    args = parser.parse_args()
    benchmark_path = HERE / "benchmark-comparison-data.json"
    benchmark = json.loads(benchmark_path.read_text())
    models = {key: next(model for model in benchmark["models"] if model["label"] == label)
              for key, label in MODELS.items()}
    score_path = HERE / "all-category-scores.csv"
    with score_path.open() as stream:
        saved_scores = list(csv.DictReader(stream))
    for key, model in models.items():
        run, checkpoint = FINAL_CHECKPOINTS[key]
        selected = [row for row in saved_scores if row["run"] == run and row["checkpoint"] == checkpoint
                    and row["readout"] == "Linear probe" and row["metric"] in PROBE_METRICS[:-1]]
        assert len(selected) == len(PROBE_METRICS) - 1
        assert {row["metric"] for row in selected} == set(PROBE_METRICS[:-1])
        assert all(float(row["DNA_tokens_B"]) == model["DNA_tokens"] / 1e9 for row in selected)
        model["probe_scores"] = {row["metric"]: float(row["score"]) for row in selected}
    summaries_path = HERE / "marin-summaries.json"
    summaries = json.loads(summaries_path.read_text())
    probe_rows, _, checked_summaries = load_probe_rows()
    assert checked_summaries == summaries
    for key, model in models.items():
        identity = "short-216b" if key == "216b" else "long-1000b"
        model["probe_scores"]["Win Rate"] = summaries["probe_win_rates"][identity]["score"]
    task_wins = {}
    for family in PROBE_METRICS[:-1]:
        tasks = summaries["families"][family]
        task_wins[family] = {}
        for key in models:
            identity = "short-216b" if key == "216b" else "long-1000b"
            points = sum(summaries["probe_win_rates"][identity]["tasks"][task] for task in tasks)
            share = 100 * points / len(tasks)
            np.testing.assert_allclose(share, summaries["probe_win_rates"][identity]["families"][family], rtol=0, atol=1e-12)
            task_wins[family][key] = {"points": points, "tasks": len(tasks), "percent": share}
        np.testing.assert_allclose(sum(row["percent"] for row in task_wins[family].values()), 100, rtol=0, atol=1e-12)
    win_rates = {"Family-balanced": {}, "Pooled": {}}
    pooled_points = {}
    for key, model in models.items():
        points = sum(.5 if row["winner"] == "tie" else float(row["winner"] == key) for row in probe_rows)
        pooled_points[key] = points
        np.testing.assert_allclose(points, sum(family[key]["points"] for family in task_wins.values()), rtol=0, atol=1e-12)
        win_rates["Pooled"][key] = 100 * points / len(probe_rows)
        win_rates["Family-balanced"][key] = model["probe_scores"]["Win Rate"]
        np.testing.assert_allclose(model["probe_scores"]["Win Rate"],
                                   np.mean([family[key]["percent"] for family in task_wins.values()]), rtol=0, atol=1e-12)
    for rates in win_rates.values():
        np.testing.assert_allclose(sum(rates.values()), 100, rtol=0, atol=1e-12)
    win_rate_comparison = {
        "percentages": win_rates, "pooled_task_count": len(probe_rows), "pooled_win_points": pooled_points,
        "family_balanced": "Average task-win fractions within each family, then weight the five families equally",
        "pooled": "Average the 21 task-win points equally, without family balancing",
        "ties": "Half a win to each model for an exact stored-score tie",
    }
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
        assert model["scores"][benchmark["categories"].index("Conservation")] == saved["scores"]["Conservation"]
        common_hashes.add(common["input_sha256"])
    assert common_hashes == {reference["D216"]["validation"]["samples"]["common-65536"]["input_sha256"]}
    provenance = {
        "figure_title": "Linear probe PlantCAD2 benchmark",
        "layout": "Three top facets: family-average scores, family-balanced/pooled win-rate bars, and species-labeled task tiles; three loss panels below",
        "probe_bar_annotations": 10,
        "win_rate_bar_annotations": 4,
        "benchmark_source_sha256": digest(benchmark_path),
        "probe_source_sha256": digest(score_path),
        "summary_source_sha256": digest(summaries_path),
        "frozen_reference_sha256": digest(HERE / "frozen-references.json"),
        "palette_sha256": digest(PALETTE_PATH),
        "run_colors": {key: model["color"] for key, model in models.items()},
        "sources": {}, "normalization": "Latest W&B step at each identical input-token count",
        "curve_start_tokens": START_TOKENS,
        "training_smoothing": {"method": "Exponential smoothing", "half_life_tokens": HALF_LIFE_TOKENS,
                               "initialization": "First raw observation at or above 1B tokens"},
        "validation_smoothing": "None; every unique logged evaluation at or above 1B tokens",
        "uncertainty": "None estimated: two individual training runs; connecting segments are descriptive",
        "scatter_loss_source": {
            "comment": REFERENCE_COMMENT, "commit": "d67c9ab6267d636a9cbeb7ac12e8a7c586bc68d8",
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
        "scatter_readout": "Linear probe: family-balanced win rate (left), Conservation AUROC (right)",
        "probe_benchmark": {
            "metrics": PROBE_METRICS, "models": {key: model["probe_scores"] for key, model in models.items()},
            "family_score_bars": PROBE_METRICS[:-1],
            "display_multipliers": {metric: 1 if metric == "Win Rate" else 100 for metric in PROBE_METRICS},
            "aggregation": "Native family means; Win Rate compares only the two final models per task, then averages task wins within families and five families equally",
        },
        "task_wins": task_wins,
        "win_rate_comparison": win_rate_comparison,
        "scatter": [],
    }
    plot_data = {"probe_benchmark": provenance["probe_benchmark"],
                 "task_wins": task_wins, "win_rate_comparison": win_rate_comparison,
                 "training": {}, "validation": {}, "scatter": [], "dense_rugs": rugs}
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "axes.axisbelow": True})
    fig = plt.figure(figsize=(20.8, 12.1))
    title = fig.text(.5225, .989, "Linear probe PlantCAD2 benchmark", weight="bold",
                     fontsize="large", ha="center", va="top")
    probe_ax = fig.add_axes((.055, .603, .318, .322))
    rate_ax = fig.add_axes((.430, .603, .160, .322))
    win_ax = fig.add_axes((.645, .568, .342, .357))
    bar_labels = []
    family_metrics = PROBE_METRICS[:-1]
    for index, (key, model) in enumerate(models.items()):
        x = np.arange(len(family_metrics)) + (index - .5) * .36
        scores = [model["probe_scores"][metric] * 100 for metric in family_metrics]
        assert all(0 <= score <= 100 for score in scores)
        bars = probe_ax.bar(x, scores, width=.36, color=model["color"], edgecolor="white", linewidth=.4)
        np.testing.assert_array_equal([bar.get_height() for bar in bars], scores)
        for xpos, score in zip(x, scores, strict=True):
            bar_labels.append(probe_ax.annotate(f"{score:.2f}", (xpos, score), xytext=(0, 5),
                                                textcoords="offset points", ha="center", va="bottom", fontsize=10))
    probe_ax.set_xticks(np.arange(len(family_metrics)), family_metrics)
    probe_ax.set_xlim(-.5, len(family_metrics) - .5)
    probe_ax.set_ylim(0, 104)
    probe_ax.set_yticks([0, 20, 40, 60, 80, 100])
    probe_ax.set_title("Family-average scores", weight="bold", pad=12)
    probe_ax.set_ylabel("Metric score (×100)")
    probe_ax.spines[["top", "right"]].set_visible(False)
    probe_ax.grid(axis="y", color="#e4e4e4", linewidth=.7)
    rate_labels = []
    for index, (key, model) in enumerate(models.items()):
        x = np.arange(len(win_rates)) + (index - .5) * .32
        scores = [rates[key] for rates in win_rates.values()]
        bars = rate_ax.bar(x, scores, width=.32, color=model["color"], edgecolor="white", linewidth=.4)
        np.testing.assert_array_equal([bar.get_height() for bar in bars], scores)
        for xpos, score in zip(x, scores, strict=True):
            rate_labels.append(rate_ax.annotate(f"{score:.2f}", (xpos, score), xytext=(0, 5),
                                                 textcoords="offset points", ha="center", va="bottom", fontsize=10))
    rate_ax.set_xticks([0, 1], ["Family-balanced\n(equal family weight)", "Pooled\n(equal task weight)"], fontsize=10)
    rate_ax.set_xlim(-.5, 1.5)
    rate_ax.set_ylim(0, 104)
    rate_ax.set_yticks([0, 20, 40, 60, 80, 100])
    rate_ax.set_title("Task win rates", weight="bold", pad=12)
    rate_ax.set_ylabel("Win rate (%)")
    rate_ax.spines[["top", "right"]].set_visible(False)
    rate_ax.grid(axis="y", color="#e4e4e4", linewidth=.7)
    # Identify every measured task without mixing wins with native-score bars.
    winner_map = task_winner_map(win_ax, probe_rows, models)
    win_labels = winner_map.pop("texts")
    tile_texts = winner_map.pop("tile_texts")
    provenance["task_winner_map"] = winner_map
    plot_data["task_winner_map"] = winner_map
    training_ax = fig.add_axes((.054, .075, .247, .345))
    validation_ax = fig.add_axes((.382, .075, .247, .345))
    scatter_ax = fig.add_axes((.710, .075, .237, .345))
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
               **{name: model["probe_scores"][name] for name in ["Win Rate", "Conservation"]}}
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
    scatter_ax.set_title("Final loss vs probe scores", weight="bold", pad=12)
    smoothing_legend = training_ax.legend(
        [Line2D([], [], color="#666666", alpha=.25, linewidth=.8),
         Line2D([], [], color="#666666", linewidth=2.4)],
        ["Raw", "Smoothed · 1B-token half-life"], loc="lower left", frameon=False,
        handlelength=1.5, fontsize=10)
    rows = sorted(provenance["scatter"], key=lambda row: row["final_validation_loss"])
    metric_handles = []
    dot_count = 0
    for metric, axis, marker, style in [
        ("Win Rate", scatter_ax, "o", "-"), ("Conservation", conservation_ax, "D", "--"),
    ]:
        xs = [row["final_validation_loss"] for row in rows]
        ys = [row[metric] for row in rows]
        line, = axis.plot(xs, ys, color="#626262", linestyle=style, linewidth=1.5, zorder=2)
        for row, x, y in zip(rows, xs, ys, strict=True):
            axis.scatter(x, y, marker=marker, color=models[row["model"]]["color"], s=78,
                         edgecolor="white", linewidth=.8, zorder=4)
            dot_count += 1
        metric_handles.append(Line2D([], [], color="#626262", marker=marker, linestyle=style,
                                     label="Family-balanced (left)" if metric == "Win Rate" else "Conservation (right)",
                                     markersize=7, linewidth=1.8))
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
    dense_token_label = next(label for label in rug_labels if label.get_gid() == "dense-rug-label-d562")
    dense_note = scatter_ax.annotate(
        "Dense MarinDNA 1B\ntraining token count",
        xy=(0, .5), xycoords=dense_token_label.get_bbox_patch(),
        xytext=(.285, .10), textcoords="axes fraction",
        ha="center", multialignment="center", va="bottom", fontsize=9.5, color="#505050", zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": .2},
        arrowprops={"arrowstyle": "->", "color": "black",
                    "linewidth": 1.1, "mutation_scale": 11, "relpos": (1, .25),
                    "connectionstyle": "arc3,rad=-.10", "shrinkA": 6, "shrinkB": 3},
    )
    dense_note.set_gid("dense-rug-note")
    dense_note.arrow_patch.set_gid("dense-rug-note-arrow")
    scatter_ax.set_xlim(.875, .99)
    scatter_ax.set_xticks([.89, .91, .93, .95, .97, .99])
    # Win rates and AUROC have different units; label both scales.
    scatter_ax.set_ylim(0, 100)
    scatter_ax.yaxis.set_major_locator(MultipleLocator(25))
    conservation_ax.set_ylim(.88, .90)
    conservation_ax.yaxis.set_major_locator(MultipleLocator(.005))
    scatter_ax.set_ylabel("Family-balanced win rate (%)")
    conservation_ax.set_ylabel("Conservation (AUROC)", labelpad=10)
    scatter_ax.set_xlabel("Matched validation loss")
    metric_legend = scatter_ax.legend(handles=metric_handles, loc="upper right", ncols=1,
                                      frameon=False, fontsize=9, columnspacing=1.0, handlelength=3.6)
    model_legend = fig.legend(smooth_lines, ["216B-token run", "1T-token run"],
                              title="DNA training horizon", loc="center", bbox_to_anchor=(.51, .523),
                              ncols=2, frameon=False, columnspacing=2.2)
    assert dot_count == 4 and len(rugs) == 3 and len(scatter_ax.lines) == len(conservation_ax.lines) == 1
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_boxes = [axis.title.get_window_extent(renderer) for axis in [training_ax, validation_ax, scatter_ax]]
    model_box = model_legend.get_window_extent(renderer)
    boxes = [label.get_window_extent(renderer).padded(1) for label in bar_labels]
    assert len(bar_labels) == len(probe_ax.patches) == 10
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert all(probe_ax.bbox.contains(box.x0, box.y0) and probe_ax.bbox.contains(box.x1, box.y1) for box in boxes)
    rate_boxes = [label.get_window_extent(renderer).padded(1) for label in rate_labels]
    assert len(rate_labels) == len(rate_ax.patches) == 4
    assert not any(a.overlaps(b) for i, a in enumerate(rate_boxes) for b in rate_boxes[i + 1:])
    assert all(rate_ax.bbox.contains(box.x0, box.y0) and rate_ax.bbox.contains(box.x1, box.y1) for box in rate_boxes)
    win_boxes = [label.get_window_extent(renderer).padded(1) for label in win_labels]
    assert len(win_ax.patches) == 21
    assert not any(a.overlaps(b) for i, a in enumerate(win_boxes) for b in win_boxes[i + 1:])
    assert all(win_ax.bbox.contains(box.x0, box.y0) and win_ax.bbox.contains(box.x1, box.y1) for box in win_boxes)
    for tile, text in tile_texts:
        tile_box = tile.get_window_extent(renderer)
        text_box = text.get_window_extent(renderer).padded(1)
        assert tile_box.contains(text_box.x0, text_box.y0) and tile_box.contains(text_box.x1, text_box.y1), (
            tile.get_gid(), text.get_text(), tuple(tile_box.bounds), tuple(text_box.bounds))
    top_boxes = [axis.get_tightbbox(renderer) for axis in [probe_ax, rate_ax, win_ax]]
    assert not any(a.overlaps(b) for i, a in enumerate(top_boxes) for b in top_boxes[i + 1:])
    assert title.get_window_extent(renderer).y0 > max(box.y1 for box in top_boxes)
    assert min(box.y0 for box in top_boxes) > model_box.y1
    assert model_box.y0 > max(box.y1 for box in title_boxes)
    assert not any(box.overlaps(model_box) for box in title_boxes)
    for label in [*title_boxes, model_box, smoothing_legend.get_window_extent(renderer),
                  metric_legend.get_window_extent(renderer), dense_note.get_window_extent(renderer)]:
        assert fig.bbox.contains(label.x0, label.y0) and fig.bbox.contains(label.x1, label.y1)
    note_boxes = [dense_note.get_bbox_patch().get_window_extent(renderer)] + [label.get_bbox_patch().get_window_extent(renderer) for label in rug_labels]
    for index, box in enumerate(note_boxes):
        assert scatter_ax.bbox.contains(box.x0, box.y0) and scatter_ax.bbox.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in note_boxes[index + 1:])
        for row in rows:
            for metric, axis in [("Win Rate", scatter_ax), ("Conservation", conservation_ax)]:
                dot = axis.transData.transform((row["final_validation_loss"], row[metric]))
                assert not box.expanded(1.05, 1.1).contains(*dot)
    assert all(scatter_ax.get_xlim()[0] < row["validation_loss"] < scatter_ax.get_xlim()[1] for row in rugs)
    target = HERE / "figures" / "loss-comparison"
    fig.savefig(target.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(target.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    normalize_svg(target.with_suffix(".svg"))
    plt.close(fig)
    (HERE / "loss-figure-data.json").write_text(json.dumps(plot_data, indent=2) + "\n")
    provenance["plot_data_sha256"] = digest(HERE / "loss-figure-data.json")
    provenance["figure_sha256"] = {extension: digest(target.with_suffix("." + extension)) for extension in ["png", "svg"]}
    (HERE / "loss-figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"scatter": provenance["scatter"], "figure": str(target), "dots": dot_count}, indent=2))


if __name__ == "__main__":
    main()
