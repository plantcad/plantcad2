"""Reuse saved full-benchmark metrics for a draft eight-model comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "zero-shot-leaderboard/marindna_exp586"
CATEGORIES = ["Conservation", "Masked motif", "Core/non-core", "Structural variant"]
DISPLAY = ["Conservation", "Motifs", "Core/non-core", "SV impact", "Composite"]
PUBLISHED = [
    ("PlantCAD2.5-Large", "PlantCAD2.5-Large", "#174a7e"),
    ("PlantCAD2-Large", "PlantCAD2-Large", "#2f78a8"),
    ("PlantCAD2-Medium", "PlantCAD2-Medium", "#4e90b6"),
    ("PlantCAD2-Small", "PlantCAD2-Small", "#66a6c9"),
    ("evo2_20b", "Evo 2 20B", "#7a5195"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(models: list[dict]) -> None:
    """Retain the reference's grouped bars, warm Marin colors and blue baselines."""
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "axes.axisbelow": True})
    fig = plt.figure(figsize=(19.2, 12.0))
    ax = fig.add_axes((.055, .58, .94, .37))
    centers = np.arange(5)
    width = .115
    annotations = []
    for index, model in enumerate(models):
        x = centers + (index - (len(models) - 1) / 2) * width
        bars = ax.bar(x, model["scores"], width=width, color=model["color"],
                      edgecolor="white", linewidth=.4, label=model["label"])
        if index < 3 or model["label"] == "Evo 2 20B":
            for bar, value in zip(bars, model["scores"], strict=True):
                annotations.append(ax.annotate(
                    f"{value:.3f}", (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9))
    ax.set_xticks(centers, DISPLAY)
    ax.set_xlim(-.5, 4.5)
    ax.set_ylim(0, 1.025)
    ax.set_ylabel("Group score / composite")
    ax.set_title("Zero-shot PlantCAD2 benchmark", weight="bold", pad=10)
    ax.grid(axis="y", color="#e4e4e4", linewidth=.7)
    legend = fig.legend(*ax.get_legend_handles_labels(), loc="center", ncols=3,
                        bbox_to_anchor=(.51, .495), frameon=False,
                        columnspacing=2.3, handletextpad=.7)
    ranking_axes = []
    for metric, position, xlabel in [
        ("Conservation", (2.42 / 19.2, .075, 3.15 / 19.2, .345), "AUROC"),
        ("SV impact", (8.82 / 19.2, .075, 3.15 / 19.2, .345), "Average precision"),
        ("Composite", (15.22 / 19.2, .075, 3.15 / 19.2, .345), "Mean category score"),
    ]:
        category_index = DISPLAY.index(metric)
        ranked = sorted(models, key=lambda model: model["scores"][category_index], reverse=True)
        ranking = fig.add_axes(position)
        ranking_axes.append(ranking)
        values = [model["scores"][category_index] for model in ranked]
        ranking.barh(np.arange(len(ranked)), values, height=.72,
                     color=[model["color"] for model in ranked])
        # Keep complete model names, wrapping only the shared MoE token budgets.
        labels = [model["label"].replace("1.5B–60M ", "1.5B–60M\n") for model in ranked]
        ranking.set_yticks(np.arange(len(ranked)), labels)
        ranking.invert_yaxis()
        ranking.set_xlim(0, .9)
        ranking.set_xticks([0, .2, .4, .6, .8])
        ranking.set_title(metric, weight="bold", pad=10)
        ranking.set_xlabel(xlabel)
        ranking.grid(axis="x", color="#e4e4e4", linewidth=.7)
        ranking.spines[["top", "right"]].set_visible(False)
        ranking.tick_params(axis="y", length=0, pad=8)
        for row, value in enumerate(values):
            annotations.append(ranking.annotate(
                f"{value:.4f}", (value, row), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=11))
    assert len(ax.patches) == 40 and len(ranking_axes) == 3
    assert all(len(panel.patches) == 8 for panel in ranking_axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer).padded(1.8 * fig.dpi / 72)
             for text in annotations]
    overlaps = [(annotations[i].get_text(), annotations[j].get_text(),
                 boxes[i].bounds, boxes[j].bounds)
                for i in range(len(boxes)) for j in range(i+1, len(boxes))
                if boxes[i].overlaps(boxes[j])]
    assert not overlaps, overlaps
    panel_boxes = [panel.get_tightbbox(renderer) for panel in [ax, *ranking_axes]]
    assert all(not a.overlaps(b) for i, a in enumerate(panel_boxes[1:]) for b in panel_boxes[i+2:])
    assert all(not panel.overlaps(legend.get_window_extent(renderer)) for panel in panel_boxes)
    assert panel_boxes[0].y0 > max(panel.y1 for panel in panel_boxes[1:])
    for extension in ("svg", "png"):
        options = {"pil_kwargs": {"optimize": True}} if extension == "png" else {}
        fig.savefig(HERE / "figures" / f"benchmark-comparison.{extension}", dpi=156,
                    bbox_inches="tight", pad_inches=.12, facecolor="white", **options)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-results", type=Path, required=True)
    parser.add_argument("--long-results", type=Path, required=True)
    args = parser.parse_args()
    historical_path = PROJECT / "historical/exp472-lr5e4-full-coreweave-20260831.json"
    historical = json.loads(historical_path.read_text())
    manifest = json.loads((PROJECT / "historical/manifest.json").read_text())
    record = next(r for r in manifest["files"] if r["path"] == "historical/" + historical_path.name)
    assert sha256(historical_path) == record["sha256"]
    dense = historical["results"]["056t"]
    tokens = dense["checkpoint"]["tokens"]
    assert tokens == (dense["checkpoint"]["step"] + 1) * 128 * 8192 == 562_022_055_936
    expected_keys = {row["key"] for row in dense["rows"]}
    dense_scores = [mean(r["selected_value"] for r in dense["rows"] if r["category"] == c)
                    for c in CATEGORIES]
    dense_scores.append(mean(dense_scores))
    np.testing.assert_allclose(dense_scores, historical["groups"]["056t"], rtol=0, atol=1e-14)
    models = [{"label": f"MarinDNA 1B {tokens/1e9:.0f}B", "color": "#8c2d27",
               "scores": dense_scores, "DNA_tokens": tokens, "source": "historical/056t"}]
    sources = {"historical": {"path": str(historical_path.relative_to(PROJECT)),
                              "sha256": sha256(historical_path)}}
    with (HERE / "all-category-scores.csv").open() as stream:
        trajectory = list(csv.DictReader(stream))
    for path, checkpoint, run, label, color in [
        (args.short_results, "dna-216b", "216B-token run", "MarinMoE 1.5B–60M 216B", "#f28e2b"),
        (args.long_results, "dna-1000b", "1T-token run", "MarinMoE 1.5B–60M 1T", "#c43c39"),
    ]:
        raw = json.loads(path.read_text())[checkpoint]
        tasks = raw["zero_shot"]["profiles"]["context-8192"]["tasks"]
        assert len(tasks["rows"]) == 20 and {r["key"] for r in tasks["rows"]} == expected_keys
        assert sum(r["samples"] for r in tasks["rows"]) == sum(r["samples"] for r in dense["rows"]) == 1_727_943
        scores = [mean(r["natural"] for r in tasks["rows"] if r["category"] == c) for c in CATEGORIES]
        scores.append(mean(scores))
        np.testing.assert_allclose(scores, [tasks["groups"]["natural"][c] for c in [*CATEGORIES, "Composite"]],
                                   rtol=0, atol=1e-14)
        for metric, value in zip(["Conservation", "Motifs", "Core/non-core", "SV impact", "Composite +SV"], scores, strict=True):
            previous = next(r for r in trajectory if r["run"] == run and r["checkpoint"] == checkpoint
                            and r["metric"] == metric and r["readout"] == "Zero-shot likelihood")
            assert abs(float(previous["score"]) - value) < 1e-14
        metadata = json.loads(path.with_name("provenance.json").read_text())["catalog"]["models"][checkpoint]
        models.append({"label": label, "color": color, "scores": scores,
                       "DNA_tokens": metadata["DNA_tokens"], "source": checkpoint,
                       "checkpoint": metadata["checkpoint"], "initialization": "language-pretrained",
                       "task_scores": tasks["rows"]})
        sources[run] = {"file": path.name, "sha256": sha256(path),
                        "provenance_sha256": sha256(path.with_name("provenance.json")),
                        "native_source": raw["sources"]["zero_shot"]}
    for key, label, color in PUBLISHED:
        scores = historical["published_groups"][key]
        assert abs(mean(scores[:4]) - scores[4]) < 1e-14
        models.append({"label": label, "color": color, "scores": scores,
                       "source": "historical/published_groups/" + key})
    assert len(models) == 8
    payload = {"categories": DISPLAY, "models": models}
    (HERE / "benchmark-comparison-data.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "benchmark-comparison-provenance.json").write_text(json.dumps({
        "sources": sources,
        "historical_source_commit": manifest["commit"],
        "published_leaderboard_revision": historical["published_revision"],
        "original_plot": "https://github.com/plantcad/plantcad2/pull/20#issuecomment-5585800169",
        "dense_tokens_exact": tokens, "dense_tokens_billions": tokens / 1e9,
        "dense_token_crosscheck": "(checkpoint step 535985 + 1) × batch 128 × sequence 8192",
        "readout": "Original full-benchmark zero-shot rule: max completed forward/RC task metric for non-SV; SV forward",
        "aggregation": "Equal-weight task mean within each category; Composite gives each of four categories 25%; AF separate",
        "moe_scope": "Final language-pretrained checkpoints only, 8192-base examples, character-forced nucleotide tokenization",
        "historical_values": "Previously computed values reused without running models or fitting probes",
        "colors": "Exact dense and published-model fills sampled from original plot; original orange/red Marin colors reused for the two MoE runs",
        "excluded": ["PlantCAD 512 bp", "other dense checkpoints", "Composite without SV", "probe scores"],
        "values": 40, "GitHub_comment_updates": False,
        "panels": ["All five benchmark categories", "Conservation ranking", "SV impact ranking", "Composite ranking"],
        "ranking_order": {
            metric: [{"model": model["label"], "score": model["scores"][DISPLAY.index(metric)]}
                     for model in sorted(models, key=lambda m: m["scores"][DISPLAY.index(metric)], reverse=True)]
            for metric in ["Conservation", "SV impact", "Composite"]
        },
        "displayed_bars": 64,
        "model_summary": {
            "DNA_tokens": 1_000_000_192_512,
            "DNA_tokens_per_epoch": 21_615_869_952,
            "DNA_epochs": 1_000_000_192_512 / 21_615_869_952,
            "rounded_total_parameters": 1_500_000_000,
            "rounded_active_parameters_per_token": 60_000_000,
            "heuristic_dense_equivalent_parameters": (1_500_000_000 * 60_000_000) ** .5,
            "approximation_sources": [
                "https://proceedings.mlr.press/v162/clark22a.html",
                "https://marin.community/analysis/scaling-laws-analysis/",
            ],
            "interpretation": "Geometric-mean size heuristic using rounded counts; not a measured dense-model match on DNA",
            "attribution": "Marin explicitly uses the geometric mean; Clark et al. 2022 Section 5.1 derives the more general effective-parameter scaling framework",
        },
    }, indent=2) + "\n")
    render(models)
    print(json.dumps({model["label"]: dict(zip(DISPLAY, model["scores"], strict=True)) for model in models}, indent=2))


if __name__ == "__main__":
    main()
