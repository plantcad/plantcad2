"""Render the 21 native per-task probe scores, grouped by task family."""

import csv
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean

import matplotlib
from artifact_io import normalize_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
UNITS = {"Conservation": "AUROC", "Motifs": "Exact-match accuracy", "Core/non-core": "AUROC",
         "SV impact": "Average precision", "Allele frequency": "Spearman correlation"}
MODEL_IDS = {"216b": "short-216b", "1t": "long-1000b"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> tuple[list[dict], int, dict]:
    """Check task membership, family means and the exact task-win tallies."""
    native = json.loads((HERE / "native-task-scores.json").read_text())["checkpoints"]
    summary = json.loads((HERE / "marin-summaries.json").read_text())
    assert summary["source_task_scores_sha256"] == digest(HERE / "native-task-scores.json")
    benchmark = json.loads((HERE / "benchmark-comparison-data.json").read_text())
    labels = {row["key"]: row["task"] for row in next(
        model for model in benchmark["models"] if model["label"] == "MarinMoE 1.5B–60M 216B")["task_scores"]}
    labels["af"] = "Alternate-allele frequency"
    cached_means = list(csv.DictReader((HERE / "all-category-scores.csv").open()))
    rows = []
    for family, tasks in summary["families"].items():
        for task in tasks:
            scores = {key: native[identity]["values"]["Linear probe"][task] for key, identity in MODEL_IDS.items()}
            delta = scores["1t"] - scores["216b"]
            winner = "1t" if delta > 0 else "216b" if delta < 0 else "tie"
            rows.append({"family": family, "metric": UNITS[family], "task_id": task,
                         "task": labels[task], "216b": scores["216b"], "1t": scores["1t"],
                         "delta_1t_minus_216b": delta, "winner": winner})
        for key, identity in MODEL_IDS.items():
            checkpoint = "dna-216b" if key == "216b" else "dna-1000b"
            original = next(float(row["score"]) for row in cached_means if row["checkpoint"] == checkpoint
                            and row["readout"] == "Linear probe" and row["metric"] == family)
            assert isclose(mean(row[key] for row in rows if row["family"] == family), original, rel_tol=0, abs_tol=1e-14)
            for row in [row for row in rows if row["family"] == family]:
                expected = .5 if row["winner"] == "tie" else float(row["winner"] == key)
                assert expected == summary["probe_win_rates"][identity]["tasks"][row["task_id"]]
    assert len(rows) == len({row["task_id"] for row in rows}) == 21
    precision = 4
    while any(row["winner"] != "tie" and (
        f"{row['216b']:.{precision}f}" == f"{row['1t']:.{precision}f}"
        or round(row["delta_1t_minus_216b"], precision) == 0) for row in rows):
        precision += 1
        assert precision <= 8, "A meaningful visible difference needs more precision than this scorecard supports"
    return rows, precision, summary


def main() -> None:
    rows, precision, summary = load_rows()
    palette = json.loads((HERE / "figure-palette.json").read_text())["marin"]
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(12.6, 9.8))
    title = fig.text(.5, .985, "Per-task linear-probe scores", ha="center", va="top", weight="bold", fontsize="large")
    ax = fig.add_axes((.03, .025, .94, .91))
    total_rows = 1 + len(UNITS) + len(rows)
    ax.set_xlim(0, 1)
    ax.set_ylim(total_rows, 0)
    ax.axis("off")
    texts = []
    for x, label, alignment in [(.02, "Task", "left"), (.60, "216B tokens", "center"),
                                 (.76, "1T tokens", "center"), (.925, "Δ (1T − 216B)", "center")]:
        texts.append(ax.text(x, .45, label, ha=alignment, va="center", weight="bold"))
    ax.plot([.53, .67], [.87, .87], color=palette["216b"], linewidth=3)
    ax.plot([.69, .83], [.87, .87], color=palette["1t"], linewidth=3)
    y = 1
    for family in summary["families"]:
        ax.add_patch(Rectangle((0, y), 1, 1, facecolor="#edf0f3", edgecolor="none"))
        texts.append(ax.text(.015, y + .5, f"{family} · {UNITS[family]}", ha="left", va="center", weight="bold"))
        y += 1
        for index, row in enumerate(row for row in rows if row["family"] == family):
            ax.add_patch(Rectangle((0, y), 1, 1, facecolor="white" if index % 2 == 0 else "#f8f9fb", edgecolor="none"))
            background = "#e4e7eb" if row["winner"] == "tie" else palette[row["winner"]]
            ax.add_patch(Rectangle((.85, y + .035), .15, .93, facecolor=background, edgecolor="none"))
            texts.append(ax.text(.025, y + .5, row["task"], ha="left", va="center"))
            for key, x in [("216b", .60), ("1t", .76)]:
                texts.append(ax.text(x, y + .5, f"{row[key]:.{precision}f}", ha="center", va="center"))
            texts.append(ax.text(.925, y + .5, f"{row['delta_1t_minus_216b']:+.{precision}f}",
                                  ha="center", va="center", color="white" if row["winner"] == "1t" else "#1f2328"))
            ax.plot([0, 1], [y + 1, y + 1], color="#e4e7eb", linewidth=.45)
            y += 1
    assert y == total_rows
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer).padded(1) for text in texts]
    assert all(ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1) for box in boxes)
    assert not [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes) if i < j and a.overlaps(b)]
    assert title.get_window_extent(renderer).y0 > ax.bbox.y1
    target = HERE / "figures" / "probe-scorecard"
    fig.savefig(target.with_suffix(".png"), dpi=160, facecolor="white")
    fig.savefig(target.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    normalize_svg(target.with_suffix(".svg"))
    plt.close(fig)
    provenance = {"rows": rows, "precision": precision, "difference": "1T − 216B, native metric units",
                  "color": "Winning model only; no magnitude encoding",
                  "source_sha256": {name: digest(HERE / name) for name in ["native-task-scores.json", "marin-summaries.json", "benchmark-comparison-data.json"]},
                  "validated": "21 unique tasks, family means match displayed bars, unrounded task wins match summary; no false visible ties"}
    (HERE / "probe-scorecard-data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (HERE / "probe-scorecard.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rendered {len(rows)} tasks at {precision} decimal places; all source and layout checks passed.")


if __name__ == "__main__":
    main()
