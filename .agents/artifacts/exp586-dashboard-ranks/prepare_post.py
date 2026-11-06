"""Recompute and audit dashboard-compatible ranks; prepare one comment update."""

import ast
import csv
import difflib
import hashlib
import importlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from dashboard_ranks import aggregate
from scipy.stats import percentileofscore

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.with_name("exp586-rank-composites")
SOURCE = HERE / "dashboard-source"
REVISION = "f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29"
SPACE = "https://huggingface.co/spaces/plantcad/plantcad2-zeroshot-leaderboard"
POST = "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065"


def published_id(task: str) -> str:
    if task.startswith("conservation_"):
        return task.replace("conservation_", "cons_").replace("non_tis", "nontis")
    if task.startswith("motif_"):
        return task.replace("_tis", "_start_sites").replace("_tts", "_stop_sites")
    if task.startswith("core_"):
        return task.replace("core_", "coreclass_", 1)
    return "sv_impact_auprc"


def main() -> None:
    live = json.loads((HERE / "live-source.json").read_text())
    original = live["body"].replace("\r\n", "\n")
    frozen = json.loads((PUBLISHED / "frozen-references.json").read_text())["Avg Rank"]
    reference, families = frozen["reference"], frozen["families"]
    assert frozen["reference_revision"] == REVISION and len(reference) == 11
    task_ids = {published_id(task): task for group in families.values() for task in group}
    assert len(task_ids) == 20
    raw = pd.read_csv(SOURCE / "data/results.csv")
    tasks = pd.read_csv(SOURCE / "data/tasks.csv")
    assert len(raw) == 220 and not raw.duplicated(["model", "task_id"]).any()
    assert set(raw.model) == set(reference) and set(raw.task_id) == set(task_ids)
    assert tasks.higher_is_better.eq(1).all()
    assert all(row.value == reference[row.model][task_ids[row.task_id]] for row in raw.itertuples())

    # Execute the actual dashboard function, isolated from its Streamlit UI.
    tree = ast.parse((SOURCE / "app.py.txt").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rank_within_task")
    namespace = {"pd": pd, "np": np}
    # The checked-in, pinned dashboard function is intentionally run for an exact audit.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE / "app.py.txt"), "exec"), namespace)  # noqa: S102
    dashboard_tasks = namespace["rank_within_task"](raw, tasks)
    dashboard_means = dashboard_tasks.groupby("model").pct_rank.mean().sort_values(ascending=False)
    verified = {}
    for identity, score in dashboard_means.items():
        ours = aggregate(reference[identity], reference, families)
        expected_tasks = dashboard_tasks[dashboard_tasks.model.eq(identity)].set_index("task_id").pct_rank
        errors = [abs(ours["tasks"][task_ids[task]] - expected) for task, expected in expected_tasks.items()]
        assert max(errors) < 1e-12 and abs(ours["score"] - score) < 1e-12
        verified[identity] = {"dashboard": score, "ours": ours["score"],
                              "max_per_task_error": max(errors), "tasks_checked": len(errors)}
    assert abs(verified["evo2_20b"]["ours"] - 82.27272727272728) < 1e-12

    prior = json.loads((PUBLISHED / "benchmark-comparison-data.json").read_text())
    native = json.loads((PUBLISHED / "native-task-scores.json").read_text())["checkpoints"]
    repo = PUBLISHED.parents[2]
    historical = json.loads((repo / "zero-shot-leaderboard/marindna_exp586/historical/exp472-lr5e4-full-coreweave-20260831.json").read_text())
    baseline_ids = {"PlantCAD2.5-Large": "PlantCAD2.5-L", "PlantCAD2-Large": "PlantCAD2-L",
                    "PlantCAD2-Medium": "PlantCAD2-M", "PlantCAD2-Small": "PlantCAD2-S", "Evo 2 20B": "evo2_20b"}
    data, rankings = deepcopy(prior), {}
    for model in data["models"]:
        label = model["label"]
        if label == "MarinDNA 1B 562B":
            scores = {row["key"]: row["selected_value"] for row in historical["results"]["056t"]["rows"]}
        elif label.startswith("MarinMoE"):
            identity = "short-216b" if label.endswith("216B") else "long-1000b"
            scores = {task: score for task, score in native[identity]["values"]["Zero-shot"].items() if task != "af"}
        else:
            scores = reference[baseline_ids[label]]
        result = aggregate(scores, reference, families)
        for task, score in scores.items():
            assert abs(result["tasks"][task] - percentileofscore([row[task] for row in reference.values()], score, kind="rank")) < 1e-12
        assert all(abs(mean(scores[task] for task in group) - model["scores"][index]) < 1e-14
                   for index, group in enumerate(families.values()))
        rankings[label] = result
        model["scores"][-1] = result["score"]
    for before, after in zip(prior["models"], data["models"], strict=True):
        restored = deepcopy(after)
        restored["scores"][-1] = before["scores"][-1]
        assert restored == before
    (HERE / "benchmark-comparison-data.json").write_text(json.dumps(data, indent=2) + "\n")
    (HERE / "benchmark-ranks.json").write_text(json.dumps(rankings, indent=2) + "\n")
    audit = {"baseline_revision": REVISION, "raw_values_verified": 220, "task_ranks_verified": 220,
             "all_baseline_models": verified, "native_scores_unchanged": True,
             "other_five_figures_unchanged": True, "baseline_ranks_unchanged_by_Marin_models": True,
             "external_model_rule": "scipy.stats.percentileofscore(kind='rank') against fixed 11 baselines"}
    (HERE / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (HERE / "baseline-validation.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "dashboard_Avg_Rank", "post_Avg_Rank", "difference", "tasks_checked"])
        for identity, row in verified.items():
            writer.writerow([identity, row["dashboard"], row["ours"], row["ours"] - row["dashboard"], 20])

    (HERE / "figures").mkdir(exist_ok=True)
    sys.path.insert(0, str(PUBLISHED))
    plotter = importlib.import_module("plot_benchmark_comparison")
    plotter.HERE = HERE
    plotter.RANGES["Avg Rank"] = (45, 100)
    plotter.TICKS["Avg Rank"] = [50, 60, 70, 80, 90, 100]
    plotter.main()

    body, edits = original, []

    def replace_once(old: str, new: str) -> None:
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new, 1)
        edits.append({"old": old, "new": new})

    figure = re.search(r'<img src="([^"]+/benchmark-comparison\.png)" alt="([^"]+)" />', body)
    assert figure
    replace_once(figure[1], "figures/benchmark-comparison.png")
    replace_once("Avg Rank averages per-task percentiles within four equally weighted families against 11 fixed published baseline models.",
                 "Avg Rank uses the dashboard's average one-based percentile ranks against the fixed 11-model baseline reference, with each of 20 tasks weighted 5%: Conservation 15%, Motifs 40%, Core/non-core 40% and SV impact 5%.")
    caption = "_Top: AUROC, accuracy and average precision ×100; Avg Rank is a percentile. Higher is better._"
    explanation = "Avg Rank uses the leaderboard’s baseline scores and percentile calculation:[^rank-baselines] each of 20 tasks contributes 5%—Conservation 15%, Motifs 40%, Core/non-core 40% and SV impact 5%; AF is excluded.[^rank-weighting]"
    replace_once(caption, caption + "\n\n" + explanation)
    old_definition = next(line for line in body.splitlines() if line.startswith("- **Avg Rank**"))
    replace_once(old_definition, "- **Avg Rank** equally averages the 20 zero-shot task percentiles (5% each), using the dashboard’s average one-based ranks against the fixed 11-model baseline reference.[^rank-baselines][^rank-weighting] AF is excluded.")
    footnotes = f"""\n\n[^rank-baselines]: [Leaderboard baseline scores]({SPACE}/blob/{REVISION}/data/results.csv) and [task definitions]({SPACE}/blob/{REVISION}/data/tasks.csv), frozen at `f3c4ddb`: 11 models, 220 scores.
[^rank-weighting]: Dashboard [percentiles]({SPACE}/blob/{REVISION}/app.py#L42) and [equal-task mean]({SPACE}/blob/{REVISION}/app.py#L137). All 11 baseline averages match the unfiltered dashboard. Marin models use [rank percentiles](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html) against that fixed reference without changing baseline ranks.
"""
    body += footnotes
    restored = body.removesuffix(footnotes)
    for edit in reversed(edits):
        assert restored.count(edit["new"]) == 1
        restored = restored.replace(edit["new"], edit["old"], 1)
    assert restored == original
    assert body.split('<p align="center">', 1)[0] == original.split('<p align="center">', 1)[0]
    assert body.count("<details>") == original.count("<details>") == 6
    original_images = re.findall(r'<img src="([^"]+)" alt="([^"]+)"', original)
    new_images = re.findall(r'<img src="([^"]+)" alt="([^"]+)"', body)
    assert len(new_images) == 6 and original_images[1:] == new_images[1:]
    (HERE / "prepared.md").write_text(body)
    (HERE / "text-edits.json").write_text(json.dumps({"replacements": edits, "append": footnotes}, indent=2) + "\n")
    (HERE / "post.diff").write_text("".join(difflib.unified_diff(original.splitlines(keepends=True),
        body.splitlines(keepends=True), fromfile="live-comment", tofile="dashboard-compatible-comment", n=0)))
    (HERE / "provenance.json").write_text(json.dumps({
        "destination": POST, "source_updated_at": live["updated_at"],
        "source_body_sha256": hashlib.sha256(live["body"].encode()).hexdigest(),
        "baseline_revision": REVISION, "raw_sha256": hashlib.sha256((SOURCE / "data/results.csv").read_bytes()).hexdigest(),
        "reference_models": 11, "tasks": 20, "task_weight": .05,
        "family_weights": {family: len(group) / 20 for family, group in families.items()},
        "manual_wording_preserved_outside_explicit_edits": True,
        "native_scores_probe_win_rates_normalized_scores_and_losses_unchanged": True,
    }, indent=2) + "\n")
    print(json.dumps({"baseline_validation": verified,
                      "displayed_Avg_Rank": {label: row["score"] for label, row in rankings.items()}}, indent=2))


if __name__ == "__main__":
    main()
