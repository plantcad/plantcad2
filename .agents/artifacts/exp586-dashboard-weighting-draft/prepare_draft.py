"""Draft dashboard task weighting from the current author-edited results post."""

import csv
import difflib
import hashlib
import importlib
import io
import json
import re
import subprocess
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.with_name("exp586-rank-composites")
REVISION = "f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29"
SPACE = "https://huggingface.co/spaces/plantcad/plantcad2-zeroshot-leaderboard"
POST = "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def published_id(task: str) -> str:
    if task.startswith("conservation_"):
        return task.replace("conservation_", "cons_").replace("non_tis", "nontis")
    if task.startswith("motif_"):
        return task.replace("_tis", "_start_sites").replace("_tts", "_stop_sites")
    if task.startswith("core_"):
        return task.replace("core_", "coreclass_", 1)
    return "sv_impact_auprc"


def main() -> None:
    # This file is the freshly fetched live source, not the historical draft.
    live = json.loads((HERE / "live-source.json").read_text())
    original = live["body"].replace("\r\n", "\n")
    frozen = json.loads((PUBLISHED / "frozen-references.json").read_text())["Avg Rank"]
    reference, families = frozen["reference"], frozen["families"]
    assert len(reference) == 11 and sum(map(len, families.values())) == 20
    assert frozen["reference_revision"] == REVISION
    task_ids = {published_id(task): task for tasks in families.values() for task in tasks}
    with urllib.request.urlopen(f"{SPACE}/raw/{REVISION}/data/results.csv", timeout=30) as response:
        raw_csv = response.read()
    raw = list(csv.DictReader(io.StringIO(raw_csv.decode())))
    assert len(raw) == 220
    identities = {(row["model"], task_ids[row["task_id"]]) for row in raw}
    assert len(identities) == 220
    assert all(float(row["value"]) == reference[row["model"]][task_ids[row["task_id"]]] for row in raw)
    (HERE / "reference-results.csv").write_bytes(raw_csv)

    sys.path.insert(0, str(PUBLISHED))
    rank_module = importlib.import_module("rank_scores")
    prior = json.loads((PUBLISHED / "benchmark-comparison-data.json").read_text())
    old_ranks = json.loads((PUBLISHED / "benchmark-ranks.json").read_text())
    native = json.loads((PUBLISHED / "native-task-scores.json").read_text())["checkpoints"]
    repo = PUBLISHED.parents[2]
    historical = json.loads((repo / "zero-shot-leaderboard/marindna_exp586/historical/exp472-lr5e4-full-coreweave-20260831.json").read_text())
    baseline_ids = {"PlantCAD2.5-Large": "PlantCAD2.5-L", "PlantCAD2-Large": "PlantCAD2-L",
                    "PlantCAD2-Medium": "PlantCAD2-M", "PlantCAD2-Small": "PlantCAD2-S", "Evo 2 20B": "evo2_20b"}
    data = deepcopy(prior)
    rankings = {}
    for model in data["models"]:
        label = model["label"]
        if label == "MarinDNA 1B 562B":
            scores = {row["key"]: row["selected_value"] for row in historical["results"]["056t"]["rows"]}
        elif label.startswith("MarinMoE"):
            identity = "short-216b" if label.endswith("216B") else "long-1000b"
            scores = {task: score for task, score in native[identity]["values"]["Zero-shot"].items() if task != "af"}
        else:
            scores = reference[baseline_ids[label]]
        result = rank_module.aggregate(scores, reference, families)
        assert result == old_ranks[label]
        assert len(result["tasks"]) == 20
        assert all(abs(mean(scores[task] for task in tasks) - model["scores"][index]) < 1e-14
                   for index, tasks in enumerate(families.values()))
        new_score = mean(result["tasks"].values())
        weighted_check = sum(len(tasks) * result["families"][family] for family, tasks in families.items()) / 20
        assert abs(new_score - weighted_check) < 1e-12
        rankings[label] = {"previous_family_balanced": model["scores"][-1], "Avg Rank": new_score,
                           "task_percentiles": result["tasks"]}
        model["scores"][-1] = new_score
    for before, after in zip(prior["models"], data["models"], strict=True):
        restored = deepcopy(after)
        restored["scores"][-1] = before["scores"][-1]
        assert restored == before
    (HERE / "benchmark-comparison-data.json").write_text(json.dumps(data, indent=2) + "\n")
    (HERE / "task-weighted-ranks.json").write_text(json.dumps(rankings, indent=2) + "\n")
    (HERE / "figures").mkdir(exist_ok=True)
    # Reuse the inspected renderer without modifying the published artifact files.
    plotter = importlib.import_module("plot_benchmark_comparison")
    plotter.HERE = HERE
    plotter.main()

    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    percentile_url = f"https://github.com/plantcad/plantcad2/blob/{code_revision}/.agents/artifacts/exp586-rank-composites/rank_scores.py#L7"
    body = original
    edits = []

    def replace_once(old: str, new: str) -> None:
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new, 1)
        edits.append((old, new))

    figure = re.search(r'<img src="([^"]+/benchmark-comparison\.png)" alt="([^"]+)" />', body)
    assert figure
    replace_once(figure[1], "figures/benchmark-comparison.png")
    old_alt = "Avg Rank averages per-task percentiles within four equally weighted families against 11 fixed published baseline models."
    new_alt = "Avg Rank equally averages the 20 per-task percentiles against 11 fixed published baseline models: each task has 5% weight, giving Conservation 15%, Motifs 40%, Core/non-core 40% and SV impact 5%."
    replace_once(old_alt, new_alt)
    caption = "_Top: AUROC, accuracy and average precision ×100; Avg Rank is a percentile. Higher is better._"
    explanation = "Avg Rank uses the leaderboard’s baseline scores[^rank-baselines], with each of 20 tasks weighted 5%: Conservation 15%, Motifs 40%, Core/non-core 40% and SV impact 5%; AF is excluded.[^rank-weighting]"
    replace_once(caption, caption + "\n\n" + explanation)
    old_definition = next(line for line in body.splitlines() if line.startswith("- **Avg Rank**"))
    new_definition = "- **Avg Rank** equally averages the 20 zero-shot task percentiles (5% each) against the fixed 11-model baseline reference.[^rank-baselines][^rank-weighting] AF is excluded."
    replace_once(old_definition, new_definition)
    footnotes = f"""\n\n[^rank-baselines]: [Raw baseline scores]({SPACE}/blob/{REVISION}/data/results.csv) and [task definitions]({SPACE}/blob/{REVISION}/data/tasks.csv), frozen at `f3c4ddb`: 11 models, 220 scores.
[^rank-weighting]: [Dashboard weighting]({SPACE}/blob/{REVISION}/app.py#L137); [percentile rule used here]({percentile_url}). Fixed 11-model reference, midpoint ties; the dashboard’s one-based ranks add 4.545 points for baseline models.
"""
    body += footnotes
    restored = body.removesuffix(footnotes)
    for old, new in reversed(edits):
        assert restored.count(new) == 1
        restored = restored.replace(new, old, 1)
    assert restored == original, "A change outside the approved weighting/source edits occurred"
    assert body.split('<p align="center">', 1)[0] == original.split('<p align="center">', 1)[0]
    assert body.count("<details>") == original.count("<details>") == 6
    assert len(re.findall(r'<img src="', body)) == 6
    (HERE / "draft.md").write_text(body)
    (HERE / "draft.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), body.splitlines(keepends=True),
        fromfile="live-comment", tofile="unpublished-draft", n=0)))
    provenance = {
        "status": "Unpublished draft; no GitHub post changes", "destination": POST,
        "source_updated_at": live["updated_at"], "source_body_sha256_LF": digest(original.encode()),
        "manual_wording_preserved_outside_explicit_edits": True,
        "approved_text_changes": "First image URL/alt, main-text weighting/source paragraph, Avg Rank definition and two source footnotes",
        "baseline_revision": REVISION, "baseline_raw_sha256": digest(raw_csv), "baseline_values_equal": 220,
        "reference_models": 11, "tasks": 20, "per_task_weight": .05,
        "family_weights": {family: len(tasks) / 20 for family, tasks in families.items()},
        "percentiles": "Unchanged fixed-reference midpoint percentiles; only the task aggregation changes",
        "native_scores_unchanged": True, "probe_and_checkpoint_summaries_unchanged": True,
        "source_plotter": str((PUBLISHED / "plot_benchmark_comparison.py").relative_to(repo)),
        "source_code_revision": code_revision,
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"draft": str(HERE / "draft.md"), "Avg Rank": {label: row["Avg Rank"] for label, row in rankings.items()}}, indent=2))


if __name__ == "__main__":
    main()
