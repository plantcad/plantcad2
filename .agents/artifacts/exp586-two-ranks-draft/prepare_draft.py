"""Draft family-unweighted and task-weighted ranks without changing the live post."""

import difflib
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
CURRENT = HERE.with_name("exp586-dashboard-ranks")
PUBLISHED = HERE.with_name("exp586-rank-composites")


def main() -> None:
    source = json.loads((HERE / "live-source.json").read_text())
    original = source["body"].replace("\r\n", "\n")
    ranks = json.loads((CURRENT / "benchmark-ranks.json").read_text())
    prior = json.loads((CURRENT / "benchmark-comparison-data.json").read_text())
    frozen = json.loads((PUBLISHED / "frozen-references.json").read_text())["Avg Rank"]
    data = deepcopy(prior)
    data["categories"][-1:] = ["Unweighted Avg Rank", "Weighted Avg Rank"]
    counts = {family: len(tasks) for family, tasks in frozen["families"].items()}
    per_task = {task: 1 / (4 * len(tasks)) for tasks in frozen["families"].values() for task in tasks}
    assert counts == {"Conservation": 3, "Motifs": 8, "Core/non-core": 8, "SV impact": 1}
    assert abs(sum(per_task.values()) - 1) < 1e-14
    comparisons = {}
    for before, model in zip(prior["models"], data["models"], strict=True):
        label = model["label"]
        result = ranks[label]
        assert result["score"] == model["scores"][-1]
        assert abs(mean(result["tasks"].values()) - result["score"]) < 1e-12
        score = mean(result["families"].values())
        assert abs(sum(result["tasks"][task] * weight for task, weight in per_task.items()) - score) < 1e-12
        model["scores"][-1:] = [score, result["score"]]
        restored = deepcopy(model)
        restored["scores"][-2:] = [before["scores"][-1]]
        assert restored == before
        comparisons[label] = {"Unweighted Avg Rank": score, "Weighted Avg Rank": result["score"]}
    # Independently recompute every baseline using the validated percentile code.
    sys.path.insert(0, str(CURRENT))
    from dashboard_ranks import aggregate

    audit = json.loads((CURRENT / "validation.json").read_text())
    for identity, values in frozen["reference"].items():
        result = aggregate(values, frozen["reference"], frozen["families"])
        assert abs(result["score"] - audit["all_baseline_models"][identity]["dashboard"]) < 1e-12
        assert abs(mean(result["families"].values()) - sum(result["tasks"][task] * weight for task, weight in per_task.items())) < 1e-12
    (HERE / "benchmark-comparison-data.json").write_text(json.dumps(data, indent=2) + "\n")
    (HERE / "rank-comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (HERE / "validation.json").write_text(json.dumps({
        "status": "Unpublished draft", "baseline_averages_checked": 11,
        "source_task_ranks_verified": 220, "percentile_convention_unchanged": True,
        "weighted_scores_match_PlantCAD2_leaderboard_and_current_post": True, "native_scores_unchanged": True,
        "default_Avg_Rank_is_unweighted": True, "weighted_rank_limited_to_first_figure": True,
        "unweighted_task_weights": per_task, "unweighted_family_weights": {family: .25 for family in counts},
        "weighted_task_weights": {task: .05 for task in per_task},
        "weighted_family_weights": {family: count / 20 for family, count in counts.items()},
    }, indent=2) + "\n")

    body, edits = original, []

    def replace_once(old: str, new: str) -> None:
        nonlocal body
        assert body.count(old) == 1, old
        body = body.replace(old, new, 1)
        edits.append({"old": old, "new": new})

    image = re.search(r'<img src="([^"]+/benchmark-comparison\.png)" alt="([^"]+)" />', body)
    assert image
    replace_once(image[1], "figures/benchmark-comparison.png")
    replace_once(image[2], "Zero-shot comparison of eight models. Top: Conservation, Motifs, Core/non-core, SV impact, Unweighted Avg Rank and Weighted Avg Rank share a marked broken axis, with a value on every bar. Native family scores are multiplied by 100 only for display. Bottom: four sorted horizontal comparisons show Conservation AUROC, SV average precision, Unweighted Avg Rank and Weighted Avg Rank, each with native values and marked breaks. Both ranks use the same one-based task percentiles against 11 fixed PlantCAD2 leaderboard baselines. Unweighted Avg Rank first averages within each family, then gives the four families 25% each. Weighted Avg Rank matches the PlantCAD2 leaderboard by weighting each task 5%, equivalently weighting family means by their task counts. Vertical-chart value labels are rotated 90 degrees; horizontal-chart value labels remain horizontal.")
    replace_once("_Top: AUROC, accuracy and average precision ×100; Avg Rank is a percentile. Higher is better._",
                 "_Top: AUROC, accuracy and average precision ×100; both average ranks are percentiles. Higher is better._")
    old_explanation = next(line for line in body.splitlines() if line.startswith("Avg Rank uses the leaderboard’s"))
    new_explanation = "Both ranks use the PlantCAD2 leaderboard’s baseline scores.[^rank-baselines] Unweighted Avg Rank averages task ranks within each family, then averages the four families equally (25% each). Weighted Avg Rank matches the PlantCAD2 leaderboard: family means are weighted by task count—Conservation 15%, Motifs 40%, Core/non-core 40% and SV impact 5%, equivalent to 5% per task. Both exclude AF; elsewhere, Avg Rank means unweighted.[^rank-weighting]"
    replace_once(old_explanation, new_explanation)
    old_definition = next(line for line in body.splitlines() if line.startswith("- **Avg Rank**"))
    replace_once(old_definition, "- **Avg Rank** means the unweighted family average: calculate each task’s percentile rank against the fixed 11-model reference, average within each family, then average the four family means equally (25% each).[^rank-baselines][^rank-weighting] The first figure also shows **Weighted Avg Rank**, the PlantCAD2 leaderboard calculation: weight family means by their task counts (3/8/8/1), equivalently averaging all 20 tasks equally. Neither rank includes AF.")
    old_baselines = next(line for line in body.splitlines() if line.startswith("[^rank-baselines]:"))
    replace_once(old_baselines, old_baselines.replace("[Leaderboard baseline scores]", "[PlantCAD2 leaderboard baseline scores]"))
    old_weighting = next(line for line in body.splitlines() if line.startswith("[^rank-weighting]:"))
    replace_once(old_weighting, old_weighting.replace("Dashboard [percentiles]", "PlantCAD2 leaderboard [percentiles]").replace("All 11 baseline averages match the unfiltered dashboard.", "Weighted Avg Rank matches all 11 baseline averages in the unfiltered PlantCAD2 leaderboard; unweighted uses the same percentiles with equal family weights."))
    assert "dashboard" not in body.lower()
    restored = body
    for edit in reversed(edits):
        assert restored.count(edit["new"]) == 1
        restored = restored.replace(edit["new"], edit["old"], 1)
    assert restored == original
    assert body.split('<p align="center">', 1)[0] == original.split('<p align="center">', 1)[0]
    old_images = re.findall(r'<img src="([^"]+)" alt="([^"]+)"', original)
    new_images = re.findall(r'<img src="([^"]+)" alt="([^"]+)"', body)
    assert len(new_images) == 6 and old_images[1:] == new_images[1:]
    (HERE / "draft.md").write_text(body)
    (HERE / "text-edits.json").write_text(json.dumps(edits, indent=2) + "\n")
    (HERE / "draft.diff").write_text("".join(difflib.unified_diff(original.splitlines(keepends=True),
        body.splitlines(keepends=True), fromfile="live-comment", tofile="unpublished-two-ranks-draft", n=0)))
    (HERE / "provenance.json").write_text(json.dumps({
        "status": "Unpublished draft; do not publish without user approval",
        "destination": "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065",
        "source_updated_at": source["updated_at"],
        "source_body_sha256": hashlib.sha256(source["body"].encode()).hexdigest(),
        "manual_wording_preserved_outside_explicit_edits": True,
        "other_five_figures_unchanged": True, "no_GitHub_mutations": True,
    }, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
