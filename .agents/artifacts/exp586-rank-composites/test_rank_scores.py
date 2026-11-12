"""Scientific contracts for frozen-reference ranks and unchanged native scores."""

import csv
import json
from pathlib import Path
from statistics import mean

import pytest
from rank_scores import aggregate, percentile
from scipy.stats import percentileofscore

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("reference", [[1, 2, 3], [1, 1, 2, 3, 3], [2] * 9])
@pytest.mark.parametrize("score", [0, 1, 1.00000000001, 2, 3, 4])
def test_ties_and_bounds_against_scipy(reference, score):
    assert percentile(score, reference) == pytest.approx(percentileofscore(reference, score, kind="mean"))


def test_equal_family_weight_instead_of_equal_task_weight():
    families = {"eight_tasks": [str(i) for i in range(8)], "one_task": ["last"]}
    reference = {"a": {**{str(i): 0 for i in range(8)}, "last": 1}}
    candidate = {**{str(i): 1 for i in range(8)}, "last": 0}
    assert aggregate(candidate, reference, families)["score"] == 50
    assert mean(percentile(candidate[k], [reference["a"][k]]) for k in candidate) != 50


def test_candidates_never_enter_reference():
    reference = {"a": {"x": 1}, "b": {"x": 2}}
    before = json.dumps(reference)
    original = aggregate({"x": 1.5}, reference, {"family": ["x"]})
    aggregate({"x": 100}, reference, {"family": ["x"]})
    assert aggregate({"x": 1.5}, reference, {"family": ["x"]}) == original
    assert json.dumps(reference) == before


def test_missing_or_duplicate_tasks_rejected():
    for candidate, reference, families in [
        ({"a": 1}, {"r": {"a": 1}}, {"f": ["a", "af"]}),
        ({"a": 1}, {"r": {}}, {"f": ["a"]}),
        ({"a": 1}, {"r": {"a": 1}}, {"f": ["a"], "g": ["a"]}),
    ]:
        with pytest.raises(ValueError):
            aggregate(candidate, reference, families)


def test_frozen_member_ids_and_native_values():
    reference = json.loads((HERE / "frozen-references.json").read_text())
    marin = reference["Marin Avg Rank"]["reference"]
    assert len(reference["Avg Rank"]["reference"]) == 11
    assert set(marin["Zero-shot"]) == set(marin["Linear probe"]) == {
        "base-000b", "short-002b", "short-031b", "short-124b", "short-216b",
        "long-010b", "long-182b", "long-545b", "long-1000b"}
    old = list(csv.DictReader((HERE.with_name("exp586-checkpoint-trajectories") / "all-category-scores.csv").open()))
    new = list(csv.DictReader((HERE / "all-category-scores.csv").open()))
    native = [row for row in old if row["metric"] in reference["Marin Avg Rank"]["families"]]
    assert len(native) == 100
    for row in native:
        row = {**row, "readout": row["readout"].replace("Zero-shot likelihood", "Zero-shot")}
        assert row in new
    old_b = json.loads((HERE.with_name("exp586-checkpoint-trajectories") / "benchmark-comparison-data.json").read_text())
    new_b = json.loads((HERE / "benchmark-comparison-data.json").read_text())
    assert [(r["label"], r["scores"][:4]) for r in old_b["models"]] == [(r["label"], r["scores"][:4]) for r in new_b["models"]]


def test_saved_ranks_independently_recomputed():
    frozen = json.loads((HERE / "frozen-references.json").read_text())
    native = json.loads((HERE / "native-task-scores.json").read_text())["checkpoints"]
    reference = frozen["Marin Avg Rank"]
    for observation in native.values():
        for readout, values in observation["values"].items():
            task_values = {task: percentileofscore([r[task] for r in reference["reference"][readout].values()],
                                                  score, kind="mean") for task, score in values.items()}
            expected = sum(sum(task_values[t] for t in tasks) / len(tasks) for tasks in reference["families"].values()) / 5
            assert aggregate(values, reference["reference"][readout], reference["families"])["score"] == pytest.approx(expected)
    # Every unique member occupies a place in the same frozen distribution.
    for readout in ["Zero-shot", "Linear probe"]:
        assert mean(aggregate(v, reference["reference"][readout], reference["families"])["score"]
                    for v in reference["reference"][readout].values()) == pytest.approx(50)
