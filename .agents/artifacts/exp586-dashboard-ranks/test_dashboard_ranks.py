"""Check actual dashboard reproduction, ties, and external-model boundaries."""

import json
from pathlib import Path

import pandas as pd
import pytest
from dashboard_ranks import aggregate, percentile
from scipy.stats import percentileofscore

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("reference", [[1, 2, 3], [1, 1, 2, 3, 3], [2] * 11])
@pytest.mark.parametrize("score", [-1, 1, 1.000001, 2, 3, 4])
def test_external_percentiles_ties_and_bounds(reference, score):
    result = percentile(score, reference)
    assert result == pytest.approx(percentileofscore(reference, score, kind="rank"))
    assert 0 <= result <= 100


@pytest.mark.parametrize("reference", [[1, 2, 3], [1, 1, 2, 3, 3], [2] * 11])
def test_reference_members_match_pandas(reference):
    expected = pd.Series(reference).rank(pct=True) * 100
    assert [percentile(score, reference) for score in reference] == pytest.approx(expected.tolist())


def test_task_weights_and_frozen_reference():
    families = {"large": ["a", "b"], "small": ["c"]}
    reference = {"r1": {"a": 0, "b": 0, "c": 1}, "r2": {"a": 1, "b": 1, "c": 2}}
    before = json.dumps(reference)
    candidate = {"a": 2, "b": 2, "c": 0}
    assert aggregate(candidate, reference, families)["score"] == pytest.approx(200 / 3)
    aggregate({"a": 5, "b": 5, "c": 5}, reference, families)
    assert json.dumps(reference) == before
    assert aggregate(candidate, reference, families)["score"] == pytest.approx(200 / 3)


def test_all_dashboard_baselines_and_task_ranks_verified():
    audit = json.loads((HERE / "validation.json").read_text())
    assert audit["raw_values_verified"] == audit["task_ranks_verified"] == 220
    assert len(audit["all_baseline_models"]) == 11
    for row in audit["all_baseline_models"].values():
        assert row["tasks_checked"] == 20
        assert row["ours"] == pytest.approx(row["dashboard"], abs=1e-12)
        assert row["max_per_task_error"] < 1e-12
    assert audit["all_baseline_models"]["evo2_20b"]["ours"] == pytest.approx(82.27272727272728)


def test_missing_duplicate_and_nonfinite_values_rejected():
    with pytest.raises(ValueError):
        aggregate({"a": 0}, {"r": {"a": 0}}, {"one": ["a"], "two": ["a"]})
    with pytest.raises(ValueError):
        aggregate({"a": 0}, {"r": {"a": 0}}, {"one": ["a", "b"]})
    for score, reference in [(1, []), (float("nan"), [1]), (1, [float("inf")])]:
        with pytest.raises(ValueError):
            percentile(score, reference)
