"""Check family weighting, min–max semantics and final-only pairwise wins."""

import csv
import json
from pathlib import Path

import pytest
from marin_summaries import normalized_average, pairwise_win_rate

HERE = Path(__file__).resolve().parent


def test_normalize_family_means_before_equal_family_average():
    values = {"base": {"a": 10, "b": .3}, "mid": {"a": 20, "b": .35}, "end": {"a": 30, "b": .5}}
    normalized = normalized_average(values)
    assert normalized["checkpoints"]["base"]["score"] == 0
    assert normalized["checkpoints"]["end"]["score"] == 1
    assert normalized["checkpoints"]["mid"]["score"] == pytest.approx((.5 + .25) / 2)
    assert normalized_average({**values, "shared_duplicate": values["base"]})["bounds"] == normalized["bounds"]


def test_pairwise_wins_family_weighting_and_ties():
    models = {"a": {"x": 1, "y": 2, "z": 3, "singleton": 2},
              "b": {"x": 0, "y": 2, "z": 4, "singleton": 1}}
    result = pairwise_win_rate(models, {"three": ["x", "y", "z"], "one": ["singleton"]})
    assert result["a"]["score"] == 75
    assert result["b"]["score"] == 25
    assert result["a"]["tasks"]["y"] == .5
    assert sum(row["score"] for row in result.values()) == 100


def test_reject_incomplete_or_undefined_normalization():
    with pytest.raises(ValueError):
        normalized_average({"a": {"f": 1}, "b": {"f": 1}})
    with pytest.raises(ValueError):
        normalized_average({"a": {"f": 1}, "b": {"g": 2}})
    with pytest.raises(ValueError):
        pairwise_win_rate({"a": {"f": 1}, "b": {}}, {"family": ["f"]})


def test_saved_native_families_unchanged_and_normalization_independent():
    old = [row for row in csv.DictReader((HERE / "all-category-scores.csv").open()) if row["metric"] != "Marin Avg Rank"]
    new = list(csv.DictReader((HERE / "normalized-category-scores.csv").open()))
    assert len(old) == 100 and all(row in new for row in old)
    result = json.loads((HERE / "marin-summaries.json").read_text())
    for readout, data in result["normalization"].items():
        for row in new:
            if row["metric"] != "Avg normalized score" or row["readout"] != readout:
                continue
            native = {r["metric"]: float(r["score"]) for r in old
                      if all(r[k] == row[k] for k in ["run", "checkpoint", "readout"])}
            expected = sum((score - data["bounds"][family]["min"]) /
                           (data["bounds"][family]["max"] - data["bounds"][family]["min"])
                           for family, score in native.items()) / 5
            assert float(row["score"]) == pytest.approx(expected)
    assert set(result["probe_win_rates"]) == {"short-216b", "long-1000b"}
    assert result["probe_win_rates"]["long-1000b"]["score"] == pytest.approx(65.83333333333333)
