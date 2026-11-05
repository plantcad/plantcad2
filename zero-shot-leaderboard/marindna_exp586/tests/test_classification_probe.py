"""The classifier fitting boundary never receives final-test rows."""

import numpy as np
import pandas as pd
import pytest

from exp586_evals.classification_probe import fit_head, fit_task, metric


def test_scaling_and_penalty_selection_use_training_rows_only():
    rng = np.random.default_rng(586)
    features = rng.normal(size=(180, 4))
    labels = (features[:, 0] > 0).astype(int)
    chromosomes = np.repeat(["1", "2", "3", "4", "5", "6"], 30)
    model, metadata = fit_head(features, labels, chromosomes, kind="core", jobs=1)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, features.mean(axis=0))
    assert metadata["development_score"] > 0.95
    assert metric(labels, model.decision_function(features), "core") > 0.98


def test_shared_chromosomes_are_rejected_before_fitting(tmp_path):
    frame = pd.DataFrame(
        {
            "row_id": [0, 1],
            "chrom": ["1", "1"],
            "split": ["train", "test"],
            "visible_sha256": ["a", "b"],
            "label": [0, 1],
        }
    )
    path = tmp_path / "membership.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="chromosome"):
        fit_task(
            {"key": "core_example", "kind": "core"},
            path,
            {"row_id": np.array([0, 1])},
            tmp_path / "output",
            jobs=1,
        )


def test_misaligned_rows_are_rejected_before_fitting(tmp_path):
    frame = pd.DataFrame({"row_id": [0, 1]})
    path = tmp_path / "membership.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="rows"):
        fit_task(
            {"key": "core_example", "kind": "core"},
            path,
            {"row_id": np.array([1, 0])},
            tmp_path / "output",
            jobs=1,
        )


def test_completed_heads_resume_but_changed_source_data_is_rejected(
    tmp_path, monkeypatch
):
    import exp586_evals.classification_probe as module

    rng = np.random.default_rng(586)
    fwd, rc = rng.normal(size=(160, 4)), rng.normal(size=(160, 4))
    frame = pd.DataFrame(
        {
            "row_id": np.arange(160),
            "chrom": np.repeat(np.arange(8).astype(str), 20),
            "split": ["train"] * 120 + ["test"] * 40,
            "visible_sha256": [f"visible-{i}" for i in range(160)],
            "label": ((fwd[:, 0] + rc[:, 0]) > 0).astype(int),
        }
    )
    membership = tmp_path / "membership.parquet"
    frame.to_parquet(membership)
    arrays = {"row_id": np.arange(160), "fwd": fwd, "rc": rc}
    spec = {"key": "core_example", "kind": "core"}
    result = fit_task(spec, membership, arrays, tmp_path / "fits", jobs=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete cached head was fitted again")

    monkeypatch.setattr(module, "fit_head", forbidden)
    assert fit_task(spec, membership, arrays, tmp_path / "fits", jobs=1) == result
    arrays["fwd"][0, 0] += 1
    with pytest.raises(ValueError, match="different source data"):
        fit_task(spec, membership, arrays, tmp_path / "fits", jobs=1)
