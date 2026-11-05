"""Whole-motif heads preserve target exclusion, held-out rows and feature identity."""

import numpy as np
import pandas as pd
import pytest

from exp586_evals.classification_data import visible_digest
from exp586_evals.motif_data import motif_code, ordered_visible_digest
from exp586_evals.motif_probe import fit_head, fit_task
from exp586_evals.motif_runner import assemble_features


def test_fixed_motif_alphabet_and_ambiguous_outcomes():
    assert motif_code("AAA") == 0
    assert motif_code("TTT") == 63
    assert motif_code("atg") == 14
    assert motif_code("AN") == -1


def test_exact_directional_identity_is_stricter_than_duplicate_identity():
    spec = {"kind": "core", "positions": [4095, 4096]}
    left = {"sequence": "A" * 4095 + "GT" + "G" * 4095}
    right = {"sequence": "C" * 4095 + "AC" + "T" * 4095}
    assert visible_digest(spec, left) == visible_digest(spec, right)
    assert ordered_visible_digest(spec, left) != ordered_visible_digest(spec, right)
    changed_target = {"sequence": "A" * 4095 + "AA" + "G" * 4095}
    assert ordered_visible_digest(spec, left) == ordered_visible_digest(
        spec, changed_target
    )


def test_merge_keeps_both_row_and_strand_order_and_rejects_missing_inputs():
    membership = pd.DataFrame(
        {
            "split": ["train", "test", "invalid_target", "train"],
            "feature_origin": ["cached_core", "new_motif", "new_motif", "cached_core"],
            "feature_row_id": [1, 0, -1, 0],
        }
    )
    cached = {
        "row_id": np.arange(2),
        "fwd": np.full((2, 768), 4.0),
        "rc": np.full((2, 768), 9.0),
    }
    cached["fwd"][1] = 3
    missing = {
        "row_id": np.arange(1),
        "fwd": np.full((1, 768), 5.0),
        "rc": np.full((1, 768), 7.0),
    }
    fwd, rc = assemble_features(membership, cached, missing)
    np.testing.assert_array_equal(fwd[:, 0], [3, 5, 0, 4])
    np.testing.assert_array_equal(rc[:, 0], [9, 7, 0, 9])
    with pytest.raises(ValueError, match="Missing required"):
        assemble_features(membership, cached, None)
    cached["row_id"] = np.array([1, 0])
    with pytest.raises(AssertionError):
        assemble_features(membership, cached, missing)


def test_whole_motif_head_retains_unseen_test_outcomes_and_train_only_scaling(tmp_path):
    rng = np.random.default_rng(58)
    labels = np.tile([0, 14, 63], 24)
    features = rng.normal(size=(72, 8)).astype(np.float32)
    features[:, 0] += labels / 10
    chromosomes = np.repeat(np.arange(1, 7), 12)
    model, development = fit_head(
        features,
        labels,
        chromosomes,
        tmp_path / "head",
        jobs=1,
        regularization=(0.001, 0.1),
    )
    np.testing.assert_allclose(
        model.named_steps["scaler"].mean_, features.mean(axis=0), atol=1e-6
    )
    assert set(development["development_chromosomes"]) == {"1", "2", "3", "4", "5", "6"}
    test_features = rng.normal(size=(12, 8)).astype(np.float32) + 100
    x = np.concatenate([features, test_features])
    y = np.concatenate([labels, np.full(12, 32)])
    membership = pd.DataFrame(
        {
            "row_id": np.arange(84),
            "label": y,
            "target_motif": ["AAA"] * 84,
            "chrom": np.concatenate([chromosomes.astype(str), np.repeat("7", 12)]),
            "split": ["train"] * 72 + ["test"] * 12,
            "visible_sha256": [f"visible-{index}" for index in range(84)],
        }
    )
    result = fit_task(membership, x[:, :4], x[:, 4:], tmp_path / "task", jobs=1)
    assert result["test_rows"] == 12 and result["test_score"] == 0
    predictions = pd.read_parquet(tmp_path / "task/test-predictions.parquet")
    assert predictions.label.eq(32).all() and len(predictions) == 12
    assert result["serialized_predictions_exact"]
