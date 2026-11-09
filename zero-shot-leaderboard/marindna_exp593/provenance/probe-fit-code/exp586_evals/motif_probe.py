"""Frozen whole-motif multinomial probes with chromosome-held-out selection."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp586_evals.classification_probe import REGULARIZATION, CheckedLogisticRegression


def estimator(penalty: float) -> Pipeline:
    """Use the existing standardized L2 logistic convention for whole-motif labels."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                CheckedLogisticRegression(
                    C=penalty,
                    solver="lbfgs",
                    max_iter=10000,
                    tol=1e-5,
                ),
            ),
        ]
    )


def fold_score(
    features: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    penalty: float,
) -> float:
    """Fit preprocessing and classifier using the development training fold only."""
    model = estimator(penalty).fit(features[fit], labels[fit])
    return float(
        accuracy_score(labels[validation], model.predict(features[validation]))
    )


def fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    chromosomes: np.ndarray,
    output: Path,
    *,
    jobs: int = 3,
    progress: Callable[[dict[str, Any]], None] | None = None,
    regularization: tuple[float, ...] = REGULARIZATION,
) -> tuple[Pipeline, dict[str, Any]]:
    """Receive development rows only; select C by whole-motif accuracy."""
    if (
        not np.isfinite(features).all()
        or np.any(labels < 0)
        or len(np.unique(labels)) < 2
    ):
        raise ValueError(
            "Expected finite features and at least two valid motif outcomes"
        )
    folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=586).split(
            features,
            labels,
            chromosomes,
        )
    )
    for fit, validation in folds:
        if set(chromosomes[fit]) & set(chromosomes[validation]):
            raise ValueError("A development chromosome crosses a CV fold")
        if len(np.unique(labels[fit])) < 2:
            raise ValueError("A CV training fold has only one motif outcome")
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "development.json"
    history = (
        json.loads(history_path.read_text())
        if history_path.exists()
        else {
            "C_grid": list(regularization),
            "fold_scores": [],
            "development_scores": [],
        }
    )
    if history["C_grid"] != list(regularization):
        raise ValueError("A resumed head uses a different regularization grid")
    for index, penalty in enumerate(regularization):
        if index >= len(history["fold_scores"]):
            scores = Parallel(n_jobs=jobs, pre_dispatch=jobs)(
                delayed(fold_score)(features, labels, fit, validation, penalty)
                for fit, validation in folds
            )
            history["fold_scores"].append(scores)
            history["development_scores"].append(float(np.mean(scores)))
            history_path.write_text(json.dumps(history, indent=2) + "\n")
        if progress:
            progress(
                {
                    "phase": "cross-validation",
                    "C_index": index + 1,
                    "C_total": len(regularization),
                    "development_accuracy": history["development_scores"][index],
                }
            )
    selected = int(np.argmax(history["development_scores"]))
    penalty = float(regularization[selected])
    model_path = output / "directional.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        if progress:
            progress({"phase": "refitting", "C": penalty})
        model = estimator(penalty).fit(features, labels)
        joblib.dump(model, model_path, compress=3)
    development = {
        **history,
        "C": penalty,
        "development_score": history["development_scores"][selected],
        "development_chromosomes": sorted(set(map(str, chromosomes))),
        "training_rows": len(labels),
        "features": features.shape[1],
        "classes": model.classes_.tolist(),
        "iterations": model.named_steps["classifier"].n_iter_.tolist(),
        "max_iterations": int(model.named_steps["classifier"].max_iter),
    }
    (output / "directional-development.json").write_text(
        json.dumps(development, indent=2) + "\n"
    )
    return model, development


def fit_task(
    membership: pd.DataFrame,
    fwd: np.ndarray,
    rc: np.ndarray,
    output: Path,
    *,
    jobs: int = 3,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Keep test labels out of fitting; persist an independently checked exact match."""
    train = membership.split.eq("train").to_numpy()
    test = membership.split.eq("test").to_numpy()
    if set(membership.loc[train, "chrom"]) & set(membership.loc[test, "chrom"]):
        raise ValueError("Training/test chromosomes overlap")
    if set(membership.loc[train, "visible_sha256"]) & set(
        membership.loc[test, "visible_sha256"]
    ):
        raise ValueError("Identical visible inputs cross the test boundary")
    if fwd.shape != rc.shape or fwd.shape[0] != len(membership):
        raise ValueError("Flank activations do not match motif membership")
    features = np.concatenate([fwd, rc], axis=1)
    labels = membership.label.to_numpy(dtype=np.int64)
    model, development = fit_head(
        features[train],
        labels[train],
        membership.loc[train, "chrom"].to_numpy(),
        output,
        jobs=jobs,
        progress=progress,
    )
    predicted = model.predict(features[test])
    restored = joblib.load(output / "directional.joblib")
    np.testing.assert_array_equal(restored.predict(features[test]), predicted)
    np.testing.assert_allclose(
        restored.decision_function(features[test]),
        model.decision_function(features[test]),
        rtol=0,
        atol=1e-10,
    )
    values, counts = np.unique(labels[train], return_counts=True)
    majority = int(values[np.argmax(counts)])
    # This control uses the selected development penalty, never test feedback.
    control_path = output / "shuffled-labels.joblib"
    if control_path.exists():
        control = joblib.load(control_path)
    else:
        if progress:
            progress({"phase": "shuffled-control"})
        shuffled = np.random.default_rng(586).permutation(labels[train])
        control = estimator(development["C"]).fit(features[train], shuffled)
        joblib.dump(control, control_path, compress=3)
    shuffled_predictions = control.predict(features[test])
    result = {
        "metric": "whole_motif_exact_match_accuracy",
        "readout": "directional",
        "feature_contract": "[last(forward left flank), last(RC right flank)]",
        "development": development,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "test_score": float(np.mean(predicted == labels[test])),
        "majority_motif_label_from_training": majority,
        "majority_baseline": float(np.mean(labels[test] == majority)),
        "shuffled_labels": float(np.mean(shuffled_predictions == labels[test])),
        "train_outcome_counts": {
            str(v): int(n) for v, n in zip(values, counts, strict=True)
        },
        "test_outcome_counts": {
            str(v): int(n)
            for v, n in zip(*np.unique(labels[test], return_counts=True), strict=True)
        },
        "serialized_predictions_exact": True,
    }
    assert result["test_score"] == float(accuracy_score(labels[test], predicted))
    predictions = membership.loc[
        test, ["row_id", "target_motif", "label", "chrom"]
    ].copy()
    predictions["predicted_label"] = predicted
    predictions["majority_label"] = majority
    predictions["shuffled_label_prediction"] = shuffled_predictions
    predictions.to_parquet(output / "test-predictions.parquet", index=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
