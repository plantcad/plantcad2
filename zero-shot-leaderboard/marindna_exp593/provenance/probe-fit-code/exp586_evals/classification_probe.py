"""Fit frozen linear classifiers with chromosome-separated development folds."""

import argparse
import hashlib
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp586_evals.probe_protocol import linear_features

REGULARIZATION = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)


class CheckedLogisticRegression(LogisticRegression):
    """Reject non-convergence in each CV child as well as the final refit."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            result = super().fit(X, y, sample_weight=sample_weight)
        if int(self.n_iter_.max()) >= self.max_iter:
            raise ValueError("Classifier optimizer exhausted its iteration budget")
        return result


def metric(labels: np.ndarray, scores: np.ndarray, kind: str) -> float:
    """Retain each benchmark's existing ranking metric."""
    return float(
        (average_precision_score if kind == "sv" else roc_auc_score)(labels, scores)
    )


def fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    chromosomes: np.ndarray,
    *,
    kind: str,
    jobs: int = 3,
) -> tuple[Pipeline, dict[str, Any]]:
    """This API receives training rows only, including for feature scaling."""
    if not np.isfinite(features).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected finite features and two outcome classes")
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=586)
    folds = list(splitter.split(features, labels, chromosomes))
    for train, validation in folds:
        if set(chromosomes[train]) & set(chromosomes[validation]):
            raise ValueError("An inner validation chromosome appears in training")
        if any(len(np.unique(labels[index])) != 2 for index in (train, validation)):
            raise ValueError("Both classes are required in every development fold")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                CheckedLogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-5),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"classifier__C": list(REGULARIZATION)},
        scoring="average_precision" if kind == "sv" else "roc_auc",
        cv=folds,
        n_jobs=jobs,
        pre_dispatch=jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(features, labels)
    model = search.best_estimator_
    if int(model.named_steps["classifier"].n_iter_.max()) >= 2000:
        raise ValueError("Classifier optimizer exhausted its iteration budget")
    return model, {
        "C": float(search.best_params_["classifier__C"]),
        "development_score": float(search.best_score_),
        "C_grid": list(REGULARIZATION),
        "development_scores": search.cv_results_["mean_test_score"].tolist(),
        "development_chromosomes": sorted(set(map(str, chromosomes))),
        "features": features.shape[1],
        "training_rows": len(labels),
        "iterations": model.named_steps["classifier"].n_iter_.tolist(),
    }


def fit_task(
    spec: dict[str, Any],
    membership: Path,
    arrays: dict[str, np.ndarray],
    output: Path,
    *,
    jobs: int = 3,
    checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Fit both prespecified readouts; never select between them on test results."""
    frame = pd.read_parquet(membership).sort_values("row_id").reset_index(drop=True)
    if not np.array_equal(frame.row_id, arrays["row_id"]):
        raise ValueError("Frozen feature rows do not match the saved membership")
    train, test = frame.split.eq("train").to_numpy(), frame.split.eq("test").to_numpy()
    if set(frame.loc[train, "chrom"]) & set(frame.loc[test, "chrom"]):
        raise ValueError("The final holdout shares a training chromosome")
    if set(frame.loc[train, "visible_sha256"]) & set(frame.loc[test, "visible_sha256"]):
        raise ValueError("The final holdout repeats visible training inputs")
    labels = frame.label.to_numpy(dtype=np.int64)
    if any(len(np.unique(labels[index])) != 2 for index in (train, test)):
        raise ValueError("Both classes are required in train and final test")
    output.mkdir(parents=True, exist_ok=True)
    with membership.open("rb") as stream:
        membership_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    binding = {"task": spec, "membership_sha256": membership_digest, "features": {}}
    for name, values in sorted(arrays.items()):
        values = np.ascontiguousarray(values)
        binding["features"][name] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
        }
    binding_path = output / "source-binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Existing probe fits belong to different source data")
    binding_path.write_text(json.dumps(binding, sort_keys=True, indent=2) + "\n")
    predictions = frame[["row_id", "label", "chrom", "split"]].copy()
    readouts = {}
    for directional in (False, True):
        name = "directional" if directional else "strand_average"
        features = linear_features(arrays, kind=spec["kind"], directional=directional)
        path = output / (name + ".joblib")
        development_path = output / (name + "-development.json")
        if path.exists() and development_path.exists():
            model = joblib.load(path)
            development = json.loads(development_path.read_text())
        else:
            model, development = fit_head(
                features[train],
                labels[train],
                frame.loc[train, "chrom"].to_numpy(),
                kind=spec["kind"],
                jobs=jobs,
            )
            joblib.dump(model, path, compress=3)
            development_path.write_text(json.dumps(development, indent=2) + "\n")
            if checkpoint is not None:
                checkpoint()
        score = model.decision_function(features[test])
        restored = joblib.load(path)
        np.testing.assert_allclose(
            restored.decision_function(features[test]), score, rtol=0, atol=1e-10
        )
        predictions.loc[test, name] = score
        readouts[name] = {
            "development": development,
            "test_score": metric(labels[test], score, spec["kind"]),
            "test_rows": int(test.sum()),
            "test_positive_fraction": float(labels[test].mean()),
        }
        print(
            json.dumps({"task": spec["key"], "readout": name, **readouts[name]}),
            flush=True,
        )
        if not directional:
            # A negative control using the same representation and penalty.
            shuffled = np.random.default_rng(586).permutation(labels[train])
            control = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        CheckedLogisticRegression(
                            C=development["C"], solver="lbfgs", max_iter=2000, tol=1e-5
                        ),
                    ),
                ]
            )
            control_path = output / "shuffled-labels.joblib"
            if control_path.exists():
                control = joblib.load(control_path)
            else:
                control.fit(features[train], shuffled)
                joblib.dump(control, control_path, compress=3)
                if checkpoint is not None:
                    checkpoint()
            control_score = control.decision_function(features[test])
            predictions.loc[test, "shuffled_labels"] = control_score
            readouts["shuffled_labels"] = {
                "test_score": metric(labels[test], control_score, spec["kind"])
            }
        del features
    if "deletion_length" in frame:
        features = np.log1p(frame.deletion_length.to_numpy(dtype=np.float64))[:, None]
        model, development = fit_head(
            features[train],
            labels[train],
            frame.loc[train, "chrom"].to_numpy(),
            kind=spec["kind"],
            jobs=jobs,
        )
        score = model.decision_function(features[test])
        predictions.loc[test, "deletion_length_only"] = score
        readouts["deletion_length_only"] = {
            "development": development,
            "test_score": metric(labels[test], score, spec["kind"]),
        }
    predictions.loc[test].to_parquet(output / "test-predictions.parquet", index=False)
    result = {
        "task": spec["key"],
        "kind": spec["kind"],
        "metric": "auprc" if spec["kind"] == "sv" else "auroc",
        "membership": str(membership),
        "readouts": readouts,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    spec = next(spec for spec in manifest["tasks"] if spec["key"] == args.task)
    arrays = {
        path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
        for path in args.features.glob("*.npy")
    }
    fit_task(spec, args.membership, arrays, args.output, jobs=args.jobs)


if __name__ == "__main__":
    main()
