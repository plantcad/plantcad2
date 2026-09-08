"""Fit ridge probes and compute paired genomic-block bootstrap intervals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from allele_frequency_probe.config import BOOTSTRAP_REPLICATES, INNER_FOLDS, PRIMARY_REPRESENTATION, REPRESENTATIONS, RIDGE_ALPHAS, SPLIT_SEED
from allele_frequency_probe.split import build_split, inner_splits


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    return float(spearmanr(y, score).statistic)


def _scorer(estimator: Any, x: np.ndarray, y: np.ndarray) -> float:
    return _spearman(y, estimator.predict(x))


def _feature(ref: np.ndarray, alt: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float32)
    alt = np.asarray(alt, dtype=np.float32)
    return np.concatenate([ref, alt - ref], axis=1)


def _fit_one(x: np.ndarray, y: np.ndarray, cv: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Pipeline, dict]:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(solver="lsqr", tol=1e-4, max_iter=2_000)),
    ])
    search = GridSearchCV(pipeline, {"ridge__alpha": RIDGE_ALPHAS}, scoring=_scorer, cv=cv, n_jobs=5, refit=True, return_train_score=False, verbose=1)
    search.fit(x, y)
    means = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    alphas = np.asarray(search.cv_results_["param_ridge__alpha"], dtype=float)
    order = np.argsort(alphas)
    alphas, means = alphas[order], means[order]
    best_alpha = float(search.best_params_["ridge__alpha"])
    at_low, at_high = best_alpha == alphas[0], best_alpha == alphas[-1]
    truncation = bool((at_low and means[0] > means[1] + 0.002) or (at_high and means[-1] > means[-2] + 0.002))
    return search.best_estimator_, {
        "best_alpha": best_alpha,
        "best_cv_spearman": float(search.best_score_),
        "alpha_grid": alphas.tolist(),
        "mean_cv_spearman": means.tolist(),
        "at_grid_edge": bool(at_low or at_high),
        "truncation_risk": truncation,
    }


def _bootstrap(frame: pd.DataFrame, score: str, baseline: str = "zero_shot_llr") -> dict[str, Any]:
    y = frame["AF"].to_numpy(dtype=float)
    probe = frame[score].to_numpy(dtype=float)
    zero = frame[baseline].to_numpy(dtype=float)
    blocks = frame["block_id"].astype(str).to_numpy()
    unique = np.unique(blocks)
    lookup = {block: np.flatnonzero(blocks == block) for block in unique}
    rng = np.random.default_rng(SPLIT_SEED)
    draws = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([lookup[block] for block in selected])
        draws[index, 0] = _spearman(y[rows], probe[rows])
        draws[index, 1] = _spearman(y[rows], zero[rows])
        draws[index, 2] = draws[index, 0] - draws[index, 1]
    point_probe, point_zero = _spearman(y, probe), _spearman(y, zero)
    names = ("probe", "zero_shot", "delta")
    points = (point_probe, point_zero, point_probe - point_zero)
    return {
        name: {
            "value": float(points[column]),
            "ci95": np.nanquantile(draws[:, column], [0.025, 0.975]).tolist(),
        }
        for column, name in enumerate(names)
    }


def fit_probes(
    frame: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    zero_shot: pd.DataFrame,
    output: Path,
) -> dict[str, Any]:
    split, split_manifest = build_split(frame)
    data = frame.merge(split, on="row_id", validate="1:1").merge(zero_shot[["row_id", "llr_acgt_avg"]], on="row_id", validate="1:1")
    data = data.rename(columns={"llr_acgt_avg": "zero_shot_llr"})
    if not np.array_equal(data["row_id"].to_numpy(), frame["row_id"].to_numpy()):
        raise RuntimeError("Probe joins changed embedding row order")
    if data["zero_shot_llr"].isna().any():
        raise RuntimeError("Zero-shot predictions do not cover the probe sample")
    train = data["split"].eq("train").to_numpy()
    test = data["split"].eq("test").to_numpy()
    cv_global = inner_splits(frame, split, INNER_FOLDS)
    cv = [(np.searchsorted(np.flatnonzero(train), tr), np.searchsorted(np.flatnonzero(train), val)) for tr, val in cv_global]
    y_train = data.loc[train, "AF"].to_numpy(dtype=np.float32)

    fit_metadata: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        print(json.dumps({"phase": "probe_fit_started", "representation": representation, "rows": int(train.sum()), "features": int(embeddings[f"{representation}_ref"].shape[1] * 2)}), flush=True)
        fit_started = time.monotonic()
        x = _feature(embeddings[f"{representation}_ref"], embeddings[f"{representation}_alt"])
        estimator, metadata = _fit_one(x[train], y_train, cv)
        prediction = estimator.predict(x)
        data[f"probe_{representation}"] = prediction
        joblib.dump(estimator, output / f"probe-{representation}.joblib", compress=3)
        fit_seconds = time.monotonic() - fit_started
        fit_metadata[representation] = {**metadata, "feature": "[emb_ref, emb_alt - emb_ref]", "train_rows": int(train.sum()), "test_rows": int(test.sum()), "fit_seconds": fit_seconds}
        print(json.dumps({"phase": "probe_fit_complete", "representation": representation, "best_alpha": metadata["best_alpha"], "best_cv_spearman": metadata["best_cv_spearman"], "truncation_risk": metadata["truncation_risk"], "fit_seconds": fit_seconds}), flush=True)
        del x

    test_frame = data.loc[test].copy()
    metrics: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        score = f"probe_{representation}"
        metrics[representation] = {
            "overall": _bootstrap(test_frame, score),
            "by_consequence": {str(name): _bootstrap(group, score) for name, group in test_frame.groupby("consequence", sort=True)},
        }
    prediction_columns = [
        "row_id", "chrom", "pos", "ref", "alt", "AF", "consequence", "original_consequence",
        "block_id", "af_bin", "outer_fold", "split", "zero_shot_llr",
        *[f"probe_{name}" for name in REPRESENTATIONS],
    ]
    prediction_columns = [name for name in prediction_columns if name in data.columns]
    data[prediction_columns].to_parquet(output / "predictions.parquet", index=False)
    split.to_parquet(output / "split.parquet", index=False)
    result = {
        "primary_representation": PRIMARY_REPRESENTATION,
        "target": "raw alternate-allele frequency (AF)",
        "probe": "StandardScaler -> Ridge; alpha selected by train-only grouped CV Spearman",
        "split": split_manifest,
        "fits": fit_metadata,
        "metrics": metrics,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
