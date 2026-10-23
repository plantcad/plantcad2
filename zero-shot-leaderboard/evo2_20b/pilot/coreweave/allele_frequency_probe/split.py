"""Deterministic, consequence/AF-stratified genomic-block train/test split."""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from allele_frequency_probe.config import AF_BINS, BLOCK_BP, OUTER_FOLDS, PURGE_BP, SPLIT_SEED, TEST_FOLDS, TEST_FRACTION


def _digest(frame: pd.DataFrame) -> str:
    payload = frame[["row_id", "split", "block_id", "af_bin"]].sort_values("row_id").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _af_bins(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("consequence", sort=False)["AF"].rank(method="average", pct=True)
    return np.minimum((ranks * AF_BINS).astype(int), AF_BINS - 1)


def _purge_overlaps(frame: pd.DataFrame, test: np.ndarray) -> np.ndarray:
    purge = np.zeros(len(frame), dtype=bool)
    for chrom, chrom_frame in frame.groupby("chrom", sort=False):
        indices = chrom_frame.index.to_numpy()
        test_positions = np.sort(frame.loc[indices[test[indices]], "pos"].to_numpy(dtype=np.int64))
        if not len(test_positions):
            continue
        positions = frame.loc[indices, "pos"].to_numpy(dtype=np.int64)
        insertion = np.searchsorted(test_positions, positions)
        left = np.maximum(insertion - 1, 0)
        right = np.minimum(insertion, len(test_positions) - 1)
        distance = np.minimum(np.abs(positions - test_positions[left]), np.abs(positions - test_positions[right]))
        purge[indices] = (~test[indices]) & (distance < PURGE_BP)
    return purge


def build_split(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"row_id", "chrom", "pos", "AF", "consequence"}
    if not required.issubset(sample.columns) or sample["row_id"].duplicated().any():
        raise RuntimeError("Malformed AF sample for probe splitting")
    frame = sample.sort_values("row_id").reset_index(drop=True).copy()
    frame["chrom"] = frame["chrom"].astype(str)
    frame["block_id"] = frame["chrom"] + ":" + (frame["pos"].astype(np.int64) // BLOCK_BP).astype(str)
    frame["af_bin"] = _af_bins(frame)
    frame["stratum"] = frame["consequence"].astype(str) + "|" + frame["af_bin"].astype(str)

    splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold = np.full(len(frame), -1, dtype=np.int8)
    for index, (_, test_indices) in enumerate(splitter.split(np.zeros(len(frame)), frame["stratum"], frame["block_id"])):
        fold[test_indices] = index
    if (fold < 0).any():
        raise RuntimeError("Some blocks were not assigned an outer fold")
    frame["outer_fold"] = fold

    counts = pd.crosstab(frame["stratum"], frame["outer_fold"])
    target = counts.sum(axis=1).to_numpy(dtype=float) * TEST_FRACTION
    best: tuple[float, tuple[int, ...]] | None = None
    for choice in itertools.combinations(range(OUTER_FOLDS), TEST_FOLDS):
        observed = counts[list(choice)].sum(axis=1).to_numpy(dtype=float)
        score = float(np.mean(np.square(observed - target) / (target + 1)))
        candidate = (score, choice)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    selected = best[1]
    test = frame["outer_fold"].isin(selected).to_numpy()
    purge = _purge_overlaps(frame, test)
    frame["split"] = np.where(test, "test", np.where(purge, "dropped_overlap", "train"))

    train_blocks = set(frame.loc[frame["split"] == "train", "block_id"])
    test_blocks = set(frame.loc[frame["split"] == "test", "block_id"])
    if train_blocks & test_blocks:
        raise RuntimeError("Genomic blocks cross train/test")
    test_counts = frame.loc[frame["split"] == "test"].groupby("consequence").size()
    if test_counts.min() < 800:
        raise RuntimeError(f"Too few held-out examples in a consequence: {test_counts.to_dict()}")
    output = frame[["row_id", "block_id", "af_bin", "stratum", "outer_fold", "split"]].copy()
    manifest = {
        "seed": SPLIT_SEED,
        "method": "ten stratified genomic-block folds; best three-fold combination by consequence/within-consequence-AF-decile balance",
        "block_bp": BLOCK_BP,
        "purge_bp": PURGE_BP,
        "test_folds": list(selected),
        "rows": {name: int(value) for name, value in output["split"].value_counts().items()},
        "test_by_consequence": {str(name): int(value) for name, value in test_counts.items()},
        "sha256": _digest(output),
    }
    return output, manifest


def inner_splits(frame: pd.DataFrame, split: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    joined = frame[["row_id"]].merge(split, on="row_id", validate="1:1")
    train = joined["split"].eq("train").to_numpy()
    train_rows = np.flatnonzero(train)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SPLIT_SEED)
    return [
        (train_rows[tr], train_rows[val])
        for tr, val in splitter.split(np.zeros(train.sum()), joined.loc[train, "stratum"], joined.loc[train, "block_id"])
    ]
