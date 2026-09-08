#!/usr/bin/env python3
"""Validate row-level AF-probe outputs and bootstrap cross-checkpoint trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CHECKPOINTS = ("022t", "039t", "056t")
SCORES = {
    "zero_shot": "zero_shot_llr",
    "whole_window": "probe_whole_window",
    "variant_token": "probe_variant_token",
}
BOOTSTRAP_REPLICATES = 1_000
SEED = 0


def _spearman(y: np.ndarray, score: np.ndarray) -> float:
    return float(spearmanr(y, score).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=PREDICTIONS.PARQUET")
    parser.add_argument("--result", action="append", required=True, metavar="NAME=RESULT.JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction_paths = dict(item.split("=", 1) for item in args.prediction)
    result_paths = dict(item.split("=", 1) for item in args.result)
    if set(prediction_paths) != set(CHECKPOINTS) or set(result_paths) != set(CHECKPOINTS):
        raise ValueError(f"Expected inputs for {CHECKPOINTS}")
    frames = {name: pd.read_parquet(path).sort_values("row_id").reset_index(drop=True) for name, path in prediction_paths.items()}
    results = {name: json.loads(Path(path).read_text()) for name, path in result_paths.items()}
    first = frames[CHECKPOINTS[0]]
    identity = ["row_id", "chrom", "pos", "ref", "alt", "AF", "consequence", "block_id", "split"]
    for name, frame in frames.items():
        if len(frame) != 94_075 or frame["row_id"].duplicated().any() or not np.isfinite(frame[["AF", *SCORES.values()]].to_numpy()).all():
            raise ValueError(f"Malformed predictions for {name}")
        if not frame[identity].equals(first[identity]):
            raise ValueError(f"Input or split mismatch for {name}")
        if results[name]["split"]["sha256"] != results[CHECKPOINTS[0]]["split"]["sha256"]:
            raise ValueError(f"Split hash mismatch for {name}")

    test = first["split"].eq("test").to_numpy()
    if int(test.sum()) != 28_378:
        raise ValueError("Unexpected held-out row count")
    validation: dict[str, dict] = {}
    for name, frame in frames.items():
        held_out = frame.loc[test]
        observed = {score: _spearman(held_out["AF"].to_numpy(), held_out[column].to_numpy()) for score, column in SCORES.items()}
        expected = {
            "zero_shot": results[name]["metrics"]["whole_window"]["overall"]["zero_shot"]["value"],
            "whole_window": results[name]["metrics"]["whole_window"]["overall"]["probe"]["value"],
            "variant_token": results[name]["metrics"]["variant_token"]["overall"]["probe"]["value"],
        }
        if any(abs(observed[key] - expected[key]) > 1e-12 for key in SCORES):
            raise ValueError(f"Metric mismatch for {name}: {observed} != {expected}")
        validation[name] = observed

    end_to_end: dict[str, dict] = {}
    held_out = first.loc[test].copy()
    blocks = held_out["block_id"].astype(str).to_numpy()
    unique_blocks = np.unique(blocks)
    lookup = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(SEED)
    draws = {score: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for score in SCORES}
    low, high = frames[CHECKPOINTS[0]].loc[test], frames[CHECKPOINTS[-1]].loc[test]
    y = held_out["AF"].to_numpy()
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        rows = np.concatenate([lookup[block] for block in selected])
        for score, column in SCORES.items():
            draws[score][replicate] = _spearman(y[rows], high[column].to_numpy()[rows]) - _spearman(y[rows], low[column].to_numpy()[rows])
    for score in SCORES:
        point = validation[CHECKPOINTS[-1]][score] - validation[CHECKPOINTS[0]][score]
        end_to_end[score] = {"value": point, "ci95": np.quantile(draws[score], [0.025, 0.975]).tolist()}

    adjacent = {
        f"{left}_to_{right}": {score: validation[right][score] - validation[left][score] for score in SCORES}
        for left, right in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:], strict=True)
    }
    output = {
        "validation": validation,
        "adjacent_point_changes": adjacent,
        "022t_to_056t_paired_block_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "block": "chromosome:1Mb",
            "metrics": end_to_end,
        },
        "split_sha256": results[CHECKPOINTS[0]]["split"]["sha256"],
        "test_rows": int(test.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
