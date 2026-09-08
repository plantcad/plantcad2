"""Pinned configuration for the supervised maize-AF linear-probe companion eval."""

from __future__ import annotations

import numpy as np


SPLIT_SEED = 0
TEST_FRACTION = 0.30
BLOCK_BP = 1_000_000
PURGE_BP = 8_192
OUTER_FOLDS = 10
TEST_FOLDS = 3
INNER_FOLDS = 5
AF_BINS = 10
BOOTSTRAP_REPLICATES = 1_000
RIDGE_ALPHAS = np.logspace(-4, 8, 13)
REPRESENTATIONS = ("whole_window", "variant_token")
PRIMARY_REPRESENTATION = "whole_window"

ZERO_SHOT_RESULTS = {
    "022t": "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2/results/step-206144/coreweave/plantcad2-maize-af-022t-20260904-v1/predictions.parquet",
    "039t": "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/results/step-371065/coreweave/plantcad2-maize-af-039t-20260904-v1/predictions.parquet",
    "056t": "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/results/step-535985/coreweave/plantcad2-maize-af-056t-20260904-v1/predictions.parquet",
}
