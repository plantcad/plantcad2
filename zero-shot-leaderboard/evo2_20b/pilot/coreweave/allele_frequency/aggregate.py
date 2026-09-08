"""Merge row-level LLRs and compute AF correlations for every retained sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import EXCLUDED_VARIANTS, HISTORICAL_10K, SAMPLE_SIZES


PREDICTION_COLUMNS = (
    "llr_acgt_fwd",
    "llr_acgt_rc",
    "llr_full_vocab_fwd",
    "llr_full_vocab_rc",
    "non_acgt_targets_fwd",
    "non_acgt_targets_rc",
)


def correlations(frame: pd.DataFrame, score: str) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(frame["AF"], frame[score]).statistic),
        "spearman": float(spearmanr(frame["AF"], frame[score]).statistic),
    }


def aggregate(artifact: Path, union_path: Path, plan: list[dict], chunks: list[tuple[dict, Path]], only_sample_size: int | None = None) -> dict:
    expected = sorted((row["start"], row["stop"], row["worker"]) for row in plan)
    observed = sorted((row["start"], row["stop"], row["worker"]) for row, _ in chunks)
    if expected != observed:
        raise RuntimeError("Completed chunk coverage differs from the execution plan")
    arrays = []
    for metadata, path in sorted(chunks, key=lambda item: item[0]["start"]):
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        rows = metadata["stop"] - metadata["start"]
        if set(values) != {"row_id", *PREDICTION_COLUMNS} or any(len(value) != rows for value in values.values()):
            raise RuntimeError(f"Malformed prediction chunk: {path.name}")
        arrays.append(values)
    merged = {name: np.concatenate([chunk[name] for chunk in arrays]) for name in arrays[0]}
    union = pd.read_parquet(union_path).sort_values("row_id").reset_index(drop=True)
    if only_sample_size is not None:
        union = union.loc[union[f"in_sample_{only_sample_size}"]].reset_index(drop=True)
    if not np.array_equal(merged["row_id"], union["row_id"].to_numpy()):
        raise RuntimeError("Prediction row IDs do not align with the sampled union")
    for name in PREDICTION_COLUMNS:
        union[name] = merged[name]
    union["llr_acgt_avg"] = (union["llr_acgt_fwd"] + union["llr_acgt_rc"]) / 2
    union["llr_full_vocab_avg"] = (union["llr_full_vocab_fwd"] + union["llr_full_vocab_rc"]) / 2
    union["llr_acgt_minus_full_vocab_avg"] = union["llr_acgt_avg"] - union["llr_full_vocab_avg"]

    metrics = {}
    for requested_n in (only_sample_size,) if only_sample_size is not None else SAMPLE_SIZES:
        sample = union.loc[union[f"in_sample_{requested_n}"]].copy()
        metrics[str(requested_n)] = {
            "requested_rows": requested_n,
            "actual_rows": len(sample),
            "llr_acgt_avg": correlations(sample, "llr_acgt_avg"),
            "llr_full_vocab_avg": correlations(sample, "llr_full_vocab_avg"),
            "llr_acgt_fwd": correlations(sample, "llr_acgt_fwd"),
            "llr_acgt_rc": correlations(sample, "llr_acgt_rc"),
            "rows_with_non_acgt_targets": int(((sample["non_acgt_targets_fwd"] + sample["non_acgt_targets_rc"]) > 0).sum()),
            "max_abs_acgt_minus_full_vocab_llr": float(sample["llr_acgt_minus_full_vocab_avg"].abs().max()),
        }
    predictions = artifact / "predictions.parquet"
    union.to_parquet(predictions, index=False)
    result = {
        "primary_score": "llr_acgt_avg",
        "primary_metric": "direct Pearson/Spearman correlation of AF with (LLR_fwd + LLR_rc) / 2",
        "llr_orientation": "log P(ALT sequence) - log P(REF sequence)",
        "metrics": metrics,
        "historical_10k": HISTORICAL_10K,
        "exclusions": list(EXCLUDED_VARIANTS),
        "predictions": {"path": predictions.name, "rows": len(union), "columns": list(union.columns)},
    }
    (artifact / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
