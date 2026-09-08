"""Profiler-verify FA2 and compare cached with uncached full-sequence LLRs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
COREWEAVE = HERE.parent
sys.path[:0] = [str(HERE), str(COREWEAVE)]

from common import storage, upload_json
from config import CONTEXT_LENGTHS, SAMPLE_SIZES, WINDOW_SIZE
from scoring import profile_scoring, score_rows, validate_center_crop


def main(root: Path, prefix: str, rows: int = 128) -> dict:
    preparation = json.loads((root / "preparation.json").read_text())
    frame = pd.read_parquet(preparation["union"])
    only_sample_size = int(os.environ.get("PLANTCAD_AF_ONLY_SAMPLE", "0"))
    if only_sample_size not in (0, *SAMPLE_SIZES):
        raise RuntimeError(f"Unsupported sample size: {only_sample_size}")
    if only_sample_size:
        frame = frame.loc[frame[f"in_sample_{only_sample_size}"]]
    frame = frame.iloc[:rows].copy()
    model = AutoModelForCausalLM.from_pretrained(preparation["model"], trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval().to("cuda:0")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(preparation["model"], trust_remote_code=True)
    fasta = Fasta(preparation["genome"]["path"], as_raw=True)
    window_size = int(os.environ.get("PLANTCAD_AF_WINDOW_SIZE", str(WINDOW_SIZE)))
    if window_size not in CONTEXT_LENGTHS:
        raise RuntimeError(f"Unsupported context length: {window_size}")
    crop_validation = validate_center_crop(fasta, frame.iloc[0], window_size)
    profiles = {method: profile_scoring(model, tokenizer, fasta, frame.iloc[:1], window_size=window_size, method=method) for method in ("cached", "uncached")}
    if any(profile["resolved_attention"] != "flash_attention_2" or not profile["external_flash_attention_2_verified"] for profile in profiles.values()):
        raise RuntimeError(f"External FA2 was not verified in both scoring paths: {profiles}")
    values = {}
    timings = {}
    for method in ("cached", "uncached"):
        values[method], timings[method] = score_rows(model, tokenizer, fasta, frame, window_size=window_size, batch_size=32, method=method)
    comparisons = {}
    for strand in ("fwd", "rc"):
        comparisons[strand] = {
            "max_abs_acgt_llr_delta": float(np.max(np.abs(values["cached"][strand][:, 0] - values["uncached"][strand][:, 0]))),
            "max_abs_full_vocab_llr_delta": float(np.max(np.abs(values["cached"][strand][:, 1] - values["uncached"][strand][:, 1]))),
            "unknown_counts_exact": bool(np.array_equal(values["cached"][strand][:, 2], values["uncached"][strand][:, 2])),
        }
    for method in values:
        averaged = (values[method]["fwd"][:, 0] + values[method]["rc"][:, 0]) / 2
        comparisons[method] = {"pearson": float(pearsonr(frame["AF"], averaged).statistic), "spearman": float(spearmanr(frame["AF"], averaged).statistic)}
    result = {
        "rows": len(frame),
        "window_size": window_size,
        "only_sample_size": only_sample_size or None,
        "crop_validation": crop_validation,
        "profiles": profiles,
        "timings": timings,
        "cached_speedup": timings["uncached"]["wall_seconds"] / timings["cached"]["wall_seconds"],
        "comparisons": comparisons,
        "production_recommendation": "cached" if timings["cached"]["wall_seconds"] < timings["uncached"]["wall_seconds"] else "uncached",
        "note": "The prefix call alone enables cache; model.config.use_cache remains false and suffix/uncached calls explicitly disable it.",
    }
    (root / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    upload_json(storage(), f"{prefix}/smoke/result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
