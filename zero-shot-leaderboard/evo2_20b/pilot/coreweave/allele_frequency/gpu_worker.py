"""Persistent one-GPU maize-AF worker."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import flash_attn
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from pyfaidx import Fasta
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
COREWEAVE = HERE.parent
sys.path[:0] = [str(HERE), str(COREWEAVE)]

from common import digest, storage, upload_file, upload_json, write_json
from config import CONTEXT_LENGTHS, SAMPLE_SIZES, WINDOW_SIZE
from scoring import profile_scoring, score_rows, validate_center_crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    preparation = json.loads((args.root / "preparation.json").read_text())
    plan = json.loads((args.root / "plan.json").read_text())
    chunks = [chunk for chunk in plan if chunk["worker"] == args.worker]
    if not chunks:
        raise RuntimeError("Worker has no assigned chunks")
    method = os.environ.get("PLANTCAD_AF_METHOD", "cached")
    if method not in ("cached", "uncached"):
        raise RuntimeError(f"Unknown scorer: {method}")
    batch_size = int(os.environ.get("PLANTCAD_AF_BATCH_SIZE", "32"))
    window_size = int(os.environ.get("PLANTCAD_AF_WINDOW_SIZE", str(WINDOW_SIZE)))
    if window_size not in CONTEXT_LENGTHS:
        raise RuntimeError(f"Unsupported context length: {window_size}")
    only_sample_size = int(os.environ.get("PLANTCAD_AF_ONLY_SAMPLE", "0"))
    if only_sample_size not in (0, *SAMPLE_SIZES):
        raise RuntimeError(f"Unsupported sample size: {only_sample_size}")
    output = args.root / "workers" / f"gpu{args.worker:03d}"
    output.mkdir(parents=True, exist_ok=True)
    s3 = storage()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    np.random.seed(0)

    loaded = time.time()
    model = AutoModelForCausalLM.from_pretrained(preparation["model"], trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval().to("cuda:0")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(preparation["model"], trust_remote_code=True)
    fasta = Fasta(preparation["genome"]["path"], as_raw=True)
    union = pd.read_parquet(preparation["union"])
    if only_sample_size:
        union = union.loc[union[f"in_sample_{only_sample_size}"]].reset_index(drop=True)
    torch.cuda.synchronize()
    load_seconds = time.time() - loaded
    crop_validation = validate_center_crop(fasta, union.iloc[chunks[0]["start"]], window_size)
    profile = profile_scoring(model, tokenizer, fasta, union.iloc[chunks[0]["start"] : chunks[0]["start"] + 1], window_size=window_size, method=method)
    if profile["resolved_attention"] != "flash_attention_2" or not profile["external_flash_attention_2_verified"]:
        raise RuntimeError(f"External FlashAttention-2 was not verified: {profile}")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(model.dtype),
        "attention": model.config._attn_implementation,
        "tf32": True,
        "batch_size": batch_size,
        "window_size": window_size,
        "only_sample_size": only_sample_size or None,
        "method": method,
        "unused_cache_disabled": model.config.use_cache is False,
    }
    started = time.time()
    metadata = {"worker": args.worker, "environment": environment, "crop_validation": crop_validation, "flash_verification": profile, "load_seconds": load_seconds, "started_epoch": started, "planned_chunks": len(chunks), "planned_rows": sum(chunk["stop"] - chunk["start"] for chunk in chunks), "sampling_manifest_sha256": digest(Path(preparation["samples"]) / "sampling-manifest.json"), "model_sha256": preparation["model_files"]["model.safetensors"]["sha256"], "scoring_code_sha256": digest(HERE / "scoring.py")}
    write_json(output / "environment.json", metadata)
    upload_file(s3, output / "environment.json", f"{args.prefix}/workers/gpu{args.worker:03d}/environment.json")
    completed_rows = 0
    for index, chunk in enumerate(chunks):
        frame = union.iloc[chunk["start"] : chunk["stop"]].copy()
        key = f"maize-af--{chunk['start']:06d}-{chunk['stop']:06d}"
        chunk_started = time.time()

        def progress(strand: str, completed: int, total: int) -> None:
            record = {"worker": args.worker, "chunk": key, "completed_chunks": index, "planned_chunks": len(chunks), "completed_rows": completed_rows, "strand": strand, "strand_completed": completed, "strand_total": total, "elapsed_seconds": time.time() - started}
            write_json(output / "progress.json", record)
            upload_json(s3, f"{args.prefix}/workers/gpu{args.worker:03d}/progress.json", record)
            print(json.dumps(record), flush=True)

        values, timing = score_rows(model, tokenizer, fasta, frame, window_size=window_size, batch_size=batch_size, method=method, progress=progress)
        archive = {
            "row_id": frame["row_id"].to_numpy(dtype=np.int64),
            "llr_acgt_fwd": values["fwd"][:, 0],
            "llr_acgt_rc": values["rc"][:, 0],
            "llr_full_vocab_fwd": values["fwd"][:, 1],
            "llr_full_vocab_rc": values["rc"][:, 1],
            "non_acgt_targets_fwd": values["fwd"][:, 2].astype(np.int32),
            "non_acgt_targets_rc": values["rc"][:, 2].astype(np.int32),
        }
        if any(not np.isfinite(value).all() for value in archive.values()):
            raise RuntimeError(f"Non-finite prediction in {key}")
        array_path = output / f"{key}.npz"
        np.savez_compressed(array_path, **archive)
        record = {**chunk, "key": key, "timing": timing, "array_sha256": digest(array_path), "started_epoch": chunk_started, "finished_epoch": time.time()}
        write_json(output / f"{key}.json", record)
        upload_file(s3, array_path, f"{args.prefix}/chunks/{key}.npz")
        upload_file(s3, output / f"{key}.json", f"{args.prefix}/chunks/{key}.json")
        completed_rows += len(frame)
        print(json.dumps({"worker": args.worker, "completed_chunk": key, "completed_chunks": index + 1, "total_chunks": len(chunks)}), flush=True)
    final = {**metadata, "finished_epoch": time.time(), "run_wall_seconds": time.time() - started, "completed_chunks": len(chunks), "completed_rows": completed_rows}
    write_json(output / "done.json", final)
    upload_file(s3, output / "done.json", f"{args.prefix}/workers/gpu{args.worker:03d}/done.json")


if __name__ == "__main__":
    main()
