"""Persistent single-GPU worker calling the unchanged Lambda-tested scoring code."""

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import PILOT, digest, storage, upload_file, upload_json, write_json
from inputs import read_frame

sys.path.insert(0, str(PILOT / "sensitivity"))
import run_condition as scoring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    prep = json.loads((args.root / "preparation.json").read_text())
    plan = json.loads((args.root / "plan.json").read_text())
    chunks = [chunk for chunk in plan if chunk["worker"] == args.worker]
    if not chunks:
        raise RuntimeError("Worker has no assigned chunks")
    specs = {task["key"]: task for task in scoring.LEADERBOARD_TASKS}
    output = args.root / "workers" / f"gpu{args.worker:03d}"
    output.mkdir(parents=True, exist_ok=True)
    s3 = storage()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    np.random.seed(0)
    settings = argparse.Namespace(dtype="bf16", attention="flash_attention_2", batch_size=32, sv_batch_size=32, softmax_dtype="fp32", tf32=True, use_cache=False)
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(prep["model"], trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(prep["model"], trust_remote_code=True)
    model.config.use_cache = False
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.time() - load_started

    first = read_frame(prep, specs[chunks[0]["task"]], chunks[0]["start"], chunks[0]["start"] + 1)
    sequence = str(first["sequence"].iloc[0] if "sequence" in first else first["RefSeq"].iloc[0])
    profile = scoring._profile_attention(model, tokenizer, sequence, require_flash=True, use_cache=False)
    external = [key for key in profile["matching_kernel_events"] if "flash_attn" in key.lower() or "flash::" in key.lower()]
    if model.config._attn_implementation != "flash_attention_2" or not external:
        raise RuntimeError("External FlashAttention-2 was not verified")
    profile.update(external_flash_attention_2_verified=True, external_flash_attention_2_events=external)
    environment = scoring._environment(model, settings)
    environment["torch_cxx11_abi"] = torch._C._GLIBCXX_USE_CXX11_ABI
    started = time.time()
    worker_meta = {"worker": args.worker, "environment": environment, "flash_verification": profile, "load_seconds": load_seconds, "started_epoch": started, "planned_chunks": len(chunks), "planned_rows": sum(chunk["stop"] - chunk["start"] for chunk in chunks), "manifest_sha256": prep["manifest_sha256"], "model_sha256": prep["model_files"]["model.safetensors"]["sha256"], "scoring_code_sha256": digest(PILOT / "sensitivity" / "run_condition.py"), "causal_code_sha256": digest(PILOT.parent / "zero-shot-eval-causal.py")}
    write_json(output / "environment.json", worker_meta)
    upload_file(s3, output / "environment.json", f"{args.prefix}/workers/gpu{args.worker:03d}/environment.json")
    completed_rows = 0
    for chunk_index, chunk in enumerate(chunks):
        spec = specs[chunk["task"]]
        # The original conservation/SV helpers use positional DataFrame indices.
        frame = read_frame(prep, spec, chunk["start"], chunk["stop"])
        key = f"{chunk['task']}--{chunk['start']:06d}-{chunk['stop']:06d}"
        chunk_started = time.time()

        def progress(phase, phase_index, phase_count, done, total):
            record = {"worker": args.worker, "chunk": key, "completed_chunks": chunk_index, "planned_chunks": len(chunks), "completed_rows": completed_rows, "current_phase": phase, "phase_completed": done, "phase_total": total, "completed_forwards": 2 * completed_rows + phase_index * total + done, "elapsed_seconds": time.time() - started}
            write_json(output / "progress.json", record)
            upload_json(s3, f"{args.prefix}/workers/gpu{args.worker:03d}/progress.json", record)
            print(json.dumps(record), flush=True)

        progress("starting", 0, 2, 0, len(frame))
        if spec["kind"] == "sv":
            result, arrays = scoring._score_sv(spec, frame, model, tokenizer, settings, progress, compute_metrics=prep["input_format"] == "tsv")
        else:
            result, arrays = scoring._score_standard_task(spec, frame, model, tokenizer, settings, progress, compute_metrics=prep["input_format"] == "tsv")
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise RuntimeError(f"Non-finite predictions: {key}")
        array_path = output / f"{key}.npz"
        np.savez_compressed(array_path, **arrays)
        payload = {**chunk, "key": key, "result": result, "array_sha256": digest(array_path), "started_epoch": chunk_started, "finished_epoch": time.time()}
        write_json(output / f"{key}.json", payload)
        # JSON is the completion marker; publish only after the array is verified.
        upload_file(s3, array_path, f"{args.prefix}/chunks/{key}.npz")
        upload_file(s3, output / f"{key}.json", f"{args.prefix}/chunks/{key}.json")
        completed_rows += len(frame)
        print(json.dumps({"worker": args.worker, "completed_chunk": key, "completed_chunks": chunk_index + 1, "total_chunks": len(chunks)}), flush=True)
    final = {**worker_meta, "finished_epoch": time.time(), "run_wall_seconds": time.time() - started, "completed_chunks": len(chunks), "completed_rows": completed_rows}
    write_json(output / "done.json", final)
    upload_file(s3, output / "done.json", f"{args.prefix}/workers/gpu{args.worker:03d}/done.json")


if __name__ == "__main__":
    main()
