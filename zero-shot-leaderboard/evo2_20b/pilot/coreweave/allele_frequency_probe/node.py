"""Iris multi-node driver for maize-AF embedding extraction and ridge probing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

HERE = Path(__file__).resolve().parent
AF = HERE.parent / "allele_frequency"
COREWEAVE = HERE.parent
PILOT = COREWEAVE.parent
sys.path[:0] = [str(AF), str(COREWEAVE), str(PILOT), str(HERE)]

from common import BUCKET, HF_REPO, digest, identity, list_keys, result_prefix, storage, upload_file, upload_json, write_json
from config import EXPECTED_SAMPLE_ROWS
from partition import plan_chunks, serialize_plan
from prepare import prepare
from allele_frequency_probe.probe import fit_probes
from allele_frequency_probe.config import ZERO_SHOT_RESULTS
from allele_frequency_probe.split import build_split


def _zero_shot(root: Path, checkpoint_name: str) -> tuple[Path, dict]:
    relative = ZERO_SHOT_RESULTS[checkpoint_name]
    api = HfApi(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    revision = api.repo_info(HF_REPO).sha
    snapshot = Path(snapshot_download(HF_REPO, revision=revision, allow_patterns=[relative], token=api.token, local_dir=root / "zero-shot", max_workers=4))
    path = snapshot / relative
    return path, {"repo": HF_REPO, "revision": revision, "path": relative, "sha256": digest(path)}


def collect(root: Path, preparation: dict, prefix: str, nodes: int, plan: list[dict], checkpoint_name: str, smoke_rows: int) -> None:
    s3 = storage()
    deadline = time.monotonic() + 43_200
    while True:
        keys = list_keys(s3, prefix)
        failures = [key for key in keys if key.endswith("/failed.json")]
        if failures:
            raise RuntimeError(f"A peer failed: {failures}")
        done = [key for key in keys if "/workers/" in key and key.endswith("/done.json")]
        if len(done) == nodes * 8:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for AF-probe workers")
        print(json.dumps({"phase": "waiting_for_probe_workers", "done_workers": len(done), "total_workers": nodes * 8}), flush=True)
        time.sleep(20)

    artifact = root / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    wanted = [key for key in keys if "/chunks/" in key or "/workers/" in key and key.endswith(("/done.json", "/environment.json")) or "/nodes/" in key]

    def download(key: str) -> None:
        local = artifact / key.removeprefix(prefix + "/")
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local))

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(download, wanted))
    metadata_paths = sorted((artifact / "chunks").glob("*.json"))
    chunks: list[tuple[dict, Path]] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        array_path = metadata_path.with_suffix(".npz")
        if digest(array_path) != metadata["array_sha256"]:
            raise RuntimeError(f"Chunk hash mismatch: {array_path.name}")
        chunks.append((metadata, array_path))
    observed = sorted((row["start"], row["stop"], row["worker"]) for row, _ in chunks)
    expected = sorted((row["start"], row["stop"], row["worker"]) for row in plan)
    if observed != expected:
        raise RuntimeError("Completed embedding coverage differs from plan")
    arrays: dict[str, list[np.ndarray]] = {}
    for metadata, path in sorted(chunks, key=lambda item: item[0]["start"]):
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                arrays.setdefault(name, []).append(archive[name])
    merged = {name: np.concatenate(parts) for name, parts in arrays.items()}
    union = pd.read_parquet(preparation["union"])
    sample = union.loc[union["in_sample_100000"]].sort_values("row_id").reset_index(drop=True)
    if len(sample) != EXPECTED_SAMPLE_ROWS[100_000]:
        raise RuntimeError("Prepared 100k-target sample has the wrong size")
    evaluated = sample.iloc[:smoke_rows].copy() if smoke_rows else sample
    if not np.array_equal(evaluated["row_id"].to_numpy(), merged.pop("row_id")):
        raise RuntimeError("Aggregated embeddings do not align to the 100k-target sample")
    if smoke_rows:
        _, split_manifest = build_split(sample)
        result = {
            "phase": "smoke_complete",
            "rows": len(evaluated),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(np.isfinite(value).all())} for name, value in merged.items()},
            "full_sample_split": split_manifest,
        }
        write_json(artifact / "smoke-result.json", result)
        shutil.rmtree(artifact / "chunks")
        for path in artifact.iterdir():
            if path.is_file():
                upload_file(s3, path, f"{prefix}/final/{path.name}")
        upload_json(s3, f"{prefix}/complete.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    zero_shot_path, zero_shot_manifest = _zero_shot(root, checkpoint_name)
    zero_shot = pd.read_parquet(zero_shot_path)
    result = fit_probes(sample, merged, zero_shot, artifact)

    workers = [json.loads(path.read_text()) for path in sorted((artifact / "workers").glob("gpu*/done.json"))]
    if len(workers) != nodes * 8 or sum(row["completed_rows"] for row in workers) != len(sample):
        raise RuntimeError("Worker inventory is incomplete")
    if not all(row["flash_verification"]["external_flash_attention_2_verified"] for row in workers):
        raise RuntimeError("Not every worker verified external FlashAttention-2")
    raw_manifest = {
        "cws3_prefix": f"s3://{BUCKET}/{prefix}/chunks/",
        "chunk_count": len(chunks),
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in merged.items()},
        "zero_shot": zero_shot_manifest,
    }
    write_json(artifact / "raw-embeddings.json", raw_manifest)
    shutil.rmtree(artifact / "chunks")
    shutil.copy2(root / "preparation.json", artifact / "preparation.json")
    shutil.copy2(root / "plan.json", artifact / "plan.json")
    write_json(artifact / "provenance.json", {
        "cluster": os.environ["PLANTCAD_CW_CLUSTER"], "iris_job": prefix.rsplit("/", 1)[-1], "nodes": nodes, "gpus": nodes * 8,
        "priority": "batch", "checkpoint": preparation["checkpoint"], "window_size": 8192,
        "representation": "FP32 final-layer whole-window mean and variant-token states; FWD/RC averaged per allele",
        "zero_shot_comparison": zero_shot_manifest, "result": result,
    })
    for path in artifact.iterdir():
        if path.is_file():
            upload_file(s3, path, f"{prefix}/final/{path.name}")
    checkpoint = preparation["checkpoint"]
    destination = f"{checkpoint['run_id']}/results/step-{checkpoint['step']}/coreweave/{prefix.rsplit('/', 1)[-1]}"
    subprocess.run([sys.executable, str(PILOT / "upload_results.py"), "--local-dir", str(artifact), "--path-in-repo", destination, "--commit-message", f"Maize AF linear probe: {checkpoint['run_id']}"], check=True)
    upload_json(s3, f"{prefix}/complete.json", {"hf_destination": destination, "metrics": result["metrics"], "completed_epoch": time.time()})
    print(json.dumps({"phase": "maize_af_probe_complete", "hf_destination": destination, "metrics": result["metrics"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    checkpoint_name = os.environ.get("PLANTCAD_CHECKPOINT", "056t")
    job_name, rank, nodes = identity()
    root = Path("/tmp") / f"plantcad2-{job_name}"
    prefix = result_prefix(job_name)
    s3 = storage()
    processes: list[subprocess.Popen] = []
    logs = []
    started = time.time()
    try:
        preparation = prepare(root)
        if args.smoke_rows and (nodes != 1 or rank != 0 or args.smoke_rows < 8):
            raise RuntimeError("Smoke runs require one node and at least eight rows")
        evaluation_rows = args.smoke_rows or EXPECTED_SAMPLE_ROWS[100_000]
        plan = serialize_plan(plan_chunks({"maize-af-probe": evaluation_rows}, nodes * 8, args.chunk_size))
        write_json(root / "plan.json", plan)
        if rank == 0:
            upload_json(s3, f"{prefix}/execution.json", {"unique_variants": evaluation_rows, "workers": nodes * 8, "chunks": len(plan), "checkpoint": preparation["checkpoint"], "window_size": 8192, "representations": ["whole_window", "variant_token"]})
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}.json", {"rank": rank, "nodes": nodes, "task": os.environ["IRIS_TASK_ID"], "prepare_seconds": preparation["prepare_seconds"], "plan_sha256": digest(root / "plan.json"), "started_epoch": started})
        for local_gpu in range(8):
            worker = rank * 8 + local_gpu
            log = (root / f"gpu{worker:03d}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(HERE / "gpu_worker.py"), "--root", str(root), "--worker", str(worker), "--prefix", prefix], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(local_gpu)}, stdout=log, stderr=subprocess.STDOUT))
        while True:
            statuses = [process.poll() for process in processes]
            if any(status not in (None, 0) for status in statuses):
                raise RuntimeError(f"AF-probe worker failed: {statuses}; inspect {root}/gpu*.log")
            peer_failures = [key for key in list_keys(s3, f"{prefix}/nodes") if key.endswith("/failed.json")]
            if peer_failures:
                raise RuntimeError(f"Stopping because a peer failed: {peer_failures}")
            progress = [json.loads(path.read_text()) for path in sorted((root / "workers").glob("gpu*/progress.json"))]
            print(json.dumps({"phase": "af_probe_node_progress", "node": rank, "done_workers": sum(status == 0 for status in statuses), "progress": progress}), flush=True)
            if all(status == 0 for status in statuses):
                break
            time.sleep(20)
        for local_gpu in range(8):
            worker = rank * 8 + local_gpu
            upload_file(s3, root / f"gpu{worker:03d}.log", f"{prefix}/workers/gpu{worker:03d}/run.log")
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}-done.json", {"rank": rank, "finished_epoch": time.time(), "wall_seconds": time.time() - started})
        if rank == 0:
            collect(root, preparation, prefix, nodes, plan, checkpoint_name, args.smoke_rows)
    except BaseException as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}/failed.json", {"error": repr(error), "epoch": time.time()})
        for path in root.glob("gpu*.log"):
            upload_file(s3, path, f"{prefix}/failure-logs/{path.name}")
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
