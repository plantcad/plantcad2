"""One Iris task per H100x8 node for distributed maize-AF evaluation."""

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

HERE = Path(__file__).resolve().parent
COREWEAVE = HERE.parent
PILOT = COREWEAVE.parent
sys.path[:0] = [str(HERE), str(COREWEAVE), str(PILOT)]

from common import BUCKET, digest, identity, list_keys, result_prefix, storage, upload_file, upload_json, write_json
from partition import plan_chunks, serialize_plan
from aggregate import aggregate
from config import EXPECTED_SAMPLE_ROWS, EXPECTED_UNION_ROWS, SAMPLE_SIZES
from prepare import prepare
from smoke import main as run_smoke


def collect(root: Path, preparation: dict, prefix: str, nodes: int, plan: list[dict]) -> None:
    s3 = storage()
    deadline = time.monotonic() + 21_600
    while True:
        keys = list_keys(s3, prefix)
        failures = [key for key in keys if key.endswith("/failed.json")]
        if failures:
            raise RuntimeError(f"A peer failed: {failures}")
        done = [key for key in keys if "/workers/" in key and key.endswith("/done.json")]
        if len(done) == nodes * 8:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for AF workers")
        print(json.dumps({"phase": "waiting_for_af_workers", "done_workers": len(done), "total_workers": nodes * 8}), flush=True)
        time.sleep(20)

    artifact = root / "artifact"
    wanted = [key for key in keys if "/chunks/" in key or "/workers/" in key and key.endswith(("/done.json", "/environment.json")) or "/nodes/" in key]

    def download(key: str) -> None:
        local = artifact / key.removeprefix(prefix + "/")
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, wanted))
    chunk_metadata = sorted((path for path in (artifact / "chunks").glob("*.json")))
    chunks = []
    for metadata_path in chunk_metadata:
        metadata = json.loads(metadata_path.read_text())
        array_path = metadata_path.with_suffix(".npz")
        if digest(array_path) != metadata["array_sha256"]:
            raise RuntimeError(f"Chunk hash mismatch: {array_path.name}")
        chunks.append((metadata, array_path))
    workers = [json.loads(path.read_text()) for path in sorted((artifact / "workers").glob("gpu*/done.json"))]
    only_sample_size = int(os.environ.get("PLANTCAD_AF_ONLY_SAMPLE", "0"))
    expected_rows = EXPECTED_SAMPLE_ROWS[only_sample_size] if only_sample_size else EXPECTED_UNION_ROWS
    if len(workers) != nodes * 8 or sum(record["completed_rows"] for record in workers) != expected_rows:
        raise RuntimeError("Worker completion inventory is incomplete")
    if len({record["scoring_code_sha256"] for record in workers}) != 1 or not all(record["flash_verification"]["external_flash_attention_2_verified"] for record in workers):
        raise RuntimeError("Workers did not share one profiler-verified scorer")
    window_size = int(os.environ.get("PLANTCAD_AF_WINDOW_SIZE", "8192"))
    if {record["environment"]["window_size"] for record in workers} != {window_size} or not all(record["crop_validation"]["entire_window_matches_center_crop"] and record["crop_validation"]["reverse_complement_exact"] for record in workers):
        raise RuntimeError("Workers did not verify one consistent centered context crop")
    result = aggregate(artifact, Path(preparation["union"]), plan, chunks, only_sample_size or None)
    result["runtime"] = {
        "workers": len(workers),
        "max_worker_seconds": max(record["run_wall_seconds"] for record in workers),
        "sum_worker_seconds": sum(record["run_wall_seconds"] for record in workers),
        "aggregate_variants_per_second": expected_rows / max(record["run_wall_seconds"] for record in workers),
    }
    (artifact / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for source, destination in ((root / "preparation.json", artifact / "preparation.json"), (root / "plan.json", artifact / "plan.json"), (Path(preparation["samples"]) / "sampling-manifest.json", artifact / "sampling-manifest.json")):
        shutil.copy2(source, destination)
    provenance = {"cluster": os.environ["PLANTCAD_CW_CLUSTER"], "iris_job": prefix.rsplit("/", 1)[-1], "nodes": nodes, "gpus": nodes * 8, "priority": "batch", "checkpoint": preparation["checkpoint"], "scoring_method": os.environ.get("PLANTCAD_AF_METHOD", "cached"), "window_size": window_size, "only_sample_size": only_sample_size or None, "forward_reverse_combination": "arithmetic mean of raw strand LLRs", "row_level_predictions_retained": True}
    write_json(artifact / "provenance.json", provenance)
    for path in artifact.iterdir():
        if path.is_file():
            upload_file(s3, path, f"{prefix}/final/{path.name}")
    checkpoint = preparation["checkpoint"]
    destination = f"{checkpoint['run_id']}/results/step-{checkpoint['step']}/coreweave/{prefix.rsplit('/', 1)[-1]}"
    subprocess.run([sys.executable, str(PILOT / "upload_results.py"), "--local-dir", str(artifact), "--path-in-repo", destination, "--commit-message", f"Maize AF evaluation: {checkpoint['run_id']}"], check=True)
    upload_json(s3, f"{prefix}/complete.json", {"hf_destination": destination, "metrics": result["metrics"], "runtime": result["runtime"], "completed_epoch": time.time()})
    print(json.dumps({"phase": "maize_af_complete", "hf_destination": destination, "metrics": result["metrics"], "runtime": result["runtime"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=128)
    args = parser.parse_args()
    job_name, rank, nodes = identity()
    root = Path("/tmp") / f"plantcad2-{job_name}"
    prefix = result_prefix(job_name)
    s3 = storage()
    processes = []
    logs = []
    started = time.time()
    try:
        preparation = prepare(root)
        if args.smoke:
            if nodes != 1 or rank != 0:
                raise RuntimeError("The AF smoke requires exactly one H100x8 node")
            run_smoke(root, prefix, args.smoke_rows)
            return
        only_sample_size = int(os.environ.get("PLANTCAD_AF_ONLY_SAMPLE", "0"))
        if only_sample_size not in (0, *SAMPLE_SIZES):
            raise RuntimeError(f"Unsupported sample size: {only_sample_size}")
        evaluation_rows = EXPECTED_SAMPLE_ROWS[only_sample_size] if only_sample_size else EXPECTED_UNION_ROWS
        chunks = plan_chunks({"maize-af": evaluation_rows}, nodes * 8, args.chunk_size)
        plan = serialize_plan(chunks)
        write_json(root / "plan.json", plan)
        if rank == 0:
            upload_json(s3, f"{prefix}/execution.json", {"unique_variants": evaluation_rows, "strand_scores": 2 * evaluation_rows, "workers": nodes * 8, "chunks": len(plan), "checkpoint": preparation["checkpoint"], "method": os.environ.get("PLANTCAD_AF_METHOD", "cached"), "window_size": int(os.environ.get("PLANTCAD_AF_WINDOW_SIZE", "8192")), "only_sample_size": only_sample_size or None})
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}.json", {"rank": rank, "nodes": nodes, "task": os.environ["IRIS_TASK_ID"], "prepare_seconds": preparation["prepare_seconds"], "plan_sha256": digest(root / "plan.json"), "started_epoch": started})
        for local_gpu in range(8):
            worker = rank * 8 + local_gpu
            log = (root / f"gpu{worker:03d}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(HERE / "gpu_worker.py"), "--root", str(root), "--worker", str(worker), "--prefix", prefix], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(local_gpu)}, stdout=log, stderr=subprocess.STDOUT))
        while True:
            statuses = [process.poll() for process in processes]
            if any(status not in (None, 0) for status in statuses):
                raise RuntimeError(f"AF GPU worker failed: {statuses}; inspect {root}/gpu*.log")
            peer_failures = [key for key in list_keys(s3, f"{prefix}/nodes") if key.endswith("/failed.json")]
            if peer_failures:
                raise RuntimeError(f"Stopping because a peer failed: {peer_failures}")
            progress = [json.loads(path.read_text()) for path in sorted((root / "workers").glob("gpu*/progress.json"))]
            print(json.dumps({"phase": "af_node_progress", "node": rank, "done_workers": sum(status == 0 for status in statuses), "progress": progress}), flush=True)
            if all(status == 0 for status in statuses):
                break
            time.sleep(20)
        for local_gpu in range(8):
            worker = rank * 8 + local_gpu
            upload_file(s3, root / f"gpu{worker:03d}.log", f"{prefix}/workers/gpu{worker:03d}/run.log")
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}-done.json", {"rank": rank, "finished_epoch": time.time(), "wall_seconds": time.time() - started})
        if rank == 0:
            collect(root, preparation, prefix, nodes, plan)
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
