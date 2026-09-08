"""One Iris task per H100 node; eight persistent workers, one final global reducer."""

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

from common import BUCKET, EVAL_MODE, HERE, PILOT, REFERENCE_REVISION, RUN_ID, STEP, digest, identity, list_keys, result_prefix, storage, upload_file, upload_json, write_json
from prepare import prepare, prepare_full_inputs
from inputs import read_frame

sys.path.insert(0, str(PILOT))
from partition import plan_chunks, serialize_plan
from aggregate import aggregate, recompute_task, scoring
from compare import compare, group_scores


def verify_reference_metrics(preparation: dict) -> None:
    reference = Path(preparation["reference"])
    baseline = json.loads((reference / "result.json").read_text())
    arrays = {}
    for path in sorted((reference / "workers").glob("gpu*/scores.npz")):
        with np.load(path, allow_pickle=False) as archive:
            arrays.update({key: archive[key] for key in archive.files})
    for spec in scoring.LEADERBOARD_TASKS:
        frame = pd.read_csv(Path(preparation["samples"]) / spec["file"], sep="\t")
        prefix = spec["key"] + "__"
        result = recompute_task(spec, frame, {key.removeprefix(prefix): value for key, value in arrays.items() if key.startswith(prefix)})
        expected = baseline["tasks"][spec["key"]]
        if result["best_context"] != expected["best_context"]:
            raise ValueError("Reference strand selection differs under the reducer")
        for context, record in result["contexts"].items():
            for metric, value in record["metrics"].items():
                if abs(value - expected["contexts"][context]["metrics"][metric]) > 1e-14:
                    raise ValueError(f"Reference metric recomputation mismatch: {spec['key']}/{context}/{metric}")
    print("REFERENCE_METRICS_VERIFIED all 20 tasks and all forward/RC metrics", flush=True)


def collect_and_reduce(root: Path, preparation: dict, prefix: str, nodes: int, expected_chunks: int) -> None:
    s3 = storage()
    deadline = time.monotonic() + 21600
    while True:
        keys = list_keys(s3, prefix)
        failed = [key for key in keys if key.endswith("/failed.json")]
        if failed:
            raise RuntimeError(f"A peer failed: {failed}")
        done = [key for key in keys if "/workers/" in key and key.endswith("/done.json")]
        if len(done) == nodes * 8:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for global worker completion")
        print(json.dumps({"phase": "waiting_for_peer_workers", "done_workers": len(done), "total_workers": nodes * 8}), flush=True)
        time.sleep(20)
    artifact = root / "artifact"
    wanted = [key for key in keys if "/chunks/" in key or "/workers/" in key and key.endswith(("/done.json", "/environment.json")) or "/nodes/" in key]

    def download(key):
        local = artifact / key.removeprefix(prefix + "/")
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(download, wanted))
    chunk_files = list((artifact / "chunks").glob("*.json"))
    if len(chunk_files) != expected_chunks:
        raise RuntimeError(f"Expected {expected_chunks} chunks, got {len(chunk_files)}")
    for path in chunk_files:
        piece = json.loads(path.read_text())
        if digest(path.with_suffix(".npz")) != piece["array_sha256"]:
            raise RuntimeError(f"Chunk integrity mismatch: {path.name}")
    records = [json.loads(path.read_text()) for path in sorted((artifact / "workers").glob("gpu*/done.json"))]
    expected_plan = json.loads((root / "plan.json").read_text())
    if sorted((piece["task"], piece["start"], piece["stop"], piece["worker"]) for piece in (json.loads(path.read_text()) for path in chunk_files)) != sorted((piece["task"], piece["start"], piece["stop"], piece["worker"]) for piece in expected_plan):
        raise RuntimeError("Completed chunks differ from the submitted plan")
    if len({record["scoring_code_sha256"] for record in records}) != 1 or not all(record["flash_verification"]["external_flash_attention_2_verified"] for record in records):
        raise RuntimeError("Scoring version or FlashAttention verification mismatch")
    if any(record["manifest_sha256"] != preparation["manifest_sha256"] or record["model_sha256"] != preparation["model_files"]["model.safetensors"]["sha256"] for record in records):
        raise RuntimeError("Workers did not use identical verified inputs/model weights")
    if sum(record["completed_rows"] for record in records) != preparation["sampling"]["total_rows"]:
        raise RuntimeError("Worker row totals do not cover the evaluation")
    result = aggregate(artifact, preparation, records, nodes)
    comparison = compare(artifact, Path(preparation["reference"])) if preparation["reference"] and preparation.get("reference_same_checkpoint", False) else None
    for filename in ("preparation.json", "plan.json"):
        shutil.copy2(root / filename, artifact / filename)
    shutil.copy2(Path(preparation["samples"]) / "manifest.json", artifact / "sample-manifest.json")
    write_json(artifact / "provenance.json", {"cluster": os.environ["PLANTCAD_CW_CLUSTER"], "iris_task": os.environ["IRIS_TASK_ID"], "nodes": nodes, "priority": "batch", "user": "eczech", "s3_prefix": f"s3://{BUCKET}/{prefix}", "reference_revision": preparation["reference_revision"], "sampling": preparation["sampling"], "checkpoint": preparation["checkpoint"], "evaluation_code_sha256": records[0]["scoring_code_sha256"], "causal_code_sha256": records[0]["causal_code_sha256"], "node_count_affects_partition_only": True})
    if EVAL_MODE == "full":
        # Keep individual chunks in CWS3; HF retains their manifest and all merged
        # raw arrays, avoiding thousands of duplicate HF files per full run.
        write_json(artifact / "chunk-manifest.json", [json.loads(path.read_text()) for path in sorted(chunk_files)])
        shutil.move(str(artifact / "chunks"), str(root / "verified-chunks"))
    for path in artifact.iterdir():
        if path.is_file():
            upload_file(s3, path, f"{prefix}/final/{path.name}")
    print(json.dumps({"phase": "global_results", "runtime": result["runtime"], "max_selected_metric_delta": comparison["max_selected_metric_delta"] if comparison else None, "raw_predictions_exact": comparison["raw_predictions_exact"] if comparison else None, "groups": group_scores(result)}, indent=2), flush=True)
    hf_destination = f"{RUN_ID}/results/step-{STEP}/coreweave/{prefix.rsplit('/', 1)[-1]}"
    subprocess.run([sys.executable, str(PILOT / "upload_results.py"), "--local-dir", str(artifact), "--path-in-repo", hf_destination, "--commit-message", f"CoreWeave {EVAL_MODE} evaluation: {nodes}x8 H100, {RUN_ID}"], check=True)
    upload_json(s3, f"{prefix}/complete.json", {"hf_destination": hf_destination, "sampling": preparation["sampling"], "groups": group_scores(result), "completed_epoch": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", help="Validate downloads, reference reducer and one real batch; no full evaluation")
    parser.add_argument("--validate-full-inputs", action="store_true", help="With --smoke, also verify the complete pinned Parquet inventory")
    args = parser.parse_args()
    if args.validate_full_inputs and not args.smoke:
        parser.error("--validate-full-inputs is only used with --smoke")
    job_name, rank, nodes = identity()
    root = Path("/tmp") / f"plantcad2-{job_name}"
    prefix = result_prefix(job_name)
    s3 = storage()
    processes = []
    logs = []
    node_started = time.time()
    try:
        preparation = prepare(root)
        if rank == 0 and preparation["reference"]:
            verify_reference_metrics(preparation)
        expected_files = {spec["file"] for spec in scoring.LEADERBOARD_TASKS}
        if set(preparation["inputs"]) != expected_files:
            raise ValueError("Dataset task inventory differs from the scorer's 20 tasks")
        plan = plan_chunks({spec["key"]: preparation["inputs"][spec["file"]]["rows"] for spec in scoring.LEADERBOARD_TASKS}, nodes * 8, args.chunk_size)
        write_json(root / "plan.json", serialize_plan(plan))
        if rank == 0:
            upload_json(s3, f"{prefix}/execution.json", {"total_forwards": 2 * preparation["sampling"]["total_rows"], "total_chunks": len(plan), "total_workers": nodes * 8, "sampling": preparation["sampling"], "checkpoint": preparation["checkpoint"], "manifest_sha256": preparation["manifest_sha256"]})
        info = {"rank": rank, "nodes": nodes, "task": os.environ["IRIS_TASK_ID"], "started_epoch": node_started, "prepare_seconds": preparation["prepare_seconds"], "plan_sha256": digest(root / "plan.json"), "hostname": os.uname().nodename, "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True), "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)}
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}.json", info)
        if args.smoke:
            subprocess.run([sys.executable, str(HERE / "test_aggregation.py")], check=True)
            subprocess.run([sys.executable, str(HERE / "test_inputs.py")], check=True)
            if EVAL_MODE != "sampled":
                raise ValueError("Use sampled mode for the saved-reference smoke; full slices have separate unit tests")
            command = [sys.executable, str(PILOT / "sensitivity" / "run_condition.py"), "--model", preparation["model"], "--sample-dir", preparation["samples"], "--output-dir", str(root / "smoke"), "--condition", "cw-smoke", "--dtype", "bf16", "--attention", "flash_attention_2", "--batch-size", "32", "--sv-batch-size", "32", "--softmax-dtype", "fp32", "--tf32", "--no-use-cache", "--require-flash", "--max-samples", "32"]
            subprocess.run(command, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            if preparation.get("reference_same_checkpoint", False):
                reference_arrays = {}
                for path in (Path(preparation["reference"]) / "workers").glob("gpu*/scores.npz"):
                    with np.load(path, allow_pickle=False) as archive:
                        reference_arrays.update({key: archive[key] for key in archive.files})
                with np.load(root / "smoke" / "scores.npz", allow_pickle=False) as archive:
                    for key in archive.files:
                        if not np.array_equal(archive[key], reference_arrays[key][:32]):
                            raise ValueError(f"Sampled regression predictions differ: {key}")
                print("SAMPLED_PREDICTIONS_BYTE_IDENTICAL", flush=True)
            if args.validate_full_inputs:
                from huggingface_hub import HfApi
                full = prepare_full_inputs(root, HfApi(token=os.environ["HUGGING_FACE_HUB_TOKEN"]))
                for spec in scoring.LEADERBOARD_TASKS:
                    rows = full["inputs"][spec["file"]]["rows"]
                    assert len(read_frame(full, spec, max(0, rows - 33), rows)) == min(33, rows)
                upload_json(s3, f"{prefix}/smoke/full-inputs.json", full)
                print(json.dumps({"phase": "full_inventory_verified", "sampling": full["sampling"], "inputs": full["inputs"], "manifest_sha256": full["manifest_sha256"]}), flush=True)
            for path in (root / "smoke").iterdir():
                if path.is_file():
                    upload_file(s3, path, f"{prefix}/smoke/node{rank:03d}/{path.name}")
            print("SMOKE_COMPLETE", flush=True)
            return
        for local_gpu in range(8):
            worker = rank * 8 + local_gpu
            log = (root / f"gpu{worker:03d}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(HERE / "gpu_worker.py"), "--root", str(root), "--worker", str(worker), "--prefix", prefix], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(local_gpu)}, stdout=log, stderr=subprocess.STDOUT))
        while True:
            statuses = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in statuses):
                raise RuntimeError(f"GPU worker failed: {statuses}; inspect {root}/gpu*.log")
            peer_failures = [key for key in list_keys(s3, f"{prefix}/nodes") if key.endswith("/failed.json")]
            if peer_failures:
                raise RuntimeError(f"Stopping local workers because a peer failed: {peer_failures}")
            progress = [json.loads(path.read_text()) for path in sorted((root / "workers").glob("gpu*/progress.json"))]
            print(json.dumps({"phase": "node_progress", "node": rank, "done_workers": sum(code == 0 for code in statuses), "progress": progress}), flush=True)
            if all(code == 0 for code in statuses):
                break
            time.sleep(20)
        for local_gpu in range(8):
            worker = rank * 8 + local_gpu
            upload_file(s3, root / f"gpu{worker:03d}.log", f"{prefix}/workers/gpu{worker:03d}/run.log")
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}-done.json", {"rank": rank, "finished_epoch": time.time(), "wall_seconds": time.time() - node_started})
        if rank == 0:
            collect_and_reduce(root, preparation, prefix, nodes, len(plan))
    except BaseException as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        upload_json(s3, f"{prefix}/nodes/node{rank:03d}/failed.json", {"error": repr(error), "epoch": time.time()})
        for path in root.glob("gpu*.log"):
            upload_file(s3, path, f"{prefix}/failure-logs/{path.name}")
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
