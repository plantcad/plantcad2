"""Download pinned model/reference data once per node and verify every fixture."""

import json
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, snapshot_download
import pyarrow.parquet as pq

from common import CHECKPOINT, EVAL_MODE, HF_REPO, MANIFEST_SHA256, MODEL_PREFIX, MODEL_REVISION, REFERENCE_PREFIX, REFERENCE_RESULT, REFERENCE_REVISION, digest, write_json
from checkpoints import DATASET_REPO, DATASET_REVISION, FULL_ROWS


def prepare_full_inputs(root: Path, api: HfApi) -> dict:
    """Pin every source file; retain every row in HF shard/row order, without sampling."""
    inventory = sorted((entry for entry in api.list_repo_tree(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, recursive=True) if entry.path.endswith(".parquet")), key=lambda entry: entry.path)
    snapshot = Path(snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, allow_patterns=[entry.path for entry in inventory], token=api.token, local_dir=root / "full-inputs", max_workers=8))

    def inspect(entry):
        path = snapshot / entry.path
        sha256 = digest(path)
        if path.stat().st_size != entry.size or entry.lfs and sha256 != entry.lfs.sha256:
            raise ValueError(f"Full dataset source integrity mismatch: {entry.path}")
        rows = pq.ParquetFile(path).metadata.num_rows
        return {"path": entry.path, "bytes": entry.size, "rows": rows, "sha256": sha256}

    with ThreadPoolExecutor(max_workers=8) as pool:
        sources = list(pool.map(inspect, inventory))
    inputs = {}
    for source in sources:
        task, filename = source["path"].split("/")
        split = filename.split("-", 1)[0]
        record = inputs.setdefault(f"{task}__{split}.tsv", {"rows": 0, "files": []})
        record["rows"] += source["rows"]
        record["files"].append(source)
    if len(inputs) != 20 or sum(record["rows"] for record in inputs.values()) != FULL_ROWS:
        raise ValueError("Full input inventory differs from the pinned 20-task / 1,727,943-row dataset")
    manifest = {"repo_id": DATASET_REPO, "revision": DATASET_REVISION, "sampling": "none", "order": "sorted HF shards, original row order", "total_rows": FULL_ROWS, "inputs": inputs}
    write_json(snapshot / "manifest.json", manifest)
    return {"samples": str(snapshot), "reference": None, "reference_revision": None, "input_format": "parquet", "inputs": inputs, "manifest_sha256": digest(snapshot / "manifest.json"), "sample_count": len(inputs), "sampling": {"method": "full", "samples_per_task": None, "seed": None, "dataset_revision": DATASET_REVISION, "total_rows": FULL_ROWS}}


def prepare(root: Path) -> dict:
    started = time.time()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if (usage.used + (32 if EVAL_MODE == "full" else 12) * 2**30) / usage.total >= 0.90:
        raise RuntimeError("Checkpoint/fixture staging would exceed 90% disk usage")
    api = HfApi(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    api.repo_info(HF_REPO, repo_type="model")
    model_snapshot = Path(snapshot_download(HF_REPO, revision=MODEL_REVISION, allow_patterns=[f"{MODEL_PREFIX}/*"], token=api.token, local_dir=root / "model-download", max_workers=8))
    model = model_snapshot / MODEL_PREFIX
    inventory = list(api.list_repo_tree(HF_REPO, path_in_repo=MODEL_PREFIX, recursive=True, expand=True, revision=MODEL_REVISION))
    model_files = {}
    for entry in inventory:
        if not hasattr(entry, "size"):
            continue
        relative = entry.path.removeprefix(MODEL_PREFIX + "/")
        path = model / relative
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"Model byte mismatch: {relative}")
        sha256 = digest(path)
        if entry.lfs and sha256 != entry.lfs.sha256:
            raise RuntimeError(f"Model SHA-256 mismatch: {relative}")
        model_files[relative] = {"bytes": entry.size, "sha256": sha256}
    if len(model_files) != 4 or sum(row["bytes"] for row in model_files.values()) != 3892739156:
        raise RuntimeError("Unexpected model inventory")
    model_info = {"model": str(model), "model_revision": MODEL_REVISION, "model_prefix": MODEL_PREFIX, "checkpoint": CHECKPOINT, "model_files": model_files}
    if EVAL_MODE == "full":
        result = {**model_info, **prepare_full_inputs(root, api), "prepare_seconds": time.time() - started}
        write_json(root / "preparation.json", result)
        print(json.dumps({"phase": "full_inputs_verified", "total_rows": FULL_ROWS, "manifest_sha256": result["manifest_sha256"]}), flush=True)
        return result
    patterns = [f"{REFERENCE_PREFIX}/samples/*", f"{REFERENCE_PREFIX}/{REFERENCE_RESULT}/result.json", f"{REFERENCE_PREFIX}/{REFERENCE_RESULT}/workers/gpu*/scores.npz"]
    reference_snapshot = Path(snapshot_download(HF_REPO, revision=REFERENCE_REVISION, allow_patterns=patterns, token=api.token, local_dir=root / "reference-download", max_workers=16))
    reference = reference_snapshot / REFERENCE_PREFIX
    samples = reference / "samples"
    manifest_path = samples / "manifest.json"
    if digest(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Reference sample-manifest hash mismatch")
    records = json.loads(manifest_path.read_text())
    if len(records) != 20:
        raise RuntimeError("Expected 20 task fixtures")
    for record in records:
        if record["samples"] != 10000 or record["seed"] != 0 or record["sampling"] != "random_unstratified":
            raise RuntimeError("Unexpected sampling contract")
        if digest(samples / Path(record["path"]).name) != record["sha256"]:
            raise RuntimeError(f"Fixture hash mismatch: {record['path']}")
    inputs = {Path(record["path"]).name: {"rows": record["samples"]} for record in records}
    result = {**model_info, "samples": str(samples), "reference": str(reference / REFERENCE_RESULT), "reference_revision": REFERENCE_REVISION, "manifest_sha256": MANIFEST_SHA256, "sample_count": len(records), "prepare_seconds": time.time() - started, "input_format": "tsv", "inputs": inputs, "sampling": {"samples_per_task": 10000, "seed": 0, "method": "retained_random_unstratified", "total_rows": 200000}}
    result["reference_same_checkpoint"] = MODEL_REVISION == "2972ca5abb575ccb9878d2525aafd396e6b73d7c"
    write_json(root / "preparation.json", result)
    return result
