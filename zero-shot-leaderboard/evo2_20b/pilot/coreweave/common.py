"""Pinned reproduction inputs and CoreWeave-only object-store helpers."""

import hashlib
import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
from checkpoints import checkpoint

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
CHECKPOINT = checkpoint(os.environ.get("PLANTCAD_CHECKPOINT", "056t"))
RUN_ID = CHECKPOINT["run_id"]
STEP = CHECKPOINT["step"]
EVAL_MODE = os.environ.get("PLANTCAD_EVAL_MODE", "sampled")
if EVAL_MODE not in ("sampled", "full"):
    raise ValueError("PLANTCAD_EVAL_MODE must be sampled or full")
HF_REPO = "eczech/marindna-exp472"
MODEL_PREFIX = CHECKPOINT["model_prefix"]
MODEL_REVISION = CHECKPOINT["revision"]
REFERENCE_PREFIX = "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/results/step-535985/leaderboard-seed0-n10000"
REFERENCE_REVISION = "dfd4d41f0fc70f95cc22081de28a1546412f1bdc"
REFERENCE_RESULT = "results/leaderboard-seed0-n10000"
MANIFEST_SHA256 = "6549414b71fe76e8d1f5e0f10b0559a07b35e064607fae0de20b81fc9dfc1345"
BUCKET = "marin-us-east-02a"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def storage():
    return boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 5, "mode": "standard"}, max_pool_connections=32))


def identity() -> tuple[str, int, int]:
    task, attempt = os.environ["IRIS_TASK_ID"].rsplit(":", 1)
    if int(attempt) != 0:
        raise RuntimeError("This reproduction runner requires a fresh job after preemption; refusing to reuse earlier-attempt chunk markers")
    job, rank = task.rsplit("/", 1)
    if not job.startswith("/eczech/"):
        raise RuntimeError(f"Unexpected task owner: {task}")
    nodes = int(os.environ["IRIS_NUM_TASKS"])
    if nodes != int(os.environ["PLANTCAD_CW_NODES"]):
        raise RuntimeError("Replica count differs from submitted node count")
    if os.environ["PLANTCAD_CW_CLUSTER"] not in ("cw-us-east-02a", "cw-rno2a"):
        raise RuntimeError("Cluster outside H100 allowlist")
    return job.rsplit("/", 1)[-1], int(rank), nodes


def result_prefix(job_name: str) -> str:
    return f"MarinDNA/plantcad2-evals/{RUN_ID}/step-{STEP}/coreweave/{job_name}"


def list_keys(s3, prefix: str) -> list[str]:
    return [obj["Key"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix.rstrip("/") + "/") for obj in page.get("Contents", [])]


def upload_file(s3, local: Path, key: str) -> None:
    s3.upload_file(str(local), BUCKET, key)
    remote = s3.head_object(Bucket=BUCKET, Key=key)
    if remote["ContentLength"] != local.stat().st_size:
        raise RuntimeError(f"Upload size mismatch: {key}")


def upload_json(s3, key: str, value: object) -> None:
    s3.put_object(Bucket=BUCKET, Key=key, Body=(json.dumps(value, indent=2) + "\n").encode(), ContentType="application/json")
