"""Reuse explicitly pinned, finished CPU jobs after an interrupted controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exp593_evals.storage import read_json

CONTROLLERS = {
    "src/exp593_evals/analysis_bootstrap.py",
    "src/exp593_evals/analysis_resume.py",
}


def scientific_files(manifest: dict) -> dict:
    """Pin every computational dependency except the two recovery controllers."""
    return {
        name: sha
        for name, sha in manifest["files"].items()
        if name not in CONTROLLERS
        and (
            name.startswith(("src/", "native/", "provenance/"))
            or name
            in {"binding.json", "catalog-record.json", "pyproject.toml", "uv.lock"}
        )
    }


def job_name(job: dict) -> str:
    name = job["profile"] + "/" + job["variant"] + "/" + job["kind"]
    return name + "/" + job["task"] if job.get("task") else name


def recovered_jobs(source: Path, phase: str, work: list, owner: str, storage) -> dict:
    plan_path = source / "recovery-plan.json"
    if not plan_path.exists():
        return {}
    plan = json.loads(plan_path.read_text())
    binding = json.loads((source / "binding.json").read_text())
    manifest = json.loads((source / "source-manifest.json").read_text())
    if plan["binding"] != binding or plan["scientific_files"] != scientific_files(
        manifest
    ):
        raise ValueError("Recovery scientific source or checkpoint differs")
    expected = {job_name(job): job for job in work}
    completed = {}
    for entry in plan["phases"].get(phase, []):
        job, record = entry["job"], entry["receipt"]
        name = job_name(job)
        prefix = owner.rstrip("/") + "/analysis/" + phase + "/attempts/"
        if name in completed or expected.get(name) != job:
            raise ValueError("Unexpected or duplicate recovered job")
        if not record["uri"].startswith(prefix) or not record["uri"].endswith(
            "/results/" + name + "/analysis-receipt.json"
        ):
            raise ValueError("Recovered receipt is outside its checkpoint/phase")
        receipt, raw = read_json(storage, record["uri"])
        if (
            len(raw) != record["bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("Recovered receipt digest differs")
        profile = (
            "context-8192" if job["profile"] == "frozen-probes" else job["profile"]
        )
        if receipt["binding"] != binding or any(
            receipt.get(key) != value
            for key, value in {
                "profile": profile,
                "kind": job["kind"],
                "variant": job["variant"],
                "task": job.get("task"),
            }.items()
        ):
            raise ValueError("Recovered job binding differs")
        artifacts = receipt["artifacts"]
        if sum(a["uri"].endswith("/result.json") for a in artifacts) != 1:
            raise ValueError("Recovered job has no unique finalized result")
        artifact_prefix = record["uri"].rsplit("/", 1)[0] + "/"
        for artifact in artifacts:
            if not artifact["uri"].startswith(artifact_prefix):
                raise ValueError("Recovered artifact is outside its finalized job")
            bucket, key = artifact["uri"].removeprefix("s3://").split("/", 1)
            body = storage.get_object(Bucket=bucket, Key=key)["Body"]
            digest, size = hashlib.sha256(), 0
            try:
                while chunk := body.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            finally:
                body.close()
            if size != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
                raise ValueError("Recovered artifact digest differs")
        completed[name] = {
            "job": job,
            "receipt": record["uri"],
            "receipt_sha256": record["sha256"],
            "reused": True,
        }
    return completed
