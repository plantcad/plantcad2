"""Content-addressed task inputs and outputs, isolated from historical results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exp593_evals import ARTIFACT_ROOT
from exp593_evals.storage import client, upload_verified


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(uri: str) -> bytes:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return client().get_object(Bucket=bucket, Key=key)["Body"].read()


def fetch(record: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / record["uri"].rsplit("/", 1)[-1]
    if not target.exists():
        bucket, key = record["uri"].removeprefix("s3://").split("/", 1)
        client().download_file(bucket, key, str(target))
    if digest(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
        raise ValueError("Input differs from its immutable source receipt")
    return target


def publish(path: Path, uri: str) -> dict:
    if not uri.startswith(ARTIFACT_ROOT):
        raise ValueError("Evaluation writes must stay under the exp593 owner")
    return upload_verified(client(), path, uri)


def publish_json(value: dict, path: Path, uri: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return publish(path, uri)
