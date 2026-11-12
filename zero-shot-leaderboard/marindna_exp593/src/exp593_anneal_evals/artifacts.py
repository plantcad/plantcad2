"""Annealing-only writes; reuse the unchanged checksum-verified read helpers."""

import json
from pathlib import Path

from exp593_anneal_evals import ARTIFACT_ROOT
from exp593_evals.storage import client, upload_verified


def publish(path: Path, uri: str) -> dict:
    if not uri.startswith(ARTIFACT_ROOT):
        raise ValueError("Annealing results may not overwrite historical artifacts")
    return upload_verified(client(), path, uri)


def publish_json(value: dict, path: Path, uri: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return publish(path, uri)
