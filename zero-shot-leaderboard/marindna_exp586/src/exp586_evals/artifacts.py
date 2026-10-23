"""Verify and retrieve original-format worker artifacts from native inference."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(
    root: str,
    directory: Path,
    preparation: Path,
    condition: str,
    read: Callable[[str], bytes],
    checkpoint_digest: str,
    *,
    verify_only: bool = False,
) -> dict:
    report = json.loads(read(root + "/manifest.json"))
    binding = report["binding"]
    if (
        binding["requests_sha256"] != sha256(preparation)
        or binding["condition"] != condition
        or binding["checkpoint_digest"] != checkpoint_digest
    ):
        raise ValueError("Input/checkpoint condition mismatch")
    prep = json.loads(preparation.read_text())
    for record in prep.get("inputs", {}).values():
        for source in record["files"]:
            if sha256(preparation.parent / source["path"]) != source["sha256"]:
                raise ValueError("Input Parquet checksum differs")
    expected = [("tasks", c) for c in prep["plan"]] + [
        ("af", c) for c in prep["af_plan"]
    ]
    if len(expected) != len(report["chunks"]):
        raise ValueError("Incomplete output coverage")
    for (kind, chunk), item in zip(expected, report["chunks"], strict=True):
        if item["kind"] != kind:
            raise ValueError("Chunk kind differs")
        records = [(kind, item["receipt"])]
        if kind == "af":
            records.append(("embeddings", item["embeddings"]))
        for folder, receipt in records:
            if receipt["binding"] != binding or any(
                receipt[k] != v for k, v in chunk.items()
            ):
                raise ValueError("Chunk input/checkpoint/source binding differs")
            stem = f"{folder}/chunks/{chunk['key']}"
            committed = json.loads(read(f"{root}/{stem}.json"))
            if committed != receipt:
                raise ValueError("Completion receipt differs")
            blob = read(f"{root}/{stem}.npz")
            if hashlib.sha256(blob).hexdigest() != receipt["array_sha256"]:
                raise ValueError("Prediction checksum differs")
            if not verify_only:
                path = directory / stem
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_suffix(".npz").write_bytes(blob)
                path.with_suffix(".json").write_text(json.dumps(receipt))
    if not verify_only:
        (directory / "manifest.json").write_text(json.dumps(report, indent=2))
    return report
