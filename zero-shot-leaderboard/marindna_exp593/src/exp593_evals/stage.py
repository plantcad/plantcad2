"""Pinned inputs and finite work assignment for matched deployment evaluations."""

import argparse
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from exp593_evals import VARIANTS
from exp593_evals.artifacts import fetch


def digest(path: Path) -> str:
    """Hash a local immutable artifact without loading a model shard into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def materialize(record: dict[str, Any], root: Path) -> Path:
    """Verify one source and expose its original relative input path."""
    cached = fetch(record, root / "cache" / record["sha256"]).resolve()
    relative = record.get("path", cached.name)
    target = root / "samples" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() != cached:
        raise ValueError(f"Conflicting source identities at {relative}")
    if not target.exists():
        target.symlink_to(cached)
    if digest(target) != record["sha256"]:
        raise ValueError(f"Staged input digest differs at {relative}")
    return target


def jobs(
    full: dict[str, Any], classification: dict[str, Any], catalog: dict[str, Any]
) -> list[dict[str, Any]]:
    """Preserve original row chunks; only the assignment to physical workers changes."""
    result = []
    for profile, length in (("context-8192", 8192), ("context-2048", 2048)):
        for chunk in full["plan"]:
            if length == 2048 and chunk["task"] == "sv_impact":
                continue
            for variant in VARIANTS:
                result.append(
                    {
                        "profile": profile,
                        "length": length,
                        "kind": "tasks",
                        "variant": variant,
                        "chunk": chunk,
                    }
                )
        for chunk in full["af_plan"]:
            for variant in VARIANTS:
                result.append(
                    {
                        "profile": profile,
                        "length": length,
                        "kind": "af",
                        "variant": variant,
                        "chunk": chunk,
                    }
                )
    for task, value in sorted(catalog["probe_tasks"].items()):
        manifest = (
            classification if value["input_manifest"] == "classification" else full
        )
        for chunk in manifest["plan"]:
            if chunk["task"] == task:
                result.append(
                    {
                        "profile": "frozen-probes",
                        "variant": "ar-lower-features",
                        "length": 8192,
                        "kind": "features",
                        "chunk": chunk,
                    }
                )
    identities = [
        (job["profile"], job["variant"], job["kind"], job["chunk"]["key"])
        for job in result
    ]
    if len(set(identities)) != len(result):
        raise ValueError("Evaluation plan contains duplicate output identities")
    return result


def stage(catalog_record: dict[str, Any], root: Path) -> None:
    """Fetch hash-bound original inputs once per node before any inference."""
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = fetch(catalog_record, root / "catalog")
    catalog = json.loads(catalog_path.read_text())
    manifests = {}
    for name, field in (
        ("full", "full_requests"),
        ("classification", "classification_requests"),
    ):
        path = fetch(catalog[field], root / "manifests" / name)
        manifest = json.loads(path.read_text())
        manifest["samples"] = str((root / "samples").resolve())
        manifests[name] = manifest
        (root / (name + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    # Load the old zero-shot science unchanged. The later probe wheel supplies
    # only the pure visibility protocol to inference; new heads are refitted on
    # the unchanged original development/test memberships.
    for name, wheel_record in (
        ("scientific-code", catalog["scientific_wheel"]),
        ("probe-code", manifests["classification"]["evaluation_wheel"]),
    ):
        wheel = fetch(wheel_record, root / "wheels" / wheel_record["sha256"])
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root / name)
    records = {}
    for manifest in manifests.values():
        for value in manifest["inputs"].values():
            for record in value["files"]:
                identity = (record["path"], record["sha256"])
                records[identity] = record
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: materialize(value, root), records.values()))
    for task, value in catalog["probe_tasks"].items():
        path = fetch(value["membership"], root / "memberships" / task)
        shutil.copyfile(path, root / "memberships" / (task + ".parquet"))
    (root / "jobs.json").write_text(
        json.dumps(
            jobs(manifests["full"], manifests["classification"], catalog), indent=2
        )
        + "\n"
    )
    (root / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(
        json.dumps(
            {
                "staged": True,
                "input_files": len(records),
                "catalog_sha256": catalog_record["sha256"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-record", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    stage(json.loads(args.catalog_record.read_text()), args.root)


if __name__ == "__main__":
    main()
