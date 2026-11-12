"""Original staging with an explicit final-only lower-case work manifest."""

import argparse
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from exp593_evals.artifacts import fetch

# Reuse unchanged, already evaluated functions; only new readouts live here.
from exp593_evals.stage import materialize


def jobs(full: dict, classification: dict, catalog: dict) -> list[dict]:
    """One8K context; no uppercase/metadata variants or redundant single-base exact passes."""
    result = []
    for chunk in full["plan"]:
        variants = (
            ("ar-lower-control", "fim-minimal-control")
            if chunk["task"] == "sv_impact"
            else (
                "ar-lower-control",
                "ar-lower-exact",
                "fim-minimal-control",
                "fim-minimal-exact",
            )
        )
        for variant in variants:
            result.append(
                {
                    "profile": "context-8192",
                    "length": 8192,
                    "kind": "tasks",
                    "variant": variant,
                    "chunk": chunk,
                }
            )
    for chunk in full["af_plan"]:
        for variant in ("ar-lower-control", "fim-minimal-control"):
            result.append(
                {
                    "profile": "context-8192",
                    "length": 8192,
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
    priority = {
        "ar-lower-features": 0,
        "ar-lower-control": 1,
        "fim-minimal-control": 2,
        "ar-lower-exact": 3,
        "fim-minimal-exact": 4,
    }
    result.sort(key=lambda j: (priority[j["variant"]], j["kind"], j["chunk"]["key"]))
    identities = [
        (j["profile"], j["variant"], j["kind"], j["chunk"]["key"]) for j in result
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate annealing work identity")
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
