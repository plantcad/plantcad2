"""Read previously computed dense metrics; never invoke a model or fit a probe."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def verify_exp582_results(root: Path) -> dict:
    """Verify the copied exp582 comparison artifacts without recomputing them."""
    archive = root / "historical" / "exp582-final"
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported exp582 historical manifest")
    for relative, record in manifest["files"].items():
        path = archive / relative
        payload = path.read_bytes()
        if len(payload) != record["bytes"]:
            raise ValueError(f"Historical exp582 artifact size changed: {path}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Historical exp582 artifact changed: {path}")
    return manifest


def dense_results(root: Path) -> dict:
    """Verify provenance and assemble the three historical token budgets."""
    archive = root / "historical"
    for manifest_name in ("manifest.json", "hf-manifest.json"):
        manifest = json.loads((archive / manifest_name).read_text())
        records = manifest["files"] if isinstance(manifest, dict) else manifest
        for record in records:
            path = root / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Historical artifact changed: {path}")

    full = json.loads(
        (archive / "exp472-lr5e4-full-coreweave-20260831.json").read_text()
    )
    catalog = json.loads((root / "task_catalog.json").read_text())
    expected_keys = {task["key"] for task in catalog["tasks"]}
    output = {}
    for rung, result in full["results"].items():
        rows = result["rows"]
        if {row["key"] for row in rows} != expected_keys or len(rows) != 20:
            raise ValueError("Historical task inventory differs")
        if sum(row["samples"] for row in rows) != catalog["full_rows"]:
            raise ValueError("Historical task row count differs")
        if result["sampling"]["dataset_revision"] != catalog["revision"]:
            raise ValueError("Historical task dataset revision differs")
        # Reassemble only saved scalar metrics, without rescoring any dense predictions.
        groups = {
            category: mean(
                row["selected_value"] for row in rows if row["category"] == category
            )
            for category in {row["category"] for row in rows}
        }
        if abs(mean(groups.values()) - full["groups"][rung][-1]) > 1e-14:
            raise ValueError("Historical aggregate does not reproduce its saved value")
        af = json.loads(
            (archive / f"plantcad2-maize-af-{rung}-20260904-v1.json").read_text()
        )
        probe = json.loads(
            (archive / f"plantcad2-maize-af-probe-{rung}-20260904-v1.json").read_text()
        )
        if (
            probe["split"]["sha256"]
            != "c37e713eb0449af48f99a14de5e3c86fab8cb5504c4872ccfd24e8988f0b89a0"
        ):
            raise ValueError("Historical probe split differs")
        output[rung] = {
            "label": full["labels"][rung],
            "tokens": result["checkpoint"]["tokens"],
            "tasks": rows,
            "groups": groups,
            "twenty_task_composite": full["groups"][rung][-1],
            "allele_frequency": af["metrics"],
            "probes": probe["metrics"],
            "probe_split": probe["split"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Directory containing historical/ and task_catalog.json"
    )
    args = parser.parse_args()
    print(json.dumps(dense_results(args.root), indent=2))


if __name__ == "__main__":
    main()
