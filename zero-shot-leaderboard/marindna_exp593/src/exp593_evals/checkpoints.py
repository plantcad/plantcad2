"""Read-only native checkpoint discovery; no placement or launch decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exp593_evals import MAX_BATCH_TOKENS, OWNER, RUNS, STAGE_INPUT_TOKENS
from exp593_evals.storage import client, read_json


def select_milestones(
    records: list[dict[str, Any]], run_id: str
) -> list[dict[str, Any]]:
    """Select only committed permanent saves at the actual quarter thresholds."""
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        metadata = record["metadata"]
        if metadata.get("is_temporary") is not False:
            continue
        bindings = metadata.get("exp593_bindings", {})
        clock = metadata.get("exp593_clock", {})
        if metadata.get("exp593_schema") != 1 or bindings.get("run_id") != run_id:
            raise ValueError("Checkpoint scientific/run identity differs")
        if bindings.get("schedule", {}).get("training_tokens") != STAGE_INPUT_TOKENS:
            raise ValueError("Checkpoint token horizon differs")
        step, count = clock.get("updates"), clock.get("input_tokens")
        if (
            not isinstance(step, int)
            or not isinstance(count, int)
            or metadata.get("step") != step
        ):
            raise ValueError("Checkpoint clock/step differs")
        if clock.get("examples") != step * 64:
            raise ValueError("Checkpoint example cursor differs from global batch")
        for quarter in range(1, 5):
            target = STAGE_INPUT_TOKENS * quarter // 4
            if target <= count < target + MAX_BATCH_TOKENS:
                candidate = {
                    "quarter": quarter,
                    "run_id": run_id,
                    "step": step,
                    "input_tokens": count,
                    "target_tokens": target,
                    "checkpoint": record["checkpoint"],
                    "clock": clock,
                    "metadata_sha256": record["metadata_sha256"],
                    "metadata_canonical_sha256": hashlib.sha256(
                        json.dumps(
                            metadata, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "bindings": bindings,
                }
                if (
                    quarter in selected
                    and selected[quarter]["checkpoint"] != candidate["checkpoint"]
                ):
                    raise ValueError("Multiple permanent saves claim one milestone")
                selected[quarter] = candidate
    return [selected[q] for q in sorted(selected)]


def observe(storage: Any) -> dict[str, Any]:
    result = []
    for condition, run_id in RUNS.items():
        uri = OWNER + "training/" + run_id + "/checkpoints/"
        authoritative, training_payload = read_json(storage, uri + "training.json")
        bucket, prefix = uri[5:].split("/", 1)
        records = []
        for page in storage.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for item in page.get("Contents", []):
                relative = item["Key"][len(prefix) :]
                if (
                    not relative.startswith("step-")
                    or relative.count("/") != 1
                    or not relative.endswith("/metadata.json")
                ):
                    continue
                metadata, payload = read_json(
                    storage, "s3://" + bucket + "/" + item["Key"]
                )
                if metadata.get("exp593_bindings") != authoritative:
                    raise ValueError(
                        "Checkpoint bindings differ from authoritative training.json"
                    )
                if relative.split("/", 1)[0] != "step-" + str(metadata.get("step")):
                    raise ValueError("Checkpoint directory and metadata step differ")
                records.append(
                    {
                        "checkpoint": "s3://"
                        + bucket
                        + "/"
                        + item["Key"].removesuffix("/metadata.json"),
                        "metadata": metadata,
                        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        result.append(
            {
                "condition": condition,
                "run_id": run_id,
                "training_json_sha256": hashlib.sha256(training_payload).hexdigest(),
                "permanent_count": sum(
                    r["metadata"].get("is_temporary") is False for r in records
                ),
                "milestones": select_milestones(records, run_id),
            }
        )
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "models": result,
        "verification": "Native metadata commit markers, identities and clocks; export must separately validate all arrays",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-credentials", action="store_true")
    args = parser.parse_args()
    result = observe(client(refresh=args.refresh_credentials))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "observed_at": result["observed_at"],
                "milestones": [
                    {
                        "condition": m["condition"],
                        "steps": [c["step"] for c in m["milestones"]],
                    }
                    for m in result["models"]
                ],
            }
        )
    )
