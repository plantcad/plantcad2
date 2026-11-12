"""Read-only completion checks; this module cannot submit or cancel jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import requests
from botocore.exceptions import ClientError

from exp593_anneal_evals import OWNER, TRIALS
from exp593_evals.storage import client, read_json


def observe(storage, api_key: str) -> dict:
    fields = " ".join(
        f"r{i}:run(name:{json.dumps(t.run_id)}){{name state summaryMetrics}}"
        for i, t in enumerate(TRIALS.values())
    )
    response = requests.post(
        "https://forge.coreweave.com/api/wandb/graphql",
        auth=("api", api_key),
        json={
            "query": 'query{project(name:"marin",entityName:"eric-czech"){'
            + fields
            + "}}"
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise ValueError(payload["errors"])
    models = []
    for i, trial in enumerate(TRIALS.values()):
        run = payload["data"]["project"][f"r{i}"]
        if run is None:
            raise ValueError(f"Training run unavailable: {trial.run_id}")
        summary = run["summaryMetrics"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        root = OWNER + f"annealing/training/{trial.run_id}/checkpoints/"
        pointer, _ = read_json(storage, root + "latest.json")
        checkpoint = pointer["checkpoint"]
        if not checkpoint.startswith(root + "step-"):
            raise ValueError("Recovery pointer escaped the selected run")
        metadata, raw = read_json(storage, checkpoint + "/metadata.json")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != pointer["metadata_sha256"]:
            raise ValueError("Recovery pointer metadata mismatch")
        bindings = metadata["exp593_bindings"]
        clock = metadata["exp593_anneal_clock"]
        if (
            bindings["run_id"] != trial.run_id
            or bindings["validation_updates"] is not None
        ):
            raise ValueError("Checkpoint belongs to another run or a smoke test")
        saved_tokens = clock["stage"]["input_tokens"]
        final = (
            metadata.get("is_temporary") is False and saved_tokens == trial.stage_tokens
        )
        training_complete = (
            run["state"] == "finished"
            and summary.get("run_progress", 0) >= 1
            and summary.get("train/input_tokens") == trial.stage_tokens
            and final
        )
        report_uri = (
            OWNER
            + f"annealing/native_audits/{trial.run_id}/step-{metadata['step']}-{digest}.json"
        )
        audit = None
        try:
            audit, _ = read_json(storage, report_uri)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise
        audit_complete = (
            audit is not None
            and audit.get("final") is True
            and audit.get("checkpoint") == checkpoint
            and audit.get("metadata_sha256") == digest
            and audit.get("native_state_array_leaves") == 128
            and all(
                audit.get(k) is True
                for k in [
                    "all_arrays_read_and_finite",
                    "compute_matches_fp32_master_cast",
                    "optimizer_count_matches_global_step",
                ]
            )
        )
        row = {
            "key": trial.key,
            "run_id": trial.run_id,
            "label": trial.label,
            "wandb_state": run["state"],
            "run_progress": summary.get("run_progress"),
            "training_input_tokens": summary.get("train/input_tokens"),
            "training_phase": summary.get("training/phase"),
            "last_logged_epoch": summary.get("_timestamp"),
            "stage_horizon": trial.stage_tokens,
            "training_complete": training_complete,
            "native_final_audit": audit_complete,
            "native_final_audit_uri": report_uri if audit_complete else None,
            "ready_for_export": training_complete and audit_complete,
            "checkpoint_record": {
                "run_id": trial.run_id,
                "condition": trial.key,
                "final": final,
                "checkpoint": checkpoint,
                "step": metadata["step"],
                "input_tokens": saved_tokens,
                "clock": clock,
                "metadata_sha256": digest,
                "metadata_canonical_sha256": hashlib.sha256(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "bindings": bindings,
            },
        }
        models.append(row)
    return {"observed_at": datetime.now(UTC).isoformat(), "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-credentials", action="store_true")
    args = parser.parse_args()
    value = observe(
        client(refresh=args.refresh_credentials), os.environ["WANDB_API_KEY"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "observed_at": value["observed_at"],
                "models": [
                    {
                        k: r[k]
                        for k in [
                            "key",
                            "wandb_state",
                            "run_progress",
                            "training_input_tokens",
                            "training_complete",
                            "native_final_audit",
                            "ready_for_export",
                        ]
                    }
                    for r in value["models"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
