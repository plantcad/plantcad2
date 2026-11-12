import hashlib
import json

import pytest
from botocore.exceptions import ClientError

from exp593_anneal_evals import OWNER, TRIALS
from exp593_anneal_evals import observe as module


@pytest.mark.parametrize(
    "state,progress,audited,ready",
    [
        ("running", 0.99, True, False),
        ("finished", 0.99, True, False),
        ("finished", 1.0, False, False),
        ("finished", 1.0, True, True),
    ],
)
def test_only_finished_exact_final_with_matching_full_state_audit_is_ready(
    monkeypatch, state, progress, audited, ready
):
    objects = {}
    runs = {}
    for i, trial in enumerate(TRIALS.values()):
        root = OWNER + f"annealing/training/{trial.run_id}/checkpoints/"
        checkpoint = root + "step-123"
        metadata = {
            "step": 123,
            "is_temporary": False,
            "exp593_bindings": {"run_id": trial.run_id, "validation_updates": None},
            "exp593_anneal_clock": {"stage": {"input_tokens": trial.stage_tokens}},
        }
        blob = json.dumps(metadata).encode()
        digest = hashlib.sha256(blob).hexdigest()
        objects[root + "latest.json"] = {
            "checkpoint": checkpoint,
            "metadata_sha256": digest,
        }
        objects[checkpoint + "/metadata.json"] = metadata
        audit = OWNER + f"annealing/native_audits/{trial.run_id}/step-123-{digest}.json"
        if audited:
            objects[audit] = {
                "final": True,
                "checkpoint": checkpoint,
                "metadata_sha256": digest,
                "native_state_array_leaves": 128,
                "all_arrays_read_and_finite": True,
                "compute_matches_fp32_master_cast": True,
                "optimizer_count_matches_global_step": True,
            }
        runs[f"r{i}"] = {
            "state": state,
            "summaryMetrics": json.dumps(
                {
                    "run_progress": progress,
                    "train/input_tokens": trial.stage_tokens,
                }
            ),
        }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"project": runs}}

    def read(_storage, uri):
        if uri not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return objects[uri], json.dumps(objects[uri]).encode()

    monkeypatch.setattr(module.requests, "post", lambda *a, **kw: Response())
    monkeypatch.setattr(module, "read_json", read)
    result = module.observe(None, "fake")
    assert len(result["models"]) == 4
    assert all(row["ready_for_export"] is ready for row in result["models"])
