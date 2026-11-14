import hashlib
import io
import json

import pytest

from exp593_evals.analysis_resume import recovered_jobs, scientific_files


def fixture(tmp_path, *, probe=False):
    binding = {"checkpoint": "model-a", "catalog": "frozen-inputs"}
    manifest = {
        "files": {
            "src/exp593_evals/analysis.py": "unchanged-science",
            "binding.json": "binding-hash",
        }
    }
    (tmp_path / "binding.json").write_text(json.dumps(binding))
    (tmp_path / "source-manifest.json").write_text(json.dumps(manifest))
    owner = "s3://owner/checkpoints/model-a/v1"
    job = {
        "kind": "probe" if probe else "tasks",
        "profile": "frozen-probes" if probe else "context-2048",
        "variant": "ar-lower-features" if probe else "ar-lower-control",
    }
    if probe:
        job["task"] = "conservation_poaceae_tis"
    name = "/".join(
        [job["profile"], job["variant"], job["kind"]] + ([job["task"]] if probe else [])
    )
    phase = "probes-p2" if probe else "tasks2048"
    prefix = owner + "/analysis/" + phase + "/attempts/original/results/" + name
    result = b'{"score": 0.75}'
    artifact = {
        "uri": prefix + "/result.json",
        "sha256": hashlib.sha256(result).hexdigest(),
        "bytes": len(result),
    }
    receipt = {
        "binding": binding,
        **job,
        "profile": "context-8192" if probe else job["profile"],
        "task": job.get("task"),
        "artifacts": [artifact],
    }
    raw = json.dumps(receipt).encode()
    record = {
        "uri": prefix + "/analysis-receipt.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    plan = {
        "binding": binding,
        "scientific_files": scientific_files(manifest),
        "phases": {phase: [{"job": job, "receipt": record}]},
    }
    objects = {record["uri"]: raw, artifact["uri"]: result}

    class Storage:
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(objects["s3://" + Bucket + "/" + Key])}

    return Storage(), objects, job, record, receipt, plan, phase, owner


def run(tmp_path, values):
    storage, _, job, _, _, plan, phase, owner = values
    (tmp_path / "recovery-plan.json").write_text(json.dumps(plan))
    return recovered_jobs(tmp_path, phase, [job], owner, storage)


@pytest.mark.parametrize("probe", [False, True])
def test_finished_job_keeps_its_original_receipt_and_probe_profile(tmp_path, probe):
    values = fixture(tmp_path, probe=probe)
    result = run(tmp_path, values)
    assert len(result) == 1
    saved = next(iter(result.values()))
    assert saved["reused"] and saved["receipt"] == values[3]["uri"]


def test_no_plan_keeps_the_original_execution_path(tmp_path):
    assert recovered_jobs(tmp_path, "tasks2048", [], "s3://owner", None) == {}


@pytest.mark.parametrize(
    "change", ["source", "binding", "artifact", "receipt", "duplicate", "phase"]
)
def test_mixed_or_corrupted_recovery_is_rejected(tmp_path, change):
    values = fixture(tmp_path)
    _, objects, job, record, receipt, plan, phase, _ = values
    if change == "source":
        plan["scientific_files"]["src/exp593_evals/analysis.py"] = "different-science"
    elif change == "binding":
        receipt["binding"] = {"checkpoint": "other-model"}
        raw = json.dumps(receipt).encode()
        objects[record["uri"]] = raw
        record.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    elif change == "artifact":
        objects[receipt["artifacts"][0]["uri"]] = b"tampered"
    elif change == "receipt":
        record["sha256"] = "wrong-digest"
    elif change == "duplicate":
        plan["phases"][phase].append({"job": job, "receipt": record})
    else:
        record["uri"] = record["uri"].replace(
            "/analysis/tasks2048/", "/analysis/af2048/"
        )
    with pytest.raises(ValueError):
        run(tmp_path, values)
