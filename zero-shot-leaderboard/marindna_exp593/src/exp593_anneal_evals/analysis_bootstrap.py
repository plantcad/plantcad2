"""Finite CPU reduction and new-head fitting over completed GPU predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from exp593_anneal_evals import VARIANTS

# Reuse unchanged, already evaluated functions; only new readouts live here.
from exp593_evals.analysis_bootstrap import environment, prior_af_predictions
from exp593_evals.analysis_resume import job_name, recovered_jobs
from exp593_evals.storage import client, upload_verified

PHASES = (
    "tasks8192",
    "tasks2048",
    "af8192",
    "af2048",
    "probes-p0",
    "probes-p1",
    "probes-p2",
    "probes-p3",
    "af-probe8192",
    "af-probe2048",
    "all",
)


def jobs(phase: str, catalog: dict) -> list[dict]:
    """Finite8K tasks; fitting protocol and development/test memberships unchanged."""
    result = []
    if phase in {"tasks8192", "all"}:
        result.extend(
            {"kind": "tasks", "profile": "context-8192", "variant": v} for v in VARIANTS
        )
    if phase in {"af8192", "all"}:
        result.extend(
            {"kind": "af", "profile": "context-8192", "variant": v}
            for v in ("ar-lower-control", "fim-minimal-control")
        )
    if phase == "all" or phase.startswith("probes-p"):
        tasks = sorted(catalog["probe_tasks"])
        if phase.startswith("probes-p"):
            tasks = tasks[int(phase.removeprefix("probes-p")) :: 4]
        result.extend(
            {
                "kind": "probe",
                "profile": "frozen-probes",
                "variant": "ar-lower-features",
                "task": t,
            }
            for t in tasks
        )
    if phase in {"af-probe8192", "all"}:
        result.append(
            {
                "kind": "af-probe",
                "profile": "context-8192",
                "variant": "ar-lower-features",
            }
        )
    return result


def shared_environment(source, root):
    import fcntl

    destination = Path(os.environ.get("EXP593_CPU_ENV", str(root / "cpu")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(
        (source / "provenance/analysis-cpu-packages.json").read_bytes()
    ).hexdigest()
    with destination.with_suffix(".lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        marker = destination / "source-binding.json"
        if marker.exists():
            if (
                json.loads(marker.read_text())["package_manifest_sha256"] != expected
                or not (destination / "bin/python").exists()
            ):
                raise ValueError(
                    "Shared CPU environment belongs to another package pin"
                )
            return destination / "bin/python"
        python = environment(source, destination)
        marker.write_text(json.dumps({"package_manifest_sha256": expected}) + "\n")
        return python


def wait_for_inputs(job, inputs, artifact_root, run):
    catalog = json.loads((inputs / "catalog.json").read_text())
    if job["kind"] == "probe":
        entry = catalog["probe_tasks"][job["task"]]
        manifest = json.loads(
            (inputs / (entry["input_manifest"] + ".json")).read_text()
        )
        plan = [p for p in manifest["plan"] if p["task"] == job["task"]]
        profile, variant, kind = "frozen-probes", "ar-lower-features", "features"
    else:
        manifest = json.loads((inputs / "full.json").read_text())
        plan = (
            manifest["af_plan"]
            if job["kind"] in {"af", "af-probe"}
            else manifest["plan"]
        )
        if job["kind"] == "tasks" and job["variant"].endswith("-exact"):
            plan = [p for p in plan if p["task"] != "sv_impact"]
        profile = job["profile"]
        variant = "ar-lower-features" if job["kind"] == "af-probe" else job["variant"]
        kind = "embeddings" if job["kind"] == "af-probe" else job["kind"]
    expected = {p["key"] for p in plan}
    uri = artifact_root.rstrip("/") + f"/{profile}/{variant}/{kind}/chunks/"
    bucket, prefix = uri.removeprefix("s3://").split("/", 1)
    storage = client()
    while True:
        actual = set()
        for page in storage.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            actual.update(
                o["Key"].rsplit("/", 1)[-1].removesuffix(".json")
                for o in page.get("Contents", [])
            )
        if expected <= actual:
            return
        run.log(
            {
                "eval/phase": "awaiting-finite-inputs",
                "eval/input_chunks_present": len(expected & actual),
                "eval/input_chunks_expected": len(expected),
            }
        )
        time.sleep(30)


def completed_phase(source, phase, work, recovered, owner, storage):
    """Preserve a pinned immutable phase only after verifying every saved job."""
    from exp593_evals.storage import read_json

    path = source / "recovery-plan.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text()).get("phase_receipts", {}).get(phase)
    if record is None:
        return None
    expected = {job_name(job): job for job in work}
    if (
        record["uri"]
        != owner.rstrip("/") + "/analysis/" + phase + "/analysis-receipt.json"
    ):
        raise ValueError("Completed phase is outside the selected owner")
    receipt, raw = read_json(storage, record["uri"])
    if (
        len(raw) != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("Completed phase receipt differs from its pin")
    if (
        receipt["phase"] != phase
        or receipt["error"] is not None
        or set(recovered) != set(expected)
    ):
        raise ValueError("Cannot preserve an incomplete or failed phase")
    saved = {job_name(row["job"]): row for row in receipt["finished"]}
    if len(saved) != len(receipt["finished"]) or set(saved) != set(expected):
        raise ValueError("Completed phase job coverage differs")
    for name, job in expected.items():
        if (
            saved[name]["job"] != job
            or saved[name]["receipt"] != recovered[name]["receipt"]
        ):
            raise ValueError("Completed phase uses another finalized job")
    return record


def run(args: argparse.Namespace) -> None:
    import wandb

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    source, root = args.source, args.root
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if (
            hashlib.file_digest((source / name).open("rb"), "sha256").hexdigest()
            != expected
        ):
            raise ValueError("CPU source manifest differs: " + name)
    attempt = uuid4().hex
    owner = (
        args.artifact_root.rstrip("/")
        + "/analysis/"
        + args.phase
        + "/attempts/"
        + attempt
    )
    run = wandb.init(
        entity="eric-czech",
        project="marin",
        id=args.run_id,
        name=args.run_id,
        group="dna-exp593-evals",
        resume="allow",
        config={
            "phase": args.phase,
            "cpu_only": True,
            "artifact_root": args.artifact_root,
            "source": manifest,
        },
    )
    result = {
        "attempt": attempt,
        "phase": args.phase,
        "source": manifest,
        "finished": [],
        "error": None,
    }
    try:
        run.log({"eval/phase": "cpu-environment", "run_progress": 0})
        python = shared_environment(source, root)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source / "src")
        inputs = Path(os.environ.get("EXP593_INPUTS_ROOT", str(root / "inputs")))
        if not os.environ.get("EXP593_INPUTS_ROOT"):
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "exp593_anneal_evals.stage",
                    "--catalog-record",
                    str(source / "catalog-record.json"),
                    "--root",
                    str(inputs),
                ],
                env=env,
                check=True,
            )
        work = jobs(args.phase, json.loads((inputs / "catalog.json").read_text()))
        recovered = recovered_jobs(
            source, args.phase, work, args.artifact_root, client()
        )
        previous = completed_phase(
            source, args.phase, work, recovered, args.artifact_root, client()
        )
        if previous is not None:
            result["finished"] = [recovered[job_name(job)] for job in work]
            result["preserved_phase_receipt"] = previous
            run.log(
                {"run_progress": 1, "eval/phase": "reused-verified-completed-phase"}
            )
            return
        for index, job in enumerate(work):
            name = job["profile"] + "/" + job["variant"] + "/" + job["kind"]
            if job.get("task"):
                name += "/" + job["task"]
            if name in recovered:
                result["finished"].append(recovered[name])
                run.log(
                    {
                        "eval/phase": "reused-verified-result",
                        "eval/current_task": name,
                        "eval/completed_jobs": index + 1,
                        "eval/reused_jobs": sum(
                            bool(r.get("reused")) for r in result["finished"]
                        ),
                        "eval/assigned_jobs": len(work),
                        "run_progress": (index + 1) / len(work),
                    }
                )
                continue
            wait_for_inputs(job, inputs, args.artifact_root, run)
            output = root / "results" / name
            cmd = [
                str(python),
                "-m",
                "exp593_anneal_evals.analysis",
                "--source",
                str(source),
                "--inputs",
                str(inputs),
                "--downloads",
                str(root / "downloads"),
                "--output",
                str(output),
                "--binding",
                str(source / "binding.json"),
                "--artifact-root",
                args.artifact_root,
                "--publish-owner",
                owner + "/results/" + name,
                "--jobs",
                "3",
                "--kind",
                job["kind"],
                "--profile",
                "context-8192" if job["profile"] == "frozen-probes" else job["profile"],
                "--variant",
                job["variant"],
            ]
            if job.get("task"):
                cmd += ["--task", job["task"]]
            if job["kind"] == "af-probe":
                local_af = (
                    root
                    / "results"
                    / job["profile"]
                    / "ar-lower-control/af/predictions.parquet"
                )
                if args.phase != "all":
                    while True:
                        try:
                            local_af = prior_af_predictions(
                                args.artifact_root,
                                job["profile"],
                                json.loads((source / "binding.json").read_text()),
                                root,
                            )
                            break
                        except Exception as error:
                            from botocore.exceptions import ClientError

                            if not isinstance(error, ClientError) or error.response[
                                "Error"
                            ]["Code"] not in {"404", "NoSuchKey"}:
                                raise
                            run.log({"eval/phase": "awaiting-fixed-AF-metrics"})
                            time.sleep(30)
                    # AF reference is now immutable and matching.
                    local_af = prior_af_predictions(
                        args.artifact_root,
                        job["profile"],
                        json.loads((source / "binding.json").read_text()),
                        root,
                    )
                cmd += [
                    "--af-zero-shot",
                    str(local_af),
                ]
            run.log(
                {
                    "eval/phase": "cpu-fit" if "probe" in job["kind"] else "metrics",
                    "eval/current_task": name,
                    "eval/completed_jobs": index,
                    "eval/assigned_jobs": len(work),
                    "run_progress": index / len(work),
                }
            )
            log = root / ("analysis-" + str(index) + ".log")
            with log.open("w") as stream:
                process = subprocess.Popen(
                    cmd, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                while process.poll() is None:
                    time.sleep(30)
                    run.log(
                        {
                            "eval/current_task": name,
                            "eval/log_bytes": log.stat().st_size,
                            "eval/completed_jobs": index,
                            "run_progress": index / len(work),
                        }
                    )
            if process.returncode:
                raise RuntimeError(
                    "Scientific reduction failed for "
                    + name
                    + ": "
                    + log.read_text()[-5000:]
                )
            result["finished"].append(
                {
                    "job": job,
                    "receipt": owner + "/results/" + name + "/analysis-receipt.json",
                }
            )
        run.log(
            {
                "eval/phase": "complete",
                "eval/completed_jobs": len(work),
                "eval/assigned_jobs": len(work),
                "run_progress": 1,
            }
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        storage = client()
        for log in sorted(root.glob("analysis-*.log")):
            upload_verified(storage, log, owner + "/logs/" + log.name)
        if (root / "cpu/environment.json").exists():
            result["environment"] = upload_verified(
                storage, root / "cpu/environment.json", owner + "/environment.json"
            )
        path = root / "analysis-receipt.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        upload_verified(storage, path, owner + "/analysis-receipt.json")
        if result["error"] is None and not result.get("preserved_phase_receipt"):
            upload_verified(
                storage,
                path,
                args.artifact_root.rstrip("/")
                + "/analysis/"
                + args.phase
                + "/analysis-receipt.json",
            )
        run.finish(exit_code=0 if result["error"] is None else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--phase",
        choices=PHASES,
        required=True,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
