"""Finite CPU reduction and new-head fitting over completed GPU predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from exp593_evals import VARIANTS
from exp593_evals.analysis_resume import recovered_jobs
from exp593_evals.artifacts import fetch
from exp593_evals.storage import client, read_json, upload_verified

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


def environment(source: Path, root: Path) -> Path:
    """Restore the original analysis versions in a separate CPU interpreter."""
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "uv==0.13.5"],
        check=True,
    )
    uv = str(Path(sys.executable).parent / "uv")
    subprocess.run([uv, "venv", "--python", "3.13.14", str(root)], check=True)
    python = root / "bin/python"
    packages = json.loads(
        (source / "provenance/analysis-cpu-packages.json").read_text()
    )
    excluded = {"marindna-exp586-evals", "mappy", "pytest", "ruff"}
    requirements = root / "requirements.txt"
    requirements.write_text(
        "\n".join(
            sorted(
                p["name"] + "==" + p["version"]
                for p in packages
                if p["name"] not in excluded
            )
        )
        + "\n"
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--prerelease",
            "allow",
            "--index-strategy",
            "unsafe-best-match",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "-r",
            str(requirements),
        ],
        check=True,
    )
    installed = json.loads(
        subprocess.check_output(
            [uv, "pip", "list", "--python", str(python), "--format", "json"], text=True
        )
    )
    versions = {p["name"]: p["version"] for p in installed}
    if any(
        versions.get(p["name"]) != p["version"]
        for p in packages
        if p["name"] not in excluded
    ):
        raise ValueError("CPU analysis environment differs from the original")
    (root / "environment.json").write_text(
        json.dumps(
            {
                "python": "3.13.14",
                "packages": installed,
                "excluded_unused": sorted(excluded),
            },
            indent=2,
        )
        + "\n"
    )
    return python


def jobs(phase: str, catalog: dict) -> list[dict]:
    """A fixed scientific work list, with no placement or recovery decisions."""
    result = []
    if phase in {"tasks8192", "all"}:
        result.extend(
            {"kind": "tasks", "profile": "context-8192", "variant": v} for v in VARIANTS
        )
    if phase in {"tasks2048", "all"}:
        result.extend(
            {"kind": "tasks", "profile": "context-2048", "variant": v} for v in VARIANTS
        )
    for profile, selected in (("context-8192", "af8192"), ("context-2048", "af2048")):
        if phase in {selected, "all"}:
            result.extend(
                {"kind": "af", "profile": profile, "variant": v} for v in VARIANTS
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
    if phase == "all" or phase.startswith("af-probe"):
        result.extend(
            {"kind": "af-probe", "profile": p, "variant": "ar-lower-features"}
            for p in ("context-8192", "context-2048")
            if phase == "all" or phase == "af-probe" + p.removeprefix("context-")
        )
    return result


def prior_af_predictions(
    artifact_root: str, profile: str, binding: dict, root: Path
) -> Path:
    """Reuse only the verified lowercase AF predictions from this checkpoint."""
    storage = client()
    parent_uri = (
        artifact_root.rstrip("/")
        + "/analysis/af"
        + profile.removeprefix("context-")
        + "/analysis-receipt.json"
    )
    parent, payload = read_json(storage, parent_uri)
    if parent["error"] is not None:
        raise ValueError("AF reduction has not completed successfully")
    expected = {"kind": "af", "profile": profile, "variant": "ar-lower-control"}
    selected = [r for r in parent["finished"] if r["job"] == expected]
    if len(selected) != 1:
        raise ValueError("No unique matching lowercase AF reduction")
    receipt, _ = read_json(storage, selected[0]["receipt"])
    if receipt["binding"] != binding or any(
        receipt[k] != v for k, v in expected.items()
    ):
        raise ValueError("AF predictions belong to a different checkpoint or readout")
    records = [
        r for r in receipt["artifacts"] if r["uri"].endswith("/predictions.parquet")
    ]
    if len(records) != 1:
        raise ValueError("No unique AF prediction artifact")
    destination = root / "prior-af" / profile
    path = fetch(records[0], destination)
    (destination / "provenance.json").write_text(
        json.dumps(
            {
                "phase_receipt": parent_uri,
                "phase_receipt_sha256": hashlib.sha256(payload).hexdigest(),
                "analysis_receipt": selected[0]["receipt"],
                "predictions": records[0],
                "checkpoint_binding": binding,
            },
            indent=2,
        )
        + "\n"
    )
    return path


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
        python = environment(source, root / "cpu")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source / "src")
        inputs = root / "inputs"
        subprocess.run(
            [
                str(python),
                "-m",
                "exp593_evals.stage",
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
            output = root / "results" / name
            cmd = [
                str(python),
                "-m",
                "exp593_evals.analysis",
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
        if result["error"] is None:
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
