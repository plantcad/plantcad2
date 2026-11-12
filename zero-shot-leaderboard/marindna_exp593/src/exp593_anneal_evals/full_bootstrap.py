"""Finite, disjoint full-evaluation workers on one owned eight-H100 node."""

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

from exp593_evals.artifacts import fetch
from exp593_evals.bootstrap import restore_provider
from exp593_evals.storage import client, upload_verified


def run(args: argparse.Namespace) -> None:
    import wandb

    source, root = args.source, args.root
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "uv==0.13.5"],
        check=True,
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    manifest = json.loads((source / "source-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        with (source / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Staged source differs: {relative}")
    binding = json.loads((source / "binding.json").read_text())
    review = binding["native_gate"]
    if (
        review.get("accepted") is not True
        or review["checkpoint"] != binding["checkpoint"]
    ):
        raise ValueError("A checkpoint-specific reviewed validation is required")
    storage = client()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source / "src")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        environment[key] = "1"
    environment["TORCHINDUCTOR_FREEZING"] = "1"
    wb = wandb.init(
        entity="eric-czech",
        project="marin",
        id=args.run_id,
        name=args.run_id,
        group="dna-exp593-evals",
        resume="allow",
        config={
            "checkpoint": binding["checkpoint"],
            "configuration": "fast-fa3",
            "source": manifest,
            "logical_workers": args.workers,
            "worker_offset": args.offset,
            "profile": "full-vocabulary input; lowercase-only a/c/g/t predictions; FP32 historical AF readout",
        },
    )
    processes = []
    cpu_processes = []
    error = None
    artifact_root = args.artifact_root.rstrip("/")
    execution_owner = artifact_root + f"/execution/{args.run_id}"
    owner = execution_owner + "/attempts/" + uuid4().hex
    try:
        wb.log({"run_progress": 0.0, "eval/phase": "setup"})
        provider = restore_provider(source, root)
        model = root / "model"
        model.mkdir(exist_ok=True)
        for record in json.loads((source / "export-manifest.json").read_text()):
            fetch(record, model)
        native_receipt = json.loads((model / "native-receipt.json").read_text())
        if native_receipt["checkpoint"] != binding["checkpoint"]:
            raise ValueError("Export and reviewed checkpoint differ")
        subprocess.run(
            [
                str(provider),
                "-m",
                "exp593_anneal_evals.stage",
                "--catalog-record",
                str(source / "catalog-record.json"),
                "--root",
                str(root / "inputs"),
            ],
            env=environment,
            check=True,
        )
        output = root / "outputs"
        output.mkdir(exist_ok=True)
        jobs = json.loads((root / "inputs/jobs.json").read_text())
        assigned = sum(
            args.offset <= index % args.workers < args.offset + 8
            for index in range(len(jobs))
        )
        for gpu in range(8):
            worker = args.offset + gpu
            log = (root / f"worker-{worker:03d}.log").open("w")
            env = {**environment, "CUDA_VISIBLE_DEVICES": str(gpu)}
            process = subprocess.Popen(
                [
                    str(provider),
                    "-m",
                    "exp593_anneal_evals.worker",
                    "--inputs",
                    str(root / "inputs"),
                    "--model-dir",
                    str(model),
                    "--binding",
                    str(source / "binding.json"),
                    "--output",
                    str(output),
                    "--artifact-root",
                    artifact_root,
                    "--worker",
                    str(worker),
                    "--workers",
                    str(args.workers),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((worker, process, log))
        if args.offset == 0:
            # All phases are finite and belong to this checkpoint. The foreground
            # parent owns them through completion; no idle future-checkpoint lease.
            for phase in [
                "tasks8192",
                "af8192",
                "af-probe8192",
                "probes-p0",
                "probes-p1",
                "probes-p2",
                "probes-p3",
            ]:
                log = (root / ("cpu-" + phase + ".log")).open("w")
                env = {
                    **environment,
                    "CUDA_VISIBLE_DEVICES": "",
                    "EXP593_INPUTS_ROOT": str(root / "inputs"),
                    "EXP593_CPU_ENV": str(root / "shared-cpu"),
                }
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "exp593_anneal_evals.analysis_bootstrap",
                        "--source",
                        str(source),
                        "--root",
                        str(root / ("cpu-" + phase)),
                        "--phase",
                        phase,
                        "--run-id",
                        args.run_id + "-" + phase,
                        "--artifact-root",
                        args.artifact_root,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                cpu_processes.append((phase, process, log))
        while any(p.poll() is None for _, p, _ in processes):
            statuses = []
            for path in output.glob("worker-*.json"):
                try:
                    statuses.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    pass
            failed = {
                w: p.returncode for w, p, _ in processes if p.poll() not in (None, 0)
            }
            if failed:
                raise RuntimeError(f"Full evaluation worker failed: {failed}")
            cpu_failed = {
                phase: p.returncode
                for phase, p, _ in cpu_processes
                if p.poll() not in (None, 0)
            }
            if cpu_failed:
                raise RuntimeError(f"Finite CPU phase failed: {cpu_failed}")
            completed = sum(item["completed"] for item in statuses)
            wb.log(
                {
                    "run_progress": completed / assigned,
                    "eval/phase": "inference",
                    "eval/completed_chunks": completed,
                    "eval/assigned_chunks": assigned,
                    "eval/host_memory_current_bytes": int(
                        Path("/sys/fs/cgroup/memory.current").read_text()
                    ),
                }
            )
            print(
                json.dumps(
                    {"completed": completed, "assigned": assigned, "workers": statuses}
                ),
                flush=True,
            )
            time.sleep(30)
        for worker, process, _ in processes:
            status = json.loads((output / f"worker-{worker:03d}.json").read_text())
            if (
                process.returncode
                or status["phase"] != "complete"
                or status["completed"] != status["total"]
            ):
                raise ValueError(f"Worker {worker} did not finish its exact assignment")
        wb.log({"run_progress": 1.0, "eval/phase": "extraction-complete"})
        while any(p.poll() is None for _, p, _ in cpu_processes):
            failed = {
                phase: p.returncode
                for phase, p, _ in cpu_processes
                if p.poll() not in (None, 0)
            }
            if failed:
                raise RuntimeError(f"Finite CPU phase failed: {failed}")
            wb.log(
                {
                    "run_progress": 1.0,
                    "eval/phase": "finite-cpu-analysis",
                    "eval/cpu_phases_complete": sum(
                        p.poll() == 0 for _, p, _ in cpu_processes
                    ),
                    "eval/cpu_phases_total": len(cpu_processes),
                }
            )
            time.sleep(30)
        if any(p.returncode for _, p, _ in cpu_processes):
            raise RuntimeError("A finite CPU phase did not succeed")
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        for _, process, log in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
        for _, process, log in cpu_processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
        records = []
        for pattern in (
            "worker-*.log",
            "cpu-*.log",
            "outputs/worker-*.json",
            "provider-environment.json",
        ):
            for path in sorted(root.glob(pattern)):
                records.append(upload_verified(storage, path, owner + "/" + path.name))
        receipt = root / "extraction-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "error": error,
                    "binding": binding,
                    "source": manifest,
                    "worker_offset": args.offset,
                    "workers": args.workers,
                    "artifacts": records,
                    "finished_epoch": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        upload_verified(storage, receipt, owner + "/extraction-receipt.json")
        if error is None:
            upload_verified(
                storage, receipt, execution_owner + "/extraction-receipt.json"
            )
        wb.finish(exit_code=1 if error else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "artifact-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.offset % 8 or args.workers % 8 or not 0 <= args.offset <= args.workers - 8:
        raise ValueError("Each node owns eight distinct logical workers")
    run(args)


if __name__ == "__main__":
    main()
