"""One finite batch H100 node: real-task native and three-backend comparison."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from exp593_evals.bootstrap import restore_provider
from exp593_evals.storage import client, upload_verified


def run(a):
    import wandb

    source, root = a.source, a.root
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source / "source-manifest.json").read_text())
    for name, sha in source_manifest["files"].items():
        with (source / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != sha:
                raise ValueError("Source manifest differs")
    os.environ["PYTHONPATH"] = str(source / "src")
    os.environ["EXP593_S3_ENDPOINT"] = "http://cwlota.com"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        os.environ[key] = "1"
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "uv==0.13.5"],
        check=True,
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    s = client()
    wb = wandb.init(
        entity="eric-czech",
        project="marin",
        id=a.run_id,
        name=a.run_id,
        resume="allow",
        allow_val_change=True,
        group="dna-exp593-plantcad2-evals",
        config={
            "scope": "finite practical backend validation, not final evaluation",
            "source": source_manifest,
            "profiles": ["native", "torch-fa3", "vllm-fa3", "fast-fa3"],
            "priority": "batch",
            "user": "eczech",
        },
        settings=wandb.Settings(base_url="https://forge.coreweave.com/api/wandb"),
    )
    processes = []
    error = None
    attempt_owner = a.artifact_root.rstrip("/") + "/attempts/" + uuid4().hex
    from botocore.exceptions import ClientError

    final_bucket, final_key = (
        (a.artifact_root.rstrip("/") + "/comparison-receipt.json")
        .removeprefix("s3://")
        .split("/", 1)
    )
    try:
        previous = json.loads(
            s.get_object(Bucket=final_bucket, Key=final_key)["Body"].read()
        )
    except ClientError as e:
        if e.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
            raise
    else:
        if previous["error"] is not None or previous["source"] != source_manifest:
            raise ValueError(
                "A completed comparison belongs to different source semantics"
            )
        wb.log(
            {
                "run_progress": 1.0,
                "validation/phase": "complete-reused-verified-receipt",
            }
        )
        wb.finish()
        return
    try:
        wb.log({"run_progress": 0.0, "validation/phase": "setup"})
        wb.log({"validation/attempt_owner": attempt_owner})
        panel_record = json.loads((source / "panel-record.json").read_text())
        uri = panel_record["uri"]
        bucket, key = uri.removeprefix("s3://").split("/", 1)
        packed = root / "panel.json.gz"
        s.download_file(bucket, key, str(packed))
        with packed.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != panel_record["sha256"]:
                raise ValueError("Panel archive differs")
        with gzip.open(packed, "rb") as f, (root / "panel.json").open("wb") as out:
            import shutil

            shutil.copyfileobj(f, out)
        from exp593_evals import panel_stream

        streamed = panel_stream.prepare(root / "panel.json", root / "panel.cases.jsonl")
        if streamed["source_sha256"] != panel_record["uncompressed_sha256"]:
            raise ValueError("Uncompressed panel differs from its source receipt")
        # Reuse the fully verified first export, without reusing its failed decision.
        old = a.previous_export.rstrip("/")
        bucket, key = (
            (old + "/export-manifest.json").removeprefix("s3://").split("/", 1)
        )
        export = json.loads(s.get_object(Bucket=bucket, Key=key)["Body"].read())
        model = root / "model"
        model.mkdir(exist_ok=True)
        for record in export:
            uri = record["uri"]
            b, k = uri.removeprefix("s3://").split("/", 1)
            if k.rsplit("/", 1)[-1] == "native-reference.npz":
                continue
            path = model / k.rsplit("/", 1)[-1]
            s.download_file(b, k, str(path))
            with path.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError("Export differs")
        native_env = os.environ.copy()
        native_env.pop("VIRTUAL_ENV", None)
        native_env["UV_PROJECT_ENVIRONMENT"] = str(root / "native-runtime")
        native_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(source / "native"),
                "--python",
                "3.12.13",
                "--locked",
                "--extra",
                "gpu",
            ],
            env=native_env,
            check=True,
        )
        provider = restore_provider(source, root / "vllm-runtime")
        torch_provider = restore_provider(source, root / "torch-runtime", "no-vllm")
        output = root / "predictions"
        output.mkdir(exist_ok=True)
        for stale in output.glob("*"):
            if stale.is_file():
                stale.unlink()
        configurations = [
            ("native", i, 5, i, root / "native-runtime/bin/python", native_env)
            for i in range(5)
        ]
        configurations += [
            ("fast-fa3", 0, 1, 5, provider, os.environ.copy()),
            ("vllm-fa3", 0, 1, 6, provider, os.environ.copy()),
            ("torch-fa3", 0, 1, 7, torch_provider, os.environ.copy()),
        ]
        for backend, worker, workers, gpu, python, env in configurations:
            env = env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if backend == "fast-fa3":
                env["TORCHINDUCTOR_FREEZING"] = "1"
            else:
                env.pop("TORCHINDUCTOR_FREEZING", None)
            log = (root / f"{backend}-{worker}.log").open("w")
            argv = [
                str(python),
                "-m",
                "exp593_evals.panel_runtime",
                "--panel",
                str(root / "panel.json"),
                "--record",
                str(source / "checkpoint-record.json"),
                "--case-stream",
                str(root / "panel.cases.jsonl"),
                "--model",
                str(model),
                "--output",
                str(output),
                "--backend",
                backend,
                "--worker",
                str(worker),
                "--workers",
                str(workers),
            ]
            p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((backend, worker, p, log))
        while any(p.poll() is None for _, _, p, _ in processes):
            progress = []
            for path in output.glob("*-progress.json"):
                try:
                    progress.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    pass
            fraction = sum(x["completed"] for x in progress) / (4 * streamed["cases"])
            wb.log(
                {
                    "run_progress": fraction,
                    "validation/phase": "task-panel",
                    "validation/finished_processes": sum(
                        p.poll() is not None for _, _, p, _ in processes
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "progress": progress,
                        "processes": {f"{b}-{w}": p.poll() for b, w, p, _ in processes},
                    }
                ),
                flush=True,
            )
            time.sleep(30)
        failures = {
            f"{b}-{w}": p.returncode for b, w, p, _ in processes if p.returncode
        }
        if failures:
            raise RuntimeError(f"Finite comparison processes failed: {failures}")
        wb.log(
            {
                "run_progress": 1.0,
                "validation/phase": "predictions-complete-awaiting-analysis",
            }
        )
    except BaseException as e:
        error = repr(e)
        raise
    finally:
        for _, _, p, log in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
            log.close()
        records = []
        for pattern in (
            "*.log",
            "*-runtime/provider-environment.json",
            "predictions/*.npz",
            "predictions/*.json",
        ):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    records.append(
                        upload_verified(
                            s,
                            path,
                            attempt_owner + "/" + str(path.relative_to(root)),
                        )
                    )
        receipt = root / "comparison-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "error": error,
                    "source": source_manifest,
                    "artifacts": records,
                    "finished_epoch": time.time(),
                    "status": "failed"
                    if error
                    else "predictions-complete-needs-scientific-review",
                    "attempt_owner": attempt_owner,
                    "host_memory_peak_bytes": int(
                        Path("/sys/fs/cgroup/memory.peak").read_text()
                    )
                    if Path("/sys/fs/cgroup/memory.peak").exists()
                    else None,
                },
                indent=2,
            )
            + "\n"
        )
        upload_verified(s, receipt, attempt_owner + "/comparison-receipt.json")
        if error is None:
            upload_verified(
                s, receipt, a.artifact_root.rstrip("/") + "/comparison-receipt.json"
            )
        wb.finish(exit_code=1 if error else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["source", "root"]:
        p.add_argument("--" + n, type=Path, required=True)
    for n in ["run-id", "artifact-root", "previous-export"]:
        p.add_argument("--" + n, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
