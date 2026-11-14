"""Finite owned-node validation: restore providers, export, gate, upload, exit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Reuse unchanged, already evaluated functions; only new readouts live here.
from exp593_evals.bootstrap import restore_provider
from exp593_evals.storage import client, upload_verified


def run(args: argparse.Namespace) -> None:
    import wandb

    source, root = args.source, args.root
    root.mkdir(parents=True, exist_ok=True)
    # The base task image currently carries uv 0.10.3. The coherent native
    # dependency lock requires 0.13.5; keep that installer in tools, not providers.
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "uv==0.13.5"],
        check=True,
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    if not subprocess.check_output(["uv", "--version"], text=True).startswith(
        "uv 0.13.5 "
    ):
        raise RuntimeError("Pinned native installer is not selected")
    manifest = json.loads((source / "source-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        with (source / relative).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Staged source differs: {relative}")
    os.environ["EXP593_S3_ENDPOINT"] = "http://cwlota.com"
    os.environ["PYTHONPATH"] = str(source / "src")
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        os.environ[variable] = "1"
    record = json.loads((source / "checkpoint-record.json").read_text())
    wb = wandb.init(
        entity="eric-czech",
        project="marin",
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        allow_val_change=True,
        group="dna-exp593-plantcad2-evals",
        config={
            "checkpoint": record,
            "source": manifest,
            "scope": "finite export and readout validation",
            "backend": "fast-fa3",
            "priority": "batch",
            "user": "eczech",
        },
        settings=wandb.Settings(base_url="https://forge.coreweave.com/api/wandb"),
    )
    storage = client()
    receipts = []
    state = {"phase": "setup", "progress": 0.0, "error": None}
    log = root / "validation.log"
    lock = threading.Lock()

    def command(argv: list[str], phase: str, *, env: dict | None = None) -> None:
        state["phase"] = phase
        with subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                with lock, log.open("a") as stream:
                    stream.write(line)
            if process.wait() != 0:
                raise RuntimeError(f"Finite validation phase failed: {phase}")

    def progress() -> None:
        while not done.wait(30):
            wb.log(
                {"run_progress": state["progress"], "validation/phase": state["phase"]}
            )

    done = threading.Event()
    observer = threading.Thread(target=progress, daemon=True)
    observer.start()
    try:
        wb.log({"run_progress": 0.0, "validation/phase": "setup"})
        # Preserve a separate native environment; never install JAX over providers.
        native_environment = os.environ.copy()
        native_environment.pop("VIRTUAL_ENV", None)
        native_environment["UV_PROJECT_ENVIRONMENT"] = str(root / "native-runtime")
        native_environment["CUDA_VISIBLE_DEVICES"] = "0"
        native_environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
        native_python = root / "native-runtime/bin/python"
        command(
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
            "native-environment",
            env=native_environment,
        )
        command(
            [
                str(native_python),
                "-c",
                "import sys,jax; from exp593_anneal_evals.native import model_config; c=model_config(); print('Native interpreter:',sys.executable,'JAX:',jax.__version__); assert jax.__version__=='0.11.1'; assert c.vocab_size==128256 and c.max_seq_len==8192 and not c.compact_input_lookup_ids and not c.compact_input_lookup_one_hot",
            ],
            "native-imports",
            env=native_environment,
        )
        state["progress"] = 0.1
        provider = None if args.export_only else restore_provider(source, root)
        state["progress"] = 0.25
        command(
            [
                str(native_python),
                "-m",
                "exp593_anneal_evals.native",
                "--checkpoint-record",
                str(source / "checkpoint-record.json"),
                "--fixtures",
                str(source / "fixtures.json"),
                "--output",
                str(root / "model"),
            ],
            "native-export-reference",
            env=native_environment,
        )
        state["progress"] = 0.6
        model_records = []
        for path in sorted((root / "model").iterdir()):
            if path.is_file():
                model_records.append(
                    upload_verified(
                        storage, path, args.artifact_root + "/model/" + path.name
                    )
                )
        (root / "export-manifest.json").write_text(
            json.dumps(model_records, indent=2) + "\n"
        )
        provider_environment = os.environ.copy()
        provider_environment["CUDA_VISIBLE_DEVICES"] = "1"
        if args.export_only:
            state["phase"] = "export-complete-needs-practical-review"
        else:
            assert provider is not None
            command(
                [
                    str(provider),
                    "-m",
                    "exp593_anneal_evals.gate",
                    "--model-dir",
                    str(root / "model"),
                    "--fixtures",
                    str(source / "fixtures.json"),
                    "--output",
                    str(root / "gate.json"),
                ],
                "torch-native-gate",
                env=provider_environment,
            )
        state["progress"] = 1.0
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        done.set()
        observer.join()
        if (root / "model/export-receipt.json").exists() and not (
            root / "export-manifest.json"
        ).exists():
            # Preserve a completed export for diagnosis even if a later native
            # reference check failed. This does not mark it accepted for scoring.
            model_records = [
                upload_verified(
                    storage, path, args.artifact_root + "/model/" + path.name
                )
                for path in sorted((root / "model").iterdir())
                if path.is_file()
            ]
            (root / "export-manifest.json").write_text(
                json.dumps(model_records, indent=2) + "\n"
            )
        for name in (
            "validation.log",
            "provider-environment.json",
            "export-manifest.json",
            "gate.json",
        ):
            path = root / name
            if path.exists():
                receipts.append(
                    upload_verified(storage, path, args.artifact_root + "/" + name)
                )
        receipt = root / "validation-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "state": state,
                    "artifacts": receipts,
                    "source": manifest,
                    "checkpoint": record,
                    "finished_epoch": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        upload_verified(
            storage, receipt, args.artifact_root + "/validation-receipt.json"
        )
        wb.log(
            {
                "run_progress": state["progress"],
                "validation/phase": state["phase"],
                "validation/diagnostics_complete": state["error"] is None
                and not args.export_only,
                "validation/agent_review_required": True,
            }
        )
        wb.finish(exit_code=0 if state["error"] is None else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export without accepting for scoring; requires separate real-task review.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
