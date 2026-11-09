"""Finite owned-node validation: restore providers, export, gate, upload, exit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from exp593_evals.storage import client, upload_verified


def restore_provider(source: Path, root: Path) -> Path:
    """Restore the exact previously audited provider versions and binary archive."""
    manifest = json.loads(
        (source / "provenance/provider-environments.json").read_text()
    )
    protected = manifest["vllm"]["packages"]
    excluded = {"exp586-torch-deployment", "vllm", "flash-attn", "flash-attn-3"}
    environment = root / "provider"
    subprocess.run(["uv", "venv", "--python", "3.12.13", str(environment)], check=True)
    python = environment / "bin/python"
    requirements = root / "provider-requirements.txt"
    requirements.write_text(
        "\n".join(
            sorted(
                k + "==" + v
                for k, v in protected.items()
                if k.lower().replace("_", "-") not in excluded
            )
        )
        + "\n"
    )
    subprocess.run(
        [
            "uv",
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
            "https://download.pytorch.org/whl/cu132",
            "-r",
            str(requirements),
        ],
        check=True,
    )
    archive = root / "vllm-providers.tar.gz"
    client().download_file(
        "marin-us-east-02a",
        "MarinDNA/exp586_plantcad2_moe/inference-deployment/2026-11-07-r1/binaries/vllm-providers.tar.gz",
        str(archive),
    )
    with archive.open("rb") as stream:
        if (
            hashlib.file_digest(stream, "sha256").hexdigest()
            != "80882098871615638f5a2706d978310d6eb02c0c7054e820cd93341adf6bbc53"
        ):
            raise ValueError("Audited provider archive differs")
    site = subprocess.check_output(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    with tarfile.open(archive) as package:
        package.extractall(site, filter="data")
    extras = json.loads(
        (source / "provenance/provider-extra-packages.json").read_text()
    )
    protected_names = {key.lower().replace("_", "-") for key in protected}
    requirements.write_text(
        "\n".join(
            sorted(
                item["name"] + "==" + item["version"]
                for item in extras
                if item["name"].lower().replace("_", "-")
                not in protected_names
                | excluded
                | {"exp586-backend-quality", "torch", "pytest", "ruff"}
            )
        )
        + "\n"
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--prerelease",
            "allow",
            "-r",
            str(requirements),
        ],
        check=True,
    )
    current = json.loads(
        subprocess.check_output(
            ["uv", "pip", "list", "--python", str(python), "--format", "json"],
            text=True,
        )
    )
    versions = {v["name"].lower().replace("_", "-"): v["version"] for v in current}
    for name, expected in protected.items():
        if (
            name != "exp586-torch-deployment"
            and versions.get(name.lower().replace("_", "-")) != expected
        ):
            raise ValueError(f"Protected provider version differs: {name}")
    (root / "provider-environment.json").write_text(
        json.dumps(current, indent=2) + "\n"
    )
    return python


def run(args: argparse.Namespace) -> None:
    import wandb

    source, root = args.source, args.root
    root.mkdir(parents=True, exist_ok=True)
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
        command(
            [
                "uv",
                "sync",
                "--project",
                str(source / "native"),
                "--locked",
                "--extra",
                "gpu",
            ],
            "native-environment",
        )
        state["progress"] = 0.1
        provider = restore_provider(source, root)
        state["progress"] = 0.25
        native_environment = os.environ.copy()
        native_environment["CUDA_VISIBLE_DEVICES"] = "0"
        native_environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
        command(
            [
                str(source / "native/.venv/bin/python"),
                "-m",
                "exp593_evals.native",
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
        command(
            [
                str(provider),
                "-m",
                "exp593_evals.gate",
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
                "validation/accepted": state["error"] is None,
            }
        )
        wb.finish(exit_code=0 if state["error"] is None else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
