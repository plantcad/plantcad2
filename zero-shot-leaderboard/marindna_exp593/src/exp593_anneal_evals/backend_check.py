"""Finite diagnostic replay of one unchanged native reference on three FA3 paths.

This gathers evidence and preserves every failed check. It cannot accept a
checkpoint, change thresholds, launch full evaluations, or publish results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from exp593_evals.artifacts import fetch
from exp593_evals.bootstrap import restore_provider
from exp593_evals.storage import client, upload_verified

PROFILES = ("fast-fa3", "vllm-fa3", "torch-fa3")


def replay(args):
    # A fresh interpreter sets the environment before importing Torch.
    if args.profile == "fast-fa3":
        os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    else:
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
    from exp593_anneal_evals import gate
    from exp593_evals import torch_scorer

    original = torch_scorer.FastScorer
    torch_scorer.FastScorer = lambda *a, **kw: original(
        *a, configuration=args.profile, **kw
    )
    try:
        gate.check(args.model_dir, args.fixtures, args.output)
    except ValueError:
        if not args.output.exists():
            raise  # An integration/runtime error is not a numerical result.
        saved = json.loads(args.output.read_text())
        if (
            saved.get("native_base_gate_passed") is not False
            or saved.get("canonical_adapter_passed") is False
        ):
            raise
    saved = json.loads(args.output.read_text())
    saved["configuration"] = args.profile
    saved["diagnostic_only"] = True
    saved["profile_selection"] = (
        "The existing FastScorer configuration argument selects a frozen provider/class profile. "
        "Native fixtures, weights, output rows, compiler invariants and thresholds are unchanged."
    )
    args.output.write_text(json.dumps(saved, indent=2) + "\n")


def run(args):
    import wandb

    source, root = args.source, args.root
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source / "src")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        environment[name] = "1"
    wb = wandb.init(
        entity="eric-czech",
        project="marin",
        id=args.run_id,
        name=args.run_id,
        group="dna-exp593-annealing-evals",
        resume="allow",
        config={
            "scope": "Finite backend diagnosis only",
            "profiles": PROFILES,
            "source": json.loads((source / "source-manifest.json").read_text()),
            "priority": "batch",
            "user": "eczech",
        },
    )
    processes = []
    error = None
    records = []
    try:
        wb.log({"run_progress": 0, "validation/phase": "restore-unchanged-reference"})
        model = root / "model"
        model.mkdir()
        for record in json.loads((source / "export-manifest.json").read_text()):
            fetch(record, model)
        providers = {}
        for kind in ("vllm", "no-vllm"):
            directory = root / kind
            directory.mkdir()
            providers[kind] = restore_provider(source, directory, kind=kind)
        for gpu, profile in enumerate(PROFILES):
            log = (root / (profile + ".log")).open("w")
            env = {**environment, "CUDA_VISIBLE_DEVICES": str(gpu)}
            if profile == "fast-fa3":
                env["TORCHINDUCTOR_FREEZING"] = "1"
            else:
                env.pop("TORCHINDUCTOR_FREEZING", None)
            provider = providers["no-vllm" if profile == "torch-fa3" else "vllm"]
            process = subprocess.Popen(
                [
                    str(provider),
                    "-m",
                    "exp593_anneal_evals.backend_check",
                    "--profile",
                    profile,
                    "--model-dir",
                    str(model),
                    "--fixtures",
                    str(source / "fixtures.json"),
                    "--output",
                    str(root / (profile + ".json")),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append((profile, process, log))
        while any(p.poll() is None for _, p, _ in processes):
            done = sum(p.poll() == 0 for _, p, _ in processes)
            wb.log(
                {
                    "run_progress": done / 3,
                    "validation/phase": "three-profile-replay",
                    "validation/completed_profiles": done,
                    "validation/failed_profiles": sum(
                        p.poll() not in (None, 0) for _, p, _ in processes
                    ),
                }
            )
            import time

            time.sleep(30)
        if any(p.returncode for _, p, _ in processes):
            raise RuntimeError("A diagnostic did not finish")
        wb.log(
            {"run_progress": 1, "validation/phase": "diagnostics-complete-needs-review"}
        )
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        for _, process, log in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
        for path in sorted(root.glob("*.json")) + sorted(root.glob("*.log")):
            records.append(
                upload_verified(client(), path, args.artifact_root + "/" + path.name)
            )
        receipt = root / "diagnostic-receipt.json"
        receipt.write_text(
            json.dumps(
                {"diagnostic_only": True, "error": error, "records": records}, indent=2
            )
            + "\n"
        )
        upload_verified(
            client(), receipt, args.artifact_root + "/diagnostic-receipt.json"
        )
        wb.finish(exit_code=0 if error is None else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    for name in ["source", "root", "model-dir", "fixtures", "output"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    if args.profile:
        if not all([args.model_dir, args.fixtures, args.output]):
            parser.error("Profile replay needs model, fixtures and output")
        replay(args)
    else:
        if not all([args.source, args.root, args.run_id, args.artifact_root]):
            parser.error(
                "Controller needs source, root, run identity and artifact owner"
            )
        run(args)


if __name__ == "__main__":
    main()
