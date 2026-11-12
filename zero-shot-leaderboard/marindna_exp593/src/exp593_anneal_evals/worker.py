"""Finite lower-only extraction, preserving original row chunks and probe inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from exp593_anneal_evals.artifacts import digest, publish, publish_json


def commit(root, chunk, arrays, binding, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / (chunk["key"] + ".npz")
    np.savez_compressed(path, **arrays)
    record = publish(path, root + "/arrays/" + digest(path) + ".npz")
    publish_json(
        {
            **chunk,
            "binding": binding,
            "array_sha256": record["sha256"],
            "artifact": record,
            "rows": chunk["stop"] - chunk["start"],
            "finished_epoch": time.time(),
        },
        path.with_suffix(".json"),
        root + "/chunks/" + chunk["key"] + ".json",
    )


def existing(root, chunk, binding):
    # The original read-only checker verifies binding, array hashes and all row counts.
    from exp593_evals.worker import existing as original_existing

    return original_existing(root, chunk, binding)


def run(args):
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ]:
        os.environ[k] = "1"
    from exp593_anneal_evals.scoring import LowercaseScorer, af_fim, standard, sv_fim
    from exp593_evals.ar_readout import ARAdapter
    from exp593_evals.torch_scorer import FastScorer
    from exp593_evals.worker import features

    sys.path.insert(0, str(args.inputs / "scientific-code"))
    from exp586_evals.protocol import original

    protocol = original()
    mod = importlib.util.spec_from_file_location(
        "anneal_visible_protocol",
        args.inputs / "probe-code/exp586_evals/probe_protocol.py",
    )
    visibility = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(visibility)
    catalog = json.loads((args.inputs / "catalog.json").read_text())
    full = json.loads((args.inputs / "full.json").read_text())
    classification = json.loads((args.inputs / "classification.json").read_text())
    binding_base = json.loads(args.binding.read_text())
    if not binding_base["native_gate"]["accepted"]:
        raise ValueError("Checkpoint needs reviewed lower-only native validation")
    jobs = json.loads((args.inputs / "jobs.json").read_text())
    assigned = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    if args.limit:
        assigned = assigned[: args.limit]
    specs = {s["key"]: s for s in protocol.tasks.LEADERBOARD_TASKS}
    scorer = FastScorer(args.model_dir)
    adapter = LowercaseScorer(scorer)
    if (
        scorer.checkpoint != binding_base["checkpoint"]
        or binding_base["native_gate"]["checkpoint"] != scorer.checkpoint
    ):
        raise ValueError("Weights, native review and scientific checkpoint differ")
    ar = ARAdapter(scorer, case="lower", collapse_case=False, sequence_length=8192)
    options = argparse.Namespace(
        batch_size=64, sv_batch_size=64, softmax_dtype="fp32", use_cache=False
    )
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / f"worker-{args.worker:03d}.json"
    completed = 0

    def status(phase, **extra):
        value = {
            "phase": phase,
            "completed": completed,
            "total": len(assigned),
            "updated_epoch": time.time(),
            **extra,
        }
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(status_path)

    status("loaded")
    try:
        for job in assigned:
            chunk, variant, kind = job["chunk"], job["variant"], job["kind"]
            root = (
                args.artifact_root.rstrip("/") + f"/{job['profile']}/{variant}/{kind}"
            )
            binding = {
                **binding_base,
                "job_profile": {k: v for k, v in job.items() if k != "chunk"},
            }
            embedding_root = (
                args.artifact_root.rstrip("/")
                + "/context-8192/ar-lower-features/embeddings"
            )
            needs_embeddings = kind == "af" and variant == "ar-lower-control"
            ready = existing(root, chunk, binding)
            embedding_ready = (
                existing(embedding_root, chunk, binding) if needs_embeddings else True
            )
            if ready and embedding_ready:
                completed += 1
                status("inference", task=chunk["task"], variant=variant)
                continue
            if kind == "features":
                entry = catalog["probe_tasks"][chunk["task"]]
                spec = entry["spec"]
                manifest = (
                    classification
                    if entry["input_manifest"] == "classification"
                    else full
                )
            else:
                spec = (
                    specs[chunk["task"]]
                    if kind == "tasks"
                    else {"key": "maize-af", "file": "maize-af.parquet", "kind": "af"}
                )
                manifest = full
            frame = protocol.inputs.read_frame(
                manifest, spec, chunk["start"], chunk["stop"]
            )
            status("inference", task=chunk["task"], variant=variant, chunk=chunk["key"])
            if kind == "features":
                visible = {**spec, "kind": "core"} if spec["kind"] == "motif" else spec
                requests = [
                    visibility.visible_sequences(visible, r)
                    for r in frame.to_dict("records")
                ]
                names = (
                    ("ref_fwd", "ref_rc", "alt_fwd", "alt_rc")
                    if spec["kind"] == "sv"
                    else ("fwd", "rc")
                )
                arrays = {
                    "row_id": np.arange(chunk["start"], chunk["stop"], dtype=np.int64),
                    **{
                        name: features(
                            scorer,
                            [r[name] for r in requests],
                            "mean" if spec["kind"] == "sv" else "last",
                        )
                        for name in names
                    },
                }
            elif kind == "af":
                if variant == "ar-lower-control":
                    arrays, embeddings = ar.allele_frequency(
                        frame, include_strand_average=False
                    )
                    if not embedding_ready:
                        commit(
                            embedding_root,
                            chunk,
                            embeddings,
                            binding,
                            args.output / "embeddings",
                        )
                else:
                    arrays = af_fim(adapter, frame, protocol)
            elif spec["kind"] == "sv":
                if variant == "ar-lower-control":
                    _, arrays = protocol.tasks._score_sv(
                        spec, frame, ar, ar.tokenizer, options, compute_metrics=False
                    )
                else:
                    arrays = sv_fim(adapter, frame.to_dict("records"))
                    arrays["left__scores"] = arrays["forward__scores"]
            elif variant == "ar-lower-control":
                # Keep the literal original benchmark path, including its full
                # source windows and query coordinates, as the TF AR baseline.
                _, arrays = protocol.tasks._score_standard_task(
                    spec, frame, ar, ar.tokenizer, options, compute_metrics=False
                )
            else:
                arrays = standard(
                    adapter,
                    frame.sequence.tolist(),
                    spec["positions"],
                    presentation="ar" if variant.startswith("ar-") else "fim",
                    decoding="exact" if variant.endswith("-exact") else "tf",
                )
            arrays = (
                {
                    "row_id": np.arange(chunk["start"], chunk["stop"], dtype=np.int64),
                    **arrays,
                }
                if kind != "af"
                else arrays
            )
            if any(
                len(v) != len(frame) or not np.isfinite(v).all()
                for v in arrays.values()
            ):
                raise ValueError("Readout has missing rows or nonfinite values")
            if not ready:
                commit(
                    root,
                    chunk,
                    arrays,
                    binding,
                    args.output / f"{job['profile']}-{variant}-{kind}",
                )
            completed += 1
            status("inference", task=chunk["task"], variant=variant)
        status("complete")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "model-dir", "binding", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
