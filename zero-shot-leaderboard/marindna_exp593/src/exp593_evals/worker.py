"""Finite checkpoint workers over the original PlantCAD2 source rows and tasks."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from exp593_evals.artifacts import digest, publish, publish_json, read


def commit(root: str, chunk: dict, arrays: dict, binding: dict, output: Path) -> None:
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


def existing(root: str, chunk: dict, binding: dict) -> bool:
    from botocore.exceptions import ClientError

    try:
        receipt = json.loads(read(root + "/chunks/" + chunk["key"] + ".json"))
    except ClientError as error:
        if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise
    if receipt["binding"] != binding or any(receipt[k] != v for k, v in chunk.items()):
        raise ValueError("A committed chunk belongs to another model/readout/input")
    import hashlib

    blob = read(receipt["artifact"]["uri"])
    if hashlib.sha256(blob).hexdigest() != receipt["array_sha256"]:
        raise ValueError("An existing chunk is corrupt")
    with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
        if any(
            len(saved[k]) != chunk["stop"] - chunk["start"]
            or not np.isfinite(saved[k]).all()
            for k in saved.files
        ):
            raise ValueError("A committed chunk has invalid rows or values")
    return True


def features(scorer, sequences: list[str], pooling: str) -> np.ndarray:
    parts = []
    for start in range(0, len(sequences), scorer.batch_size):
        ids = [
            scorer.dna_tokens(s, case="lower")
            for s in sequences[start : start + scorer.batch_size]
        ]
        parts.append(scorer.features(ids, pooling=pooling))
    return np.concatenate(parts)


def run(args: argparse.Namespace) -> None:
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        os.environ[key] = "1"
    from exp593_evals.ar_readout import ARAdapter
    from exp593_evals.fim_readout import FIMAdapter, af_site_readout, sv_boundary_scores
    from exp593_evals.metadata import metadata_for
    from exp593_evals.torch_scorer import FastScorer

    sys.path.insert(0, str(args.inputs / "scientific-code"))
    from exp586_evals.protocol import original

    protocol = original()
    module = importlib.util.spec_from_file_location(
        "exp593_visible_protocol",
        args.inputs / "probe-code/exp586_evals/probe_protocol.py",
    )
    assert module and module.loader
    visibility = importlib.util.module_from_spec(module)
    module.loader.exec_module(visibility)
    catalog = json.loads((args.inputs / "catalog.json").read_text())
    full = json.loads((args.inputs / "full.json").read_text())
    classification = json.loads((args.inputs / "classification.json").read_text())
    binding_base = json.loads(args.binding.read_text())
    gate = binding_base["native_gate"]
    if gate.get("accepted") is not True:
        raise ValueError("Full scoring requires an accepted native/fast-fa3 gate")
    jobs = json.loads((args.inputs / "jobs.json").read_text())
    assigned = [
        job for index, job in enumerate(jobs) if index % args.workers == args.worker
    ]
    if args.limit:
        assigned = assigned[: args.limit]
    specifications = {spec["key"]: spec for spec in protocol.tasks.LEADERBOARD_TASKS}
    scorer = FastScorer(args.model_dir)
    if (
        gate.get("checkpoint") != scorer.checkpoint
        or binding_base["checkpoint"] != scorer.checkpoint
    ):
        raise ValueError(
            "Gate, deployed weights and scientific checkpoint identity differ"
        )
    adapters = {
        (variant, length): ARAdapter(
            scorer,
            case="upper" if variant == "ar-upper-folded" else "lower",
            collapse_case=variant != "ar-lower-control",
            sequence_length=length,
        )
        for variant in ("ar-lower-control", "ar-lower-folded", "ar-upper-folded")
        for length in (2048, 8192)
    }
    function_args = argparse.Namespace(
        batch_size=64, sv_batch_size=64, softmax_dtype="fp32", use_cache=False
    )
    completed = 0
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / f"worker-{args.worker:03d}.json"

    def status(phase: str, **kwargs) -> None:
        value = {
            "phase": phase,
            "completed": completed,
            "total": len(assigned),
            "updated_epoch": time.time(),
            **kwargs,
        }
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(status_path)

    status("loaded")
    try:
        for job in assigned:
            chunk, variant, kind, length = (
                job["chunk"],
                job["variant"],
                job["kind"],
                job["length"],
            )
            root = (
                args.artifact_root.rstrip("/") + f"/{job['profile']}/{variant}/{kind}"
            )
            binding = {
                **binding_base,
                "job_profile": {k: v for k, v in job.items() if k != "chunk"},
            }
            if existing(root, chunk, binding):
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
                    specifications[chunk["task"]]
                    if kind == "tasks"
                    else {"key": "maize-af", "file": "maize-af.parquet", "kind": "af"}
                )
                manifest = full
            frame = protocol.inputs.read_frame(
                manifest, spec, chunk["start"], chunk["stop"]
            )
            status("inference", task=chunk["task"], variant=variant, chunk=chunk["key"])
            if kind == "features":
                # Refit new heads: extract ALL original rows, not just old test rows.
                visible = {**spec, "kind": "sv" if spec["kind"] == "sv" else "core"}
                requests = [
                    visibility.visible_sequences(visible, row)
                    for row in frame.to_dict("records")
                ]
                names = (
                    ("ref_fwd", "ref_rc", "alt_fwd", "alt_rc")
                    if spec["kind"] == "sv"
                    else ("fwd", "rc")
                )
                arrays = {
                    "row_id": np.arange(chunk["start"], chunk["stop"], dtype=np.int64)
                }
                for name in names:
                    arrays[name] = features(
                        scorer,
                        [r[name] for r in requests],
                        "mean" if spec["kind"] == "sv" else "last",
                    )
            elif kind == "af":
                if variant.startswith("fim-"):
                    arrays = af_site_readout(
                        scorer,
                        frame,
                        protocol,
                        sequence_length=length,
                        presentation="minimal"
                        if variant == "fim-minimal-folded"
                        else "metadata",
                    )
                else:
                    arrays, embeddings = adapters[(variant, length)].allele_frequency(
                        frame, include_strand_average=False
                    )
                    if variant == "ar-lower-control":
                        embedding_root = (
                            args.artifact_root.rstrip("/")
                            + f"/{job['profile']}/ar-lower-features/embeddings"
                        )
                        commit(
                            embedding_root,
                            chunk,
                            embeddings,
                            binding,
                            args.output / "embeddings",
                        )
            elif variant.startswith("ar-"):
                fn = (
                    protocol.tasks._score_sv
                    if spec["kind"] == "sv"
                    else protocol.tasks._score_standard_task
                )
                _, arrays = fn(
                    spec,
                    frame,
                    adapters[(variant, length)],
                    adapters[(variant, length)].tokenizer,
                    function_args,
                    compute_metrics=False,
                )
            else:
                presentation = (
                    "minimal" if variant == "fim-minimal-folded" else "metadata"
                )
                if spec["kind"] == "sv":
                    fields = [
                        metadata_for(spec, row) for row in frame.to_dict("records")
                    ]
                    arrays = {
                        "left__scores": sv_boundary_scores(
                            scorer, frame, fields, presentation=presentation
                        )
                    }
                else:
                    arrays = {}
                    for context in spec["contexts"]:
                        fields = [
                            metadata_for(
                                spec,
                                row,
                                reverse_complement=context
                                == "right_reverse_complement",
                            )
                            for row in frame.to_dict("records")
                        ]
                        adapter = FIMAdapter(
                            scorer,
                            fields,
                            presentation=presentation,
                            sequence_length=length,
                        )
                        _, value = protocol.tasks._score_standard_task(
                            {**spec, "contexts": [context]},
                            frame,
                            adapter,
                            None,
                            function_args,
                            compute_metrics=False,
                        )
                        arrays.update(value)
            if any(
                len(value) != chunk["stop"] - chunk["start"]
                or not np.isfinite(value).all()
                for value in arrays.values()
            ):
                raise ValueError(
                    "Prediction/features contain incorrect rows or nonfinite values"
                )
            commit(
                root,
                chunk,
                arrays,
                binding,
                args.output / job["profile"] / variant / kind,
            )
            completed += 1
            status("inference", task=chunk["task"], variant=variant)
        status("complete")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "model-dir", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        raise ValueError("Invalid disjoint worker assignment")
    run(args)


if __name__ == "__main__":
    main()
