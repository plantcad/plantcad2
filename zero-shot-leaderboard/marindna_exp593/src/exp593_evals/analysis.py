"""Reuse the original scoring and fitting code on complete, verified new features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from exp593_evals.artifacts import digest, fetch, read
from exp593_evals.storage import client, upload_verified


def aligned_features(arrays: dict, ids: np.ndarray) -> dict:
    """Original memberships may exclude rows; never infer membership from offsets."""
    rows = arrays["row_id"]
    if not np.array_equal(rows, np.unique(rows)) or not np.array_equal(
        ids, np.unique(ids)
    ):
        raise ValueError("Feature and membership IDs must be unique and ordered")
    take = np.searchsorted(rows, ids)
    if np.any(take >= len(rows)) or not np.array_equal(rows[take], ids):
        raise ValueError("A saved training/test membership has no extracted feature")
    return {name: value[take] for name, value in arrays.items()}


def chunks(
    owner: str,
    plan: list[dict],
    output: Path,
    binding: dict,
    *,
    profile: str,
    variant: str,
    kind: str,
) -> Path:
    """Retain the historical chunk layout while checking the new artifact bindings."""
    directory = output / profile / variant / kind / "chunks"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {piece["key"]: piece for piece in plan}
    if len(expected) != len(plan):
        raise ValueError("Duplicate planned chunks")
    root = owner.rstrip("/") + f"/{profile}/{variant}/{kind}/chunks/"
    bucket, prefix = root.removeprefix("s3://").split("/", 1)
    storage = client()
    present = set()
    for page in storage.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for value in page.get("Contents", []):
            key = value["Key"].rsplit("/", 1)[-1].removesuffix(".json")
            if key not in expected:
                continue  # Other tasks share the same immutable kind owner.
            record = json.loads(read("s3://" + bucket + "/" + value["Key"]))
            if any(record[k] != v for k, v in expected[key].items()):
                raise ValueError("A completed chunk differs from the original plan")
            actual_binding = dict(record["binding"])
            job = actual_binding.pop("job_profile")
            expected_job = {
                "profile": profile,
                "variant": "ar-lower-control" if kind == "embeddings" else variant,
                "kind": "af" if kind == "embeddings" else kind,
                "length": 8192
                if profile == "frozen-probes"
                else int(profile.removeprefix("context-")),
            }
            if actual_binding != binding or any(
                job[k] != v for k, v in expected_job.items()
            ):
                raise ValueError("Mixed model, code, data, readout or context bindings")
            if Path(key).name != key:
                raise ValueError("A chunk key cannot contain a path")
            cached = fetch(
                record["artifact"], output / "cache" / record["array_sha256"]
            ).resolve()
            if digest(cached) != record["array_sha256"]:
                raise ValueError("Committed feature/prediction digest differs")
            target = directory / (key + ".npz")
            if target.exists() and target.resolve() != cached:
                raise ValueError("A previous local chunk has another identity")
            if not target.exists():
                target.symlink_to(cached)
            (directory / (key + ".json")).write_text(
                json.dumps(record, indent=2) + "\n"
            )
            present.add(key)
    if present != set(expected):
        raise ValueError(
            f"Incomplete input coverage: {len(present)}/{len(expected)} chunks"
        )
    return directory


def fitting_code(inputs: Path, source: Path) -> None:
    """Keep the original scientific wheel and append only the pinned fitting files."""
    sys.path.insert(0, str(inputs / "scientific-code"))
    import exp586_evals

    manifest = json.loads((source / "src/exp593_evals/sources.json").read_text())
    for relative, expected in manifest["probe_fitting"]["files"].items():
        if digest(source / relative) != expected:
            raise ValueError("Original probe fitting source differs from its pin")
    for relative, expected in manifest["probe_fitting"][
        "scientific_wheel_files"
    ].items():
        if digest(inputs / "scientific-code/exp586_evals" / relative) != expected:
            raise ValueError("Original AF fitting/feature code differs from its pin")
    location = str(source / "provenance/probe-fit-code/exp586_evals")
    if location not in exp586_evals.__path__:
        exp586_evals.__path__.insert(0, location)


def fit_task(args: argparse.Namespace, binding: dict, catalog: dict) -> dict:
    import pandas as pd
    from exp586_evals import classification_probe, full_metrics, motif_probe

    for module in (classification_probe, motif_probe):
        if (
            Path(module.__file__).resolve().parent
            != (args.source / "provenance/probe-fit-code/exp586_evals").resolve()
        ):
            raise ValueError("An unexpected fitting module was already imported")

    entry = catalog["probe_tasks"][args.task]
    preparation = json.loads(
        (args.inputs / (entry["input_manifest"] + ".json")).read_text()
    )
    plan = [p for p in preparation["plan"] if p["task"] == args.task]
    directory = chunks(
        args.artifact_root,
        plan,
        args.downloads,
        binding,
        profile="frozen-probes",
        variant="ar-lower-features",
        kind="features",
    )
    arrays = full_metrics.merge(
        plan, directory, max(p["stop"] for p in plan), task=args.task
    )
    membership = args.inputs / "memberships" / (args.task + ".parquet")
    frame = pd.read_parquet(membership).sort_values("row_id").reset_index(drop=True)
    arrays = aligned_features(arrays, frame.row_id.to_numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    fit_binding = {
        "checkpoint_binding": binding,
        "membership_sha256": digest(membership),
        "fitting_source": json.loads(
            (args.source / "src/exp593_evals/sources.json").read_text()
        )["probe_fitting"],
        "feature_hashes": {
            k: hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast("B")).hexdigest()
            for k, v in arrays.items()
        },
    }
    path = args.output / "evaluation-binding.json"
    if path.exists() and json.loads(path.read_text()) != fit_binding:
        raise ValueError("Existing fits belong to different features or memberships")
    path.write_text(json.dumps(fit_binding, indent=2) + "\n")
    if entry["spec"]["kind"] == "motif":
        return motif_probe.fit_task(
            frame, arrays["fwd"], arrays["rc"], args.output, jobs=args.jobs
        )
    return classification_probe.fit_task(
        entry["spec"], membership, arrays, args.output, jobs=args.jobs
    )


def run(args: argparse.Namespace) -> None:
    fitting_code(args.inputs, args.source)
    from exp586_evals import full_metrics, full_probe

    binding = json.loads(args.binding.read_text())
    if binding["native_gate"].get("accepted") is not True:
        raise ValueError("Scientific reduction requires reviewed checkpoint validation")
    catalog = json.loads((args.inputs / "catalog.json").read_text())
    preparation_path = args.inputs / "full.json"
    preparation = json.loads(preparation_path.read_text())
    if args.kind == "probe":
        result = fit_task(args, binding, catalog)
    else:
        length = int(args.profile.removeprefix("context-"))
        plan = (
            preparation["af_plan"]
            if args.kind in {"af", "af-probe"}
            else preparation["plan"]
        )
        if length == 2048 and args.kind == "tasks":
            plan = [p for p in plan if p["task"] != "sv_impact"]
        variant = "ar-lower-features" if args.kind == "af-probe" else args.variant
        storage_kind = "embeddings" if args.kind == "af-probe" else args.kind
        chunks(
            args.artifact_root,
            plan,
            args.downloads,
            binding,
            profile=args.profile,
            variant=variant,
            kind=storage_kind,
        )
        profile = args.downloads / args.profile / variant
        args.output.mkdir(parents=True, exist_ok=True)
        if args.kind == "tasks":
            result = full_metrics.task_metrics(preparation, profile, length)
        elif args.kind == "af":
            result = full_metrics.af_metrics(
                preparation, profile, args.output / "predictions.parquet"
            )
            if args.variant.startswith("fim-"):
                result["natural_method"] = (
                    "Common target-free FIM prompt; forward/RC mean single-site REF/ALT odds, separate FP32 eight-row projection"
                )
                result["bidirectional_method"] = (
                    "Equal-weight forward/RC product at the FIM variant target"
                )
        else:
            if args.af_zero_shot is None:
                raise ValueError(
                    "AF probes require the matching new lowercase-control predictions"
                )
            full_probe.fit_probes(
                preparation_path, profile / "embeddings", args.af_zero_shot, args.output
            )
            result = json.loads((args.output / "result.json").read_text())
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    records = []
    if args.publish_owner:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                records.append(
                    upload_verified(
                        client(),
                        path,
                        args.publish_owner.rstrip("/")
                        + "/"
                        + str(path.relative_to(args.output)),
                    )
                )
        receipt = args.output / "analysis-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "binding": binding,
                    "kind": args.kind,
                    "profile": args.profile,
                    "variant": args.variant,
                    "task": args.task,
                    "artifacts": records,
                },
                indent=2,
            )
            + "\n"
        )
        upload_verified(
            client(), receipt, args.publish_owner.rstrip("/") + "/analysis-receipt.json"
        )
    print(json.dumps({"completed": True, "result": str(result_path)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "inputs", "downloads", "output", "binding"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--publish-owner")
    parser.add_argument(
        "--kind", choices=("tasks", "af", "probe", "af-probe"), required=True
    )
    parser.add_argument(
        "--profile", choices=("context-8192", "context-2048"), default="context-8192"
    )
    parser.add_argument("--variant", default="ar-lower-control")
    parser.add_argument("--task")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--af-zero-shot", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
