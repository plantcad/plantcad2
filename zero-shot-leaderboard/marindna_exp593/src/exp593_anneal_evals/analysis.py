"""Reuse the original scoring and fitting code on complete, verified new features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reuse unchanged, already evaluated functions; only new readouts live here.
from exp593_evals.analysis import (
    chunks,
    fit_task,
    fitting_code,
)
from exp593_evals.artifacts import digest
from exp593_evals.storage import client, upload_verified


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
        if args.kind == "tasks" and (length == 2048 or args.variant.endswith("-exact")):
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
            from exp593_anneal_evals.metrics import task_metrics

            result = task_metrics(preparation, profile, length, args.variant)
        elif args.kind == "af":
            result = full_metrics.af_metrics(
                preparation, profile, args.output / "predictions.parquet"
            )
            if args.variant.startswith("fim-"):
                result["natural_method"] = (
                    "Common target-free FIM prompt; forward/RC mean single-site REF/ALT odds, separate FP32 lowercase-only projection"
                )
                result["bidirectional_method"] = (
                    "Equal-weight forward/RC product at the FIM variant target"
                )
        else:
            if args.af_zero_shot is None:
                raise ValueError(
                    "AF probes require the matching new lowercase-control predictions"
                )
            # The unchanged fitter resolves samples from the manifest's parent.
            # Put its copy beside the original sample links, preserving all
            # original membership fields and the actual source-file locations.
            probe_preparation = (
                args.inputs
                / "samples"
                / ("af-probe-preparation-" + digest(preparation_path) + ".json")
            )
            probe_preparation.write_bytes(preparation_path.read_bytes())
            full_probe.fit_probes(
                probe_preparation,
                profile / "embeddings",
                args.af_zero_shot,
                args.output,
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
