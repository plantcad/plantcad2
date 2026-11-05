"""Fit one checkpoint/task motif head from source-bound cached flank features."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp586_evals.classification_runner import (
    collect_features,
    fetch,
    read_bytes,
    upload_directory,
)
from exp586_evals.motif_probe import fit_task


def assemble_features(
    membership: pd.DataFrame,
    cached: dict[str, np.ndarray],
    missing: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve motif row order and exact forward/RC orientation for both sources."""
    selected = membership.split.isin(["train", "test"]).to_numpy()
    outputs = [np.zeros((len(membership), 768), dtype=np.float32) for _ in range(2)]
    for source, arrays in (("cached_core", cached), ("new_motif", missing)):
        mask = selected & membership.feature_origin.eq(source).to_numpy()
        if not mask.any():
            continue
        if arrays is None:
            raise ValueError("Missing required motif flank embeddings")
        indices = membership.loc[mask, "feature_row_id"].to_numpy(dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(arrays["row_id"])):
            raise ValueError("Feature row mapping lies outside the bound source")
        np.testing.assert_array_equal(arrays["row_id"][indices], indices)
        for name, output in zip(("fwd", "rc"), outputs, strict=True):
            if (
                arrays[name].shape[1:] != (768,)
                or not np.isfinite(arrays[name][indices]).all()
            ):
                raise ValueError("Unexpected non-finite or non-d768 activations")
            output[mask] = arrays[name][indices]
    if set(membership.loc[selected, "feature_origin"]) - {"cached_core", "new_motif"}:
        raise ValueError("Unknown motif feature origin")
    return outputs[0], outputs[1]


def run(args: argparse.Namespace) -> None:
    import wandb

    payload = read_bytes(args.manifest_uri)
    if hashlib.sha256(payload).hexdigest() != args.manifest_sha256:
        raise ValueError("Motif analysis manifest identity differs")
    config = json.loads(payload)
    model = config["models"][args.model_label]
    task = config["tasks"][args.task]
    directory = Path(tempfile.mkdtemp(prefix="exp586-motif-head-"))
    tracking = wandb.init(
        entity="eric-czech",
        project="marin",
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        group="exp586-motif-checkpoint-probes",
        tags=[
            "dna-exp586",
            "evaluation",
            "motif-probe",
            "CPU",
            args.model_label,
            args.task,
        ],
        config={
            "manifest_sha256": args.manifest_sha256,
            "model": model,
            "task": args.task,
            "training_updates": 0,
        },
    )
    tracking.summary["evaluation/phase"] = "collecting"
    binding = {
        "manifest_sha256": args.manifest_sha256,
        "checkpoint": model["checkpoint"],
        "task": args.task,
    }
    try:
        membership = pd.read_parquet(
            fetch(task["membership"], directory / "membership.parquet")
        )
        membership = membership.sort_values("row_id").reset_index(drop=True)
        np.testing.assert_array_equal(membership.row_id, np.arange(len(membership)))
        cached_model = model["cached_features"]
        cached_requests = json.loads(
            fetch(
                cached_model["requests"], directory / "cached-requests.json"
            ).read_text()
        )
        cached_binding = cached_requests["models"][cached_model["label"]]
        if any(
            cached_binding[name] != model[name]
            for name in ("checkpoint", "raw_metadata_sha256")
        ):
            raise ValueError("Cached activations refer to a different checkpoint")
        arrays = collect_features(
            cached_requests,
            cached_model["requests"]["sha256"],
            cached_model["label"],
            cached_model["root"],
            args.task.replace("motif_", "core_"),
            directory / "cached",
        )
        tracking.log({"run_progress": 0.10})
        missing = None
        if (
            (membership.feature_origin == "new_motif")
            & membership.split.isin(["train", "test"])
        ).any():
            new_requests_record = config["new_feature_requests"]
            new_requests = json.loads(
                fetch(new_requests_record, directory / "new-requests.json").read_text()
            )
            if (
                new_requests["models"][args.model_label]["checkpoint"]
                != model["checkpoint"]
            ):
                raise ValueError("New activations refer to a different checkpoint")
            missing = collect_features(
                new_requests,
                new_requests_record["sha256"],
                args.model_label,
                model["new_feature_root"],
                args.task,
                directory / "missing",
            )
        fwd, rc = assemble_features(membership, arrays, missing)
        tracking.log({"run_progress": 0.15})
        fits = directory / "fits"
        fits.mkdir()
        partial = args.output_prefix.rstrip("/") + "/partial"
        import fsspec

        fs, key = fsspec.core.url_to_fs(partial + "/receipt.json")
        if fs.exists(key):
            receipt = json.loads(read_bytes(partial + "/receipt.json"))
            if receipt["binding"] != binding or receipt["complete"] is not False:
                raise ValueError("A partial head belongs to a different motif study")
            for name, record in receipt["files"].items():
                fetch({**record, "uri": partial + "/" + name}, fits / name)

        def report(value: dict[str, Any]) -> None:
            tracking.summary.update(
                {"evaluation/phase": value["phase"], "evaluation/task": args.task}
            )
            if "C_index" in value:
                tracking.log(
                    {
                        "run_progress": 0.15
                        + 0.65 * value["C_index"] / value["C_total"],
                        "probe/development_accuracy": value["development_accuracy"],
                    }
                )
                upload_directory(fits, partial, binding, complete=False)
            else:
                tracking.log({"evaluation/phase": value["phase"]})

        result = fit_task(membership, fwd, rc, fits, jobs=args.jobs, progress=report)
        result["binding"] = binding
        (fits / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        upload_directory(fits, args.output_prefix, binding)
        tracking.summary.update(
            {
                "evaluation/phase": "complete",
                "evaluation/task": args.task,
                "evaluation/result_uri": args.output_prefix + "/result.json",
                "probe/accuracy": result["test_score"],
                "probe/majority_baseline": result["majority_baseline"],
                "probe/shuffled_accuracy": result["shuffled_labels"],
            }
        )
        tracking.log({"run_progress": 1.0})
        tracking.finish()
    except Exception:
        tracking.summary["evaluation/phase"] = "failed"
        tracking.finish(exit_code=1)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
