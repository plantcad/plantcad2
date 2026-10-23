"""Reduce native predictions with the original PR20 task/AF/probe implementations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp586_evals.artifacts import collect
from exp586_evals.protocol import original


def legacy_af_chunks(
    plan: list[dict], source: Path, output: Path, prediction_columns: tuple[str, ...]
) -> list[tuple[dict, Path]]:
    """Project richer native AF chunks onto the original PR20 reducer schema."""
    output.mkdir(parents=True, exist_ok=True)
    required = {"row_id", *prediction_columns, "target_probs_fwd", "target_probs_rc"}
    chunks = []
    for metadata in plan:
        source_path = source / f"{metadata['key']}.npz"
        with np.load(source_path, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                missing = sorted(required.difference(archive.files))
                raise RuntimeError(
                    f"Native AF chunk {source_path.name} is missing {missing}"
                )
            values = {name: archive[name] for name in ("row_id", *prediction_columns)}
            target_probabilities = [
                archive[name] for name in ("target_probs_fwd", "target_probs_rc")
            ]
        rows = metadata["stop"] - metadata["start"]
        if any(
            len(value) != rows for value in [*values.values(), *target_probabilities]
        ):
            raise RuntimeError(
                f"Native AF chunk {source_path.name} has wrong row count"
            )
        if any(
            probabilities.shape != (rows, 4) or not np.isfinite(probabilities).all()
            for probabilities in target_probabilities
        ):
            raise RuntimeError(
                f"Native AF chunk {source_path.name} has invalid target probabilities"
            )
        output_path = output / source_path.name
        np.savez_compressed(output_path, **values)
        chunks.append((metadata, output_path))
    return chunks


def score(
    preparation_path: Path, directory: Path, condition: str, checkpoint_digest: str
) -> dict:
    protocol = original()
    prep = json.loads(preparation_path.read_text())
    prep["samples"] = str(preparation_path.parent.resolve())
    worker = collect(
        str(directory),
        directory,
        preparation_path,
        condition,
        lambda uri: Path(uri).read_bytes(),
        checkpoint_digest,
        verify_only=True,
    )
    task_chunks = [
        item["receipt"] for item in worker["chunks"] if item["kind"] == "tasks"
    ]
    task_worker = {
        **worker,
        "started_epoch": min(c["started_epoch"] for c in task_chunks),
        "finished_epoch": max(c["finished_epoch"] for c in task_chunks),
        "run_wall_seconds": sum(
            c["finished_epoch"] - c["started_epoch"] for c in task_chunks
        ),
    }
    prep.update(
        model=worker["binding"]["condition"],
        checkpoint=worker["binding"],
        condition="bf16-native-jax-sonic-fp32-acgt-no-cache-b8",
        manifest_sha256=worker["binding"]["requests_sha256"],
    )
    tasks = protocol.aggregate.aggregate(
        directory / "tasks", prep, [task_worker], nodes=1
    )
    af_root = directory / "af"
    union = preparation_path.parent / "maize-af.parquet"
    chunks = legacy_af_chunks(
        prep["af_plan"],
        af_root / "chunks",
        af_root / "legacy-chunks",
        protocol.af_aggregate.PREDICTION_COLUMNS,
    )
    af = protocol.af_aggregate.aggregate(
        af_root, union, prep["af_plan"], chunks, only_sample_size=2112
    )
    frame = pd.read_parquet(union)
    pieces = prep["af_plan"]
    embeddings = protocol.aggregate.merge_arrays(
        pieces, directory / "embeddings/chunks", len(frame)
    )
    np.testing.assert_array_equal(embeddings.pop("row_id"), frame.row_id)
    split_columns = ["row_id", "split", "block_id", "af_bin", "stratum", "outer_fold"]
    split = frame[split_columns].copy()
    if set(split[split.split.eq("train")].block_id) & set(
        split[split.split.eq("test")].block_id
    ):
        raise ValueError("Historical split has overlapping genomic blocks")
    probe_root = directory / "probes"
    probe_root.mkdir(exist_ok=True)
    probes = protocol.probe.fit_probes(
        frame.drop(columns=split_columns[1:] + ["sequence"]),
        embeddings,
        pd.read_parquet(af_root / "predictions.parquet"),
        probe_root,
        provided_split=(
            split,
            {
                **prep["af_source"],
                "pilot_only": True,
                "method": "Subset of archived exp472 membership; no resplitting",
            },
        ),
    )
    # Saved estimators and predictions must reproduce each other after serialization.
    import joblib

    predictions = pd.read_parquet(probe_root / "predictions.parquet")
    for name in ("whole_window", "variant_token"):
        estimator = joblib.load(probe_root / f"probe-{name}.joblib")
        x = protocol.probe._feature(
            embeddings[f"{name}_ref"], embeddings[f"{name}_alt"]
        )
        np.testing.assert_array_equal(
            estimator.predict(x), predictions[f"probe_{name}"]
        )
    result = {
        "pilot_only": True,
        "groups": protocol.compare.group_scores(tasks),
        "allele_frequency": af,
        "probes": probes,
        "binding": worker["binding"],
    }
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--condition", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    args = parser.parse_args()
    result = score(
        args.preparation, args.directory, args.condition, args.checkpoint_digest
    )
    print(json.dumps(result["groups"]), flush=True)


if __name__ == "__main__":
    main()
