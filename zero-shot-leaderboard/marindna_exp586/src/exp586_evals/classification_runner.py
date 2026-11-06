"""Collect verified frozen features and fit one model's classification probes."""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd

from exp586_evals.classification_probe import fit_task, metric


def read_bytes(uri: str) -> bytes:
    """Read through the configured CoreWeave object-store client."""
    with fsspec.open(uri, "rb") as stream:
        return stream.read()


def digest(path: Path) -> str:
    """Hash a file without a second full-sized memory allocation."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(record: dict[str, Any], path: Path) -> Path:
    """Require exact bytes and identity for every immutable input."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        filesystem, key = fsspec.core.url_to_fs(record["uri"])
        filesystem.get_file(key, str(path))
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("Artifact differs from its immutable receipt")
    return path


def upload_directory(
    directory: Path, root: str, binding: dict[str, Any], *, complete: bool = True
) -> dict[str, Any]:
    """Commit small fitted models and predictions only after direct readback."""
    files = {}
    filesystem, prefix = fsspec.core.url_to_fs(root)
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        sha = digest(path)
        key = prefix.rstrip("/") + "/" + path.name
        filesystem.put_file(str(path), key)
        if (
            hashlib.sha256(read_bytes(root.rstrip("/") + "/" + path.name)).hexdigest()
            != sha
        ):
            raise ValueError("Analysis artifact readback differs")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha}
    receipt = {"binding": binding, "files": files, "complete": complete}
    with filesystem.open(prefix.rstrip("/") + "/receipt.json", "wb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True).encode())
    return receipt


def collect_features(
    requests: dict[str, Any],
    request_sha256: str,
    model_label: str,
    feature_root: str,
    task: str,
    output: Path,
) -> dict[str, np.ndarray]:
    """Merge disjoint chunks into disk-backed arrays, checking every row ID."""
    plan = sorted(
        (piece for piece in requests["plan"] if piece["task"] == task),
        key=lambda piece: piece["start"],
    )
    output.mkdir(parents=True, exist_ok=True)
    model = requests["models"][model_label]
    spec = next(spec for spec in requests["tasks"] if spec["key"] == task)
    expected_rows = requests["inputs"][spec["file"]]["rows"]
    if not plan or plan[0]["start"] != 0 or plan[-1]["stop"] != expected_rows:
        raise ValueError("Frozen feature plan does not cover the complete task")

    def download(piece: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        uri = feature_root.rstrip("/") + "/chunks/" + piece["key"]
        receipt = json.loads(read_bytes(uri + ".json"))
        binding = receipt["binding"]
        if any(receipt[name] != value for name, value in piece.items()):
            raise ValueError("Feature chunk addresses different source rows")
        if (
            binding["requests_sha256"] != request_sha256
            or binding["model_label"] != model_label
            or binding["checkpoint_metadata_raw_sha256"] != model["raw_metadata_sha256"]
        ):
            raise ValueError(
                "Feature chunk belongs to a different checkpoint or protocol"
            )
        path = output / (piece["key"] + ".npz")
        if not path.exists():
            path.write_bytes(read_bytes(uri + ".npz"))
        if digest(path) != receipt["array_sha256"]:
            raise ValueError("Frozen feature checksum differs")
        return receipt, path

    with ThreadPoolExecutor(max_workers=12) as executor:
        pieces = list(executor.map(download, plan))
    bindings = {json.dumps(receipt["binding"], sort_keys=True) for receipt, _ in pieces}
    if len(bindings) != 1:
        raise ValueError("A task mixes extraction implementations or checkpoints")
    rows = plan[-1]["stop"]
    arrays = {}
    previous_stop = 0
    for receipt, path in pieces:
        begin, end = receipt["start"], receipt["stop"]
        if begin != previous_stop:
            raise ValueError(
                "Frozen feature coverage is not exactly disjoint and complete"
            )
        with np.load(path, allow_pickle=False) as saved:
            expected = (
                {"row_id", "ref_fwd", "ref_rc", "alt_fwd", "alt_rc"}
                if task == "sv_impact"
                else {"row_id", "fwd", "rc"}
            )
            if set(saved.files) != expected:
                raise ValueError("Unexpected fields in frozen feature chunk")
            np.testing.assert_array_equal(saved["row_id"], np.arange(begin, end))
            for name in saved.files:
                value = saved[name]
                if len(value) != end - begin or not np.isfinite(value).all():
                    raise ValueError("Frozen feature dimensions or values differ")
                if name != "row_id" and value.shape[1:] != (768,):
                    raise ValueError("Unexpected d768 hidden-state width")
                if name not in arrays:
                    arrays[name] = np.lib.format.open_memmap(
                        output / (name + ".npy"),
                        mode="w+",
                        dtype=value.dtype,
                        shape=(rows, *value.shape[1:]),
                    )
                arrays[name][begin:end] = value
        previous_stop = end
    for array in arrays.values():
        array.flush()
    return arrays


def run(args: argparse.Namespace) -> None:
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError(
            "The CoreWeave bootstrap must forward the operator's W&B credential"
        )
    payload = read_bytes(args.manifest_uri)
    if hashlib.sha256(payload).hexdigest() != args.manifest_sha256:
        raise ValueError("Analysis manifest differs")
    analysis = json.loads(payload)
    directory = Path(tempfile.mkdtemp(prefix="exp586-classification-analysis-"))
    request_record = analysis["feature_requests"]
    request_file = fetch(request_record, directory / "requests.json")
    requests = json.loads(request_file.read_text())
    requested = set(args.task or [])
    known = {spec["key"] for spec in requests["tasks"]}
    if requested - known:
        raise ValueError("An unknown classification task was requested")
    selected = [
        spec for spec in requests["tasks"] if not requested or spec["key"] in requested
    ]
    run = wandb.init(
        entity="eric-czech",
        project="marin",
        id=args.run_id,
        name=args.run_id,
        resume="allow",
        group="exp586-plantcad2-moe-frozen-classification",
        tags=["dna-exp586", "evaluation", "linear-probes", "CPU", args.model_label],
        config={
            "model_label": args.model_label,
            "analysis_manifest_sha256": args.manifest_sha256,
            "feature_requests_sha256": request_record["sha256"],
            "training_updates": 0,
            "requested_tasks": [spec["key"] for spec in selected],
        },
    )
    run.summary["evaluation/phase"] = "collecting"
    environment = {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "scikit-learn", "scipy", "pandas", "joblib")
        },
    }
    run.config.update({"environment": environment})
    binding = {
        "analysis_manifest_sha256": args.manifest_sha256,
        "feature_requests_sha256": request_record["sha256"],
        "model_label": args.model_label,
    }
    results = {}
    for index, spec in enumerate(selected):
        task = spec["key"]
        root = args.output_prefix.rstrip("/") + "/" + task
        filesystem, key = fsspec.core.url_to_fs(root + "/receipt.json")
        if filesystem.exists(key):
            receipt = json.loads(read_bytes(root + "/receipt.json"))
            if receipt["binding"] != binding or receipt["complete"] is not True:
                raise ValueError("Existing analysis belongs to different inputs")
            record = receipt["files"]["result.json"]
            result = json.loads(
                fetch(
                    {**record, "uri": root + "/result.json"},
                    directory / task / "result.json",
                ).read_text()
            )
        else:
            run.summary.update(
                {"evaluation/phase": "collecting", "evaluation/task": task}
            )
            arrays = collect_features(
                requests,
                request_record["sha256"],
                args.model_label,
                analysis["feature_roots"][args.model_label],
                task,
                directory / task / "features",
            )
            membership = fetch(
                analysis["memberships"][task], directory / task / "membership.parquet"
            )
            run.summary["evaluation/phase"] = "fitting"
            fits = directory / task / "fits"
            partial = root + "/partial"
            partial_fs, partial_key = fsspec.core.url_to_fs(partial + "/receipt.json")
            if partial_fs.exists(partial_key):
                receipt = json.loads(read_bytes(partial + "/receipt.json"))
                if receipt["binding"] != binding or receipt["complete"] is not False:
                    raise ValueError("Partial probe fits belong to different inputs")
                for name, record in receipt["files"].items():
                    fetch({**record, "uri": partial + "/" + name}, fits / name)
            result = fit_task(
                spec,
                membership,
                arrays,
                fits,
                jobs=args.jobs,
                checkpoint=functools.partial(
                    upload_directory, fits, partial, binding, complete=False
                ),
            )
            frame = pd.read_parquet(membership).sort_values("row_id")
            test = frame.split.eq("test")
            row_ids = frame.loc[test, "row_id"].to_numpy(dtype=np.int64)
            labels = frame.loc[test, "label"].to_numpy(dtype=np.int64)
            baseline_path = fetch(
                analysis["baselines"][args.model_label][task],
                directory / task / "baseline.npz",
            )
            with np.load(baseline_path, allow_pickle=False) as baseline:
                np.testing.assert_array_equal(baseline["row_id"][row_ids], row_ids)
                for name in ("natural", "poe"):
                    if name in baseline:
                        scores = baseline[name][row_ids]
                        if not np.isfinite(scores).all():
                            raise ValueError(
                                "Cached zero-shot scores do not cover the probe test set"
                            )
                        result["readouts"][name] = {
                            "test_score": metric(labels, scores, spec["kind"]),
                            "test_rows": len(labels),
                        }
            (directory / task / "fits/result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            upload_directory(directory / task / "fits", root, binding)
            del arrays
        results[task] = result
        run.summary.update(
            {
                "evaluation/phase": "fitting",
                "evaluation/tasks_done": index + 1,
                "evaluation/tasks_total": len(selected),
                "evaluation/task": task,
            }
        )
        run.log(
            {
                "run_progress": (index + 1) / len(selected),
                **{
                    f"probe/{task}/{name}": value["test_score"]
                    for name, value in result["readouts"].items()
                },
            }
        )
    summary = {
        "binding": binding,
        "environment": environment,
        "tasks": results,
        "full_task_set": not requested,
        "complete": True,
    }
    summary_path = directory / "summary"
    summary_path.mkdir()
    (summary_path / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary_root = (
        args.output_prefix.rstrip("/")
        + ("/qualification/" + "-".join(sorted(requested)) if requested else "")
        + "/summary"
    )
    upload_directory(summary_path, summary_root, binding)
    run.summary.update(
        {
            "evaluation/phase": "complete",
            "evaluation/result_uri": summary_root + "/result.json",
        }
    )
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--task",
        action="append",
        help="Qualify a task while preserving its reusable, source-bound fits",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
