"""Compare full predictions to saved preview predictions on matched input rows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def verify(full: Path, sampled: Path, mapping: Path, sample_manifest: Path) -> dict:
    metadata = json.loads((mapping / "sample-indices.json").read_text())
    indices_path = mapping / "sample-indices.npz"
    assert hashlib.sha256(indices_path.read_bytes()).hexdigest() == metadata["sha256"]
    full_result = json.loads((full / "result.json").read_text())
    sampled_result = json.loads((sampled / "result.json").read_text())
    assert full_result["sampling"]["method"] == "full"
    assert full_result["sampling"]["manifest_sha256"] == metadata["manifest_sha256"]
    assert hashlib.sha256(sample_manifest.read_bytes()).hexdigest() == metadata["reference_manifest_sha256"]
    assert sampled_result["sampling"]["seed"] == 0 and sampled_result["sampling"]["samples_per_task"] == 10000
    assert sampled_result["sampling"]["dataset_revision"] == full_result["sampling"]["dataset_revision"]
    report, observed, changed_rows = {}, set(), {}
    with np.load(indices_path, allow_pickle=False) as indices, np.load(full / "scores.npz", allow_pickle=False) as complete:
        for path in sorted(sampled.glob("workers/gpu*/scores.npz")):
            with np.load(path, allow_pickle=False) as preview:
                for name in preview.files:
                    assert name not in observed
                    observed.add(name)
                    key = name.split("__", 1)[0]
                    old = preview[name]
                    full_values = complete[name]
                    new = full_values[indices[key]]
                    assert new.shape == old.shape and new.dtype == old.dtype
                    assert np.isfinite(new).all() and np.isfinite(old).all()
                    differences = np.abs(new.astype(np.float64) - old.astype(np.float64))
                    report[name] = {"shape": list(new.shape), "dtype": str(new.dtype), "elements": int(new.size), "different_elements": int(np.count_nonzero(new != old)), "max_abs_delta": float(differences.max()), "mean_abs_delta": float(differences.mean()), "bytes_equal": new.tobytes() == old.tobytes()}
                    for row in np.flatnonzero((new != old).reshape(len(old), -1).any(axis=1)):
                        full_row = int(indices[key][row])
                        record = changed_rows.setdefault((key, int(row)), {"task": key, "preview_row": int(row), "full_row": full_row, "preview_batch_size": min(32, len(old) - int(row) // 32 * 32), "full_batch_size": min(32, len(full_values) - full_row // 32 * 32), "arrays": []})
                        record["arrays"].append(name.split("__", 1)[1])
        assert observed == set(complete.files) and len(report) == 77
    return {"arrays": report, "array_count": len(report), "elements": sum(row["elements"] for row in report.values()), "different_elements": sum(row["different_elements"] for row in report.values()), "max_abs_delta": max(row["max_abs_delta"] for row in report.values()), "all_bytes_equal": all(row["bytes_equal"] for row in report.values()), "different_input_rows": len(changed_rows), "all_differences_at_changed_batch_size": all(row["preview_batch_size"] != row["full_batch_size"] for row in changed_rows.values()), "different_row_batches": list(changed_rows.values()), "mapping_sha256": metadata["sha256"], "full_manifest_sha256": metadata["manifest_sha256"], "sample_manifest_sha256": metadata["reference_manifest_sha256"]}


def subset_metrics(full: Path, sampled: Path, mapping: Path, targets_dir: Path) -> dict:
    """Measure any numerical change on the same examples, not the larger split."""
    result = json.loads((full / "result.json").read_text())
    reference = json.loads((sampled / "result.json").read_text())
    target_metadata = json.loads((targets_dir / "targets.json").read_text())
    assert target_metadata["manifest_sha256"] == result["sampling"]["manifest_sha256"]
    target_path = targets_dir / "targets.npz"
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == target_metadata["sha256"]
    comparisons = {}
    with np.load(mapping / "sample-indices.npz", allow_pickle=False) as indices, np.load(target_path, allow_pickle=False) as targets, np.load(full / "scores.npz", allow_pickle=False) as predictions:
        for key, task in result["tasks"].items():
            order = indices[key]
            metrics = {}
            for context in task["contexts"]:
                prefix = f"{key}__{context}"
                scores = predictions[f"{prefix}__scores"][order]
                if task["kind"] == "motif":
                    true = targets[f"{prefix}__true"][order]
                    valid = (true >= 0).all(axis=1)
                    correct = (predictions[f"{prefix}__probs"][order].argmax(axis=2) == true).all(axis=1)
                    metrics[context] = float(correct[valid].mean())
                else:
                    labels = targets[f"{key}__label"][order]
                    metric = average_precision_score if task["kind"] == "sv" else roc_auc_score
                    metrics[context] = float(metric(labels, scores))
            selected = max(metrics, key=metrics.get)
            old = reference["tasks"][key]
            old_value = old["contexts"][old["best_context"]]["metrics"][old["primary_metric"]]
            comparisons[key] = {"preview_value": old_value, "full_run_on_preview_rows": metrics[selected], "delta": metrics[selected] - old_value, "preview_context": old["best_context"], "subset_context": selected, "context_deltas": {context: value - old["contexts"][context]["metrics"][old["primary_metric"]] for context, value in metrics.items()}}
    return {"tasks": comparisons, "max_selected_metric_delta": max(abs(row["delta"]) for row in comparisons.values()), "max_primary_context_metric_delta": max(abs(value) for row in comparisons.values() for value in row["context_deltas"].values()), "changed_strand_selections": sum(row["preview_context"] != row["subset_context"] for row in comparisons.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--sampled", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.full, args.sampled, args.mapping, args.sample_manifest)
    result["subset_metrics"] = subset_metrics(args.full, args.sampled, args.mapping, args.targets_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "arrays"}))
