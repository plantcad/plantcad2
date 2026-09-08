"""Independent CPU-only verification of full saved probabilities, scores, and metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from checkpoints import FULL_ROWS


def verify(artifact: Path, targets_dir: Path, expected_nodes: int | None = None) -> dict:
    result = json.loads((artifact / "result.json").read_text())
    metadata = json.loads((targets_dir / "targets.json").read_text())
    target_path = targets_dir / "targets.npz"
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == metadata["sha256"]
    assert result["sampling"]["method"] == "full"
    assert result["sampling"]["manifest_sha256"] == metadata["manifest_sha256"]
    assert len(result["tasks"]) == 20
    assert sum(task["samples"] for task in result["tasks"].values()) == FULL_ROWS
    assert result["runtime"]["scored_sequences"] == 2 * FULL_ROWS
    assert result["runtime"]["scored_tokens"] == 2 * FULL_ROWS * 8192
    nodes = result["runtime"]["nodes"]
    assert isinstance(nodes, int) and nodes > 0
    assert result["runtime"]["gpus"] == nodes * 8
    if expected_nodes is not None:
        assert nodes == expected_nodes
    assert len(result["flash_verification"]) == nodes * 8
    assert all(item["external_flash_attention_2_verified"] for item in result["flash_verification"])
    rows = {row["key"]: row for row in result["rows"]}
    differences, counts, elements = [], {}, 0
    with np.load(target_path, allow_pickle=False) as targets, np.load(artifact / "scores.npz", allow_pickle=False) as predictions:
        assert len(predictions.files) == 77
        for name in predictions.files:
            values = predictions[name]
            assert np.isfinite(values).all(), name
            elements += values.size
        for key, task in result["tasks"].items():
            n = task["samples"]
            counts[key] = n
            computed = {}
            for context, recorded in task["contexts"].items():
                prefix = f"{key}__{context}"
                scores = predictions[f"{prefix}__scores"]
                assert scores.shape == (n,)
                if task["kind"] == "sv":
                    metrics = {"auprc": float(average_precision_score(targets[f"{key}__label"], scores))}
                else:
                    true = targets[f"{prefix}__true"]
                    probs = predictions[f"{prefix}__probs"]
                    assert probs.shape == (*true.shape, 4)
                    assert len(true) == n and ((probs >= 0) & (probs <= 1)).all()
                    np.testing.assert_allclose(probs.sum(axis=2), 1.0, rtol=0, atol=3e-7)
                    valid = true >= 0
                    if task["kind"] == "motif":
                        correct = probs.argmax(axis=2) == true
                        expected_scores = correct.all(axis=1).astype(np.float32)
                        valid_rows = valid.all(axis=1)
                        metrics = {"token_accuracy": float(correct[valid].mean()) if valid.any() else 0.0, "motif_accuracy": float(correct.all(axis=1)[valid_rows].mean()) if valid_rows.any() else 0.0}
                    else:
                        base_probs = np.zeros(true.shape, dtype=np.float64)
                        r, c = np.where(valid)
                        base_probs[r, c] = probs[r, c, true[r, c]]
                        expected_scores = base_probs[:, 0] if task["kind"] == "conservation" else base_probs.mean(axis=1)
                        labels = targets[f"{key}__label"]
                        metrics = {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores))}
                    np.testing.assert_array_equal(scores, expected_scores)
                computed[context] = metrics
                assert set(metrics) == set(recorded["metrics"])
                for metric, value in metrics.items():
                    difference = abs(value - recorded["metrics"][metric])
                    assert difference < 1e-13, (key, context, metric, difference)
                    differences.append(difference)
            primary = task["primary_metric"]
            selected = max(computed, key=lambda context: computed[context][primary])
            assert selected == task["best_context"] == rows[key]["selected_context"]
            assert abs(computed[selected][primary] - rows[key]["selected_value"]) < 1e-13
    assert len(differences) == 77
    return {"verified": True, "tasks": len(counts), "rows": FULL_ROWS, "prediction_arrays": 77, "prediction_elements": elements, "recomputed_context_metrics": len(differences), "max_metric_difference": max(differences), "non_sv_scores_recomputed_exactly_from_saved_probs": True, "all_selected_strands_verified": True, "flash_verified_workers": nodes * 8, "row_counts": counts, "target_sha256": metadata["sha256"], "manifest_sha256": metadata["manifest_sha256"], "verification": "Independent direct-target extraction and NumPy/sklearn metric recomputation; no inference code imported"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, help="Optionally require the requested allocation size")
    args = parser.parse_args()
    report = verify(args.artifact, args.targets_dir, args.expected_nodes)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
