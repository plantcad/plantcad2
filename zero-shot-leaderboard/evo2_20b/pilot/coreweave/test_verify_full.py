"""Small CPU-only fixtures exercise the independent full-result verifier."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import verify_full


def fixture(root):
    # Same 20-task / 77-array contract, but only three rows/task for unit tests.
    specs = [(f"conservation_{i}", "conservation", 1) for i in range(3)]
    specs += [(f"motif_{i}", "motif", 3 if i % 4 < 2 else 2) for i in range(8)]
    specs += [(f"core_{i}", "core", 3 if i % 4 < 2 else 2) for i in range(8)]
    specs += [("sv", "sv", 0)]
    targets, predictions, tasks, rows = {}, {}, {}, []
    labels = np.array([0, 1, 1], dtype=np.int8)
    for key, kind, width in specs:
        contexts = {}
        if kind != "motif":
            targets[f"{key}__label"] = labels
        for context in (["left"] if kind == "sv" else ["left", "right_reverse_complement"]):
            if kind == "sv":
                scores = np.array([-0.3, 0.2, 0.1])
                metrics = {"auprc": float(average_precision_score(labels, scores))}
            else:
                true = np.repeat(np.array([[0], [1], [-1]], dtype=np.int8), width, axis=1)
                if context == "right_reverse_complement":
                    true = np.where(true >= 0, 3 - true[:, ::-1], -1).astype(np.int8)
                probs = np.repeat(np.array([[[0.8, 0.1, 0.05, 0.05]], [[0.2, 0.65, 0.1, 0.05]], [[0.25, 0.25, 0.25, 0.25]]], dtype=np.float32), width, axis=1)
                targets[f"{key}__{context}__true"] = true
                predictions[f"{key}__{context}__probs"] = probs
                if kind == "motif":
                    correct = probs.argmax(axis=2) == true
                    scores = correct.all(axis=1).astype(np.float32)
                    metrics = {"motif_accuracy": float(correct.all(axis=1)[:2].mean()), "token_accuracy": float(correct[:2].mean())}
                else:
                    base_probs = np.zeros(true.shape, dtype=np.float64)
                    for r in range(3):
                        for c in range(width):
                            if true[r, c] >= 0:
                                base_probs[r, c] = probs[r, c, true[r, c]]
                    scores = base_probs.mean(axis=1)
                    metrics = {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores))}
            predictions[f"{key}__{context}__scores"] = scores
            contexts[context] = {"metrics": metrics}
        primary = "auprc" if kind == "sv" else "motif_accuracy" if kind == "motif" else "auroc"
        best = max(contexts, key=lambda context: contexts[context]["metrics"][primary])
        tasks[key] = {"kind": kind, "samples": 3, "contexts": contexts, "primary_metric": primary, "best_context": best}
        rows.append({"key": key, "selected_context": best, "selected_value": contexts[best]["metrics"][primary]})
    result = {"sampling": {"method": "full", "manifest_sha256": "fixture"}, "runtime": {"scored_sequences": 120, "scored_tokens": 120 * 8192, "gpus": 32, "nodes": 4}, "flash_verification": [{"external_flash_attention_2_verified": True}] * 32, "rows": rows, "tasks": tasks}
    np.savez(root / "targets.npz", **targets)
    np.savez(root / "scores.npz", **predictions)
    (root / "targets.json").write_text(json.dumps({"sha256": hashlib.sha256((root / "targets.npz").read_bytes()).hexdigest(), "manifest_sha256": "fixture"}))
    (root / "result.json").write_text(json.dumps(result))
    return result, predictions


class VerifierTests(unittest.TestCase):
    def test_other_allocation_sizes(self):
        with tempfile.TemporaryDirectory(prefix="plantcad-verify-test-") as temporary, patch.object(verify_full, "FULL_ROWS", 60):
            root = Path(temporary)
            result, _ = fixture(root)
            for nodes in (1, 2, 4, 8):
                result["runtime"].update(nodes=nodes, gpus=nodes * 8)
                result["flash_verification"] = [{"external_flash_attention_2_verified": True}] * (nodes * 8)
                (root / "result.json").write_text(json.dumps(result))
                self.assertEqual(verify_full.verify(root, root, expected_nodes=nodes)["flash_verified_workers"], nodes * 8)
                with self.assertRaises(AssertionError):
                    verify_full.verify(root, root, expected_nodes=nodes + 1)

    def test_complete_fixture_and_corruption_detection(self):
        with tempfile.TemporaryDirectory(prefix="plantcad-verify-test-") as temporary, patch.object(verify_full, "FULL_ROWS", 60):
            root = Path(temporary)
            result, predictions = fixture(root)
            self.assertTrue(verify_full.verify(root, root)["verified"])
            result["tasks"]["conservation_0"]["contexts"]["left"]["metrics"]["auroc"] += 0.1
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaises(AssertionError):
                verify_full.verify(root, root)
            result, predictions = fixture(root)
            predictions["conservation_0__left__scores"][0] += 0.01
            np.savez(root / "scores.npz", **predictions)
            with self.assertRaises(AssertionError):
                verify_full.verify(root, root)


if __name__ == "__main__":
    unittest.main()
