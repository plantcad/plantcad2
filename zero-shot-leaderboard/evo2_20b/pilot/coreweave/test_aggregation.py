"""Reducer checks run in the remote inference environment (no local CUDA install)."""

import tempfile
import unittest
import argparse
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import merge_arrays, recompute_task, scoring


class AggregationTests(unittest.TestCase):
    def test_single_class_chunks_skip_metrics_not_predictions(self):
        spec = {"kind": "conservation", "positions": [1], "contexts": ["left"]}
        frame = pd.DataFrame({"sequence": ["AAA", "ACA"], "label": [0, 0]})
        probs = np.array([[[0.7, 0.1, 0.1, 0.1]], [[0.1, 0.7, 0.1, 0.1]]], dtype=np.float32)
        args = argparse.Namespace(batch_size=32, softmax_dtype="fp32", use_cache=False)
        with ExitStack() as stack:
            stack.enter_context(patch.object(scoring, "_score_positions", return_value=(probs, {})))
            for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(scoring.torch.cuda, name, return_value=0))
            result, arrays = scoring._score_standard_task(spec, frame, None, None, args, compute_metrics=False)
            self.assertEqual(result["contexts"]["left"]["metrics"], {})
            self.assertIsNone(result["best_context"])
            frame["label"] = [0, 1]
            normal, expected = scoring._score_standard_task(spec, frame, None, None, args)
            self.assertIn("auroc", normal["contexts"]["left"]["metrics"])
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], expected[key])

    def test_reconstruct_original_order(self):
        expected = np.arange(48, dtype=np.float32).reshape(4, 3, 4)
        with tempfile.TemporaryDirectory(prefix="plantcad-reducer-test-") as temporary:
            root = Path(temporary)
            pieces = [{"key": "second", "start": 2, "stop": 4}, {"key": "first", "start": 0, "stop": 2}]
            np.savez(root / "first.npz", left__probs=expected[:2])
            np.savez(root / "second.npz", left__probs=expected[2:])
            actual = merge_arrays(pieces, root, 4)
            np.testing.assert_array_equal(actual["left__probs"], expected)

    def test_auc_is_global_not_average_of_chunk_aucs(self):
        frame = pd.DataFrame({"label": [0, 1, 0, 1]})
        result = recompute_task({"kind": "conservation", "contexts": ["left"]}, frame, {"left__scores": np.array([0.1, 0.2, 0.3, 0.4])})
        # Each two-row chunk has AUROC 1, but the complete task has AUROC 0.75.
        self.assertEqual(result["best_value"], 0.75)

    def test_motif_validity_mask_survives_reduction(self):
        frame = pd.DataFrame({"sequence": ["AACA", "AANA", "ACCA"]})
        probs = np.eye(4)[np.array([[0, 1], [0, 0], [0, 1]])]
        result = recompute_task({"kind": "motif", "positions": [1, 2], "contexts": ["left"]}, frame, {"left__scores": np.array([1, 0, 0]), "left__probs": probs})
        self.assertEqual(result["best_value"], 0.5)

    def test_select_strand_after_full_task_metric(self):
        frame = pd.DataFrame({"label": [0, 1, 0, 1]})
        result = recompute_task({"kind": "core", "contexts": ["left", "right_reverse_complement"]}, frame, {"left__scores": np.array([0.1, 0.4, 0.3, 0.2]), "right_reverse_complement__scores": np.array([0.2, 0.3, 0.4, 0.1])})
        self.assertEqual(result["best_context"], "left")
        self.assertEqual(result["best_value"], 0.75)


if __name__ == "__main__":
    unittest.main()
