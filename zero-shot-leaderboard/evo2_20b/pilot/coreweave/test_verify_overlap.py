"""CPU-only overlap verification regression, including reordered rows/corruption."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from verify_overlap import subset_metrics, verify


class OverlapTest(unittest.TestCase):
    def test_same_subset_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full, sampled, mapping, targets = (root / name for name in ("full", "sampled", "mapping", "targets"))
            for path in (full, sampled, mapping, targets):
                path.mkdir()
            task = {"kind": "sv", "primary_metric": "auprc", "best_context": "left", "contexts": {"left": {"metrics": {"auprc": 1.0}}}}
            (full / "result.json").write_text(json.dumps({"sampling": {"manifest_sha256": "full-hash"}, "tasks": {"sv": task}}))
            (sampled / "result.json").write_text(json.dumps({"tasks": {"sv": task}}))
            np.savez_compressed(mapping / "sample-indices.npz", sv=np.array([3, 0, 2]))
            np.savez_compressed(targets / "targets.npz", sv__label=np.array([0, 1, 1, 0]))
            (targets / "targets.json").write_text(json.dumps({"manifest_sha256": "full-hash", "sha256": hashlib.sha256((targets / "targets.npz").read_bytes()).hexdigest()}))
            np.savez_compressed(full / "scores.npz", sv__left__scores=np.array([0.1, 0.8, 0.7, 0.3]))
            self.assertEqual(subset_metrics(full, sampled, mapping, targets)["max_selected_metric_delta"], 0)
            np.savez_compressed(full / "scores.npz", sv__left__scores=np.array([0.1, 0.8, 0.0, 0.3]))
            changed = subset_metrics(full, sampled, mapping, targets)
            self.assertAlmostEqual(changed["tasks"]["sv"]["delta"], -2 / 3)
            self.assertEqual(changed["changed_strand_selections"], 0)

    def test_reordering_and_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full, sampled, mapping = root / "full", root / "sampled", root / "mapping"
            for path in (full, sampled / "workers/gpu0", mapping):
                path.mkdir(parents=True)
            sample_manifest = root / "sample-manifest.json"
            sample_manifest.write_text("[]\n")
            sample_hash = hashlib.sha256(sample_manifest.read_bytes()).hexdigest()
            (full / "result.json").write_text(json.dumps({"sampling": {"method": "full", "manifest_sha256": "full-hash", "dataset_revision": "dataset-hash"}}))
            (sampled / "result.json").write_text(json.dumps({"sampling": {"seed": 0, "samples_per_task": 10000, "dataset_revision": "dataset-hash"}}))
            order = np.random.default_rng(0).permutation(10020)[:10000].astype(np.int32)
            indices = {f"task{i}": order for i in range(20)}
            np.savez_compressed(mapping / "sample-indices.npz", **indices)
            index_hash = hashlib.sha256((mapping / "sample-indices.npz").read_bytes()).hexdigest()
            metadata = {"sha256": index_hash, "manifest_sha256": "full-hash", "reference_manifest_sha256": sample_hash}
            (mapping / "sample-indices.json").write_text(json.dumps(metadata))
            complete, preview = {}, {}
            for task in range(20):
                suffixes = ("left__scores",) if task == 19 else ("left__scores", "left__probs", "right_reverse_complement__scores", "right_reverse_complement__probs")
                for suffix in suffixes:
                    name = f"task{task}__{suffix}"
                    complete[name] = np.arange(10020, dtype=np.float32)
                    preview[name] = complete[name][order]
            np.savez_compressed(full / "scores.npz", **complete)
            archive = sampled / "workers/gpu0/scores.npz"
            np.savez_compressed(archive, **preview)
            correct = verify(full, sampled, mapping, sample_manifest)
            self.assertTrue(correct["all_bytes_equal"])
            self.assertEqual(correct["array_count"], 77)
            preview["task0__left__scores"][0] += 1
            np.savez_compressed(archive, **preview)
            changed = verify(full, sampled, mapping, sample_manifest)
            self.assertFalse(changed["all_bytes_equal"])
            self.assertEqual(changed["different_elements"], 1)
            self.assertEqual(changed["max_abs_delta"], 1)
            self.assertEqual(changed["different_input_rows"], 1)
            self.assertEqual(changed["different_row_batches"][0]["full_row"], int(order[0]))
            self.assertEqual(changed["different_row_batches"][0]["preview_batch_size"], 32)
            del preview["task0__left__scores"]
            np.savez_compressed(archive, **preview)
            with self.assertRaises(AssertionError):
                verify(full, sampled, mapping, sample_manifest)


if __name__ == "__main__":
    unittest.main()
