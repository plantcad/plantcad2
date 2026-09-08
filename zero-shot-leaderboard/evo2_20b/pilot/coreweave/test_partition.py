"""CPU-only checks; no Torch/CUDA dependencies or new local virtualenv required."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partition import plan_chunks, serialize_plan, validate_task_coverage


class PartitionTests(unittest.TestCase):
    def test_variable_full_split_sizes_and_tiny_final_batches(self):
        sizes = {"a": 183685, "b": 153869, "c": 36662, "d": 35477, "e": 33}
        for nodes in (1, 2, 4, 8):
            chunks = serialize_plan(plan_chunks(sizes, workers=8 * nodes))
            self.assertEqual(sum(chunk["stop"] - chunk["start"] for chunk in chunks), sum(sizes.values()))
            for task, size in sizes.items():
                ordered = validate_task_coverage([chunk for chunk in chunks if chunk["task"] == task], size)
                self.assertEqual([(start, min(start + 32, size)) for start in range(0, size, 32)], [(start, min(start + 32, chunk["stop"])) for chunk in ordered for start in range(chunk["start"], chunk["stop"], 32)])

    def test_all_node_counts_preserve_original_batches(self):
        sizes = {f"task{index}": 10000 for index in range(20)}
        for nodes in (1, 2, 4, 8):
            chunks = plan_chunks(sizes, workers=nodes * 8)
            self.assertEqual(chunks, plan_chunks(sizes, workers=nodes * 8))
            loads = [0] * (nodes * 8)
            for chunk in chunks:
                loads[chunk.worker] += chunk.stop - chunk.start
            self.assertLessEqual(max(loads) - min(loads), 1024)
            self.assertTrue(all(load > 0 for load in loads))
            for task, size in sizes.items():
                ordered = validate_task_coverage([chunk for chunk in serialize_plan(chunks) if chunk["task"] == task], size)
                original_batches = [(start, min(start + 32, size)) for start in range(0, size, 32)]
                sharded_batches = [(start, min(start + 32, chunk["stop"])) for chunk in ordered for start in range(chunk["start"], chunk["stop"], 32)]
                self.assertEqual(original_batches, sharded_batches)

    def test_reject_invalid_plan(self):
        for workers, chunk_size in ((0, 1024), (1, 1000), (1, 0), (100, 1024)):
            with self.assertRaises(ValueError):
                plan_chunks({"task": 10000}, workers, chunk_size)

    def test_reject_gap_overlap_and_duplicate(self):
        bad = ([{"start": 0, "stop": 32}, {"start": 64, "stop": 96}], [{"start": 0, "stop": 64}, {"start": 32, "stop": 96}], [{"start": 0, "stop": 96}, {"start": 0, "stop": 96}], [{"start": 0, "stop": 64}])
        for pieces in bad:
            with self.assertRaises(ValueError):
                validate_task_coverage(pieces, 96)


if __name__ == "__main__":
    unittest.main()
