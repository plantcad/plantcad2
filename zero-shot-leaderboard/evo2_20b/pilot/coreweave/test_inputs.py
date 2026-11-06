"""CPU tests for exact full-split row coverage across Parquet shards/row groups."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inputs import parquet_file, read_frame, tsv_frame


class InputTests(unittest.TestCase):
    def test_full_parquet_slices_match_original_rows(self):
        frame = pd.DataFrame({"sequence": [f"ACGT{i}" for i in range(113)], "label": [i % 2 for i in range(113)]})
        with tempfile.TemporaryDirectory(prefix="plantcad-inputs-") as temporary:
            root = Path(temporary)
            sources = []
            for index, (start, stop) in enumerate(((0, 37), (37, 79), (79, 113))):
                path = f"part{index}.parquet"
                pq.write_table(pa.Table.from_pandas(frame.iloc[start:stop], preserve_index=False), root / path, row_group_size=7)
                sources.append({"path": path, "rows": stop - start})
            prep = {"samples": str(root), "input_format": "parquet", "inputs": {"test.tsv": {"rows": 113, "files": sources}}}
            spec = {"file": "test.tsv"}
            for start, stop in ((0, 113), (0, 1), (112, 113), (32, 64), (37, 79), (35, 81)):
                pd.testing.assert_frame_equal(read_frame(prep, spec, start, stop), frame.iloc[start:stop].reset_index(drop=True))
                pd.testing.assert_frame_equal(read_frame(prep, spec, start, stop, ["label"]), frame.iloc[start:stop][["label"]].reset_index(drop=True))
            pieces = [read_frame(prep, spec, start, min(start + 32, 113)) for start in range(0, 113, 32)]
            pd.testing.assert_frame_equal(pd.concat(pieces, ignore_index=True), frame)
            for start, stop in ((-1, 1), (0, 114), (1, 1)):
                with self.assertRaises(ValueError):
                    read_frame(prep, spec, start, stop)
            parquet_file.cache_clear()

    def test_sampled_tsv_path_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="plantcad-tsv-") as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({"sequence": ["ACG", "ANN", "TTT"], "label": [0, 1, 0]})
            frame.to_csv(root / "sample.tsv", sep="\t", index=False)
            prep = {"samples": str(root), "input_format": "tsv", "inputs": {"sample.tsv": {"rows": 3}}}
            pd.testing.assert_frame_equal(read_frame(prep, {"file": "sample.tsv"}, 1, 3), frame.iloc[1:].reset_index(drop=True))
            tsv_frame.cache_clear()


if __name__ == "__main__":
    unittest.main()
