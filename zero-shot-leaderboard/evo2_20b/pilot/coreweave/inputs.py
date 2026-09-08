"""Exact row-range access for retained TSV previews and complete Parquet splits."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@lru_cache(maxsize=64)
def parquet_file(path: str):
    return pq.ParquetFile(path)


@lru_cache(maxsize=3)
def tsv_frame(path: str):
    return pd.read_csv(path, sep="\t")


def read_frame(preparation: dict, spec: dict, start: int = 0, stop: int | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    record = preparation["inputs"][spec["file"]]
    stop = record["rows"] if stop is None else stop
    if not 0 <= start < stop <= record["rows"]:
        raise ValueError(f"Invalid input row range: {start}:{stop}/{record['rows']}")
    root = Path(preparation["samples"])
    if preparation["input_format"] == "tsv":
        frame = tsv_frame(str(root / spec["file"])).iloc[start:stop]
        if columns is not None:
            frame = frame[columns]
        return frame.reset_index(drop=True)
    parts = []
    offset = 0
    for source in record["files"]:
        parquet = parquet_file(str(root / source["path"]))
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if start < offset + size and stop > offset:
                table = parquet.read_row_group(group, columns=columns)
                local_start = max(start - offset, 0)
                local_stop = min(stop - offset, size)
                parts.append(table.slice(local_start, local_stop - local_start))
            offset += size
            if offset >= stop:
                break
        if offset >= stop:
            break
    table = pa.concat_tables(parts)
    if table.num_rows != stop - start:
        raise ValueError("Parquet slice does not cover the requested rows")
    return table.to_pandas().reset_index(drop=True)
