"""Prepare and verify the shared validation windows without running a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    # These imports belong to the pinned Marin execution environment, not the
    # lightweight plotting environment.
    import fsspec
    from exp586_moe.corpus import CORPUS_PATH, open_dataset
    from exp586_moe.data import scratch_tokenizer
    from levanter.store.cache import TreeCache

    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = (
        "s3://marin-us-east-02a/MarinDNA/tokenized/plantcad/"
        "Angiosperm_65_genomes_8192bp/validation"
    )
    cache = TreeCache.load(cache_root, {"input_ids": np.zeros(0, np.int32)})
    dataset = open_dataset("validation")
    tokenizer = scratch_tokenizer()
    manifest = {
        "schema": 1,
        "corpus_sha256": hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest(),
        "dense_cache": cache_root,
        "dense_cache_shard_order": cache.ledger.finished_shards,
        "validation_rows": len(dataset),
        "sequence_length": 8192,
        "logical_batch_size": 128,
        "logical_batches": 512,
        "samples": {},
    }
    for name, indices in (
        ("original-moe-512", [i * len(dataset) // 512 for i in range(512)]),
        ("common-65536", list(range(512 * 128))),
    ):
        target = args.output / f"{name}.parquet"
        digest = hashlib.sha256()
        canonical_digest = hashlib.sha256()
        schema = pa.schema([("source_row", pa.int32()), ("seq", pa.string())])
        with pq.ParquetWriter(target, schema, compression="zstd") as writer:
            for start in range(0, len(indices), 512):
                batch_indices = indices[start : start + 512]
                sequences = dataset.get_batch(batch_indices)
                encoded = tokenizer.encode_batch(sequences, add_special_tokens=False)
                if name == "common-65536":
                    cached = cache.get_batch_sync(batch_indices)
                    for sequence, row, item in zip(
                        sequences, encoded, cached, strict=True
                    ):
                        tokens = np.asarray(item["input_ids"], dtype=np.int32)
                        if len(sequence) != 8192 or not np.array_equal(tokens, row.ids):
                            raise ValueError(
                                "Dense cache differs from raw validation windows"
                            )
                        canonical_digest.update(tokens.astype("<i4").tobytes())
                for sequence in sequences:
                    digest.update(sequence.encode("ascii"))
                writer.write_table(
                    pa.Table.from_pydict(
                        {"source_row": batch_indices, "seq": sequences}, schema=schema
                    )
                )
                if start % 8192 == 0:
                    print(
                        f"{name}: verified {start + len(sequences)}/{len(indices)}",
                        flush=True,
                    )
        payload = target.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        uri = args.prefix.rstrip("/") + f"/{sha}-{name}.parquet"
        with fsspec.open(uri, "wb") as stream:
            stream.write(payload)
        with fsspec.open(uri, "rb") as stream:
            if hashlib.sha256(stream.read()).hexdigest() != sha:
                raise ValueError("Prepared validation upload failed verification")
        manifest["samples"][name] = {
            "uri": uri,
            "sha256": sha,
            "bytes": len(payload),
            "examples": len(indices),
            "input_bases": 8192 * len(indices),
            "targets": 8191 * len(indices),
            "sequence_sha256": digest.hexdigest(),
            "historical_token_sha256": canonical_digest.hexdigest()
            if name == "common-65536"
            else None,
        }
    path = args.output / "inputs.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
