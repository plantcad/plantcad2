#!/usr/bin/env python3
"""Reproduce the legacy seed-42 balanced samples and build their scored union."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from config import CONSEQUENCES, EXCLUDED_VARIANTS, EXPECTED_SAMPLE_ROWS, EXPECTED_UNION_ROWS, SAMPLE_SEED, SAMPLE_SIZES


KEYS = ["chrom", "pos", "ref", "alt"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def balanced_sample(full: pl.DataFrame, requested_n: int) -> pl.DataFrame:
    eligible = full.filter(~pl.col("is_repeat")).drop("is_repeat")
    per_consequence = requested_n // len(CONSEQUENCES)
    pieces = []
    for consequence in CONSEQUENCES:
        subset = eligible.filter(pl.col("consequence") == consequence)
        pieces.append(subset.sample(n=min(per_consequence, subset.height), seed=SAMPLE_SEED))
    return pl.concat(pieces).sort(KEYS)


def apply_exclusions(frame: pl.DataFrame) -> pl.DataFrame:
    for variant in EXCLUDED_VARIANTS:
        matches = (
            (pl.col("chrom").cast(pl.String) == variant["chrom"])
            & (pl.col("pos") == variant["pos"])
            & (pl.col("ref") == variant["ref"])
            & (pl.col("alt") == variant["alt"])
        )
        if frame.filter(matches).height != 1:
            raise RuntimeError(f"Expected exactly one occurrence of excluded variant: {variant}")
        frame = frame.filter(~matches)
    return frame


def build_samples(full_path: Path, exact_10k: Path, exact_20k: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    full = pl.read_parquet(full_path)
    generated_before_exclusion = {requested_n: balanced_sample(full, requested_n) for requested_n in SAMPLE_SIZES}
    published = {10_000: pl.read_parquet(exact_10k), 20_000: pl.read_parquet(exact_20k)}
    for requested_n, expected in published.items():
        if not generated_before_exclusion[requested_n].equals(expected):
            raise RuntimeError(f"Polars 1.34.0 seed-42 sample does not reproduce the published {requested_n:,} fixture")
    generated = {requested_n: apply_exclusions(frame) for requested_n, frame in generated_before_exclusion.items()}
    for requested_n, frame in generated.items():
        if frame.height != EXPECTED_SAMPLE_ROWS[requested_n] or frame.select(KEYS).n_unique() != frame.height:
            raise RuntimeError(f"Unexpected sample inventory for requested n={requested_n:,}")

    union = pl.concat([frame.select(KEYS) for frame in generated.values()]).unique(KEYS).sort(KEYS).with_row_index("row_id")
    if union.height != EXPECTED_UNION_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_UNION_ROWS:,} unique variants, got {union.height:,}")
    union = union.join(full.drop("is_repeat"), on=KEYS, how="left", validate="1:1")
    if union.null_count().sum_horizontal().item() != 0:
        raise RuntimeError("Union failed to join uniquely onto the full input")

    memberships: dict[str, dict] = {}
    for requested_n, frame in generated.items():
        sample = frame.join(union.select(["row_id", *KEYS]), on=KEYS, how="left", validate="1:1").sort(KEYS)
        path = output_dir / f"sample-{requested_n}.parquet"
        sample.write_parquet(path)
        memberships[str(requested_n)] = {"requested_rows": requested_n, "actual_rows": sample.height, "path": path.name, "sha256": digest(path)}
        member_ids = set(sample["row_id"].to_list())
        union = union.with_columns(pl.col("row_id").is_in(member_ids).alias(f"in_sample_{requested_n}"))

    union_path = output_dir / "union.parquet"
    union.write_parquet(union_path)
    manifest = {
        "method": "legacy consequence-balanced sample without replacement",
        "seed": SAMPLE_SEED,
        "polars": pl.__version__,
        "key_columns": KEYS,
        "consequences": list(CONSEQUENCES),
        "sample_sizes": memberships,
        "union": {"rows": union.height, "path": union_path.name, "sha256": digest(union_path)},
        "published_10k_and_20k_reproduced_exactly_before_exclusion": True,
        "exclusions": list(EXCLUDED_VARIANTS),
    }
    manifest_path = output_dir / "sampling-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--exact-10k", type=Path, required=True)
    parser.add_argument("--exact-20k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_samples(args.full, args.exact_10k, args.exact_20k, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
