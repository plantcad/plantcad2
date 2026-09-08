#!/usr/bin/env python3
"""Count ambiguous bases in the exact historical maize-AF evaluation windows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from huggingface_hub import hf_hub_download

from sampling import build_samples


DATASET_REPO = "plantcad/maize-allele-frequency"
DATASET_REVISION = "b2e5138cb7e8d952e6c674c488e89e35e86abe52"
DATASET_FILE = "10k_all_consequences/test.parquet"
DATASET_FULL_FILE = "full/test.parquet"
DATASET_20K_FILE = "20k_all_consequences/test.parquet"
GENOME_URL = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
GENOME_BYTES = 672_057_339
GENOME_S3_URI = "s3://marin-us-east-02a/MarinDNA/plantcad2-evals/references/zea_mays/release-62/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"


def download_ranges(url: str, path: Path, size: int, workers: int = 8) -> None:
    """Download a range-capable immutable object concurrently and atomically."""
    if path.exists() and path.stat().st_size == size:
        return
    parts = [path.with_suffix(path.suffix + f".part{i:02d}") for i in range(workers)]

    def one(index: int) -> None:
        start = size * index // workers
        stop = size * (index + 1) // workers - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}"})
        with urllib.request.urlopen(request) as source, parts[index].open("wb") as target:
            if source.status != 206:
                raise RuntimeError(f"Server ignored byte range {start}-{stop}: HTTP {source.status}")
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
        if parts[index].stat().st_size != stop - start + 1:
            raise RuntimeError(f"Incomplete byte range {start}-{stop}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(workers)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
            part.unlink()
    if temporary.stat().st_size != size:
        raise RuntimeError("Assembled download has the wrong size")
    temporary.replace(path)


def s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(s3={"addressing_style": "virtual"}))


def fetch_s3(uri: str, path: Path, size: int) -> bool:
    """Use an already-staged copy when its byte count is intact."""
    from botocore.exceptions import ClientError

    parsed = urlparse(uri)
    client = s3_client()
    try:
        remote = client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    except ClientError:
        return False
    if remote["ContentLength"] != size:
        return False
    temporary = path.with_suffix(path.suffix + ".tmp")
    client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(temporary))
    if temporary.stat().st_size != size:
        raise RuntimeError("Downloaded CWS3 genome has the wrong size")
    temporary.replace(path)
    return True


def stage_s3(path: Path, uri: str, sha256: str) -> None:
    parsed = urlparse(uri)
    client = s3_client()
    client.upload_file(str(path), parsed.netloc, parsed.path.lstrip("/"), ExtraArgs={"Metadata": {"sha256": sha256}})
    remote = client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    if remote["ContentLength"] != path.stat().st_size:
        raise RuntimeError("Staged CWS3 genome did not verify")


def read_selected_fasta(path: Path, wanted: set[str]) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith(">"):
                current = line[1:].split()[0]
                if current in wanted:
                    records[current] = []
                continue
            if current in records:
                records[current].append(line.strip().upper())
    missing = wanted - records.keys()
    if missing:
        raise ValueError(f"Genome is missing chromosomes: {sorted(missing)}")
    return {chrom: "".join(parts) for chrom, parts in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/maize-af-window-inspection"))
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stage-s3-uri", default=GENOME_S3_URI)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        filename: Path(
            hf_hub_download(
                DATASET_REPO,
                filename,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=args.work_dir / "dataset",
            )
        )
        for filename in (DATASET_FILE, DATASET_FULL_FILE, DATASET_20K_FILE)
    }
    samples_dir = args.work_dir / "samples"
    sampling = build_samples(fixtures[DATASET_FULL_FILE], fixtures[DATASET_FILE], fixtures[DATASET_20K_FILE], samples_dir)
    genome_path = args.work_dir / "genome.fa.gz"
    staged = bool(args.stage_s3_uri) and fetch_s3(args.stage_s3_uri, genome_path, GENOME_BYTES)
    if not staged:
        download_ranges(GENOME_URL, genome_path, GENOME_BYTES)
    genome_sha256 = hashlib.file_digest(genome_path.open("rb"), "sha256").hexdigest()
    if args.stage_s3_uri and not staged:
        stage_s3(genome_path, args.stage_s3_uri, genome_sha256)
    frame = pd.read_parquet(samples_dir / "union.parquet")
    genome = read_selected_fasta(genome_path, set(frame["chrom"].astype(str)))
    center = args.window_size // 2
    totals = {
        "rows": len(frame),
        "rows_with_non_acgt_full_window": 0,
        "rows_with_non_acgt_left_of_variant": 0,
        "rows_with_non_acgt_right_of_variant": 0,
        "non_acgt_bases": 0,
        "reference_mismatches": 0,
        "rows_with_non_acgt_by_sample": {requested: 0 for requested in (10_000, 20_000, 50_000, 100_000)},
    }
    examples: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        seq = genome[str(row.chrom)]
        variant_index = int(row.pos) - 1
        start = variant_index - center
        end = start + args.window_size
        window = "N" * max(0, -start) + seq[max(0, start) : min(len(seq), end)] + "N" * max(0, end - len(seq))
        if len(window) != args.window_size:
            raise AssertionError((row.chrom, row.pos, len(window)))
        non_acgt = sum(base not in "ACGT" for base in window)
        left_bad = any(base not in "ACGT" for base in window[:center])
        right_bad = any(base not in "ACGT" for base in window[center + 1 :])
        totals["rows_with_non_acgt_full_window"] += bool(non_acgt)
        totals["rows_with_non_acgt_left_of_variant"] += left_bad
        totals["rows_with_non_acgt_right_of_variant"] += right_bad
        totals["non_acgt_bases"] += non_acgt
        totals["reference_mismatches"] += window[center] != str(row.ref).upper()
        if non_acgt:
            for requested in (10_000, 20_000, 50_000, 100_000):
                totals["rows_with_non_acgt_by_sample"][requested] += bool(getattr(row, f"in_sample_{requested}"))
        if non_acgt and len(examples) < 10:
            examples.append({"chrom": str(row.chrom), "pos": int(row.pos), "non_acgt": non_acgt, "left_bad": left_bad, "right_bad": right_bad})
    print(json.dumps({"dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "files": list(fixtures), "sampling": sampling}, "genome": {"url": GENOME_URL, "bytes": GENOME_BYTES, "sha256": genome_sha256, "staged_s3_uri": args.stage_s3_uri}, "window_size": args.window_size, "totals": totals, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
