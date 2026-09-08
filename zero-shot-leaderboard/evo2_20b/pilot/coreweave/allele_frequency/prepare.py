"""Stage and byte-verify one checkpoint, the AF samples, and the maize genome."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import HfApi, snapshot_download
from pyfaidx import Fasta

HERE = Path(__file__).resolve().parent
COREWEAVE = HERE.parent
sys.path[:0] = [str(HERE), str(COREWEAVE)]

from checkpoints import checkpoint
from common import HF_REPO, digest, storage, write_json
from config import DATASET_FILES, DATASET_REPO, DATASET_REVISION, EXPECTED_GENERATED_SHA256, GENOME_BYTES, GENOME_S3_URI, GENOME_SHA256
from sampling import build_samples


def prepare(root: Path) -> dict:
    started = time.time()
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if (usage.used + 16 * 2**30) / usage.total >= 0.90:
        raise RuntimeError("AF-eval staging would exceed 90% disk usage")
    if not GENOME_SHA256:
        raise RuntimeError("Pin GENOME_SHA256 after the inspection/staging job")

    api = HfApi(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    if api.whoami()["name"] != "eczech":
        raise RuntimeError("Expected eczech Hugging Face credentials")
    selected = checkpoint(os.environ.get("PLANTCAD_CHECKPOINT", "056t"))
    model_prefix = selected["model_prefix"]
    model_snapshot = Path(snapshot_download(HF_REPO, revision=selected["revision"], allow_patterns=[f"{model_prefix}/*"], token=api.token, local_dir=root / "model-download", max_workers=8))
    model = model_snapshot / model_prefix
    model_files = {}
    for entry in api.list_repo_tree(HF_REPO, path_in_repo=model_prefix, recursive=True, expand=True, revision=selected["revision"]):
        if not hasattr(entry, "size"):
            continue
        relative = entry.path.removeprefix(model_prefix + "/")
        path = model / relative
        sha256 = digest(path)
        if path.stat().st_size != entry.size or entry.lfs and sha256 != entry.lfs.sha256:
            raise RuntimeError(f"Model integrity mismatch: {relative}")
        model_files[relative] = {"bytes": entry.size, "sha256": sha256}
    if len(model_files) != 4 or sum(record["bytes"] for record in model_files.values()) != 3_892_739_156:
        raise RuntimeError("Unexpected checkpoint inventory")

    dataset_snapshot = Path(snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, allow_patterns=list(DATASET_FILES), token=api.token, local_dir=root / "dataset", max_workers=3))
    for relative, expected in DATASET_FILES.items():
        path = dataset_snapshot / relative
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"Dataset integrity mismatch: {relative}")
    sample_dir = root / "samples"
    sampling = build_samples(
        dataset_snapshot / "full/test.parquet",
        dataset_snapshot / "10k_all_consequences/test.parquet",
        dataset_snapshot / "20k_all_consequences/test.parquet",
        sample_dir,
    )
    for filename, expected in EXPECTED_GENERATED_SHA256.items():
        if digest(sample_dir / filename) != expected:
            raise RuntimeError(f"Generated sample hash mismatch: {filename}")

    parsed = urlparse(GENOME_S3_URI)
    s3 = storage()
    genome_key = parsed.path.lstrip("/")
    remote = s3.head_object(Bucket=parsed.netloc, Key=genome_key)
    if remote["ContentLength"] != GENOME_BYTES:
        raise RuntimeError("Staged genome size mismatch")
    compressed = root / "genome.fa.gz"
    s3.download_file(parsed.netloc, genome_key, str(compressed))
    if compressed.stat().st_size != GENOME_BYTES or digest(compressed) != GENOME_SHA256:
        raise RuntimeError("Downloaded genome integrity mismatch")
    genome = root / "genome.fa"
    with gzip.open(compressed, "rb") as source, genome.open("wb") as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    fasta = Fasta(str(genome), as_raw=True, rebuild=True)
    chroms = {str(key): len(fasta[key]) for key in fasta.keys()}
    for chrom in ("2", "4", "6", "8", "10"):
        if chrom not in chroms:
            raise RuntimeError(f"Genome lacks test chromosome {chrom}")
    fasta.close()

    result = {
        "checkpoint": selected,
        "model": str(model),
        "model_files": model_files,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "files": DATASET_FILES},
        "sampling": sampling,
        "samples": str(sample_dir),
        "union": str(sample_dir / "union.parquet"),
        "genome": {"path": str(genome), "compressed_s3_uri": GENOME_S3_URI, "compressed_bytes": GENOME_BYTES, "compressed_sha256": GENOME_SHA256, "chromosome_lengths": chroms},
        "prepare_seconds": time.time() - started,
    }
    write_json(root / "preparation.json", result)
    print(json.dumps({"phase": "af_inputs_prepared", "checkpoint": selected, "sampling": sampling, "prepare_seconds": result["prepare_seconds"]}, indent=2), flush=True)
    return result
