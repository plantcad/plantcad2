"""Map retained preview rows onto full inputs for a no-inference overlap check.

Run on an existing allocation. Downloads fixtures there, not on the laptop;
returns only compact indices and hashes. Does not modify evaluation inputs.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from common import HF_REPO, MANIFEST_SHA256, PILOT, REFERENCE_PREFIX, REFERENCE_REVISION, digest, storage, upload_file, write_json
from inputs import read_frame

sys.path.insert(0, str(PILOT / "sensitivity"))
from run_condition import LEADERBOARD_TASKS


def row_hashes(frame, kind):
    columns = ["RefSeq", "MutSeq", "left", "right"] if kind == "sv" else ["sequence"]
    labeled = kind != "motif"
    if labeled:
        columns.append("label")
    for row in frame[columns].itertuples(index=False, name=None):
        values = [str(value) for value in row]
        if kind == "sv":
            values[2:] = [str(int(value)) for value in row[2:]]
        elif labeled:
            values[-1] = str(int(row[-1]))
        yield hashlib.sha256("\0".join(values).encode()).digest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s3-prefix", required=True)
    args = parser.parse_args()
    preparation = json.loads((args.root / "preparation.json").read_text())
    assert preparation["sampling"]["method"] == "full"
    usage = shutil.disk_usage(args.root)
    if (usage.used + 4 * 2**30) / usage.total >= 0.90:
        raise RuntimeError("Preview fixture staging would exceed 90% disk usage")
    staging = args.root / "overlap-reference-download"
    snapshot = Path(snapshot_download(HF_REPO, revision=REFERENCE_REVISION, allow_patterns=[f"{REFERENCE_PREFIX}/samples/*"], token=os.environ["HUGGING_FACE_HUB_TOKEN"], local_dir=staging, max_workers=4))
    samples = snapshot / REFERENCE_PREFIX / "samples"
    assert digest(samples / "manifest.json") == MANIFEST_SHA256
    manifest = {Path(row["path"]).name: row for row in json.loads((samples / "manifest.json").read_text())}
    indices, records = {}, {}
    for spec in LEADERBOARD_TASKS:
        path = samples / spec["file"]
        assert digest(path) == manifest[spec["file"]]["sha256"]
        preview = pd.read_csv(path, sep="\t")
        preview_hashes = list(row_hashes(preview, spec["kind"]))
        wanted = set(preview_hashes)
        matched, duplicates = {}, 0
        size = preparation["inputs"][spec["file"]]["rows"]
        columns = ["RefSeq", "MutSeq", "left", "right", "label"] if spec["kind"] == "sv" else ["sequence"] + ([] if spec["kind"] == "motif" else ["label"])
        for start in range(0, size, 4096):
            frame = read_frame(preparation, spec, start, min(start + 4096, size), columns)
            for offset, fingerprint in enumerate(row_hashes(frame, spec["kind"])):
                if fingerprint in wanted:
                    if fingerprint in matched:
                        duplicates += 1
                    else:
                        matched[fingerprint] = start + offset
        assert set(matched) == wanted, (spec["key"], len(wanted - set(matched)))
        indices[spec["key"]] = np.array([matched[value] for value in preview_hashes], dtype=np.int32)
        records[spec["key"]] = {"preview_rows": len(preview), "unique_preview_rows": len(wanted), "full_rows": size, "additional_identical_full_rows": duplicates, "fixture_sha256": manifest[spec["file"]]["sha256"]}
        print(json.dumps({"matched_task": spec["key"], **records[spec["key"]]}), flush=True)
    output = args.root / "independent-sample-overlap"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "sample-indices.npz", **indices)
    write_json(output / "sample-indices.json", {"manifest_sha256": preparation["manifest_sha256"], "reference_manifest_sha256": MANIFEST_SHA256, "reference_revision": REFERENCE_REVISION, "sha256": digest(output / "sample-indices.npz"), "tasks": records, "method": "Exact input strings, binary labels, and SV coordinates matched by SHA-256; identical duplicate rows use their first full occurrence"})
    s3 = storage()
    for filename in ("sample-indices.npz", "sample-indices.json"):
        upload_file(s3, output / filename, f"{args.s3_prefix}/validation/{filename}")
    # This directory was created exclusively by this diagnostic; staged full
    # evaluation inputs and checkpoints are never touched.
    shutil.rmtree(staging)
    print(json.dumps({"sample_overlap_complete": True, "bytes": (output / "sample-indices.npz").stat().st_size, "fixture_staging_removed": not staging.exists()}), flush=True)


if __name__ == "__main__":
    main()
