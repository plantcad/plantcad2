"""Extract small independent metric targets from already-staged complete inputs.

Run on a live node without changing any GPU worker or its evaluation files.
No model inference and no full-sequence downloads to the submitter are needed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import PILOT, digest, storage, upload_file, write_json
from inputs import read_frame

sys.path.insert(0, str(PILOT / "sensitivity"))
from run_condition import LEADERBOARD_TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s3-prefix", required=True)
    args = parser.parse_args()
    preparation = json.loads((args.root / "preparation.json").read_text())
    if preparation["sampling"]["method"] != "full":
        raise ValueError("Independent targets require the complete dataset")
    targets = {}
    for spec in LEADERBOARD_TASKS:
        size = preparation["inputs"][spec["file"]]["rows"]
        labels, nucleotides = [], []
        for start in range(0, size, 8192):
            frame = read_frame(preparation, spec, start, min(start + 8192, size), ["label"] if spec["kind"] == "sv" else None)
            if "label" in frame:
                labels.append(frame["label"].to_numpy(dtype=np.int8))
            if spec["kind"] != "sv":
                # Independent direct base extraction, not the evaluator's
                # sequence transform or metric helpers. Non-ACGT is -1.
                values = np.stack([frame["sequence"].str[position].str.upper().map({"A": 0, "C": 1, "G": 2, "T": 3}).fillna(-1).to_numpy(dtype=np.int8) for position in spec["positions"]], axis=1)
                nucleotides.append(values)
        if labels:
            targets[f"{spec['key']}__label"] = np.concatenate(labels)
        if nucleotides:
            left = np.concatenate(nucleotides)
            targets[f"{spec['key']}__left__true"] = left
            reverse = left[:, ::-1]
            targets[f"{spec['key']}__right_reverse_complement__true"] = np.where(reverse >= 0, 3 - reverse, -1).astype(np.int8)
        print(json.dumps({"validation_targets_task": spec["key"], "rows": size}), flush=True)
    output = args.root / "independent-validation-targets"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "targets.npz", **targets)
    write_json(output / "targets.json", {"manifest_sha256": preparation["manifest_sha256"], "sampling": preparation["sampling"], "arrays": {key: list(value.shape) for key, value in targets.items()}, "sha256": digest(output / "targets.npz"), "method": "Direct original-sequence target extraction; RC target equals complement of reversed left targets; no evaluator metric/transform helpers used"})
    s3 = storage()
    for filename in ("targets.npz", "targets.json"):
        upload_file(s3, output / filename, f"{args.s3_prefix}/validation/{filename}")
    print(json.dumps({"validation_targets_complete": True, "bytes": (output / "targets.npz").stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
