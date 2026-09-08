#!/usr/bin/env python3
"""Create deterministic TSV samples for causal eval task sets."""

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import datasets
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


@dataclass(frozen=True)
class TaskSpec:
    category: str
    task: str
    split: str
    label_column: str | None = None


REPRESENTATIVE_TASKS = (
    TaskSpec("evo_cons", "conservation_within_poaceae_tis", "test", "label"),
    TaskSpec("evo_cons", "conservation_within_poaceae_non_tis", "test", "label"),
    TaskSpec("motif_acc", "tis_recovery", "test_maize"),
    TaskSpec("motif_acc", "donor_recovery", "test_tomato"),
    TaskSpec("core_noncore", "tis_core_noncore_classification", "test_maize", "label"),
    TaskSpec("core_noncore", "donor_core_noncore_classification", "test_tomato", "label"),
    TaskSpec("sv_effect", "structural_variant_effect_prediction", "test", "label"),
)

SENSITIVITY_TASKS = (
    TaskSpec("evo_cons", "conservation_within_poaceae_non_tis", "test", "label"),
    TaskSpec("motif_acc", "acceptor_recovery", "test_tomato"),
    TaskSpec("core_noncore", "tis_core_noncore_classification", "test_maize", "label"),
    TaskSpec("sv_effect", "structural_variant_effect_prediction", "test", "label"),
)

LEADERBOARD_TASKS = (
    TaskSpec("evo_cons", "conservation_within_andropogoneae", "test", "label"),
    TaskSpec("evo_cons", "conservation_within_poaceae_non_tis", "test", "label"),
    TaskSpec("evo_cons", "conservation_within_poaceae_tis", "test", "label"),
    *(
        TaskSpec("motif_acc", f"{motif}_recovery", f"test_{species}")
        for species in ("maize", "tomato")
        for motif in ("tis", "tts", "donor", "acceptor")
    ),
    *(
        TaskSpec(
            "core_noncore",
            f"{motif}_core_noncore_classification",
            f"test_{species}",
            "label",
        )
        for species in ("maize", "tomato")
        for motif in ("tis", "tts", "donor", "acceptor")
    ),
    TaskSpec("sv_effect", "structural_variant_effect_prediction", "test", "label"),
)

TASK_SETS = {
    "representative": REPRESENTATIVE_TASKS,
    "sensitivity": SENSITIVITY_TASKS,
    "leaderboard": LEADERBOARD_TASKS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_seed(seed: int, spec: TaskSpec) -> int:
    digest = hashlib.blake2b(
        f"{spec.task}/{spec.split}".encode(), digest_size=4
    ).digest()
    return seed ^ int.from_bytes(digest, "big")


def _sample_rows(
    dataset,
    samples: int,
    label_column: str | None,
    seed: int,
    balance_binary: bool,
    shuffle_buffer_size: int,
) -> list[dict]:
    shuffled = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    if label_column is None or not balance_binary:
        rows = list(itertools.islice(shuffled, samples))
        if len(rows) != samples:
            raise RuntimeError(f"requested {samples} rows, found {len(rows)}")
        return rows

    targets = {0: samples // 2, 1: samples - samples // 2}
    rows_by_label: dict[int, list[dict]] = {0: [], 1: []}
    for row in shuffled:
        label = int(row[label_column])
        if label in targets and len(rows_by_label[label]) < targets[label]:
            rows_by_label[label].append(row)
        if all(len(rows_by_label[label]) == target for label, target in targets.items()):
            break
    if any(len(rows_by_label[label]) != target for label, target in targets.items()):
        found = {label: len(rows) for label, rows in rows_by_label.items()}
        raise RuntimeError(f"could not construct balanced sample: requested={targets}, found={found}")

    rows = rows_by_label[0] + rows_by_label[1]
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="plantcad/PlantCAD2_zero_shot_tasks")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=472)
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="representative")
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=0,
        help="Streaming shuffle buffer; 0 chooses max(1024, samples * 32).",
    )
    parser.add_argument(
        "--shared-seed",
        action="store_true",
        help="Use --seed directly for every task instead of deriving a task-specific seed.",
    )
    parser.add_argument(
        "--balance-binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stratify binary tasks 50/50; disable for a prevalence-preserving random sample.",
    )
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        raise ValueError("samples must be an even integer of at least 2")
    shuffle_buffer_size = args.shuffle_buffer_size or max(1_024, args.samples * 32)
    if shuffle_buffer_size < args.samples:
        raise ValueError("shuffle-buffer-size must be at least samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = HfApi().repo_info(
        args.repo_id, repo_type="dataset", revision=args.revision
    ).sha
    manifest = []
    for spec in TASK_SETS[args.task_set]:
        seed = args.seed if args.shared_seed else _task_seed(args.seed, spec)
        dataset = load_dataset(
            args.repo_id,
            spec.task,
            split=spec.split,
            streaming=True,
            revision=resolved_revision,
        )
        rows = _sample_rows(
            dataset,
            args.samples,
            spec.label_column,
            seed,
            balance_binary=args.balance_binary,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        frame = pd.DataFrame(rows)
        output_path = output_dir / f"{spec.task}__{spec.split}.tsv"
        frame.to_csv(output_path, sep="\t", index=False)
        output_sha256 = _sha256_file(output_path)
        label_counts = (
            frame[spec.label_column].value_counts().sort_index().to_dict()
            if spec.label_column
            else None
        )
        record = {
            "category": spec.category,
            "repo_id": args.repo_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "datasets_version": datasets.__version__,
            "pandas_version": pd.__version__,
            "task": spec.task,
            "split": spec.split,
            "samples": len(frame),
            "seed": seed,
            "shuffle_buffer_size": shuffle_buffer_size,
            "sampling": (
                "balanced_binary"
                if spec.label_column and args.balance_binary
                else "random_unstratified"
            ),
            "label_counts": label_counts,
            "path": str(output_path),
            "sha256": output_sha256,
        }
        print(json.dumps(record, sort_keys=True))
        manifest.append(record)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Some datasets streaming transports leave a non-daemon network thread alive after
    # all requested rows have been materialized. The command has no remaining cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
