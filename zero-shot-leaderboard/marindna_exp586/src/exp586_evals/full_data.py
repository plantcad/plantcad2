"""Prepare immutable full-split inputs for the exp586 MoE evaluations."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_url
from pyfaidx import Fasta

from exp586_evals.artifacts import sha256
from exp586_evals.protocol import REFERENCE_COMMIT, original
from exp586_evals.real_data import TASK_REVISION, af_membership

DATASET_REPO = "plantcad/PlantCAD2_zero_shot_tasks"
REFERENCE_SHA256 = "058292473dc0dc7e4dbc384a97b381e75dbad533665d93b5ea9e8cd216d73b1e"


def assign_task_files(
    tasks: list[dict], inputs: dict, workers: int
) -> dict[tuple[str, str], int]:
    """Balance immutable Parquet files without duplicating a file across workers."""
    if workers < 1 or not tasks:
        raise ValueError("Need positive workers and task inputs")
    units = [
        (task["key"], record["path"], record["rows"])
        for task in tasks
        for record in inputs[task["file"]]["files"]
    ]
    if any(rows < 1 for _, _, rows in units):
        raise ValueError("Task input files must be nonempty")
    loads = [0] * workers
    assignment = {}
    for task, path, rows in sorted(units, key=lambda item: (-item[2], item[:2])):
        worker = min(range(workers), key=lambda index: (loads[index], index))
        assignment[(task, path)] = worker
        loads[worker] += rows
    return assignment


def task_plan(
    tasks: list[dict], inputs: dict, workers: int, *, chunk_size: int = 2048
) -> list[dict]:
    if chunk_size < 8 or chunk_size % 8:
        raise ValueError("Task chunks must preserve eight-example inference batches")
    assignment = assign_task_files(tasks, inputs, workers)
    plan, task_offset = [], {task["key"]: 0 for task in tasks}
    for task in tasks:
        for record in inputs[task["file"]]["files"]:
            source_start = task_offset[task["key"]]
            for local_start in range(0, record["rows"], chunk_size):
                start = source_start + local_start
                stop = source_start + min(local_start + chunk_size, record["rows"])
                plan.append(
                    {
                        "task": task["key"],
                        "start": start,
                        "stop": stop,
                        "worker": assignment[(task["key"], record["path"])],
                        "source_path": record["path"],
                        "source_start": source_start,
                        "key": f"{task['key']}--{start:06d}-{stop:06d}",
                    }
                )
            task_offset[task["key"]] += record["rows"]
    return plan


def task_inventory(tasks: list[dict], staged_prefix: str) -> tuple[dict, dict]:
    tree = {
        item.path: item
        for item in HfApi().list_repo_tree(
            DATASET_REPO,
            repo_type="dataset",
            revision=TASK_REVISION,
            recursive=True,
            expand=True,
        )
        if getattr(item, "path", "").endswith(".parquet")
    }
    fs = HfFileSystem()
    inputs, sources = {}, {}
    for task in tasks:
        config, split = task["file"].removesuffix(".tsv").split("__")
        paths = sorted(path for path in tree if path.startswith(f"{config}/{split}-"))
        if not paths:
            raise ValueError(f"No immutable Parquet sources for {task['key']}")
        records, rows = [], 0
        for path in paths:
            item = tree[path]
            if item.lfs is None or item.size != item.lfs.size:
                raise ValueError(f"Missing LFS identity for {path}")
            source = f"datasets/{DATASET_REPO}@{TASK_REVISION}/{path}"
            with fs.open(source, "rb", block_size=2**20) as stream:
                metadata = pq.ParquetFile(stream).metadata
            groups = [
                metadata.row_group(index).num_rows
                for index in range(metadata.num_row_groups)
            ]
            if sum(groups) != metadata.num_rows:
                raise ValueError(f"Row-group inventory differs for {path}")
            records.append(
                {
                    "path": "tasks/" + path,
                    "uri": staged_prefix.rstrip("/") + "/tasks/" + path,
                    "sha256": item.lfs.sha256,
                    "bytes": item.size,
                    "rows": metadata.num_rows,
                    "row_groups": groups,
                    "source_path": path,
                    "source_uri": hf_hub_url(
                        DATASET_REPO,
                        path,
                        repo_type="dataset",
                        revision=TASK_REVISION,
                    ),
                }
            )
            rows += metadata.num_rows
        inputs[task["file"]] = {"rows": rows, "files": records}
        sources[task["key"]] = {
            "rows": rows,
            "files": paths,
            "dataset_revision": TASK_REVISION,
        }
    return inputs, sources


def prepare_af(genome: Path, output: Path) -> tuple[dict, dict, dict]:
    frame, source = af_membership()
    protocol = original()
    fasta = Fasta(str(genome), as_raw=True)
    frame["sequence"] = [
        protocol.af.extract_window(
            fasta, str(row.chrom), int(row.pos), row.ref.upper(), 8192, "fwd"
        )
        for row in frame.itertuples()
    ]
    if any(
        len(sequence) != 8192 or set(sequence) - set("ACGT")
        for sequence in frame.sequence
    ):
        raise ValueError("Historical full AF membership contains an invalid window")
    crop = protocol.af.validate_center_crop(fasta, next(frame.itertuples()), 8192)
    fasta.close()
    frame["in_sample_94075"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, row_group_size=128, compression="zstd")
    record = {
        "path": "af/maize-af.parquet",
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "rows": len(frame),
    }
    return record, source, crop


def prepare(
    catalog: Path,
    genome: Path,
    output: Path,
    staged_prefix: str,
    workers: int,
) -> None:
    if sha256(genome) != REFERENCE_SHA256:
        raise ValueError("Reference genome differs from the pinned maize assembly")
    protocol = original()
    tasks = list(protocol.tasks.LEADERBOARD_TASKS)
    catalog_data = json.loads(catalog.read_text())
    if tasks != catalog_data["tasks"]:
        raise ValueError("Task catalog differs from PR20")
    inputs, sources = task_inventory(tasks, staged_prefix)
    task_sizes = {task["key"]: inputs[task["file"]]["rows"] for task in tasks}
    if sum(task_sizes.values()) != catalog_data["full_rows"]:
        raise ValueError("Full task row inventory differs")
    af_path = output.parent / "af/maize-af.parquet"
    af_record, af_source, crop = prepare_af(genome, af_path)
    af_record["uri"] = staged_prefix.rstrip("/") + "/af/maize-af.parquet"
    inputs["maize-af.parquet"] = {"rows": af_record["rows"], "files": [af_record]}
    af_chunks = protocol.partition.plan_chunks(
        {"maize-af": af_record["rows"]},
        workers=workers,
        chunk_size=128,
        batch_size=8,
    )
    preparation = {
        "schema": 5,
        "real_data_full": True,
        "protocol_commit": REFERENCE_COMMIT,
        "input_format": "parquet",
        "samples": str(output.parent.resolve()),
        "inputs": inputs,
        "plan": task_plan(tasks, inputs, workers),
        "af_plan": [{**vars(chunk), "key": chunk.key} for chunk in af_chunks],
        "sampling": {
            "total_rows": sum(task_sizes.values()),
            "pilot_only": False,
            "full_splits": True,
        },
        "workers": workers,
        "reference": {
            "assembly": "Zm-B73-REFERENCE-NAM-5.0",
            "ensembl_release": 62,
            "fasta_sha256": REFERENCE_SHA256,
        },
        "catalog_sha256": sha256(catalog),
        "dataset_revision": TASK_REVISION,
        "task_sources": sources,
        "af_source": af_source,
        "crop_validation": crop,
    }
    output.write_text(json.dumps(preparation, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256(output),
                "task_rows": sum(task_sizes.values()),
                "af_rows": af_record["rows"],
                "task_files": sum(len(value["files"]) for value in inputs.values()) - 1,
                "workers": workers,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staged-prefix", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
