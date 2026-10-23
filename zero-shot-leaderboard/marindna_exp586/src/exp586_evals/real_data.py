"""Prepare a bounded real-data pilot with immutable historical memberships."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from pyfaidx import Fasta

from exp586_evals.artifacts import sha256
from exp586_evals.protocol import REFERENCE_COMMIT, original

TASK_REVISION = "d340debe0c8402c84f0696cd2002f87c2f7ba6db"
ARCHIVE_REVISION = "69656cc98d0ad0d17203e1d18b213c47a9d8187e"
ARCHIVE_PREFIX = (
    "exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2/results/step-206144/coreweave"
)


def sample_task(
    task: dict, files: list[str], fs: HfFileSystem
) -> tuple[list[dict], dict]:
    config, split = task["file"].removesuffix(".tsv").split("__")
    groups, offset = [], 0
    for path in sorted(p for p in files if p.startswith(config + "/" + split + "-")):
        with fs.open(
            f"datasets/plantcad/PlantCAD2_zero_shot_tasks@{TASK_REVISION}/{path}",
            "rb",
            block_size=2**20,
        ) as source:
            meta = pq.ParquetFile(source).metadata
        for index in range(meta.num_row_groups):
            size = meta.row_group(index).num_rows
            groups.append((path, index, offset, size))
            offset += size
    if len(groups) < 4:
        raise ValueError("Pilot requires four row groups")
    rng = np.random.default_rng(582)
    rows, selected = [], []
    for i in np.linspace(0, len(groups) - 1, 4, dtype=int):
        path, group, begin, size = groups[i]
        with fs.open(
            f"datasets/plantcad/PlantCAD2_zero_shot_tasks@{TASK_REVISION}/{path}",
            "rb",
            block_size=2**20,
        ) as source:
            table = pq.ParquetFile(source).read_row_group(group)
        take = sorted(rng.choice(size, 32, replace=False).tolist())
        rows.extend(
            {**row, "source_row": begin + index}
            for index, row in zip(take, table.take(take).to_pylist(), strict=True)
        )
        selected.append(
            {"path": path, "row_group": group, "rows": [begin + x for x in take]}
        )
    return rows, {
        "full_rows": offset,
        "sampling": "32 seed-582 rows from each of four evenly spaced parquet row groups",
        "groups": selected,
    }


def af_membership() -> tuple[pd.DataFrame, dict]:
    def download(suffix: str) -> Path:
        return Path(
            hf_hub_download(
                "plantcad/marindna-exp472",
                ARCHIVE_PREFIX + suffix,
                revision=ARCHIVE_REVISION,
            )
        )

    split_path = download("/plantcad2-maize-af-probe-022t-20260904-v1/split.parquet")
    input_path = download("/plantcad2-maize-af-022t-20260904-v1/predictions.parquet")
    # Read only input columns; no dense predictions are recomputed or used.
    columns = [
        "row_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "AF",
        "consequence",
        "original_consequence",
        "in_sample_100000",
    ]
    frame = pd.read_parquet(input_path, columns=columns)
    frame = frame[frame.in_sample_100000].drop(columns="in_sample_100000")
    split = pd.read_parquet(split_path)
    digest = hashlib.sha256(
        split[["row_id", "split", "block_id", "af_bin"]]
        .sort_values("row_id")
        .to_csv(index=False)
        .encode()
    ).hexdigest()
    if (
        digest != "c37e713eb0449af48f99a14de5e3c86fab8cb5504c4872ccfd24e8988f0b89a0"
        or len(frame) != 94075
    ):
        raise ValueError("Historical membership/split mismatch")
    frame = frame.merge(split, on="row_id", validate="1:1")
    if set(frame[frame.split.eq("train")].block_id) & set(
        frame[frame.split.eq("test")].block_id
    ):
        raise ValueError("Train/test blocks overlap")
    return frame.sort_values("row_id").reset_index(drop=True), {
        "archive_revision": ARCHIVE_REVISION,
        "historical_split_sha256": digest,
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "input_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "train": int(frame.split.eq("train").sum()),
        "test": int(frame.split.eq("test").sum()),
    }


def af_inputs() -> tuple[pd.DataFrame, dict]:
    frame, source = af_membership()
    samples = []
    for (_, membership), group in frame[frame.split.isin(["train", "test"])].groupby(
        ["consequence", "split"], sort=True
    ):
        samples.append(
            group.sample(n=128 if membership == "train" else 64, random_state=582)
        )
    pilot = pd.concat(samples).sort_values("row_id").reset_index(drop=True)
    if len(pilot) != 2112 or pilot.row_id.duplicated().any():
        raise ValueError("Pilot AF inventory differs")
    # Check against ALL historical test windows, including unsampled ones.
    for chrom, group in pilot[pilot.split == "train"].groupby("chrom"):
        positions = np.sort(
            frame[(frame.chrom == chrom) & (frame.split == "test")].pos.to_numpy()
        )
        for position in group.pos:
            j = np.searchsorted(positions, position)
            if any(abs(position - p) < 8192 for p in positions[max(0, j - 1) : j + 1]):
                raise ValueError("Training window overlaps historical test window")
    return pilot, {
        **source,
        "rows": len(pilot),
        "train": 1408,
        "test": 704,
    }


def prepare(catalog: Path, genome: Path, output: Path) -> None:
    reference = {
        "assembly": "Zm-B73-REFERENCE-NAM-5.0",
        "ensembl_release": 62,
        "fasta_sha256": sha256(genome),
        "source": "s3://marin-us-east-02a/MarinDNA/plantcad2-evals/references/zea_mays/release-62/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz",
    }
    if (
        reference["fasta_sha256"]
        != "058292473dc0dc7e4dbc384a97b381e75dbad533665d93b5ea9e8cd216d73b1e"
    ):
        raise ValueError("Reference genome differs from pinned assembly")
    protocol = original()
    tasks = protocol.tasks.LEADERBOARD_TASKS
    if list(tasks) != json.loads(catalog.read_text())["tasks"]:
        raise ValueError("Task catalog differs from PR20")
    files = HfApi().list_repo_files(
        "plantcad/PlantCAD2_zero_shot_tasks",
        repo_type="dataset",
        revision=TASK_REVISION,
    )
    fs = HfFileSystem()
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs, sources = {}, {}

    def save_frame(name: str, frame: pd.DataFrame) -> dict:
        path = output.parent / name
        frame.to_parquet(path, index=False, row_group_size=64)
        return {"path": name, "sha256": sha256(path), "rows": len(frame)}

    for task in tasks:
        rows, sources[task["key"]] = sample_task(task, files, fs)
        frame = pd.DataFrame(rows)
        if task["kind"] != "motif" and set(frame.label) != {0, 1}:
            raise ValueError("Pilot must cover both classes")
        sequence_columns = (
            ["RefSeq", "MutSeq"] if task["kind"] == "sv" else ["sequence"]
        )
        for column in sequence_columns:
            if any(
                len(x) != 8192 or set(x.upper()) - set("ACGTN") for x in frame[column]
            ):
                raise ValueError("Unexpected task window/alphabet")
        source = save_frame(task["file"].removesuffix(".tsv") + ".parquet", frame)
        inputs[task["file"]] = {"rows": len(frame), "files": [source]}
        print(f"Prepared {task['key']}: {len(frame)} original rows", flush=True)
    af, af_source = af_inputs()
    fasta = Fasta(str(genome), as_raw=True)
    af["sequence"] = [
        protocol.af.extract_window(
            fasta, str(row.chrom), int(row.pos), row.ref.upper(), 8192, "fwd"
        )
        for row in af.itertuples()
    ]
    if any(set(seq) - set("ACGT") for seq in af.sequence):
        raise ValueError("Historical AF pilot includes ambiguous bases")
    crop = protocol.af.validate_center_crop(fasta, next(af.itertuples()), 8192)
    fasta.close()
    af["in_sample_2112"] = True
    af_file = save_frame("maize-af.parquet", af)
    inputs["maize-af.parquet"] = {"rows": len(af), "files": [af_file]}
    chunks = protocol.partition.plan_chunks(
        {spec["key"]: 128 for spec in tasks}, workers=1, chunk_size=64, batch_size=8
    )
    af_chunks = protocol.partition.plan_chunks(
        {"maize-af": len(af)}, workers=1, chunk_size=32, batch_size=8
    )
    preparation = {
        "schema": 4,
        "real_data_pilot": True,
        "protocol_commit": REFERENCE_COMMIT,
        "input_format": "parquet",
        "samples": str(output.parent.resolve()),
        "inputs": inputs,
        "plan": [{**vars(c), "key": c.key} for c in chunks],
        "af_plan": [{**vars(c), "key": c.key} for c in af_chunks],
        "sampling": {
            "total_rows": 2560,
            "rows_per_task": 128,
            "seed": 582,
            "pilot_only": True,
        },
        "reference": reference,
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
                "task_rows": 2560,
                "af_rows": len(af),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--genome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.catalog, args.genome, args.output)


if __name__ == "__main__":
    main()
