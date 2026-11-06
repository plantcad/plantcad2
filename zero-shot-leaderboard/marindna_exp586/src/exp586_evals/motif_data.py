"""Bind whole-motif outcomes to chromosome splits and exact cached flank inputs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mappy
import pandas as pd
import pyarrow.parquet as pq

from exp586_evals.classification_data import (
    assign_membership,
    recover_location,
    task_species,
    visible_digest,
)
from exp586_evals.probe_protocol import visible_sequences


def motif_code(sequence: str) -> int:
    """Use a fixed base-four alphabet; ambiguous targets cannot be scored."""
    result = 0
    for base in sequence.upper():
        if base not in "ACGT":
            return -1
        result = 4 * result + "ACGT".index(base)
    return result


def prepare_task(
    spec: dict[str, Any],
    sources: list[dict[str, Any]],
    source_root: Path,
    core_membership: Path,
    core_sources: list[dict[str, Any]],
    core_source_root: Path,
    reference_index: Path,
    output: Path,
) -> dict[str, Any]:
    """Reuse only identical visible inputs; recover coordinates for other rows."""
    # The existing core protocol is precisely the required whole-motif exclusion.
    flank_spec = {**spec, "kind": "core"}
    core = pd.read_parquet(core_membership)
    core_rows = core.to_dict("records")
    core_by_hash = {}
    conflicts = set()
    offset = 0
    for record in core_sources:
        parquet = pq.ParquetFile(core_source_root / record["path"])
        for batch in parquet.iter_batches(batch_size=1000, columns=["sequence"]):
            for source in batch.to_pylist():
                row = core_rows[offset]
                assert int(row["row_id"]) == offset
                assert visible_digest(flank_spec, source) == row["visible_sha256"]
                key = ordered_visible_digest(flank_spec, source)
                if key in core_by_hash and core_by_hash[key]["chrom"] != row["chrom"]:
                    conflicts.add(key)
                core_by_hash.setdefault(key, row)
                offset += 1
    assert offset == len(core)
    for key in conflicts:
        core_by_hash.pop(key)
    aligner = None
    rows = []
    missing = []
    for record in sources:
        parquet = pq.ParquetFile(source_root / record["path"])
        for batch in parquet.iter_batches(batch_size=1000, columns=["sequence"]):
            for source in batch.to_pylist():
                sequence = source["sequence"].upper()
                target = "".join(sequence[position] for position in spec["positions"])
                key = visible_digest(flank_spec, source)
                cached = core_by_hash.get(ordered_visible_digest(flank_spec, source))
                if cached is None:
                    if aligner is None:
                        aligner = mappy.Aligner(str(reference_index), preset="map-hifi")
                    location = recover_location(aligner, sequence)
                    origin_row = len(missing)
                    missing.append({"sequence": sequence, "label": 0})
                else:
                    location = {
                        name: cached[name]
                        for name in (
                            "chrom",
                            "window_start",
                            "window_end",
                            "strand",
                            "mapq",
                            "identity",
                            "mapping_query_bases",
                        )
                    }
                    origin_row = int(cached["row_id"])
                rows.append(
                    {
                        "row_id": len(rows),
                        "target_motif": target,
                        "label": motif_code(target),
                        "visible_sha256": key,
                        "feature_origin": "cached_core"
                        if cached is not None
                        else "new_motif",
                        "feature_row_id": origin_row,
                        **location,
                    }
                )
            print(
                json.dumps(
                    {"task": spec["key"], "rows": len(rows), "new_inputs": len(missing)}
                ),
                flush=True,
            )
    frame = assign_membership(pd.DataFrame(rows), task_species(spec))
    frame.loc[frame.label.lt(0), "split"] = "invalid_target"
    required = frame.split.isin(["train", "test"]) & frame.feature_origin.eq(
        "new_motif"
    )
    required_missing = []
    for index in frame.index[required]:
        original = int(frame.loc[index, "feature_row_id"])
        frame.loc[index, "feature_row_id"] = len(required_missing)
        required_missing.append(missing[original])
    frame.loc[~required & frame.feature_origin.eq("new_motif"), "feature_row_id"] = -1
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "membership.parquet", index=False)
    import pyarrow as pa

    pq.write_table(
        pa.Table.from_pylist(
            required_missing,
            schema=pa.schema(
                [
                    ("sequence", pa.string()),
                    ("label", pa.int64()),
                ]
            ),
        ),
        output / "missing.parquet",
        compression="zstd",
        row_group_size=512,
    )
    result = {
        "task": spec,
        "rows": len(frame),
        "cached_core_rows": int(frame.feature_origin.eq("cached_core").sum()),
        "new_input_rows": len(missing),
        "new_required_input_rows": len(required_missing),
        "split_counts": frame.split.value_counts().to_dict(),
        "outcome_counts": dict(Counter(frame.target_motif)),
        "train_outcome_counts": dict(
            Counter(frame.loc[frame.split.eq("train"), "target_motif"])
        ),
        "test_outcome_counts": dict(
            Counter(frame.loc[frame.split.eq("test"), "target_motif"])
        ),
        "matching": "SHA256 of both complete withheld flanks in their exact forward/RC order; strand swaps are not reused",
    }
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if "outcome" not in k}), flush=True
    )
    return result


def ordered_visible_digest(spec: dict[str, Any], row: dict[str, Any]) -> str:
    """Directional probes require identical ordering, as well as identical flanks."""
    sequences = visible_sequences(spec, row)
    return hashlib.sha256(
        (sequences["fwd"] + "|" + sequences["rc"]).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--core-memberships", type=Path, required=True)
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--core-source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    spec = next(t for t in manifest["tasks"] if t["key"] == args.task)
    index = "tomato.mmi" if "tomato" in spec["key"] else "maize-k21w31.mmi"
    core_file = spec["file"].replace("_recovery__", "_core_noncore_classification__")
    core_sources = json.loads(args.core_manifest.read_text())["inputs"][core_file][
        "files"
    ]
    prepare_task(
        spec,
        manifest["inputs"][spec["file"]]["files"],
        args.source_root,
        args.core_memberships / (spec["key"].replace("motif_", "core_") + ".parquet"),
        core_sources,
        args.core_source_root,
        args.reference_root / index,
        args.output / spec["key"],
    )


if __name__ == "__main__":
    main()
