"""Real benchmark targets for practical backend comparisons, selected before inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exp593_dna.pipeline import load_encoder
from exp593_evals.metadata import metadata_for
from exp593_evals.prompts import matched_fim
from exp593_evals.readouts import canonical_ids, normalize_dna


def selected(frame, count: int, *, balanced: bool, seed: int):
    rng = np.random.default_rng(seed)
    if balanced:
        parts = []
        for label in (0, 1):
            values = np.flatnonzero(frame.label.to_numpy() == label)
            if len(values) < count // 2:
                raise ValueError("Not enough source examples of both classes")
            parts.extend(rng.choice(values, count // 2, replace=False).tolist())
        indices = np.sort(parts)
    else:
        indices = np.sort(rng.choice(len(frame), count, replace=False))
    result = frame.iloc[indices].copy()
    result["panel_source_row"] = (
        frame.iloc[indices]["panel_source_row"].to_numpy()
        if "panel_source_row" in frame
        else indices
    )
    return result.reset_index(drop=True)


def prepare(
    root: Path, output: Path, *, af_only: bool = False, af_count: int = 32
) -> None:
    sys.path.insert(0, str(root / "scientific-code"))
    from exp586_evals.protocol import original

    protocol = original()
    manifest = json.loads((root / "full.json").read_text())
    manifest["samples"] = str(root / "samples")
    tokenizer = load_encoder().tokenizer
    dna_ids = canonical_ids(tokenizer)
    lookup = dict(zip("ACGTacgtNn", dna_ids, strict=True))
    cases, tasks, sources = [], {}, {}

    def add(
        sequence, positions, base, *, metadata, fim: bool = True, full: bool = False
    ):
        sequence = normalize_dna(sequence, case="lower")
        answer = "".join(sequence[p] for p in positions)
        targets = ["acgt".find(x) for x in answer]
        for case in ("lower", "upper"):
            text = sequence if case == "lower" else sequence.upper()
            cases.append(
                {
                    **base,
                    "readout": "ar-" + case,
                    "ids": [lookup[x] for x in text],
                    "target_positions": positions,
                    "target_bases": targets,
                    "full_vocab": full,
                }
            )
        if fim:
            if positions != list(range(min(positions), max(positions) + 1)):
                raise ValueError("FIM panel gap must be contiguous")
            lo, hi = min(positions), max(positions) + 1
            paired = matched_fim(
                tokenizer, sequence[:lo], sequence[hi:], sequence[lo:hi], metadata
            )
            if (
                paired["minimal"]["left"] != paired["metadata"]["left"]
                or paired["minimal"]["right"] != paired["metadata"]["right"]
            ):
                raise ValueError("FIM variants do not share the same visible DNA")
            for kind, rendered in paired.items():
                # Do not repeat long visible flanks in every serialized case.
                cases.append(
                    {
                        **base,
                        "readout": "fim-" + kind,
                        "ids": rendered["ids"],
                        "target_positions": rendered["target_positions"],
                        "target_bases": targets,
                        "full_vocab": False,
                        "prompt_length": rendered["prompt_length"],
                    }
                )

    for number, spec in enumerate(() if af_only else protocol.tasks.LEADERBOARD_TASKS):
        source = manifest["inputs"][spec["file"]]
        import pyarrow.parquet as pq

        from exp593_evals.artifacts import fetch

        count = 128 if spec["kind"] == "motif" else 16 if spec["kind"] == "sv" else 64
        candidates, records, offset = [], [], 0
        for record in source["files"]:
            path = root / "samples" / record["path"]
            if not path.exists():
                cached = fetch(record, root / "cache" / record["sha256"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(cached.resolve())
            with path.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != record["sha256"]
                ):
                    raise ValueError("Panel input differs from its retained manifest")
            sources[record["path"]] = record
            records.append((path, offset))
            if spec["kind"] == "motif":
                candidate = pd.DataFrame(
                    {"panel_source_row": np.arange(offset, offset + record["rows"])}
                )
            else:
                candidate = pq.read_table(path, columns=["label"]).to_pandas()
                candidate["panel_source_row"] = np.arange(
                    offset, offset + len(candidate)
                )
            candidates.append(candidate)
            candidate = pd.concat(candidates, ignore_index=True)
            if (
                spec["kind"] == "motif"
                or (
                    candidate.label.value_counts().reindex([0, 1], fill_value=0)
                    >= count // 2
                ).all()
            ):
                break
            offset += record["rows"]
        chosen = selected(
            candidate, count, balanced=spec["kind"] != "motif", seed=593000 + number
        ).panel_source_row.to_numpy()
        pieces = []
        for path, offset in records:
            parquet = pq.ParquetFile(path)
            local = 0
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                points = chosen[
                    (chosen >= offset + local) & (chosen < offset + local + size)
                ]
                if len(points):
                    part = (
                        parquet.read_row_group(group)
                        .take(points - offset - local)
                        .to_pandas()
                    )
                    part["panel_source_row"] = points
                    pieces.append(part)
                local += size
        frame = (
            pd.concat(pieces, ignore_index=True)
            .sort_values("panel_source_row")
            .reset_index(drop=True)
        )
        tasks[spec["key"]] = {"spec": spec, "rows": frame.to_dict("records")}
        for row_index, row in enumerate(frame.to_dict("records")):
            base = {
                "task": spec["key"],
                "kind": spec["kind"],
                "row": row_index,
                "source_row": int(row["panel_source_row"]),
                "label": int(row.get("label", -1)),
            }
            if spec["kind"] == "sv":
                ref_points = [
                    *range(int(row["left"]) - 6, int(row["left"]) - 1),
                    *range(int(row["right"]), int(row["right"]) + 5),
                ]
                for allele, field, positions in (
                    ("ref", "RefSeq", ref_points),
                    ("alt", "MutSeq", list(range(4091, 4101))),
                ):
                    points = sorted({p for p in positions if p > 0})
                    add(
                        row[field],
                        points,
                        {
                            **base,
                            "allele": allele,
                            "context": "left",
                            "original_positions": points,
                        },
                        metadata=metadata_for(spec, row),
                        fim=False,
                    )
                    for position in points:
                        # The FIM-SV readout is leave-one-base-out at each old boundary target.
                        begin = len(cases)
                        add(
                            row[field],
                            [position],
                            {
                                **base,
                                "allele": allele,
                                "context": "left",
                                "original_positions": [position],
                            },
                            metadata=metadata_for(spec, row),
                        )
                        cases[begin:] = [
                            c for c in cases[begin:] if c["readout"].startswith("fim-")
                        ]
            else:
                for context in spec["contexts"]:
                    sequences, positions = (
                        protocol.tasks.EVAL._transform_sequences_and_positions(
                            frame.sequence.iloc[[row_index]], spec["positions"], context
                        )
                    )
                    add(
                        sequences.iloc[0],
                        positions,
                        {**base, "context": context, "original_positions": positions},
                        metadata=metadata_for(
                            spec,
                            row,
                            reverse_complement=context == "right_reverse_complement",
                        ),
                    )
        print(
            json.dumps(
                {"prepared": spec["key"], "examples": count, "cases": len(cases)}
            ),
            flush=True,
        )

    spec = {"key": "maize-af", "kind": "af", "file": "maize-af.parquet"}
    source = manifest["inputs"][spec["file"]]
    record = source["files"][0]
    with (root / "samples" / record["path"]).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError("AF source digest changed")
    sources[record["path"]] = record
    if af_only:
        import pyarrow.parquet as pq

        if len(source["files"]) != 1:
            raise ValueError(
                "Expanded AF sampler expects the retained single Parquet source"
            )
        parquet = pq.ParquetFile(root / "samples" / record["path"])
        chosen = np.sort(
            np.random.default_rng(593021).choice(
                source["rows"], af_count, replace=False
            )
        )
        pieces, offset = [], 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            points = chosen[(chosen >= offset) & (chosen < offset + size)]
            if len(points):
                part = parquet.read_row_group(group).take(points - offset).to_pandas()
                part["panel_source_row"] = points
                pieces.append(part)
            offset += size
        frame = (
            pd.concat(pieces, ignore_index=True)
            .sort_values("panel_source_row")
            .reset_index(drop=True)
        )
    else:
        frame = protocol.inputs.read_frame(manifest, spec, 0, min(4000, source["rows"]))
        frame = selected(frame, af_count, balanced=False, seed=593021)
    tasks[spec["key"]] = {"spec": spec, "rows": frame.to_dict("records")}
    for i, row in enumerate(frame.to_dict("records")):
        sequence = row["sequence"]
        if len(sequence) != 8192 or sequence[4096].upper() != row["ref"].upper():
            raise ValueError("AF source target alignment changed")
        alt = sequence[:4096] + row["alt"] + sequence[4097:]
        for strand in ("fwd", "rc"):
            position = protocol.af.variant_position(8192, strand)
            for allele, value in (("ref", sequence), ("alt", alt)):
                if strand == "rc":
                    value = protocol.af.reverse_complement(value)
                base = {
                    "task": "maize-af",
                    "kind": "af",
                    "row": i,
                    "source_row": int(row["panel_source_row"]),
                    "context": strand,
                    "allele": allele,
                    "original_positions": [position],
                }
                # AR suffix scoring retains the original target-and-following-base reduction.
                add(
                    value,
                    list(range(position, len(value))),
                    base,
                    metadata=metadata_for(spec, row, reverse_complement=strand == "rc"),
                    fim=False,
                    full=True,
                )
                if allele == "ref":
                    begin = len(cases)
                    add(
                        value,
                        [position],
                        base,
                        metadata=metadata_for(
                            spec, row, reverse_complement=strand == "rc"
                        ),
                    )
                    cases[begin:] = [
                        c for c in cases[begin:] if c["readout"].startswith("fim-")
                    ]
    for index, case in enumerate(cases):
        case["index"] = index
        if min(case["target_positions"]) <= 0 or max(case["target_positions"]) >= len(
            case["ids"]
        ):
            raise ValueError("Target has no causal predecessor")
    output.mkdir(parents=True, exist_ok=True)
    (output / "panel.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "selection_seed": 593000,
                "tokenizer_sha256": hashlib.sha256(
                    tokenizer.to_str().encode()
                ).hexdigest(),
                "sources": sources,
                "tasks": tasks,
                "cases": cases,
            },
            separators=(",", ":"),
        )
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "tasks": len(tasks),
                "path": str(output / "panel.json"),
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--af-only", action="store_true")
    p.add_argument("--af-count", type=int, default=32)
    a = p.parse_args()
    prepare(a.inputs, a.output, af_only=a.af_only, af_count=a.af_count)


if __name__ == "__main__":
    main()
