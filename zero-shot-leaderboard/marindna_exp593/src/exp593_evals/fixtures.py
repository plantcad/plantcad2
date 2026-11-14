"""Prepare finite real-data fixtures before acquiring any GPU capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from exp593_dna.pipeline import load_encoder, load_registry
from exp593_evals.prompts import matched_fim
from exp593_evals.readouts import canonical_ids, normalize_dna


def prepare(manifest: Path, samples: Path, output: Path) -> None:
    source = json.loads(manifest.read_text())
    tokenizer = load_encoder().tokenizer
    lookup = dict(zip("ACGTacgtNn", canonical_ids(tokenizer), strict=True))
    record = source["inputs"]["conservation_within_andropogoneae__test.tsv"]["files"][0]
    path = samples / record["path"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError("Actual benchmark fixture file differs from manifest")
    rows = pq.ParquetFile(path).read_row_group(0, columns=["sequence"]).to_pylist()[:32]
    cases = []
    for index, row in enumerate(rows):
        dna = normalize_dna(row["sequence"], case="lower")
        cases.append(
            {
                "name": f"sorghum-ar-lower-{index:02d}",
                "ids": [lookup[b] for b in dna],
                "target_positions": list(range(1, len(dna))),
                "row": index,
            }
        )
    species = load_registry()["Sbicolor"]
    fields = {
        "species": "Sorghum bicolor",
        "assembly": "Sbicolor",
        "taxonomy": species["taxonomy_path"],
    }
    if not isinstance(fields["taxonomy"], str):
        raise TypeError("Training taxonomy must be the actual registry string")
    for index in range(4):
        dna = normalize_dna(rows[index]["sequence"], case="lower")
        cases.append(
            {
                "name": f"sorghum-ar-upper-{index:02d}",
                "ids": [lookup[b.upper()] for b in dna],
                "target_positions": list(range(1, len(dna))),
                "row": index,
            }
        )
        # These are backend/prompt gates, not benchmark accuracy estimates.
        start, stop = len(dna) // 2 - 1, len(dna) // 2 + 2
        for kind, value in matched_fim(
            tokenizer, dna[:start], dna[stop:], dna[start:stop], fields
        ).items():
            cases.append(
                {"name": f"sorghum-fim-{kind}-{index:02d}", "row": index, **value}
            )
    result = {
        "purpose": "Native/Torch readout validation on real unchanged benchmark DNA; not task-score estimates",
        "source": record,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.samples, args.output)


if __name__ == "__main__":
    main()
