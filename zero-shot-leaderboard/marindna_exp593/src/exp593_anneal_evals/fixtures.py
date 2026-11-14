"""Freeze real task, complete-prefix and nativeAR fixtures before GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from exp593_anneal_evals.exhaustive import candidates
from exp593_anneal_evals.scoring import contexts, rc_dna
from exp593_dna.pipeline import load_encoder
from exp593_evals.readouts import canonical_ids, normalize_dna


def prepare(panel: Path, manifest: Path, samples: Path, output: Path):
    import pyarrow.parquet as pq

    tokenizer = load_encoder().tokenizer
    dna_ids = canonical_ids(tokenizer)

    class Adapter:
        def __init__(self):
            self.tokenizer = tokenizer

        def ids(self, text):
            return [
                dna_ids[4 + "acgt".index(b)] if b in "acgt" else dna_ids[9]
                for b in normalize_dna(text, case="lower")
            ]

    adapter = Adapter()
    panel_data = json.loads(panel.read_text())
    source = json.loads(manifest.read_text())
    cases = []
    record = source["inputs"]["conservation_within_andropogoneae__test.tsv"]["files"][0]
    path = samples / record["path"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
            raise ValueError("AR source checksum mismatch")
    rows = pq.ParquetFile(path).read_row_group(0, columns=["sequence"]).to_pylist()[:32]
    for i, row in enumerate(rows):
        ids = adapter.ids(row["sequence"])
        cases.append(
            {
                "name": f"ar-window-{i:02d}",
                "kind": "ar-window",
                "ids": ids,
                "target_positions": list(range(1, len(ids))),
            }
        )
    for task, item in panel_data["tasks"].items():
        spec = item["spec"]
        if spec["kind"] not in ["conservation", "core", "motif"]:
            continue
        selected = item["rows"][: 128 if spec["kind"] == "motif" else 64]
        positions = spec["positions"]
        length = len(positions)
        for i, row in enumerate(selected):
            for presentation in ["ar", "fim"]:
                f, r, answers = contexts(
                    adapter, [row["sequence"]], positions, presentation
                )
                answer = answers[0]
                for strand, prompt, target in [
                    ("fwd", f[0], answer),
                    ("rc", r[0], rc_dna(answer)),
                ]:
                    name = f"{task}-{i:03d}-{presentation}-{strand}"
                    if presentation == "ar":
                        ids = adapter.ids(
                            row["sequence"]
                            if strand == "fwd"
                            else rc_dna(row["sequence"])
                        )
                        query_positions = (
                            positions
                            if strand == "fwd"
                            else sorted(8191 - p for p in positions)
                        )
                    else:
                        ids = prompt + adapter.ids(target)
                        query_positions = list(range(len(prompt), len(prompt) + length))
                    cases.append(
                        {
                            "name": name,
                            "kind": "tf",
                            "task": task,
                            "presentation": presentation,
                            "strand": strand,
                            "example": i,
                            "answer": target,
                            "task_kind": spec["kind"],
                            "label": row.get("label"),
                            "ids": ids,
                            "target_positions": query_positions,
                        }
                    )
                    if spec["kind"] == "core" or spec["kind"] == "motif" and i < 16:
                        for depth in range(length):
                            prefixes = [""] if depth == 0 else candidates(depth)
                            for prefix in prefixes:
                                ids = prompt + adapter.ids(prefix) + adapter.ids("a")
                                cases.append(
                                    {
                                        "name": name + "-prefix-" + (prefix or "empty"),
                                        "kind": "candidate-prefix",
                                        "task": task,
                                        "presentation": presentation,
                                        "strand": strand,
                                        "example": i,
                                        "gap_length": length,
                                        "prefix": prefix,
                                        "ids": ids,
                                        "target_positions": [len(ids) - 1],
                                    }
                                )
    # Explicit boundary/variant one-base cases exercise every added task family.
    from exp593_anneal_evals.prompts import matched_flanks, prompt_ids
    from exp593_anneal_evals.scoring import sv_positions

    for task, item in panel_data["tasks"].items():
        if item["spec"]["kind"] not in {"sv", "af"}:
            continue
        for i, row in enumerate(item["rows"]):
            if item["spec"]["kind"] == "sv":
                pieces = [
                    (field, normalize_dna(row[field], case="lower"), points)
                    for field, points in zip(
                        ("RefSeq", "MutSeq"), sv_positions(row), strict=True
                    )
                ]
                for allele_index, (allele, sequence, points) in enumerate(pieces):
                    valid = [j for j, point in enumerate(points) if 0 < point < 8192]
                    cases.append(
                        {
                            "name": f"{task}-{i}-{allele}-AR",
                            "kind": "sv-ar",
                            "task": task,
                            "ids": adapter.ids(sequence),
                            "target_positions": [points[j] for j in valid],
                            "sv_query": {
                                "presentation": "ar",
                                "row": i,
                                "allele": allele_index,
                                "boundaries": valid,
                                "strand": "fwd",
                            },
                        }
                    )
            else:
                pieces = [("REF", normalize_dna(row["sequence"], case="lower"), [4096])]
            for allele, sequence, points in pieces:
                for boundary_index, point in enumerate(points):
                    if point <= 0:
                        continue
                    if point >= 8191:
                        raise ValueError(
                            "A FIM boundary lacks its required right flank"
                        )
                    left, right = matched_flanks(
                        tokenizer, sequence[:point], sequence[point + 1 :], 1
                    )
                    for strand, l, r, a in [
                        ("fwd", left, right, sequence[point]),
                        ("rc", rc_dna(right), rc_dna(left), rc_dna(sequence[point])),
                    ]:
                        prompt = prompt_ids(tokenizer, l, r, 1)
                        ids = prompt + adapter.ids(a)
                        cases.append(
                            {
                                "name": f"{task}-{i}-{allele}-{point}-{strand}",
                                "kind": "single-base",
                                "task": task,
                                "ids": ids,
                                "target_positions": [len(ids) - 1],
                                **(
                                    {
                                        "sv_query": {
                                            "presentation": "fim",
                                            "row": i,
                                            "allele": 0 if allele == "RefSeq" else 1,
                                            "boundaries": [boundary_index],
                                            "strand": strand,
                                        }
                                    }
                                    if item["spec"]["kind"] == "sv"
                                    else {}
                                ),
                            }
                        )
    if any(len(c["ids"]) > 8192 or min(c["target_positions"]) <= 0 for c in cases):
        raise ValueError("Invalid real fixture geometry")
    value = {
        "purpose": "Native lower-only practical validation; true answers never choose candidate prefixes",
        "tokenizer_sha256": hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
        "source_panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
        "cases": cases,
        "sv_rows": panel_data["tasks"]["sv_impact"]["rows"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "bytes": output.stat().st_size,
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["panel", "manifest", "samples", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.panel, a.manifest, a.samples, a.output)


if __name__ == "__main__":
    main()
