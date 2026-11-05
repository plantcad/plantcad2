"""Recover genomic provenance and freeze chromosome-held-out probe memberships."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mappy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyfaidx import Fasta

from exp586_evals.probe_protocol import visible_sequences

TEST_CHROMOSOMES = {
    "maize": {"1", "5", "9"},
    "tomato": {"SL4.0ch02", "SL4.0ch06", "SL4.0ch10"},
    "sorghum": {"1", "5", "9"},
    "arabidopsis": {"2", "4"},
    "rice": {"1", "5", "9"},
    "pharus": {"CM033319.1", "CM033323.1", "CM033327.1"},
}


def task_species(spec: dict[str, Any]) -> str:
    """Keep unrelated benchmark reference genomes distinct."""
    if spec["kind"] == "sv":
        return "arabidopsis"
    if spec["key"] == "conservation_andropogoneae":
        return "sorghum"
    if spec["kind"] == "conservation":
        return "pharus"
    return "tomato" if "tomato" in spec["key"] else "maize"


def file_digest(path: Path) -> str:
    """Compute a SHA-256 without materializing a complete input in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recover_location(aligner: Any, sequence: str) -> dict[str, Any]:
    """Map a central fragment, then the full window when uniqueness is unclear.

    Coordinates are 0-based half-open, expanded conservatively for unaligned
    query ends. They identify a source chromosome, not a variant-effect target.
    """
    sequence = sequence.upper()
    for offset, query in ((3072, sequence[3072:5120]), (0, sequence)):
        hits = list(aligner.map(query))
        accepted = [
            hit
            for hit in hits
            if hit.is_primary
            and hit.mapq >= 20
            and hit.mlen / max(hit.blen, 1) >= 0.98
            and hit.q_st <= (20 if offset else 3072)
            and hit.q_en >= (len(query) - 20 if offset else 5120)
        ]
        if len(accepted) != 1:
            continue
        hit = accepted[0]
        before, after = offset + hit.q_st, len(sequence) - offset - hit.q_en
        if hit.strand < 0:
            before, after = after, before
        return {
            "chrom": hit.ctg,
            "window_start": max(0, hit.r_st - before),
            "window_end": hit.r_en + after,
            "strand": hit.strand,
            "mapq": hit.mapq,
            "identity": hit.mlen / hit.blen,
            "mapping_query_bases": len(query),
        }
    return {
        "chrom": "",
        "window_start": -1,
        "window_end": -1,
        "strand": 0,
        "mapq": 0,
        "identity": 0.0,
        "mapping_query_bases": 0,
    }


def visible_digest(spec: dict[str, Any], row: dict[str, Any]) -> str:
    """Recognize identical visible information, including strand swaps."""
    values = visible_sequences(spec, row)
    if spec["kind"] == "sv":
        representations = (
            values["ref_fwd"] + "|" + values["alt_fwd"],
            values["ref_rc"] + "|" + values["alt_rc"],
        )
    else:
        representations = (
            values["fwd"] + "|" + values["rc"],
            values["rc"] + "|" + values["fwd"],
        )
    return hashlib.sha256(min(representations).encode()).hexdigest()


def assign_membership(frame: pd.DataFrame, species: str) -> pd.DataFrame:
    """Fix test chromosomes without consulting labels or model predictions."""
    frame = frame.copy()
    frame["split"] = "train"
    frame.loc[frame.chrom.isin(TEST_CHROMOSOMES[species]), "split"] = "test"
    frame.loc[frame.chrom.eq(""), "split"] = "unmapped"
    test_hashes = set(frame.loc[frame.split.eq("test"), "visible_sha256"])
    duplicate = frame.split.eq("train") & frame.visible_sha256.isin(test_hashes)
    frame.loc[duplicate, "split"] = "dropped_duplicate"
    train = frame.split.eq("train")
    test = frame.split.eq("test")
    if set(frame.loc[train, "chrom"]) & set(frame.loc[test, "chrom"]):
        raise ValueError("A source chromosome crosses the holdout boundary")
    if set(frame.loc[train, "visible_sha256"]) & test_hashes:
        raise ValueError("Identical visible sequences cross the holdout boundary")
    return frame


def prepare_sv_membership(source: Path, genome: Path, output: Path) -> dict[str, Any]:
    """Validate the provided 1-based closed SV coordinates against TAIR10.

    Known source coordinates retain repetitive variants that cannot be uniquely
    remapped. Internal deletion and reference-window intervals are half-open.
    """
    frame = pq.read_table(source).to_pandas()
    fasta = Fasta(str(genome), as_raw=True)
    metadata = []
    for index, row in enumerate(frame.to_dict("records")):
        chromosome = str(row["Chromosome"])
        begin, end = int(row["Start"]) - 1, int(row["End"])
        left, right = int(row["left"]) - 1, int(row["right"])
        if end - begin != int(row["DelLen"]) or right - left != int(row["DelLen"]):
            raise ValueError("Genomic and sequence deletion lengths differ")
        start = begin - left
        reference = fasta[chromosome][start : start + 8192].upper()
        if reference != row["RefSeq"].upper():
            raise ValueError(
                f"SV reference sequence disagrees with TAIR10 at row {index}"
            )
        retained = reference[:left] + reference[right:]
        offset = 4096 - left
        if row["MutSeq"][offset : offset + len(retained)].upper() != retained:
            raise ValueError(
                f"ALT does not contain the specified deletion at row {index}"
            )
        metadata.append(
            {
                "row_id": index,
                "label": int(row["label"]),
                "chrom": chromosome,
                "window_start": start,
                "window_end": start + 8192,
                "deletion_start": begin,
                "deletion_end": end,
                "deletion_length": end - begin,
                "visible_sha256": visible_digest({"kind": "sv"}, row),
            }
        )
    fasta.close()
    membership = assign_membership(pd.DataFrame(metadata), "arabidopsis")
    membership.to_parquet(output, index=False, compression="zstd")
    result = {
        "rows": len(frame),
        "reference_and_deletion_checks": "all rows exact",
        "source_sha256": file_digest(source),
        "reference_sha256": file_digest(genome),
        "membership_sha256": file_digest(output),
        "membership": membership.split.value_counts().to_dict(),
        "source_coordinates": "1-based closed",
        "internal_coordinates": "0-based half-open",
    }
    return result


def map_task(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    source_root: Path,
    output: Path,
    aligner: Any,
    *,
    threads: int = 2,
) -> dict[str, Any]:
    """Map every original row and persist labels with immutable source-row IDs."""
    rows: list[dict[str, Any]] = []
    offset = 0
    cache = output.parent / (output.stem + "-mapping-parts")
    cache.mkdir(parents=True, exist_ok=True)
    binding = {"spec": spec, "sources": records, "mapper": "central2048-full-v1"}
    binding_path = cache / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Mapping cache belongs to different source inputs")
    binding_path.write_text(json.dumps(binding, sort_keys=True) + "\n")
    for record in records:
        path = source_root / record["path"]
        if file_digest(path) != record["sha256"]:
            raise ValueError("Source Parquet identity differs")
        parquet = pq.ParquetFile(path)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for group in range(parquet.num_row_groups):
                part = cache / f"{offset:09d}.parquet"
                count = parquet.metadata.row_group(group).num_rows
                if part.exists():
                    cached = pq.read_table(part).to_pylist()
                    if [row["row_id"] for row in cached] != list(
                        range(offset, offset + count)
                    ):
                        raise ValueError("Cached genomic mapping row IDs differ")
                    rows.extend(cached)
                    offset += count
                    continue
                begin = len(rows)
                source = parquet.read_row_group(group).to_pylist()
                sequences = [row.get("sequence", row.get("RefSeq")) for row in source]
                locations = list(
                    pool.map(lambda seq: recover_location(aligner, seq), sequences)
                )
                for index, (row, location) in enumerate(
                    zip(source, locations, strict=True)
                ):
                    rows.append(
                        {
                            "row_id": offset + index,
                            "label": int(row["label"]),
                            "visible_sha256": visible_digest(spec, row),
                            **location,
                            **(
                                {"deletion_length": int(row["DelLen"])}
                                if spec["kind"] == "sv"
                                else {}
                            ),
                        }
                    )
                offset += len(source)
                temporary = part.with_suffix(".tmp")
                pq.write_table(
                    pa.Table.from_pylist(rows[begin:]), temporary, compression="zstd"
                )
                temporary.replace(part)
                if group % 10 == 0:
                    print(
                        json.dumps(
                            {"task": spec["key"], "mapped_rows_processed": offset}
                        ),
                        flush=True,
                    )
    species = task_species(spec)
    frame = assign_membership(pd.DataFrame(rows), species)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), output, compression="zstd"
    )
    result = {
        "task": spec["key"],
        "species": species,
        "rows": len(frame),
        "membership": {str(k): int(v) for k, v in frame.split.value_counts().items()},
        "labels_by_split": frame.groupby(["split", "label"]).size().to_dict(),
        "test_chromosomes": sorted(TEST_CHROMOSOMES[species]),
        "sha256": file_digest(output),
    }
    result["labels_by_split"] = {
        "/".join(map(str, key)): value
        for key, value in result["labels_by_split"].items()
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--species", choices=list(TEST_CHROMOSOMES), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    catalog = json.loads(args.catalog.read_text())
    index = args.reference_root / (
        "maize-k21w31.mmi" if args.species == "maize" else args.species + ".mmi"
    )
    aligner = mappy.Aligner(
        str(index),
        preset="splice:hq" if args.species == "pharus" else "map-hifi",
        n_threads=2,
        best_n=3,
    )
    if aligner is None:
        raise ValueError("Reference index is unavailable")
    results = []
    for spec in catalog["tasks"]:
        species = task_species(spec)
        if (
            spec["kind"] not in ("conservation", "core", "sv")
            or species != args.species
        ):
            continue
        output = args.output / (spec["key"] + ".parquet")
        if output.exists():
            print(
                json.dumps(
                    {"reusing_membership": str(output), "sha256": file_digest(output)}
                ),
                flush=True,
            )
            continue
        results.append(
            map_task(
                spec,
                manifest["inputs"][spec["file"]]["files"],
                args.source_root,
                output,
                aligner,
            )
        )
    (args.output / (args.species + "-mapping.json")).write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
