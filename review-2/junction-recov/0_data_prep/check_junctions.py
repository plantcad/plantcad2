#!/usr/bin/env python3
"""
Check that junction TSV files contain canonical motifs at expected positions.

Expected sequence layout (all 8192 bp):
  start/stop sites : upstream=4094 + codon(3) + downstream=4095  → motif at [4094:4097]
  donor/acceptor   : upstream=4095 + dinucleotide(2) + downstream=4095 → motif at [4095:4097]

Canonical motifs:
  start    : ATG
  stop     : TAA, TAG, TGA
  donor    : GT
  acceptor : AG
"""

import argparse
import pandas as pd

CANONICAL = {
    'start':    ({'ATG'},           4094, 4097),
    'stop':     ({'TAA', 'TAG', 'TGA'}, 4094, 4097),
    'donor':    ({'GT'},            4095, 4097),
    'acceptor': ({'AG'},            4095, 4097),
}


def check(tsv_file, junction_type):
    motifs, start, end = CANONICAL[junction_type]
    df = pd.read_csv(tsv_file, sep='\t')

    if 'sequence' not in df.columns:
        raise ValueError(f"No 'sequence' column found in {tsv_file}. Columns: {list(df.columns)}")

    total = len(df)
    extracted = df['sequence'].str[start:end].str.upper()
    canonical_mask = extracted.isin(motifs)
    n_canonical = canonical_mask.sum()

    print(f"\n{'='*60}")
    print(f"File     : {tsv_file}")
    print(f"Type     : {junction_type}  |  expected motif at [{start}:{end}]: {motifs}")
    print(f"Total    : {total}")
    print(f"Canonical: {n_canonical} ({100*n_canonical/total:.1f}%)")
    print(f"Non-canonical examples (up to 10):")
    non_canon = extracted[~canonical_mask].value_counts().head(10)
    for motif, count in non_canon.items():
        print(f"  {motif!r:10s}  {count}")


def main():
    parser = argparse.ArgumentParser(description="Check canonical junction motifs in TSV files")
    parser.add_argument("--results-dir", default="../results", help="Directory with TSV files (default: ../results)")
    parser.add_argument("--prefix", nargs='+',
                        default=["GCF_000001405.26_GRCh38", "GCF_000001635.27_GRCm39"],
                        help="Species prefixes to check")
    args = parser.parse_args()

    for prefix in args.prefix:
        for jtype, (_, _, _) in CANONICAL.items():
            fname_map = {
                'start':    f"{args.results_dir}/{prefix}_start_sites.tsv",
                'stop':     f"{args.results_dir}/{prefix}_stop_sites.tsv",
                'donor':    f"{args.results_dir}/{prefix}_donor.tsv",
                'acceptor': f"{args.results_dir}/{prefix}_acceptor.tsv",
            }
            check(fname_map[jtype], jtype)


if __name__ == "__main__":
    main()
