#!/usr/bin/env python3
"""Convert FASTA file(s) to TSV with columns: seq_id, sequence."""

import argparse
import sys
from Bio import SeqIO


def fasta_to_tsv(fasta_path, out):
    for record in SeqIO.parse(fasta_path, "fasta"):
        if len(record.seq) == 8192:
            out.write(f"{record.id}\t{str(record.seq)}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert FASTA to two-column TSV (seq_id, sequence)")
    parser.add_argument("fastas", nargs="+", help="Input FASTA file(s)")
    parser.add_argument("-o", "--output", default=None, help="Output TSV file (default: stdout)")
    args = parser.parse_args()

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        out.write("seq_id\tsequence\n")
        for fasta in args.fastas:
            fasta_to_tsv(fasta, out)
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
