#!/usr/bin/env python3
"""
Add mean phastCons scores to test.tsv file.
For each region (chr, start, end), calculate the mean phastCons score.
"""

import sys
from collections import defaultdict

# Path to phastCons file
phastcons_file = "../../results/review/7_phastcons/Ath_PhastCons.txt"
test_file = "/workdir/jz963/utils/plantcad2/results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_scores.tsv"
output_file = "/workdir/jz963/utils/plantcad2/results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_scores_with_phastcons.tsv"

print("Loading phastCons scores...", file=sys.stderr)
# Dictionary to store phastCons scores: (chr, pos) -> score
phastcons = {}

with open(phastcons_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        if line_num % 10000000 == 0:
            print(f"Loaded {line_num} phastCons entries...", file=sys.stderr)

        parts = line.strip().split()
        if len(parts) != 2:
            continue

        chr_pos, score = parts
        chrom, pos = chr_pos.split('_')
        pos = int(pos)
        score = float(score)

        phastcons[(chrom, pos)] = score

print(f"Loaded {len(phastcons)} phastCons scores.", file=sys.stderr)

print("Processing test.tsv...", file=sys.stderr)
# Process test.tsv and add mean phastCons scores
with open(test_file, 'r') as infile, open(output_file, 'w') as outfile:
    header = infile.readline().strip()
    # Add phastCons column to header
    outfile.write(header + "\tmean_phastCons\n")

    for line_num, line in enumerate(infile, 1):
        if line_num % 1000 == 0:
            print(f"Processed {line_num} regions...", file=sys.stderr)

        parts = line.strip().split('\t')
        chrom = parts[0]
        start = int(parts[1])  # 1-based
        end = int(parts[2])    # 1-based

        # Calculate mean phastCons score for this region
        scores = []
        for pos in range(start, end + 1):  # inclusive range
            if (chrom, pos) in phastcons:
                scores.append(phastcons[(chrom, pos)])

        # Calculate mean (or NA if no scores found)
        if scores:
            mean_score = sum(scores) / len(scores)
            outfile.write(line.strip() + f"\t{mean_score:.6f}\n")
        else:
            outfile.write(line.strip() + "\tNA\n")

print(f"Done! Output written to: {output_file}", file=sys.stderr)
