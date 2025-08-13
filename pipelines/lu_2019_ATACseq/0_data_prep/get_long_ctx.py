# =====================================
# File: get_long_ctx.py
# Author: JINGJING ZHAI
# Created: 2025-07-24
# Last Modified: 2025-07-24
# Description: This script is used to extract long context sequences from the original curated data
# =====================================

import pyfaidx
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract long context sequences from the original curated data")
    parser.add_argument("--input", type=str, required=True, help="Path to the input TSV file")
    parser.add_argument("--fasta", type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output TSV file")
    parser.add_argument("--flank", type=int, default=500, help="Flank size to add to each sequence (default: 500)")
    args = parser.parse_args()

    # Load the input TSV file
    df = pd.read_csv(args.input, sep="\t")
    genome = pyfaidx.Fasta(args.fasta)

    expected_length = 600 + 2 * args.flank # 600 is the length of the original sequence
    
    # Iterate over each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        chrom = str(row['Chr'])
        start = int(row['Start']) - args.flank  # Adjust start position by flank size
        end = int(row['End']) + args.flank
        
        # Adjust to chromosome length
        if start < 0:
            start = 0
        
        if end > len(genome[chrom]):
            end = len(genome[chrom])
        
        # Extract the sequence from the genome
        seq = genome[chrom][start:end].seq
        
        # Replace the sequence column with the extracted sequence
        df.loc[index, 'Seq'] = seq

    # Drop rows with incorrect sequence lengths
    df = df[df['Seq'].str.len() == expected_length]
    
    # Save the modified DataFrame to output TSV
    df.to_csv(args.output, sep="\t", index=False)

if __name__ == "__main__":
    main()