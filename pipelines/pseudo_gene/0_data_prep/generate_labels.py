#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14:44 2024/12/03
@author: JINGJING ZHAI (jz963@cornell.edu; zhaijingjing603@gmail.com)
Description: Generate labels for junction fastas based on NAM pan gene table
"""

import pandas as pd
from Bio import SeqIO
import argparse

def flatten_dataframe(df, col_to_split='P39', separator=';'):
    # Convert to string type first to handle NaN values
    df[col_to_split] = df[col_to_split].astype(str)
    
    # Split and explode
    flattened = df.assign(**{col_to_split: df[col_to_split].str.split(separator)})\
                 .explode(col_to_split)\
                 .reset_index(drop=True)
    
    return flattened

def main():
    parser = argparse.ArgumentParser(description="Extract translation start and stop site sequences from GFF3 and genome FASTA")
    parser.add_argument("--input", required=True, help="Input fasta file")
    parser.add_argument("--taxa", required=True, type=str, help="Taxa name")
    parser.add_argument("--output", required=True, help="Output fasta file")
    args = parser.parse_args()

    # Load NAM pan gene table
    nam_pan_gene_table = pd.read_csv("/workdir/jz963/Expression_modeling/a2z/a2z_dataset/tasks/exp-max/pan_gene_matrix_v3_cyverse.csv", low_memory=False)

    with open(args.input) as handle:
        input_fasta = list(SeqIO.parse(handle, "fasta"))

    # Extract transcript IDs
    seq_ids = [record.id for record in input_fasta]
    transcript_ids = ['_'.join(seq_id.split('_')[:2]) for seq_id in seq_ids]

    # Process pan gene table
    subset_pan_gene = nam_pan_gene_table[[args.taxa, 'class']]
    subset_pan_gene = subset_pan_gene.dropna()
    subset_pan_gene = subset_pan_gene[subset_pan_gene.iloc[:, 0].str.startswith('Zm')]

    # Create label mapping
    flattened_subset_pan_gene = flatten_dataframe(subset_pan_gene, col_to_split=args.taxa)
    class_dict = dict(zip(flattened_subset_pan_gene[args.taxa], flattened_subset_pan_gene['class']))
    labels = [class_dict.get(transcript_id, 'Unknown') for transcript_id in transcript_ids]

    transcript_labels_df = pd.DataFrame({'Transcript_ID': transcript_ids, 'Label': labels})
    
    # Get indices where Label is not 'Unknown'
    keep_indices = transcript_labels_df[transcript_labels_df['Label'] != 'Unknown'].index

    # Filter both FASTA and DataFrame using indices
    filtered_fasta = [input_fasta[i] for i in keep_indices]
    filtered_labels_df = transcript_labels_df.iloc[keep_indices]

    # Verify order matches (optional)
    for record, row in zip(filtered_fasta, filtered_labels_df.itertuples()):
        transcript_id = '_'.join(record.id.split('_')[:2])
        assert transcript_id == row.Transcript_ID, f"Mismatch: {transcript_id} != {row.Transcript_ID}"

    # Write filtered sequences
    with open(args.output, 'w') as output_handle:
        SeqIO.write(filtered_fasta, output_handle, "fasta")

    # Optionally save filtered labels
    filtered_labels_df.to_csv(args.output.replace('.fasta', '_labels.tsv'), sep='\t', index=False)

    # Print statistics
    print(f"Input sequences: {len(input_fasta)}")
    print(f"Filtered sequences: {len(filtered_fasta)}")
    print(f"Removed sequences: {len(input_fasta) - len(filtered_fasta)}")

if __name__ == "__main__":
    main()