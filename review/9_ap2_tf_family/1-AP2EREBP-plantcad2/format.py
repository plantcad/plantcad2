#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Convert zero_shot_score output to (N, 201, 4) npz format using R script logic.')
    parser.add_argument('--bed', required=True, help='Input BED file')
    parser.add_argument('--tsv', required=True, help='Input TSV file')
    parser.add_argument('--output', required=True, help='Output NPZ file')
    args = parser.parse_args()

    # 1. Read BED
    print(f"Reading BED: {args.bed}")
    bed_cols = ['Chrom', 'Start', 'End']
    bed = pd.read_csv(args.bed, sep='\t', header=None, usecols=[0,1,2], names=bed_cols, dtype={'Chrom': str, 'Start': int, 'End': int})
    
    # 2. Read TSV
    print(f"Reading TSV: {args.tsv}")
    tsv_cols = ['Chrom', 'Start', 'End', 'Ref', 'Score', 'RefProb', 'AltProbs']
    try:
        df = pd.read_csv(args.tsv, sep='\t', header=None, names=tsv_cols, dtype=str)
    except Exception as e:
        print(f"Error reading TSV: {e}")
        sys.exit(1)
        
    # Filter headers
    df = df[df['Start'].str.lower() != 'start']
    
    # Convert types
    df['Start'] = df['Start'].astype(int)
    df['End'] = df['End'].astype(int)
    df['RefProb'] = df['RefProb'].astype(float)
    
    # 3. Process Probabilities
    print("Processing probabilities...")
    # Initialize A, C, G, T columns with 0
    bases = ['A', 'C', 'G', 'T']
    for b in bases:
        df[f'prob_{b}'] = 0.0
        

    alt_probs = df['AltProbs'].str.split(',', expand=True).astype(float)
    
    # Vectorized assignment
    for ref_base in bases:
        mask = (df['Ref'] == ref_base)
        if not mask.any():
            continue
            
        df.loc[mask, f'prob_{ref_base}'] = df.loc[mask, 'RefProb']
        
        alt_bases = [b for b in bases if b != ref_base]
        
        for i, alt_b in enumerate(alt_bases):
            df.loc[mask, f'prob_{alt_b}'] = alt_probs.loc[mask, i]
            
    # Now df has prob_A, prob_C, prob_G, prob_T
    
    # 4. Iterate over peaks and construct result
    print("Constructing tensor...")
    num_peaks = len(bed)
    final_tensor = np.full((num_peaks, 201, 4), 0.25, dtype=np.float32)
    
    df_grouped = dict(tuple(df.groupby('Chrom')))
    
    for i in tqdm(range(num_peaks)):
        chrom = bed.loc[i, 'Chrom']
        curstart = bed.loc[i, 'Start'] + 1
        
        curend = bed.loc[i, 'End']
        
        if chrom not in df_grouped:
            continue
            
        sub_df = df_grouped[chrom]
        
        mask = (sub_df['End'] >= curstart) & (sub_df['End'] <= curend)
        curBED = sub_df[mask].copy()
        
        curBED = curBED.drop_duplicates()
        
        curBED = curBED.sort_values('Start')
        
        prob_cols = ['prob_A', 'prob_C', 'prob_G', 'prob_T']
        
        if len(curBED) == 201:
            vals = curBED[prob_cols].values
            final_tensor[i, :, :] = vals
        else:
            
            indices = (curBED['End'] - curstart).values
            
            valid_mask = (indices >= 0) & (indices < 201)
            indices = indices[valid_mask]
            vals = curBED.loc[valid_mask, prob_cols].values
            
            # Assign to tensor
            # final_tensor is already zeros
            final_tensor[i, indices, :] = vals
            
    print(f"Saving to {args.output}")
    np.savez(args.output, probs=final_tensor)
    print("Done.")

if __name__ == '__main__':
    main()
