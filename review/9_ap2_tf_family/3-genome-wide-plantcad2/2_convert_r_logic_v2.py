#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Convert zero_shot_score output to N*4 dataframe (TSV) with variable peak lengths.')
    parser.add_argument('--bed', required=True, help='Input BED file')
    parser.add_argument('--tsv', required=True, help='Input TSV file')
    parser.add_argument('--output', required=True, help='Output TSV file')
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
    bases = ['A', 'C', 'G', 'T']
    for b in bases:
        df[f'prob_{b}'] = 0.0
        
    alt_probs = df['AltProbs'].str.split(',', expand=True).astype(float)
    
    for ref_base in bases:
        mask = (df['Ref'] == ref_base)
        if not mask.any():
            continue
        df.loc[mask, f'prob_{ref_base}'] = df.loc[mask, 'RefProb']
        alt_bases = [b for b in bases if b != ref_base]
        for i, alt_b in enumerate(alt_bases):
            df.loc[mask, f'prob_{alt_b}'] = alt_probs.loc[mask, i]
            
    # 4. Iterate over peaks and construct result
    print("Constructing dataframe...")
    df_grouped = dict(tuple(df.groupby('Chrom')))
    
    results = []
    
    for i in tqdm(range(len(bed))):
        chrom = bed.loc[i, 'Chrom']
        curstart = bed.loc[i, 'Start'] + 1
        curend = bed.loc[i, 'End']
        
        # Prepare target ends for this peak
        target_ends = np.arange(curstart, curend + 1)
        target_df = pd.DataFrame({'End': target_ends})
        
        if chrom in df_grouped:
            sub_df = df_grouped[chrom]
            # Filter range
            mask = (sub_df['End'] >= curstart) & (sub_df['End'] <= curend)
            curBED = sub_df[mask].copy()
            curBED = curBED.drop_duplicates(subset=['End']) # Ensure unique End
            
            # Merge
            merged = pd.merge(target_df, curBED[['End', 'prob_A', 'prob_C', 'prob_G', 'prob_T']], on='End', how='left')
        else:
            # No data for chrom, create empty with NaNs
            merged = target_df
            for col in ['prob_A', 'prob_C', 'prob_G', 'prob_T']:
                merged[col] = np.nan
        
        # Fill missing with 0.25
        merged = merged.fillna(0.25)
        
        # Append just the probs
        results.append(merged[['prob_A', 'prob_C', 'prob_G', 'prob_T']])
        
    print("Concatenating results...")
    final_df = pd.concat(results, ignore_index=True)
    
    print(f"Saving to {args.output}")
    final_df.to_csv(args.output, sep='\t', index=False)
    print("Done.")

if __name__ == '__main__':
    main()
