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
    # R: read.table(header=F) -> V1, V2, V3
    bed = pd.read_csv(args.bed, sep='\t', header=None, usecols=[0,1,2], names=bed_cols, dtype={'Chrom': str, 'Start': int, 'End': int})
    
    # 2. Read TSV
    print(f"Reading TSV: {args.tsv}")
    tsv_cols = ['Chrom', 'Start', 'End', 'Ref', 'Score', 'RefProb', 'AltProbs']
    # R: read.table(header=T, quote='', stringsAsFactors=F)
    # We need to handle the header issue we saw earlier (header lines in middle of file)
    try:
        df = pd.read_csv(args.tsv, sep='\t', header=None, names=tsv_cols, dtype=str)
    except Exception as e:
        print(f"Error reading TSV: {e}")
        sys.exit(1)
        
    # Filter headers
    # R: bed <- read.table(..., header=T) implies it handles one header. 
    # But we saw multiple headers in previous attempts.
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
        
    # We need to parse AltProbs
    # AltProbs is "p1,p2,p3"
    # We can split it
    alt_probs = df['AltProbs'].str.split(',', expand=True).astype(float)
    
    # Vectorized assignment
    # Iterate over each Ref type
    for ref_base in bases:
        # Mask for rows where Ref == ref_base
        mask = (df['Ref'] == ref_base)
        if not mask.any():
            continue
            
        # Assign RefProb
        df.loc[mask, f'prob_{ref_base}'] = df.loc[mask, 'RefProb']
        
        # Assign AltProbs
        # The alt bases are bases excluding ref_base, sorted alphabetically (implied by setdiff in R usually? 
        # R setdiff(x, y): "The elements of setdiff(x,y) are those elements in x but not in y."
        # R bases <- c("A", "C", "G", "T")
        # setdiff(bases, "A") -> "C", "G", "T" (preserves order of x)
        # So yes, A, C, G, T order excluding Ref.
        
        alt_bases = [b for b in bases if b != ref_base]
        
        # alt_probs has columns 0, 1, 2 corresponding to the 3 alt bases
        for i, alt_b in enumerate(alt_bases):
            df.loc[mask, f'prob_{alt_b}'] = alt_probs.loc[mask, i]
            
    # Now df has prob_A, prob_C, prob_G, prob_T
    
    # 4. Iterate over peaks and construct result
    print("Constructing tensor...")
    num_peaks = len(bed)
    # R: resDF <- rbind(resDF, curDF)
    # We want (N, 201, 4)
    final_tensor = np.full((num_peaks, 201, 4), 0.25, dtype=np.float32)
    
    # To speed up, we can index the TSV by Chrom
    # But filtering by range is still needed.
    # Since N is small (982), we can just filter.
    # But let's at least group by Chrom to avoid full scan.
    df_grouped = dict(tuple(df.groupby('Chrom')))
    
    for i in tqdm(range(num_peaks)):
        chrom = bed.loc[i, 'Chrom']
        # R: curstart <- peaks$V2[i] + 1
        # BED V2 is Start (0-based).
        # So curstart is Start + 1.
        curstart = bed.loc[i, 'Start'] + 1
        
        # R: curend <- peaks$V3[i]
        curend = bed.loc[i, 'End']
        
        # R: curBED <- unique(bed_new[which(bed_new$chr == curchr & bed_new$end >= curstart & bed_new$end <= curend), ])
        # Note: unique() in R removes duplicate rows.
        
        if chrom not in df_grouped:
            # No data for this chrom
            # curDF is all zeros (already initialized in final_tensor)
            continue
            
        sub_df = df_grouped[chrom]
        
        # Filter range
        # bed_new$end >= curstart & bed_new$end <= curend
        # In Python:
        mask = (sub_df['End'] >= curstart) & (sub_df['End'] <= curend)
        curBED = sub_df[mask].copy()
        
        # Unique
        # R: unique(curBED). 
        # In pandas: drop_duplicates()
        curBED = curBED.drop_duplicates()
        
        # R: curBED <- curBED[order(curBED$start), c(1:3,9:12)]
        # Sort by start
        curBED = curBED.sort_values('Start')
        
        # Extract probs
        # Columns: prob_A, prob_C, prob_G, prob_T
        prob_cols = ['prob_A', 'prob_C', 'prob_G', 'prob_T']
        
        # R: if(nrow(curBED) == 201)
        if len(curBED) == 201:
            # Take values directly
            vals = curBED[prob_cols].values
            final_tensor[i, :, :] = vals
        else:
            # R: curDF <- data.frame(pos = curstart:curend, ...)
            # R: idx <- match(curBED$end, curDF$pos)
            # R: curDF[idx, 2:5] <- curBED[, 4:7]
            
            # Create target positions
            # pos goes from curstart to curend (inclusive)
            # Length should be 201
            # In Python, we map these positions to indices 0..200
            
            # Map curBED['End'] to index
            # index = curBED['End'] - curstart
            
            indices = (curBED['End'] - curstart).values
            
            # Filter indices that are valid (should be all given the range filter, but safety check)
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
