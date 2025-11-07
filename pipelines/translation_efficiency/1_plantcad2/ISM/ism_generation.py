import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def generate_ism_data(input_tsv, output_tsv, mutate_length=100):
    df = pd.read_csv(input_tsv, sep='\t')

    bases = ['A', 'T', 'C', 'G']
    ism_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['seq'].upper()
        gene = row['gene']
        if len(seq) != 500:
            continue 

        ism_start = 500 - mutate_length
        for pos in range(ism_start, 500):
            original_base = seq[pos]
            for mut_base in bases:
                if mut_base == original_base:
                    continue
                mutated_seq = seq[:pos] + mut_base + seq[pos+1:]
                ism_records.append({
                    'gene': gene,
                    'original_index': idx,
                    'pos': pos - 500, 
                    'original_base': original_base,
                    'mutated_base': mut_base,
                    'seq': mutated_seq
                })

    ism_df = pd.DataFrame(ism_records)
    ism_df.to_csv(output_tsv, sep='\t', index=False)

if __name__ == "__main__":
    generate_ism_data(sys.argv[1], sys.argv[2], mutate_length=100)