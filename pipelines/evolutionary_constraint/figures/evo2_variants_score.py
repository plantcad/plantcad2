from Bio import SeqIO
import gzip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

# Set root path
os.chdir('../..')

df = pd.read_csv(sys.argv[1], sep='\t')
df['REF'] = df['sequences'].str[4095]


def expand_snps_to_variants(df):
    """Expand each SNP to 3 rows"""
    expanded_rows = []
    
    for idx, row in df.iterrows():
        ref_allele = row['REF']
        bases = ['A', 'T', 'C', 'G']
        alt_alleles = [base for base in bases if base != ref_allele]
        
        for alt in alt_alleles:
            new_row = row.copy()
            new_row['alt'] = alt
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows).reset_index(drop=True)

expanded_df = expand_snps_to_variants(df)
expanded_df = expanded_df[['chrom', 'pos', 'REF', 'alt', 'sequences', 'label']]

def load_reference_genome(fasta_path):
   """Load reference genome sequences for all chromosomes"""
   genome_dict = {}
   with open(fasta_path, "r") as handle:  #
       for record in SeqIO.parse(handle, "fasta"):
           chr_name = record.id.split()[0]  
           genome_dict[chr_name] = str(record.seq)
   return genome_dict

# Load the genome once
genome_dict = load_reference_genome('/workdir/jz963/genomes/Sorghum_bicolor_v3.1.1/assembly/Sbicolor_454_v3.0.1.fa')


def parse_sequences(chrom, pos, ref, alt):
   """
   Parse reference and variant sequences from the reference genome sequence.
   """
   # Get chromosome sequence
   if str(chrom) not in genome_dict:
       raise ValueError(f"Chromosome {chrom} not found in reference genome")
   
   chr_seq = genome_dict[str(chrom)]
   
   p = pos - 1  # Convert to 0-indexed position
   ref_seq_start = max(0, p - WINDOW_SIZE//2)
   ref_seq_end = min(len(chr_seq), p + WINDOW_SIZE//2)
   ref_seq = chr_seq[ref_seq_start:ref_seq_end]
   
   snv_pos_in_ref = min(WINDOW_SIZE//2, p)
   var_seq = ref_seq[:snv_pos_in_ref] + alt + ref_seq[snv_pos_in_ref+1:]
   
   # Sanity checks
   assert len(var_seq) == len(ref_seq)
   assert ref_seq[snv_pos_in_ref] == ref
   assert var_seq[snv_pos_in_ref] == alt
   
   return ref_seq, var_seq


from evo2.models import Evo2

# Load model
model = Evo2('evo2_7b')

WINDOW_SIZE = 8192
ref_seqs = []
ref_seq_to_index = {}

ref_seq_indexes = []
var_seqs = []

for _, row in expanded_df.iterrows():
    ref_seq, var_seq = parse_sequences(row['chrom'], row['pos'], row['REF'], row['alt'])

    # Get or create index for reference sequence
    if ref_seq not in ref_seq_to_index:
        ref_seq_to_index[ref_seq] = len(ref_seqs)
        ref_seqs.append(ref_seq)
    
    ref_seq_indexes.append(ref_seq_to_index[ref_seq])
    var_seqs.append(var_seq)

ref_seq_indexes = np.array(ref_seq_indexes)


print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
ref_scores = model.score_sequences(ref_seqs)

print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
var_scores = model.score_sequences(var_seqs)

# Add all scores to dataframe
expanded_df['ref_score'] = np.array(ref_scores)[ref_seq_indexes]
expanded_df['var_score'] = var_scores
expanded_df['delta_score'] = delta_scores

# Save complete dataframe
expanded_df.to_csv(sys.argv[2], sep='\t', index=False)



