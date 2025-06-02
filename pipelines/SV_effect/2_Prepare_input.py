import pandas as pd
import pysam
from tqdm import tqdm
import sys

df = pd.read_csv(sys.argv[1], sep='\t', comment='#', header=None)
df.columns = ['Uploaded_variation', 'Location', 'Allele', 'Gene', 'Feature', 'Feature_type', 'Consequence', 'cDNA_position',
              'CDS_position','Protein_position', 'Amino_acids','Codons','Existing_variation','Extra']

unique_locations = df['Location'].unique()

split_data = []
for loc in unique_locations:
    chrom, pos_range = loc.split(':')
    split_result = pos_range.split('-')
    if len(split_result) == 1:  # Single position, no range
        start = end = int(split_result[0])
    else:  # Range provided
        start, end = map(int, split_result)
    split_data.append([chrom, start, end])

location_df = pd.DataFrame(split_data, columns=['Chromosome', 'Start', 'End'])

genome_file = '../../results/SV_effect/Arabidopsis_thaliana.TAIR10.dna.toplevel.noMtPt.fa'
fasta = pysam.FastaFile(genome_file)

context_size = 8192
half = context_size // 2

records = []

for _, row in tqdm(location_df.iterrows(), total=len(location_df), desc="Extracting sequences"):
    chrom = row['Chromosome']
    start = int(row['Start'])
    end = int(row['End'])
    del_len = end - start + 1
    center = (start + end) // 2

    # Reference sequence (with deletion present)
    left_flank = max(0, center - half)
    right_flank = center + half
    ref_seq = fasta.fetch(chrom, left_flank, right_flank)

    # Mutated sequence (deletion removed)
    del_start = start - 1  # 0-based
    del_end = end          # exclusive
    left = fasta.fetch(chrom, max(0, del_start - half), del_start)
    right = fasta.fetch(chrom, del_end, del_end + half)
    mut_seq = left + right

    records.append({
        "Chromosome": chrom,
        "Start": start,
        "End": end,
        'RefStart': left_flank + 1,
        'RefEnd': right_flank,
        "DelLen": del_len,
        "RefSeq": ref_seq,
        "MutSeq": mut_seq
    })

# Convert to DataFrame
sv_df = pd.DataFrame(records)


left_positions = []
right_positions = []

for i, row in sv_df.iterrows():
    ref_len = len(row["RefSeq"])  # should be 512
    del_len = row["DelLen"]

    del_start = (ref_len - del_len) // 2
    del_end = del_start + del_len

    left_pos = [p + 1 for p in range(max(0, del_start - 5), del_start)]
    right_pos = [p + 1 for p in range(del_end, min(ref_len, del_end + 5))]

    left_positions.append(left_pos)
    right_positions.append(right_pos)

sv_df["Left5_Positions"] = left_positions
sv_df["Right5_Positions"] = right_positions

sv_df = sv_df[
    (sv_df["RefSeq"].str.len() == context_size) &
    (sv_df["MutSeq"].str.len() == context_size)
].reset_index(drop=True)

sv_df.to_csv(sys.argv[2], index=False, sep='\t')