import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

def one_hot_encode(sequence):
    base_to_index = {'A':0, 'C':1, 'G':2, 'T':3}
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in base_to_index:
            encoding[i, base_to_index[base]] = 1.0
        else:
            encoding[i, :] = 0.25  # Handle N bases
    return encoding

windows = pd.read_csv('../../results/review/9_ap2_tf_family/ath_upstream_1.5k.bed', sep='\t', header=None)
windows.columns = ['chr', 'start', 'end', 'four', 'five', 'strand']

sequences = []
names = []
for record in SeqIO.parse('../../results/review/9_ap2_tf_family/ath_upstream_1.5k.fasta', "fasta"):
    sequences.append(str(record.seq))
    names.append(record.id)

logits = pd.read_csv('../../results/review/9_ap2_tf_family/genome_wide_plantcad2-large-context-2k-prob.tsv', sep='\t')
logits.columns = ['A', 'C', 'G', 'T']
logits.head()

resProb_list = []
resOneHot_list = []
insert_rows = pd.DataFrame(0.25, index=range(20), columns=logits.columns)
current_logit_pos = 0  # Track position in logits DataFrame

for idx, row in tqdm(windows.iterrows(), total=len(windows)):
    start_pos, end_pos = row['start'], row['end']
    width = end_pos - start_pos
    
    # Extract current logits
    curLogits = logits.iloc[current_logit_pos: current_logit_pos + width]
    current_logit_pos += width
    
    # Append insert_rows and process sequence
    curLogits = pd.concat([curLogits, insert_rows], axis=0)
    curSeq = sequences[idx] + 'N'*20
    encoded = one_hot_encode(curSeq)
    
    resProb_list.append(curLogits)
    resOneHot_list.append(encoded)

# Concatenate all results
resProb = pd.concat(resProb_list, axis=0).reset_index(drop=True)
resOneHot = np.vstack(resOneHot_list)

resProb_array = resProb.to_numpy()

resProb_array = resProb_array[np.newaxis, :, :]
resProb_array.shape

resOneHot = resOneHot[np.newaxis, :, :]

epsilon = 1e-10
centered_probs = np.log(resProb_array + epsilon) - np.log(0.25)

import modiscolite


pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=centered_probs,
    one_hot=resOneHot,
    max_seqlets_per_metacluster=100_000,
    sliding_window_size=20,
    flank_size=5,
    verbose=True)

modiscolite.io.save_hdf5('genome_wide_plantcad2_modisco_results.h5', pos_patterns, neg_patterns, window_size = 20)
