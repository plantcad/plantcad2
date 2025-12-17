import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import modiscolite


def one_hot_encode(sequence):
    base_to_index = {'A':0, 'C':1, 'G':2, 'T':3}
    encoding = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in base_to_index:
            encoding[i, base_to_index[base]] = 1.0
        else:
            encoding[i, :] = 0  # Handle N bases
    return encoding

sequences = []
names = []
for record in SeqIO.parse('../../results/review/9_ap2_tf_family/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events.fasta', "fasta"):
    sequences.append(str(record.seq))
    names.append(record.id)

resOneHot_list = []
for seq in sequences:
    encoded = one_hot_encode(seq)
    resOneHot_list.append(encoded)

resOneHot = np.stack(resOneHot_list, axis = 0)
resOneHot.shape

logits = np.load('/workdir/jz963/utils/plantcad2/results/review/9_ap2_tf_family/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_evo2_context-2k-formatted.npz')
probs = logits['probs']

print("min:", probs.min())
print("max:", probs.max())

epsilon = 1e-10
centered_probs = np.log(probs + epsilon) - np.log(0.25)

centered_probs.min(),centered_probs.max()

pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=centered_probs,
    one_hot=resOneHot,
    max_seqlets_per_metacluster=25000,
    sliding_window_size=15,
    flank_size=5,
    verbose=True)

modiscolite.io.save_hdf5('evo2-context-2k_modisco_results.h5', pos_patterns, neg_patterns, window_size = 15)
