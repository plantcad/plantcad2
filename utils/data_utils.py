import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from Bio import SeqIO

def load_fastas(args):
    sequences = []
    names = []
    for record in SeqIO.parse(args.inputFASTA, "fasta"):
        sequences.append(str(record.seq))
        names.append(record.id)
    return sequences, names

class maskedTokenDataset(Dataset):
    def __init__(self, sequences, tokenizer, tokenIdx):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tokenIdx = tokenIdx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        true_ids = input_ids[0,self.tokenIdx].clone()
        input_ids[0, self.tokenIdx] = self.tokenizer.mask_token_id
        return {
            'sequence': sequence,
            'masked_ids': input_ids,
            'true_ids': true_ids
        }

