import pandas as pd
import numpy as np
from torch.utils.data import Dataset
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
        """
        Args:
            sequences: list of sequences
            tokenizer: tokenizer object
            tokenIdx: list of indices to mask (e.g., [255, 256, 257])
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tokenIdx = tokenIdx  # List of positions to mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        input_ids = encoding['input_ids']
        # Store original tokens for all positions to be masked
        true_ids = input_ids[0, self.tokenIdx].clone()
        # Mask all specified positions
        input_ids[0, self.tokenIdx] = self.tokenizer.mask_token_id
        
        return {
            'sequence': sequence,
            'masked_ids': input_ids,
            'true_ids': true_ids
        }

