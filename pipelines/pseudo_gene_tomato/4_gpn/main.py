import pandas as pd
import torch,sys
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse, sys, os
from tqdm import tqdm
from datasets import load_dataset
import gpn.model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", dest="inputFasta", type=str, default=None,
                        help="The name of the input dataset, either 'valid' or 'test'.")
    parser.add_argument("-outLogit", dest = "outLogit", default=None, help = "The directory of output")
    parser.add_argument("-model", dest = "modelDir", default=None, help = "The directory of pre-trained model")
    parser.add_argument("-device", dest = "device", default="cuda:0", help = "The device to run the model")
    parser.add_argument("-batchSize", dest = "batchSize", default=128, type=int, help = "The batch size for the model")
    parser.add_argument("-numWorkers", dest = "numWorkers", default=4, type=int, help = "The number of workers for the model")
    parser.add_argument("-tokenIdx", dest="tokenIdx", 
                       type=lambda x: [int(i.strip()) for i in x.split(',')], 
                       default=[255, 256, 257],
                       help="The indices of the tokens to be masked (comma-separated)")
    args = parser.parse_args()
    return args

args = parse_args()

class SequenceDataset(Dataset):
    def __init__(self, sequences, names, tokenizer, mask_token_ids):
        self.sequences = sequences
        self.names = names
        self.tokenizer = tokenizer
        self.mask_token_ids = mask_token_ids  # Now a list

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        name = self.names[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )
        input_ids = encoding['input_ids']
        # Mask multiple tokens
        for mask_idx in self.mask_token_ids:
            input_ids[0, mask_idx] = self.tokenizer.mask_token_id
        return {
            'sequence': sequence,
            'name': name,
            'input_ids': input_ids
        }

device = args.device
model_path = args.modelDir

try:
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).to(device)
except:
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    print("Note: The model is not supported for torch.bfloat16, running with torch.float32")

model.to(device)
model.eval()

# Initialize your tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


sequences = [str(record.seq) for record in SeqIO.parse(args.inputFasta, "fasta")]
sequences = [x[(len(x)-512)//2:(len(x)-512)//2+512] for x in sequences]
names = [record.id for record in SeqIO.parse(args.inputFasta, "fasta")]
# Create dataset
dataset = SequenceDataset(
    sequences=sequences,
    tokenizer=tokenizer,
    names=names,
    mask_token_ids=args.tokenIdx  # Pass the list directly
)

# Create your data loader
loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)

# if output exists, remove it
if args.outLogit and os.path.exists(args.outLogit):
    os.remove(args.outLogit)

nucleotides = list('acgt')
all_batch_probs = []  # Collect all results

for batch in tqdm(loader, desc="Inference..."):
    curName = np.array(batch['name'])[:,np.newaxis]
    curIDs = batch['input_ids'].to(device)
    curIDs = curIDs.squeeze(1)
    with torch.inference_mode():
        outputs = model(input_ids=curIDs)

    if args.outLogit:
        all_logits = outputs.logits
        batch_size = all_logits.shape[0]
        
        # Collect logits for all masked positions
        batch_probs_list = []
        for token_idx in args.tokenIdx:
            logits = all_logits[:, token_idx, [tokenizer.get_vocab()[nc] for nc in nucleotides]]
            probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
            batch_probs_list.append(probs)
        
        # Stack and reshape: (batch_size, num_masks, 4) -> (batch_size * num_masks, 4)
        batch_probs = np.stack(batch_probs_list, axis=1)  # (batch_size, num_masks, 4)
        batch_probs_reshaped = batch_probs.reshape(-1, 4)  # (batch_size * num_masks, 4)
        
        all_batch_probs.append(batch_probs_reshaped)

# Concatenate all batches and save
if args.outLogit and all_batch_probs:
    final_probs = np.vstack(all_batch_probs)  # (total_sequences * num_masks, 4)
    with open(args.outLogit, 'w') as f:
        np.savetxt(f, final_probs, delimiter='\t')