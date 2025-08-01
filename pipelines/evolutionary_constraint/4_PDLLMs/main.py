import pandas as pd
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse, os
from tqdm import tqdm
from datasets import load_dataset

repo_id = 'kuleshov-group/cross-species-single-nucleotide-annotation'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", dest="inputDF", type=str, default=None,
                        help="The name of the input dataset, either 'valid' or 'test'.")
    parser.add_argument("-outLogit", dest="outLogit", default=None, help="The path to save output logits.")
    parser.add_argument("-model", dest="modelDir", default=None, help="The directory of pre-trained model.")
    parser.add_argument("-device", dest="device", default="cuda:0", help="The device to run the model.")
    parser.add_argument("-batchSize", dest="batchSize", default=128, type=int, help="Batch size.")
    parser.add_argument("-numWorkers", dest="numWorkers", default=4, type=int, help="Number of workers.")
    parser.add_argument("-tokenIdx", dest="tokenIdx", default=255, type=int, help="Token index to use for logits extraction.")
    args = parser.parse_args()
    return args

args = parse_args()

class SequenceDataset(Dataset):
    def __init__(self, sequences, names, tokenizer):
        self.sequences = sequences
        self.names = names
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        name = self.names[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=True,
            max_length=512  # Adjust max_length as needed
        )
        input_ids = encoding['input_ids']
        return {
            'sequence': sequence,
            'name': name,
            'input_ids': input_ids
        }

device = args.device
model_path = args.modelDir

try:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
except:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    print("Note: The model is not supported for torch.bfloat16, running with torch.float32")

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load dataset
if args.inputDF == 'valid':
    data = load_dataset(repo_id, data_files={'valid': 'Evolutionary_constraint/valid.tsv'})
    df = data['valid'].to_pandas()
elif args.inputDF == 'test':
    data = load_dataset(repo_id, data_files={'test': 'Evolutionary_constraint/test.tsv'})
    df = data['test'].to_pandas()
else:
    try:
        df = pd.read_csv(args.inputDF, sep='\t')
    except FileNotFoundError:
        print(f"File {args.inputDF} not found. Please provide a valid input file.")
        sys.exit(1)

sequences = df['sequences'].tolist()
names = df['pos'].tolist()

# Dataset and loader
dataset = SequenceDataset(sequences=sequences, names=names, tokenizer=tokenizer)
loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorkers)

# Remove output file if it exists
if args.outLogit and os.path.exists(args.outLogit):
    os.remove(args.outLogit)

nucleotides = list('ACGT')
nucleotide_token_ids = [tokenizer.get_vocab()[nt] for nt in nucleotides]

for batch in tqdm(loader, desc="Inference..."):
    curName = np.array(batch['name'])[:, np.newaxis]
    curIDs = batch['input_ids'].to(device)
    curIDs = curIDs.squeeze(1)

    with torch.inference_mode():
        outputs = model(input_ids=curIDs)
        logits_all = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
    
    # print(f"Logits shape: {logits_all.shape}, batch size: {curIDs.shape[0]}")

    # For CLM: model predicts token i+1 at position i
    # Get logits at target_index, for each batch element
    logits = logits_all[:, args.tokenIdx, nucleotide_token_ids]  # shape: (batch_size, 4)

    if args.outLogit:
        probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
        with open(args.outLogit, 'a') as f:
            np.savetxt(f, probs, delimiter='\t')