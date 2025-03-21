import torch
from evo2 import Evo2
from tqdm import tqdm
from Bio import SeqIO
import argparse
import numpy as np

evo2_model = Evo2('evo2_7b')

def read_fasta(fasta_file, length=1000):
    """
    Read sequences from FASTA file
    """
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs.append(str(record.seq)[0:length])
    return seqs

def evo2_generate(seqs, batch_size=32, n_tokens=3, top_k=4):
    """
    Extract logit and generated sequences from Evo2 model
    """
    logits = []
    sequences = []
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]
        output = evo2_model.generate(prompt_seqs=batch, n_tokens=n_tokens, temperature=1.0, top_k=top_k)
        sequences.append(output.sequences)
        output = output.logits[0][:,:,[65,67,71,84]] # A, C, G, T
        probs = torch.nn.functional.softmax(output, dim=2).cpu().numpy()
        logits.append(probs)
        
    # return numpy array
    return np.concatenate(logits), np.concatenate(sequences)

def main():
    parser = argparse.ArgumentParser(description="Extract logit scores from Evo2 model")
    parser.add_argument("--fasta", type=str, help="Path to input FASTA file")
    parser.add_argument("--length", type=int, default=1000, help="Length of sequences to extract (default: 1000)")
    parser.add_argument("--output", type=str, help="Path to output logit scores")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing sequences (default: 32)")
    parser.add_argument("--n-tokens", type=int, default=3, help="Number of tokens to generate (default: 3)")
    parser.add_argument("--top-k", type=int, default=1, help="Top k tokens to consider (default: 1)")
    args = parser.parse_args()

    seqs = read_fasta(args.fasta, length=args.length)
    logits, sequences = evo2_generate(seqs, batch_size=args.batch_size, n_tokens=args.n_tokens, top_k=args.top_k)
    np.savez_compressed(args.output, logits=logits)
    np.savetxt(f'{args.output}.tsv', sequences, fmt='%s')

if __name__ == "__main__":
    main()


