#!/usr/bin/env python3
import requests
import os
import json
import time
import argparse
from pathlib import Path
from Bio import SeqIO 
import numpy as np

key = "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"
url = os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate")

def read_fasta(fasta_file):
    """
    Read sequences from FASTA file
    """
    seqs = []
    names = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs.append(str(record.seq))
        names.append(record.id)
    return seqs, names

def softmax(x, axis=None):
    """
    Apply softmax function to numpy array.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def submit_request(sequence, name, num_tokens=1):
    """
    Submit a request to the Evo2 API
    """
    r = requests.post(
        url=url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "sequence": sequence,
            "num_tokens": num_tokens,
            "temperature": 1.0,
            "top_k": 1,
            "enable_sampled_probs": False,
            "enable_logits": True
        },
    )
    response = r.json()
    generatedSeq = response.get("sequence")
    if "logits" in response:
        logits = np.array(response["logits"])[...,[65, 67, 71, 84]]
        probs = softmax(logits, axis=1)
        return generatedSeq, probs, name
    else:
        raise ValueError("No logits found in the response")  
      


def main():
    parser = argparse.ArgumentParser(description="Extract logit scores from Evo2 model")
    parser.add_argument("--fasta", type=str, help="Path to input FASTA file")
    parser.add_argument("--length", type=int, default=2048, help="Length of sequences to extract (default: 1000)")
    parser.add_argument("--n-tokens", type=int, default=3, help="Number of tokens to generate (default: 3)")
    parser.add_argument("--reverse", action='store_true', help="Reverse complement the sequences")
    parser.add_argument("--output", type=str, help="Path to output logit scores")
    args = parser.parse_args()

    seqs, names = read_fasta(args.fasta)
    seqLen = len(seqs[0]) # assuming all sequences have the same length, TODO: support variable length
    print(f"Length of sequences: {seqLen}")
    
    short_sequence_length = 4096 # Longer than this won't be responsed from Evo2 api
    start = (seqLen - short_sequence_length) // 2
    seqs = [seq[start:(start+short_sequence_length)] for seq in seqs]

    if args.reverse:
        print("Reversing sequences")
        seqs = [str(record.seq.reverse_complement()) for record in SeqIO.parse(args.fasta, "fasta")]
        
    seqs = [seq[:args.length] for seq in seqs]
    for seq, name in zip(seqs, names):
        # if output file already exists, skip
        if os.path.exists(f"{args.output}/{name}.npz"):
            print(f"Skipping {name}, already exists")
            continue
        else:
            try:
                print(f"Processing {name}")
                curseq, logit, matched_name = submit_request(seq, name, num_tokens=args.n_tokens)
                # Save per-sequence
                np.savez_compressed(f"{args.output}/{matched_name}.npz", logits=logit, name=matched_name, sequence=curseq)
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        # time.sleep(1)


if __name__ == "__main__": 
    main()

