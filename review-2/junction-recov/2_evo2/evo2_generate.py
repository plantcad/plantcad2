import os
import time
import torch
from evo2 import Evo2
from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def get_slurm_info():
    def _get_first_env_var(possible, default):
        for var in possible:
            if var in os.environ:
                return int(os.environ[var])
        return default
    return {
        "rank": _get_first_env_var(["PMIX_RANK", "SLURM_PROCID", "PMI_RANK"], 0),
        "world_size": _get_first_env_var(["PMIX_SIZE", "SLURM_NTASKS", "PMI_SIZE"], 1),
    }


def partition_indices(n, rank, world_size):
    chunk = n // world_size
    remainder = n % world_size
    start = rank * chunk + min(rank, remainder)
    end = start + chunk + (1 if rank < remainder else 0)
    return start, end


def read_fasta(fasta_file, length=1000, reverse=False):
    """
    Read sequences from FASTA file
    """
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        if reverse:
            seqs.append(str(record.seq.reverse_complement())[0:length])
        else:
            seqs.append(str(record.seq)[0:length])
    return [s.upper() for s in seqs]

def read_tsv(tsv_file, length=1000, reverse=False):
    """
    Read sequences from TSV file with a 'sequence' column
    """
    df = pd.read_csv(tsv_file, sep='\t')
    seqs = []
    for seq in df['sequence']:
        if reverse:
            from Bio.Seq import Seq
            seqs.append(str(Seq(seq).reverse_complement())[0:length])
        else:
            seqs.append(str(seq)[0:length])
    return [s.upper() for s in seqs]

def evo2_generate(seqs, model, batch_size=32, n_tokens=3, top_k=4):
    """
    Extract logit and generated sequences from Evo2 model
    """
    logits = []
    sequences = []
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]
        output = model.generate(prompt_seqs=batch, n_tokens=n_tokens, temperature=1.0, top_k=top_k)
        sequences.append(output.sequences)
        output = output.logits[0][:,:,[65,67,71,84]] # A, C, G, T
        probs = torch.nn.functional.softmax(output, dim=2).cpu().numpy()
        logits.append(probs)

    # return numpy array
    return np.concatenate(logits), np.concatenate(sequences)

def main():
    parser = argparse.ArgumentParser(description="Extract logit scores from Evo2 model")
    parser.add_argument("--fasta", type=str, help="Path to input FASTA file")
    parser.add_argument("--tsv", type=str, help="Path to input TSV file (must have a 'sequence' column)")
    parser.add_argument("--length", type=int, default=1000, help="Length of sequences to extract (default: 1000)")
    parser.add_argument("--reverse", action='store_true', help="Reverse complement the sequences")
    parser.add_argument("--output", type=str, help="Path to output logit scores")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing sequences (default: 32)")
    parser.add_argument("--n-tokens", type=int, default=3, help="Number of tokens to generate (default: 3)")
    parser.add_argument("--top-k", type=int, default=1, help="Top k tokens to consider (default: 1)")
    parser.add_argument("--model", type=str, default="evo2_7b", help="Evo2 model name (default: evo2_7b)")
    args = parser.parse_args()

    slurm = get_slurm_info()
    rank, world_size = slurm["rank"], slurm["world_size"]
    print(f"[Rank {rank}] Starting ({rank + 1}/{world_size})")

    if args.tsv:
        seqs = read_tsv(args.tsv, length=args.length, reverse=args.reverse)
    else:
        seqs = read_fasta(args.fasta, length=args.length, reverse=args.reverse)

    start_idx, end_idx = partition_indices(len(seqs), rank, world_size)
    seqs = seqs[start_idx:end_idx]
    print(f"[Rank {rank}] Processing {len(seqs)} sequences (index {start_idx}–{end_idx})")

    print(f"[Rank {rank}] Loading Evo2 model...")
    model = Evo2(args.model)

    logits, sequences = evo2_generate(seqs, model, batch_size=args.batch_size, n_tokens=args.n_tokens, top_k=args.top_k)

    # Save per-rank outputs
    rank_npz = Path(args.output).with_suffix(f".rank{rank}.npz")
    rank_tsv = Path(f"{args.output}.tsv").with_suffix(f".rank{rank}.tsv")
    np.savez_compressed(rank_npz, logits=logits)
    np.savetxt(rank_tsv, sequences, fmt='%s')
    print(f"[Rank {rank}] Saved to {rank_npz}")

    if rank == 0:
        print("[Rank 0] Waiting for all ranks to finish...")
        while True:
            done = sum(
                1 for r in range(world_size)
                if Path(args.output).with_suffix(f".rank{r}.npz").exists()
            )
            if done == world_size:
                break
            time.sleep(5)

        print("[Rank 0] Aggregating results...")
        all_logits = []
        all_sequences = []
        for r in range(world_size):
            all_logits.append(np.load(Path(args.output).with_suffix(f".rank{r}.npz"))['logits'])
            all_sequences.append(np.loadtxt(Path(f"{args.output}.tsv").with_suffix(f".rank{r}.tsv"), dtype=str))

        np.savez_compressed(args.output, logits=np.concatenate(all_logits))
        np.savetxt(f'{args.output}.tsv', np.concatenate(all_sequences), fmt='%s')
        print(f"[Rank 0] Final output saved to {args.output}")

        for r in range(world_size):
            Path(args.output).with_suffix(f".rank{r}.npz").unlink(missing_ok=True)
            Path(f"{args.output}.tsv").with_suffix(f".rank{r}.tsv").unlink(missing_ok=True)

if __name__ == "__main__":
    main()


