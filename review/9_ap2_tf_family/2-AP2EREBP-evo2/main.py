import torch
from evo2 import Evo2
from tqdm import tqdm
import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Union


import gzip
from Bio import SeqIO
import logging

evo2_model = Evo2('evo2_7b')

def load_fasta(fasta_path):
    """Loads a FASTA file, handling .gz compression."""
    logging.info(f"Loading reference genome from {fasta_path}")
    if fasta_path.endswith(".gz"):
        with gzip.open(fasta_path, "rt") as file:
            return SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    else:
        return SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))

def load_bed_file(bed_path):
    """Load BED file and return DataFrame with chr, start, end columns."""
    logging.info(f"Reading BED file from {bed_path}")
    bed_df = pd.read_csv(bed_path, sep='\t', header=None,
                         names=['chr', 'start', 'end'], usecols=[0, 1, 2])
    return bed_df

def extract_sequences_from_bed(args, bed_df, fasta_dict):
    """Extract sequences using windowed approach for BED regions."""
    logging.info("Extracting sequences from BED regions with windowed approach")

    sequences = []
    position_info = []  # Store chr, pos, ref_allele for each position
    
    addIdx = args.contextSize - args.tokenIdx

    logging.info(f"Using step_size={args.stepSize}")

    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df)):
        chrom = str(row['chr'])
        start = int(row['start'])
        end = int(row['end'])

        if chrom not in fasta_dict:
            logging.warning(f"Chromosome {chrom} not found in FASTA file, skipping region")
            continue

        for window_start in range(start, end, args.stepSize):
            window_end = min(window_start + args.stepSize, end)
            num_positions = window_end - window_start

            if num_positions == 0:
                continue

            center_pos = window_start + (num_positions - 1) / 2.0
            center_pos_int = int(center_pos)

            try:
                window_refs = []
                window_positions = []

                for pos in range(window_start, window_end):
                    ref_allele = str(fasta_dict[chrom].seq[pos]).upper()
                    if ref_allele not in ['A', 'C', 'G', 'T']:
                        continue
                    window_refs.append(ref_allele)
                    window_positions.append(pos)

                if not window_refs:
                    continue

                seq_start = center_pos_int - args.tokenIdx
                seq_end = center_pos_int + addIdx

                if seq_start < 0:
                    seq = str(fasta_dict[chrom].seq[0:seq_end]).upper().rjust(args.contextSize, "N")
                else:
                    seq = str(fasta_dict[chrom].seq[seq_start:seq_end]).upper().ljust(args.contextSize, "N")

                sequences.append(seq)
                
                for pos, ref in zip(window_positions, window_refs):
                    position_info.append({'chr': chrom, 'pos': pos, 'ref': ref})

            except Exception as e:
                logging.warning(f"Error processing window {chrom}:{window_start}-{window_end}, skipping. Error: {e}")
                continue

    logging.info(f"Extracted {len(sequences)} windows covering {len(position_info)} positions")
    return sequences, position_info

def prepare_batch(
        seqs: List[str],
        tokenizer: object,
        prepend_bos: bool = False,
        device: str = 'cuda:0'
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes in a list of sequences, tokenizes them, and puts them in a tensor batch.
    If the sequences have differing lengths, then pad up to the maximum sequence length.
    """
    seq_lengths = [ len(seq) for seq in seqs ]
    max_seq_length = max(seq_lengths)

    input_ids = []
    for seq in seqs:
        padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
        input_ids.append(
            torch.tensor(
                ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
                dtype=torch.long,
            ).to(device).unsqueeze(0)
        )
    input_ids = torch.cat(input_ids, dim=0)

    return input_ids, seq_lengths

def evo2_token_logits(seqs, batch_size=32, tokenIdx=255, device='cuda:0', prepend_bos=False):
    """
    Get logit scores for each token in the sequences using the Evo2 model.
    """
    logits = []
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]
        input_ids, seq_lengths = prepare_batch(batch, evo2_model.tokenizer, device=device, prepend_bos=prepend_bos)
        assert len(seq_lengths) == input_ids.shape[0]

        with torch.inference_mode():
            output, _ = evo2_model(input_ids) # (batch, length, vocab)

        output = output[0][:,tokenIdx-1,[65,67,71,84]] # A, C, G, T
        probs = torch.nn.functional.softmax(output, dim=1).float().cpu().numpy()
        logits.append(probs)
        
    return np.concatenate(logits)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Extract logit scores from Evo2 model")
    
    inputGroup = parser.add_mutually_exclusive_group(required=True)
    inputGroup.add_argument("--input-bed", dest='inputBed', type=str, help="Path to input BED file")
    inputGroup.add_argument("--input-tsv", dest='inputTSV', type=str, help="Path to input TSV file with sequences (old format)")
    
    parser.add_argument("--input-fasta", dest="inputFasta", type=str, help="Path to reference genome fasta (required for BED input)")
    parser.add_argument("--output", dest='output', type=str, required=True, help="Path to output logit scores")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing sequences (default: 32)")
    parser.add_argument("--tokenIdx", type=int, default=255, help="Index of the token to score (0-based index in the extracted sequence)")
    parser.add_argument("--contextSize", type=int, default=512, help="Context window size (default: 512)")
    parser.add_argument("--step-size", dest="stepSize", default=1, type=int, help="Step size for windowing (default: 1)")
    
    args = parser.parse_args()

    if args.inputBed:
        if not args.inputFasta:
            sys.exit("--input-fasta is required with --input-bed")
            
        fasta_dict = load_fasta(args.inputFasta)
        bed_df = load_bed_file(args.inputBed)
        
        # Calculate tokenIdx if not manually set? 
        # The user's zero_shot_score sets tokenIdx = contextSize // 2 - 1
        # But here tokenIdx is an argument. Let's respect the argument but maybe default it if needed.
        # Actually, let's just use the logic from zero_shot_score for extraction, which relies on tokenIdx.
        # If the user passes --tokenIdx, we use it.
        
        sequences, position_info = extract_sequences_from_bed(args, bed_df, fasta_dict)
        
        if not sequences:
            logging.error("No sequences extracted.")
            sys.exit(1)
            
        seqs = sequences
        
    elif args.inputTSV:
        df = pd.read_csv(args.inputTSV, sep='\t')
        seqs = df['sequence'].tolist()
    
    logits = evo2_token_logits(seqs, batch_size=args.batch_size, tokenIdx=args.tokenIdx)
    
    # If we have position info (from BED), we might want to save it?
    # The original code just saved logits. The user asked to "extract sequences... then generate logits".
    # They didn't explicitly ask to change the output format, but saving just logits for BED input might be useless without coordinates.
    # However, the request was "change it to taking bed as input... logic of extracting sequences is the exactly the same".
    # I will stick to the original output format (just logits) to be safe, or maybe add coordinates if it seems appropriate.
    # Given the previous context of zero_shot_score producing TSVs with scores, maybe I should output that?
    # But the original code outputted a simple text file of logits.
    # "np.savetxt(args.output, logits, delimiter='\t', fmt='%.6f')"
    # I will keep the output format as is for now, just saving the logits.
    
    np.savetxt(args.output, logits, delimiter='\t', fmt='%.6f')

if __name__ == "__main__":
    main()


