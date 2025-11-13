import torch
from evo2 import Evo2
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Union


evo2_model = Evo2('evo2_7b')

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
    parser = argparse.ArgumentParser(description="Extract logit scores from Evo2 model")
    parser.add_argument("--input", dest='inputDF', type=str, help="Path to input")
    parser.add_argument("--output", dest='output', type=str, help="Path to output logit scores")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing sequences (default: 32)")
    parser.add_argument("--tokenIdx", type=int, default=255, help="Number of tokens to generate (default: 255)")
    args = parser.parse_args()

    df = pd.read_csv(args.inputDF, sep='\t')
    seqs = df['sequence'].tolist()
    
    logits = evo2_token_logits(seqs, batch_size=args.batch_size, tokenIdx=args.tokenIdx)
    np.savetxt(args.output, logits, delimiter='\t', fmt='%.6f')

if __name__ == "__main__":
    main()


