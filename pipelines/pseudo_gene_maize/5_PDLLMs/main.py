import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from Bio import SeqIO
import argparse
import numpy as np

class PlantDNAMamba:
    def __init__(self, model_name='plant-dnamamba-singlebase'):
        self.model = AutoModelForCausalLM.from_pretrained(
            f'zhangtaolab/{model_name}', 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            f'zhangtaolab/{model_name}', 
            trust_remote_code=True
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Get token IDs for nucleotides
        self.nucleotide_ids = {
            'A': self.tokenizer.encode('A', add_special_tokens=False)[0],
            'C': self.tokenizer.encode('C', add_special_tokens=False)[0],
            'G': self.tokenizer.encode('G', add_special_tokens=False)[0],
            'T': self.tokenizer.encode('T', add_special_tokens=False)[0]
        }
        self.nucleotide_indices = [
            self.nucleotide_ids['A'],
            self.nucleotide_ids['C'],
            self.nucleotide_ids['G'],
            self.nucleotide_ids['T']
        ]
    
    def generate_with_logits(self, sequences, n_tokens=3, temperature=1.0, top_k=0):
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        ).to(self.device)

        batch_size = inputs['input_ids'].shape[0]
        all_probs = []
        chosen_bases = []

        id2base = {v:k for k,v in self.nucleotide_ids.items()}
        nuc_ids = torch.tensor(self.nucleotide_indices, device=self.device)  # (4,)

        with torch.inference_mode():
            current_ids = inputs['input_ids']
            for _ in range(n_tokens):
                out = self.model(input_ids=current_ids)
                next_logits_full = out.logits[:, -1, :]                  # (B, V)
                next_logits_nuc  = next_logits_full.index_select(1, nuc_ids)  # (B, 4)

                # optional top-k on the 4 bases (mostly useless unless you allow >4)
                if top_k and top_k < 4:
                    tk_vals, tk_idx = torch.topk(next_logits_nuc, k=top_k, dim=1)
                    thresh = tk_vals[:, -1].unsqueeze(1)
                    mask = next_logits_nuc < thresh
                    next_logits_nuc = next_logits_nuc.masked_fill(mask, float('-inf'))

                probs_nuc = torch.softmax(next_logits_nuc / temperature, dim=-1)  # (B,4)
                all_probs.append(probs_nuc.detach().cpu().numpy())

                # sample within A/C/G/T, then map back to full IDs
                next_idx_in4 = torch.multinomial(probs_nuc, num_samples=1)        # (B,1)
                next_token = nuc_ids.gather(0, next_idx_in4.squeeze(1)).unsqueeze(1)  # (B,1)

                # record bases directly
                chosen_bases.append(next_token)

                current_ids = torch.cat([current_ids, next_token], dim=1)

        chosen_ids = torch.cat(chosen_bases, dim=1)  # (B, n_tokens)
        # turn IDs → bases without tokenizer
        inv = {v:k for k,v in self.nucleotide_ids.items()}
        generated_seqs = [
            ''.join(inv[int(t)] for t in chosen_ids[i].tolist())
            for i in range(batch_size)
        ]
        probs = np.stack(all_probs, axis=1)  # (B, n_tokens, 4)
        return probs, generated_seqs

def read_fasta(fasta_file, length=1000, reverse=False):
    """Read sequences from FASTA file"""
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        curseq = record.seq
        curseq = curseq[(len(curseq)-512)//2:(len(curseq)-512)//2+512] # Center the 512bp
        if reverse:
            seqs.append(str(curseq.reverse_complement())[0:length])
        else:
            seqs.append(str(curseq)[0:length])
    return seqs

def mamba_generate(model, seqs, batch_size=32, n_tokens=3, top_k=4):
    """Extract logits and generated sequences from DNAMamba model"""
    all_logits = []
    all_sequences = []
    
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]
        logits, sequences = model.generate_with_logits(
            batch, 
            n_tokens=n_tokens, 
            temperature=1.0, 
            top_k=top_k
        )
        all_logits.append(logits)
        all_sequences.extend(sequences)
    
    # Concatenate all logits
    return np.concatenate(all_logits, axis=0), np.array(all_sequences)

def main():
    parser = argparse.ArgumentParser(description="Extract logit scores from Plant DNAMamba model")
    parser.add_argument("--fasta", type=str, help="Path to input FASTA file")
    parser.add_argument("--length", type=int, default=1000, help="Length of sequences to extract (default: 1000)")
    parser.add_argument("--reverse", action='store_true', help="Reverse complement the sequences")
    parser.add_argument("--output", type=str, help="Path to output logit scores")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing sequences (default: 32)")
    parser.add_argument("--n-tokens", type=int, default=3, help="Number of tokens to generate (default: 3)")
    parser.add_argument("--top-k", type=int, default=1, help="Top k tokens to consider (default: 1)")
    parser.add_argument("--model", type=str, default="plant-dnamamba-singlebase", help="Model name (default: plant-dnamamba-singlebase)")
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model: {args.model}")
    model = PlantDNAMamba(args.model)
    
    # Read sequences
    if args.reverse:
        seqs = read_fasta(args.fasta, length=args.length, reverse=True)
    else:
        seqs = read_fasta(args.fasta, length=args.length, reverse=False)
    
    print(f"Processing {len(seqs)} sequences...")
    
    # Generate and extract logits
    logits, sequences = mamba_generate(
        model, 
        seqs, 
        batch_size=args.batch_size, 
        n_tokens=args.n_tokens, 
        top_k=args.top_k
    )
    
    # Save results
    np.savez_compressed(args.output, logits=logits)
    np.savetxt(f'{args.output}.tsv', sequences, fmt='%s')
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()