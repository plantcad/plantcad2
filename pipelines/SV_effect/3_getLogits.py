import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).eval()
    return tokenizer, model

def extract_masked_sequences(sv_df, tokenizer, shift):
    ref_masked_seqs, ref_row_ids = [], []
    mut_masked_seqs, mut_row_ids = [], []

    for i, row in tqdm(sv_df.iterrows(), total=len(sv_df), desc = "Extracting masked sequences......"):
        ref_seq = list(row['RefSeq'])
        mut_seq = list(row['MutSeq'])

        left_pos = literal_eval(row['Left5_Positions']) if isinstance(row['Left5_Positions'], str) else row['Left5_Positions']
        right_pos = literal_eval(row['Right5_Positions']) if isinstance(row['Right5_Positions'], str) else row['Right5_Positions']
        if shift != 0:
            left_pos = [pos - shift for pos in left_pos]
            right_pos = [pos + shift for pos in right_pos]

        for pos in left_pos + right_pos:
            ref_copy = ref_seq.copy()
            ref_copy[pos - 1] = tokenizer.mask_token
            ref_masked_seqs.append("".join(ref_copy))
            ref_row_ids.append(i)
        
        left_pos = list(range(4092, 4097))
        right_pos = list(range(4097, 4102))
        
        if shift != 0:
            left_pos = [pos - shift for pos in left_pos]
            right_pos = [pos + shift for pos in right_pos]

        for pos in left_pos + right_pos:
            mut_copy = mut_seq.copy()
            mut_copy[pos - 1] = tokenizer.mask_token
            mut_masked_seqs.append("".join(mut_copy))
            mut_row_ids.append(i)

    return ref_masked_seqs, ref_row_ids, mut_masked_seqs, mut_row_ids

def run_batched_inference(seqs, row_ids, tokenizer, model, N, batch_size, label):
    logits_all = np.zeros((N, 10, 4), dtype=np.float32)
    logits_dict = {i: [] for i in range(N)}

    for i in tqdm(range(0, len(seqs), batch_size), desc=f"Running inference for {label}"):
        batch_seqs = seqs[i:i+batch_size]
        batch_rows = row_ids[i:i+batch_size]

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        inputs = tokenizer(
            batch_seqs,
            truncation=False,
            padding=False,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )['input_ids'].to(model.device)

        with torch.no_grad():
            outputs = model(inputs)
        
        nucleotides = list('acgt')
        logits = outputs.logits[..., [tokenizer.get_vocab()[nc] for nc in nucleotides]] # only retain atcg
        probs = torch.nn.functional.softmax(logits, dim=2).cpu().numpy()

        for j in range(len(batch_seqs)):
            mask_idx = (inputs[j] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            assert len(mask_idx) == 1, f"Expected one [MASK], got {len(mask_idx)}"

            cur_logits = probs[j, mask_idx.item(), :]
            row_id = batch_rows[j]
            logits_dict[row_id].append(cur_logits)

    for row_id, logits_list in logits_dict.items():
        logits_all[row_id] = np.stack(logits_list, axis=0)

    return logits_all

def main():
    parser = argparse.ArgumentParser(description="Compute masked token logits for RefSeq and MutSeq")
    parser.add_argument("--input", required=True, help="Input TSV file with sv_df")
    parser.add_argument("--model", required=True, help="Masked language model name or path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--output", required=True, help="Output .npz filename")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--label", default="RefSeq", help="Label for the model (e.g., 'RefSeq' or 'MutSeq')")
    parser.add_argument("--shift", default=0, type=int, help="Shift base positions for the masked sequences")
    args = parser.parse_args()

    sv_df = pd.read_csv(args.input, sep="\t")

    tokenizer, model = load_model(args.model)
    model.to(args.device)
    ref_seqs, ref_ids, mut_seqs, mut_ids = extract_masked_sequences(sv_df, tokenizer, shift=args.shift)

    if args.label == "RefSeq":
        logits = run_batched_inference(ref_seqs, ref_ids, tokenizer, model, len(sv_df), args.batch_size, args.label)
    elif args.label == "MutSeq":
        logits = run_batched_inference(mut_seqs, mut_ids, tokenizer, model, len(sv_df), args.batch_size, args.label)
    else:
        raise ValueError("Invalid label. Use 'RefSeq' or 'MutSeq'.")

    
    np.savez_compressed(args.output, logits=logits)
    print(f"Saved logits to {args.output}")

if __name__ == "__main__":
    main()
