import numpy as np
import pandas as pd
import argparse
from ast import literal_eval


def zero_shot(df, refLogits, mutLogits, shift):
    nucleotides = "ACGT"
    res = []
    for i, ((_, row), ref, mut) in enumerate(zip(df.iterrows(), refLogits, mutLogits)):
        upstream_idx = 251 - shift
        downstream_idx = 256 + shift
        curSeq = row['MutSeq'][upstream_idx:(upstream_idx + 5)] + row['MutSeq'][downstream_idx:(downstream_idx + 5)]
        
        mut_left = mut[upstream_idx:(upstream_idx + 5), :]
        mut_right = mut[downstream_idx:(downstream_idx + 5), :]
        mut = np.concatenate((mut_left, mut_right), axis=0)

        left_pos = literal_eval(row['Left5_Positions']) if isinstance(row['Left5_Positions'], str) else row['Left5_Positions']
        right_pos = literal_eval(row['Right5_Positions']) if isinstance(row['Right5_Positions'], str) else row['Right5_Positions']

        left_pos = [pos - 3840 for pos in left_pos] # (8192-512)/2 = 3840
        right_pos = [pos - 3840 for pos in right_pos]

        ref_left = ref[np.array(left_pos) - shift - 1, :]
        ref_right = ref[np.array(right_pos) + shift - 1, :]
        ref = np.concatenate((ref_left, ref_right), axis=0)
        
        scores = []
        for idx, nt in enumerate(curSeq):
            if nt in nucleotides:
                refProb = ref[idx, nucleotides.index(nt)]
                mutProb = mut[idx, nucleotides.index(nt)]
                scores.append(np.log(mutProb / refProb))
            else:
                scores.append(0)
        res.append(scores)
    return res

def main():
    parser = argparse.ArgumentParser(description='Parse logits for SV effect prediction.')
    parser.add_argument('--input', type=str, required=True, help='Input TSV file with SV data.')
    parser.add_argument('--ref_logits', type=str, required=True, help='Path to reference logits file.')
    parser.add_argument('--mut_logits', type=str, required=True, help='Path to mutated logits file.')
    parser.add_argument('--output', type=str, required=True, help='Output file name for results.')
    parser.add_argument('--shift', type=int, default=0, help='Shift value for the sequence.')
    args = parser.parse_args()

    shift = args.shift

    df = pd.read_csv(args.input, sep='\t')
    df['RefSeq'] = df['RefSeq'].apply(lambda x: x[(len(x)-512)//2:(len(x)-512)//2+512])
    df['MutSeq'] = df['MutSeq'].apply(lambda x: x[(len(x)-512)//2:(len(x)-512)//2+512])
    df['Location'] = df['Chromosome'].astype(str) + ":" + df['Start'].astype(str) + "-" + df['End'].astype(str)

    loaded = np.load(args.ref_logits)
    refLogits = loaded['logits']

    loaded = np.load(args.mut_logits)
    mutLogits = loaded['logits']

    # check if refLogits, mutLogits and df have the same number of rows
    if len(refLogits) != len(df) or len(mutLogits) != len(df):
        raise ValueError("Number of rows in logits files does not match the number of rows in the input DataFrame.")

    res = zero_shot(df, refLogits, mutLogits, shift=shift)
    score_df = pd.DataFrame(res, columns=[f'score_{i}' for i in range(10)])
    df = pd.concat([df, score_df], axis=1)
    df.to_csv(args.output, sep='\t', index=False)

if __name__ == "__main__":
    main()
