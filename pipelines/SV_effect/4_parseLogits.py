import numpy as np
import pandas as pd
import argparse

def zero_shot(df, refLogits, mutLogits, shift):
    nucleotides = "ACGT"
    res = []
    for i, ((_, row), ref, mut) in enumerate(zip(df.iterrows(), refLogits, mutLogits)):
        upstream_idx = 4091 - shift
        downstream_idx = 4096 + shift
        curSeq = row['MutSeq'][upstream_idx:(upstream_idx + 5)] + row['MutSeq'][downstream_idx:(downstream_idx + 5)]
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
