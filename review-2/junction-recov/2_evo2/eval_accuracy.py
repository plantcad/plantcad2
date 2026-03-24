"""
Evaluate Evo2 junction recovery accuracy by comparing generated tokens
to the true junction sequence extracted from the input TSV.

For each input sequence (length L), the model predicts n_tokens starting at position L.
The true label is input_sequence[L : L + n_tokens].
"""

import argparse
import pandas as pd
import numpy as np
from Bio.Seq import Seq


def extract_true_labels(tsv_file, length, n_tokens, reverse=False):
    df = pd.read_csv(tsv_file, sep='\t')
    labels = []
    for seq in df['sequence']:
        seq = seq.upper()
        if reverse:
            seq = str(Seq(seq).reverse_complement())
        labels.append(seq[length: length + n_tokens])
    return labels


def read_predictions(pred_file):
    with open(pred_file) as f:
        return [line.strip().upper() for line in f if line.strip()]


def compute_accuracy(true_labels, predictions):
    assert len(true_labels) == len(predictions), (
        f"Length mismatch: {len(true_labels)} true labels vs {len(predictions)} predictions"
    )
    n = len(true_labels)
    exact = sum(t == p for t, p in zip(true_labels, predictions))

    # Per-position accuracy
    n_tokens = max(len(t) for t in true_labels)
    pos_correct = np.zeros(n_tokens, dtype=int)
    pos_total = np.zeros(n_tokens, dtype=int)
    for t, p in zip(true_labels, predictions):
        for i, (tc, pc) in enumerate(zip(t, p)):
            pos_correct[i] += tc == pc
            pos_total[i] += 1

    return {
        "n": n,
        "exact_match": exact / n,
        "per_position": [pos_correct[i] / pos_total[i] if pos_total[i] else None
                         for i in range(n_tokens)],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Evo2 junction recovery accuracy")
    parser.add_argument("--input-tsv", required=True, help="Input TSV with seq_id and sequence columns")
    parser.add_argument("--pred-tsv", required=True, help="Evo2 output TSV (one predicted sequence per line)")
    parser.add_argument("--length", type=int, required=True, help="Context length used for generation")
    parser.add_argument("--n-tokens", type=int, required=True, help="Number of tokens predicted")
    parser.add_argument("--reverse", action='store_true', help="Reverse complement was applied to input")
    args = parser.parse_args()

    true_labels = extract_true_labels(args.input_tsv, args.length, args.n_tokens, args.reverse)
    predictions = read_predictions(args.pred_tsv)

    result = compute_accuracy(true_labels, predictions)

    print(f"N sequences:   {result['n']}")
    print(f"Exact match:   {result['exact_match']:.4f} ({int(result['exact_match'] * result['n'])}/{result['n']})")
    print("Per-position accuracy:")
    for i, acc in enumerate(result['per_position']):
        print(f"  Position {i+1}: {acc:.4f}" if acc is not None else f"  Position {i+1}: N/A")


if __name__ == "__main__":
    main()
