"""Summarize zero-shot masked-LM scores on the EMS/Kono 2018 SNVs.

Reports AUROC/AUPRC for P(ref) and -log(P(alt)/P(ref)) at the SNV level, at the amino-acid-substitution
level (SNVs grouped by transcript, aa_pos, mut_aa), and for the Kono et al. 2018 predictors on the same
substitutions.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
NUCLEOTIDES = "ACGT"
# Kono et al. predictors and the sign that makes larger = more deleterious.
PAPER_SCORES = {"SiftScore": -1, "PolyPhen2": 1, "Provean": -1, "MAPP": 1, "Gerp++": 1, "LRT-logistic": 1}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", required=True)
parser.add_argument("--logits", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--table", default=REPO_ROOT / "results/ems/table-s2.csv")
args = parser.parse_args()


def metrics(y, s):
    return {"auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s), "n": len(y), "n_pos": int(y.sum())}


d = pd.read_csv(args.data, sep="\t", usecols=lambda c: c != "sequence")
probs = pd.read_csv(args.logits, sep="\t")[list(NUCLEOTIDES)].to_numpy()
assert len(probs) == len(d)
idx = {b: i for i, b in enumerate(NUCLEOTIDES)}
rows = np.arange(len(d))
d["p_ref"] = probs[rows, d["ref"].map(idx)]
d["p_alt"] = probs[rows, d["alt"].map(idx)]
d["neg_llr"] = np.log(d["p_ref"]) - np.log(d["p_alt"])
d.to_csv(f"{args.out_dir}/scores.tsv", sep="\t", index=False)

results = []
print(f"Masked-base accuracy (argmax == ref): {(probs.argmax(1) == d['ref'].map(idx)).mean():.4f}")
for score in ["p_ref", "neg_llr"]:
    results.append({"level": "snv", "score": score, **metrics(d["label"], d[score])})

aa = (
    d.groupby(["transcript", "aa_pos", "mut_aa"])
    .agg(label=("label", "first"), p_ref=("p_ref", "mean"), neg_llr=("neg_llr", "mean"))
    .reset_index()
)
table = pd.read_csv(args.table, skiprows=1).rename(columns={"Protein": "transcript", "Position": "aa_pos", "MUT": "mut_aa"})
aa = aa.merge(table[["transcript", "aa_pos", "mut_aa", *PAPER_SCORES]], on=["transcript", "aa_pos", "mut_aa"], how="left")
# Compare on substitutions scored by every Kono et al. predictor.
common = aa.dropna(subset=list(PAPER_SCORES))
for score in ["p_ref", "neg_llr"]:
    results.append({"level": "aa_all", "score": score, **metrics(aa["label"], aa[score])})
    results.append({"level": "aa_common", "score": score, **metrics(common["label"], common[score])})
for score, sign in PAPER_SCORES.items():
    results.append({"level": "aa_common", "score": score, **metrics(common["label"], sign * common[score])})

res = pd.DataFrame(results)
res.to_csv(f"{args.out_dir}/summary.tsv", sep="\t", index=False)
print(res.to_string(index=False, float_format="%.4f"))
