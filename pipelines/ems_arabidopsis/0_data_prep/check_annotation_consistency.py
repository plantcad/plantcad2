"""Check Kono et al. 2018 Table S2 training substitutions against the TAIR10 (Ensembl 56) annotation."""

import argparse
import re
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import SimpleFastaParser

REPO_ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--table", default=REPO_ROOT / "results/ems/table-s2.csv")
parser.add_argument("--genome-dir", default="/workdir/jz963/genomes/Arabidopsis_thaliana")
parser.add_argument("--output", default=REPO_ROOT / "results/ems/training_annotation_check.tsv")
args = parser.parse_args()
GENOME_DIR, TABLE = args.genome_dir, args.table
LABEL_COL = "Training (1=True positive, 0=True negative)"


def read_fasta(path):
    with open(path) as f:
        return {h.split()[0]: s for h, s in SimpleFastaParser(f)}


df = pd.read_csv(TABLE, skiprows=1)
print(f"Rows in table: {len(df)}")
df = df[df[LABEL_COL].notna()].copy()
df["label"] = df[LABEL_COL].astype(int)
print(f"Rows with non-NA training label: {len(df)}")
print(df["label"].value_counts().rename("n").to_string(), "\n")
print("Class x label:\n", pd.crosstab(df["Class"], df["label"]), "\n")

pep = read_fasta(f"{GENOME_DIR}/Arabidopsis_thaliana.TAIR10.pep.all.fa")
cds = read_fasta(f"{GENOME_DIR}/Arabidopsis_thaliana.TAIR10.cds.all.fa")

# Sanity: does CDS translation match the peptide file for the transcripts we need?
tx = df["Protein"].unique()
mism_tr = [t for t in tx if t in cds and t in pep and str(Seq(cds[t]).translate()).rstrip("*") != pep[t]]
print(f"Unique transcripts: {len(tx)}; in pep: {sum(t in pep for t in tx)}; "
      f"in cds: {sum(t in cds for t in tx)}; CDS translation != pep: {len(mism_tr)}\n")

TRANSITIONS_EMS = {("G", "A"), ("C", "T")}  # coding-strand view of G:C -> A:T


def check(row):
    t, pos, wt, mut = row["Protein"], int(row["Position"]), row["WT"], row["MUT"]
    if t not in pep:
        return pd.Series({"status": "transcript_missing"})
    p = pep[t]
    if pos < 1 or pos > len(p):
        return pd.Series({"status": "pos_out_of_range"})
    if p[pos - 1] != wt:
        return pd.Series({"status": "wt_mismatch", "ref_aa": p[pos - 1]})
    out = {"status": "wt_match", "ref_aa": p[pos - 1]}
    c = cds.get(t)
    if c is not None:
        codon = c[3 * (pos - 1): 3 * pos]
        snvs = []  # all single-nucleotide changes that produce MUT
        for i in range(3):
            for b in "ACGT":
                if b != codon[i]:
                    alt = codon[:i] + b + codon[i + 1:]
                    if str(Seq(alt).translate()) == mut:
                        snvs.append((codon[i], b))
        out["codon"] = codon
        out["n_snv"] = len(snvs)
        out["ems_compatible"] = any(s in TRANSITIONS_EMS for s in snvs)
    return pd.Series(out)


res = pd.concat([df, df.apply(check, axis=1)], axis=1)

print("WT-residue check (all training rows):")
print(res["status"].value_counts().to_string(), "\n")
print("WT-residue check by label:\n", pd.crosstab(res["status"], res["label"], margins=True), "\n")
print("WT-residue check by Class:\n", pd.crosstab(res["Class"], res["status"]), "\n")

mm = res[res["status"] != "wt_match"]
if len(mm):
    print("Mismatching / missing rows by transcript (top 20):")
    print(mm.groupby(["Protein", "status"]).size().sort_values(ascending=False).head(20).to_string(), "\n")
    # Is the mismatch explained by an alternative isoform of the same gene?
    gene_tx = {}
    for k in pep:
        gene_tx.setdefault(re.sub(r"\.\d+$", "", k), []).append(k)
    def alt_iso(row):
        g = re.sub(r"\.\d+$", "", row["Protein"])
        pos = int(row["Position"])
        return [k for k in gene_tx.get(g, []) if k != row["Protein"] and len(pep[k]) >= pos and pep[k][pos - 1] == row["WT"]]
    mm = mm.assign(alt_isoforms_matching=mm.apply(alt_iso, axis=1))
    print(f"Of {len(mm)} non-matching rows, {(mm['alt_isoforms_matching'].str.len() > 0).sum()} "
          f"match WT in another isoform of the same gene")
    print(mm[["Protein", "Position", "WT", "MUT", "ref_aa", "label", "Class", "alt_isoforms_matching"]].head(30).to_string(), "\n")

ok = res[res["status"] == "wt_match"]
print("Among WT-matching rows:")
print("  reachable by a single-nucleotide change:\n", pd.crosstab(ok["n_snv"] > 0, ok["label"], margins=True))
print("  reachable by an EMS-type (G>A / C>T) change:\n", pd.crosstab(ok["ems_compatible"], ok["label"], margins=True))

res.to_csv(args.output, sep="\t", index=False)
