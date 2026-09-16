"""Convert Kono et al. 2018 Table S2 training amino-acid substitutions to genomic SNVs (TAIR10).

Every single-nucleotide change of the WT codon that yields the MUT amino acid is emitted as one row.
Rows whose WT residue does not match the annotation (or whose transcript is absent) are removed.
Each SNV gets a + strand reference window of `--window` bp with the SNV at 1-based index window // 2.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import SimpleFastaParser

REPO_ROOT = Path(__file__).resolve().parents[3]
LABEL_COL = "Training (1=True positive, 0=True negative)"
COMP = str.maketrans("ACGT", "TGCA")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--table", default=REPO_ROOT / "results/ems/table-s2.csv")
parser.add_argument("--genome-dir", default="/workdir/jz963/genomes/Arabidopsis_thaliana")
parser.add_argument("--window", type=int, default=8192)
parser.add_argument("--output", default=REPO_ROOT / "results/ems/ems_training_snvs.tsv")
args = parser.parse_args()
GENOME_DIR = args.genome_dir


def read_fasta(path):
    with open(path) as f:
        return {h.split()[0]: s for h, s in SimpleFastaParser(f)}


df = pd.read_csv(args.table, skiprows=1)
df = df[df[LABEL_COL].notna()].copy()
df["label"] = df[LABEL_COL].astype(int)
print(f"Training rows: {len(df)}")

pep = read_fasta(f"{GENOME_DIR}/Arabidopsis_thaliana.TAIR10.pep.all.fa")
cds = read_fasta(f"{GENOME_DIR}/Arabidopsis_thaliana.TAIR10.cds.all.fa")
genome = read_fasta(f"{GENOME_DIR}/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa")

# Remove rows whose transcript is missing or whose WT residue disagrees with the annotation.
keep = df.apply(
    lambda r: r["Protein"] in pep and int(r["Position"]) <= len(pep[r["Protein"]])
    and pep[r["Protein"]][int(r["Position"]) - 1] == r["WT"],
    axis=1,
)
print(f"Removed {(~keep).sum()} rows failing the WT check; {keep.sum()} remain")
df = df[keep]

# CDS genomic coordinates (1-based), in transcript 5'->3' order, from the GTF.
needed = set(df["Protein"])
cds_exons = defaultdict(list)
strand_of, chrom_of, gene_of = {}, {}, {}
with open(f"{GENOME_DIR}/Arabidopsis_thaliana.TAIR10.56.gtf") as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if fields[2] not in ("CDS", "stop_codon"):
            continue
        attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
        tid = attrs["transcript_id"]
        if tid not in needed:
            continue
        cds_exons[tid].append((int(fields[3]), int(fields[4])))
        strand_of[tid], chrom_of[tid], gene_of[tid] = fields[6], fields[0], attrs["gene_id"]

tx_pos = {}
for tid, exons in cds_exons.items():
    exons = sorted(set(exons))
    positions = [p for s, e in exons for p in range(s, e + 1)]
    if strand_of[tid] == "-":
        positions = positions[::-1]
    seq = "".join(genome[chrom_of[tid]][p - 1] for p in positions)
    if strand_of[tid] == "-":
        seq = seq.translate(COMP)
    assert seq == cds[tid], f"GTF-derived CDS != cds.all.fa for {tid}"
    tx_pos[tid] = positions
print(f"GTF CDS reconstructions match cds.all.fa for all {len(tx_pos)} transcripts")

records = []
for _, r in df.iterrows():
    tid, aa_pos, wt, mut = r["Protein"], int(r["Position"]), r["WT"], r["MUT"]
    codon = cds[tid][3 * (aa_pos - 1): 3 * aa_pos]
    assert str(Seq(codon).translate()) == wt
    for i in range(3):
        for b in "ACGT":
            if b == codon[i]:
                continue
            alt_codon = codon[:i] + b + codon[i + 1:]
            if str(Seq(alt_codon).translate()) != mut:
                continue
            gpos = tx_pos[tid][3 * (aa_pos - 1) + i]
            ref, alt = (codon[i], b) if strand_of[tid] == "+" else (codon[i].translate(COMP), b.translate(COMP))
            assert genome[chrom_of[tid]][gpos - 1] == ref
            records.append({
                "chrom": chrom_of[tid], "pos": gpos, "ref": ref, "alt": alt, "gene": gene_of[tid],
                "label": r["label"], "transcript": tid, "strand": strand_of[tid], "aa_pos": aa_pos,
                "wt_aa": wt, "mut_aa": mut, "codon_pos": i + 1, "ref_codon": codon, "alt_codon": alt_codon,
                "source": r["Class"] if r["label"] == 1 else "EnsemblVariation",
            })

out = pd.DataFrame(records)
n_aa_with_snv = out.groupby(["transcript", "aa_pos", "mut_aa"]).ngroups
print(f"AA substitutions reachable by >=1 SNV: {n_aa_with_snv} / {len(df)} "
      f"(dropped {len(df) - n_aa_with_snv} needing >1 nucleotide change)")
print(f"SNV rows: {len(out)}")
print("SNVs per AA substitution:\n", out.groupby(["transcript", "aa_pos", "mut_aa"]).size().value_counts().sort_index().to_string())

dup = out.duplicated(["chrom", "pos", "ref", "alt"], keep=False)
if dup.any():
    conflicting = (out[dup].groupby(["chrom", "pos", "ref", "alt"])["label"].nunique() > 1).sum()
    print(f"Duplicated genomic SNVs: {dup.sum()} rows ({conflicting} sites with conflicting labels)")
    assert conflicting == 0
    # Same substitution listed under multiple isoforms of a gene; keep one row per SNV.
    out = out.drop_duplicates(["chrom", "pos", "ref", "alt"])
    print(f"SNV rows after deduplication: {len(out)}")

out = out.sort_values(["chrom", "pos", "alt"]).reset_index(drop=True)

# + strand reference window with the SNV at 1-based index window // 2
# (window // 2 - 1 bp upstream, window // 2 bp downstream); 0-based index window // 2 - 1.
WINDOW, CENTER = args.window, args.window // 2
starts = out["pos"] - CENTER  # 0-based start
assert all(s >= 0 and s + WINDOW <= len(genome[c]) for c, s in zip(out["chrom"], starts))
out["sequence"] = [genome[c][s:s + WINDOW] for c, s in zip(out["chrom"], starts)]
assert (out["sequence"].str.len() == WINDOW).all()
assert (out["sequence"].str[CENTER - 1] == out["ref"]).all()

print("Label counts (SNV level):\n", out["label"].value_counts().to_string())
out.to_csv(args.output, sep="\t", index=False)
print(f"Wrote {args.output} ({WINDOW} bp windows, SNV at 1-based index {CENTER})")
