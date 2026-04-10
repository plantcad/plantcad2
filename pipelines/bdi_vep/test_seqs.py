"""
Quick sanity-check: extract ref/alt windows for the first N SNPs and write them to a TSV.

Usage:
    python test_seqs.py \
        --vcf variants.vcf.gz \
        --fasta genome.fa \
        --output seq_check.tsv \
        --n 10 \
        --half_window 300

Output TSV columns:
    CHROM  POS  REF  ALT  REF_SEQ  ALT_SEQ  MATCH_CENTER_REF  MATCH_CENTER_ALT

MATCH_CENTER_REF  : True if the central base(s) of REF_SEQ equal REF (sanity check).
MATCH_CENTER_ALT  : True if the central base(s) of ALT_SEQ equal ALT (sanity check for SNPs).
"""

import sys
from pathlib import Path
from typing import Optional

import fire
from cyvcf2 import VCF
from pyfaidx import Fasta

# Reuse helpers from vep.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from vep import VALID_DNA_BASES, _extract_window, _make_allele_seq


def test_seqs(
    vcf: str,
    fasta: str,
    output: str = "seq_check.tsv",
    n: int = 10,
    half_window: int = 300,
) -> None:
    """Extract ref and alt windows for the first *n* scoreable SNPs and write to TSV.

    Parameters
    ----------
    vcf:         Path to input VCF/BCF.
    fasta:       Path to reference genome FASTA.
    output:      Output TSV path.
    n:           Number of variants to extract (default 10).
    half_window: Flanking bases on each side of the variant (default 300 -> 600 bp total).
    """
    genome = Fasta(fasta, as_raw=False, sequence_always_upper=True)
    vcf_reader = VCF(vcf)

    rows = []
    for variant in vcf_reader:
        if len(rows) >= n:
            break

        chrom = variant.CHROM
        pos = variant.POS
        ref = variant.REF
        alts = variant.ALT
        if alts is None:
            continue

        for alt in alts:
            if len(rows) >= n:
                break

            alt = (alt or "").upper()
            if not alt or any(base not in VALID_DNA_BASES for base in alt):
                continue

            try:
                ref_seq = _extract_window(genome, chrom, pos, half=half_window)
            except KeyError:
                print(f"WARNING: chromosome {chrom} not found in FASTA, skipping", file=sys.stderr)
                continue

            alt_seq = _make_allele_seq(ref_seq, ref, alt, half=half_window)
            if alt_seq is None:
                print(f"WARNING: could not construct alt window for {chrom}:{pos} {ref}>{alt}", file=sys.stderr)
                continue

            center = half_window
            ref_len = len(ref)
            match_ref = ref_seq[center : center + ref_len] == ref.upper()
            match_alt = alt_seq[center : center + len(alt)] == alt.upper()

            rows.append((chrom, pos, ref, alt, ref_seq, alt_seq, match_ref, match_alt))

    vcf_reader.close()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("CHROM\tPOS\tREF\tALT\tREF_SEQ\tALT_SEQ\tMATCH_CENTER_REF\tMATCH_CENTER_ALT\n")
        for chrom, pos, ref, alt, ref_seq, alt_seq, m_ref, m_alt in rows:
            fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{ref_seq}\t{alt_seq}\t{m_ref}\t{m_alt}\n")

    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)

    # Print a summary to stderr for quick visual check
    all_ref_ok = all(r[6] for r in rows)
    all_alt_ok = all(r[7] for r in rows)
    print(f"MATCH_CENTER_REF all True: {all_ref_ok}", file=sys.stderr)
    print(f"MATCH_CENTER_ALT all True (SNPs only): {all_alt_ok}", file=sys.stderr)

    if not all_ref_ok or not all_alt_ok:
        print("FAILURES:", file=sys.stderr)
        for chrom, pos, ref, alt, ref_seq, alt_seq, m_ref, m_alt in rows:
            if not m_ref or not m_alt:
                center = half_window
                print(
                    f"  {chrom}:{pos} {ref}>{alt}  "
                    f"ref_center={ref_seq[center:center+len(ref)]}  "
                    f"alt_center={alt_seq[center:center+len(alt)]}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    fire.Fire(test_seqs)
