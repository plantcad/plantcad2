# Genomic-class annotation for the Evolutionary_constraint benchmark

`annotate_genomic_class.R` labels every SNP in the benchmark with a genomic class
(`CDS`, `utr5`, `utr3`, `promoter`, `intron`, `intergenic`) and a TE overlap flag,
so zero-shot and fine-tuned results can be broken down by sequence context.

```bash
Rscript annotate_genomic_class.R <input.tsv> <output.tsv> [gene_gff] [repeat_gff] [promoter_bp]
```

Input needs `chrom` and `pos` columns; any other columns are carried through
untouched. Defaults point at the Sorghum v3.1.1 GFFs and the benchmark tables on
cbsupennell01.

## Relationship to `../stratify_noncoding_type.R`

Both assign the same non-coding classes from the same GFF with the same 1 kb
promoter window and the same precedence. They differ in scope:

| | `stratify_noncoding_type.R` | `annotate_genomic_class.R` |
| --- | --- | --- |
| input | needs a `type` column | needs only `chrom`, `pos` |
| scope | rewrites rows typed `Noncoding` | annotates every row |
| CDS | assumed already assigned upstream | assigned by the script |
| TE flag | no | yes, as an independent column |

Use the existing script to refine an already-typed table; use this one to
annotate a raw benchmark table from scratch. Neither replaces the other, and
`stratify_noncoding_type.R` is unchanged.

## Coordinates

`pos` is **1-based** on `Sbicolor_454_v3.0.1.fa` and is the center base of the
window, i.e. sequence index 4096 with 1-based indexing; the window spans
`pos-4095` through `pos+4096`. This was verified against the assembly with
`samtools faidx` on sampled rows, where the extracted 8,192 bp interval matched
the stored sequence byte-for-byte. `GRanges` is 1-based, so the script uses
`start = end = pos` with no offset. The Sorghum GFF uses `Chr01`..`Chr10`, which
match the benchmark tables directly, so no `chr` prefix is added, unlike the
B73 v5 equivalents elsewhere in this repo.

## Class precedence

Later overrides earlier:

```
intergenic < intron < promoter < utr5 < utr3 < CDS
```

A site can fall in several categories across different transcripts. CDS is last
so coding sites are never counted as regulatory. `is_TE` comes from the
RepeatMasker GFF and is deliberately a separate column: it does not override
`finalType`, so the two can be conditioned on independently.

## Results on the current benchmark

Splits are **held-out chromosome**, not random: train is Chr01-Chr09 (858,086
rows), valid is Chr10 (38,060 rows). They share 0 positions and 0 identical
8,192 bp sequences.

| finalType | train | valid | % label=1 |
| --- | ---: | ---: | ---: |
| CDS | 593,553 | 27,969 | 50.5 |
| intron | 145,970 | 6,471 | 52.2 |
| intergenic | 44,637 | 1,136 | 25.2 |
| promoter | 31,249 | 1,065 | 37.9 |
| utr3 | 27,056 | 797 | 68.7 |
| utr5 | 15,621 | 622 | 70.8 |

Two things worth knowing before stratifying on these:

- **`finalType` is not balanced across labels.** UTRs are strongly enriched for
  conserved sites (utr5 71%, utr3 69% label=1) while intergenic is depleted
  (25%). A per-class accuracy comparison is partly reading this skew, not model
  behavior. TE overlap, by contrast, is matched exactly 1:1 between labels within
  each split.
- **Negatives sit close to their matched positives.** On Chr01 every positive has
  a negative within 8,192 bp, median distance 283 bp, so a positive and its
  matched negative have heavily overlapping windows and differ chiefly at the
  center base. The pairing never crosses the train/valid boundary, since it is
  always within a chromosome.

The generated tables are not committed here. They are published alongside the
sequences at
[JingjingZhai/full_conservation_within_andropogoneae](https://huggingface.co/datasets/JingjingZhai/full_conservation_within_andropogoneae),
row-aligned with the benchmark files so they can be concatenated column-wise with
no join.

`test.tsv` in this benchmark is maize B73, not Sorghum, so it needs the B73 v5
GFF rather than the defaults above. It is not annotated yet.
