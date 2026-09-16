# Arabidopsis amino-acid substitution benchmark (Kono et al. 2018)

Zero-shot evaluation of genomic language models on *A. thaliana* missense variants with
phenotypic effects versus common natural polymorphisms, using the training labels from
Kono et al. 2018, Table S2.

## Source data

Table S2 from the [Kono et al. 2018 supplemental material](https://figshare.com/articles/dataset/Supplemental_Material_for_Kono_et_al_2018/6998387),
saved as `results/ems/table-s2.csv` (ignored by git, as is everything under `results/`).
Only rows with a non-NA `Training (1=True positive, 0=True negative)` value are used:

| label | meaning | source | n |
| --- | --- | --- | ---: |
| 1 | substitution with a phenotypic effect | SwissProt / manual curation (mostly site-directed mutagenesis) | 2,910 |
| 0 | natural polymorphism, frequency 0.11-0.89 | Ensembl Variation (`ENSVATH…`) | 1,583 |

The annotation is Ensembl TAIR10 release 56 (`Arabidopsis_thaliana.TAIR10.{pep.all,cds.all,dna.toplevel}.fa`
and `.56.gtf`), default `/workdir/jz963/genomes/Arabidopsis_thaliana`.

## 0_data_prep

```bash
python 0_data_prep/check_annotation_consistency.py   # diagnostic, writes results/ems/training_annotation_check.tsv
python 0_data_prep/make_snv_dataset.py [--window 8192]  # writes results/ems/ems_training_snvs.tsv
```

`check_annotation_consistency.py` compares the table's `WT` residue with the annotated
protein at `Position`. 4,476 / 4,493 rows match. The 17 that don't are removed by
`make_snv_dataset.py`:

- 13 rows whose transcript is absent: `AT1G55860.2` (7; only `.1` is annotated), `AT4G16110.1` (4), `AT3G55270.1` (2)
- 4 rows whose WT residue differs, with no consistent offset: `AT4G14570.1` 618/707, `AT4G00450.1` 1665, `AT4G24580.1` 793

`make_snv_dataset.py` then, for each substitution, enumerates every single-nucleotide change
of the WT codon that produces the MUT amino acid and maps it to the genome through the GTF
CDS features. Only the substitution the table labels is emitted; other changes of the same
codon produce different amino acids and are not labeled.

| step | label 1 | label 0 |
| --- | ---: | ---: |
| substitutions after WT check | 2,904 | 1,572 |
| reachable by one nucleotide change | 2,084 | 1,572 |
| after removing the same SNV listed under a second isoform | 2,082 | 1,555 |
| SNV rows | 2,398 | 1,874 |

820 positives (mostly alanine scans such as K>A, H>A, W>A and phospho-mimics S>D / T>D)
need more than one nucleotide change and cannot be expressed as an SNV. Before isoform
deduplication, 3,076 substitutions map to one SNV, 522 to two and 58 to three; in the final table 451 of 3,805 positions carry more than one ALT.
No position or SNV has conflicting labels, and no position falls in more than one gene.

Built-in checks (all assertions in the script):

- CDS rebuilt from the GTF + genome equals `cds.all.fa` for all 996 transcripts
- WT codon translates to `WT`; genome base at `pos` equals `ref`
- window base at the center equals `ref`

### Output columns

`chrom, pos, ref, alt, gene, label, transcript, strand, aa_pos, wt_aa, mut_aa, codon_pos, ref_codon, alt_codon, source, sequence`

`pos` is 1-based on TAIR10. `ref`, `alt` and `sequence` are on the **+ strand** regardless of
gene strand. `sequence` spans `pos-4095` through `pos+4096`, so the SNV is at 1-based index
4096 (0-based 4095); 20 sampled windows matched `samtools faidx` byte-for-byte. No window
reaches a chromosome end.

## 1_plantcad2

```bash
conda activate transformers
BATCH_SIZE=16 bash 1_plantcad2/run_zero_shot.sh [model=kuleshov-group/PlantCAD2-Large-l48-d1536] [gpu=0]
```

Runs `review/8_tiny_model_eval/zero-shot-eval.py evo_cons` with `--token_idx 4095`, which
masks the SNV and scores it by P(ref). `summarize.py` then adds `-log(P(alt)/P(ref))`,
aggregates SNVs to substitutions (mean over `transcript, aa_pos, mut_aa`), and compares
against the Kono et al. predictors on the substitutions all of them score.
Outputs go to `results/ems/<model>/` (`logits.tsv`, `metrics.json`, `scores.tsv`, `summary.tsv`).

`PlantCaduceus_l32` only supports 512 bp, so it was run on the same table with
`--crop_length 512 --token_idx 255` (the center crop keeps the SNV at 0-based index 255).

### Results

SNV level (n = 4,272, 2,398 positive):

| model | context | masked-base acc. | P(ref) AUROC / AUPRC | -log(P(alt)/P(ref)) AUROC / AUPRC |
| --- | ---: | ---: | --- | --- |
| PlantCAD2-Large-l48-d1536 | 8,192 | 0.769 | 0.922 / 0.942 | 0.929 / 0.944 |
| PlantCaduceus_l32 | 512 | 0.787 | 0.924 / 0.943 | 0.930 / 0.944 |

Substitution level, on the 3,416 substitutions (2,057 positive) scored by every Kono et al.
predictor; model scores are the mean over each substitution's SNVs:

| method | AUROC | AUPRC |
| --- | ---: | ---: |
| LRT-logistic | 0.954 | 0.967 |
| PolyPhen2 | 0.938 | 0.953 |
| PlantCaduceus_l32, -log(P(alt)/P(ref)) | 0.929 | 0.951 |
| PlantCAD2-Large-l48-d1536, -log(P(alt)/P(ref)) | 0.926 | 0.949 |
| Provean | 0.925 | 0.941 |
| PlantCAD2-Large-l48-d1536, P(ref) | 0.920 | 0.948 |
| SIFT | 0.915 | 0.940 |
| GERP++ | 0.912 | 0.942 |
| MAPP | 0.912 | 0.937 |

PlantCAD2-Large with 8,192 bp does not outperform PlantCaduceus_l32 with 512 bp here, and their
`-log(P(alt)/P(ref))` scores have Spearman ρ = 0.83. Model and context length differ between
the two runs, so this does not isolate the effect of context.

## Caveats

- **Negatives are common polymorphisms, positives are mostly engineered mutations in
  functional sites.** Anything tracking conservation separates them easily; every method
  above reaches AUROC > 0.91, so the gaps between methods are small.
- **Negatives with multiple SNVs include unobserved alleles.** Only one ALT was observed per
  polymorphism; the others are enumerated because the table records only the amino acid.
  Substitution-level aggregation avoids weighting these extra rows.
- `LRT-logistic` may have been fit on these labels, so it is likely an optimistic baseline.
