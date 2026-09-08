# Maize allele-frequency frozen linear probe on CoreWeave

This is a supervised companion to, not a replacement for, `../allele_frequency/`. It extracts frozen final-layer REF/ALT representations for the same 94,075 variants at 8,192 bp, fits an independent ridge probe for each checkpoint, and compares held-out predictions with the existing zero-shot LLR on identical rows.

## Protocol

- Primary representation: FP32 whole-window mean of the final hidden layer, averaged across forward/reverse-complement views per allele; feature `[REF, ALT - REF]`.
- Diagnostic representation: the analogous variant-token state, retained from the same forward pass.
- Split: seed 0, genomic 1-Mb blocks, approximately 70% train / 30% test, stratified by consequence and within-consequence AF decile; train windows overlapping test windows are purged.
- Probe: `StandardScaler -> Ridge`; alpha is selected by five-fold grouped train-only Spearman CV. Raw ALT AF is the target.
- Report: held-out pooled and within-consequence Spearman with paired 1-Mb-block bootstrap intervals for the probe, matched zero-shot LLR, and their difference.
- Raw FP32 embeddings remain as chunked CWS3 artifacts. Split, fitted probes, row-level predictions, metrics, and provenance are uploaded under the checkpoint's Hugging Face result path.

## Run

```bash
source ~/oa.env
source ~/.zshrc

CHECKPOINT=056t NODES=4 JOB_NAME=plantcad2-maize-af-probe-056t-20260904-v1 \
  bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/allele_frequency_probe/submit.sh
```

`CHECKPOINT` is `022t`, `039t`, or `056t`. The wrapper always uses batch priority and an H100x8 cluster.
