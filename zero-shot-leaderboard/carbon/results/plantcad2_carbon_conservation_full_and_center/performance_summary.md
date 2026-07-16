# Carbon-3B PlantCAD2 Zero-Shot Performance Summary

Model: `HuggingFaceBio/Carbon-3B`

Completed outputs:

- Conservation: `zero-shot-leaderboard/carbon/results/plantcad2_carbon_conservation_full_and_center/carbon_conservation_summary.json`
- Core/noncore classification: `zero-shot-leaderboard/carbon/results/plantcad2_carbon_core_noncore/carbon_core_noncore_summary.json`
- Motif recovery: `zero-shot-leaderboard/carbon/results/plantcad2_carbon_recovery/carbon_recovery_summary.json`
- Structural variant effect prediction: `zero-shot-leaderboard/carbon/results/plantcad2_carbon_structural_variant/carbon_structural_variant_summary.json`

## Conservation Tasks

Scoring:

- `Center 6-mer`: average of forward and reverse-complement log-probability for the center 6-mer.
- `Full mean LL`: average teacher-forced full-sequence log-likelihood per token, forward and reverse-complement averaged.

| Task | n | Pos frac | Center 6-mer AUROC | Center 6-mer AUPRC | Full mean LL AUROC | Full mean LL AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| `conservation_within_andropogoneae` | 38,060 | 0.500 | 0.6580 | 0.6570 | 0.6232 | 0.5801 |
| `conservation_within_poaceae_tis` | 36,662 | 0.727 | 0.6460 | 0.8281 | 0.5202 | 0.7442 |
| `conservation_within_poaceae_non_tis` | 183,685 | 0.563 | 0.7036 | 0.7404 | 0.5565 | 0.6134 |

## Core/Noncore Classification Tasks

Scoring:

- Prompt ends before the target motif.
- Scores the true motif by log-summing all compatible next-6-mer suffix completions.
- Score is `0.5 * (forward true-motif next-6mer logsumexp + reverse-complement true-motif next-6mer logsumexp)`.
- Examples with ambiguous true motifs outside `ACGT` are skipped for AUROC/AUPRC.
- `context_bp=1020`.

| Task | Split | n | Scored n | Skipped | Pos frac | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| `tis_core_noncore_classification` | `test_maize` | 36,409 | 36,409 | 0 | 0.777 | 0.6125 | 0.8175 |
| `tis_core_noncore_classification` | `test_tomato` | 35,478 | 35,477 | 1 | 0.840 | 0.5693 | 0.8525 |
| `tts_core_noncore_classification` | `test_maize` | 36,409 | 36,409 | 0 | 0.777 | 0.6159 | 0.8265 |
| `tts_core_noncore_classification` | `test_tomato` | 35,477 | 35,475 | 2 | 0.840 | 0.5584 | 0.8561 |
| `donor_core_noncore_classification` | `test_maize` | 144,550 | 144,550 | 0 | 0.852 | 0.6781 | 0.9122 |
| `donor_core_noncore_classification` | `test_tomato` | 140,451 | 140,451 | 0 | 0.941 | 0.6449 | 0.9555 |
| `acceptor_core_noncore_classification` | `test_maize` | 144,550 | 144,550 | 0 | 0.852 | 0.6814 | 0.9114 |
| `acceptor_core_noncore_classification` | `test_tomato` | 140,451 | 140,451 | 0 | 0.941 | 0.6462 | 0.9585 |

## Motif Recovery Tasks

Scoring:

- Prompt ends before the target motif.
- Predicts the best motif by scoring all motif candidates with forward and reverse-complement next-6-mer log-probability.
- `context_bp=1020`.

| Task | Split | n | Token Accuracy | Motif Accuracy |
|---|---|---:|---:|---:|
| `tis_recovery` | `test_maize` | 39,035 | 0.4860 | 0.3208 |
| `tis_recovery` | `test_tomato` | 35,484 | 0.5802 | 0.3743 |
| `tts_recovery` | `test_maize` | 39,035 | 0.4845 | 0.2237 |
| `tts_recovery` | `test_tomato` | 35,483 | 0.5309 | 0.1895 |
| `donor_recovery` | `test_maize` | 153,869 | 0.7287 | 0.6366 |
| `donor_recovery` | `test_tomato` | 140,456 | 0.7081 | 0.6025 |
| `acceptor_recovery` | `test_maize` | 153,869 | 0.7334 | 0.6566 |
| `acceptor_recovery` | `test_tomato` | 140,455 | 0.6882 | 0.6000 |

## Structural Variant Effect Prediction

Scoring:

- Score is the reference minus mutant mean log-likelihood, averaged over forward and reverse-complement sequences.
- Higher scores mean the mutant sequence is less likely than the reference.
- Sequences are trimmed to a multiple of the Carbon tokenizer k-mer size before scoring.

| Task | Split | n | Pos frac | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| `structural_variant_effect_prediction` | `test` | 18,075 | 0.424 | 0.8365 | 0.8314 |

## Remaining Tasks

| Task group | Status |
|---|---|
| All expected PlantCAD2 zero-shot task groups | Complete |
