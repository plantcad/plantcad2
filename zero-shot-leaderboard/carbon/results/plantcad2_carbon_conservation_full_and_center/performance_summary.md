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

- Uses the true motif in the sequence.
- Aligns the local sequence crop so the motif starts at a Carbon 6-mer boundary.
- Score is `0.5 * (forward true-motif 6-mer logp + reverse-complement true-motif 6-mer logp)`.
- `context_bp=1020`.

| Task | Split | n | Pos frac | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|
| `tis_core_noncore_classification` | `test_maize` | 36,409 | 0.777 | 0.5855 | 0.8019 |
| `tis_core_noncore_classification` | `test_tomato` | 35,478 | 0.840 | 0.5323 | 0.8378 |
| `tts_core_noncore_classification` | `test_maize` | 36,409 | 0.777 | 0.5670 | 0.7951 |
| `tts_core_noncore_classification` | `test_tomato` | 35,477 | 0.840 | 0.5113 | 0.8320 |
| `donor_core_noncore_classification` | `test_maize` | 144,550 | 0.852 | 0.6446 | 0.9009 |
| `donor_core_noncore_classification` | `test_tomato` | 140,451 | 0.941 | 0.5854 | 0.9472 |
| `acceptor_core_noncore_classification` | `test_maize` | 144,550 | 0.852 | 0.6557 | 0.9048 |
| `acceptor_core_noncore_classification` | `test_tomato` | 140,451 | 0.941 | 0.5919 | 0.9495 |

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
