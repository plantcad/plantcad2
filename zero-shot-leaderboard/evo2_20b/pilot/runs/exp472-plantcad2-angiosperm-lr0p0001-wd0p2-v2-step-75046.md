# exp472 PlantCAD2 angiosperm pilot — step 75046

This is a smoke/pilot evaluation, not a leaderboard submission. Each task uses 64
deterministically sampled examples. Binary tasks are balanced 32/32, so these values
should only be used to validate the evaluation path and identify promising checkpoints.

- W&B run ID: `exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2`
- Levanter checkpoint: `step-75046`
- Model: Qwen3, 973,178,880 parameters, vocabulary size 7
- Hardware: one Lambda Labs H100 SXM5 80 GB
- Conservation, motif, and core/non-core: maximum over forward and reverse-complement strands
- Structural variants: forward strand, matching the leaderboard protocol
- Artifacts: `plantcad/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2/results/step-75046/pilot-64`

| Category | Task / split | Pilot metric |
| :--- | :--- | :--- |
| Conservation | `conservation_within_poaceae_tis` / `test` | AUROC 0.448, AUPRC 0.475 |
| Conservation | `conservation_within_poaceae_non_tis` / `test` | AUROC 0.647, AUPRC 0.698 |
| Motif recovery | `tis_recovery` / `test_maize` | token accuracy 0.531, motif accuracy 0.219 |
| Motif recovery | `donor_recovery` / `test_tomato` | token accuracy 0.789, motif accuracy 0.703 |
| Structural variant | `structural_variant_effect_prediction` / `test` | AUPRC 0.763 |
| Core/non-core | `tis_core_noncore_classification` / `test_maize` | AUROC 0.539, AUPRC 0.539 |
| Core/non-core | `donor_core_noncore_classification` / `test_tomato` | AUROC 0.675, AUPRC 0.678 |
