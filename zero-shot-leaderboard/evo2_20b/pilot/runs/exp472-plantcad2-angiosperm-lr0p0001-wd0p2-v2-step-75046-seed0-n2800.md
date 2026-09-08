# exp472 PlantCAD2 angiosperm sampled leaderboard — step 75046

This is an early-training estimate over all 20 leaderboard rows, not a leaderboard
submission. Each row uses 2,800 unstratified random examples selected with the same
seed 0. The 200,000-row shuffle buffer exceeds every current split (maximum 183,685),
so the sample is a deterministic full-split shuffle rather than a local streaming
window. The resulting 56,000 examples are retained for exact checkpoint-to-checkpoint
comparisons.

- W&B run ID: `exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2`
- Levanter checkpoint: `step-75046`
- Model: Qwen3, 973,178,880 parameters, vocabulary size 7
- Context: 8,192 bp
- Hardware: one Lambda Labs H100 SXM5 80 GB
- Evaluation wall time: 1:58:04; about $8.44 at $4.29/H100-hour
- Conservation, motif, and core/non-core: maximum over forward and
  reverse-complement contexts
- Structural variants: left context only, matching the leaderboard comparison path
- Published comparison source: `plantcad/plantcad2-zeroshot-leaderboard` commit
  `f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29`
- Artifacts: `plantcad/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2/results/step-75046/leaderboard-seed0-n2800`
- Artifact commit: `93f1f5088bb83747ecca0e1866b3d0e9f48c3fcf`

Published baselines below are full-split results, whereas exp472 uses the sampled
2,800 rows per task. PlantCAD is shown at its only published context, 512 bp; the
other published models and exp472 use 8,192 bp.

| Category | Species / task | Metric | exp472 | PlantCAD2.5-L | PlantCAD2-L | PlantCAD (512 bp) | evo2_20b |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Conservation | Andropogoneae, genome-wide | AUROC | 0.672 | 0.717 | 0.725 | 0.690 | 0.732 |
| Conservation | Poaceae, non-TIS CDS | AUROC | 0.657 | 0.729 | 0.713 | 0.726 | 0.862 |
| Conservation | Poaceae, TIS CDS | AUROC | 0.545 | 0.683 | 0.670 | 0.644 | 0.772 |
| Masked motif | Maize TIS (start) | accuracy | 0.220 | 0.696 | 0.657 | 0.520 | 0.602 |
| Masked motif | Maize TTS (stop) | accuracy | 0.164 | 0.446 | 0.410 | 0.237 | 0.453 |
| Masked motif | Maize splice donor | accuracy | 0.624 | 0.921 | 0.910 | 0.849 | 0.822 |
| Masked motif | Maize splice acceptor | accuracy | 0.594 | 0.913 | 0.900 | 0.829 | 0.826 |
| Masked motif | Tomato TIS (start) | accuracy | 0.214 | 0.612 | 0.596 | 0.389 | 0.586 |
| Masked motif | Tomato TTS (stop) | accuracy | 0.123 | 0.294 | 0.285 | 0.157 | 0.379 |
| Masked motif | Tomato splice donor | accuracy | 0.548 | 0.846 | 0.839 | 0.755 | 0.789 |
| Masked motif | Tomato splice acceptor | accuracy | 0.502 | 0.835 | 0.826 | 0.727 | 0.785 |
| Core/non-core | Maize TIS (start) | AUROC | 0.560 | 0.743 | 0.696 | 0.682 | 0.686 |
| Core/non-core | Maize TTS (stop) | AUROC | 0.545 | 0.626 | 0.608 | 0.608 | 0.686 |
| Core/non-core | Maize splice donor | AUROC | 0.658 | 0.842 | 0.808 | 0.708 | 0.776 |
| Core/non-core | Maize splice acceptor | AUROC | 0.656 | 0.873 | 0.836 | 0.707 | 0.804 |
| Core/non-core | Tomato TIS (start) | AUROC | 0.565 | 0.668 | 0.646 | 0.587 | 0.644 |
| Core/non-core | Tomato TTS (stop) | AUROC | 0.557 | 0.606 | 0.598 | 0.522 | 0.639 |
| Core/non-core | Tomato splice donor | AUROC | 0.649 | 0.783 | 0.767 | 0.720 | 0.766 |
| Core/non-core | Tomato splice acceptor | AUROC | 0.704 | 0.790 | 0.774 | 0.709 | 0.778 |
| Structural variant | Impact prediction | AUPRC | 0.750 | 0.745 | 0.841 | 0.823 | 0.860 |

The clearest early-training signal is task dependent. Conservation is already
meaningful, tomato core/non-core splice acceptor is close to older PlantCAD, and SV is
roughly level with PlantCAD2.5-L on this sample. Exact masked-motif recovery, especially
start and stop codons, remains far behind the mature models. Because the sampled and
published columns use different example sets, small differences such as the SV result
should not be treated as a genuine leaderboard win without a full-split run or
same-sample baseline rescoring.
