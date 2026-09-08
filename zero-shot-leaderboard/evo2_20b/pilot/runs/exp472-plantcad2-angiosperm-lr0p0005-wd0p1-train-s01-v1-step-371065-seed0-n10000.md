# MarinDNA 1B 20E sampled leaderboard — step 371065

This evaluates the final post-cooldown checkpoint of `exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1` over all 20 current PlantCAD2 leaderboard rows. It is a 10,000-example-per-row estimate, not a leaderboard submission. It reuses the exact materialized seed-0 fixtures from MarinDNA 1B 10E, so checkpoint deltas are paired; published baselines use full splits.

- Display name: MarinDNA 1B 20E; the previous step-206144 checkpoint is MarinDNA 1B 10E
- Final native artifact: `step-371065` after cooldown; final HF export: <https://huggingface.co/plantcad/marindna-exp472/tree/e56696e49dbc4c5d904507983df901fbe9d6d32d/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/hf/step-371065>
- Model: Qwen3, approximately 1B parameters, vocabulary size 7, 8,192-bp context
- Sampling: 10,000 random unstratified examples per row, shared seed 0, exact same 20 fixture files and SHA-256 manifest as MarinDNA 1B 10E, dataset revision `d340debe0c8402c84f0696cd2002f87c2f7ba6db`
- Inference: BF16 model, explicit external FlashAttention-2, FP32 A/C/G/T softmax, `use_cache=False`, batch 32
- Evaluation logic: maximum of the completed forward and reverse-complement task-level metrics for conservation, motif, and core/non-core; left/reference-versus-mutant score for SV
- Hardware: Lambda `gpu_2x_h100_sxm5`, two H100 SXM5 80 GB GPUs
- Evaluation: 3:17:34 wall time, 400,000 sequence forwards, 3,276,800,000 scored tokens, 33.76 aggregate sequences/s, 21.59 GiB peak reserved per worker
- FlashAttention: `flash_attn::_flash_attn_forward` and `flash::flash_fwd_kernel` profiler-verified independently on both workers
- Full-scale estimate: 56.51 H100-hours, or 28.26 hours on two H100s, for all 1,727,943 current leaderboard rows
- Published comparison source: `plantcad/plantcad2-zeroshot-leaderboard` commit `f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29`
- Artifact: <https://huggingface.co/plantcad/marindna-exp472/tree/2090e4c6c52393b5b8626ea9be70205c79935d53/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/results/step-371065/leaderboard-seed0-n10000>
- Artifact commit: `2090e4c6c52393b5b8626ea9be70205c79935d53` (42 files, 1,734,923,623 bytes, verified exactly after upload)

MarinDNA 1B 20E improved all 20 paired task rows over MarinDNA 1B 10E. Group means moved from 0.661 to 0.669 for conservation, 0.518 to 0.561 for masked motif, 0.678 to 0.696 for core/non-core, and 0.847 to 0.848 for SV. The composite, defined as the unweighted mean of those four group scores, moved from 0.676 to 0.693. The largest task gains were tomato start-codon recovery (+0.065) and maize start-codon recovery (+0.060). Masked-motif recovery improved the most as a group (+0.043), but remains the clearest gap to PlantCAD2-Small (0.606) and the larger published models. SV is effectively flat (+0.001).

Published baselines are full-split results, whereas both MarinDNA columns use the exact same sampled rows. Apparent differences against published models are directional; the 20E-versus-10E deltas are the directly paired comparison. Each group mean weights its task rows equally, while the composite weights the four groups equally regardless of their row counts. The composite is descriptive because its groups use AUROC, accuracy, and AUPRC.

| Category | Species / task | Metric | MarinDNA 1B 20E | MarinDNA 1B 10E | Δ | PlantCAD2.5-L | PlantCAD2-L | PlantCAD2-S | PlantCAD¹ | evo2_20b |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservation | Andropogoneae, genome-wide | AUROC | **0.722** | 0.720 | +0.002 | 0.717 | 0.725 | 0.655 | 0.690 | 0.732 |
| Conservation | Poaceae, non-TIS CDS | AUROC | **0.699** | 0.689 | +0.010 | 0.729 | 0.713 | 0.646 | 0.726 | 0.862 |
| Conservation | Poaceae, TIS CDS | AUROC | **0.584** | 0.575 | +0.009 | 0.683 | 0.670 | 0.632 | 0.644 | 0.772 |
| Masked motif | Maize TIS (start) | Accuracy | **0.464** | 0.404 | +0.060 | 0.696 | 0.657 | 0.545 | 0.520 | 0.602 |
| Masked motif | Maize TTS (stop) | Accuracy | **0.371** | 0.341 | +0.030 | 0.446 | 0.410 | 0.230 | 0.237 | 0.453 |
| Masked motif | Maize splice donor | Accuracy | **0.816** | 0.781 | +0.035 | 0.921 | 0.910 | 0.875 | 0.849 | 0.822 |
| Masked motif | Maize splice acceptor | Accuracy | **0.806** | 0.771 | +0.034 | 0.913 | 0.900 | 0.854 | 0.829 | 0.826 |
| Masked motif | Tomato TIS (start) | Accuracy | **0.391** | 0.326 | +0.065 | 0.612 | 0.596 | 0.527 | 0.389 | 0.586 |
| Masked motif | Tomato TTS (stop) | Accuracy | **0.215** | 0.186 | +0.029 | 0.294 | 0.285 | 0.205 | 0.157 | 0.379 |
| Masked motif | Tomato splice donor | Accuracy | **0.715** | 0.672 | +0.044 | 0.846 | 0.839 | 0.817 | 0.755 | 0.789 |
| Masked motif | Tomato splice acceptor | Accuracy | **0.711** | 0.667 | +0.044 | 0.835 | 0.826 | 0.794 | 0.727 | 0.785 |
| Core/non-core | Maize TIS (start) | AUROC | **0.612** | 0.587 | +0.025 | 0.743 | 0.696 | 0.707 | 0.682 | 0.686 |
| Core/non-core | Maize TTS (stop) | AUROC | **0.643** | 0.640 | +0.003 | 0.626 | 0.608 | 0.598 | 0.608 | 0.686 |
| Core/non-core | Maize splice donor | AUROC | **0.768** | 0.739 | +0.029 | 0.842 | 0.808 | 0.764 | 0.708 | 0.776 |
| Core/non-core | Maize splice acceptor | AUROC | **0.789** | 0.751 | +0.038 | 0.873 | 0.836 | 0.763 | 0.707 | 0.804 |
| Core/non-core | Tomato TIS (start) | AUROC | **0.611** | 0.600 | +0.012 | 0.668 | 0.646 | 0.622 | 0.587 | 0.644 |
| Core/non-core | Tomato TTS (stop) | AUROC | **0.632** | 0.617 | +0.015 | 0.606 | 0.598 | 0.535 | 0.522 | 0.639 |
| Core/non-core | Tomato splice donor | AUROC | **0.752** | 0.738 | +0.014 | 0.783 | 0.767 | 0.735 | 0.720 | 0.766 |
| Core/non-core | Tomato splice acceptor | AUROC | **0.765** | 0.755 | +0.010 | 0.790 | 0.774 | 0.734 | 0.709 | 0.778 |
| Structural variant | Impact prediction | AUPRC | **0.848** | 0.847 | +0.001 | 0.745 | 0.841 | 0.795 | 0.823 | 0.860 |

¹ PlantCAD has only a published 512-bp row; all other columns use 8,192 bp.

## Strand/context details

| Category | Species / task | Forward | Reverse complement | Selected |
| :--- | :--- | ---: | ---: | :--- |
| Conservation | Andropogoneae, genome-wide | 0.721915 | 0.715406 | forward |
| Conservation | Poaceae, non-TIS CDS | 0.699297 | 0.691149 | forward |
| Conservation | Poaceae, TIS CDS | 0.504201 | 0.584367 | reverse complement |
| Masked motif | Maize TIS (start) | 0.114000 | 0.463500 | reverse complement |
| Masked motif | Maize TTS (stop) | 0.371000 | 0.097600 | forward |
| Masked motif | Maize splice donor | 0.815900 | 0.362100 | forward |
| Masked motif | Maize splice acceptor | 0.369000 | 0.805600 | reverse complement |
| Masked motif | Tomato TIS (start) | 0.077015 | 0.390978 | reverse complement |
| Masked motif | Tomato TTS (stop) | 0.215308 | 0.059230 | forward |
| Masked motif | Tomato splice donor | 0.715172 | 0.230023 | forward |
| Masked motif | Tomato splice acceptor | 0.264700 | 0.710800 | reverse complement |
| Core/non-core | Maize TIS (start) | 0.340754 | 0.612099 | reverse complement |
| Core/non-core | Maize TTS (stop) | 0.643036 | 0.378630 | forward |
| Core/non-core | Maize splice donor | 0.767609 | 0.545065 | forward |
| Core/non-core | Maize splice acceptor | 0.549732 | 0.788859 | reverse complement |
| Core/non-core | Tomato TIS (start) | 0.442138 | 0.611476 | reverse complement |
| Core/non-core | Tomato TTS (stop) | 0.631608 | 0.423395 | forward |
| Core/non-core | Tomato splice donor | 0.752166 | 0.548914 | forward |
| Core/non-core | Tomato splice acceptor | 0.535515 | 0.765075 | reverse complement |
| Structural variant | Impact prediction | 0.847536 | — | forward/ref-mut |

The selected context flipped from reverse complement at 10E to forward at 20E for the first two conservation rows. Motif and core/non-core winning orientations were unchanged. The aggregation rule itself was unchanged: selection is still the maximum of the two completed task-level metrics, never a per-example maximum.
