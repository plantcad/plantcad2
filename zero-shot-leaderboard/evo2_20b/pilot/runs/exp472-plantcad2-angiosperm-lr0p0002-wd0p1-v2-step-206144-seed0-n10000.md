# exp472 PlantCAD2 angiosperm sampled leaderboard — step 206144

This evaluates the final checkpoint of `exp472-plantcad2-angiosperm-lr0p0002-wd0p1-v2` over all 20 current PlantCAD2 leaderboard rows. It is a 10,000-example-per-row estimate, not a leaderboard submission. Each row uses an unstratified random sample with shared seed 0; the materialized fixtures and SHA-256 manifest are retained so future checkpoints can use the exact same examples.

- Experiment/run name: `exp472-plantcad2-angiosperm-lr0p0002-wd0p1-v2`
- W&B group: <https://wandb.ai/eric-czech/marin/groups/exp472-plantcad2-baseline-sweep>
- Final artifact: `step-206144` after 206,145 completed training steps
- Checkpoint: <https://huggingface.co/plantcad/marindna-exp472/tree/fe3b167ff53d4d5d0fefbc92652acbef3b801831/exp472-plantcad2-angiosperm-lr0p0002-wd0p1-v2/hf/step-206144>
- Model: Qwen3, 973,178,880 parameters, vocabulary size 7, 8,192-bp context
- Sampling: 10,000 random unstratified examples per row, shared seed 0, dataset revision `d340debe0c8402c84f0696cd2002f87c2f7ba6db`
- Inference: BF16 model, explicit external FlashAttention-2, FP32 A/C/G/T softmax, `use_cache=False`, batch 32
- Hardware: Lambda `gpu_2x_h100_sxm5`, two H100 SXM5 80 GB GPUs
- Evaluation: 3:17:39 wall time, 400,000 sequence forwards, 3,276,800,000 scored tokens, 33.75 aggregate sequences/s, 21.59 GiB peak reserved per worker
- FlashAttention: `flash_attn::_flash_attn_forward` and `flash::flash_fwd_kernel` profiler-verified independently on both workers
- Full-scale estimate: 56.77 H100-hours, or 28.39 hours on two H100s, for all 1,727,943 current leaderboard rows
- Published comparison source: `plantcad/plantcad2-zeroshot-leaderboard` commit `f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29`
- Artifact: <https://huggingface.co/plantcad/marindna-exp472/tree/ba0bba0ef3bcb52ad21e1e13a95b87e6ff06ef02/exp472-plantcad2-angiosperm-lr0p0002-wd0p1-v2/results/step-206144/leaderboard-seed0-n10000>
- Artifact commit: `ba0bba0ef3bcb52ad21e1e13a95b87e6ff06ef02` (37 files, 1,734,863,972 bytes, verified exactly after upload)
- Lambda lifecycle: approximately 3 hours 41 minutes total rental time and about $30.9 at $8.38/hour; the exact evaluation instance was terminated and disappearance from `/instances` was verified

Published baselines are full-split results, whereas exp472 uses 10,000 sampled rows per task. PlantCAD is shown at its only published 512-bp context; exp472 and the other published models use 8,192 bp. Small apparent wins should therefore be treated as directional until a same-sample baseline rescore or full-split evaluation.

| Category | Species / task | Metric | exp472 | PlantCAD2.5-L | PlantCAD2-L | PlantCAD2-S | PlantCAD (512 bp) | evo2_20b |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservation | Andropogoneae, genome-wide | AUROC | 0.720 | 0.717 | 0.725 | 0.656 | 0.690 | 0.732 |
| Conservation | Poaceae, non-TIS CDS | AUROC | 0.689 | 0.729 | 0.713 | 0.646 | 0.726 | 0.862 |
| Conservation | Poaceae, TIS CDS | AUROC | 0.575 | 0.683 | 0.670 | 0.632 | 0.644 | 0.772 |
| Masked motif | Maize TIS (start) | accuracy | 0.404 | 0.696 | 0.657 | 0.545 | 0.520 | 0.602 |
| Masked motif | Maize TTS (stop) | accuracy | 0.341 | 0.446 | 0.410 | 0.230 | 0.237 | 0.453 |
| Masked motif | Maize splice donor | accuracy | 0.781 | 0.921 | 0.910 | 0.875 | 0.849 | 0.822 |
| Masked motif | Maize splice acceptor | accuracy | 0.771 | 0.913 | 0.900 | 0.854 | 0.829 | 0.826 |
| Masked motif | Tomato TIS (start) | accuracy | 0.326 | 0.612 | 0.596 | 0.527 | 0.389 | 0.586 |
| Masked motif | Tomato TTS (stop) | accuracy | 0.186 | 0.294 | 0.285 | 0.205 | 0.157 | 0.379 |
| Masked motif | Tomato splice donor | accuracy | 0.672 | 0.846 | 0.839 | 0.817 | 0.755 | 0.789 |
| Masked motif | Tomato splice acceptor | accuracy | 0.667 | 0.835 | 0.826 | 0.794 | 0.727 | 0.785 |
| Core/non-core | Maize TIS (start) | AUROC | 0.587 | 0.743 | 0.696 | 0.707 | 0.682 | 0.686 |
| Core/non-core | Maize TTS (stop) | AUROC | 0.640 | 0.626 | 0.608 | 0.598 | 0.608 | 0.686 |
| Core/non-core | Maize splice donor | AUROC | 0.739 | 0.842 | 0.808 | 0.764 | 0.708 | 0.776 |
| Core/non-core | Maize splice acceptor | AUROC | 0.751 | 0.873 | 0.836 | 0.763 | 0.707 | 0.804 |
| Core/non-core | Tomato TIS (start) | AUROC | 0.600 | 0.668 | 0.646 | 0.622 | 0.587 | 0.644 |
| Core/non-core | Tomato TTS (stop) | AUROC | 0.617 | 0.606 | 0.598 | 0.535 | 0.522 | 0.639 |
| Core/non-core | Tomato splice donor | AUROC | 0.738 | 0.783 | 0.767 | 0.735 | 0.720 | 0.766 |
| Core/non-core | Tomato splice acceptor | AUROC | 0.755 | 0.790 | 0.774 | 0.734 | 0.709 | 0.778 |
| Structural variant | Impact prediction | AUPRC | 0.847 | 0.745 | 0.841 | 0.795 | 0.823 | 0.860 |

Category means are descriptive because they combine AUROC, accuracy, and AUPRC. Exp472 scored 0.661 for conservation, 0.518 for masked motif, 0.678 for core/non-core, 0.847 for SV, and 0.620 across all 20 rows; PlantCAD2-Small scored 0.645, 0.606, 0.682, 0.795, and 0.652 respectively. The clearest exp472 gap is still masked-motif recovery. Core/non-core is approximately level with PlantCAD2-Small at the category-mean level, and sampled SV is already close to evo2_20b, but the sampled-versus-full comparison prevents interpreting small differences as leaderboard wins.

## Strand/context details

| Category | Species / task | Left | Reverse complement | Selected |
| :--- | :--- | ---: | ---: | :--- |
| Conservation | Andropogoneae, genome-wide | 0.719238 | 0.720114 | reverse complement |
| Conservation | Poaceae, non-TIS CDS | 0.688033 | 0.689103 | reverse complement |
| Conservation | Poaceae, TIS CDS | 0.509954 | 0.574973 | reverse complement |
| Masked motif | Maize TIS (start) | 0.104700 | 0.403700 | reverse complement |
| Masked motif | Maize TTS (stop) | 0.340700 | 0.094400 | left |
| Masked motif | Maize splice donor | 0.780800 | 0.333800 | left |
| Masked motif | Maize splice acceptor | 0.345500 | 0.771200 | reverse complement |
| Masked motif | Tomato TIS (start) | 0.064913 | 0.326165 | reverse complement |
| Masked motif | Tomato TTS (stop) | 0.185993 | 0.050625 | left |
| Masked motif | Tomato splice donor | 0.671567 | 0.198320 | left |
| Masked motif | Tomato splice acceptor | 0.240500 | 0.666700 | reverse complement |
| Core/non-core | Maize TIS (start) | 0.350382 | 0.587274 | reverse complement |
| Core/non-core | Maize TTS (stop) | 0.639995 | 0.372008 | left |
| Core/non-core | Maize splice donor | 0.738901 | 0.542906 | left |
| Core/non-core | Maize splice acceptor | 0.540326 | 0.751120 | reverse complement |
| Core/non-core | Tomato TIS (start) | 0.451157 | 0.599813 | reverse complement |
| Core/non-core | Tomato TTS (stop) | 0.616750 | 0.421366 | left |
| Core/non-core | Tomato splice donor | 0.737710 | 0.548149 | left |
| Core/non-core | Tomato splice acceptor | 0.544994 | 0.755485 | reverse complement |
| Structural variant | Impact prediction | 0.846537 | — | left/ref-mut |
