# evo2_20b (ours) vs PlantCAD2 zero-shot leaderboard

## Masked-motif recovery

Metric: motif accuracy. Leaderboard values are the 8192 bp context rows.
Local column is max over left, right_reverse_complement. `*` = only one strand available; `--` = not run locally.

| Task | Split | Evo2 | PlantCAD2-L | PlantCAD2.5-L | evo2_20b (ours) | strand | vs PC2.5-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| acceptor | maize | 0.738 | 0.900 | 0.913 | 0.826 | rc | -0.087 |
| donor | maize | 0.741 | 0.910 | 0.921 | 0.822 | fwd | -0.099 |
| tis | maize | 0.447 | 0.657 | 0.696 | 0.602 | rc | -0.094 |
| tts | maize | 0.256 | 0.410 | 0.446 | 0.453 | fwd | +0.007 |
| acceptor | tomato | 0.722 | 0.826 | 0.835 | 0.785 | rc | -0.050 |
| donor | tomato | 0.731 | 0.839 | 0.846 | 0.789 | fwd | -0.057 |
| tis | tomato | 0.485 | 0.596 | 0.612 | 0.586 | rc | -0.026 |
| tts | tomato | 0.268 | 0.285 | 0.294 | 0.379 | fwd | +0.085 |

## Evolutionary conservation

Metric: AUROC. Leaderboard values are the 8192 bp context rows.
Local column is max over left, right_reverse_complement. `*` = only one strand available; `--` = not run locally.

| Task | Split | Evo2 | PlantCAD2-L | PlantCAD2.5-L | evo2_20b (ours) | strand | vs PC2.5-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| andropogoneae | test | 0.691 | 0.725 | 0.717 | 0.732 | fwd | +0.015 |
| poaceae_non_tis | test | 0.822 | 0.713 | 0.729 | 0.862* | fwd | +0.133 |
| poaceae_tis | test | 0.533 | 0.670 | 0.683 | 0.772 | rc | +0.089 |

## Core vs non-core site classification

Metric: AUROC. Leaderboard values are the 8192 bp context rows.
Local column is max over left, right_reverse_complement. `*` = only one strand available; `--` = not run locally.

| Task | Split | Evo2 | PlantCAD2-L | PlantCAD2.5-L | evo2_20b (ours) | strand | vs PC2.5-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| acceptor | maize | 0.761 | 0.836 | 0.873 | 0.804 | rc | -0.069 |
| donor | maize | 0.754 | 0.808 | 0.842 | 0.776 | fwd | -0.066 |
| tis | maize | 0.624 | 0.696 | 0.743 | -- | -- | -- |
| tts | maize | 0.628 | 0.608 | 0.626 | 0.686 | fwd | +0.060 |
| acceptor | tomato | 0.744 | 0.774 | 0.790 | 0.778 | rc | -0.012 |
| donor | tomato | 0.745 | 0.767 | 0.783 | 0.766 | fwd | -0.017 |
| tis | tomato | 0.587 | 0.646 | 0.668 | 0.644 | rc | -0.024 |
| tts | tomato | 0.598 | 0.598 | 0.606 | 0.639 | fwd | +0.033 |

## Structural-variant effect prediction

Metric: AUPRC. Leaderboard values are the 8192 bp context rows.
Local column is max over left, right_reverse_complement. `*` = only one strand available; `--` = not run locally.

| Task | Split | Evo2 | PlantCAD2-L | PlantCAD2.5-L | evo2_20b (ours) | strand | vs PC2.5-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sv_impact | test | 0.771 | 0.841 | 0.745 | 0.860* | fwd | +0.115 |
