# evo2_20b vs PlantCAD2 leaderboard -- masked-motif recovery

Metric: motif accuracy (exact match over all motif positions).
Local column is max over left, right_reverse_complement; the leaderboard's own Evo2 row
(evo2_7b) uses the same best=fwd/rc protocol. `*` = only one strand available.

| Task | Split | Evo2 | PlantCAD2-L | PlantCAD2.5-L | evo2_20b (ours) | best strand | vs PC2.5-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| acceptor | maize | 0.738 | 0.900 | 0.913 | 0.826 | rc | -0.087 |
| donor | maize | 0.741 | 0.910 | 0.921 | 0.822 | fwd | -0.099 |
| tis | maize | 0.447 | 0.657 | 0.696 | 0.602 | rc | -0.094 |
| tts | maize | 0.256 | 0.410 | 0.446 | 0.453 | fwd | +0.007 |
| acceptor | tomato | 0.722 | 0.826 | 0.835 | 0.339* | fwd | -0.496 |
| donor | tomato | 0.731 | 0.839 | 0.846 | -- | -- | -- |
| tis | tomato | 0.485 | 0.596 | 0.612 | 0.586 | rc | -0.026 |
| tts | tomato | 0.268 | 0.285 | 0.294 | 0.379 | fwd | +0.085 |
