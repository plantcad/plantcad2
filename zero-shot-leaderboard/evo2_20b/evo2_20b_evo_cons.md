# evo2_20b -- evo_cons

Max over context modes: left, right_reverse_complement. Modes present on disk: left, right_reverse_complement.

| Task | Split | AUROC | AUPRC | Best strand |
| :--- | :--- | :--- | :--- | :--- |
| **conservation_within_andropogoneae** | test | 0.732 | 0.751 | fwd |
| **conservation_within_poaceae_non_tis** | test | 0.862 | 0.909 | rc |
| **conservation_within_poaceae_tis** | test | 0.772 | 0.902 | rc |

## Per context mode

| Task | Split | Mode | AUROC | AUPRC |
| :--- | :--- | :--- | :--- | :--- |
| conservation_within_andropogoneae | test | left | 0.732 | 0.751 |
| conservation_within_andropogoneae | test | right_reverse_complement | 0.730 | 0.747 |
| conservation_within_poaceae_non_tis | test | left | 0.862 | 0.909 |
| conservation_within_poaceae_non_tis | test | right_reverse_complement | 0.862 | 0.909 |
| conservation_within_poaceae_tis | test | left | 0.535 | 0.767 |
| conservation_within_poaceae_tis | test | right_reverse_complement | 0.772 | 0.902 |
