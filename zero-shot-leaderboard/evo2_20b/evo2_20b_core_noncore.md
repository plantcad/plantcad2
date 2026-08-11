# evo2_20b -- core_noncore

Max over context modes: left, right_reverse_complement. Modes present on disk: left, right_reverse_complement.

| Task | Split | AUROC | AUPRC | Best strand |
| :--- | :--- | :--- | :--- | :--- |
| **acceptor_core_noncore_classification** | test_maize | 0.804 | 0.962 | rc |
|  | test_tomato | 0.778 | 0.983 | rc |
| **donor_core_noncore_classification** | test_maize | 0.776 | 0.953 | fwd |
|  | test_tomato | 0.766 | 0.980 | fwd |
| **tis_core_noncore_classification** | test_tomato | 0.644 | 0.905 | rc |
| **tts_core_noncore_classification** | test_maize | 0.686 | 0.848 | fwd |
|  | test_tomato | 0.639 | 0.886 | fwd |

## Per context mode

| Task | Split | Mode | AUROC | AUPRC |
| :--- | :--- | :--- | :--- | :--- |
| acceptor_core_noncore_classification | test_maize | left | 0.546 | 0.850 |
| acceptor_core_noncore_classification | test_maize | right_reverse_complement | 0.804 | 0.962 |
| acceptor_core_noncore_classification | test_tomato | left | 0.489 | 0.934 |
| acceptor_core_noncore_classification | test_tomato | right_reverse_complement | 0.778 | 0.983 |
| donor_core_noncore_classification | test_maize | left | 0.776 | 0.953 |
| donor_core_noncore_classification | test_maize | right_reverse_complement | 0.549 | 0.853 |
| donor_core_noncore_classification | test_tomato | left | 0.766 | 0.980 |
| donor_core_noncore_classification | test_tomato | right_reverse_complement | 0.499 | 0.933 |
| tis_core_noncore_classification | test_tomato | left | 0.444 | 0.808 |
| tis_core_noncore_classification | test_tomato | right_reverse_complement | 0.644 | 0.905 |
| tts_core_noncore_classification | test_maize | left | 0.686 | 0.848 |
| tts_core_noncore_classification | test_maize | right_reverse_complement | 0.378 | 0.690 |
| tts_core_noncore_classification | test_tomato | left | 0.639 | 0.886 |
| tts_core_noncore_classification | test_tomato | right_reverse_complement | 0.386 | 0.783 |

## Incomplete

- `tis_core_noncore_classification_test_maize`: no output at all (array element produced nothing)
