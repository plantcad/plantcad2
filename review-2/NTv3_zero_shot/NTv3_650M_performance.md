# Evaluation Results

## Core/Non-core Classification Tasks
| Task | Split | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| **acceptor_core_noncore_classification** | test_maize | 0.627 | 0.899 |
|  | test_tomato | 0.584 | 0.955 |
| **donor_core_noncore_classification** | test_maize | 0.651 | 0.904 |
|  | test_tomato | 0.632 | 0.959 |
| **tis_core_noncore_classification** | test_maize | 0.656 | 0.867 |
|  | test_tomato | 0.579 | 0.871 |
| **tts_core_noncore_classification** | test_maize | 0.577 | 0.816 |
|  | test_tomato | 0.502 | 0.833 |

## Conservation Tasks
| Task | Split | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| **conservation_within_andropogoneae** | test | 0.612 | 0.611 |
| **conservation_within_poaceae_non_tis** | test | 0.684 | 0.767 |
| **conservation_within_poaceae_tis** | test | 0.613 | 0.787 |

## Motif Recovery Tasks
| Task | Split | Token Accuracy | Motif Accuracy |
| :--- | :--- | :--- | :--- |
| **acceptor_recovery** | test_maize | 0.670 | 0.560 |
|  | test_tomato | 0.626 | 0.506 |
| **donor_recovery** | test_maize | 0.778 | 0.686 |
|  | test_tomato | 0.742 | 0.645 |
| **tis_recovery** | test_maize | 0.522 | 0.288 |
|  | test_tomato | 0.592 | 0.342 |
| **tts_recovery** | test_maize | 0.432 | 0.114 |
|  | test_tomato | 0.492 | 0.132 |

## Structural Variant Effect Prediction
| Task | Split | AUPRC |
| :--- | :--- | :--- |
| **structural_variant_effect_prediction** | test | 0.642 |