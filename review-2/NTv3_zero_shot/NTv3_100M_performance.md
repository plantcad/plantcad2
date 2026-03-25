# Evaluation Results

## Core/Non-core Classification Tasks
| Task | Split | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| **acceptor_core_noncore_classification** | test_maize | 0.596 | 0.887 |
|  | test_tomato | 0.572 | 0.952 |
| **donor_core_noncore_classification** | test_maize | 0.594 | 0.886 |
|  | test_tomato | 0.624 | 0.958 |
| **tis_core_noncore_classification** | test_maize | 0.531 | 0.811 |
|  | test_tomato | 0.521 | 0.843 |
| **tts_core_noncore_classification** | test_maize | 0.438 | 0.740 |
|  | test_tomato | 0.430 | 0.795 |

## Conservation Tasks
| Task | Split | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| **conservation_within_andropogoneae** | test | 0.547 | 0.526 |
| **conservation_within_poaceae_non_tis** | test | 0.533 | 0.576 |
| **conservation_within_poaceae_tis** | test | 0.533 | 0.744 |

## Motif Recovery Tasks
| Task | Split | Token Accuracy | Motif Accuracy |
| :--- | :--- | :--- | :--- |
| **acceptor_recovery** | test_maize | 0.577 | 0.442 |
|  | test_tomato | 0.518 | 0.378 |
| **donor_recovery** | test_maize | 0.684 | 0.549 |
|  | test_tomato | 0.666 | 0.529 |
| **tis_recovery** | test_maize | 0.315 | 0.055 |
|  | test_tomato | 0.445 | 0.107 |
| **tts_recovery** | test_maize | 0.287 | 0.031 |
|  | test_tomato | 0.349 | 0.036 |

## Structural Variant Effect Prediction
| Task | Split | AUPRC |
| :--- | :--- | :--- |
| **structural_variant_effect_prediction** | test | 0.514 |