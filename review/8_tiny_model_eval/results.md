# Evaluation Results

## Core/Non-core Classification Tasks
| Task | Split | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| **acceptor_core_noncore_classification** | test_maize | 0.750 | 0.939 |
|  | test_tomato | 0.707 | 0.971 |
| **donor_core_noncore_classification** | test_maize | 0.761 | 0.943 |
|  | test_tomato | 0.719 | 0.972 |
| **tis_core_noncore_classification** | test_maize | 0.716 | 0.896 |
|  | test_tomato | 0.609 | 0.885 |
| **tts_core_noncore_classification** | test_maize | 0.567 | 0.809 |
|  | test_tomato | 0.486 | 0.828 |

## Conservation Tasks
| Task | Split | AUROC | AUPRC |
| :--- | :--- | :--- | :--- |
| **conservation_within_andropogoneae** | test | 0.593 | 0.599 |
| **conservation_within_poaceae_non_tis** | test | 0.606 | 0.691 |
| **conservation_within_poaceae_tis** | test | 0.616 | 0.779 |

## Motif Recovery Tasks
| Task | Split | Token Accuracy | Motif Accuracy |
| :--- | :--- | :--- | :--- |
| **acceptor_recovery** | test_maize | 0.861 | 0.817 |
|  | test_tomato | 0.819 | 0.760 |
| **donor_recovery** | test_maize | 0.891 | 0.845 |
|  | test_tomato | 0.849 | 0.792 |
| **tis_recovery** | test_maize | 0.626 | 0.445 |
|  | test_tomato | 0.664 | 0.454 |
| **tts_recovery** | test_maize | 0.455 | 0.149 |
|  | test_tomato | 0.488 | 0.147 |

## Structural Variant Effect Prediction
| Task | Split | AUPRC |
| :--- | :--- | :--- |
| **structural_variant_effect_prediction** | test | 0.635 |