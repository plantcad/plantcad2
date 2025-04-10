# Context Window Size Experiments — PlantCAD2 Expression Pipeline

This README documents experiments to evaluate how different genomic context window sizes affect gene expression prediction performance using the PlantCAD2 pipeline.

## Objective

Test the impact of varying upstream and downstream context lengths on model performance.

## Default Setup

The default context window used in the original setup is:

- **Upstream**: 4000 bp  
- **Downstream**: 4000 bp

## Experimental Variants

The following alternative context window configurations will be tested:

| Variant | Upstream (bp) | Downstream (bp) |
|---------|----------------|------------------|
| A       | 1000           | 1000             |
| B       | 2000           | 2000             |

These context windows will be applied consistently across all training, validation, and testing phases.

## Notes

- Each experiment will use the same dataset and model architecture, with only the input sequence length changing.
- Metrics such as ROC-AUC, PR-AUC, and loss will be tracked to evaluate performance shifts.
