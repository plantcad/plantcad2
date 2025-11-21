#!/usr/bin/env python3
"""
Evaluate model predictions for gene expression tasks.

For binary classification tasks:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)

For regression tasks:
- Spearman correlation
- Pearson correlation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr


def load_ground_truth(file_path):
    """Load ground truth file."""
    df = pd.read_csv(file_path, sep='\t')
    return df


def load_predictions(file_path):
    """Load prediction file."""
    df = pd.read_csv(file_path, sep='\t')
    return df


def evaluate_binary_classification(y_true, y_pred_proba):
    """Calculate AUROC and AUPRC for binary classification."""
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    return {
        'AUROC': auroc,
        'AUPRC': auprc
    }


def evaluate_regression(y_true, y_pred):
    """Calculate Spearman and Pearson correlation for regression."""
    spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
    pearson_corr, pearson_pval = pearsonr(y_true, y_pred)
    return {
        'Spearman': spearman_corr,
        'Pearson': pearson_corr
    }


def main():
    # Define base directory
    base_dir = Path(__file__).parent
    pred_dir = base_dir / 'predictions'

    # Find all ground truth files
    ground_truth_files = list(base_dir.glob('*.tsv'))

    # Results storage
    results = []

    # Process each ground truth file
    for gt_file in ground_truth_files:
        # Skip files in predictions directory
        if 'predictions' in str(gt_file):
            continue

        # Load ground truth
        gt_df = load_ground_truth(gt_file)

        # Determine task type and dataset name
        file_name = gt_file.stem  # e.g., 'osa_leaf_bin' or 'sly_leaf_exp'

        if '_bin' in file_name:
            task_type = 'binary'
        elif '_exp' in file_name:
            task_type = 'regression'
        else:
            continue

        # Find all prediction files for this dataset
        pred_files = list(pred_dir.glob(f'{file_name}_*.tsv'))

        print(f"\nProcessing {file_name} ({task_type})...")
        print(f"Found {len(pred_files)} prediction files")

        # Process each prediction file
        for pred_file in pred_files:
            # Extract model name
            model_name = pred_file.stem.replace(f'{file_name}_', '')

            # Load predictions
            pred_df = load_predictions(pred_file)

            # Ensure matching number of samples
            if len(gt_df) != len(pred_df):
                print(f"  Warning: Sample count mismatch for {model_name}")
                continue

            # Evaluate based on task type
            if task_type == 'binary':
                y_true = gt_df['Label'].values
                y_pred_proba = pred_df['probability_positive'].values

                metrics = evaluate_binary_classification(y_true, y_pred_proba)

                results.append({
                    'dataset': file_name,
                    'model': model_name,
                    'task': 'binary',
                    'AUROC': f"{metrics['AUROC']:.4f}",
                    'AUPRC': f"{metrics['AUPRC']:.4f}",
                    'Spearman': '-',
                    'Pearson': '-'
                })

                print(f"  {model_name:25s} | AUROC: {metrics['AUROC']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")

            elif task_type == 'regression':
                y_true = gt_df['Label'].values

                # Check for different possible column names
                # cnn-lstm uses 'probability_positive', others might use 'predicted_value'
                if 'probability_positive' in pred_df.columns:
                    y_pred = pred_df['probability_positive'].values
                elif 'predicted_value' in pred_df.columns:
                    y_pred = pred_df['predicted_value'].values
                else:
                    # Try to get the first non-index column
                    pred_columns = [col for col in pred_df.columns if col.lower() not in ['index', 'geneid', 'gene_id']]
                    if pred_columns:
                        y_pred = pred_df[pred_columns[0]].values
                        print(f"  Warning: Using column '{pred_columns[0]}' for {model_name}")
                    else:
                        print(f"  Error: No prediction column found for {model_name}")
                        continue

                metrics = evaluate_regression(y_true, y_pred)

                results.append({
                    'dataset': file_name,
                    'model': model_name,
                    'task': 'regression',
                    'AUROC': '-',
                    'AUPRC': '-',
                    'Spearman': f"{metrics['Spearman']:.4f}",
                    'Pearson': f"{metrics['Pearson']:.4f}"
                })

                print(f"  {model_name:25s} | Spearman: {metrics['Spearman']:.4f} | Pearson: {metrics['Pearson']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    output_file = base_dir / 'evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)


if __name__ == '__main__':
    main()
