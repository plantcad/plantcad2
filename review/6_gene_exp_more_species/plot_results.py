#!/usr/bin/env python3
"""
Plot evaluation results as grouped bar plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_model_style(model_name):
    """
    Get color and alpha for a model following the original color scheme.

    Original scheme:
    - plantcad2-small/medium/large: C0 (blue), alpha 0.4/0.7/1.0
    - agront: gray, alpha 1.0
    - sup-plantcad2: C1 (orange), alpha 1.0
    - cnn-lstm: C1 (orange), alpha 0.5
    """
    if 'agront' in model_name.lower():
        return 'gray', 1.0
    elif 'sup-plantcad2' in model_name.lower() or 'sup_pcv2' in model_name.lower():
        return 'C1', 1.0
    elif 'cnn' in model_name.lower():
        return 'C1', 0.5
    elif 'plantcad2-small' in model_name.lower() or 'pcv2_1' in model_name.lower():
        return 'C0', 0.4
    elif 'plantcad2-medium' in model_name.lower() or 'pcv2_2' in model_name.lower():
        return 'C0', 0.7
    elif 'plantcad2-large' in model_name.lower() or 'pcv2_3' in model_name.lower():
        return 'C0', 1.0
    else:
        return 'gray', 0.5


def plot_metrics(df, task_type, metric_names, output_prefix):
    """
    Create grouped bar plots for specified metrics.

    Args:
        df: DataFrame with evaluation results
        task_type: 'binary' or 'regression'
        metric_names: list of metric column names to plot
        output_prefix: prefix for output files
    """
    # Filter data for this task type
    task_df = df[df['task'] == task_type].copy()

    if task_df.empty:
        print(f"No data found for task type: {task_type}")
        return

    # Define dataset order
    if task_type == 'binary':
        dataset_order = ['osa_leaf_bin', 'sly_leaf_bin']
        dataset_labels = ['Rice', 'Tomato']
    else:
        dataset_order = ['osa_leaf_exp', 'sly_leaf_exp']
        dataset_labels = ['Rice', 'Tomato']

    # Define model order (following original code)
    model_order = ['plantcad2-small', 'plantcad2-medium', 'plantcad2-large', 'agront', 'sup-plantcad2', 'cnn-lstm']

    # Get models that exist in the data, in the specified order
    available_models = set(task_df['model'].unique())
    models = [m for m in model_order if m in available_models]
    n_models = len(models)

    # Create a figure for each metric
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar width and positions
        bar_w = 0.15
        x = np.arange(len(dataset_order))

        # Plot bars for each model
        for i, model in enumerate(models):
            sub = task_df[task_df['model'] == model].copy()
            sub = sub.set_index('dataset').reindex(dataset_order).reset_index()

            # Convert metric values to float (they might be strings with '-')
            values = []
            for val in sub[metric]:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    values.append(0.0)

            # Get color and alpha for this model
            color, alpha = get_model_style(model)

            ax.bar(
                x + i * bar_w,
                values,
                bar_w,
                label=model,
                color=color,
                alpha=alpha
            )

        # Set x-axis labels
        ax.set_xticks(x + bar_w * (n_models - 1) / 2)
        ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=11, fontweight='bold')

        # Set y-axis label and limits
        ax.set_ylabel(metric, fontsize=14, fontweight='bold')

        if metric in ['Spearman', 'Pearson']:
            ax.set_ylim(0, 1.0)
        else:  # AUROC, AUPRC
            ax.set_ylim(0, 1.0)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add legend
        ax.legend(title='Model', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        # Save figure
        output_file = f"{output_prefix}_{task_type}_{metric.lower()}.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")

        plt.close()


def plot_combined_metrics(df, output_file):
    """
    Create a combined plot with all metrics in subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define dataset order
    binary_datasets = ['osa_leaf_bin', 'sly_leaf_bin']
    regression_datasets = ['osa_leaf_exp', 'sly_leaf_exp']
    binary_labels = ['Rice', 'Tomato']
    regression_labels = ['Rice', 'Tomato']

    # Define model order (following original code)
    model_order = ['plantcad2-small', 'plantcad2-medium', 'plantcad2-large', 'agront', 'sup-plantcad2', 'cnn-lstm']

    # Plot configurations
    plot_configs = [
        {'task': 'binary', 'metric': 'AUROC', 'datasets': binary_datasets, 'labels': binary_labels, 'ax': axes[0, 0]},
        {'task': 'binary', 'metric': 'AUPRC', 'datasets': binary_datasets, 'labels': binary_labels, 'ax': axes[0, 1]},
        {'task': 'regression', 'metric': 'Spearman', 'datasets': regression_datasets, 'labels': regression_labels, 'ax': axes[1, 0]},
        {'task': 'regression', 'metric': 'Pearson', 'datasets': regression_datasets, 'labels': regression_labels, 'ax': axes[1, 1]}
    ]

    for config in plot_configs:
        ax = config['ax']
        task_df = df[df['task'] == config['task']].copy()

        if task_df.empty:
            continue

        # Get models in the specified order
        available_models = set(task_df['model'].unique())
        models = [m for m in model_order if m in available_models]
        n_models = len(models)

        bar_w = 0.15
        x = np.arange(len(config['datasets']))

        # Plot bars for each model
        for i, model in enumerate(models):
            sub = task_df[task_df['model'] == model].copy()
            sub = sub.set_index('dataset').reindex(config['datasets']).reset_index()

            # Convert metric values to float
            values = []
            for val in sub[config['metric']]:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    values.append(0.0)

            # Get color and alpha for this model
            color, alpha = get_model_style(model)

            ax.bar(
                x + i * bar_w,
                values,
                bar_w,
                label=model,
                color=color,
                alpha=alpha
            )

        # Set x-axis labels
        ax.set_xticks(x + bar_w * (n_models - 1) / 2)
        ax.set_xticklabels(config['labels'], fontsize=11, fontweight='bold')

        # Set y-axis
        ax.set_ylabel(config['metric'], fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add legend only to first subplot
        if config['ax'] == axes[0, 0]:
            ax.legend(title='Model', fontsize=9, loc='lower right')

    plt.tight_layout()
    output_file_pdf = output_file.replace('.png', '.pdf')
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_file_pdf}")
    plt.close()


def main():
    # Define base directory
    base_dir = Path(__file__).parent
    results_file = base_dir / 'evaluation_results.csv'

    if not results_file.exists():
        print(f"Error: {results_file} not found. Please run evaluate_models.py first.")
        return

    # Load results
    df = pd.read_csv(results_file)

    print("Generating plots...")

    # Plot binary classification metrics
    plot_metrics(df, 'binary', ['AUROC', 'AUPRC'], str(base_dir / 'plot'))

    # Plot regression metrics
    plot_metrics(df, 'regression', ['Spearman', 'Pearson'], str(base_dir / 'plot'))

    # Create combined plot
    plot_combined_metrics(df, str(base_dir / 'plot_combined_all_metrics.pdf'))

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
