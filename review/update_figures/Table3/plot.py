
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark():
    # Load data
    file_path = 'benchmark_results_torch2.1.1_trans4.49.0_mamba2.2.2_causal1.4.0.tsv'
    df = pd.read_csv(file_path, sep='\t')

    # Clean tokens/s column
    df['tokens/s'] = df['tokens/s'].str.replace(',', '').astype(float)

    # predefined order
    model_order = ['pcv2-small', 'pcv2-medium', 'pcv2-large']
    
    # Filter/Sort to match order if needed
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    
    # Group by Model and calculate mean and std
    grouped = df.groupby('Model', observed=True)['tokens/s'].agg(['mean', 'std'])

    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot
    plt.figure(figsize=(8, 6))
    
    # Bar plot
    # x positions
    x = range(len(grouped))
    
    plt.bar(x, grouped['mean'], yerr=grouped['std'], capsize=10, color=colors, alpha=0.9)
    
    # Add mean values on top of bars
    for i, (mean, std) in enumerate(zip(grouped['mean'], grouped['std'])):
        # Calculate y position: mean + std + small offset
        # Handle cases where std might be NaN if only one sample (though not the case here likely)
        err = std if not np.isnan(std) else 0
        y_pos = mean + err + (0.02 * grouped['mean'].max())
        plt.text(x[i], y_pos, f"{mean:,.0f}", ha='center', va='bottom', fontsize=12)
    
    plt.xticks(x, grouped.index, fontsize=12)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Tokens/s', fontsize=14)
    plt.title('Training Throughput (Tokens/s)', fontsize=16)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    # Save plot
    output_file = 'benchmark_tokens_per_second.pdf'
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_benchmark()
