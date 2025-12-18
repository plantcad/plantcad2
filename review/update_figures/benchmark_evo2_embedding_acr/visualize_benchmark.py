
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_markdown_table(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract table lines
    table_lines = []
    for line in lines:
        if line.strip().startswith('|'):
            table_lines.append(line.strip())
            
    if not table_lines:
        return None
        
    # Parse header
    header = [c.strip() for c in table_lines[0].strip('|').split('|')]
    
    # Parse data
    data = []
    for line in table_lines[2:]: # Skip header and separator
        row = [c.strip() for c in line.strip('|').split('|')]
        if len(row) == len(header):
            data.append(row)
            
    df = pd.DataFrame(data, columns=header)
    
    # Convert numeric columns
    numeric_cols = ['AUPRC (XGBoost)', 'AUPRC (Neural Network)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
        
    return df

def plot_grouped_benchmark(df, classifier_col, model_order, model_colors, output_path):
    species_list = df['Species'].unique()
    species_labels = [s.replace('_', ' ').title() for s in species_list]
    
    n_species = len(species_list)
    n_models = len(model_order)
    
    # Pivot the dataframe to get a matrix of scores: Rows=Species, Cols=Models
    # This ensures we handle data alignment correctly
    pivot_df = df.pivot(index='Species', columns='Model', values=classifier_col)
    
    # Reindex to match specific orders
    pivot_df = pivot_df.loc[species_list, model_order]
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Calculation for bar width and positioning
    # Calculation for bar width and positioning
    # Total width allocated for one group (species) is usually around 0.8
    total_group_width = 0.9
    bar_width = total_group_width / n_models
    
    # X locations for the groups
    indices = np.arange(n_species)
    
    # Plot bars for each model
    for i, model in enumerate(model_order):
        scores = pivot_df[model].values
        color = model_colors[i]
        
        # Calculate the offset for this specific model within the group
        # center index - half_width + offset
        # Or more simply: start from left edge of the group
        # Left edge relative to center: -total_group_width / 2
        # Offset for this bar: i * bar_width
        # Plus half bar width to center the bar at its position
        
        # Formula: x + (i - n_models/2 + 0.5) * bar_width
        offset = (i - n_models / 2 + 0.5) * bar_width
        
        rects = ax.bar(indices + offset, scores, bar_width, label=model, color=color)
        
        # Add labels to the bars
        for rect in rects:
            height = rect.get_height()
            # Only add label if height is positive to avoid overlapping with axis
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=18, rotation=0)
    
    # Aesthetics
    ax.set_ylabel('AUPRC', fontsize=24)
    title_suffix = "XGBoost" if "XGBoost" in classifier_col else "Neural Network"
    ax.set_title(f'Model Comparison: {title_suffix}', fontsize=28)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(species_labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, 1.0) # Ensure scale is consistent 0-1
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metric_file = os.path.join(current_dir, "metric.md")
    
    if not os.path.exists(metric_file):
        print(f"Error: {metric_file} not found.")
        exit(1)
        
    df = parse_markdown_table(metric_file)
    if df is None:
        print("Failed to parse table.")
        return

    # Filter out unwanted models based on user request
    # PlantCAD: keep only 'mean' (remove 'max')
    # Evo2: remove 'concatenate'
    df = df[~((df['Model'].str.contains('plantcad')) & (df['Model'].str.contains('max')))]
    df = df[~(df['Model'].str.contains('evo2_concatenate'))]

    # Define model ordering
    plantcad_models = [m for m in df['Model'].unique() if 'plantcad' in m]
    evo2_models = [m for m in df['Model'].unique() if 'evo2' in m]
    
    # Sort PlantCAD: small -> medium -> large
    def plantcad_sort_key(m):
        size_order = {'small': 0, 'medium': 1, 'large': 2}
        parts = m.split('_')
        size = parts[1]
        return (0, size_order.get(size, 99))
    plantcad_models.sort(key=plantcad_sort_key)
    
    # Sort Evo2: forward -> reverse -> mean
    def evo2_sort_key(m):
        order = ['forward', 'reverse', 'mean']
        parts = m.split('_')
        strat = parts[1]
        return (1, order.index(strat) if strat in order else 99)
    evo2_models.sort(key=evo2_sort_key)
    
    model_order = plantcad_models + evo2_models
    
    # Define Colors
    color_map = {
        'plantcad2_small_mean': '#1f77b466',
        'plantcad2_medium_mean': '#6baed6',
        'plantcad2_large_mean': '#1f77b4b3',
    }
    
    # Evo2 Grays
    gray_palette = ['#e0e0e0', '#bdbdbd', '#969696', '#636363'] 
    
    evo2_colors = {}
    for i, m in enumerate(evo2_models):
        evo2_colors[m] = gray_palette[i % len(gray_palette)]
        
    full_color_map = {**color_map, **evo2_colors}
    
    # Get ordered color list
    ordered_colors = [full_color_map.get(m, '#333333') for m in model_order]
    
    # Plot for each classifier
    classifiers = ['AUPRC (XGBoost)', 'AUPRC (Neural Network)']
    
    for classifier in classifiers:
        short_name = "XGBoost" if "XGBoost" in classifier else "Neural_Network"
        output_filename = f"benchmark_{short_name}.pdf"
        output_path = os.path.join(current_dir, output_filename)
        
        plot_grouped_benchmark(df, classifier, model_order, ordered_colors, output_path)

if __name__ == "__main__":
    main()
