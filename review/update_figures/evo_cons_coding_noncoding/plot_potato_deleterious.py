import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load main dataframe
df = pd.read_csv('/workdir/jz963/utils/plantcad2/results/review/3_potato_deleterious_mutations/processed_v2/all_downsampled_with_types_detailed.tsv', sep='\t')
REF = df['sequence'].str[4095]

# Base types mapping and vectorization setup
base2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ref_values = REF.values
valid_mask = np.isin(ref_values, list(base2idx.keys()))
rows = np.where(valid_mask)[0]
cols = np.array([base2idx[b] for b in ref_values[valid_mask]])

# Models configuration
# Filename -> Model Name (or Model Name -> Filename)
models_info = [
    ('plantcad', 'plantcad.tsv'),
    ('plantcad2-small', 'plantcad2-small.tsv'),
    ('plantcad2-medium', 'plantcad2-medium.tsv'),
    ('plantcad2-large', 'plantcad2-large.tsv'),
    ('evo2', 'evo2.tsv'),
    ('gpn', 'gpn.tsv')
]
model_order = [m[0] for m in models_info]
logits_dir = '/workdir/jz963/utils/plantcad2/results/review/3_potato_deleterious_mutations/logits_v2'

# Calculate scores for all models
for model_name, filename in models_info:
    print(f"Processing {model_name}...")
    logitPath = os.path.join(logits_dir, filename)
    # Assuming headers exist as per user's snippet
    logits = pd.read_csv(logitPath, sep='\t')
    
    # Extract ACGT columns
    logits_np = logits[['A', 'C', 'G', 'T']].values
    
    # Compute scores
    scores = np.zeros(len(ref_values), dtype=float)
    scores[rows] = logits_np[rows, cols]
    
    df[model_name] = scores

# Separate Coding and Noncoding analysis
types = ['CDS', 'Intron', 'Upstream', 'Downstream', 'intergenic', 'TE']
results = {t: [] for t in types}

for t in types:
    subset = df[df['finalType'] == t]
    y_true = subset['label']
    
    for mdl_name in model_order:
        y_scores = subset[mdl_name]
        
        if len(y_true.unique()) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0
            
        results[t].append({'model': mdl_name, 'roc_auc': roc_auc})

# Plotting
# Use separate y-axis logic (sharey=False)
fig, axes = plt.subplots(3, 2, figsize=(10, 15), sharey=False)
axes = axes.flatten()
colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff', '#808080', '#999999']

for i, t in enumerate(types):
    res_df = pd.DataFrame(results[t])
    # Ensure correct order
    res_df = res_df.set_index('model').reindex(model_order).reset_index()
    
    # Calculate counts
    subset = df[df['finalType'] == t]
    n_pos = (subset['label'] == 1).sum()
    n_neg = (subset['label'] == 0).sum()
    
    bars = axes[i].bar(res_df['model'], res_df['roc_auc'], color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom')
    
    axes[i].set_title(f'{t}\n({n_pos} conserved vs {n_neg} neutral)')
    axes[i].set_xticks(range(len(model_order)))
    axes[i].set_xticklabels(model_order, rotation=90)
    
    # Set dynamic y-limits based on data
    min_val = res_df['roc_auc'].min()
    max_val = res_df['roc_auc'].max() + 0.1
    lower = 0.3
    upper = max_val
    axes[i].set_ylim(lower, upper)
    
    axes[i].grid(True, alpha=0.3, axis='y')
    axes[i].set_ylabel('ROC AUC')

plt.tight_layout()
output_pdf = 'potato_deleterious_performance.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"Saved plot to {output_pdf}")
# plt.show()
