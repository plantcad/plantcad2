import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

print("Starting ISM position importance analysis...")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("Loading validation data...")
valid = pd.read_csv('../../../../results/translation_efficiency/training_data/2_bin/valid.tsv', sep='\t')
print(f"Loaded {len(valid)} validation sequences")

print("Loading prediction scores...")
scores = pd.read_csv('../../../../results/translation_efficiency/training_data/2_bin/models/pcv2-l48-d1536/valid_predictions.tsv', sep='\t')
valid['scores'] = scores['probability_positive']
print(f"Added prediction scores, range: {valid['scores'].min():.4f} - {valid['scores'].max():.4f}")

print("Loading ISM data...")
ism = pd.read_csv('../../../../results/translation_efficiency/training_data/2_bin/valid_ism.tsv', sep='\t')
print(f"Loaded {len(ism)} ISM mutations")

print("Loading ISM prediction scores...")
ism_scores = pd.read_csv('../../../../results/translation_efficiency/training_data/2_bin/models/pcv2-l48-d1536/valid_ism_scores.tsv', sep='\t')
ism['scores'] = ism_scores['probability_positive']
print(f"Added ISM scores, range: {ism['scores'].min():.4f} - {ism['scores'].max():.4f}")

# Calculate importance scores for each position and nucleotide, separated by score groups
def calculate_position_importance():
    """Calculate importance score for each position based on ISM results"""
    print("Calculating position importance scores...")
    
    # Separate data by original scores
    positive_indices = valid[valid['scores'] > 0.8].index
    negative_indices = valid[valid['scores'] < 0.7].index
    
    print(f"Positive sequences (score > 0.8): {len(positive_indices)}")
    print(f"Negative sequences (score < 0.7): {len(negative_indices)}")
    
    # Initialize dictionaries for both groups
    results = {}
    for group_name, indices in [('positive', positive_indices), ('negative', negative_indices)]:
        position_importance = {}
        nucleotide_contributions = {}
        
        processed = 0
        for idx in indices:
            if idx not in ism['original_index'].values:
                continue
                
            group = ism[ism['original_index'] == idx]
            original_score = valid.loc[idx, 'scores']
            
            for _, row in group.iterrows():
                pos = row['pos']  # Position relative to ATG (-100 to -1)
                mut_base = row['mutated_base']
                
                # Convert probabilities to log-odds
                original_logodds = np.log(original_score / (1 - original_score + 1e-8))
                mutated_logodds = np.log(row['scores'] / (1 - row['scores'] + 1e-8))
                
                delta_logodds = mutated_logodds - original_logodds  # Signed log-odds difference
                abs_delta_logodds = abs(delta_logodds)  # Absolute log-odds difference
                
                # Overall position importance (absolute log-odds)
                if pos not in position_importance:
                    position_importance[pos] = []
                position_importance[pos].append(abs_delta_logodds)
                
                # Nucleotide-specific contribution (signed log-odds)
                if pos not in nucleotide_contributions:
                    nucleotide_contributions[pos] = {'A': [], 'C': [], 'G': [], 'T': []}
                nucleotide_contributions[pos][mut_base].append(delta_logodds)
            
            processed += 1
            if processed % 500 == 0:
                print(f"Processed {processed} {group_name} sequences...")
        
        print(f"Finished processing {processed} {group_name} sequences")
        
        # Calculate statistics for this group
        pos_stats = {}
        for pos, scores in position_importance.items():
            if scores:  # Only if we have data for this position
                pos_stats[pos] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'count': len(scores)
                }
        
        nucleotide_stats = {}
        for pos in nucleotide_contributions:
            nucleotide_stats[pos] = {}
            for base in ['A', 'C', 'G', 'T']:
                if nucleotide_contributions[pos][base]:
                    nucleotide_stats[pos][base] = np.mean(nucleotide_contributions[pos][base])
                else:
                    nucleotide_stats[pos][base] = 0.0
        
        results[group_name] = (pos_stats, nucleotide_stats)
    
    return results

# Calculate importance with caching
import pickle
import os

cache_file = 'ism_importance_cache.pkl'

if os.path.exists(cache_file):
    print("Loading cached importance scores...")
    try:
        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
        # Check if it's the new format (dictionary) or old format (tuple)
        if isinstance(results, dict) and 'positive' in results:
            print("Cached data loaded!")
        else:
            print("Old cache format detected, recalculating...")
            raise ValueError("Old cache format")
    except:
        print("Cache file incompatible, recalculating...")
        results = calculate_position_importance()
        print("Importance calculation completed!")
        
        print("Saving results to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        print("Cache saved!")
else:
    print("Starting importance calculation...")
    results = calculate_position_importance()
    print("Importance calculation completed!")
    
    print("Saving results to cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    print("Cache saved!")

# Extract results for positive and negative groups
pos_importance_pos, nucleotide_stats_pos = results['positive']
pos_importance_neg, nucleotide_stats_neg = results['negative']

# Convert to DataFrame for plotting
print("Converting results to DataFrame for plotting...")

# Positive group DataFrames
pos_df_pos = pd.DataFrame(pos_importance_pos).T
pos_df_pos.index.name = 'position'
pos_df_pos = pos_df_pos.reset_index().sort_values('position')

nucleotide_df_pos = pd.DataFrame(nucleotide_stats_pos).T
nucleotide_df_pos.index.name = 'position'
nucleotide_df_pos = nucleotide_df_pos.reset_index().sort_values('position')

# Negative group DataFrames
pos_df_neg = pd.DataFrame(pos_importance_neg).T
pos_df_neg.index.name = 'position'
pos_df_neg = pos_df_neg.reset_index().sort_values('position')

nucleotide_df_neg = pd.DataFrame(nucleotide_stats_neg).T
nucleotide_df_neg.index.name = 'position'
nucleotide_df_neg = nucleotide_df_neg.reset_index().sort_values('position')

# Save low efficiency nucleotide contributions DataFrame to TSV
print("Saving low efficiency nucleotide contributions data...")
nucleotide_df_neg.to_csv('nucleotide_contributions_low_efficiency.tsv', sep='\t', index=False)
print("Low efficiency nucleotide contributions saved to: nucleotide_contributions_low_efficiency.tsv")

# Create comparison plots
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

colors = {'A': 'red', 'T': 'blue', 'G': 'green', 'C': 'orange'}

# Row 1: Overall position importance comparison
axes[0, 0].plot(pos_df_pos['position'], pos_df_pos['mean'], 'o-', linewidth=2, markersize=4, color='red', label='High efficiency (>0.8)')
axes[0, 0].set_xlabel('Position relative to ATG')
axes[0, 0].set_ylabel('Mean absolute log-odds difference')
axes[0, 0].set_title('Position Importance: High Efficiency Sequences')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(pos_df_neg['position'], pos_df_neg['mean'], 'o-', linewidth=2, markersize=4, color='blue', label='Low efficiency (<0.7)')
axes[0, 1].set_xlabel('Position relative to ATG')
axes[0, 1].set_ylabel('Mean absolute log-odds difference')
axes[0, 1].set_title('Position Importance: Low Efficiency Sequences')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Row 2: Nucleotide contributions comparison (full range)
for base in ['A', 'T', 'G', 'C']:
    axes[1, 0].plot(nucleotide_df_pos['position'], nucleotide_df_pos[base], 'o-', 
                   linewidth=2, markersize=3, color=colors[base], label=f'{base}', alpha=0.8)
axes[1, 0].set_xlabel('Position relative to ATG')
axes[1, 0].set_ylabel('Mean log-odds difference (signed)')
axes[1, 0].set_title('Nucleotide Contributions: High Efficiency (>0.8)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

for base in ['A', 'T', 'G', 'C']:
    axes[1, 1].plot(nucleotide_df_neg['position'], nucleotide_df_neg[base], 'o-', 
                   linewidth=2, markersize=3, color=colors[base], label=f'{base}', alpha=0.8)
axes[1, 1].set_xlabel('Position relative to ATG')
axes[1, 1].set_ylabel('Mean log-odds difference (signed)')
axes[1, 1].set_title('Nucleotide Contributions: Low Efficiency (<0.7)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Row 3: Nucleotide contributions zoom (-10 to -1)
zoom_pos = nucleotide_df_pos[(nucleotide_df_pos['position'] >= -10) & (nucleotide_df_pos['position'] <= -1)]
zoom_neg = nucleotide_df_neg[(nucleotide_df_neg['position'] >= -10) & (nucleotide_df_neg['position'] <= -1)]

for base in ['A', 'T', 'G', 'C']:
    axes[2, 0].plot(zoom_pos['position'], zoom_pos[base], 'o-', 
                   linewidth=2, markersize=5, color=colors[base], label=f'{base}', alpha=0.8)
axes[2, 0].set_xlabel('Position relative to ATG')
axes[2, 0].set_ylabel('Mean log-odds difference (signed)')
axes[2, 0].set_title('High Efficiency: Zoom -10 to -1 (Near ATG)')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()
axes[2, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2, 0].set_xlim(-10.5, -0.5)

for base in ['A', 'T', 'G', 'C']:
    axes[2, 1].plot(zoom_neg['position'], zoom_neg[base], 'o-', 
                   linewidth=2, markersize=5, color=colors[base], label=f'{base}', alpha=0.8)
axes[2, 1].set_xlabel('Position relative to ATG')
axes[2, 1].set_ylabel('Mean log-odds difference (signed)')
axes[2, 1].set_title('Low Efficiency: Zoom -10 to -1 (Near ATG)')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend()
axes[2, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[2, 1].set_xlim(-10.5, -0.5)

plt.tight_layout()
plt.savefig('position_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Position Importance Analysis Summary:")
print("=" * 50)

print("\nHIGH EFFICIENCY SEQUENCES (score > 0.8):")
print("-" * 40)
print(f"Total positions analyzed: {len(pos_df_pos)}")
print(f"Mean importance across all positions: {pos_df_pos['mean'].mean():.6f}")
print(f"Standard deviation: {pos_df_pos['mean'].std():.6f}")
print("\nTop 10 most important positions:")
top_positions_pos = pos_df_pos.nlargest(10, 'mean')
print(top_positions_pos[['position', 'mean', 'std']].head(10).round(6))

print("\nLOW EFFICIENCY SEQUENCES (score < 0.7):")
print("-" * 40)
print(f"Total positions analyzed: {len(pos_df_neg)}")
print(f"Mean importance across all positions: {pos_df_neg['mean'].mean():.6f}")
print(f"Standard deviation: {pos_df_neg['mean'].std():.6f}")
print("\nTop 10 most important positions:")
top_positions_neg = pos_df_neg.nlargest(10, 'mean')
print(top_positions_neg[['position', 'mean', 'std']].head(10).round(6))

# Compare top positions between groups
print("\nCOMPARISON:")
print("-" * 40)
common_top_positions = set(top_positions_pos['position'].head(5)) & set(top_positions_neg['position'].head(5))
print(f"Common positions in top 5 for both groups: {sorted(common_top_positions)}")

print(f"High efficiency group has higher average importance: {pos_df_pos['mean'].mean() > pos_df_neg['mean'].mean()}")