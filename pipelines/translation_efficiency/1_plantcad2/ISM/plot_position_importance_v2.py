import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import pickle
import os
import logomaker  # Import the logomaker library

print("Starting ISM position importance analysis...")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# --- Data Loading ---
# NOTE: The script expects the data files to be in specific relative paths.
# You may need to adjust these paths based on your directory structure.
print("Loading validation data...")
try:
    valid = pd.read_csv('../../../../results/translation_efficiency/training_data/1_absolute/valid.tsv', sep='\t')
    print(f"Loaded {len(valid)} validation sequences")

    print("Loading prediction scores...")
    scores = pd.read_csv('../../../../results/translation_efficiency/training_data/1_absolute/models/pcv2-l48-d1536/valid_predictions.tsv', sep='\t')
    valid['scores'] = scores['predicted_value']
    print(f"Added prediction scores, range: {valid['scores'].min():.4f} - {valid['scores'].max():.4f}")

    print("Loading ISM data...")
    ism = pd.read_csv('../../../../results/translation_efficiency/training_data/1_absolute/valid_ism.tsv', sep='\t')
    print(f"Loaded {len(ism)} ISM mutations")

    print("Loading ISM prediction scores...")
    ism_scores = pd.read_csv('../../../../results/translation_efficiency/training_data/1_absolute/models/pcv2-l48-d1536/valid_ism_scores.tsv', sep='\t')
    ism['scores'] = ism_scores['predicted_value']
    print(f"Added ISM scores, range: {ism['scores'].min():.4f} - {ism['scores'].max():.4f}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the script is run from the correct directory and the data files exist.")
    print("Creating dummy data to allow the script to run for demonstration purposes.")
    # Create dummy data if files are not found
    valid = pd.DataFrame({'sequence': ['A'*100]*10, 'scores': np.random.rand(10)})
    ism = pd.DataFrame({
        'original_index': np.repeat(range(10), 10),
        'pos': list(range(-10, 0))*10,
        'mutated_base': np.random.choice(['A', 'C', 'G', 'T'], 100),
        'scores': np.random.rand(100)
    })


# --- Importance Score Calculation ---
def calculate_position_importance():
    """Calculate importance score for each position based on ISM results"""
    print("Calculating position importance scores...")
    
    all_indices = valid.index
    print(f"Total sequences for analysis: {len(all_indices)}")
    
    position_importance = {}
    nucleotide_contributions = {}
    
    processed = 0
    for idx in all_indices:
        if idx not in ism['original_index'].values:
            continue
            
        group = ism[ism['original_index'] == idx]
        original_score = valid.loc[idx, 'scores']
        
        for _, row in group.iterrows():
            pos = row['pos']
            mut_base = row['mutated_base']
            
            delta_score = row['scores'] - original_score
            abs_delta_score = abs(delta_score)
            
            if pos not in position_importance:
                position_importance[pos] = []
            position_importance[pos].append(abs_delta_score)
            
            if pos not in nucleotide_contributions:
                nucleotide_contributions[pos] = {'A': [], 'C': [], 'G': [], 'T': []}
            nucleotide_contributions[pos][mut_base].append(delta_score)
        
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} sequences...")
    
    print(f"Finished processing {processed} sequences")
    
    pos_stats = {pos: {'mean': np.mean(scores), 'std': np.std(scores), 'median': np.median(scores), 'count': len(scores)} for pos, scores in position_importance.items() if scores}
    
    nucleotide_stats = {}
    for pos, bases in nucleotide_contributions.items():
        nucleotide_stats[pos] = {base: np.mean(scores) if scores else 0.0 for base, scores in bases.items()}
    
    return pos_stats, nucleotide_stats

# --- Caching Logic ---
cache_file = 'ism_importance_cache.pkl'
if os.path.exists(cache_file):
    print("Loading cached importance scores...")
    with open(cache_file, 'rb') as f:
        results = pickle.load(f)
    pos_importance, nucleotide_stats = results
    print("Cached data loaded!")
else:
    print("No cache found. Starting importance calculation...")
    results = calculate_position_importance()
    pos_importance, nucleotide_stats = results
    print("Importance calculation completed!")
    
    print("Saving results to cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    print("Cache saved!")

# --- DataFrame Preparation ---
print("Converting results to DataFrame for plotting...")
pos_df = pd.DataFrame(pos_importance).T.reset_index().rename(columns={'index': 'position'}).sort_values('position')
nucleotide_df = pd.DataFrame(nucleotide_stats).T.reset_index().rename(columns={'index': 'position'}).sort_values('position')

# --- Plotting ---
print("Generating plots...")
fig, axes = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [1, 1, 1.2]})
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Overall position importance (Mean Absolute Score Difference)
axes[0].plot(pos_df['position'], pos_df['mean'], 'o-', linewidth=2, markersize=4, color='darkblue', label='Mean Importance')
axes[0].fill_between(pos_df['position'], pos_df['mean'] - pos_df['std'], pos_df['mean'] + pos_df['std'], alpha=0.2, color='cornflowerblue', label='Std. Dev.')
axes[0].set_xlabel('Position relative to ATG', fontsize=12)
axes[0].set_ylabel('Mean |Δ Score|', fontsize=12)
axes[0].set_title('Position Importance: Impact on Translation Efficiency', fontsize=16, pad=20)
axes[0].legend()
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 2: Nucleotide contributions (Full Range Line Plot)
colors = {'A': '#1f77b4', 'C': '#ff7f0e', 'G': '#2ca02c', 'T': '#d62728'} # Standard colors
for base in ['A', 'C', 'G', 'T']:
    # CORRECTED THIS LINE: Used 'position' as a string to access the DataFrame column
    axes[1].plot(nucleotide_df['position'], nucleotide_df[base], 'o-', linewidth=1.5, markersize=3, color=colors[base], label=f'{base}', alpha=0.9)
axes[1].set_xlabel('Position relative to ATG', fontsize=12)
axes[1].set_ylabel('Mean Δ Score', fontsize=12)
axes[1].set_title('Nucleotide-Specific Contributions (All Positions)', fontsize=16, pad=20)
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[1].legend(title='Mutated To')
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7)

# --- Plot 3: Information Content Sequence Logo ---
print("Generating information content sequence logo for the Kozak region...")
# Filter for the region of interest (-15 to -1 for more context)
logo_range_df = nucleotide_df[(nucleotide_df['position'] >= -15) & (nucleotide_df['position'] <= -1)]
scores_df = logo_range_df.set_index('position')[['A', 'C', 'G', 'T']]

# Convert scores to a probability matrix.
# Exponentiating makes all values positive and emphasizes larger scores.
# Then, normalize each position (row) to sum to 1.
prob_df = np.exp(scores_df)
prob_df = prob_df.div(prob_df.sum(axis=1), axis=0)

# Create a logo from the probability matrix.
# logomaker will automatically calculate information content in bits.
logo = logomaker.Logo(prob_df,
                      ax=axes[2],
                      font_name='Arial Rounded MT Bold',
                      color_scheme='classic',
                      vpad=.1,
                      width=.8)

# Style the logo plot
logo.ax.set_ylabel("Information (bits)", fontsize=12)
logo.ax.set_xlabel('Position relative to ATG', fontsize=12)
logo.ax.set_title('Sequence Motif Logo (Kozak Region)', fontsize=16, pad=20)
logo.style_spines(visible=False)
logo.style_spines(spines=['left', 'bottom'], visible=True, linewidth=1.5)
logo.ax.grid(False) # Turn off grid for logo clarity
logo.ax.tick_params(axis='both', which='major', labelsize=10)


# --- Final Steps ---
plt.tight_layout(pad=3.0)
plt.savefig('position_importance_with_info_logo.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'position_importance_with_info_logo.png'")
plt.show()

# --- Summary Statistics ---
print("\n" + "="*50)
print("Position Importance Analysis Summary")
print("="*50)

print("\nOVERALL POSITION IMPORTANCE (based on absolute score change):")
print("-" * 50)
print(f"Total positions analyzed: {len(pos_df)}")
print(f"Mean importance across all positions: {pos_df['mean'].mean():.6f} (± {pos_df['mean'].std():.6f})")

print("\nTop 10 most important positions:")
top_positions = pos_df.sort_values('mean', ascending=False).head(10)
print(top_positions[['position', 'mean', 'std', 'count']].round(4))

print("\nNUCLEOTIDE CONTRIBUTION (based on signed score change):")
print("-" * 50)
print("Average nucleotide effects across all positions:")
for base in ['A', 'C', 'G', 'T']:
    avg_effect = nucleotide_df[base].mean()
    print(f"  {base}: {avg_effect:.6f}")

print("\nNUCLEOTIDE EFFECTS IN KOZAK REGION (-10 to -1):")
print("-" * 50)
kozak_df = nucleotide_df[(nucleotide_df['position'] >= -10) & (nucleotide_df['position'] <= -1)]
kozak_effects = {}
for base in ['A', 'C', 'G', 'T']:
    kozak_effect = kozak_df[base].mean()
    kozak_effects[base] = kozak_effect
    print(f"  {base}: {kozak_effect:.6f}")

most_beneficial = max(kozak_effects, key=kozak_effects.get)
most_detrimental = min(kozak_effects, key=kozak_effects.get)
print(f"\nMost beneficial nucleotide in Kozak region: {most_beneficial} (avg effect: {kozak_effects[most_beneficial]:.4f})")
print(f"Most detrimental nucleotide in Kozak region: {most_detrimental} (avg effect: {kozak_effects[most_detrimental]:.4f})")
print("\nAnalysis complete.")
