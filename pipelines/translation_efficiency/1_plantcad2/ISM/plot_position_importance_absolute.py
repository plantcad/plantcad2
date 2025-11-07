import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

# Try to import logomaker for motif logo visualization
try:
    import logomaker
    LOGOMAKER_AVAILABLE = True
except ImportError:
    LOGOMAKER_AVAILABLE = False
    print("Warning: logomaker not installed. Install with: pip install logomaker")
    print("Motif logo will be replaced with standard plot.")

print("Starting ISM position importance analysis...")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("Loading validation data...")
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

# Calculate importance scores for each position and nucleotide for regression model
def calculate_position_importance():
    """Calculate importance score for each position based on ISM results"""
    print("Calculating position importance scores...")
    
    # Use all available sequences for regression analysis
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
            pos = row['pos']  # Position relative to ATG (-100 to -1)
            mut_base = row['mutated_base']
            
            # For regression, use direct score difference
            delta_score = row['scores'] - original_score  # Signed score difference
            abs_delta_score = abs(delta_score)  # Absolute score difference
            
            # Apply transformation to amplify differences
            # Option 1: Square the differences to amplify larger effects
            # abs_delta_score = abs_delta_score ** 2
            # delta_score = np.sign(delta_score) * (abs(delta_score) ** 2)
            
            # Option 2: Log transformation (add small constant to avoid log(0))
            # abs_delta_score = np.log(abs_delta_score + 1e-8)
            # delta_score = np.sign(delta_score) * np.log(abs(delta_score) + 1e-8)
            
            # Option 3: Multiply by a scaling factor
            scaling_factor = 100  # Amplify by 100x
            abs_delta_score = abs_delta_score * scaling_factor
            delta_score = delta_score * scaling_factor
            
            # Overall position importance (absolute score difference)
            if pos not in position_importance:
                position_importance[pos] = []
            position_importance[pos].append(abs_delta_score)
            
            # Nucleotide-specific contribution (signed score difference)
            if pos not in nucleotide_contributions:
                nucleotide_contributions[pos] = {'A': [], 'C': [], 'G': [], 'T': []}
            nucleotide_contributions[pos][mut_base].append(delta_score)
        
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} sequences...")
    
    print(f"Finished processing {processed} sequences")
    
    # Calculate statistics
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
    
    # Apply additional normalization to nucleotide stats for better visualization
    print("Applying z-score normalization to nucleotide contributions...")
    
    # Collect all nucleotide contribution values for z-score normalization
    all_values = []
    for pos in nucleotide_stats:
        for base in ['A', 'C', 'G', 'T']:
            all_values.append(nucleotide_stats[pos][base])
    
    if all_values:
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        
        # Apply z-score normalization
        for pos in nucleotide_stats:
            for base in ['A', 'C', 'G', 'T']:
                if std_val > 0:
                    nucleotide_stats[pos][base] = (nucleotide_stats[pos][base] - mean_val) / std_val
                else:
                    nucleotide_stats[pos][base] = 0.0
    
    return pos_stats, nucleotide_stats

# Calculate importance with caching
import pickle
import os

cache_file = 'ism_importance_cache.pkl'

if os.path.exists(cache_file):
    print("Loading cached importance scores...")
    try:
        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
        # Check if it's the new format (tuple) or old format (dictionary)
        if isinstance(results, tuple) and len(results) == 2:
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

# Extract results
pos_importance, nucleotide_stats = results

# Convert to DataFrame for plotting
print("Converting results to DataFrame for plotting...")

# Position importance DataFrame
pos_df = pd.DataFrame(pos_importance).T
pos_df.index.name = 'position'
pos_df = pos_df.reset_index().sort_values('position')

# Nucleotide contributions DataFrame
nucleotide_df = pd.DataFrame(nucleotide_stats).T
nucleotide_df.index.name = 'position'
nucleotide_df = nucleotide_df.reset_index().sort_values('position')

# Create comprehensive plots
fig, axes = plt.subplots(3, 1, figsize=(14, 16))

colors = {'A': 'red', 'T': 'blue', 'G': 'green', 'C': 'orange'}

# Plot 1: Overall position importance
axes[0].plot(pos_df['position'], pos_df['mean'], 'o-', linewidth=2, markersize=4, color='darkblue')
axes[0].fill_between(pos_df['position'], 
                     pos_df['mean'] - pos_df['std'], 
                     pos_df['mean'] + pos_df['std'], 
                     alpha=0.3, color='darkblue')
axes[0].set_xlabel('Position relative to ATG')
axes[0].set_ylabel('Mean absolute score difference (×100)')
axes[0].set_title('Position Importance: Translation Efficiency (All Sequences)')
axes[0].grid(True, alpha=0.3)

# Save nucleotide contributions DataFrame to TSV
print("Saving nucleotide contributions data...")
nucleotide_df.to_csv('nucleotide_contributions.tsv', sep='\t', index=False)
print("Nucleotide contributions saved to: nucleotide_contributions.tsv")

# Plot 2: Nucleotide contributions (full range)
for base in ['A', 'T', 'G', 'C']:
    axes[1].plot(nucleotide_df['position'], nucleotide_df[base], 'o-', 
                linewidth=2, markersize=3, color=colors[base], label=f'{base}', alpha=0.8)
axes[1].set_xlabel('Position relative to ATG')
axes[1].set_ylabel('Z-score normalized contribution')
axes[1].set_title('Nucleotide Contributions: Translation Efficiency (Z-score normalized)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Plot 3: Motif logo or line plot for Kozak region (-10 to -1)
zoom_df = nucleotide_df[(nucleotide_df['position'] >= -10) & (nucleotide_df['position'] <= -1)]

def create_pwm_from_scores(zoom_df):
    """Convert nucleotide scores to PWM matrix for motif logo"""
    # Create PWM matrix with positions as rows and nucleotides as columns
    pwm_data = []
    positions = []
    
    for _, row in zoom_df.iterrows():
        pos = int(row['position'])
        positions.append(pos)
        
        # Get raw scores for each nucleotide
        scores = [row['A'], row['T'], row['G'], row['C']]
        
        # Enhanced normalization for better motif logo visualization
        # Apply exponential transformation to amplify differences
        exp_scores = [np.exp(s) for s in scores]
        
        # Normalize to sum to 1 for PWM
        total = sum(exp_scores)
        if total > 0:
            normalized_scores = [s / total for s in exp_scores]
        else:
            normalized_scores = [0.25, 0.25, 0.25, 0.25]  # Equal probabilities if all zeros
        
        pwm_data.append(normalized_scores)
    
    # Create DataFrame with proper column names for logomaker
    pwm_df = pd.DataFrame(pwm_data, columns=['A', 'T', 'G', 'C'])
    pwm_df.index = positions
    
    return pwm_df

if LOGOMAKER_AVAILABLE and len(zoom_df) > 0:
    # Create PWM and motif logo
    pwm_df = create_pwm_from_scores(zoom_df)
    
    # Create motif logo
    logo = logomaker.Logo(pwm_df, ax=axes[2], color_scheme='classic')
    axes[2].set_xlabel('Position relative to ATG')
    axes[2].set_ylabel('Normalized nucleotide contribution')
    axes[2].set_title('Motif Logo: Translation Efficiency (Positions -10 to -1)')
    
    # Set x-axis ticks to show actual positions
    axes[2].set_xticks(range(len(pwm_df)))
    axes[2].set_xticklabels(pwm_df.index)
    
else:
    # Fallback to line plot if logomaker not available
    for base in ['A', 'T', 'G', 'C']:
        axes[2].plot(zoom_df['position'], zoom_df[base], 'o-', 
                    linewidth=2, markersize=5, color=colors[base], label=f'{base}', alpha=0.8)
    axes[2].set_xlabel('Position relative to ATG')
    axes[2].set_ylabel('Mean score difference (signed)')
    axes[2].set_title('Nucleotide Contributions: Translation Efficiency (Positions -10 to -1)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlim(-10.5, -0.5)

plt.tight_layout()
plt.savefig('position_importance_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Position Importance Analysis Summary:")
print("=" * 50)

print("\nTRANSLATION EFFICIENCY REGRESSION ANALYSIS:")
print("-" * 40)
print(f"Total positions analyzed: {len(pos_df)}")
print(f"Mean importance across all positions: {pos_df['mean'].mean():.6f}")
print(f"Standard deviation: {pos_df['mean'].std():.6f}")
print(f"Median importance: {pos_df['mean'].median():.6f}")
print(f"Min importance: {pos_df['mean'].min():.6f}")
print(f"Max importance: {pos_df['mean'].max():.6f}")

print("\nTop 10 most important positions:")
top_positions = pos_df.nlargest(10, 'mean')
print(top_positions[['position', 'mean', 'std', 'median', 'count']].round(6))

print("\nNUCLEOTIDE CONTRIBUTION ANALYSIS:")
print("-" * 40)
print("Average nucleotide effects across all positions:")
for base in ['A', 'T', 'G', 'C']:
    avg_effect = nucleotide_df[base].mean()
    print(f"{base}: {avg_effect:.6f}")

print("\nNUCLEOTIDE EFFECTS IN KOZAK REGION (-10 to -1):")
print("-" * 40)
kozak_effects = {}
for base in ['A', 'T', 'G', 'C']:
    kozak_effect = zoom_df[base].mean()
    kozak_effects[base] = kozak_effect
    print(f"{base}: {kozak_effect:.6f}")

most_beneficial = max(kozak_effects, key=kozak_effects.get)
most_detrimental = min(kozak_effects, key=kozak_effects.get)
print(f"\nMost beneficial nucleotide in Kozak region: {most_beneficial} ({kozak_effects[most_beneficial]:.6f})")
print(f"Most detrimental nucleotide in Kozak region: {most_detrimental} ({kozak_effects[most_detrimental]:.6f})")