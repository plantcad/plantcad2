import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

valid_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-leaf-abs/valid.tsv', sep='\t')
test_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-leaf-abs/test.tsv', sep='\t')

# Define seeds for repeated experiments (None = no suffix for seed42)
seeds = [None, 'seed1024', 'seed123', 'seed456', 'seed789']

model_dict = [
    ('pcv2_1', 'pcv2-l24-d0768-checkpoints-lr-1e-4'),
    ('agront', 'agront-checkpoints-lr-1e-4'),
    ('sup_pcv2_1', 'sup_pcv2-l24-d0768-checkpoints-lr-1e-4'),
    ('cnn_lstm', 'cnn_lstm'),
]

# Calculate metrics for each model and seed
records = []
individual_records = []

for k, model in enumerate(model_dict):
    valid_scc_scores = []
    valid_pcc_scores = []
    test_scc_scores = []
    test_pcc_scores = []

    print(f"\nProcessing model: {model[0]}")

    for seed in seeds:
        # Construct paths with seed suffix (None = no suffix for original seed42 run)
        if seed is None:
            model_name_with_seed = model[1]
            seed_label = 'seed42'
        else:
            if 'cnn_lstm' in model[1]:
                model_name_with_seed = f"{model[1]}_{seed}"
            else:   
                model_name_with_seed = f"{model[1]}-{seed}"
            seed_label = seed

        # Load validation predictions
        valid_path = f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/valid_scores.tsv'
        test_path = f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/predictions.csv'

        scc_valid, pcc_valid, scc_test, pcc_test = None, None, None, None

        # Try different validation file names
        valid_files_to_try = [
            f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/valid_scores.tsv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/valid_predictions.tsv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/valid_predictions.csv'
        ]

        for valid_path in valid_files_to_try:
            if os.path.exists(valid_path):
                valid_preds = pd.read_csv(valid_path, sep='\t')
                # Auto-detect prediction column (exclude common non-prediction columns)
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in valid_preds.columns if col not in exclude_cols][0]

                scc_valid = valid_labels['Label'].corr(valid_preds[pred_col], method='spearman')
                pcc_valid = valid_labels['Label'].corr(valid_preds[pred_col], method='pearson')
                valid_scc_scores.append(scc_valid)
                valid_pcc_scores.append(pcc_valid)
                break

        # Try different test file names
        test_files_to_try = [
            f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/predictions.csv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/test_predictions.tsv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_name_with_seed}/test_predictions.csv'
        ]

        for test_path in test_files_to_try:
            if os.path.exists(test_path):
                test_preds = pd.read_csv(test_path, sep='\t')
                # Auto-detect prediction column (exclude common non-prediction columns)
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in test_preds.columns if col not in exclude_cols][0]

                scc_test = test_labels['Label'].corr(test_preds[pred_col], method='spearman')
                pcc_test = test_labels['Label'].corr(test_preds[pred_col], method='pearson')
                test_scc_scores.append(scc_test)
                test_pcc_scores.append(pcc_test)
                break

        # Store individual results
        individual_records.append({
            'model': model[0],
            'seed': seed_label,
            'valid_scc': scc_valid,
            'valid_pcc': pcc_valid,
            'test_scc': scc_test,
            'test_pcc': pcc_test
        })

        valid_str = f"{scc_valid:.4f}" if scc_valid is not None else "N/A"
        test_str = f"{scc_test:.4f}" if scc_test is not None else "N/A"
        print(f"  {seed_label}: Valid SCC={valid_str}, Test SCC={test_str}")

    # Calculate mean and std
    scc_valid_mean = np.mean(valid_scc_scores) if valid_scc_scores else 0
    scc_valid_std = np.std(valid_scc_scores, ddof=1) if len(valid_scc_scores) > 1 else 0
    pcc_valid_mean = np.mean(valid_pcc_scores) if valid_pcc_scores else 0
    pcc_valid_std = np.std(valid_pcc_scores, ddof=1) if len(valid_pcc_scores) > 1 else 0

    scc_test_mean = np.mean(test_scc_scores) if test_scc_scores else 0
    scc_test_std = np.std(test_scc_scores, ddof=1) if len(test_scc_scores) > 1 else 0
    pcc_test_mean = np.mean(test_pcc_scores) if test_pcc_scores else 0
    pcc_test_std = np.std(test_pcc_scores, ddof=1) if len(test_pcc_scores) > 1 else 0

    print(f"  Summary: Valid SCC = {scc_valid_mean:.4f} ± {scc_valid_std:.4f}, Test SCC = {scc_test_mean:.4f} ± {scc_test_std:.4f}")

    # Determine color and alpha
    if 'agront' in model[0].lower():
        color, alpha = 'gray', 1.0
    elif 'sup_pcv2' in model[0].lower():
        color, alpha = 'C1', 1.0
    elif 'cnn' in model[0].lower():
        color, alpha = 'C1', 0.5
    else:
        color, alpha = 'C0', 1.0  # pcv2 models use blue with full opacity

    records.append({
        'model': model[0],
        'label': 'valid',
        'scc_mean': scc_valid_mean,
        'scc_std': scc_valid_std,
        'pcc_mean': pcc_valid_mean,
        'pcc_std': pcc_valid_std,
        'color': color,
        'alpha': alpha
    })
    records.append({
        'model': model[0],
        'label': 'test',
        'scc_mean': scc_test_mean,
        'scc_std': scc_test_std,
        'pcc_mean': pcc_test_mean,
        'pcc_std': pcc_test_std,
        'color': color,
        'alpha': alpha
    })
    
df = pd.DataFrame(records)
individual_df = pd.DataFrame(individual_records)

# Save detailed metrics table
df.to_csv('exp-leaf-abs.tsv', sep='\t', index=False)

# Save individual seed results
individual_df.to_csv('individual_seeds_exp-leaf-abs.tsv', sep='\t', index=False)

# Create a formatted metrics table with mean ± std
metric_table = []
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    valid_data = model_data[model_data['label'] == 'valid'].iloc[0]
    test_data = model_data[model_data['label'] == 'test'].iloc[0]

    metric_table.append({
        'Model': model,
        'Valid SCC': f"{valid_data['scc_mean']:.3f} ± {valid_data['scc_std']:.3f}",
        'Valid PCC': f"{valid_data['pcc_mean']:.3f} ± {valid_data['pcc_std']:.3f}",
        'Test SCC': f"{test_data['scc_mean']:.3f} ± {test_data['scc_std']:.3f}",
        'Test PCC': f"{test_data['pcc_mean']:.3f} ± {test_data['pcc_std']:.3f}",
    })

metric_table_df = pd.DataFrame(metric_table)
metric_table_df.to_csv('metrics_table_exp-leaf-abs.tsv', sep='\t', index=False)
print("Metrics Table:")
print(metric_table_df.to_string(index=False))

# Plot with error bars
models = df['model'].unique()
n_models = len(models)

species_order = ['valid', 'test']
n_sp = len(species_order)

bar_w = 0.8 / n_models
x = np.arange(n_sp)

fig, ax = plt.subplots(figsize=(5 + 0.5 * n_sp, 6))

for i, model in enumerate(models):
    sub = df[df['model'] == model]
    sub = sub.set_index('label').reindex(species_order).reset_index()

    ax.bar(
        x + i * bar_w,
        sub['scc_mean'],
        bar_w,
        yerr=sub['scc_std'],
        label=model,
        color=sub['color'].iloc[0] if not sub.empty else 'gray',
        alpha=sub['alpha'].iloc[0] if not sub.empty else 0.5,
        capsize=5,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5}
    )

ax.set_xticks(x + bar_w * (n_models - 1) / 2)
ax.set_xticklabels(species_order, rotation=45, ha='right', fontsize=11, fontweight='bold')

ax.set_ylabel('SCC', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.8)
ax.legend(title='Model', fontsize=10)

plt.tight_layout()
plt.savefig('exp-leaf-abs.pdf', format='pdf', bbox_inches='tight')
print("\n" + "="*60)
print("Output files:")
print("  Plot: exp-leaf-abs.pdf")
print("  Summary metrics: exp-leaf-abs.tsv")
print("  Formatted table: metrics_table_exp-leaf-abs.tsv")
print("  Individual seeds: individual_seeds_exp-leaf-abs.tsv")
print("="*60)