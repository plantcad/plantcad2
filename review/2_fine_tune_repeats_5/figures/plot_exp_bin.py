import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve
import matplotlib.pyplot as plt
import os

valid_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-leaf-bin/valid.tsv', sep='\t')
test_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-leaf-bin/test.tsv', sep='\t')

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
    valid_auc_scores = []
    valid_ap_scores = []
    test_auc_scores = []
    test_ap_scores = []

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

        auc_valid, ap_valid, auc_test, ap_test = None, None, None, None

        # Try different validation file names
        valid_files_to_try = [
            f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_name_with_seed}/valid_scores.tsv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_name_with_seed}/valid_predictions.tsv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_name_with_seed}/valid_predictions.csv'
        ]

        for valid_path in valid_files_to_try:
            if os.path.exists(valid_path):
                valid_preds = pd.read_csv(valid_path, sep='\t')
                # Auto-detect prediction column (exclude common non-prediction columns)
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in valid_preds.columns if col not in exclude_cols][0]

                fpr, tpr, thresholds = roc_curve(valid_labels['Label'], valid_preds[pred_col])
                auc_valid = auc(fpr, tpr)
                ap_valid = average_precision_score(valid_labels['Label'], valid_preds[pred_col])
                valid_auc_scores.append(auc_valid)
                valid_ap_scores.append(ap_valid)
                break

        # Try different test file names
        test_files_to_try = [
            f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_name_with_seed}/predictions.csv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_name_with_seed}/test_predictions.tsv',
            f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_name_with_seed}/test_predictions.csv'
        ]

        for test_path in test_files_to_try:
            if os.path.exists(test_path):
                test_preds = pd.read_csv(test_path, sep='\t')
                # Auto-detect prediction column (exclude common non-prediction columns)
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in test_preds.columns if col not in exclude_cols][0]

                fpr, tpr, thresholds = roc_curve(test_labels['Label'], test_preds[pred_col])
                auc_test = auc(fpr, tpr)
                ap_test = average_precision_score(test_labels['Label'], test_preds[pred_col])
                test_auc_scores.append(auc_test)
                test_ap_scores.append(ap_test)
                break

        # Store individual results
        individual_records.append({
            'model': model[0],
            'seed': seed_label,
            'valid_auc': auc_valid,
            'valid_ap': ap_valid,
            'test_auc': auc_test,
            'test_ap': ap_test
        })

        valid_str = f"{auc_valid:.4f}" if auc_valid is not None else "N/A"
        test_str = f"{auc_test:.4f}" if auc_test is not None else "N/A"
        print(f"  {seed_label}: Valid AUROC={valid_str}, Test AUROC={test_str}")

    # Calculate mean and std
    auc_valid_mean = np.mean(valid_auc_scores) if valid_auc_scores else 0
    auc_valid_std = np.std(valid_auc_scores, ddof=1) if len(valid_auc_scores) > 1 else 0
    ap_valid_mean = np.mean(valid_ap_scores) if valid_ap_scores else 0
    ap_valid_std = np.std(valid_ap_scores, ddof=1) if len(valid_ap_scores) > 1 else 0

    auc_test_mean = np.mean(test_auc_scores) if test_auc_scores else 0
    auc_test_std = np.std(test_auc_scores, ddof=1) if len(test_auc_scores) > 1 else 0
    ap_test_mean = np.mean(test_ap_scores) if test_ap_scores else 0
    ap_test_std = np.std(test_ap_scores, ddof=1) if len(test_ap_scores) > 1 else 0

    print(f"  Summary: Valid AUROC = {auc_valid_mean:.4f} ± {auc_valid_std:.4f}, Test AUROC = {auc_test_mean:.4f} ± {auc_test_std:.4f}")

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
        'auc_mean': auc_valid_mean,
        'auc_std': auc_valid_std,
        'ap_mean': ap_valid_mean,
        'ap_std': ap_valid_std,
        'color': color,
        'alpha': alpha,
        'baseline': valid_labels['Label'].mean()
    })
    records.append({
        'model': model[0],
        'label': 'test',
        'auc_mean': auc_test_mean,
        'auc_std': auc_test_std,
        'ap_mean': ap_test_mean,
        'ap_std': ap_test_std,
        'color': color,
        'alpha': alpha,
        'baseline': test_labels['Label'].mean()
    })


df = pd.DataFrame(records)
individual_df = pd.DataFrame(individual_records)

# Save detailed metrics table
df.to_csv('exp-leaf-bin.tsv', sep='\t', index=False)

# Save individual seed results
individual_df.to_csv('individual_seeds_exp-leaf-bin.tsv', sep='\t', index=False)

# Create a formatted metrics table with mean ± std
metric_table = []
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    valid_data = model_data[model_data['label'] == 'valid'].iloc[0]
    test_data = model_data[model_data['label'] == 'test'].iloc[0]

    metric_table.append({
        'Model': model,
        'Valid AUROC': f"{valid_data['auc_mean']:.3f} ± {valid_data['auc_std']:.3f}",
        'Valid AP': f"{valid_data['ap_mean']:.3f} ± {valid_data['ap_std']:.3f}",
        'Test AUROC': f"{test_data['auc_mean']:.3f} ± {test_data['auc_std']:.3f}",
        'Test AP': f"{test_data['ap_mean']:.3f} ± {test_data['ap_std']:.3f}",
    })

metric_table_df = pd.DataFrame(metric_table)
metric_table_df.to_csv('metrics_table_exp-leaf-bin.tsv', sep='\t', index=False)
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
        sub['auc_mean'],
        bar_w,
        yerr=sub['auc_std'],
        label=model,
        color=sub['color'].iloc[0] if not sub.empty else 'gray',
        alpha=sub['alpha'].iloc[0] if not sub.empty else 0.5,
        capsize=5,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5}
    )

ax.set_xticks(x + bar_w * (n_models - 1) / 2)
ax.set_xticklabels(species_order, rotation=45, ha='right', fontsize=11, fontweight='bold')

ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(title='Model', fontsize=10)

plt.tight_layout()
plt.savefig('exp-leaf-bin.pdf', format='pdf', bbox_inches='tight')
print("\n" + "="*60)
print("Output files:")
print("  Plot: exp-leaf-bin.pdf")
print("  Summary metrics: exp-leaf-bin.tsv")
print("  Formatted table: metrics_table_exp-leaf-bin.tsv")
print("  Individual seeds: individual_seeds_exp-leaf-bin.tsv")
print("="*60)