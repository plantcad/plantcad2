import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve
import matplotlib.pyplot as plt
import os

# Define seeds for repeated experiments (None = no suffix for seed42)
seeds = [None, 'seed1024', 'seed123', 'seed456', 'seed789']

# Store results for all contexts and seeds
records = []
individual_records = []

for context in ['2k', '4k', '8k']:
    print(f"\nProcessing context: {context}")

    valid_auc_scores = []
    valid_ap_scores = []
    test_auc_scores = []
    test_ap_scores = []

    for seed in seeds:
        seed_label = 'seed42' if seed is None else seed

        # Construct model path with seed
        if context == '2k':
            base_model = 'pcv2-l24-d0768-checkpoints-lr-1e-4'
        else:
            base_model = f'pcv2-l24-d0768-{context}-checkpoints-lr-1e-4'

        if seed is None:
            model_path = base_model
        else:
            model_path = f"{base_model}-{seed}"

        # Test set
        if context == '2k':
            test_label_path = '../../../results/PlantCAD2_tasks/exp-leaf-bin/test.tsv'
            test_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/predictions.csv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/test_predictions.tsv'
            ]
        else:
            test_label_path = f'../../../results/PlantCAD2_tasks/exp-leaf-bin/test_{context}.tsv'
            test_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/test_{context}_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/test_{context}_predictions.csv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/test_predictions.tsv'
            ]

        # Validation set
        if context == '2k':
            valid_label_path = '../../../results/PlantCAD2_tasks/exp-leaf-bin/valid.tsv'
            valid_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/valid_scores.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/valid_predictions.tsv'
            ]
        else:
            valid_label_path = f'../../../results/PlantCAD2_tasks/exp-leaf-bin/valid_{context}.tsv'
            valid_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/valid_{context}_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/valid_{context}_predictions.csv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-bin/{model_path}/valid_predictions.tsv'
            ]

        auc_test, ap_test, auc_valid, ap_valid = None, None, None, None

        # Load test predictions
        test_labels = pd.read_csv(test_label_path, sep='\t')
        test_found = False
        for test_pred_path in test_pred_files:
            if os.path.exists(test_pred_path):
                test_preds = pd.read_csv(test_pred_path, sep='\t')
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in test_preds.columns if col not in exclude_cols][0]
                fpr, tpr, thresholds = roc_curve(test_labels['Label'], test_preds[pred_col])
                auc_test = auc(fpr, tpr)
                ap_test = average_precision_score(test_labels['Label'], test_preds[pred_col])
                test_auc_scores.append(auc_test)
                test_ap_scores.append(ap_test)
                test_found = True
                break

        if not test_found:
            raise FileNotFoundError(f"Test predictions not found for context {context}, seed {seed_label}. Tried:\n" +
                                    "\n".join(f"  - {f}" for f in test_pred_files))

        # Load validation predictions
        valid_labels = pd.read_csv(valid_label_path, sep='\t')
        valid_found = False
        for valid_pred_path in valid_pred_files:
            if os.path.exists(valid_pred_path):
                valid_preds = pd.read_csv(valid_pred_path, sep='\t')
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in valid_preds.columns if col not in exclude_cols][0]
                fpr, tpr, thresholds = roc_curve(valid_labels['Label'], valid_preds[pred_col])
                auc_valid = auc(fpr, tpr)
                ap_valid = average_precision_score(valid_labels['Label'], valid_preds[pred_col])
                valid_auc_scores.append(auc_valid)
                valid_ap_scores.append(ap_valid)
                valid_found = True
                break

        if not valid_found:
            raise FileNotFoundError(f"Validation predictions not found for context {context}, seed {seed_label}. Tried:\n" +
                                    "\n".join(f"  - {f}" for f in valid_pred_files))

        # Store individual results
        individual_records.append({
            'context': context,
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

    records.append({
        'context': context,
        'split': 'valid',
        'auc_mean': auc_valid_mean,
        'auc_std': auc_valid_std,
        'ap_mean': ap_valid_mean,
        'ap_std': ap_valid_std
    })
    records.append({
        'context': context,
        'split': 'test',
        'auc_mean': auc_test_mean,
        'auc_std': auc_test_std,
        'ap_mean': ap_test_mean,
        'ap_std': ap_test_std
    })

# Save results
df = pd.DataFrame(records)
individual_df = pd.DataFrame(individual_records)

df.to_csv('exp-leaf-bin-long-ctx.tsv', sep='\t', index=False)
individual_df.to_csv('individual_seeds_exp-leaf-bin-long-ctx.tsv', sep='\t', index=False)

# Prepare data for plotting
context_labels = []
auroc_valid = []
auroc_valid_err = []
auroc_test = []
auroc_test_err = []

for context in ['2k', '4k', '8k']:
    context_labels.append(context)
    valid_data = df[(df['context'] == context) & (df['split'] == 'valid')].iloc[0]
    test_data = df[(df['context'] == context) & (df['split'] == 'test')].iloc[0]

    auroc_valid.append(valid_data['auc_mean'])
    auroc_valid_err.append(valid_data['auc_std'])
    auroc_test.append(test_data['auc_mean'])
    auroc_test_err.append(test_data['auc_std'])

# Plot AUROC with error bars
plt.figure(figsize=(6, 5))
plt.errorbar(context_labels, auroc_valid, yerr=auroc_valid_err, marker='o', color='orange',
             linewidth=2, markersize=8, label='valid', capsize=5, capthick=2)
plt.errorbar(context_labels, auroc_test, yerr=auroc_test_err, marker='o', color='green',
             linewidth=2, markersize=8, label='test', capsize=5, capthick=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Context window length', fontsize=12)
plt.ylabel('AUROC', fontsize=12)
plt.legend(frameon=False)
plt.title('Model Performance vs Context Window Length')
plt.tight_layout()
plt.savefig('performance_vs_context_window_exp-leaf-bin.pdf', format='pdf', bbox_inches='tight')

print("\n" + "="*60)
print("Output files:")
print("  Plot: performance_vs_context_window_exp-leaf-bin.pdf")
print("  Summary metrics: exp-leaf-bin-long-ctx.tsv")
print("  Individual seeds: individual_seeds_exp-leaf-bin-long-ctx.tsv")
print("="*60)
