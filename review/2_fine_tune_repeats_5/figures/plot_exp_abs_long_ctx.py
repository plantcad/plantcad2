import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define seeds for repeated experiments (None = no suffix for seed42)
seeds = [None, 'seed1024', 'seed123', 'seed456', 'seed789']

# Store results for all contexts and seeds
records = []
individual_records = []

for context in ['2k', '4k', '8k']:
    print(f"\nProcessing context: {context}")

    valid_scc_scores = []
    valid_pcc_scores = []
    test_scc_scores = []
    test_pcc_scores = []

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
            test_label_path = '../../../results/PlantCAD2_tasks/exp-leaf-abs/test.tsv'
            test_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/predictions.csv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/test_{context}_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/test_predictions.tsv'
            ]
        else:
            test_label_path = f'../../../results/PlantCAD2_tasks/exp-leaf-abs/test_{context}.tsv'
            test_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/test_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/test_{context}_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/test_predictions.tsv'
            ]

        # Validation set
        if context == '2k':
            valid_label_path = '../../../results/PlantCAD2_tasks/exp-leaf-abs/valid.tsv'
            valid_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/valid_scores.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/valid_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/valid_{context}_predictions.tsv'
            ]
        else:
            valid_label_path = f'../../../results/PlantCAD2_tasks/exp-leaf-abs/valid_{context}.tsv'
            valid_pred_files = [
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/valid_predictions.tsv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/valid_predictions.csv',
                f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model_path}/valid_{context}_predictions.tsv'
            ]

        scc_test, pcc_test, scc_valid, pcc_valid = None, None, None, None

        # Load test predictions
        test_labels = pd.read_csv(test_label_path, sep='\t')
        test_found = False
        for test_pred_path in test_pred_files:
            if os.path.exists(test_pred_path):
                test_preds = pd.read_csv(test_pred_path, sep='\t')
                exclude_cols = ['Label', 'ID', 'id', 'index', 'Unnamed: 0', 'sample', 'Sample']
                pred_col = [col for col in test_preds.columns if col not in exclude_cols][0]
                scc_test = test_labels['Label'].corr(test_preds[pred_col], method='spearman')
                pcc_test = test_labels['Label'].corr(test_preds[pred_col], method='pearson')
                test_scc_scores.append(scc_test)
                test_pcc_scores.append(pcc_test)
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
                scc_valid = valid_labels['Label'].corr(valid_preds[pred_col], method='spearman')
                pcc_valid = valid_labels['Label'].corr(valid_preds[pred_col], method='pearson')
                valid_scc_scores.append(scc_valid)
                valid_pcc_scores.append(pcc_valid)
                valid_found = True
                break

        if not valid_found:
            raise FileNotFoundError(f"Validation predictions not found for context {context}, seed {seed_label}. Tried:\n" +
                                    "\n".join(f"  - {f}" for f in valid_pred_files))

        # Store individual results
        individual_records.append({
            'context': context,
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

    records.append({
        'context': context,
        'split': 'valid',
        'scc_mean': scc_valid_mean,
        'scc_std': scc_valid_std,
        'pcc_mean': pcc_valid_mean,
        'pcc_std': pcc_valid_std
    })
    records.append({
        'context': context,
        'split': 'test',
        'scc_mean': scc_test_mean,
        'scc_std': scc_test_std,
        'pcc_mean': pcc_test_mean,
        'pcc_std': pcc_test_std
    })

# Save results
df = pd.DataFrame(records)
individual_df = pd.DataFrame(individual_records)

df.to_csv('exp-leaf-abs-long-ctx.tsv', sep='\t', index=False)
individual_df.to_csv('individual_seeds_exp-leaf-abs-long-ctx.tsv', sep='\t', index=False)

# Prepare data for plotting
context_labels = []
scc_valid = []
scc_valid_err = []
scc_test = []
scc_test_err = []

for context in ['2k', '4k', '8k']:
    context_labels.append(context)
    valid_data = df[(df['context'] == context) & (df['split'] == 'valid')].iloc[0]
    test_data = df[(df['context'] == context) & (df['split'] == 'test')].iloc[0]

    scc_valid.append(valid_data['scc_mean'])
    scc_valid_err.append(valid_data['scc_std'])
    scc_test.append(test_data['scc_mean'])
    scc_test_err.append(test_data['scc_std'])


# Plot SCC with error bars
plt.figure(figsize=(6, 5))
plt.errorbar(context_labels, scc_valid, yerr=scc_valid_err, marker='o', color='orange',
             linewidth=2, markersize=8, label='valid', capsize=5, capthick=2)
plt.errorbar(context_labels, scc_test, yerr=scc_test_err, marker='o', color='green',
             linewidth=2, markersize=8, label='test', capsize=5, capthick=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Context window length', fontsize=12)
plt.ylabel('SCC', fontsize=12)
plt.legend(frameon=False)
plt.title('Model Performance vs Context Window Length')
plt.tight_layout()
plt.savefig('performance_vs_context_window_exp-leaf-abs.pdf', format='pdf', bbox_inches='tight')

print("\n" + "="*60)
print("Output files:")
print("  Plot: performance_vs_context_window_exp-leaf-abs.pdf")
print("  Summary metrics: exp-leaf-abs-long-ctx.tsv")
print("  Individual seeds: individual_seeds_exp-leaf-abs-long-ctx.tsv")
print("="*60)