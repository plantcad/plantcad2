import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

valid_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-max/valid.tsv', sep='\t')
test_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-max/test.tsv', sep='\t')

model_dict = [
    ('pcv2_1', 'pcv2-l24-d0768-checkpoints-lr-1e-4'),
    ('pcv2_2', 'pcv2-l48-d1024-checkpoints-lr-1e-4'),
    ('pcv2_3', 'pcv2-l48-d1536-checkpoints-lr-1e-4'),
    ('agront', 'agront-checkpoints-lr-1e-4'),
    ('sup_pcv2_1', 'sup_pcv2-l24-d0768-checkpoints-lr-1e-4'),
    ('cnn_lstm', 'cnn_lstm'),
]

for model in model_dict:
    curPATH = f'../../../results/PlantCAD2_tasks/exp-max/{model[1]}/valid_predictions.tsv'
    curPreds = pd.read_csv(curPATH, sep='\t')
    valid_labels[f'{model[0]}'] = curPreds['predicted_value']


for model in model_dict:
    curPATH = f'../../../results/PlantCAD2_tasks/exp-max/{model[1]}/test_predictions.tsv'
    curPreds = pd.read_csv(curPATH, sep='\t')
    test_labels[f'{model[0]}'] = curPreds['predicted_value']

records = []
pcv2_alphas = [0.4, 0.7, 1.0]
for k, model in enumerate(model_dict):
    valid_preds = valid_labels[f'{model[0]}']
    test_preds = test_labels[f'{model[0]}']

    valid_true = valid_labels['Label']
    test_true = test_labels['Label']
    
    scc_valid = valid_labels['Label'].corr(valid_labels[f'{model[0]}'], method='spearman')
    scc_test = test_labels['Label'].corr(test_labels[f'{model[0]}'], method='spearman')

    pcc_valid = valid_labels['Label'].corr(valid_labels[f'{model[0]}'], method='pearson')
    pcc_test = test_labels['Label'].corr(test_labels[f'{model[0]}'], method='pearson')
    

    if 'agront' in model[0].lower():
        color, alpha = 'gray', 1.0
    elif 'sup_pcv2' in model[0].lower():
        color, alpha = 'C1', 1.0
    elif 'cnn' in model[0].lower():
        color, alpha = 'C1', 0.5
    else:
        color, alpha = 'C0', pcv2_alphas[k]

    if 'cnn' in model[0].lower():
        lab = model[0].replace('-imbalance-lr-1e-3', '')
    else:
        lab = model[0].replace('-checkpoints-lr-1e-4', '')

    records.append({
        'model': model[0],
        'label': 'valid',
        'scc': scc_valid,
        'pcc': pcc_valid,
        'color': color,
        'alpha': alpha    })
    records.append({
        'model': model[0],
        'label': 'test',
        'scc': scc_test,
        'pcc': pcc_test,
        'color': color,
        'alpha': alpha    })

df = pd.DataFrame(records)
df.head()

df.to_csv('auc_exp-max.tsv', sep='\t', index=False)

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
        sub['scc'],
        bar_w,
        label=model,
        color=sub['color'].iloc[0] if not sub.empty else 'gray',
        alpha=sub['alpha'].iloc[0] if not sub.empty else 0.5
    )

ax.set_xticks(x + bar_w * (n_models - 1) / 2)
ax.set_xticklabels(species_order, rotation=45, ha='right', fontsize=11, fontweight='bold')

ax.set_ylabel('SCC', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.8)
ax.legend(title='Model', fontsize=10)

plt.tight_layout()
plt.savefig('exp-max-performance.pdf', format='pdf', bbox_inches='tight')

