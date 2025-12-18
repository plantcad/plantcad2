import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

valid_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-leaf-abs/valid.tsv', sep='\t')
test_labels = pd.read_csv('../../../results/PlantCAD2_tasks/exp-leaf-abs/test.tsv', sep='\t')

model_dict = [
    ('pcv2_1', 'pcv2-l24-d0768-checkpoints-lr-1e-4'),
    ('pcv2_2', 'pcv2-l48-d1024-checkpoints-lr-1e-4'),
    ('pcv2_3', 'pcv2-l48-d1536-checkpoints-lr-1e-4'),
    ('agront', 'agront-checkpoints-lr-1e-4'),
    ('sup_pcv2_1', 'sup_pcv2-l24-d0768-checkpoints-lr-1e-4'),
    ('cnn_lstm', 'cnn_lstm'),
]

for model in model_dict:
    curPATH = f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model[1]}/valid_scores.tsv'
    curPreds = pd.read_csv(curPATH, sep='\t')
    valid_labels[f'{model[0]}'] = curPreds['predicted_value']

for model in model_dict:
    curPATH = f'../../../results/PlantCAD2_tasks/exp-leaf-abs/{model[1]}/predictions.csv'
    curPreds = pd.read_csv(curPATH, sep='\t')
    test_labels[f'{model[0]}'] = curPreds['predicted_value']

valid_labels.head()
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
    
    mse_valid = np.mean((valid_preds - valid_true) ** 2)
    mse_test = np.mean((test_preds - test_true) ** 2)

    mae_valid = np.mean(np.abs(valid_preds - valid_true))
    mae_test = np.mean(np.abs(test_preds - test_true))

    # R square
    r2_valid = r2_score(valid_true, valid_preds)
    r2_test = r2_score(test_true, test_preds)
    
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
        'mse': mse_valid,
        'mae': mae_valid,
        'r2': r2_valid,
        'color': color,
        'alpha': alpha    })
    records.append({
        'model': model[0],
        'label': 'test',
        'scc': scc_test,
        'pcc': pcc_test,
        'mse': mse_test,
        'mae': mae_test,
        'r2': r2_test,
        'color': color,
        'alpha': alpha    })

df = pd.DataFrame(records)
df.to_csv('exp-leaf-abs.tsv', sep='\t', index=False)
