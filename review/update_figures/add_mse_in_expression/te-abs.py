import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

valid_labels = pd.read_csv('../../../results/translation_efficiency/training_data/1_absolute/valid.tsv', sep='\t')
test_labels = pd.read_csv('../../../results/translation_efficiency/training_data/1_absolute/test.tsv', sep='\t')

model_dict = [
    ('pcv2_1', 'pcv2-l24-d0768'),
    ('pcv2_2', 'pcv2-l48-d1024'),
    ('pcv2_3', 'pcv2-l48-d1536'),
    ('agront', 'agront'),
    ('sup_pcv2_1', 'sup_pcv2-l24-d0768'),
    ('cnn_lstm', 'cnn_lstm'),
]

for model in model_dict:
    curPATH = f'../../../results/translation_efficiency/training_data/1_absolute/models/{model[1]}/valid_predictions.tsv'
    curPreds = pd.read_csv(curPATH, sep='\t')
    valid_labels[f'{model[0]}'] = curPreds['predicted_value']

for model in model_dict:
    curPATH = f'../../../results/translation_efficiency/training_data/1_absolute/models/{model[1]}/predictions.tsv'
    curPreds = pd.read_csv(curPATH, sep='\t')
    test_labels[f'{model[0]}'] = curPreds['predicted_value']

records = []
pcv2_alphas = [0.4, 0.7, 1.0]
for k, model in enumerate(model_dict):
    valid_preds = valid_labels[f'{model[0]}']
    test_preds = test_labels[f'{model[0]}']

    valid_true = valid_labels['label']
    test_true = test_labels['label']
    
    scc_valid = valid_labels['label'].corr(valid_labels[f'{model[0]}'], method='spearman')
    scc_test = test_labels['label'].corr(test_labels[f'{model[0]}'], method='spearman')

    pcc_valid = valid_labels['label'].corr(valid_labels[f'{model[0]}'], method='pearson')
    pcc_test = test_labels['label'].corr(test_labels[f'{model[0]}'], method='pearson')

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
        lab = model[0].replace('', '')

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
df.to_csv('te-abs.tsv', sep='\t', index=False)
