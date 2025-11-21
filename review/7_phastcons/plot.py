import pandas as pd
import numpy as np
import pyranges as pr
from scipy.stats import spearmanr
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

models = ['pcv2_large', 'pcv2_medium', 'pcv2_small', 'evo2', 'gpn', 'pcv1']

res_with_scores = pd.read_csv('/workdir/jz963/utils/plantcad2/results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_scores_with_phastcons.tsv', sep='\t')

results = {}
relevant_mask = (res_with_scores['meanPhyloP'] > 1) | (res_with_scores['meanPhyloP'] < 0)
filtered_true_labels = np.where(res_with_scores['mean_phastCons'][relevant_mask] > 0.5, 1, 0)

for model in models:
   filtered_scores = res_with_scores[model][relevant_mask] * -1
   auc_value = roc_auc_score(filtered_true_labels, filtered_scores)
   ap_value = average_precision_score(filtered_true_labels, filtered_scores)
   results[model] = {'AUC': auc_value, 'AP': ap_value}

results_df = pd.DataFrame(results).T
results_df

custom_order = ['pcv1', 'pcv2_small', 'pcv2_medium', 'pcv2_large', 'evo2', 'gpn']
results_df = results_df.reindex(custom_order)
results_df.to_csv('model_performance_phastcons.tsv', sep='\t')

plt.figure(figsize=(5, 5))
colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff',  '#808080','#999999', '#d3d3d3']

bars = plt.bar(results_df.index, results_df['AP'], color=colors)

plt.xlabel('Models')
plt.ylabel('AUPRC')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.ylim(0.3, 1)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('AUPRC.pdf', format='pdf', bbox_inches='tight')
plt.show()
