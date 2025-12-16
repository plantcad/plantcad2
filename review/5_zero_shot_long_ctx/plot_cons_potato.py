import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

models = ['pcv2-small', 'pcv2-medium', 'pcv2-large']
ctx_window = ['512', '1024', '2048', '4096']

df = pd.read_csv("../../results/review/3_potato_deleterious_mutations/processed_v2/all_downsampled_with_types.tsv", sep='\t')
REF = df['sequence'].str[4095]


score_cols = []
for ctx in ctx_window:
    for model in models:
        print(f"Processing {model} of context {ctx}......")
        logits_list = []
        for i in range(1, 13):
            chrom = f"chr{i:02d}"
            logitPath = f'../../results/review/5_zero_shot_long_ctx/potato_deleterious_mutations/{model}-c{ctx}-{chrom}.tsv'
            print(f"Processing {logitPath}......")
            logits_list.append(pd.read_csv(logitPath, sep='\t'))
        logits = pd.concat(logits_list, ignore_index=True)
        scores = logits.apply(
            lambda row: row[REF.loc[row.name]] if REF.loc[row.name] in "ATCG" else 0,
            axis=1
        )
        prefix = f'{ctx}_{model}'
        df[prefix] = scores
        score_cols.append(prefix)

models = score_cols
y_true = df['label']
results = []

for mdl in models:
    y_scores = df[mdl]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    parts = mdl.split('_', 1)
    context = parts[0]
    model_name = parts[1]
    results.append({'model': model_name, 'context': context, 'roc_auc': roc_auc})
results_df = pd.DataFrame(results)

# the performance of 8192
# pcv2-small    0.652881	
# pcv2-medium   0.678841	
# pcv2-large    0.684095	
# add 8192
new_rows = [
    {'model': 'pcv2-small', 'context': '8192', 'roc_auc': 0.652881},
    {'model': 'pcv2-medium', 'context': '8192', 'roc_auc': 0.678841},
    {'model': 'pcv2-large', 'context': '8192', 'roc_auc': 0.684095}
]
results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)

# save results_df to csv
results_df.to_csv("conservation_potato_results.tsv", sep='\t', index=False)

plt.figure(figsize=(10, 6))

results_df['context'] = pd.to_numeric(results_df['context'])
models = results_df['model'].unique()

for model in models:
    model_data = results_df[results_df['model'] == model].sort_values('context')
    plt.plot(model_data['context'], model_data['roc_auc'], 
             marker='o', linewidth=2, markersize=6, label=model)

plt.xlabel('Context Window Length')
plt.ylabel('AUROC')
plt.title('Model Performance vs Context Window Size')

plt.xticks(sorted(results_df['context'].unique()))
# plt.ylim(0.6, None)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('conservation_potato.pdf', format='pdf', bbox_inches='tight')
plt.show()