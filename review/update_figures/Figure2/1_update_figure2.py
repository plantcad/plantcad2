import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datasets import load_dataset
import numpy as np
import sys

def plot_performance(models, scores, colors, filename, ylim=(0.3, 0.8), rotation=45, ha='center', xlabel=None):
    plt.figure(figsize=(5, 5))
    
    # Ensure colors match the number of bars
    if len(colors) > len(models):
        colors = colors[:len(models)]
        
    bars = plt.bar(models, scores, color=colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom')

    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel('ROC AUC')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=rotation, ha=ha)
    plt.ylim(ylim)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

def process_figure_2b():
    # for Figure 2B (conservation within andropogoneae)
    print("Processing Figure 2B...")
    repo_id = 'kuleshov-group/cross-species-single-nucleotide-annotation'
    DATA_DIR = '/workdir/jz963/utils/plantcad2/results/evolutionary_constraint'
    data = load_dataset(repo_id, data_files={'valid': 'Evolutionary_constraint/valid.tsv'})
    df = data['valid'].to_pandas()
    REF = df['sequences'].str[255]

    models = ['pcv2_l24_d0768', 'pcv2_l48_d1024', 'pcv2_l48_d1536']
    ctx_window = ['512', '1024', '2048', '4096', '8192']

    for ctx in ctx_window:
        for model in models:
            print(f"Processing {model} of context {ctx}......")
            if ctx in ['4096', '8192']:
                logitPath = f'{DATA_DIR}/2nd/valid_{ctx}_{model}_logits_batchSize_32.tsv'
            else:
                logitPath = f'{DATA_DIR}/valid_{ctx}_{model}_logits.tsv'
            logits = pd.read_csv(logitPath, header=None, sep='\t')
            logits.columns = ['A', 'C', 'G', 'T']
            scores = df.apply(
                lambda row: logits.loc[row.name, REF.loc[row.name]] if REF.loc[row.name] in "ATCG" else 0,
                axis=1
            )
            prefix = f'{ctx}_{model}'
            df[prefix] = scores

    logitPath = f'{DATA_DIR}/valid_pcv1_logits.tsv'
    logits = pd.read_csv(logitPath, header=None, sep='\t')
    logits.columns = ['A', 'C', 'G', 'T']
    scores = df.apply(
        lambda row: logits.loc[row.name, REF.loc[row.name]] if REF.loc[row.name] in "ATCG" else 0,
        axis=1
    )
    df['pcv1'] = scores

    logitPath = f'{DATA_DIR}/valid_gpn_logits.tsv'
    logits = pd.read_csv(logitPath, header=None, sep='\t')
    logits.columns = ['A', 'C', 'G', 'T']
    scores = df.apply(
        lambda row: logits.loc[row.name, REF.loc[row.name]] if REF.loc[row.name] in "ATCG" else 0,
        axis=1
    )
    df['gpn'] = scores

    logitPath = f'{DATA_DIR}/valid_8192_evo2_logits_new.tsv'
    logits = pd.read_csv(logitPath, header=None, sep='\t')
    logits.columns = ['A', 'C', 'G', 'T']
    scores = df.apply(
        lambda row: logits.loc[row.name, REF.loc[row.name]] if REF.loc[row.name] in "ATCG" else 0,
        axis=1
    )
    df['evo2'] = scores

    models = df.columns[4:].tolist()
    y_true = df['label']
    results = []

    for mdl in models:
        y_scores = df[mdl]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        if mdl == 'evo2':
            context = 8192
        elif mdl == 'pcv1' or mdl == 'gpn' or mdl == 'pdllm':
            context = 512
        else:
            context = mdl.split('_')[0]
            mdl = "_".join(mdl.split("_")[1:])
        results.append({'model': mdl, 'context': context, 'roc_auc': roc_auc})

    results_df = pd.DataFrame(results)

    # save results_df to csv
    results_df.to_csv('Figure2B_results.tsv', index=False, sep='\t')

    subsets_df = results_df[12:]
    model_order = ['pcv1', 'pcv2_l24_d0768',  'pcv2_l48_d1024', 'pcv2_l48_d1536',  'evo2', 'gpn']
    subsets_df = subsets_df.set_index('model').reindex(model_order).reset_index()

    colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff',  '#808080','#999999']
    plot_performance(
        subsets_df['model'], 
        subsets_df['roc_auc'], 
        colors, 
        'Figure2B.pdf', 
        ylim=(0.3, 0.8), 
        rotation=90
    )

def process_figure_2c():
    # for Figure 2C (potato deleterious mutations)
    print("Processing Figure 2C...")
    raw = pd.read_table(
        "../3_potato_deleterious_mutations/5_figures/metric.md",
        sep="|",
        engine="python",
        skiprows=[1]
    )

    raw = raw.dropna(how="all")

    df = raw.iloc[:, 1:-1]

    df.columns = df.columns.str.strip()
    df["Model"] = df["Model"].str.strip()

    df["AUROC"] = df["AUROC"].astype(float)
    df["AUPRC"] = df["AUPRC"].astype(float)

    # Save metric to TSV
    df.to_csv('Figure2C_results.tsv', index=False, sep='\t')

    models = df["Model"]
    scores = df["AUROC"]

    print(df)
    colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff',  '#808080','#999999']
    plot_performance(
        models, 
        scores, 
        colors, 
        'Figure2C.pdf', 
        ylim=(0.3, 0.8), 
        rotation=45, 
        ha='right'
    )

def process_figure_2d():
    # for Figure 2D (orginally Figure 2C; conservation within poaceae; non-TIS sites in coding sequences)
    print("Processing Figure 2D...")
    df = pd.read_csv('../../pipelines/Poaceae_CDS_PhyloP/figures/performance_compairson.tsv', sep='\t')
    
    # Save metric to TSV (saving the loaded dataframe for consistency)
    df.to_csv('Figure2D_results.tsv', index=False, sep='\t')

    subsets_df = df[12:]
    # remove pdllm
    subsets_df = subsets_df[subsets_df['model'] != 'pdllm']

    model_order = ['pcv1', 'pcv2-l24-d0768',  'pcv2-l48-d1024', 'pcv2-l48-d1536', 'evo2', 'gpn']
    subsets_df = subsets_df.set_index('model').reindex(model_order).reset_index()
    subsets_df

    colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff', '#808080', '#999999', '#d3d3d3']
    plot_performance(
        subsets_df['model'], 
        subsets_df['roc_auc'], 
        colors, 
        'Figure2D.pdf', 
        ylim=(0.3, 0.9), 
        rotation=45, 
        xlabel='Models'
    )

def process_figure_2e():
    # for Figure 2E (orginally Figure 2D; conservation within poaceae; TIS sites in coding sequences)
    print("Processing Figure 2E...")
    models = ['pcv1', 'pcv2-l24-d0768', 'pcv2-l48-d1024', 'pcv2-l48-d1536', 'evo2', 'gpn']
    tis_files = ['TIS_1', 'TIS_2', 'TIS_3']
    data_types = ['conserved', 'neutral']
    DATA_DIR = '/local/workdir/jz963/utils/plantcad2/results/Poaceae_CDS_PhyloP'

    results = {}

    for dtype in data_types:
        data_list = []
        
        for tis in tis_files:
            print(f"Processing {dtype} {tis}...")
            df = pd.read_csv(f'{DATA_DIR}/8k/{dtype}_8k_{tis}.tsv', sep='\t')
            REF = df['Seq'].str[4095]
            
            for model in models:
                print(f"  Processing {model}...")
                if 'pcv2' not in model:
                    logitPath = f'{DATA_DIR}/8k/{dtype}_8k_{tis}_{model}.tsv'
                else:
                    logitPath = f'{DATA_DIR}/8k/{dtype}_8k_{tis}_{model}_logits.tsv'
                
                logits = pd.read_csv(logitPath, header=None, sep='\t')
                logits.columns = ['A', 'C', 'G', 'T']
                scores = df.apply(lambda row: logits.loc[row.name, REF.loc[row.name]], axis=1)
                df[model] = scores
            
            data_list.append(df)
        
        results[dtype] = pd.concat(data_list, ignore_index=True)

    conserved = results['conserved']
    neutral = results['neutral']
    conserved['label'] = 1
    neutral['label'] = 0

    combined_df = pd.concat([conserved, neutral], ignore_index=True)
    y_true = combined_df['label']
    models = combined_df.columns[6:].tolist()[:-1]

    results = []

    for mdl in models:
        y_scores = combined_df[mdl]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        results.append({'model': mdl, 'context': 8192, 'roc_auc': roc_auc})
    results_df = pd.DataFrame(results)

    # Save results to TSV
    results_df.to_csv('Figure2E_results.tsv', index=False, sep='\t')

    colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff', '#808080', '#999999', '#d3d3d3']
    plot_performance(
        results_df['model'], 
        results_df['roc_auc'], 
        colors, 
        'Figure2E.pdf', 
        ylim=(0.3, 0.8), 
        rotation=45, 
        xlabel='Models'
    )

if __name__ == "__main__":
    if sys.argv[1] == '2b':
        process_figure_2b()
    elif sys.argv[1] == '2c':
        process_figure_2c()
    elif sys.argv[1] == '2d':
        process_figure_2d()
    elif sys.argv[1] == '2e':
        process_figure_2e()
    else:
        process_figure_2b()
        process_figure_2c()
        process_figure_2d()
        process_figure_2e()
        
