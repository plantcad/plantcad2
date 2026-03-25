import os,sys
import json
import glob

import pandas as pd
from sklearn.metrics import average_precision_score

results_dir = sys.argv[1]
output_file = sys.argv[2]

pattern = os.path.join(results_dir, "*_metrics.json")
files = glob.glob(pattern)

results = {}

for f in files:
    name = os.path.basename(f)
    if name == ".json": continue
    try:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            results[name] = data
    except Exception as e:
        print(f"Error reading {name}: {e}")

# Process SV Effect
sv_file = os.path.join(results_dir, "sv_effect_scored.tsv")
if os.path.exists(sv_file):
    try:
        df = pd.read_csv(sv_file, sep="\t")
        y_true = df["label"].astype(int)
        scores = df["score"]
        auprc = average_precision_score(y_true, scores)
        results["sv_effect_metrics.json"] = {"auprc": auprc}
    except Exception as e:
        print(f"Error processing SV effect: {e}")

# Group by task
grouped = {}
for name, data in results.items():
    # name format: {task}_{split}_metrics.json
    # but some tasks have underscores.
    # The script uses: ${task}_${split}_metrics.json
    # splits are test_maize, test_tomato
    
    if "_test_maize_" in name:
        task = name.replace("_test_maize_metrics.json", "")
        split = "test_maize"
    elif "_test_tomato_" in name:
        task = name.replace("_test_tomato_metrics.json", "")
        split = "test_tomato"
    elif name == "sv_effect_metrics.json":
        task = "structural_variant_effect_prediction"
        split = "test"
    else:
        task = name.replace("_metrics.json", "")
        split = "test"
    
    if task not in grouped:
        grouped[task] = {}
    grouped[task][split] = data

# Generate Markdown
md_lines = []
md_lines.append("# Evaluation Results")
md_lines.append("")

# Classification Tasks
md_lines.append("## Core/Non-core Classification Tasks")
md_lines.append("| Task | Split | AUROC | AUPRC |")
md_lines.append("| :--- | :--- | :--- | :--- |")

classification_tasks = []
recovery_tasks = []
conservation_tasks = []
sv_tasks = []

for task, splits in grouped.items():
    # Check if it looks like a classification task (has auroc)
    first_split = list(splits.values())[0]
    if task == "structural_variant_effect_prediction":
        sv_tasks.append(task)
    elif "conservation" in task:
        conservation_tasks.append(task)
    elif "auroc" in first_split:
        classification_tasks.append(task)
    elif "token_accuracy" in first_split:
        recovery_tasks.append(task)

for task in sorted(classification_tasks):
    splits = grouped[task]
    for i, split in enumerate(sorted(splits.keys())):
        metrics = splits[split]
        auroc = f"{metrics.get('auroc', 0):.3f}"
        auprc = f"{metrics.get('auprc', 0):.3f}"
        task_col = f"**{task}**" if i == 0 else ""
        md_lines.append(f"| {task_col} | {split} | {auroc} | {auprc} |")

md_lines.append("")
md_lines.append("## Conservation Tasks")
md_lines.append("| Task | Split | AUROC | AUPRC |")
md_lines.append("| :--- | :--- | :--- | :--- |")

for task in sorted(conservation_tasks):
    splits = grouped[task]
    for i, split in enumerate(sorted(splits.keys())):
        metrics = splits[split]
        auroc = f"{metrics.get('auroc', 0):.3f}"
        auprc = f"{metrics.get('auprc', 0):.3f}"
        task_col = f"**{task}**" if i == 0 else ""
        md_lines.append(f"| {task_col} | {split} | {auroc} | {auprc} |")

md_lines.append("")
md_lines.append("## Motif Recovery Tasks")
md_lines.append("| Task | Split | Token Accuracy | Motif Accuracy |")
md_lines.append("| :--- | :--- | :--- | :--- |")

for task in sorted(recovery_tasks):
    splits = grouped[task]
    for i, split in enumerate(sorted(splits.keys())):
        metrics = splits[split]
        token_acc = f"{metrics.get('token_accuracy', 0):.3f}"
        motif_acc = f"{metrics.get('motif_accuracy', 0):.3f}"
        task_col = f"**{task}**" if i == 0 else ""
        md_lines.append(f"| {task_col} | {split} | {token_acc} | {motif_acc} |")

md_lines.append("")
md_lines.append("## Structural Variant Effect Prediction")
md_lines.append("| Task | Split | AUPRC |")
md_lines.append("| :--- | :--- | :--- |")

for task in sorted(sv_tasks):
    splits = grouped[task]
    for i, split in enumerate(sorted(splits.keys())):
        metrics = splits[split]
        auprc = f"{metrics.get('auprc', 0):.3f}"
        task_col = f"**{task}**" if i == 0 else ""
        md_lines.append(f"| {task_col} | {split} | {auprc} |")

with open(output_file, "w") as f:
    f.write("\n".join(md_lines))

print(f"Markdown results written to {output_file}")