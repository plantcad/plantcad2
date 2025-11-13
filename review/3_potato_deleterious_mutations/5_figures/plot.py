import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re


raw = pd.read_table(
    "metric.md",
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

models = df["Model"]
scores = df["AUROC"]

print(df)
plt.figure(figsize=(5, 5))
colors = ['#1f77b466', '#6baed6', '#1f77b4b3', '#1f77b4ff',  '#808080','#999999']

colors = colors[:len(df)]

plt.figure(figsize=(5, 5))
x = np.arange(len(models))

plt.bar(x, scores, color=colors)

plt.xticks(x, models, rotation=45, ha="right")
plt.ylabel("AUROC")

plt.ylim(0.3, 0.8) 
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("auroc_plot.pdf", format="pdf", dpi=300)

plt.close()

print("Saved to auroc_plot.pdf")
