"""One figure: all categories, checkpoints, readouts and training horizons."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

out = Path(__file__).resolve().parent
source = out/'all-category-scores.csv'
with source.open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 140
metrics = ['Conservation', 'Motifs', 'Core/non-core', 'SV impact', 'Allele frequency',
           'Composite −SV', 'Composite +SV']
runs = ['216B-token run', '1T-token run']
readouts = ['Zero-shot likelihood', 'Linear probe']
palette = ['#ffe0bd', '#f1f1f1', '#dceaf6']
plt.rcParams.update({'font.size': 12, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4), sharey='row')
count = 0
for row_index, run in enumerate(runs):
    tokens = sorted({float(row['DNA_tokens_B']) for row in rows if row['run'] == run})
    assert len(tokens) == 5
    ends = ['1T' if value >= 999 else f'{value:.0f}B' for value in tokens[1:]]
    intervals = [f'{a:.0f}–{b}' for a, b in zip(tokens, ends)]
    for column, readout in enumerate(readouts):
        ax = axes[row_index, column]
        values = np.full((len(metrics), 4), np.nan)
        for i, metric in enumerate(metrics):
            points = sorted((row for row in rows if row['run'] == run and row['readout'] == readout
                             and row['metric'] == metric), key=lambda row: float(row['DNA_tokens_B']))
            if points:
                assert len(points) == 5
                values[i] = np.diff([float(point['score']) for point in points])
                count += 4
        signs = np.where(np.isnan(values), 0, np.sign(values))
        ax.imshow(signs, cmap=ListedColormap(palette), vmin=-1, vmax=1,
                  aspect='auto', interpolation='nearest')
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, '—' if np.isnan(value) else f'{value:+.4f}', ha='center', va='center',
                    color='#999999' if np.isnan(value) else '#202020')
        ax.set_xticks(range(4), intervals)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(metrics)), metrics)
        ax.tick_params(axis='both', which='major', length=0, pad=9)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(metrics), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        horizon = '216B' if row_index == 0 else '1T'
        method = 'zero-shot' if column == 0 else 'linear probes'
        ax.set_title(f'{horizon} run · {method}', pad=40)
assert count == 112
fig.suptitle('Where scores improve or decline during DNA training', y=.995)
fig.legend(handles=[Patch(facecolor=palette[2], label='Increase'),
                    Patch(facecolor=palette[0], label='Decrease'),
                    Patch(facecolor=palette[1], label='No change')],
           loc='lower center', bbox_to_anchor=(.57, .015), ncols=3, frameon=False)
fig.subplots_adjust(left=.20, right=.99, bottom=.12, top=.83, hspace=.55, wspace=.06)
fig.supxlabel('DNA-token intervals (rounded)', y=.075)
for extension in ['svg', 'png']:
    fig.savefig(out/'figures'/f'checkpoint-changes.{extension}', dpi=160,
                facecolor='white', bbox_inches='tight', pad_inches=.12)
plt.close(fig)
(out/'one-figure-provenance.json').write_text(json.dumps({
    'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'input_scores': len(rows), 'displayed_checkpoint_changes': count,
    'metrics': metrics, 'runs': runs, 'readouts': readouts,
    'context_bp': 8192, 'color_encoding': 'Sign only; annotations give magnitude',
    'GitHub_updates': False,
}, indent=2)+'\n')
