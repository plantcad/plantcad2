import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data
human = {
    'vertcad2-88M':    {'TIS': 0.355276, 'TTS': 0.074033, 'Acceptor': 0.737538, 'Donor': 0.823369},
    'vertcad2-676M':   {'TIS': 0.509045, 'TTS': 0.177086, 'Acceptor': 0.791987, 'Donor': 0.873908},
    'evo2-7b-rc':      {'TIS': 0.7317,   'TTS': 0.0889,   'Acceptor': 0.885,    'Donor': 0.1734},
    'evo2-7b-fwd':     {'TIS': 0.1915,   'TTS': 0.6908,   'Acceptor': 0.2011,   'Donor': 0.8671},
    'evo2-1b-rc':      {'TIS': 0.4715,   'TTS': 0.024,    'Acceptor': 0.6685,   'Donor': 0.0864},
    'evo2-1b-fwd':     {'TIS': 0.0531,   'TTS': 0.3907,   'Acceptor': 0.1595,   'Donor': 0.6451},
}

mouse = {
    'vertcad2-88M':    {'TIS': 0.326453, 'TTS': 0.068538, 'Acceptor': 0.730177, 'Donor': 0.803994},
    'vertcad2-676M':   {'TIS': 0.497296, 'TTS': 0.166772, 'Acceptor': 0.787372, 'Donor': 0.856771},
    'evo2-7b-rc':      {'TIS': 0.696,    'TTS': 0.0604,   'Acceptor': 0.8515,   'Donor': 0.1652},
    'evo2-7b-fwd':     {'TIS': 0.1716,   'TTS': 0.5847,   'Acceptor': 0.2165,   'Donor': 0.8279},
    'evo2-1b-rc':      {'TIS': 0.4708,   'TTS': 0.0235,   'Acceptor': 0.6505,   'Donor': 0.0776},
    'evo2-1b-fwd':     {'TIS': 0.0721,   'TTS': 0.3397,   'Acceptor': 0.1713,   'Donor': 0.6191},
}

metrics = ['TIS', 'TTS', 'Acceptor', 'Donor']

def best_evo2(data, prefix):
    rc = data[f'{prefix}-rc']
    fwd = data[f'{prefix}-fwd']
    return {m: max(rc[m], fwd[m]) for m in metrics}

def build_model_data(data):
    return {
        'vertcad2-88M':  data['vertcad2-88M'],
        'vertcad2-676M': data['vertcad2-676M'],
        'evo2-7b':       best_evo2(data, 'evo2-7b'),
        'evo2-1b':       best_evo2(data, 'evo2-1b'),
    }

human_plot = build_model_data(human)
mouse_plot = build_model_data(mouse)

model_names = list(human_plot.keys())
model_labels = ['vertcad2-88M', 'vertcad2-676M', 'evo2-7b\n(best)', 'evo2-1b\n(best)']
colors = ['#4C72B0', '#1A4A8A', '#999999', '#CCCCCC']

x = np.arange(len(metrics))
n = len(model_names)
width = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax, title, plot_data in zip(axes, ['Human', 'Mouse'], [human_plot, mouse_plot]):
    for i, (model, label) in enumerate(zip(model_names, model_labels)):
        vals = [plot_data[model][m] for m in metrics]
        ax.bar(x + offsets[i], vals, width, label=label, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

handles = [mpatches.Patch(color=colors[i], label=model_labels[i].replace('\n', ' '), alpha=0.85)
           for i in range(n)]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.05), frameon=False)

fig.suptitle('Junction Recovery — Zero-shot Accuracy\n(evo2: best of rc/fwd per metric)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/local/workdir/jz963/utils/plantcad2/review-2/junction-recov/results_plot.pdf',
            bbox_inches='tight')
plt.savefig('/local/workdir/jz963/utils/plantcad2/review-2/junction-recov/results_plot.png',
            bbox_inches='tight', dpi=150)
print("Saved results_plot.pdf and results_plot.png")
