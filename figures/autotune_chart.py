"""
Karpathy-style autotune progress chart — TinyStories wave function experiments only.
Y-axis = ASA val_loss (lower is better), like Karpathy's chart.
"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'sans-serif'

# Parse results — TinyStories only (val_loss < 5.5)
experiments = []
with open('results_full.tsv', 'r') as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        try:
            asa_val = float(parts[1])
            baseline_val = float(parts[2])
            gap = float(parts[3])
        except ValueError:
            continue
        if asa_val > 5.5:  # Skip WikiText-2 experiments
            continue
        experiments.append({
            'asa_val': asa_val,
            'baseline_val': baseline_val,
            'gap': gap,
            'status': parts[6],
            'desc': parts[7],
        })

# Running best (lowest ASA val_loss among KEEPs)
running_best = []
best = float('inf')
for exp in experiments:
    if exp['status'] == 'KEEP':
        best = min(best, exp['asa_val'])
    running_best.append(best)

n_total = len(experiments)
n_kept = sum(1 for e in experiments if e['status'] == 'KEEP')
n_disc = n_total - n_kept

# Plot
fig, ax = plt.subplots(figsize=(20, 9))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Scatter
for i, exp in enumerate(experiments):
    if exp['status'] == 'KEEP':
        color, alpha, size, zorder = '#3fb950', 0.9, 60, 3
    elif exp['status'] == 'REVERT':
        color, alpha, size, zorder = '#8b949e', 0.3, 22, 1
    else:
        color, alpha, size, zorder = '#8b949e', 0.45, 28, 1
    ax.scatter(i, exp['asa_val'], c=color, alpha=alpha, s=size, zorder=zorder, edgecolors='none')

# Running best line
ax.step(range(n_total), running_best, where='post', color='#3fb950', linewidth=2.5, alpha=0.85, zorder=2)

# Annotations for key milestones
annotations = []
for i, exp in enumerate(experiments):
    d = exp['desc']
    label = None
    if 'wave 2w+6s d512' in d and not annotations:
        label = 'first wave heads'
    elif 'WAVE WINS' in d and 'First clear' in d:
        label = 'wave wins (first!)'
    elif 'wave-late' in d and 'BIGGEST WIN' in d:
        label = 'late layers best'
    elif 'LONG-RANGE' in d and 'CONFIRMED' in d and 'seq=256' in d and '200k' not in d.lower():
        label = 'long-range confirmed'
    elif 'seq=512' in d and 'p<<' in d:
        label = 'seq=512 gap grows'
    elif 'Contextual+Res' in d and 'BREAKS' in d:
        label = 'contextual waves'
    elif 'FFN GATE d=256' in d and 'BIGGEST' in d:
        label = 'FFN wave gate!'
    elif '+0.239' in d or ('ff=256' in d and '11.4%' in d):
        label = 'half FFN > full FFN'
    elif '1.30x COMPRESSION CONFIRMED' in d:
        label = '1.30x compression!'
    elif 'DATA SCALING d256 200K' in d and 'PERSISTS' in d:
        label = '200K: advantage persists'
    elif 'DATA SCALING d256 50K' in d:
        label = 'data scaling'

    if label:
        too_close = any(abs(i - ai) < 3 for ai, _ in annotations)
        if not too_close:
            annotations.append((i, label))

for idx, label in annotations:
    y = experiments[idx]['asa_val']
    y_off = -18 if y > running_best[idx] + 0.01 else 14
    ax.annotate(label, (idx, y), textcoords="offset points",
               xytext=(8, y_off), fontsize=7.5, color='#c9d1d9', alpha=0.8,
               rotation=28, ha='left',
               arrowprops=dict(arrowstyle='-', color='#c9d1d9', alpha=0.35, lw=0.6))

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#8b949e',
           markersize=6, alpha=0.4, label=f'Discarded ({n_disc})'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#3fb950',
           markersize=8, label=f'Kept ({n_kept})'),
    Line2D([0], [0], color='#3fb950', linewidth=2.5, label='Running best'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
         facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

ax.set_xlabel('Experiment #', color='#c9d1d9', fontsize=12)
ax.set_ylabel('Validation Loss (lower is better)', color='#c9d1d9', fontsize=12)
ax.set_title(f'ASA Autoresearch: {n_total} Wave Function Experiments, {n_kept} Kept Improvements',
            color='#c9d1d9', fontsize=14, fontweight='bold')
ax.tick_params(colors='#8b949e')
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.1, color='#8b949e')

plt.tight_layout()
plt.savefig('autotune_progress.png', dpi=150, facecolor='#0d1117', bbox_inches='tight')
print(f"Saved — {n_total} experiments, {n_kept} kept, {n_disc} discarded")
