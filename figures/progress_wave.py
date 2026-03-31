"""Generate progress chart for wave function experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# All wave experiments (ordered chronologically)
experiments = [
    # (label, wave_val, std_val, wave_params_M, std_params_M)
    ("2w undiff\n(all layers)", 4.5184, 4.5163, 22.1, 22.9),
    ("4w undiff", 4.5903, 4.5163, 21.3, 22.9),
    ("8w undiff", 4.6624, 4.5163, 14.7, 22.9),
    ("2w diff\n(all layers)", 4.4960, 4.4948, 22.1, 22.9),
    ("4w diff", 4.5695, 4.4948, 21.3, 22.9),
    ("2w rank=8", 4.5277, 4.4962, 22.1, 22.9),
    ("2w rank=16", 4.5452, 4.4962, 22.1, 22.9),
    ("2w boost\n(late)", 4.5346, 4.5312, 22.7, 22.9),
    ("2w norm\n(late)", 4.5216, 4.4961, 22.5, 22.9),
    ("2w diff\n(late) *best*", 4.5107, 4.5138, 22.5, 22.9),  # 3-trial avg
]

labels = [e[0] for e in experiments]
wave_vals = [e[1] for e in experiments]
std_vals = [e[2] for e in experiments]
gaps = [e[2] - e[1] for e in experiments]
param_savings = [(1 - e[3]/e[4])*100 for e in experiments]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

# Top: val loss comparison
x = np.arange(len(labels))
width = 0.35
bars1 = ax1.bar(x - width/2, wave_vals, width, label='Wave', color='#2196F3', alpha=0.8)
bars2 = ax1.bar(x + width/2, std_vals, width, label='Standard', color='#FF9800', alpha=0.8)

ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Wave Function Attention vs Standard Transformer\n(TinyStories 50K, d=512, h=8, L=6)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
ax1.legend(fontsize=11)
ax1.set_ylim(4.45, 4.7)
ax1.axhline(y=4.5138, color='#FF9800', linestyle='--', alpha=0.5, label='Std mean (3-trial)')
ax1.grid(axis='y', alpha=0.3)

# Bottom: gap (positive = wave wins)
colors = ['#4CAF50' if g > 0 else '#F44336' for g in gaps]
ax2.bar(x, gaps, width=0.6, color=colors, alpha=0.8)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel('Gap (std - wave)', fontsize=12)
ax2.set_xlabel('Experiment Configuration', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Add param savings annotation
for i, (g, ps) in enumerate(zip(gaps, param_savings)):
    ax2.annotate(f'{ps:.1f}%\nfewer', (i, g), ha='center', va='bottom' if g > 0 else 'top',
                fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('progress_wave.png', dpi=150, bbox_inches='tight')
print('Saved progress_wave.png')

# Summary stats
print(f'\nKey findings:')
print(f'  Best config: 2w differentiated, late layers only')
print(f'  3-trial avg: wave {4.5107:.4f} vs std {4.5138:.4f} (wave wins +0.003)')
print(f'  Wave variance: ±0.005 vs std ±0.011 (2x more stable)')
print(f'  Param savings: 394K (1.7%)')
print(f'  Predetermined overlap > learned projections (rank, boost, temperature all hurt)')
