"""Generate the ASA scaling chart — single panel, clean, impressive."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

# 3-seed averaged results from GPU autoresearch
models = ['3M params', '8M params', '23M params']
gaps = [0.053, 0.074, 0.086]
stds = [0.013, 0.010, 0.004]

fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(len(models))
colors = ['#2ecc71', '#27ae60', '#1e8449']

bars = ax.bar(x, gaps, yerr=stds, capsize=10, color=colors,
              edgecolor='white', linewidth=2.5, width=0.45, zorder=3,
              error_kw={'linewidth': 2, 'capthick': 2, 'color': '#333'})

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14, fontweight='bold')
ax.set_ylabel('ASA Advantage Over Baseline', fontsize=14, fontweight='bold')
ax.set_title('Advantage Grows with Scale', fontsize=20, fontweight='bold', pad=20)

ax.grid(True, alpha=0.15, axis='y')
ax.set_ylim(0, 0.115)
ax.set_xlim(-0.5, 2.5)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_alpha(0.3)
ax.spines['bottom'].set_alpha(0.3)
ax.tick_params(axis='y', labelsize=11)

# Annotate bars with values
for i, (g, s) in enumerate(zip(gaps, stds)):
    ax.text(i, g + s + 0.004, f'+{g:.3f}',
            ha='center', va='bottom', fontsize=16, fontweight='bold', color='#1a1a1a')

# p-values below x labels
for i, p in enumerate(['p < 0.01', 'p < 0.001', 'p < 0.001']):
    ax.text(i, -0.008, p, ha='center', fontsize=10, color='#888', style='italic')

# Subtitle
ax.text(0.5, -0.14, '66 autonomous experiments on A100  ·  3-seed averages  ·  WikiText-2',
        transform=ax.transAxes, ha='center', fontsize=11, color='#999')

plt.tight_layout()
plt.savefig('scaling_chart.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved scaling_chart.png")
