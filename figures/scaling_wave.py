"""Generate scaling chart for wave function experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Scaling scan data
dims = [256, 320, 384, 448, 512]
std_params = [8.3, 11.4, 14.8, 18.7, 22.9]
wave_params = [8.2, 11.2, 14.6, 18.4, 22.5]
std_vals = [5.3388, 5.0219, 4.8506, 4.6504, 4.5277]
wave_vals = [5.2521, 5.0358, 4.8444, 4.6351, 4.5272]
gaps = [s - w for s, w in zip(std_vals, wave_vals)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

# Top: val loss vs model dimension
ax1.plot(dims, std_vals, 'o-', color='#FF9800', linewidth=2, markersize=8, label='Standard', zorder=3)
ax1.plot(dims, wave_vals, 's-', color='#2196F3', linewidth=2, markersize=8, label='Wave (2w late)', zorder=3)

# Shade the advantage regions
for i in range(len(dims)):
    if gaps[i] > 0:
        ax1.annotate(f'+{gaps[i]:.3f}', (dims[i], wave_vals[i] - 0.02),
                    ha='center', fontsize=9, color='#2196F3', fontweight='bold')

ax1.set_xlabel('Model Dimension (d_model)', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Wave Function Attention: Scaling Behavior\nTinyStories 50K, h=8, L=6, 2w sym-late', fontsize=14)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(alpha=0.3)
ax1.invert_yaxis()

# Add param annotations
for i, d in enumerate(dims):
    ax1.annotate(f'{std_params[i]:.0f}M', (d, std_vals[i] + 0.015),
                ha='center', fontsize=7, color='#FF9800', alpha=0.7)
    ax1.annotate(f'{wave_params[i]:.0f}M', (d, wave_vals[i] + 0.015),
                ha='center', fontsize=7, color='#2196F3', alpha=0.7)

# Bottom: gap vs model size
colors = ['#4CAF50' if g > 0 else '#F44336' for g in gaps]
bars = ax2.bar(dims, gaps, width=40, color=colors, alpha=0.8)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('Model Dimension (d_model)', fontsize=12)
ax2.set_ylabel('Gap (std - wave)', fontsize=12)
ax2.set_title('Wave Advantage by Scale (positive = wave wins)', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

for i, (d, g) in enumerate(zip(dims, gaps)):
    ax2.annotate(f'{g:+.3f}', (d, g), ha='center',
                va='bottom' if g > 0 else 'top',
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('scaling_wave.png', dpi=150, bbox_inches='tight')
print('Saved scaling_wave.png')
print(f'\nWave wins at {sum(1 for g in gaps if g > 0)}/{len(gaps)} scales')
print(f'Biggest win: d={dims[gaps.index(max(gaps))]} gap={max(gaps):.4f}')
