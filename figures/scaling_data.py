"""Visualize data scaling: wave vs standard at different data sizes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

with open("results_data_scaling.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax_idx, (d, d_label) in enumerate([(256, "d=256"), (512, "d=512")]):
    ax = axes[ax_idx]
    sizes = [50, 100, 200]
    std_vals = [data[f"d{d}_n{s}K"]["std_val"] for s in sizes]
    wave_vals = [data[f"d{d}_n{s}K"]["wave_val"] for s in sizes]
    gaps = [data[f"d{d}_n{s}K"]["gap"] for s in sizes]

    ax.plot(sizes, std_vals, 'o-', color='#FF9800', linewidth=2, markersize=8, label='Standard')
    ax.plot(sizes, wave_vals, 's-', color='#2196F3', linewidth=2, markersize=8, label='Wave')

    for i, (s, g) in enumerate(zip(sizes, gaps)):
        color = '#4CAF50' if g > 0 else '#F44336'
        ax.annotate(f'{g:+.3f}', (s, min(std_vals[i], wave_vals[i]) - 0.02),
                   ha='center', fontsize=10, color=color, fontweight='bold')

    ax.set_xlabel('Training Stories (K)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(f'{d_label} (wave 2w {"all" if d==256 else "late"})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# Third panel: gap vs data size
ax = axes[2]
for d, color, marker, label in [(256, '#2196F3', 's', 'd=256'), (512, '#9C27B0', 'D', 'd=512')]:
    sizes = [50, 100, 200]
    gaps = [data[f"d{d}_n{s}K"]["gap"] for s in sizes]
    ax.plot(sizes, gaps, f'{marker}-', color=color, linewidth=2, markersize=8, label=label)

ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.fill_between([0, 250], -0.005, 0.005, color='gray', alpha=0.15, label='Noise band')
ax.set_xlabel('Training Stories (K)', fontsize=12)
ax.set_ylabel('Gap (std - wave)', fontsize=12)
ax.set_title('Wave Advantage vs Data Size\n(positive = wave wins)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(30, 220)

plt.tight_layout()
plt.savefig('scaling_data.png', dpi=150, bbox_inches='tight')
print('Saved scaling_data.png')

# Summary
print("\nData Scaling Summary:")
print("  d=256: 50K→+0.048, 100K→-0.003, 200K→+0.043")
print("  d=512: 50K→+0.020, 100K→-0.016, 200K→+0.010")
print("\n  Wave advantage is NON-MONOTONIC.")
print("  100K shows a dip (possible: LR/batch interaction with data size)")
print("  200K: advantage RETURNS at both scales")
print("  This suggests STRUCTURAL advantage, not just sample efficiency")
