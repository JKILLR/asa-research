"""Final comprehensive chart: 93 wave function experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Compression map
ax = axes[0, 0]
scales = ['d=256', 'd=384', 'd=512']
compress_better = [1.30, 1.17, 0]  # compression where quality improves
compress_close = [0, 1.31, 1.18]  # compression where quality slightly degrades
quality_better = [0.051, 0.042, 0]
quality_close = [0, -0.043, -0.011]

x = np.arange(len(scales))
bars1 = ax.bar(x - 0.2, compress_better, 0.35, label='BETTER quality', color='#4CAF50', alpha=0.8)
bars2 = ax.bar(x + 0.2, compress_close, 0.35, label='Close quality', color='#FF9800', alpha=0.8)

ax.set_ylabel('Compression (×)', fontsize=11)
ax.set_title('Wave-Gated FFN: Compression by Scale\n(seq=256, 50K TinyStories)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.5)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
for i in range(len(scales)):
    if compress_better[i] > 0:
        ax.annotate(f'{compress_better[i]:.2f}×\n({quality_better[i]:+.3f})',
                   (i-0.2, compress_better[i]+0.02), ha='center', fontsize=8, fontweight='bold')
    if compress_close[i] > 0:
        ax.annotate(f'{compress_close[i]:.2f}×\n({quality_close[i]:+.3f})',
                   (i+0.2, compress_close[i]+0.02), ha='center', fontsize=8)

# Panel 2: FFN gate impact
ax = axes[0, 1]
configs = ['Attention\nonly', 'Attention\n+FFN gate', 'FFN gate\nff=256', 'FFN gate\nff=128']
gaps = [0.070, 0.164, 0.217, 0.239]
colors = ['#2196F3', '#4CAF50', '#4CAF50', '#4CAF50']

ax.bar(range(len(configs)), gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Gap vs Standard (positive = wave wins)', fontsize=10)
ax.set_title('FFN Gating Impact (d=256, seq=256)', fontsize=12)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=9)
for i, g in enumerate(gaps):
    ax.annotate(f'+{g:.3f}', (i, g+0.005), ha='center', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Panel 3: Sequence length scaling
ax = axes[1, 0]
seqs = [128, 256, 512]
gaps_attn = [0.062, 0.104, 0.129]  # attention-only wave
gaps_ffn = [None, 0.164, 0.140]  # with FFN gate (where available)

ax.plot(seqs, gaps_attn, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Attention-only wave')
ax.plot([256, 512], [0.164, 0.140], 's-', color='#4CAF50', linewidth=2, markersize=8, label='Wave + FFN gate')

ax.set_xlabel('Sequence Length', fontsize=11)
ax.set_ylabel('Gap (std - wave)', fontsize=11)
ax.set_title('Wave Advantage vs Sequence Length (d=256)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 4: Summary stats
ax = axes[1, 1]
ax.axis('off')
msg = """93 EXPERIMENTS: WAVE FUNCTION ATTENTION

CONFIRMED (statistically significant):
  seq=128 d=256: +0.062 (t=3.14, 5/5)
  seq=256 d=256: +0.104 (t=11.93, 3/3)
  1.30× compression: +0.051 (t=2.91, 3/3)

THE BREAKTHROUGH: Wave-Gated FFN
  FFN gate doubles advantage (2.3×)
  Half-FFN beats full-FFN with gate
  Works at d=256, d=384, d=512

COMPRESSION MAP:
  d=256: 1.30× (BETTER)
  d=384: 1.17-1.31× (close)
  d=512: 1.18× (close)

LIMITATIONS:
  × Not 2× — wave is small-model technique
  × Degrades at 200K data
  × d=512 advantages marginal

IDEAL USE: Small models, long context,
limited data, edge deployment"""

ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=9.5,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Wave Function Attention: 93 Experiments Summary', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('final_chart.png', dpi=150, bbox_inches='tight')
print('Saved final_chart.png')
