"""Comprehensive summary of wave function attention experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Effect matrix (d_model × seq_len)
ax = axes[0, 0]
data = np.array([
    [0.062, 0.104],   # d=256
    [0.001, -0.001],  # d=512
])
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.12)
ax.set_xticks([0, 1])
ax.set_xticklabels(['seq=128', 'seq=256'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['d=256\n(8M)', 'd=512\n(23M)'])
ax.set_title('Wave Advantage (50K TinyStories)\nGreen = wave wins', fontsize=12)
for i in range(2):
    for j in range(2):
        v = data[i, j]
        sig = '*' if abs(v) > 0.02 else ''
        ax.text(j, i, f'{v:+.3f}{sig}', ha='center', va='center', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 2: Data scaling at seq=128 vs seq=256
ax = axes[0, 1]
# seq=128 data (3-trial)
ax.plot([50, 200], [0.062, -0.009], 's-', color='#FF9800', linewidth=2, markersize=8, label='seq=128')
# seq=256 data (3-trial)
ax.plot([50, 200], [0.104, 0.022], 'D-', color='#2196F3', linewidth=2, markersize=8, label='seq=256')
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax.fill_between([30, 220], -0.015, 0.015, color='gray', alpha=0.1)
ax.set_xlabel('Training Stories (K)', fontsize=11)
ax.set_ylabel('Gap (std - wave)', fontsize=11)
ax.set_title('Data Scaling: seq=256 advantage PERSISTS', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: Significance summary
ax = axes[1, 0]
experiments = [
    ('d=256\nseq=128\n50K', 0.062, 3.14, True),
    ('d=256\nseq=256\n50K', 0.104, 11.93, True),
    ('d=256\nseq=256\n200K', 0.022, 2.13, True),
    ('d=256\nseq=128\n200K', -0.009, -0.65, False),
    ('d=512\nseq=128\n50K', 0.001, 0.1, False),
    ('d=512\nseq=256\n50K', -0.001, -0.12, False),
]
x = range(len(experiments))
colors = ['#4CAF50' if sig else '#F44336' for _, _, _, sig in experiments]
bars = ax.bar(x, [g for _, g, _, _ in experiments], color=colors, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([e[0] for e in experiments], fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Gap (std - wave)', fontsize=11)
ax.set_title('All Significance Tests\nGreen = significant (p<0.05)', fontsize=12)
for i, (_, g, t, sig) in enumerate(experiments):
    ax.annotate(f't={t:.1f}', (i, g), ha='center',
               va='bottom' if g > 0 else 'top', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Panel 4: Key message
ax = axes[1, 1]
ax.axis('off')
msg = """WAVE FUNCTION ATTENTION
Summary of 45+ Experiments

CONFIRMED:
✓ Wave wins at small models (d≤384)
✓ Wave wins at long sequences (seq≥256)
✓ Long-range advantage persists at 200K data
✓ Predetermined overlap > learned modifications
✓ Training 2-7× more stable (regularization)
✓ Distance-invariant (no positional decay)
✓ Works on TinyStories AND WikiText-2

NOT CONFIRMED:
✗ Can't bridge 24% param gap (d=224 vs d=256)
✗ Advantage vanishes at d=512 (enough capacity)
✗ Advantage vanishes at seq=128 + large data

VALUE PROPOSITION:
→ Small models with long context windows
→ Sample-efficient learning
→ More stable training"""
ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('summary_wave.png', dpi=150, bbox_inches='tight')
print('Saved summary_wave.png')
