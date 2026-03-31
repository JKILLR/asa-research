"""Final summary: 119 wave function experiments."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: 2x compression cost by d_model and seq_len
ax = axes[0, 0]
# Data: (d_model, seq_len, 2x_cost, validated)
points = [
    (256, 256, -0.205, False, '1L'),
    (512, 128, -0.188, False, '1L'),
    (512, 256, -0.089, False, '1L'),
    (512, 512, -0.038, True, '1L 3-trial'),
    (1024, 256, -0.223, False, '2L'),
]

for d, s, cost, val, label in points:
    color = '#4CAF50' if abs(cost) < 0.05 else '#FF9800' if abs(cost) < 0.15 else '#F44336'
    marker = 's' if val else 'o'
    ax.scatter(d, s, c=color, s=200 if val else 120, marker=marker, edgecolors='black', linewidth=1, zorder=3)
    ax.annotate(f'{cost:+.3f}\n{label}', (d+20, s+15), fontsize=8)

ax.set_xlabel('d_model', fontsize=11)
ax.set_ylabel('seq_len', fontsize=11)
ax.set_title('2× Compression Quality Cost\n(green=small, orange=moderate, red=large)', fontsize=12)
ax.set_xlim(200, 1100)
ax.set_ylim(80, 600)
ax.grid(alpha=0.3)

# Panel 2: Compression vs quality (Pareto frontier)
ax = axes[0, 1]
# All validated multi-trial results
results = [
    (1.19, +0.178, 't=9.05', 'd=256 6L'),
    (1.26, +0.028, 't=2.18', 'd=384 6L'),
    (1.50, +0.091, 't=6.12', 'd=256 1L'),
    (1.59, -0.014, 't=-0.70', 'd=224 3L'),
    (1.98, -0.038, 't=-9.82', 'd=512 1L'),
    (1.98, -0.023, 't=-7.08', 'WikiText-2'),
]

for comp, gap, t, label in results:
    color = '#4CAF50' if gap > 0 else '#FF9800' if gap > -0.03 else '#2196F3'
    ax.scatter(comp, gap, c=color, s=150, edgecolors='black', linewidth=1, zorder=3)
    ax.annotate(f'{label}\n{t}', (comp+0.02, gap+0.01), fontsize=7)

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Compression Ratio (×)', fontsize=11)
ax.set_ylabel('Quality Gap (positive = wave better)', fontsize=11)
ax.set_title('Multi-Trial Validated Results\n(Pareto frontier of compression)', fontsize=12)
ax.grid(alpha=0.3)

# Panel 3: The mechanism
ax = axes[1, 0]
ax.axis('off')
mech = """THE KEY MECHANISM: Wave-Gated FFN

Standard:    FFN(x) = W₂(GELU(W₁(x)))
Wave-Gated:  FFN(x) = W₂(GELU(W₁(x)) ⊙ σ(W_g(ψ)))

Where ψ = wave function from POS + features.

The gate σ(W_g(ψ)) tells each neuron when to fire
based on linguistic type:
  • Nouns → activate semantic feature neurons
  • Verbs → activate argument structure neurons
  • Determiners → activate noun-phrase neurons

This routing replaces what standard transformers
learn through DEPTH — multiple attention layers
build up the same type-specific processing that
the wave gate provides directly.

Result: 1 wave-gated layer ≈ 6 standard layers"""

ax.text(0.05, 0.95, mech, transform=ax.transAxes, fontsize=9.5,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 4: Summary stats
ax = axes[1, 1]
ax.axis('off')
summary = """119 EXPERIMENTS SUMMARY

DATASETS: TinyStories, WikiText-2
SCALES: d=128, 192, 224, 256, 320, 352,
        384, 448, 464, 512, 1024
DEPTHS: 1L, 2L, 3L, 4L, 6L
SEQ: 128, 256, 512

STATISTICALLY VALIDATED:
 ✓ 1.19× better quality (t=9.05, 5/5)
 ✓ 1.50× better quality (t=6.12, 5/5)
 ✓ 1.59× tied (t=-0.70, n.s.)
 ✓ 2.0× costs 0.02-0.04 across datasets

WHAT WORKS:
 • Wave-gated FFN (2.3× more than attention)
 • Layer reduction (1L matches 6L)
 • Contextual wave (POS bigrams)
 • Long sequences (lower cost at seq≥256)

WHAT DOESN'T:
 × 2× free at d=1024 (-0.22 cost)
 × Learned projections on wave overlap
 × 4× FFN reduction on WikiText-2"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.suptitle('Wave Function Attention: 119 Experiments\nAtomic Semantic Architecture (ASA)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('final_summary.png', dpi=150, bbox_inches='tight')
print('Saved final_summary.png')
