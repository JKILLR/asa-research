"""Data scaling study: does wave advantage vanish with more data?

Runs wave (2w sym-late at d=512, 2w all at d=256) vs standard
at 50K, 100K, 200K TinyStories. Cleans cache between sizes.
"""
import sys
import os
import shutil
import json
sys.argv = ['']

from train_wave import LMDataset, train_model
import torch

device = 'cuda'
CACHE_DIR = "cache"

results = {}

for d_model, d_ff, wave_layers, label_prefix in [
    (256, 512, "all", "d256"),
    (512, 1024, "late", "d512"),
]:
    for n_samples in [50000, 100000, 200000]:
        tag = f"{label_prefix}_n{n_samples//1000}K"
        print(f"\n{'#'*70}")
        print(f"# {tag}: d={d_model}, {n_samples//1000}K stories")
        print(f"{'#'*70}")

        # Clean cache to save disk
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print("Cleaned cache")

        # Load dataset
        train_ds = LMDataset('tinystories', 'train', 128, n_samples)
        val_ds = LMDataset('tinystories', 'validation', 128, n_samples,
                           shared_word2id=train_ds.word2id)

        # Standard
        std_r, _, _ = train_model(
            'standard', 'tinystories', 1, device,
            d_model, 8, 6, d_ff, 0, 5e-5, 16, 128, n_samples,
            train_ds, val_ds
        )

        # Wave
        wave_r, _, _ = train_model(
            'wave', 'tinystories', 1, device,
            d_model, 8, 6, d_ff, 2, 5e-5, 16, 128, n_samples,
            train_ds, val_ds, wave_layers=wave_layers
        )

        std_val = std_r['epochs'][-1]['val_loss']
        wave_val = wave_r['epochs'][-1]['val_loss']
        gap = std_val - wave_val  # positive = wave wins

        results[tag] = {
            'std_val': std_val,
            'wave_val': wave_val,
            'gap': gap,
            'std_params': std_r['n_params'],
            'wave_params': wave_r['n_params'],
            'n_samples': n_samples,
            'd_model': d_model,
        }

        print(f"\n>>> {tag}: std={std_val:.4f} wave={wave_val:.4f} gap={gap:+.4f}")

# Final summary
print("\n" + "=" * 70)
print("DATA SCALING STUDY — COMPLETE RESULTS")
print("=" * 70)
print(f"{'Tag':>16} {'Std Val':>10} {'Wave Val':>10} {'Gap':>10} {'Winner':>8}")
print("-" * 60)

for tag, r in sorted(results.items()):
    winner = "WAVE" if r['gap'] > 0 else "Std"
    print(f"{tag:>16} {r['std_val']:>10.4f} {r['wave_val']:>10.4f} {r['gap']:>+10.4f} {winner:>8}")

# Analysis
print("\n--- ANALYSIS ---")
for prefix in ["d256", "d512"]:
    print(f"\n{prefix}:")
    tags = sorted([t for t in results if t.startswith(prefix)])
    gaps = [results[t]['gap'] for t in tags]
    sizes = [results[t]['n_samples']//1000 for t in tags]
    for t, g, s in zip(tags, gaps, sizes):
        print(f"  {s}K: gap={g:+.4f}")

    if len(gaps) >= 2:
        if gaps[-1] < gaps[0]:
            print(f"  TREND: gap SHRINKING ({gaps[0]:+.4f} → {gaps[-1]:+.4f})")
            if gaps[-1] <= 0:
                print(f"  >>> Wave advantage VANISHES at scale")
            else:
                print(f"  >>> Wave advantage DIMINISHES but PERSISTS")
        else:
            print(f"  TREND: gap GROWING ({gaps[0]:+.4f} → {gaps[-1]:+.4f})")
            print(f"  >>> Wave advantage INCREASES with data!")

# Save
with open("results_data_scaling.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results_data_scaling.json")
