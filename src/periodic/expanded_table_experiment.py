#!/usr/bin/env python3
"""
Expanded Periodic Table Experiment.

Tests whether the 6 new elements (28→34 dim) improve the combined model.
The new elements are: clause_complement_pref, semantic_weight, prep_selectivity,
noun_governability, frame_complexity, embedding_depth.

The wave experiment showed that LEARNED PROJECTIONS are essential.
More features = more information for the projections to combine.

Compare at d=256, 6L, 3 seeds, 10 epochs:
1. Standard baseline (8 std heads, zero features)
2. Combined 28-dim (1proj+7std+wavegate, original features)
3. Combined 34-dim (1proj+7std+wavegate, expanded features)
"""

import os, sys, time, math, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats

sys.path.insert(0, '/home/asa/asa')

from periodic_features import FEATURE_DIM, word_to_features  # now 34
from combined_model import CombinedLanguageModel, StandardLanguageModel


class WikiTextDataset(Dataset):
    def __init__(self, token_ids, features, seq_len):
        n = len(token_ids) // (seq_len + 1)
        self.cx, self.cf, self.cy = [], [], []
        for i in range(n):
            s = i * seq_len
            self.cx.append(token_ids[s:s+seq_len])
            self.cy.append(token_ids[s+1:s+seq_len+1])
            self.cf.append(features[s:s+seq_len])
    def __len__(self): return len(self.cx)
    def __getitem__(self, i):
        return (torch.tensor(self.cx[i], dtype=torch.long),
                torch.tensor(self.cf[i], dtype=torch.float32),
                torch.tensor(self.cy[i], dtype=torch.long))


def regenerate_features_34d():
    """Regenerate WikiText-2 features with 34-dim expanded periodic table."""
    cache_34 = '/home/asa/asa/cache/wikitext2_periodic_34d.npz'
    if os.path.exists(cache_34):
        print("Loading cached 34-dim features...", flush=True)
        data = np.load(cache_34, allow_pickle=True)
        return (data['train_ids'], data['val_ids'],
                data['train_feats'], data['val_feats'],
                int(data['vocab_size']))

    print("Regenerating features with 34-dim periodic table...", flush=True)

    # Load original cache for token IDs, POS tags, and word mappings
    orig = np.load('/home/asa/asa/cache/wikitext2_periodic.npz', allow_pickle=True)
    train_ids = orig['train_ids']
    val_ids = orig['val_ids']
    train_feats_28 = orig['train_feats']
    val_feats_28 = orig['val_feats']
    vocab_size = int(orig['vocab_size'])

    # We need to re-extract features for each token with the expanded 34-dim extractor.
    # But we don't have the original words/tags stored in the cache.
    # We need to load the tokenizer and POS info.

    # Alternative approach: copy dims 0-27 from original, compute dims 28-33 separately.
    # But dims 28-33 need word-level info we don't have from the cache.
    # The original cache was built with word_to_features() per word, then aligned to tokens.

    # Let's reload WikiText-2 and recompute all features from scratch.
    print("  Loading WikiText-2 from HuggingFace...", flush=True)
    from datasets import load_dataset
    from nltk import pos_tag

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Build vocabulary (same as original)
    from collections import Counter
    word_freq = Counter()
    for split in ['train', 'validation']:
        for row in ds[split]:
            text = row['text'].strip()
            if text and not text.startswith('='):
                word_freq.update(text.split())

    vocab = ['<pad>', '<unk>'] + [w for w, c in word_freq.most_common(vocab_size - 2)]
    word2id = {w: i for i, w in enumerate(vocab)}

    def process_split(split_name):
        all_ids = []
        all_feats = []
        texts = [row['text'].strip() for row in ds[split_name]
                 if row['text'].strip() and not row['text'].strip().startswith('=')]

        # Process in chunks for POS tagging
        chunk_size = 5000
        words_buf = []
        for text in texts:
            words_buf.extend(text.split())

        # POS tag in chunks
        print(f"  POS tagging {split_name} ({len(words_buf)} words)...", flush=True)
        all_tags = []
        for i in range(0, len(words_buf), chunk_size):
            chunk = words_buf[i:i+chunk_size]
            tagged = pos_tag(chunk)
            all_tags.extend([t for _, t in tagged])
            if i % 50000 == 0 and i > 0:
                print(f"    {i}/{len(words_buf)}", flush=True)

        # Extract features and IDs
        print(f"  Extracting 34-dim features...", flush=True)
        ids = np.array([word2id.get(w, 1) for w in words_buf], dtype=np.int64)
        feats = np.stack([word_to_features(w, t) for w, t in zip(words_buf, all_tags)])

        return ids, feats

    train_ids_new, train_feats_new = process_split('train')
    val_ids_new, val_feats_new = process_split('validation')

    print(f"  Train: {len(train_ids_new)} tokens, feats shape {train_feats_new.shape}", flush=True)
    print(f"  Val: {len(val_ids_new)} tokens, feats shape {val_feats_new.shape}", flush=True)

    # Save cache
    np.savez_compressed(cache_34,
                        train_ids=train_ids_new, val_ids=val_ids_new,
                        train_feats=train_feats_new, val_feats=val_feats_new,
                        vocab_size=vocab_size)
    print(f"  Saved to {cache_34}", flush=True)

    return train_ids_new, val_ids_new, train_feats_new, val_feats_new, vocab_size


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total, n = 0, 0
    for x, f, y in loader:
        x, f, y = x.to(device), f.to(device), y.to(device)
        loss = model(x, f, y)['loss']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, tok = 0, 0
    for x, f, y in loader:
        x, f, y = x.to(device), f.to(device), y.to(device)
        out = model(x, f, y)
        nt = (y != -1).sum().item()
        total += out['loss'].item() * nt
        tok += nt
    avg = total / max(tok, 1)
    return avg, math.exp(min(avg, 100))


def run(name, model_fn, train_ds, val_ds, bs, epochs, lr, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = model_fn().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] seed={seed}, params={params:,}", flush=True)

    tdl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    vdl = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=True, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = len(tdl) * epochs
    warmup = min(500, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
        s/warmup if s < warmup else 0.5*(1+math.cos(math.pi*(s-warmup)/max(total_steps-warmup,1))))

    best_loss, best_ppl = float('inf'), float('inf')
    for ep in range(1, epochs+1):
        t0 = time.time()
        tl = train_epoch(model, tdl, opt, sched, device)
        vl, vp = evaluate(model, vdl, device)
        if vl < best_loss: best_loss, best_ppl = vl, vp
        print(f"  Epoch {ep}: train={tl:.4f} val={vl:.4f} ppl={vp:.1f} best={best_ppl:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return best_loss, best_ppl, params


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    # Load 28-dim features (original)
    orig = np.load('/home/asa/asa/cache/wikitext2_periodic.npz', allow_pickle=True)
    train_ids_28 = orig['train_ids']
    val_ids_28 = orig['val_ids']
    train_feats_28 = orig['train_feats']
    val_feats_28 = orig['val_feats']
    vocab_size = int(orig['vocab_size'])

    # Load/regenerate 34-dim features
    train_ids_34, val_ids_34, train_feats_34, val_feats_34, _ = regenerate_features_34d()

    seq_len, bs, epochs, lr = 256, 32, 10, 5e-4
    seeds = [42, 123, 7]

    ds_28 = WikiTextDataset(train_ids_28, train_feats_28, seq_len)
    ds_28v = WikiTextDataset(val_ids_28, val_feats_28, seq_len)
    ds_34 = WikiTextDataset(train_ids_34, train_feats_34, seq_len)
    ds_34v = WikiTextDataset(val_ids_34, val_feats_34, seq_len)
    ds_zero = WikiTextDataset(train_ids_28, np.zeros_like(train_feats_28), seq_len)
    ds_zerov = WikiTextDataset(val_ids_28, np.zeros_like(val_feats_28), seq_len)

    print(f"Data: {len(ds_28)} train, {len(ds_28v)} val", flush=True)

    d, L, dk, ff = 256, 6, 32, 512

    configs = {
        'standard': {
            'desc': 'Standard 8 heads (baseline)',
            'fn': lambda: StandardLanguageModel(vocab_size, d, L, 8, dk, ff, seq_len),
            'ds': (ds_zero, ds_zerov),
        },
        'combined_28d': {
            'desc': 'Combined 1p+7s+gate (28-dim)',
            'fn': lambda: CombinedLanguageModel(vocab_size, d, L, 1, 7, 28, dk, ff, seq_len, use_wave_gate=True),
            'ds': (ds_28, ds_28v),
        },
        'combined_34d': {
            'desc': 'Combined 1p+7s+gate (34-dim, expanded table)',
            'fn': lambda: CombinedLanguageModel(vocab_size, d, L, 1, 7, 34, dk, ff, seq_len, use_wave_gate=True),
            'ds': (ds_34, ds_34v),
        },
    }

    results = {}
    for cfg_name, cfg in configs.items():
        print(f"\n{'='*60}", flush=True)
        print(f"{cfg['desc']}", flush=True)
        print(f"{'='*60}", flush=True)

        losses, ppls, params = [], [], None
        tds, vds = cfg['ds']
        for seed in seeds:
            print(f"\n--- {cfg_name} seed={seed} ---", flush=True)
            loss, ppl, p = run(cfg_name, cfg['fn'], tds, vds, bs, epochs, lr, seed, device)
            losses.append(loss)
            ppls.append(ppl)
            params = p
            print(f"  RESULT: loss={loss:.4f} ppl={ppl:.1f}", flush=True)

        results[cfg_name] = {
            'desc': cfg['desc'],
            'losses': losses, 'ppls': ppls,
            'mean_loss': float(np.mean(losses)), 'mean_ppl': float(np.mean(ppls)),
            'std_ppl': float(np.std(ppls)), 'params': params,
        }

        with open('/home/asa/asa/expanded_table_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    std = results['standard']['losses']
    print(f"\n{'='*60}", flush=True)
    print("EXPANDED PERIODIC TABLE — RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<40s} {'PPL':>10s} {'Params':>12s} {'vs Std':>10s} {'p':>10s}", flush=True)
    print("─" * 85, flush=True)

    for name, r in results.items():
        t, p = stats.ttest_ind(r['losses'], std)
        gap = (r['mean_ppl'] - results['standard']['mean_ppl']) / results['standard']['mean_ppl'] * 100
        sig = ' ★' if p < 0.05 else ''
        print(f"{r['desc']:<40s} {r['mean_ppl']:>8.1f}±{r['std_ppl']:.1f} {r['params']:>10,} {gap:>+8.1f}% {p:>9.4f}{sig}", flush=True)

    # Compare 28d vs 34d directly
    t28v34, p28v34 = stats.ttest_ind(results['combined_28d']['losses'], results['combined_34d']['losses'])
    gap_34v28 = results['combined_34d']['mean_ppl'] - results['combined_28d']['mean_ppl']
    print(f"\n34-dim vs 28-dim: PPL gap={gap_34v28:+.1f}, t={t28v34:.3f}, p={p28v34:.4f}", flush=True)
    print(f"{'34-dim WINS' if gap_34v28 < 0 else '28-dim is sufficient'}", flush=True)


if __name__ == '__main__':
    main()
