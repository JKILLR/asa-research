#!/usr/bin/env python3
"""
Wave Function Transformer Experiment.

Tests the article's core claim: can raw periodic feature overlap
replace learned Q/K attention?

Configs (all d=256, 6L, dk=32, ff=512):
1. Standard: 8 standard heads (baseline)
2. Combined: 1 proj + 7 std (our proven approach)
3. Wave-4+4: 4 wave + 4 std (hybrid)
4. Wave-8: 8 pure wave heads (the radical test)
5. Direct-4+4: 4 direct overlap + 4 std (even purer)

WikiText-2, 3 seeds, 10 epochs.
"""

import os, sys, time, math, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats

sys.path.insert(0, '/home/asa/asa')

from wave_transformer import WaveTransformerLM, StandardTransformerLM
from periodic_features import FEATURE_DIM


class WikiTextDataset(Dataset):
    def __init__(self, token_ids, features, seq_len):
        n_chunks = len(token_ids) // (seq_len + 1)
        self.chunks_x, self.chunks_f, self.chunks_y = [], [], []
        for i in range(n_chunks):
            s = i * seq_len
            self.chunks_x.append(token_ids[s:s + seq_len])
            self.chunks_y.append(token_ids[s + 1:s + seq_len + 1])
            self.chunks_f.append(features[s:s + seq_len])

    def __len__(self):
        return len(self.chunks_x)

    def __getitem__(self, idx):
        return (torch.tensor(self.chunks_x[idx], dtype=torch.long),
                torch.tensor(self.chunks_f[idx], dtype=torch.float32),
                torch.tensor(self.chunks_y[idx], dtype=torch.long))


def load_data():
    data = np.load('/home/asa/asa/cache/wikitext2_periodic.npz', allow_pickle=True)
    return (data['train_ids'], data['val_ids'],
            data['train_feats'], data['val_feats'],
            int(data['vocab_size']))


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, n = 0, 0
    for x, f, y in loader:
        x, f, y = x.to(device), f.to(device), y.to(device)
        loss = model(x, f, y)['loss']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tok = 0, 0
    for x, f, y in loader:
        x, f, y = x.to(device), f.to(device), y.to(device)
        loss = model(x, f, y)['loss']
        nt = (y != -1).sum().item()
        total_loss += loss.item() * nt
        total_tok += nt
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 100))


def run(name, model_fn, train_ds, val_ds, batch_size, epochs, lr, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = model_fn().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] seed={seed}, params={n_params:,}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                         drop_last=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       drop_last=True, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = len(train_dl) * epochs
    warmup = min(500, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:
        s / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total_steps - warmup, 1))))

    best_loss, best_ppl = float('inf'), float('inf')
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl = train_epoch(model, train_dl, opt, sched, device)
        vl, vp = evaluate(model, val_dl, device)
        if vl < best_loss: best_loss, best_ppl = vl, vp
        print(f"  Epoch {ep}: train={tl:.4f} val={vl:.4f} ppl={vp:.1f} best={best_ppl:.1f} ({time.time()-t0:.0f}s)", flush=True)

    return best_loss, best_ppl, n_params


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    train_ids, val_ids, train_feats, val_feats, vocab_size = load_data()
    seq_len, batch_size, epochs, lr = 256, 32, 10, 5e-4
    seeds = [42, 123, 7]

    train_ds = WikiTextDataset(train_ids, train_feats, seq_len)
    val_ds = WikiTextDataset(val_ids, val_feats, seq_len)
    train_ds_zero = WikiTextDataset(train_ids, np.zeros_like(train_feats), seq_len)
    val_ds_zero = WikiTextDataset(val_ids, np.zeros_like(val_feats), seq_len)

    print(f"Data: {len(train_ds)} train, {len(val_ds)} val", flush=True)

    d, L, dk, ff = 256, 6, 32, 512

    configs = {
        'standard': {
            'desc': 'Standard 8 heads (baseline)',
            'fn': lambda: StandardTransformerLM(vocab_size, d, L, 8, dk, ff, seq_len),
            'ds': (train_ds_zero, val_ds_zero),
        },
        'combined_1p7s': {
            'desc': 'Combined 1proj+7std+wavegate',
            'fn': lambda: WaveTransformerLM(vocab_size, d, L, 0, 1, 7, FEATURE_DIM, dk, ff, seq_len,
                                             use_wave_gate=True),
            'ds': (train_ds, val_ds),
        },
        'wave_4w4s': {
            'desc': 'Wave 4wave+4std+wavegate',
            'fn': lambda: WaveTransformerLM(vocab_size, d, L, 4, 0, 4, FEATURE_DIM, dk, ff, seq_len,
                                             use_wave_gate=True),
            'ds': (train_ds, val_ds),
        },
        'wave_2w2p4s': {
            'desc': 'Wave 2wave+2proj+4std+wavegate',
            'fn': lambda: WaveTransformerLM(vocab_size, d, L, 2, 2, 4, FEATURE_DIM, dk, ff, seq_len,
                                             use_wave_gate=True),
            'ds': (train_ds, val_ds),
        },
        'wave_8w_pure': {
            'desc': 'Wave 8wave pure (0 Q/K params) + wavegate',
            'fn': lambda: WaveTransformerLM(vocab_size, d, L, 8, 0, 0, FEATURE_DIM, dk, ff, seq_len,
                                             use_wave_gate=True),
            'ds': (train_ds, val_ds),
        },
        'direct_4w4s': {
            'desc': 'Direct overlap 4wave+4std+wavegate',
            'fn': lambda: WaveTransformerLM(vocab_size, d, L, 4, 0, 4, FEATURE_DIM, dk, ff, seq_len,
                                             use_wave_gate=True, use_direct_overlap=True),
            'ds': (train_ds, val_ds),
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
            loss, ppl, p = run(cfg_name, cfg['fn'], tds, vds, batch_size, epochs, lr, seed, device)
            losses.append(loss)
            ppls.append(ppl)
            params = p
            print(f"  RESULT: loss={loss:.4f} ppl={ppl:.1f}", flush=True)

        results[cfg_name] = {
            'desc': cfg['desc'],
            'losses': losses,
            'ppls': ppls,
            'mean_loss': float(np.mean(losses)),
            'mean_ppl': float(np.mean(ppls)),
            'std_ppl': float(np.std(ppls)),
            'params': params,
        }

        with open('/home/asa/asa/wave_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Final summary
    std_losses = results['standard']['losses']
    print(f"\n{'='*60}", flush=True)
    print("WAVE FUNCTION TRANSFORMER — RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<30s} {'PPL':>10s} {'Params':>12s} {'vs Std':>10s} {'p':>10s}", flush=True)
    print("─" * 75, flush=True)

    for name, r in results.items():
        t, p = stats.ttest_ind(r['losses'], std_losses)
        gap_pct = (r['mean_ppl'] - results['standard']['mean_ppl']) / results['standard']['mean_ppl'] * 100
        sig = ' ★' if p < 0.05 else ''
        print(f"{r['desc']:<30s} {r['mean_ppl']:>8.1f}±{r['std_ppl']:.1f} {r['params']:>10,} "
              f"{gap_pct:>+8.1f}% {p:>9.4f}{sig}", flush=True)


if __name__ == '__main__':
    main()
