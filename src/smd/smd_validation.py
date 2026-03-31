#!/usr/bin/env python3
"""
SMD Validation: d=512 scale + n_steps sweep.

Two experiments run sequentially:
1. d=512, 5 seeds, 10 epochs: SMD 2+6 vs Combined vs Standard
2. n_steps sweep at d=256, 3 seeds: 2, 4, 8, 16 steps
"""

import os, sys, time, math, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats

sys.path.insert(0, '/home/asa/asa')

from smd_attention import SMDLM
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


def run_configs(name, configs, seeds, bs, epochs, lr, device):
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
            'desc': cfg['desc'], 'losses': losses, 'ppls': ppls,
            'mean_loss': float(np.mean(losses)), 'mean_ppl': float(np.mean(ppls)),
            'std_ppl': float(np.std(ppls)), 'params': params,
        }
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    data = np.load('/home/asa/asa/cache/wikitext2_periodic.npz', allow_pickle=True)
    train_ids, val_ids = data['train_ids'], data['val_ids']
    train_feats, val_feats = data['train_feats'], data['val_feats']
    vocab_size = int(data['vocab_size'])
    fdim = 28
    seq_len = 256

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: d=512 validation, 5 seeds
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}", flush=True)
    print("EXPERIMENT 1: d=512 VALIDATION (5 seeds, 10 epochs)", flush=True)
    print(f"{'#'*60}", flush=True)

    d, L, dk, ff = 512, 6, 64, 2048
    bs, epochs, lr = 32, 10, 5e-4
    seeds5 = [42, 123, 7, 99, 2024]

    train_ds = WikiTextDataset(train_ids, train_feats, seq_len)
    val_ds = WikiTextDataset(val_ids, val_feats, seq_len)
    train_ds_zero = WikiTextDataset(train_ids, np.zeros_like(train_feats), seq_len)
    val_ds_zero = WikiTextDataset(val_ids, np.zeros_like(val_feats), seq_len)

    configs_512 = {
        'standard_512': {
            'desc': f'Standard 8 heads (d={d})',
            'fn': lambda: StandardLanguageModel(vocab_size, d, L, 8, dk, ff, seq_len),
            'ds': (train_ds_zero, val_ds_zero),
        },
        'combined_512': {
            'desc': f'Combined 1p+7s+gate (d={d})',
            'fn': lambda: CombinedLanguageModel(vocab_size, d, L, 1, 7, fdim, dk, ff, seq_len, use_wave_gate=True),
            'ds': (train_ds, val_ds),
        },
        'smd_512': {
            'desc': f'SMD 2smd+6std+gate (d={d}, 4 steps)',
            'fn': lambda: SMDLM(vocab_size, d, L, 2, 6, fdim, dk, ff, seq_len, n_steps=4, use_wave_gate=True),
            'ds': (train_ds, val_ds),
        },
    }

    results_512 = run_configs("d512", configs_512, seeds5, bs, epochs, lr, device)
    all_results['d512'] = results_512

    std512 = results_512['standard_512']['losses']
    print(f"\n{'='*60}", flush=True)
    print("d=512 RESULTS:", flush=True)
    for name, r in results_512.items():
        t, p = stats.ttest_ind(r['losses'], std512)
        gap = (r['mean_ppl'] - results_512['standard_512']['mean_ppl']) / results_512['standard_512']['mean_ppl'] * 100
        sig = ' ★' if p < 0.05 else ''
        print(f"  {r['desc']:<40s} PPL={r['mean_ppl']:.1f}±{r['std_ppl']:.1f} {r['params']:,} {gap:+.1f}% p={p:.4f}{sig}", flush=True)

    # SMD vs Combined at d=512
    t, p = stats.ttest_ind(results_512['smd_512']['losses'], results_512['combined_512']['losses'])
    gap = results_512['smd_512']['mean_ppl'] - results_512['combined_512']['mean_ppl']
    print(f"\n  SMD vs Combined at d=512: gap={gap:+.1f} PPL, p={p:.4f}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: n_steps sweep at d=256
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'#'*60}", flush=True)
    print("EXPERIMENT 2: N_STEPS SWEEP (d=256, 3 seeds, 10 epochs)", flush=True)
    print(f"{'#'*60}", flush=True)

    d2, L2, dk2, ff2 = 256, 6, 32, 512
    seeds3 = [42, 123, 7]

    configs_steps = {}
    for n_steps in [2, 4, 8, 16]:
        ns = n_steps
        configs_steps[f'smd_steps{ns}'] = {
            'desc': f'SMD 2+6 ({ns} steps)',
            'fn': lambda ns=ns: SMDLM(vocab_size, d2, L2, 2, 6, fdim, dk2, ff2, seq_len, n_steps=ns, use_wave_gate=True),
            'ds': (train_ds, val_ds),
        }

    results_steps = run_configs("steps", configs_steps, seeds3, bs, epochs, lr, device)
    all_results['n_steps'] = results_steps

    print(f"\n{'='*60}", flush=True)
    print("N_STEPS RESULTS:", flush=True)
    for name, r in sorted(results_steps.items()):
        print(f"  {r['desc']:<25s} PPL={r['mean_ppl']:.1f}±{r['std_ppl']:.1f} {r['params']:,}", flush=True)

    # Save everything
    with open('/home/asa/asa/smd_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to smd_validation_results.json", flush=True)


if __name__ == '__main__':
    main()
