#!/usr/bin/env python3
"""
Controlled 125M comparison: SMD vs Standard.

Same architecture, same data, same everything.
Only difference: 2 heads use energy minimization vs standard Q/K.

WikiText-103 streaming, BPE tokenizer, 2K steps each.
"""

import sys, time, math, json
import torch
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, '/home/asa/asa')
from smd_1b import SMD1B


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def get_data(tokenizer, seq_len=512):
    from datasets import load_dataset
    from torch.utils.data import IterableDataset, DataLoader

    class StreamDS(IterableDataset):
        def __init__(self, ds, tok, sl):
            self.ds, self.tok, self.sl = ds, tok, sl
        def __iter__(self):
            buf = []
            for ex in self.ds:
                t = ex.get('text', '')
                if len(t) < 50: continue
                buf.extend(self.tok.encode(t))
                while len(buf) >= self.sl + 1:
                    c = buf[:self.sl + 1]
                    buf = buf[self.sl:]
                    yield torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

    train = load_dataset("wikitext", "wikitext-103-raw-v1", split='train', streaming=True)
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split='validation')

    # Validation: non-streaming
    from torch.utils.data import Dataset
    class ValDS(Dataset):
        def __init__(self, tokens, sl):
            self.chunks = [tokens[i*sl:(i+1)*sl+1] for i in range(len(tokens)//(sl+1))]
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i):
            c = self.chunks[i]
            return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

    val_text = " ".join([t for t in val['text'] if t.strip()])
    val_tokens = tokenizer.encode(val_text)
    val_ds = ValDS(val_tokens, seq_len)

    train_dl = DataLoader(StreamDS(train, tokenizer, seq_len), batch_size=4, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    return train_dl, val_dl


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, tok = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.float16):
            out = model(x, y)
        nt = (y != -1).sum().item()
        total += out['loss'].item() * nt
        tok += nt
    avg = total / max(tok, 1)
    return avg, math.exp(min(avg, 100))


def train_model(name, model, train_dl, val_dl, device, max_steps=2000, lr=3e-4):
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name}", flush=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.0f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    warmup = 200
    def lr_fn(s):
        if s < warmup: return s / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(max_steps - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    scaler = GradScaler()
    grad_accum = 8

    model.train()
    step, accum_loss, accum_n = 0, 0, 0
    t0 = time.time()
    losses_log = []

    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.float16):
            loss = model(x, y)['loss'] / grad_accum
        scaler.scale(loss).backward()
        accum_loss += loss.item() * grad_accum
        accum_n += 1

        if accum_n >= grad_accum:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            step += 1
            avg = accum_loss / accum_n
            losses_log.append(avg)
            accum_loss, accum_n = 0, 0

            if step % 200 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"  Step {step}/{max_steps}: loss={avg:.4f} ppl={math.exp(min(avg,20)):.0f} "
                      f"({elapsed:.0f}s)", flush=True)

            if step % 500 == 0:
                vl, vp = evaluate(model, val_dl, device)
                print(f"  [EVAL] step={step}: val_loss={vl:.4f} val_ppl={vp:.1f}", flush=True)
                model.train()

            if step >= max_steps:
                break

    vl, vp = evaluate(model, val_dl, device)
    print(f"  FINAL: val_loss={vl:.4f} val_ppl={vp:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return vl, vp, params, losses_log


def main():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    tok = get_tokenizer()
    vocab = len(tok)
    seq_len = 512
    max_steps = 2000

    train_dl, val_dl = get_data(tok, seq_len)

    # Common config
    cfg = dict(vocab_size=vocab, d_model=768, n_layers=12, d_k=64, d_ff=2048,
               max_seq=seq_len, use_checkpointing=True, feature_dim=28)

    results = {}

    # 1. Standard (0 SMD heads, 12 standard, no wave gate)
    model_std = SMD1B(**cfg, n_smd=0, n_std=12, n_steps=4, use_wave_gate=False).to(device)
    vl, vp, p, log = train_model("Standard 12 heads", model_std, train_dl, val_dl, device, max_steps)
    results['standard'] = {'val_loss': vl, 'val_ppl': vp, 'params': p}
    del model_std; torch.cuda.empty_cache()

    # Need fresh data iterator
    train_dl, _ = get_data(tok, seq_len)

    # 2. SMD (2 SMD + 10 standard + wave gate)
    model_smd = SMD1B(**cfg, n_smd=2, n_std=10, n_steps=4, use_wave_gate=True).to(device)
    vl, vp, p, log = train_model("SMD 2smd+10std+gate", model_smd, train_dl, val_dl, device, max_steps)
    results['smd'] = {'val_loss': vl, 'val_ppl': vp, 'params': p}
    del model_smd; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("125M COMPARISON — RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:<25s}: PPL={r['val_ppl']:.1f} params={r['params']/1e6:.0f}M", flush=True)

    gap = results['smd']['val_ppl'] - results['standard']['val_ppl']
    pct = gap / results['standard']['val_ppl'] * 100
    print(f"\n  SMD vs Standard: {gap:+.1f} PPL ({pct:+.1f}%)", flush=True)
    print(f"  {'SMD WINS' if gap < 0 else 'Standard wins'}", flush=True)

    with open('/home/asa/asa/smd_125m_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
