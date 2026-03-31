#!/usr/bin/env python3
"""
SMD Fine-tuning v2: Simpler approach.

Instead of hooks, just run SMD as a parallel attention path
and add its output after each forward pass.

Steps:
1. Load GPT-2-large, evaluate baseline PPL
2. Add SMD adapter modules, train only those (100 steps)
3. Generate text and compare
"""

import sys, time, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/asa/asa')


class SMDAdapter(nn.Module):
    """Lightweight SMD adapter that runs alongside a transformer.

    Doesn't modify the base model at all. Just processes the output
    of each selected layer and adds a small structural correction.
    """
    def __init__(self, d_model, n_layers_to_adapt=4, n_steps=4):
        super().__init__()
        self.d_k = 64
        self.n_steps = n_steps

        # One SMD module per adapted layer
        self.layers = nn.ModuleList()
        for _ in range(n_layers_to_adapt):
            self.layers.append(nn.ModuleDict({
                'feat': nn.Sequential(
                    nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 28), nn.Sigmoid()),
                'wq': nn.Linear(28, self.d_k, bias=False),
                'wk': nn.Linear(28, self.d_k, bias=False),
                'charge': nn.Linear(28, 1),
                'v': nn.Linear(d_model, self.d_k, bias=False),
                'out': nn.Linear(self.d_k, d_model, bias=False),
            }))

        # Initialize output to near-zero
        for layer in self.layers:
            nn.init.zeros_(layer['out'].weight)

        self.step_size = nn.Parameter(torch.tensor(0.5))
        self.charge_decay = nn.Parameter(torch.tensor(0.3))

    def forward_layer(self, hidden, layer_idx):
        """Apply SMD to one layer's output."""
        mod = self.layers[layer_idx]
        B, N, D = hidden.shape

        features = mod['feat'](hidden.detach())
        Q = mod['wq'](features)
        K = mod['wk'](features)
        compat = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        charge = torch.sigmoid(mod['charge'](features)).squeeze(-1)

        causal = torch.tril(torch.ones(N, N, device=hidden.device, dtype=torch.bool))
        logits = compat.masked_fill(~causal.unsqueeze(0), float('-inf'))

        for _ in range(self.n_steps):
            attn = F.softmax(logits, dim=-1)
            received = attn.sum(dim=-2)
            charge = charge * (1 - self.charge_decay * torch.sigmoid(received - 1))
            energy = -compat * charge.unsqueeze(-1) * charge.unsqueeze(-2)
            logits = (logits - self.step_size * energy).masked_fill(~causal.unsqueeze(0), float('-inf'))

        attn = F.softmax(logits, dim=-1)
        V = mod['v'](hidden)
        out = mod['out'](torch.matmul(attn, V))
        return out * 0.1  # small contribution


class SMDWrappedModel(nn.Module):
    """Wraps a HuggingFace model with SMD adapters."""
    def __init__(self, base_model, adapt_layers=None):
        super().__init__()
        self.base = base_model
        n_layers = len(base_model.transformer.h)
        d_model = base_model.config.n_embd

        if adapt_layers is None:
            adapt_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4]
        self.adapt_layers = adapt_layers

        self.adapter = SMDAdapter(d_model, len(adapt_layers))
        self.adapt_layer_set = set(adapt_layers)

    def forward(self, input_ids, labels=None):
        # Manual forward through transformer blocks
        x = self.base.transformer.wte(input_ids) + \
            self.base.transformer.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
        x = self.base.transformer.drop(x)

        adapt_idx = 0
        for i, block in enumerate(self.base.transformer.h):
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out
            if x.dim() == 2:
                x = x.unsqueeze(0)  # restore batch dim if squeezed
            if i in self.adapt_layer_set:
                x = x + self.adapter.forward_layer(x, adapt_idx)
                adapt_idx += 1

        x = self.base.transformer.ln_f(x)
        logits = self.base.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1), ignore_index=-1)

        return type('Output', (), {'loss': loss, 'logits': logits})()

    def generate(self, input_ids, max_new_tokens=60, temperature=0.8, top_p=0.9,
                 pad_token_id=None):
        """Simple autoregressive generation."""
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.forward(ids[:, -1024:])  # GPT-2 max 1024
                logits = out.logits[:, -1, :] / temperature

                # Top-p sampling
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float('-inf')
                probs = F.softmax(sorted_logits, dim=-1)
                next_token = sorted_idx.gather(-1, torch.multinomial(probs, 1))

                ids = torch.cat([ids, next_token], dim=-1)
                if next_token.item() == pad_token_id:
                    break
        return ids


def main():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    from torch.cuda.amp import autocast, GradScaler

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    seq_len = 512

    # Validation data
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split='validation')
    val_text = " ".join([t for t in val['text'] if t.strip()])
    val_tokens = tokenizer.encode(val_text)

    class ValDS(Dataset):
        def __init__(self, tokens, sl):
            self.chunks = [tokens[i*sl:(i+1)*sl+1] for i in range(len(tokens)//(sl+1))]
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i):
            c = self.chunks[i]
            return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

    val_dl = DataLoader(ValDS(val_tokens, seq_len), batch_size=4, shuffle=False)

    # Training data (streaming)
    class StreamDS(IterableDataset):
        def __init__(self, tok, sl):
            self.tok, self.sl = tok, sl
            self.ds = load_dataset("wikitext", "wikitext-103-raw-v1", split='train', streaming=True)
        def __iter__(self):
            buf = []
            for ex in self.ds:
                t = ex.get('text', '')
                if len(t) < 50: continue
                buf.extend(self.tok.encode(t))
                while len(buf) >= self.sl + 1:
                    c = buf[:self.sl + 1]; buf = buf[self.sl:]
                    yield torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        total, tok = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(dtype=torch.float16):
                out = model(x, labels=y)
            nt = (y != -1).sum().item()
            total += out.loss.item() * nt; tok += nt
        return total / max(tok, 1), math.exp(min(total/max(tok,1), 100))

    prompts = [
        "The scientist discovered that",
        "In the beginning of the 21st century,",
        "The fundamental problem with current AI is",
        "Language has structure because",
    ]

    def generate(model, label):
        model.eval()
        print(f"\n  [{label}] Generated text:", flush=True)
        for p in prompts:
            ids = tokenizer.encode(p, return_tensors='pt').to(device)
            if hasattr(model, 'generate'):
                out_ids = model.generate(ids, max_new_tokens=50, temperature=0.8,
                                         top_p=0.9, pad_token_id=tokenizer.eos_token_id)
            else:
                out_ids = model.base.generate(ids, max_new_tokens=50, do_sample=True,
                                               temperature=0.8, top_p=0.9,
                                               pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            print(f"    {text[:200]}", flush=True)

    results = {}

    # ── 1. Baseline GPT-2-large ──
    print("Loading GPT-2-large...", flush=True)
    base = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
    vl, vp = evaluate(base, val_dl)
    print(f"\nBaseline GPT-2-large: val_ppl={vp:.1f}", flush=True)
    results['baseline'] = {'val_ppl': float(vp)}
    generate(base, "Baseline GPT-2-large")

    # ── 2. GPT-2-large + SMD adapter (train adapter only) ──
    print(f"\n{'='*60}", flush=True)
    print("Adding SMD adapter and training...", flush=True)

    model_smd = SMDWrappedModel(base, adapt_layers=[0, 9, 18, 27]).to(device)

    # Freeze base
    for p in model_smd.base.parameters():
        p.requires_grad = False
    for p in model_smd.adapter.parameters():
        p.requires_grad = True

    adapter_params = sum(p.numel() for p in model_smd.adapter.parameters())
    print(f"  Adapter params: {adapter_params/1e3:.0f}K (base: 774M frozen)", flush=True)

    train_dl = DataLoader(StreamDS(tokenizer, seq_len), batch_size=2, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model_smd.adapter.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = GradScaler()
    grad_accum = 4

    model_smd.train()
    step, accum_loss, accum_n = 0, 0, 0
    max_steps = 300
    t0 = time.time()

    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.float16):
            out = model_smd(x, labels=y)
            loss = out.loss / grad_accum
        scaler.scale(loss).backward()
        accum_loss += loss.item() * grad_accum
        accum_n += 1

        if accum_n >= grad_accum:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model_smd.adapter.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            step += 1
            avg = accum_loss / accum_n
            accum_loss, accum_n = 0, 0

            if step % 50 == 0 or step == 1:
                print(f"  Step {step}/{max_steps}: loss={avg:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if step >= max_steps:
                break

    vl, vp = evaluate(model_smd, val_dl)
    print(f"\n  GPT-2-large + SMD adapter: val_ppl={vp:.1f}", flush=True)
    results['smd_adapter'] = {'val_ppl': float(vp)}
    generate(model_smd, "GPT-2-large + SMD")

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:<25s}: val_ppl={r['val_ppl']:.1f}", flush=True)

    diff = results['smd_adapter']['val_ppl'] - results['baseline']['val_ppl']
    print(f"\n  SMD adapter effect: {diff:+.1f} PPL", flush=True)

    with open('/home/asa/asa/smd_finetune_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
