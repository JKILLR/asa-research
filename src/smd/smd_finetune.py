#!/usr/bin/env python3
"""
Fine-tune GPT-2-large with SMD attention heads.

Takes a pretrained model, replaces 2 of its attention heads with
SMD energy minimization heads, fine-tunes briefly, compares quality.

Two models fine-tuned on same data:
1. GPT-2-large vanilla (baseline fine-tune)
2. GPT-2-large + SMD heads (2 SMD replacing 2 standard)

Then generates text from both to compare coherence.
"""

import sys, time, math, json, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, '/home/asa/asa')


class SMDHeadInjection(nn.Module):
    """SMD head that operates alongside GPT-2's existing attention.

    Instead of replacing the full attention layer, we ADD an SMD head
    that provides a structural attention signal mixed into the output.
    """
    def __init__(self, d_model, n_steps=4):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // 16  # small head dim
        self.n_steps = n_steps

        # Feature predictor: embedding → structural features
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 28),
            nn.Sigmoid(),
        )

        # Energy function
        self.W_q = nn.Linear(28, self.d_k, bias=False)
        self.W_k = nn.Linear(28, self.d_k, bias=False)
        self.charge_proj = nn.Linear(28, 1, bias=True)
        self.step_size = nn.Parameter(torch.tensor(0.5))
        self.charge_decay = nn.Parameter(torch.tensor(0.3))

        # Value and output projection
        self.v_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.out_proj = nn.Linear(self.d_k, d_model, bias=False)

        # Mixing weight (starts small so we don't destroy pretrained model)
        self.mix_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5, but we scale

        nn.init.zeros_(self.out_proj.weight)  # start as identity

    def forward(self, hidden_states, attention_mask=None):
        B, N, D = hidden_states.shape

        # Predict features
        features = self.feat_proj(hidden_states.detach())

        # Energy minimization
        Q = self.W_q(features)
        K = self.W_k(features)
        compat = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        charge = torch.sigmoid(self.charge_proj(features)).squeeze(-1)

        logits = compat
        if attention_mask is not None:
            causal = torch.tril(torch.ones(N, N, device=hidden_states.device, dtype=torch.bool))
            logits = logits.masked_fill(~causal.unsqueeze(0), float('-inf'))

        for step in range(self.n_steps):
            attn = F.softmax(logits, dim=-1)
            received = attn.sum(dim=-2)
            charge = charge * (1.0 - self.charge_decay * torch.sigmoid(received - 1.0))
            energy = -compat * charge.unsqueeze(-1) * charge.unsqueeze(-2)
            logits = logits - self.step_size * energy
            if attention_mask is not None:
                logits = logits.masked_fill(~causal.unsqueeze(0), float('-inf'))

        attn = F.softmax(logits, dim=-1)
        V = self.v_proj(hidden_states)
        out = torch.matmul(attn, V)
        out = self.out_proj(out)

        # Mix with residual (controlled by learned alpha, starts near 0)
        alpha = torch.sigmoid(self.mix_alpha) * 0.2  # max 20% contribution
        return alpha * out


class GPT2WithSMD(nn.Module):
    """GPT-2 with SMD heads injected into selected layers."""
    def __init__(self, base_model, smd_layers=None):
        super().__init__()
        self.base = base_model
        self.smd_layers = smd_layers or [0, 6, 12, 18]  # inject at these layers
        d_model = base_model.config.n_embd

        # Create SMD heads for selected layers
        self.smd_heads = nn.ModuleDict({
            str(i): SMDHeadInjection(d_model, n_steps=4)
            for i in self.smd_layers
        })

        # Hook: add SMD output to each selected layer's output
        self._hooks = []
        for i in self.smd_layers:
            block = self.base.transformer.h[i]
            hook = block.register_forward_hook(self._make_hook(str(i)))
            self._hooks.append(hook)

    def _make_hook(self, layer_key):
        def hook(module, input, output):
            hidden = output[0]  # (batch, seq, d_model)
            smd_out = self.smd_heads[layer_key](hidden)
            new_hidden = hidden + smd_out
            return (new_hidden,) + output[1:]
        return hook

    def forward(self, input_ids, labels=None, **kwargs):
        return self.base(input_ids=input_ids, labels=labels, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)


def load_model(with_smd=False, smd_layers=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float32)

    if with_smd:
        model = GPT2WithSMD(model, smd_layers=smd_layers)

    return model, tokenizer


def get_data(tokenizer, seq_len=512):
    from datasets import load_dataset
    from torch.utils.data import IterableDataset, DataLoader, Dataset

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

    class ValDS(Dataset):
        def __init__(self, tokens, sl):
            self.chunks = [tokens[i*sl:(i+1)*sl+1] for i in range(len(tokens)//(sl+1))]
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i):
            c = self.chunks[i]
            return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

    train = load_dataset("wikitext", "wikitext-103-raw-v1", split='train', streaming=True)
    val = load_dataset("wikitext", "wikitext-2-raw-v1", split='validation')
    val_text = " ".join([t for t in val['text'] if t.strip()])
    val_tokens = tokenizer.encode(val_text)

    train_dl = DataLoader(StreamDS(train, tokenizer, seq_len), batch_size=2, num_workers=0, pin_memory=True)
    val_dl = DataLoader(ValDS(val_tokens, seq_len), batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    return train_dl, val_dl


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total, tok = 0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.float16):
            out = model(x, labels=y)
        nt = (y != -1).sum().item()
        total += out.loss.item() * nt
        tok += nt
    avg = total / max(tok, 1)
    return avg, math.exp(min(avg, 100))


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts=None):
    if prompts is None:
        prompts = [
            "The scientist discovered that",
            "In the beginning of the 21st century,",
            "The fundamental problem with current AI is that",
            "Language is structured because",
        ]

    model.eval()
    print("\n  Generated text:", flush=True)
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        out = model.generate(ids, max_new_tokens=60, do_sample=True,
                            temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  >>> {text[:200]}", flush=True)
        print(flush=True)


def finetune(name, model, train_dl, val_dl, tokenizer, device,
             max_steps=500, lr=1e-5):
    print(f"\n{'='*60}", flush=True)
    print(f"Fine-tuning: {name}", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.0f}M, Trainable: {trainable/1e6:.0f}M", flush=True)

    # Evaluate before fine-tuning
    vl, vp = evaluate(model, val_dl, device)
    print(f"  Before fine-tune: val_ppl={vp:.1f}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scaler = GradScaler()
    grad_accum = 4

    model.train()
    step, accum_loss, accum_n = 0, 0, 0
    t0 = time.time()

    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.float16):
            out = model(x, labels=y)
            loss = out.loss / grad_accum
        scaler.scale(loss).backward()
        accum_loss += loss.item() * grad_accum
        accum_n += 1

        if accum_n >= grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            avg = accum_loss / accum_n
            accum_loss, accum_n = 0, 0

            if step % 50 == 0 or step == 1:
                print(f"  Step {step}/{max_steps}: loss={avg:.4f} ppl={math.exp(min(avg,20)):.0f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

            if step % 200 == 0:
                vl, vp = evaluate(model, val_dl, device)
                print(f"  [EVAL] step={step}: val_ppl={vp:.1f}", flush=True)
                model.train()

            if step >= max_steps:
                break

    vl, vp = evaluate(model, val_dl, device)
    print(f"  FINAL: val_ppl={vp:.1f} ({time.time()-t0:.0f}s)", flush=True)

    generate_samples(model, tokenizer, device)

    return vl, vp


def main():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    max_steps = 500
    seq_len = 512  # GPT-2 max is 1024, we use 512

    results = {}

    # ── 1. Baseline: fine-tune vanilla GPT-2-large ──
    print("\nLoading GPT-2-large (vanilla)...", flush=True)
    model_base, tokenizer = load_model(with_smd=False)
    model_base = model_base.to(device)
    train_dl, val_dl = get_data(tokenizer, seq_len)

    vl, vp = finetune("GPT-2-large vanilla", model_base, train_dl, val_dl,
                       tokenizer, device, max_steps=max_steps)
    results['vanilla'] = {'val_loss': vl, 'val_ppl': vp}

    del model_base
    torch.cuda.empty_cache()

    # ── 2. GPT-2-large + SMD heads ──
    print("\nLoading GPT-2-large + SMD heads...", flush=True)
    model_smd, tokenizer = load_model(with_smd=True, smd_layers=[0, 6, 12, 18])
    model_smd = model_smd.to(device)

    # Freeze base model, only train SMD heads
    for param in model_smd.base.parameters():
        param.requires_grad = False
    for param in model_smd.smd_heads.parameters():
        param.requires_grad = True

    train_dl, val_dl = get_data(tokenizer, seq_len)

    vl, vp = finetune("GPT-2-large + SMD (SMD heads only)", model_smd,
                       train_dl, val_dl, tokenizer, device,
                       max_steps=max_steps, lr=3e-4)  # higher LR for new params
    results['smd_frozen'] = {'val_loss': vl, 'val_ppl': vp}

    # Now unfreeze everything and fine-tune jointly
    for param in model_smd.base.parameters():
        param.requires_grad = True

    train_dl, _ = get_data(tokenizer, seq_len)

    vl, vp = finetune("GPT-2-large + SMD (all params)", model_smd,
                       train_dl, val_dl, tokenizer, device,
                       max_steps=max_steps, lr=1e-5)
    results['smd_full'] = {'val_loss': vl, 'val_ppl': vp}

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("FINE-TUNING RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:<30s}: val_ppl={r['val_ppl']:.1f}", flush=True)

    with open('/home/asa/asa/smd_finetune_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
