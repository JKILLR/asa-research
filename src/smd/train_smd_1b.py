#!/usr/bin/env python3
"""
Training pipeline for SMD-1B.

Trains on OpenWebText (HuggingFace streaming) with:
- GPT-2 BPE tokenizer (32K vocab)
- fp16 mixed precision
- Gradient checkpointing
- Gradient accumulation

Usage:
    # Test with 125M
    python train_smd_1b.py --config 125M --max-steps 1000

    # Full 1.3B training
    python train_smd_1b.py --config 1.3B --max-steps 50000
"""

import os, sys, time, math, json, argparse
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, '/home/asa/asa')
from smd_1b import SMD1B, CONFIGS


def get_tokenizer():
    """Get GPT-2 tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def get_dataloader(tokenizer, seq_len=1024, batch_size=4, split='train'):
    """Streaming dataloader from OpenWebText."""
    from datasets import load_dataset
    from torch.utils.data import IterableDataset, DataLoader

    class StreamingLMDataset(IterableDataset):
        def __init__(self, dataset, tokenizer, seq_len):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.seq_len = seq_len

        def __iter__(self):
            buffer = []
            for example in self.dataset:
                text = example.get('text', '')
                if not text or len(text) < 50:
                    continue
                ids = self.tokenizer.encode(text)
                buffer.extend(ids)

                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split='train', streaming=True)
    dataset = StreamingLMDataset(ds, tokenizer, seq_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0,
                     pin_memory=True)


def get_wikitext_loader(tokenizer, seq_len=1024, batch_size=4, split='train'):
    """WikiText-103 for validation (smaller, non-streaming)."""
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader

    class WikiTextDataset(Dataset):
        def __init__(self, tokens, seq_len):
            self.seq_len = seq_len
            n = len(tokens) // (seq_len + 1)
            self.chunks = []
            for i in range(n):
                s = i * seq_len
                self.chunks.append(tokens[s:s + seq_len + 1])

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, idx):
            c = self.chunks[idx]
            return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split='validation')
    all_text = " ".join([t for t in ds['text'] if t.strip()])
    tokens = tokenizer.encode(all_text)
    dataset = WikiTextDataset(tokens, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                     num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss, total_tok, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.float16):
            out = model(x, y)
        nt = (y != -1).sum().item()
        total_loss += out['loss'].item() * nt
        total_tok += nt
        n += 1
        if n >= max_batches:
            break
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 100))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB", flush=True)

    # Config
    cfg = CONFIGS[args.config].copy()
    cfg['max_seq'] = args.seq_len
    cfg['use_checkpointing'] = True

    print(f"\nConfig: {args.config}", flush=True)
    print(f"  {json.dumps(cfg, indent=2)}", flush=True)

    # Tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = get_tokenizer()
    cfg['vocab_size'] = len(tokenizer)  # GPT-2: 50257
    print(f"  Vocab size: {cfg['vocab_size']}", flush=True)

    # Model
    print("\nBuilding model...", flush=True)
    model = SMD1B(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.0f}M ({n_params:,})", flush=True)

    # Data
    print("\nLoading data...", flush=True)
    train_loader = get_dataloader(tokenizer, args.seq_len, args.batch_size)
    val_loader = get_wikitext_loader(tokenizer, args.seq_len, args.batch_size)
    print(f"  Batch size: {args.batch_size}, seq_len: {args.seq_len}", flush=True)
    print(f"  Gradient accumulation: {args.grad_accum}", flush=True)
    print(f"  Effective batch: {args.batch_size * args.grad_accum}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.1, betas=(0.9, 0.95))

    # LR schedule: cosine with warmup
    def lr_fn(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # decay to 10%

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = GradScaler()

    # Training loop
    print(f"\nTraining for {args.max_steps} steps...", flush=True)
    model.train()
    step = 0
    accum_loss = 0
    accum_count = 0
    t0 = time.time()
    log_interval = max(1, args.max_steps // 100)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        with autocast(dtype=torch.float16):
            out = model(x, y)
            loss = out['loss'] / args.grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item() * args.grad_accum
        accum_count += 1

        if accum_count >= args.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            avg_loss = accum_loss / accum_count
            accum_loss = 0
            accum_count = 0

            if step % log_interval == 0 or step == 1:
                elapsed = time.time() - t0
                tokens_per_sec = step * args.batch_size * args.grad_accum * args.seq_len / elapsed
                lr_now = scheduler.get_last_lr()[0]
                mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  Step {step}/{args.max_steps}: loss={avg_loss:.4f} "
                      f"ppl={math.exp(min(avg_loss, 20)):.1f} lr={lr_now:.2e} "
                      f"tok/s={tokens_per_sec:.0f} mem={mem:.1f}GB "
                      f"({elapsed:.0f}s)", flush=True)

            if step % (log_interval * 10) == 0 and step > 0:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                print(f"  [EVAL] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}", flush=True)
                model.train()

            if step >= args.max_steps:
                break

    # Final evaluation
    val_loss, val_ppl = evaluate(model, val_loader, device)
    print(f"\nFINAL: val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    # Save
    save_path = f'/home/asa/asa/smd_{args.config}.pt'
    torch.save({
        'model_state': model.state_dict(),
        'config': cfg,
        'step': step,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
    }, save_path)
    print(f"Saved to {save_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='125M', choices=list(CONFIGS.keys()))
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-accum', type=int, default=8)
    args = parser.parse_args()
    train(args)
