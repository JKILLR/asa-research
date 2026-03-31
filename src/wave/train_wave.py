"""
Wave Function Attention Training Pipeline (Phase 2a)

Trains WaveLanguageModel (wave-hybrid) vs ASALanguageModel (standard baseline)
on TinyStories dataset. Compares at matched parameter counts.

The standard baseline is LOCKED: d=512, h=8, L=6, ff=1024, lr=5e-5, bs=16,
cosine decay 1 epoch. DO NOT TUNE IT.

All innovation goes into the wave architecture.

Usage:
    # Standard baseline only
    python train_wave.py --mode standard --dataset tinystories

    # Wave hybrid only
    python train_wave.py --mode wave --dataset tinystories --n-wave-heads 2

    # Compare wave vs standard
    python train_wave.py --compare --dataset tinystories --n-wave-heads 2

    # WikiText-2 (secondary validation)
    python train_wave.py --compare --dataset wikitext2
"""

import argparse
import math
import time
import json
from pathlib import Path
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk import pos_tag

from model import ASALanguageModel, HybridLanguageModel, POS_IDS, ASA_FEATURE_DIM, NUM_POS
from wave_model import WaveLanguageModel, WAVE_DIM
from wave_expanded import build_expanded_wave_fast, EXPANDED_WAVE_DIM, VERB_CLASS_TO_IDX

# ── NLTK setup ────────────────────────────────────────────────
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

_wnl = WordNetLemmatizer()

# ── Reuse extraction from train.py ────────────────────────────
from train import (
    PTB_TO_ASA, ptb_to_asa_id, extract_properties,
    VERB_TO_CLASS, VERB_CLASS_REQUIREMENTS, SYNSET_TO_FEATURES,
    PRONOUN_FEATURES, get_noun_features, get_verb_requirements,
)


# =============================================================================
# DATASET: TinyStories + WikiText-2 unified
# =============================================================================

class LMDataset(Dataset):
    """Language modeling dataset with POS tagging and feature extraction.

    Supports both TinyStories and WikiText-2.
    Caches preprocessed data to disk.
    """

    CACHE_DIR = Path("cache")

    def __init__(self, dataset_name: str = "tinystories", split: str = "train",
                 seq_len: int = 128, max_samples: int = 0,
                 min_vocab_freq: int = 3, max_vocab: int = 20000,
                 shared_word2id: dict = None):
        self.dataset_name = dataset_name
        self.seq_len = seq_len

        # Build cache key
        sample_tag = f"_{max_samples}" if max_samples > 0 else ""
        cache_name = f"{dataset_name}_{split}{sample_tag}.pt"
        cache_path = self.CACHE_DIR / cache_name

        if cache_path.exists():
            self._load_from_cache(cache_path, seq_len)
            # If shared vocab provided, remap
            if shared_word2id is not None and shared_word2id != self.word2id:
                self._remap_vocab(shared_word2id)
        else:
            self._build_from_scratch(dataset_name, split, seq_len,
                                     min_vocab_freq, max_vocab, max_samples,
                                     shared_word2id)
            self._save_to_cache(cache_path)

    def _load_from_cache(self, cache_path: Path, seq_len: int):
        print(f"Loading cached {cache_path}...")
        data = torch.load(cache_path, weights_only=False)
        self.token_ids = data["token_ids"]
        self.pos_ids = data["pos_ids"]
        self.features = data["features"]
        self.requirements = data["requirements"]
        self.word2id = data["word2id"]
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        self.seq_len = seq_len
        self.n_sequences = (len(self.token_ids) - 1) // seq_len
        print(f"  Vocab: {self.vocab_size:,}, Sequences: {self.n_sequences:,}")

    def _save_to_cache(self, cache_path: Path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "token_ids": self.token_ids,
            "pos_ids": self.pos_ids,
            "features": self.features,
            "requirements": self.requirements,
            "word2id": self.word2id,
        }, cache_path)
        print(f"  Cached to {cache_path}")

    def _remap_vocab(self, shared_word2id: dict):
        """Remap token_ids to use shared vocabulary."""
        old2new = {}
        for word, old_id in self.word2id.items():
            new_id = shared_word2id.get(word, shared_word2id.get("<unk>", 1))
            old2new[old_id] = new_id

        self.token_ids = [old2new.get(tid, 1) for tid in self.token_ids]
        self.word2id = shared_word2id
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)

    def _build_from_scratch(self, dataset_name: str, split: str, seq_len: int,
                            min_vocab_freq: int, max_vocab: int,
                            max_samples: int, shared_word2id: dict):
        from datasets import load_dataset

        print(f"Loading {dataset_name} ({split})...")

        if dataset_name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split=split)
            text_key = "text"
        elif dataset_name == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text_key = "text"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Collect words
        all_words = []
        n_docs = 0
        for item in ds:
            text = item[text_key].strip()
            if not text:
                continue
            if dataset_name == "wikitext2" and text.startswith("="):
                continue
            words = text.split()
            all_words.extend(words)
            n_docs += 1
            if max_samples > 0 and n_docs >= max_samples:
                break

        print(f"  Documents: {n_docs:,}, Total words: {len(all_words):,}")

        # Build or use shared vocab
        if shared_word2id is not None:
            self.word2id = shared_word2id
        else:
            counts = Counter(all_words)
            vocab_words = [w for w, c in counts.most_common(max_vocab)
                           if c >= min_vocab_freq]
            self.word2id = {"<pad>": 0, "<unk>": 1}
            for w in vocab_words:
                self.word2id[w] = len(self.word2id)

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        print(f"  Vocab size: {self.vocab_size:,}")

        # Encode
        self.token_ids = [self.word2id.get(w, 1) for w in all_words]

        # POS tag + feature extraction
        print("  POS tagging + feature extraction...")
        self.pos_ids = []
        self.features = []
        self.requirements = []
        chunk_size = 5000
        n_verb_hits = 0
        n_noun_hits = 0
        n_verbs = 0
        n_nouns = 0

        for i in range(0, len(all_words), chunk_size):
            chunk = all_words[i:i + chunk_size]
            tagged = pos_tag(chunk)
            for word, ptb_tag in tagged:
                self.pos_ids.append(ptb_to_asa_id(ptb_tag))
                feat, req = extract_properties(word, ptb_tag)
                self.features.append(feat)
                self.requirements.append(req)

                asa_pos = PTB_TO_ASA.get(ptb_tag, "Other")
                if asa_pos == "Verb":
                    n_verbs += 1
                    if req.any():
                        n_verb_hits += 1
                elif asa_pos == "Noun":
                    n_nouns += 1
                    if feat.any():
                        n_noun_hits += 1

        self.seq_len = seq_len
        self.n_sequences = (len(self.token_ids) - 1) // seq_len

        verb_cov = n_verb_hits / max(n_verbs, 1) * 100
        noun_cov = n_noun_hits / max(n_nouns, 1) * 100
        print(f"  VerbNet coverage: {n_verb_hits}/{n_verbs} ({verb_cov:.1f}%)")
        print(f"  WordNet coverage: {n_noun_hits}/{n_nouns} ({noun_cov:.1f}%)")
        print(f"  Sequences (len={seq_len}): {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        ids = self.token_ids[start:end]
        pos = self.pos_ids[start:end]
        feat = self.features[start:end]
        req = self.requirements[start:end]

        while len(ids) < self.seq_len + 1:
            ids.append(0)
            pos.append(POS_IDS["Other"])
            feat.append(np.zeros(ASA_FEATURE_DIM, dtype=np.float32))
            req.append(np.zeros(ASA_FEATURE_DIM, dtype=np.float32))

        return {
            "input_ids": torch.tensor(ids[:self.seq_len], dtype=torch.long),
            "labels": torch.tensor(ids[1:self.seq_len + 1], dtype=torch.long),
            "pos_ids": torch.tensor(pos[:self.seq_len], dtype=torch.long),
            "features": torch.tensor(np.stack(feat[:self.seq_len]), dtype=torch.float32),
            "requirements": torch.tensor(np.stack(req[:self.seq_len]), dtype=torch.float32),
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, is_wave, epoch,
                scheduler=None, log_interval=100, wave_dim=24, is_hybrid=False):
    model.train()
    total_loss = 0.0
    step_losses = []
    n_batches = 0

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        pos_ids = batch["pos_ids"].to(device)
        features = batch["features"].to(device)
        requirements = batch["requirements"].to(device)

        if is_wave:
            kwargs = {}
            if wave_dim > 24:
                wave_amp_exp = build_expanded_wave_fast(
                    pos_ids, features, requirements
                )
                kwargs['wave_amp_expanded'] = wave_amp_exp
            logits = model(input_ids, pos_ids=pos_ids,
                           features=features, requirements=requirements,
                           **kwargs)
        elif is_hybrid:
            logits = model(input_ids, pos_ids=pos_ids,
                           features=features, requirements=requirements)
        else:
            logits = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=0
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        step_losses.append(loss.item())
        n_batches += 1

        if (i + 1) % log_interval == 0:
            avg = total_loss / n_batches
            ppl = math.exp(min(avg, 20))
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  ep{epoch} step {i+1}/{len(dataloader)} "
                  f"loss={avg:.4f} ppl={ppl:.2f} lr={cur_lr:.2e}")

    return total_loss / max(n_batches, 1), step_losses


def evaluate(model, dataloader, device, is_wave, wave_dim=24, is_hybrid=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            features = batch["features"].to(device)
            requirements = batch["requirements"].to(device)

            if is_wave:
                kwargs = {}
                if wave_dim > 24:
                    wave_amp_exp = build_expanded_wave_fast(
                        pos_ids, features, requirements
                    )
                    kwargs['wave_amp_expanded'] = wave_amp_exp
                logits = model(input_ids, pos_ids=pos_ids,
                               features=features, requirements=requirements,
                               **kwargs)
            elif is_hybrid:
                logits = model(input_ids, pos_ids=pos_ids,
                               features=features, requirements=requirements)
            else:
                logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0
            )
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def make_scheduler(optimizer, total_steps, warmup_frac=0.05):
    """Cosine decay over 1 epoch with warmup."""
    warmup_steps = max(int(total_steps * warmup_frac), 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN TRAINING ROUTINE
# =============================================================================

def train_model(mode: str, dataset_name: str, epochs: int, device: str,
                d_model: int, n_heads: int, n_layers: int, d_ff: int,
                n_wave_heads: int, lr: float, batch_size: int,
                seq_len: int, max_samples: int,
                train_ds=None, val_ds=None, wave_rank: int = 0,
                wave_layers: str = "all", wave_boost: bool = False,
                wave_normalize: bool = False, wave_asymmetric: bool = False,
                asa_alpha: float = 0.0, wave_pos_decay: bool = False,
                wave_residual_eps: float = 0.0, wave_dim: int = 24,
                wave_contextual: bool = False,
                wave_ffn_gate: bool = False):
    """Train a single model configuration.

    mode: 'wave', 'standard', or 'hybrid'
    """
    is_wave = (mode == "wave")
    is_hybrid = (mode == "hybrid")

    print(f"\n{'='*60}")
    print(f"Training: {mode.upper()} | d={d_model} h={n_heads} L={n_layers} ff={d_ff}")
    if is_wave:
        print(f"  Wave heads: {n_wave_heads}, Standard heads: {n_heads - n_wave_heads}")
    elif is_hybrid:
        n_std = n_heads - n_wave_heads
        print(f"  HYBRID: {n_wave_heads} wave (hard mask) + {n_std} standard (learned)")
    print(f"  lr={lr} bs={batch_size} seq={seq_len} epochs={epochs}")
    print(f"{'='*60}")

    # Load data
    if train_ds is None:
        train_ds = LMDataset(dataset_name, "train", seq_len, max_samples)
    if val_ds is None:
        split = "validation" if dataset_name != "tinystories" else "validation"
        val_ds = LMDataset(dataset_name, split, seq_len, max_samples,
                           shared_word2id=train_ds.word2id)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    vocab_size = train_ds.vocab_size

    # Build model
    if is_hybrid:
        n_std = n_heads - n_wave_heads
        model = HybridLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_wave_heads=n_wave_heads,
            n_std_heads=n_std,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=seq_len,
            dropout=0.05,
            wave_ffn_gate=wave_ffn_gate,
        ).to(device)
    elif is_wave:
        model = WaveLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_wave_heads=n_wave_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=seq_len,
            dropout=0.05,
            wave_rank=wave_rank,
            wave_layers=wave_layers,
            wave_boost=wave_boost,
            wave_normalize=wave_normalize,
            wave_asymmetric=wave_asymmetric,
            asa_alpha=asa_alpha,
            wave_pos_decay=wave_pos_decay,
            wave_residual_eps=wave_residual_eps,
            wave_dim=wave_dim,
            wave_contextual=wave_contextual,
            wave_ffn_gate=wave_ffn_gate,
        ).to(device)
    else:
        # Standard baseline — mode="none" means no ASA bias
        model = ASALanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=seq_len,
            mode="none",
            alpha=0.0,
            dropout=0.05,
        ).to(device)

    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine decay over ALL epochs (not just 1)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * epochs
    scheduler = make_scheduler(optimizer, total_steps)

    results = {
        "mode": mode, "n_params": n_params, "vocab_size": vocab_size,
        "config": {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "d_ff": d_ff, "n_wave_heads": n_wave_heads if is_wave else 0,
            "lr": lr, "batch_size": batch_size, "seq_len": seq_len,
        },
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, step_losses = train_epoch(
            model, train_dl, optimizer, device, is_wave, epoch,
            scheduler=scheduler, log_interval=200,
            wave_dim=wave_dim if is_wave else 24,
            is_hybrid=is_hybrid
        )
        elapsed = time.time() - t0

        val_loss, val_ppl = evaluate(model, val_dl, device, is_wave,
                                     wave_dim=wave_dim if is_wave else 24,
                                     is_hybrid=is_hybrid)

        print(f"\nEpoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
              f"ppl={val_ppl:.2f} time={elapsed:.1f}s")

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "time": elapsed,
        })

    return results, train_ds, val_ds


def compare(args):
    """Run hybrid/wave vs standard comparison."""
    device = args.device
    compare_mode = args.mode if args.mode in ("wave", "hybrid") else "hybrid"
    print("\n" + "=" * 70)
    print(f"{compare_mode.upper()} vs STANDARD TRANSFORMER COMPARISON")
    print("=" * 70)

    # Shared dataset
    train_ds = LMDataset(args.dataset, "train", args.seq_len, args.max_samples)
    val_ds = LMDataset(args.dataset, "validation", args.seq_len, args.max_samples,
                       shared_word2id=train_ds.word2id)

    # Standard baseline (LOCKED config)
    std_results, _, _ = train_model(
        mode="standard",
        dataset_name=args.dataset, epochs=args.epochs, device=device,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, n_wave_heads=0, lr=args.lr,
        batch_size=args.batch_size, seq_len=args.seq_len,
        max_samples=args.max_samples, train_ds=train_ds, val_ds=val_ds,
    )

    # Hybrid or wave
    wave_results, _, _ = train_model(
        mode=compare_mode,
        dataset_name=args.dataset, epochs=args.epochs, device=device,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, n_wave_heads=args.n_wave_heads, lr=args.lr,
        batch_size=args.batch_size, seq_len=args.seq_len,
        max_samples=args.max_samples, train_ds=train_ds, val_ds=val_ds,
        wave_rank=args.wave_rank,
        wave_layers=args.wave_layers,
        wave_boost=args.wave_boost,
        wave_normalize=args.wave_normalize,
        wave_asymmetric=args.wave_asymmetric,
        asa_alpha=args.asa_alpha,
        wave_ffn_gate=args.wave_ffn_gate,
    )

    # Summary
    std_final = std_results["epochs"][-1]
    wave_final = wave_results["epochs"][-1]
    gap = std_final["val_loss"] - wave_final["val_loss"]

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<12} {'Params':>10} {'Train Loss':>12} {'Val Loss':>10} {'Val PPL':>10}")
    print(f"  {'-'*56}")
    print(f"  {'Standard':<12} {std_results['n_params']:>10,} "
          f"{std_final['train_loss']:>12.4f} {std_final['val_loss']:>10.4f} {std_final['val_ppl']:>10.2f}")
    print(f"  {'Wave':<12} {wave_results['n_params']:>10,} "
          f"{wave_final['train_loss']:>12.4f} {wave_final['val_loss']:>10.4f} {wave_final['val_ppl']:>10.2f}")
    print(f"\n  Gap (std - wave): {gap:+.4f}")
    print(f"  Param savings: {std_results['n_params'] - wave_results['n_params']:,} "
          f"({(1 - wave_results['n_params']/std_results['n_params'])*100:.1f}%)")

    if gap > 0:
        print(f"\n  >>> WAVE WINS by {gap:.4f} val loss <<<")
    else:
        print(f"\n  >>> Standard wins by {-gap:.4f} val loss <<<")

    # Save results
    combined = {
        "standard": std_results,
        "wave": wave_results,
        "gap": gap,
        "dataset": args.dataset,
    }
    out_path = Path(f"results_wave_{args.dataset}_w{args.n_wave_heads}.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Function Attention Training")

    # Model config
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--n-wave-heads", type=int, default=2,
                        help="Number of wave function attention heads")
    parser.add_argument("--wave-rank", type=int, default=0,
                        help="Low-rank projection dimension for wave heads (0=raw overlap)")
    parser.add_argument("--wave-layers", type=str, default="all",
                        help="Which layers get wave heads: all, early, late, or '0,1,2'")
    parser.add_argument("--wave-boost", action="store_true",
                        help="Wave heads get half-size Q/K + wave overlap as bias")
    parser.add_argument("--wave-normalize", action="store_true",
                        help="L2-normalize wave amplitudes for cosine overlap")
    parser.add_argument("--wave-asymmetric", action="store_true",
                        help="Use separate IS/NEEDS wave functions (directional overlap)")
    parser.add_argument("--asa-alpha", type=float, default=0.0,
                        help="ASA additive bias strength on standard heads (0=disabled)")
    parser.add_argument("--wave-ffn-gate", action="store_true",
                        help="Wave-gated FFN: gate FFN output with feature-derived signal")

    # Training config
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Dataset
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "wikitext2"])
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max documents to load (0=all)")

    # Mode
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["wave", "standard", "hybrid"])
    parser.add_argument("--compare", action="store_true",
                        help="Run wave vs standard comparison")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Auto-detect device
    if torch.cuda.is_available():
        args.device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        args.device = "mps"
        print("Using MPS")
    else:
        args.device = "cpu"
        print("Using CPU")

    if args.compare:
        compare(args)
    else:
        train_model(
            mode=args.mode,
            dataset_name=args.dataset, epochs=args.epochs, device=args.device,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            d_ff=args.d_ff, n_wave_heads=args.n_wave_heads, lr=args.lr,
            batch_size=args.batch_size, seq_len=args.seq_len,
            max_samples=args.max_samples, wave_rank=args.wave_rank,
            wave_layers=args.wave_layers, wave_boost=args.wave_boost,
            wave_normalize=args.wave_normalize,
            wave_asymmetric=args.wave_asymmetric,
            asa_alpha=args.asa_alpha,
        )
