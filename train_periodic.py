"""
Train ASA v3 — Periodic Table Attention.

Compares:
  1. Standard transformer (all Q/K/V learned)
  2. Periodic-only (all attention from periodic table features)
  3. Mixed (some periodic + some standard heads)

At matched parameter counts.
"""

import argparse
import math
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk import pos_tag

from periodic_attention import PeriodicLanguageModel
from periodic_features import sentence_to_features, word_to_features, FEATURE_DIM, PTB_TO_ASA


# ── Simple tokenizer (character-level for speed) ──
class CharTokenizer:
    """Character-level tokenizer for fast experiments."""
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def build_vocab(self, text, max_vocab=256):
        chars = sorted(set(text))[:max_vocab - 1]
        self.char_to_id = {c: i + 1 for i, c in enumerate(chars)}
        self.char_to_id['<pad>'] = 0
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        return len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


# ── Word-level tokenizer ──
class WordTokenizer:
    """Simple word-level tokenizer."""
    def __init__(self):
        self.word_to_id = {'<pad>': 0, '<unk>': 1}
        self.id_to_word = {0: '<pad>', 1: '<unk>'}

    def build_vocab(self, words, max_vocab=10000):
        from collections import Counter
        freq = Counter(words)
        for w, _ in freq.most_common(max_vocab - 2):
            idx = len(self.word_to_id)
            self.word_to_id[w] = idx
            self.id_to_word[idx] = w
        return len(self.word_to_id)

    def encode(self, words):
        return [self.word_to_id.get(w, 1) for w in words]

    def decode(self, ids):
        return [self.id_to_word.get(i, '<unk>') for i in ids]


class TextDataset(Dataset):
    """Simple text dataset with periodic features."""

    def __init__(self, token_ids, features, seq_len=128):
        self.seq_len = seq_len
        # Chunk into sequences
        n = len(token_ids) - seq_len
        self.token_ids = token_ids
        self.features = features

    def __len__(self):
        return max(1, len(self.token_ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.token_ids[idx:idx + self.seq_len]
        y = self.token_ids[idx + 1:idx + self.seq_len + 1]
        f = self.features[idx:idx + self.seq_len]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(f, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))


def prepare_data(text, max_words=50000, seq_len=128):
    """Tokenize text and extract periodic features."""
    print("Preparing data...")

    # Word tokenization
    words = text.split()[:max_words]
    print(f"  {len(words)} words")

    # Build vocabulary
    tokenizer = WordTokenizer()
    vocab_size = tokenizer.build_vocab(words, max_vocab=8000)
    print(f"  Vocab: {vocab_size}")

    # Encode
    token_ids = tokenizer.encode(words)

    # POS tag and extract features
    print("  POS tagging...")
    # Tag in chunks for efficiency
    chunk_size = 500
    all_features = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        try:
            tagged = pos_tag(chunk)
            for w, t in tagged:
                all_features.append(word_to_features(w, t))
        except:
            for w in chunk:
                all_features.append(word_to_features(w, 'NN'))

    features = np.stack(all_features)
    print(f"  Features: {features.shape}")

    dataset = TextDataset(token_ids, features, seq_len)
    return dataset, tokenizer, vocab_size


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (x, f, y) in enumerate(dataloader):
        x, f, y = x.to(device), f.to(device), y.to(device)

        out = model(x, f, y)
        loss = out['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} batch {batch_idx}: loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    for x, f, y in dataloader:
        x, f, y = x.to(device), f.to(device), y.to(device)
        out = model(x, f, y)
        total_loss += out['loss'].item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['periodic', 'standard', 'mixed', 'compare'],
                       default='compare')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--max-words', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data (TinyStories or simple text)
    print("Loading text data...")
    try:
        from datasets import load_dataset
        ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        text = ' '.join(item['text'] for i, item in zip(range(200), ds))
    except:
        # Fallback: use Brown corpus
        from nltk.corpus import brown
        text = ' '.join(brown.words()[:args.max_words])

    dataset, tokenizer, vocab_size = prepare_data(
        text, max_words=args.max_words, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"  Dataset: {len(dataset)} sequences")

    results = {}

    configs = {
        'periodic': {'n_periodic': 4, 'n_standard': 0},
        'mixed':    {'n_periodic': 2, 'n_standard': 2},
        'standard': {'n_periodic': 0, 'n_standard': 4},
    }

    if args.mode == 'compare':
        modes = ['periodic', 'mixed', 'standard']
    else:
        modes = [args.mode]

    for mode in modes:
        cfg = configs[mode]
        print(f"\n{'='*60}")
        print(f"Training {mode.upper()} model")
        print(f"  Periodic heads: {cfg['n_periodic']}, Standard heads: {cfg['n_standard']}")
        print(f"{'='*60}")

        model = PeriodicLanguageModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_periodic_heads=cfg['n_periodic'],
            n_standard_heads=cfg['n_standard'],
            feature_dim=FEATURE_DIM,
            ff_dim=args.d_model * 2,
            max_seq_len=args.seq_len,
        ).to(device)

        params = model.count_params()
        print(f"  Total params: {params['total']:,}")
        print(f"  Periodic attention: {params['periodic_attention']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=0.01)

        best_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, dataloader, optimizer, device, epoch)
            dt = time.time() - t0
            val_loss, val_ppl = evaluate(model, dataloader, device)
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}, time={dt:.1f}s")
            if val_loss < best_loss:
                best_loss = val_loss

        results[mode] = {
            'best_loss': best_loss,
            'best_ppl': math.exp(min(best_loss, 20)),
            'params': params,
        }

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        for mode, r in results.items():
            print(f"  {mode:10s}: loss={r['best_loss']:.4f}, ppl={r['best_ppl']:.1f}, "
                  f"params={r['params']['total']:,}")

        if 'periodic' in results and 'standard' in results:
            p_ppl = results['periodic']['best_ppl']
            s_ppl = results['standard']['best_ppl']
            ratio = p_ppl / s_ppl
            print(f"\n  Periodic/Standard PPL ratio: {ratio:.3f}")
            if ratio < 1.0:
                print(f"  → PERIODIC WINS by {(1-ratio)*100:.1f}%")
            elif ratio > 1.0:
                print(f"  → Standard wins by {(ratio-1)*100:.1f}%")
            else:
                print(f"  → TIE")


if __name__ == '__main__':
    main()
