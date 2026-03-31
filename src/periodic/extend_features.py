#!/usr/bin/env python3
"""
Extend 28-dim features to 34-dim by appending the 6 new elements.

Uses the SAME token IDs as the original cache, just recomputes features
with the expanded 34-dim periodic_features.py.
"""

import os, sys
import numpy as np

sys.path.insert(0, '/home/asa/asa')

from periodic_features import FEATURE_DIM, word_to_features

def main():
    cache_34 = '/home/asa/asa/cache/wikitext2_periodic_34d.npz'

    print("Extending features from 28-dim to 34-dim...", flush=True)
    print(f"Using FEATURE_DIM={FEATURE_DIM}", flush=True)

    from datasets import load_dataset
    from collections import Counter
    from nltk import pos_tag

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Build vocabulary EXACTLY as original: lowercase words
    word_freq = Counter()
    for text in ds['train']['text']:
        words = text.strip().split()
        word_freq.update(w.lower() for w in words)

    vocab = {'<pad>': 0, '<unk>': 1}
    for w, _ in word_freq.most_common(20000 - 2):
        vocab[w] = len(vocab)
    vocab_size = len(vocab)
    print(f"Vocab: {vocab_size}", flush=True)

    results = {}
    for split in ['train', 'validation']:
        print(f"Processing {split}...", flush=True)
        all_words = []
        for text in ds[split]['text']:
            words = text.strip().split()
            if words:
                all_words.extend([w.lower() for w in words])

        # POS tag in chunks
        chunk_size = 5000
        all_feats = []
        for i in range(0, len(all_words), chunk_size):
            chunk = all_words[i:i + chunk_size]
            try:
                tagged = pos_tag(chunk)
                for w, t in tagged:
                    all_feats.append(word_to_features(w, t))
            except:
                for w in chunk:
                    all_feats.append(word_to_features(w, 'NN'))
            if i % 100000 == 0 and i > 0:
                print(f"  {i}/{len(all_words)}", flush=True)

        token_ids = np.array([vocab.get(w, 1) for w in all_words], dtype=np.int32)
        features = np.stack(all_feats).astype(np.float32)

        key = 'val' if split == 'validation' else split
        results[f'{key}_ids'] = token_ids
        results[f'{key}_feats'] = features
        print(f"  {len(token_ids)} tokens, features: {features.shape}", flush=True)

    results['vocab_size'] = vocab_size

    # Verify token IDs match original
    orig = np.load('/home/asa/asa/cache/wikitext2_periodic.npz', allow_pickle=True)
    match = (orig['train_ids'][:10000] == results['train_ids'][:10000]).mean()
    print(f"Token ID match with original (first 10K): {match:.3f}", flush=True)

    np.savez_compressed(cache_34, **results)
    print(f"Saved to {cache_34}", flush=True)
    print(f"Feature shape: {results['train_feats'].shape}", flush=True)

if __name__ == '__main__':
    main()
