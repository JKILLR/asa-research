"""
Experiment 2: Learn a small weight matrix over the 24-dim periodic table features.
Tests whether the features are RIGHT and just need learned combination.

Architecture: score(seeker, filler) = MLP(concat(seeker.features, filler.features))
  Input: 48 dims (24 seeker + 24 filler)
  Hidden: 32 dims
  Output: 1 (bond score)

Train on gold bonds vs random non-bonds from treebank.
Evaluate by replacing wave_overlap with learned scoring.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

import eval_treebank as et
from eval_treebank import make_token, extract_gold_bonds, PTB_TO_ASA
from nltk.corpus import dependency_treebank
import asa_toy

FEATURE_DIM = et.FEATURE_DIM

# Step 1: Extract training data from treebank gold
print("Extracting training pairs from treebank...")
positive_pairs = []  # (seeker_feats, filler_feats) for gold bonds
negative_pairs = []  # random non-bonds

for sent in dependency_treebank.parsed_sents()[:200]:
    nodes = sent.nodes
    tokens = []
    ptb_tags = []
    for idx in range(1, len(nodes)):
        node = nodes[idx]
        w, t = node.get('word'), node.get('tag')
        if w is None:
            continue
        tokens.append(make_token(w, t))
        ptb_tags.append(t)

    if len(tokens) < 3 or len(tokens) > 30:
        continue

    et._current_tokens = tokens
    et._ptb_tags = ptb_tags

    gold = extract_gold_bonds(sent)
    gold_set = set()
    for h, d, r in gold:
        if h < len(tokens) and d < len(tokens):
            gold_set.add((h, d))
            gold_set.add((d, h))

    # Positive: gold bonds (try both directions)
    for h, d, r in gold:
        if h < len(tokens) and d < len(tokens):
            positive_pairs.append((tokens[h].features.copy(), tokens[d].features.copy()))

    # Negative: random non-bonds
    import random
    for _ in range(len(gold) * 2):
        i = random.randint(0, len(tokens)-1)
        j = random.randint(0, len(tokens)-1)
        if i != j and (i, j) not in gold_set:
            negative_pairs.append((tokens[i].features.copy(), tokens[j].features.copy()))

print(f"  Positive pairs: {len(positive_pairs)}")
print(f"  Negative pairs: {len(negative_pairs)}")

# Step 2: Build training data
X_pos = np.array([np.concatenate([s, f]) for s, f in positive_pairs])
X_neg = np.array([np.concatenate([s, f]) for s, f in negative_pairs])
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

# Shuffle
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

input_dim = X.shape[1]
print(f"  Input dim: {input_dim}")

# Step 3: Train simple MLP (numpy, no PyTorch needed)
hidden_dim = 32

# Xavier init
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros(1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forward(x):
    h = np.maximum(0, x @ W1 + b1)  # ReLU
    out = sigmoid(h @ W2 + b2)
    return out.flatten(), h

lr = 0.01
n_epochs = 50
batch_size = 256

print(f"\nTraining MLP ({input_dim}→{hidden_dim}→1)...")
for epoch in range(n_epochs):
    total_loss = 0
    n_batches = 0
    for start in range(0, len(X), batch_size):
        xb = X[start:start+batch_size]
        yb = y[start:start+batch_size]

        # Forward
        pred, h = forward(xb)
        loss = -np.mean(yb * np.log(pred + 1e-8) + (1-yb) * np.log(1-pred + 1e-8))

        # Backward (manual gradients)
        dpred = (pred - yb) / len(yb)  # (batch,)
        dout = dpred.reshape(-1, 1)  # (batch, 1)
        dW2 = h.T @ dout
        db2 = np.sum(dout, axis=0)
        dh = dout @ W2.T  # (batch, hidden)
        dh[h <= 0] = 0  # ReLU grad
        dW1 = xb.T @ dh
        db1 = np.sum(dh, axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        total_loss += loss
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / n_batches
        pred_all, _ = forward(X)
        acc = np.mean((pred_all > 0.5) == y)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.3f}")

# Step 4: Evaluate by using learned scoring as wave_overlap replacement
print(f"\nEvaluating with learned scoring...")

_orig_wave = asa_toy.wave_overlap

def _learned_overlap(seeker, filler, slot_idx=None,
                      seeker_idx=None, filler_idx=None,
                      state=None, complement=None):
    """Score using learned MLP on periodic table features."""
    x = np.concatenate([seeker.features, filler.features]).reshape(1, -1)
    score, _ = forward(x)
    # Scale to match original score range (0-10+)
    return float(score[0]) * 15.0

asa_toy.wave_overlap = _learned_overlap

# Run eval (with post-processing)
et._current_tokens = None
et._ptb_tags = []
results, missed, spurious = et.evaluate_treebank(max_sents=500)

# Restore
asa_toy.wave_overlap = _orig_wave
