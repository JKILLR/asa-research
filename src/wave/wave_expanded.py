"""
Expanded Wave Functions: 128-dim relational basis

The original 24-dim wave captures basic POS compatibility (9 dims) +
semantic features (15 dims). This is ~5% of what 512-dim learned Q/K
can represent.

This module expands to 128 dims by encoding:
1. POS pair interactions (8×8 = 64 dims) — one dim per POS pair
2. Semantic features (15 dims) — same as before
3. VerbNet class membership (24 dims) — one-hot verb class
4. Dependency direction (8 dims) — head-seeking vs dependent
5. Morphological (8 dims) — tense, number, definiteness, etc.
6. Discourse (9 dims) — position in sentence, paragraph boundary, etc.
Total: 128 dims

The key insight from the researcher: 24 dims is a drop in the ocean
against 512 learned dims. We need to encode ENOUGH structure that
wave heads can genuinely replace learned Q/K heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from model import POS_IDS, NUM_POS, ASA_FEATURE_DIM

# ═══════════════════════════════════════════════════════════════
# EXPANDED WAVE BASIS: 128 dimensions
# ═══════════════════════════════════════════════════════════════

# Dimension layout:
# [0:64]   POS pair basis (8×8 matrix unrolled — what POS-pair bonds this token can form)
# [64:79]  Semantic features (same 15-dim as before)
# [79:103] VerbNet class basis (24 verb classes)
# [103:111] Dependency direction (4 head-seeking + 4 dependent)
# [111:119] Morphological features
# [119:128] Positional / discourse features
EXPANDED_WAVE_DIM = 128

# POS pair basis: for each token with POS p, set amp=1.0 in all (p, q) pair dims
# where q is a POS that p can relate to. This gives each POS a unique 64-dim fingerprint.
# Token with POS=Noun gets 1 in (Noun,Verb), (Noun,Adj), (Noun,Det), (Noun,Noun), etc.

# Verb class one-hot (24 classes from VerbNet)
from train import VERB_TO_CLASS, VERB_CLASS_REQUIREMENTS

VERB_CLASSES = sorted(set(VERB_CLASS_REQUIREMENTS.keys()))
VERB_CLASS_TO_IDX = {vc: i for i, vc in enumerate(VERB_CLASSES)}
N_VERB_CLASSES = len(VERB_CLASSES)  # 24

# POS compatibility pairs (from model.py)
from model import POS_COMPAT_MATRIX


def build_expanded_wave(pos_ids: torch.Tensor,
                        features: Optional[torch.Tensor] = None,
                        requirements: Optional[torch.Tensor] = None,
                        verb_classes: Optional[torch.Tensor] = None,
                        token_positions: Optional[torch.Tensor] = None,
                        seq_len: int = 128) -> torch.Tensor:
    """Build 128-dim wave functions per token.

    Args:
        pos_ids: [B, S] integer POS IDs (0-7)
        features: [B, S, 15] semantic features
        requirements: [B, S, 15] verb requirements
        verb_classes: [B, S] integer verb class IDs (0-23, -1 for non-verbs)
        token_positions: [B, S] position in sequence (0 to S-1)
        seq_len: max sequence length for positional features

    Returns:
        wave: [B, S, 128]
    """
    device = pos_ids.device
    B, S = pos_ids.shape

    wave = torch.zeros(B, S, EXPANDED_WAVE_DIM, device=device)

    # ── Section 1: POS pair basis (dims 0-63) ──
    # For each token with POS p, activate dims for all compatible POS pairs
    compat = POS_COMPAT_MATRIX.to(device)  # [8, 8]
    for b in range(B):
        for s in range(S):
            p = pos_ids[b, s].item()
            if p < NUM_POS:
                # Set amp=1 for all (p, q) compatible pairs
                for q in range(NUM_POS):
                    dim_idx = p * NUM_POS + q
                    if dim_idx < 64 and compat[p, q]:
                        wave[b, s, dim_idx] = 1.0

    # ── Section 2: Semantic features (dims 64-78) ──
    if features is not None:
        wave[:, :, 64:79] = features[:, :, :15]
    if requirements is not None:
        wave[:, :, 64:79] = wave[:, :, 64:79] + requirements[:, :, :15]

    # ── Section 3: VerbNet class (dims 79-102) ──
    if verb_classes is not None:
        for b in range(B):
            for s in range(S):
                vc = verb_classes[b, s].item()
                if 0 <= vc < N_VERB_CLASSES:
                    wave[b, s, 79 + vc] = 1.0

    # ── Section 4: Dependency direction (dims 103-110) ──
    # Head-seeking tokens (verbs, preps) get positive in head dims
    # Dependent tokens (nouns, adj, adv) get positive in dependent dims
    HEAD_SEEKING = {1, 5}  # Verb, Prep — actively seek arguments/complements
    DEPENDENT = {0, 3, 4, 6}  # Noun, Adj, Adv, Pron — serve as dependents
    for b in range(B):
        for s in range(S):
            p = pos_ids[b, s].item()
            if p in HEAD_SEEKING:
                wave[b, s, 103:107] = 1.0
            if p in DEPENDENT:
                wave[b, s, 107:111] = 1.0

    # ── Section 5: Morphological features (dims 111-118) ──
    # Simple heuristics from POS tag
    # [111] singular (NN), [112] plural (NNS), [113] past tense (VBD)
    # [114] present (VBP/VBZ), [115] gerund (VBG), [116] definite (the)
    # [117] indefinite (a/an), [118] proper noun (NNP)
    # These would need the raw PTB tag; skip for now, leave as zeros

    # ── Section 6: Positional / discourse (dims 119-127) ──
    if token_positions is not None:
        pos_frac = token_positions.float() / max(seq_len, 1)
        wave[:, :, 119] = pos_frac  # relative position
        wave[:, :, 120] = 1.0 - pos_frac  # reverse position
        wave[:, :, 121] = (pos_frac < 0.1).float()  # sentence start
        wave[:, :, 122] = (pos_frac > 0.9).float()  # sentence end
        # Sinusoidal position encoding in wave space
        for i in range(4):
            freq = 2 ** i
            wave[:, :, 123 + i] = torch.sin(pos_frac * freq * math.pi)
            if 123 + i + 4 < EXPANDED_WAVE_DIM:
                wave[:, :, 127] = torch.cos(pos_frac * freq * math.pi)

    return wave


# Optimized version: avoid per-token loops using vectorized ops
def build_expanded_wave_fast(pos_ids: torch.Tensor,
                             features: Optional[torch.Tensor] = None,
                             requirements: Optional[torch.Tensor] = None,
                             verb_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fast vectorized version of build_expanded_wave.

    Skips positional features (handled by pos_emb) and morphological (need raw PTB tags).
    Focus on the three most impactful sections: POS pairs, semantic, VerbNet.
    """
    device = pos_ids.device
    B, S = pos_ids.shape

    wave = torch.zeros(B, S, EXPANDED_WAVE_DIM, device=device)

    # ── POS pair basis (dims 0-63): vectorized ──
    compat = POS_COMPAT_MATRIX.float().to(device)  # [8, 8]
    # For each token, its POS row in the compat matrix gives the 8-dim compatibility vector
    # Expand to 64 dims: repeat each POS's row as the basis activation
    pos_compat_rows = compat[pos_ids]  # [B, S, 8] — which POS types this token can relate to
    # Create the 64-dim basis by placing the compat row at the token's POS offset
    for p in range(NUM_POS):
        mask = (pos_ids == p)  # [B, S]
        if mask.any():
            start = p * NUM_POS
            end = start + NUM_POS
            if end <= 64:
                wave[:, :, start:end] += mask.unsqueeze(-1).float() * compat[p].unsqueeze(0).unsqueeze(0)

    # ── Semantic features (dims 64-78) ──
    if features is not None:
        wave[:, :, 64:79] = features[:, :, :15]
    if requirements is not None:
        wave[:, :, 64:79] = wave[:, :, 64:79] + requirements[:, :, :15]

    # ── VerbNet class (dims 79-102) ──
    if verb_classes is not None:
        valid = (verb_classes >= 0) & (verb_classes < N_VERB_CLASSES)
        if valid.any():
            vc_onehot = F.one_hot(verb_classes.clamp(0, N_VERB_CLASSES-1), N_VERB_CLASSES).float()
            wave[:, :, 79:79+N_VERB_CLASSES] = vc_onehot * valid.unsqueeze(-1).float()

    # ── Dependency direction (dims 103-110) ──
    # Vectorized: create masks for head-seeking and dependent POS
    is_head = ((pos_ids == 1) | (pos_ids == 5)).float()  # Verb or Prep
    is_dep = ((pos_ids == 0) | (pos_ids == 3) | (pos_ids == 4) | (pos_ids == 6)).float()
    wave[:, :, 103:107] = is_head.unsqueeze(-1).expand(-1, -1, 4)
    wave[:, :, 107:111] = is_dep.unsqueeze(-1).expand(-1, -1, 4)

    return wave
