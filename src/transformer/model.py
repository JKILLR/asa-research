"""
ASA Attention as Transformer Drop-In (Phase 2a — Hardened)

Core idea: standard attention score Q·K^T/√d gets an additive ASA bias that
encodes predetermined linguistic structure:

    score(i,j) = Q_i · K_j^T / √d_k  +  α · asa_bias(i,j)

The ASA bias has two components:
  1. POS compatibility mask: -inf for incompatible pairs (hard filter)
  2. Directional feature compatibility: requirements_i · features_j / |requirements_i|
     (what token i NEEDS × what token j IS)

The feature compatibility is ASYMMETRIC by design — this is the seeker/filler
principle from the toy. Verbs have requirements (need animate subject). Nouns
have features (are animate). The dot product is directional.

Supports 4 ablation modes: 'full', 'pos_only', 'features_only', 'none'.

v2.2 reference: see prior ASA v2.2 implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── POS encoding ──────────────────────────────────────────────
# Maps POS tags to integer IDs for the compatibility matrix.
POS_IDS = {
    "Noun": 0, "Verb": 1, "Det": 2, "Adj": 3,
    "Adv": 4, "Prep": 5, "Pron": 6, "Other": 7,
}
NUM_POS = len(POS_IDS)

# Periodic table validation: 24-dim features (key-query asymmetric layout)
# KEY dimensions (what token provides):
#   0: animate, 1: inanimate, 2: edible, 3: physical
#   4: needs_subject, 5: needs_object, 6: can_modify_verb, 7: can_modify_noun
#   8: abstract, 9: scope_barrier
# QUERY dimensions (what token seeks) + structural:
#   10: instrument, 11: prep_head_pref
#   12: directionality, 13: argument_reach, 14: bond_absorber, 15: structural_role
#   16: NP_orbital, 17: VP_orbital, 18: HEAD_orbital, 19: ARG_orbital
#   20: NP_QUERY, 21: VP_QUERY, 22: ARG_QUERY, 23: HEAD_QUERY
ASA_FEATURE_DIM = 24

# ── POS compatibility matrix ─────────────────────────────────
# Linguistically motivated pairs only. Adapted from v2.2's curated list
# (which used Universal Dependencies v2 with 17 POS tags).
#
# Design principle: only allow pairs that represent real syntactic
# dependencies. Content-word cross-attention (Noun↔Verb etc.) is NOT
# included — that dilutes the sparsity thesis. If the model needs
# noun-to-verb context, it should learn that through Q/K projections,
# not through a permissive mask.
_POS_COMPAT = torch.zeros(NUM_POS, NUM_POS, dtype=torch.bool)

# Hybrid hard-mask POS compatibility (from program_hard_mask.md)
# Bidirectional: if A can attend to B, B can attend to A (language modeling needs both)
# Det ↔ Noun, Adj
_POS_COMPAT[POS_IDS["Det"],  POS_IDS["Noun"]] = True
_POS_COMPAT[POS_IDS["Det"],  POS_IDS["Adj"]]  = True
_POS_COMPAT[POS_IDS["Noun"], POS_IDS["Det"]]  = True
_POS_COMPAT[POS_IDS["Adj"],  POS_IDS["Det"]]  = True
# Adj ↔ Noun, Adj (stacking)
_POS_COMPAT[POS_IDS["Adj"],  POS_IDS["Noun"]] = True
_POS_COMPAT[POS_IDS["Noun"], POS_IDS["Adj"]]  = True
_POS_COMPAT[POS_IDS["Adj"],  POS_IDS["Adj"]]  = True
# Adv ↔ Verb, Adj, Adv
_POS_COMPAT[POS_IDS["Adv"],  POS_IDS["Verb"]] = True
_POS_COMPAT[POS_IDS["Adv"],  POS_IDS["Adj"]]  = True
_POS_COMPAT[POS_IDS["Adv"],  POS_IDS["Adv"]]  = True
_POS_COMPAT[POS_IDS["Verb"], POS_IDS["Adv"]]  = True
_POS_COMPAT[POS_IDS["Adj"],  POS_IDS["Adv"]]  = True
# Verb ↔ Noun, Pron, Verb (chains), Prep, Adv
_POS_COMPAT[POS_IDS["Verb"], POS_IDS["Noun"]] = True
_POS_COMPAT[POS_IDS["Verb"], POS_IDS["Pron"]] = True
_POS_COMPAT[POS_IDS["Verb"], POS_IDS["Verb"]] = True
_POS_COMPAT[POS_IDS["Verb"], POS_IDS["Prep"]] = True
_POS_COMPAT[POS_IDS["Noun"], POS_IDS["Verb"]] = True
_POS_COMPAT[POS_IDS["Pron"], POS_IDS["Verb"]] = True
# Noun ↔ Prep, Noun (compounds)
_POS_COMPAT[POS_IDS["Noun"], POS_IDS["Prep"]] = True
_POS_COMPAT[POS_IDS["Noun"], POS_IDS["Noun"]] = True
# Pron ↔ Prep
_POS_COMPAT[POS_IDS["Pron"], POS_IDS["Prep"]] = True
# Prep ↔ Noun, Pron, Verb
_POS_COMPAT[POS_IDS["Prep"], POS_IDS["Noun"]] = True
_POS_COMPAT[POS_IDS["Prep"], POS_IDS["Pron"]] = True
_POS_COMPAT[POS_IDS["Prep"], POS_IDS["Verb"]] = True

# Self-attention (diagonal)
for i in range(NUM_POS):
    _POS_COMPAT[i, i] = True

# "Other" POS can attend to/from everything (punctuation, unknown tokens)
for j in range(NUM_POS):
    _POS_COMPAT[POS_IDS["Other"], j] = True
    _POS_COMPAT[j, POS_IDS["Other"]] = True

POS_COMPAT_MATRIX = _POS_COMPAT  # [NUM_POS, NUM_POS]


def compute_asa_bias(pos_ids: torch.Tensor,
                     features: Optional[torch.Tensor] = None,
                     requirements: Optional[torch.Tensor] = None,
                     mode: str = "full") -> torch.Tensor:
    """Compute ASA attention bias matrix.

    Args:
        pos_ids: [batch, seq_len] integer POS IDs (0-7)
        features: [batch, seq_len, ASA_FEATURE_DIM] — what each token IS
        requirements: [batch, seq_len, ASA_FEATURE_DIM] — what each token NEEDS
        mode: 'full', 'pos_only', 'features_only', 'none'

    Returns:
        bias: [batch, seq_len, seq_len] additive bias for attention scores.
              -inf where POS incompatible (in 'full' and 'pos_only' modes).
    """
    if mode == "none":
        return torch.zeros(pos_ids.shape[0], pos_ids.shape[1], pos_ids.shape[1],
                           device=pos_ids.device)

    batch, seq_len = pos_ids.shape
    bias = torch.zeros(batch, seq_len, seq_len, device=pos_ids.device)

    # ── POS compatibility mask ──
    if mode in ("full", "pos_only"):
        compat = POS_COMPAT_MATRIX.to(pos_ids.device)
        pos_i = pos_ids.unsqueeze(2).expand(-1, -1, seq_len)
        pos_j = pos_ids.unsqueeze(1).expand(-1, seq_len, -1)
        mask = compat[pos_i.reshape(-1), pos_j.reshape(-1)].reshape(batch, seq_len, seq_len)
        bias = bias.masked_fill(~mask, float("-inf"))

    # ── Feature compatibility (directional) ──
    if mode in ("full", "features_only") and features is not None:
        if requirements is not None:
            # DIRECTIONAL: requirements_i · features_j / (|requirements_i| * |features_j|)
            # Cosine-like similarity: scale-invariant, normalized both sides
            # Still asymmetric: bias[verb, noun] ≠ bias[noun, verb]
            req_normed = F.normalize(requirements, p=2, dim=-1)
            feat_normed = F.normalize(features, p=2, dim=-1)
            compat = torch.bmm(req_normed, feat_normed.transpose(1, 2))
        else:
            # Fallback: symmetric dot product (for tests without requirements)
            compat = torch.bmm(features, features.transpose(1, 2))

        # No fixed POS boosts — let the features speak for themselves.
        # If the signal is too weak, that means the features need to be
        # better, which is valuable diagnostic information.
        bias = bias + compat

    return bias


class ASAAttention(nn.Module):
    """Multi-head attention with ASA bias.

    Drop-in replacement for standard MHA. When mode='none', behaves
    identically to vanilla scaled dot-product attention.
    """

    def __init__(self, d_model: int, n_heads: int, mode: str = "full",
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.mode = mode
        self.alpha = alpha

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Saved for entropy monitoring (gate 2d)
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            pos_ids: [batch, seq_len] integer POS IDs
            features: [batch, seq_len, ASA_FEATURE_DIM] — what each token IS
            requirements: [batch, seq_len, ASA_FEATURE_DIM] — what each token NEEDS
            causal_mask: [seq_len, seq_len] bool, True where attention allowed

        Returns:
            output: [batch, seq_len, d_model]
        """
        B, S, _ = x.shape

        Q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ASA bias
        if self.mode != "none" and pos_ids is not None:
            asa_bias = compute_asa_bias(pos_ids, features, requirements, self.mode)
            scores = scores + self.alpha * asa_bias.unsqueeze(1)

        # Causal mask
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = weights.nan_to_num(0.0)
        self.last_weights = weights.detach()
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


class ASATransformerBlock(nn.Module):
    """Pre-norm transformer block with ASA attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 mode: str = "full", alpha: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = ASAAttention(d_model, n_heads, mode, alpha, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), pos_ids, features, requirements, causal_mask)
        x = x + self.ff(self.norm2(x))
        return x


class ASALanguageModel(nn.Module):
    """Causal language model with ASA attention.

    Tiny config (default): d=128, heads=2, layers=2, ff=512 → ~3M params
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 2,
                 n_layers: int = 2, d_ff: int = 512, max_seq_len: int = 256,
                 mode: str = "full", alpha: float = 1.0, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            ASATransformerBlock(d_model, n_heads, d_ff, mode, alpha, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs
            pos_ids: [batch, seq_len] POS tag IDs (optional)
            features: [batch, seq_len, ASA_FEATURE_DIM] — what each token IS
            requirements: [batch, seq_len, ASA_FEATURE_DIM] — what each token NEEDS

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=input_ids.device))

        for layer in self.layers:
            x = layer(x, pos_ids, features, requirements, causal)

        return self.output(self.norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# HYBRID ATTENTION: Hard Sparse Wave Heads + Standard Learned Heads
# ═══════════════════════════════════════════════════════════════

class HybridAttention(nn.Module):
    """Hybrid attention with wave heads (hard POS mask, no Q/K) + standard heads.

    Wave heads: predetermined attention from feature overlap + hard POS mask.
    Standard heads: full learned QK^T/√d attention over all pairs.
    """

    def __init__(self, d_model: int, n_wave_heads: int, n_std_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.n_wave = n_wave_heads
        self.n_std = n_std_heads
        n_heads = n_wave_heads + n_std_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.scale = self.head_dim ** 0.5

        # Standard heads: full Q, K, V
        if n_std_heads > 0:
            self.std_q = nn.Linear(d_model, n_std_heads * self.head_dim)
            self.std_k = nn.Linear(d_model, n_std_heads * self.head_dim)
            self.std_v = nn.Linear(d_model, n_std_heads * self.head_dim)

        # Wave heads: only V (no Q/K — scores from feature overlap)
        if n_wave_heads > 0:
            self.wave_v = nn.Linear(d_model, n_wave_heads * self.head_dim)

        # Output projection (all heads combined)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        outputs = []

        # === STANDARD HEADS ===
        if self.n_std > 0:
            Q = self.std_q(x).view(B, T, self.n_std, self.head_dim).transpose(1, 2)
            K = self.std_k(x).view(B, T, self.n_std, self.head_dim).transpose(1, 2)
            V = self.std_v(x).view(B, T, self.n_std, self.head_dim).transpose(1, 2)

            std_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            if causal_mask is not None:
                if causal_mask.dim() == 2:
                    cm = causal_mask.unsqueeze(0).unsqueeze(0)
                else:
                    cm = causal_mask
                std_scores = std_scores.masked_fill(~cm, float("-inf"))

            std_attn = F.softmax(std_scores, dim=-1)
            std_attn = std_attn.nan_to_num(0.0)
            std_attn = self.dropout(std_attn)
            std_out = torch.matmul(std_attn, V)  # [B, n_std, T, head_dim]
            outputs.append(std_out)

        # === WAVE HEADS ===
        if self.n_wave > 0 and features is not None and requirements is not None:
            wave_V = self.wave_v(x).view(B, T, self.n_wave, self.head_dim).transpose(1, 2)

            # Feature overlap: requirements_i · features_j (directional IS/NEEDS)
            feat_scores = torch.bmm(requirements, features.transpose(1, 2))  # [B, T, T]

            # Hard POS mask: -inf for incompatible pairs
            if pos_ids is not None:
                compat = POS_COMPAT_MATRIX.to(pos_ids.device)
                pos_i = pos_ids.unsqueeze(2).expand(-1, -1, T)
                pos_j = pos_ids.unsqueeze(1).expand(-1, T, -1)
                pos_ok = compat[pos_i.reshape(-1), pos_j.reshape(-1)].reshape(B, T, T)
                # Hard mask: -inf where POS incompatible
                feat_scores = feat_scores.masked_fill(~pos_ok, float("-inf"))

            # Expand to wave heads (all share the same scores)
            wave_scores = feat_scores.unsqueeze(1).expand(-1, self.n_wave, -1, -1)

            # Causal mask
            if causal_mask is not None:
                if causal_mask.dim() == 2:
                    cm = causal_mask.unsqueeze(0).unsqueeze(0)
                else:
                    cm = causal_mask
                wave_scores = wave_scores.masked_fill(~cm, float("-inf"))

            wave_attn = F.softmax(wave_scores, dim=-1)
            wave_attn = wave_attn.nan_to_num(0.0)
            wave_attn = self.dropout(wave_attn)
            wave_out = torch.matmul(wave_attn, wave_V)  # [B, n_wave, T, head_dim]
            outputs.append(wave_out)

        # Concatenate all head outputs
        all_out = torch.cat(outputs, dim=1)  # [B, n_heads, T, head_dim]
        all_out = all_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(all_out)


class HybridTransformerBlock(nn.Module):
    """Pre-norm transformer block with hybrid attention and optional wave-gated FFN."""

    def __init__(self, d_model: int, n_wave_heads: int, n_std_heads: int,
                 d_ff: int, dropout: float = 0.1, wave_ffn_gate: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = HybridAttention(d_model, n_wave_heads, n_std_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.wave_ffn_gate = wave_ffn_gate

        if wave_ffn_gate:
            # Gated FFN: output = W2(GELU(W1(x)) * sigmoid(W_gate(features)))
            self.ff_up = nn.Linear(d_model, d_ff)
            self.ff_gate = nn.Linear(ASA_FEATURE_DIM, d_ff, bias=False)
            self.ff_down = nn.Linear(d_ff, d_model)
            self.ff_drop = nn.Dropout(dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x, pos_ids=None, features=None, requirements=None,
                causal_mask=None):
        x = x + self.attention(self.norm1(x), pos_ids, features, requirements, causal_mask)

        if self.wave_ffn_gate and features is not None:
            h = self.norm2(x)
            up = F.gelu(self.ff_up(h))
            # Gate based on combined features+requirements (what token IS + NEEDS)
            gate_input = features + requirements if requirements is not None else features
            gate = torch.sigmoid(self.ff_gate(gate_input))
            x = x + self.ff_drop(self.ff_down(up * gate))
        else:
            x = x + self.ff(self.norm2(x))
        return x


class HybridLanguageModel(nn.Module):
    """Language model with hybrid wave+standard attention heads.

    Wave heads use hard POS masking + feature overlap (no Q/K).
    Standard heads use full learned attention.
    """

    def __init__(self, vocab_size: int, d_model: int = 512,
                 n_wave_heads: int = 4, n_std_heads: int = 4,
                 n_layers: int = 1, d_ff: int = 256,
                 max_seq_len: int = 128, dropout: float = 0.05,
                 wave_ffn_gate: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            HybridTransformerBlock(d_model, n_wave_heads, n_std_heads, d_ff, dropout,
                                   wave_ffn_gate=wave_ffn_gate)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(vocab_size, d_model, bias=False)  # will be tied

        # Weight tying
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, pos_ids=None, features=None, requirements=None):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=input_ids.device))

        for layer in self.layers:
            x = layer(x, pos_ids, features, requirements, causal)

        return self.output(self.norm(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
