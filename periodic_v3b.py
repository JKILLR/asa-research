"""
ASA v3b — Learned Projections Over Fixed Features.

Standard:  Q = W_q @ hidden_state    (d_model x d_k params)
ASA v3b:   Q = W_q @ periodic_feats  (28 x d_k params — 18x smaller!)

The dot product structure CANNOT memorize word pairs.
It can only learn bilinear combinations of features → forces generalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PeriodicProjectionHead(nn.Module):
    """Attention head with Q/K projected from FIXED periodic features.

    Q = W_q @ features  (28 -> d_k)
    K = W_k @ features  (28 -> d_k)
    score = Q . K^T / sqrt(d_k)

    Cannot memorize word pairs — bilinear structure forces generalization.
    """

    def __init__(self, feature_dim: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(feature_dim, d_k, bias=False)
        self.W_k = nn.Linear(feature_dim, d_k, bias=False)

    def forward(self, features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        features: (B, N, feature_dim) — fixed periodic table vectors
        mask: (B, N, N) or (1, N, N) — causal mask
        Returns: (B, N, N) attention weights
        """
        Q = self.W_q(features)  # (B, N, d_k)
        K = self.W_k(features)  # (B, N, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        return F.softmax(scores, dim=-1)


class PeriodicProjectionAttention(nn.Module):
    """Multi-head attention: periodic projection heads + optional standard heads."""

    def __init__(self, d_model: int, n_periodic: int, n_standard: int,
                 feature_dim: int, d_k: int):
        super().__init__()
        self.n_periodic = n_periodic
        self.n_standard = n_standard
        self.n_heads = n_periodic + n_standard
        self.d_k = d_k

        # Periodic heads: Q/K from features (tiny)
        self.periodic_heads = nn.ModuleList([
            PeriodicProjectionHead(feature_dim, d_k)
            for _ in range(n_periodic)
        ])

        # Standard heads: Q/K from hidden state
        if n_standard > 0:
            self.std_q = nn.Linear(d_model, n_standard * d_k)
            self.std_k = nn.Linear(d_model, n_standard * d_k)

        # V always from hidden state (model needs to learn what info to pass)
        self.v_proj = nn.Linear(d_model, self.n_heads * d_k)
        self.out_proj = nn.Linear(self.n_heads * d_k, d_model)

    def forward(self, x: torch.Tensor, features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape

        # V from hidden state
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_list = []

        # Periodic heads
        for head in self.periodic_heads:
            attn_list.append(head(features, mask))

        # Standard heads
        if self.n_standard > 0:
            Q = self.std_q(x).view(B, N, self.n_standard, self.d_k).transpose(1, 2)
            K = self.std_k(x).view(B, N, self.n_standard, self.d_k).transpose(1, 2)
            std_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                std_scores = std_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            for h in range(self.n_standard):
                attn_list.append(F.softmax(std_scores[:, h], dim=-1))

        attn = torch.stack(attn_list, dim=1)  # (B, n_heads, N, N)
        out = torch.matmul(attn, V)  # (B, n_heads, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class PeriodicProjectionBlock(nn.Module):
    def __init__(self, d_model: int, n_periodic: int, n_standard: int,
                 feature_dim: int, d_k: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = PeriodicProjectionAttention(
            d_model, n_periodic, n_standard, feature_dim, d_k)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, features, mask=None):
        h = self.attn(self.ln1(x), features, mask)
        x = x + self.drop(h)
        h = self.ff(self.ln2(x))
        x = x + self.drop(h)
        return x


class PeriodicProjectionLM(nn.Module):
    """Language model with periodic projection attention (v3b)."""

    def __init__(self, vocab_size, d_model=256, n_layers=4,
                 n_periodic=4, n_standard=0, feature_dim=28,
                 d_k=32, ff_dim=512, max_seq=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            PeriodicProjectionBlock(d_model, n_periodic, n_standard,
                                    feature_dim, d_k, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, features, targets=None):
        B, N = ids.shape
        pos = torch.arange(N, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        mask = torch.tril(torch.ones(N, N, device=ids.device, dtype=torch.bool)).unsqueeze(0)
        for block in self.blocks:
            x = block(x, features, mask)
        logits = self.lm_head(self.ln_f(x))
        out = {'logits': logits}
        if targets is not None:
            out['loss'] = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                          targets.view(-1), ignore_index=-1)
        return out

    def param_breakdown(self):
        periodic = sum(p.numel() for b in self.blocks
                      for h in b.attn.periodic_heads for p in h.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {'total': total, 'periodic_qk': periodic, 'other': total - periodic}


if __name__ == "__main__":
    m = PeriodicProjectionLM(vocab_size=8000, d_model=128, n_layers=2,
                              n_periodic=4, n_standard=0, d_k=32, ff_dim=256)
    p = m.param_breakdown()
    print(f"v3b Periodic Projection LM:")
    print(f"  Total: {p['total']:,}")
    print(f"  Periodic Q/K: {p['periodic_qk']:,}")
    print(f"  Other: {p['other']:,}")

    # Standard equivalent
    m2 = PeriodicProjectionLM(vocab_size=8000, d_model=128, n_layers=2,
                               n_periodic=0, n_standard=4, d_k=32, ff_dim=256)
    p2 = m2.param_breakdown()
    print(f"\nStandard (same d_k):")
    print(f"  Total: {p2['total']:,}")
    print(f"  Periodic Q/K vs Standard Q/K savings: {p['periodic_qk']} vs {p2['total']-p['other']}")

    x = torch.randint(0, 8000, (2, 64))
    f = torch.randn(2, 64, 28)
    out = m(x, f)
    print(f"\n  Forward pass OK: {out['logits'].shape}")
