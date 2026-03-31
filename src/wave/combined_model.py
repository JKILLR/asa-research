"""
ASA Combined Model — Periodic Attention + Wave-Gated FFN.

Two independently validated mechanisms combined:
1. Periodic projection heads: Q/K from fixed 28-dim features (replaces learned Q/K)
2. Wave-gated FFN: sigmoid(W_gate(features)) gates neuron activation (replaces standard FFN)

Together they attack different param pools for maximum compression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PeriodicProjectionHead(nn.Module):
    """Q/K projected from fixed periodic features (28 -> d_k)."""
    def __init__(self, feature_dim: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(feature_dim, d_k, bias=False)
        self.W_k = nn.Linear(feature_dim, d_k, bias=False)

    def forward(self, features, mask=None):
        Q = self.W_q(features)
        K = self.W_k(features)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)


class CombinedAttention(nn.Module):
    """Multi-head attention: periodic projection + standard heads."""
    def __init__(self, d_model, n_periodic, n_standard, feature_dim, d_k):
        super().__init__()
        self.n_periodic = n_periodic
        self.n_standard = n_standard
        self.n_heads = n_periodic + n_standard
        self.d_k = d_k

        self.periodic_heads = nn.ModuleList([
            PeriodicProjectionHead(feature_dim, d_k) for _ in range(n_periodic)
        ])
        if n_standard > 0:
            self.std_q = nn.Linear(d_model, n_standard * d_k)
            self.std_k = nn.Linear(d_model, n_standard * d_k)

        self.v_proj = nn.Linear(d_model, self.n_heads * d_k)
        self.out_proj = nn.Linear(self.n_heads * d_k, d_model)

    def forward(self, x, features, mask=None):
        B, N, D = x.shape
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_list = []
        for head in self.periodic_heads:
            attn_list.append(head(features, mask))

        if self.n_standard > 0:
            Q = self.std_q(x).view(B, N, self.n_standard, self.d_k).transpose(1, 2)
            K = self.std_k(x).view(B, N, self.n_standard, self.d_k).transpose(1, 2)
            std_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                std_scores = std_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            for h in range(self.n_standard):
                attn_list.append(F.softmax(std_scores[:, h], dim=-1))

        attn = torch.stack(attn_list, dim=1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class WaveGatedFFN(nn.Module):
    """FFN with wave amplitude gating: GELU(W1(x)) ⊙ σ(W_gate(features)).

    The gate uses periodic features to control which neurons fire per token type.
    Nouns activate different neurons than verbs — by construction, not learning.
    """
    def __init__(self, d_model, d_ff, feature_dim, dropout=0.1):
        super().__init__()
        self.ff_up = nn.Linear(d_model, d_ff)
        self.ff_gate = nn.Linear(feature_dim, d_ff, bias=False)  # tiny: feature_dim × d_ff
        self.ff_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, features):
        up = F.gelu(self.ff_up(x))
        gate = torch.sigmoid(self.ff_gate(features))
        return self.dropout(self.ff_down(up * gate))


class StandardFFN(nn.Module):
    """Standard FFN for comparison."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, features=None):
        return self.net(x)


class CombinedBlock(nn.Module):
    """Transformer block with periodic attention + wave-gated FFN."""
    def __init__(self, d_model, n_periodic, n_standard, feature_dim,
                 d_k, d_ff, dropout=0.1, use_wave_gate=True):
        super().__init__()
        self.attn = CombinedAttention(d_model, n_periodic, n_standard, feature_dim, d_k)
        if use_wave_gate:
            self.ffn = WaveGatedFFN(d_model, d_ff, feature_dim, dropout)
        else:
            self.ffn = StandardFFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, features, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), features, mask))
        x = x + self.ffn(self.ln2(x), features)
        return x


class CombinedLanguageModel(nn.Module):
    """Combined periodic attention + wave-gated FFN language model.

    Periodic heads: Q/K from fixed features (saves Q/K params)
    Wave-gated FFN: gating from fixed features (adds type-specific neurons)
    Standard heads: learned Q/K for world knowledge
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6,
                 n_periodic=1, n_standard=7, feature_dim=28,
                 d_k=32, d_ff=512, max_seq=256, dropout=0.1,
                 use_wave_gate=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            CombinedBlock(d_model, n_periodic, n_standard, feature_dim,
                         d_k, d_ff, dropout, use_wave_gate)
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
        periodic_qk = sum(p.numel() for b in self.blocks
                         for h in b.attn.periodic_heads for p in h.parameters())
        std_qk = sum(p.numel() for b in self.blocks
                    if hasattr(b.attn, 'std_q')
                    for p in list(b.attn.std_q.parameters()) + list(b.attn.std_k.parameters()))
        wave_gate = sum(b.ffn.ff_gate.weight.numel() for b in self.blocks
                       if hasattr(b.ffn, 'ff_gate'))
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'periodic_qk': periodic_qk,
            'standard_qk': std_qk,
            'wave_gate': wave_gate,
            'other': total - periodic_qk - std_qk - wave_gate
        }


class StandardLanguageModel(nn.Module):
    """Standard transformer baseline for comparison."""
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8,
                 d_k=32, d_ff=512, max_seq=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            CombinedBlock(d_model, 0, n_heads, 28, d_k, d_ff, dropout,
                         use_wave_gate=False)
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


if __name__ == "__main__":
    # Test and compare param counts
    V = 20000

    print("COMBINED (1p+7s, wave-gated FFN, d=256, 6L, ff=512):")
    m1 = CombinedLanguageModel(V, d_model=256, n_layers=6, n_periodic=1,
                                n_standard=7, d_k=32, d_ff=512, use_wave_gate=True)
    p1 = m1.param_breakdown()
    for k, v in p1.items():
        print(f"  {k}: {v:,}")

    print(f"\nSTANDARD (8 heads, d=256, 6L, ff=512):")
    m2 = StandardLanguageModel(V, d_model=256, n_layers=6, n_heads=8, d_k=32, d_ff=512)
    p2 = sum(p.numel() for p in m2.parameters())
    print(f"  total: {p2:,}")

    ratio = p2 / p1['total']
    print(f"\n  Compression: {ratio:.2f}x ({p2:,} / {p1['total']:,})")
    print(f"  Savings: {p2 - p1['total']:,} params ({(1-p1['total']/p2)*100:.1f}%)")

    # Test forward
    x = torch.randint(0, V, (2, 64))
    f = torch.randn(2, 64, 28)
    out = m1(x, f)
    print(f"\n  Forward OK: {out['logits'].shape}")

    # Compression configs
    print(f"\n{'='*60}")
    print("COMPRESSION CONFIGS:")
    configs = [
        ("A gentle 1.3x", dict(d_model=256, n_layers=6, n_periodic=1, n_standard=7, d_k=32, d_ff=256)),
        ("B medium 1.5x", dict(d_model=256, n_layers=4, n_periodic=1, n_standard=5, d_k=32, d_ff=256)),
        ("C aggressive 2x", dict(d_model=256, n_layers=2, n_periodic=1, n_standard=3, d_k=32, d_ff=128)),
        ("D extreme 3x", dict(d_model=192, n_layers=2, n_periodic=1, n_standard=3, d_k=24, d_ff=96)),
        ("E matched params", dict(d_model=256, n_layers=6, n_periodic=1, n_standard=7, d_k=32, d_ff=512)),
    ]
    for name, cfg in configs:
        m = CombinedLanguageModel(V, use_wave_gate=True, **cfg)
        p = sum(p.numel() for p in m.parameters())
        ratio = p2 / p
        print(f"  {name}: {p:,} params ({ratio:.2f}x compression)")
