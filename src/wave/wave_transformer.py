"""
Wave Function Transformer — Structure IS Computation.

The article's core claim: attention between tokens should be their wave function
overlap, not a learned Q/K projection. If words carry structural codes (the
periodic table), then attention patterns are PREDETERMINED by those codes.

Three attention head types, in order of parameter cost:
1. WaveOverlapHead: score(i,j) = features_i · features_j  (ZERO Q/K params)
2. PeriodicProjectionHead: Q = W_q @ features, K = W_k @ features  (small Q/K)
3. StandardHead: Q = W_q @ hidden, K = W_k @ hidden  (full Q/K params)

If wave overlap heads work, the compression story becomes dramatic:
  Standard: d_model × d_k × 2 = 32,768 params per head (d=256, dk=64)
  Periodic: feature_dim × d_k × 2 = 3,584 params per head
  Wave:     0 params per head

This is the test of "structure IS the computation" vs "structure HELPS computation."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveOverlapHead(nn.Module):
    """Pure wave function overlap attention. ZERO learned Q/K parameters.

    Attention score = dot product of periodic features, projected to d_k
    by a FIXED (non-learned) random projection for dimensionality matching.

    The key insight: if periodic features encode real structural information,
    their raw dot products should produce linguistically meaningful attention
    patterns without any learning.
    """
    def __init__(self, feature_dim: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.feature_dim = feature_dim
        # Fixed random projection to match d_k (not learned)
        # This just reshapes features to d_k dim without adding learnable params
        proj = torch.randn(feature_dim, d_k) / math.sqrt(feature_dim)
        self.register_buffer('proj', proj)

    def forward(self, features, mask=None):
        # Project features to d_k space (fixed, not learned)
        projected = features @ self.proj  # [B, N, d_k]
        # Attention = dot product of projected features
        scores = torch.matmul(projected, projected.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)


class WaveOverlapHeadDirect(nn.Module):
    """Even purer: raw feature dot product, no projection at all.

    score(i,j) = features_i · features_j / sqrt(feature_dim)
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features, mask=None):
        scores = torch.matmul(features, features.transpose(-2, -1)) / math.sqrt(self.feature_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)


class LearnedProjectionHead(nn.Module):
    """Q/K from learned projection of periodic features (our validated approach)."""
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


class StandardHead(nn.Module):
    """Standard learned Q/K from hidden states."""
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)


class WaveAttention(nn.Module):
    """Multi-head attention mixing wave, projection, and standard heads.

    The article's architecture: bonding IS attention.
    Wave heads provide structure, standard heads provide world knowledge.
    """
    def __init__(self, d_model, n_wave, n_proj, n_std, feature_dim, d_k,
                 use_direct_overlap=False):
        super().__init__()
        self.n_wave = n_wave
        self.n_proj = n_proj
        self.n_std = n_std
        self.n_heads = n_wave + n_proj + n_std
        self.d_k = d_k

        # Wave overlap heads (ZERO learnable Q/K params)
        if use_direct_overlap:
            self.wave_heads = nn.ModuleList([
                WaveOverlapHeadDirect(feature_dim) for _ in range(n_wave)
            ])
        else:
            self.wave_heads = nn.ModuleList([
                WaveOverlapHead(feature_dim, d_k) for _ in range(n_wave)
            ])

        # Learned projection heads (small Q/K params)
        self.proj_heads = nn.ModuleList([
            LearnedProjectionHead(feature_dim, d_k) for _ in range(n_proj)
        ])

        # Standard heads (full Q/K params)
        if n_std > 0:
            self.std_q = nn.Linear(d_model, n_std * d_k)
            self.std_k = nn.Linear(d_model, n_std * d_k)

        # V from hidden states (all heads)
        self.v_proj = nn.Linear(d_model, self.n_heads * d_k)
        self.out_proj = nn.Linear(self.n_heads * d_k, d_model)

    def forward(self, x, features, mask=None):
        B, N, D = x.shape
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_list = []

        # Wave overlap heads
        for head in self.wave_heads:
            attn_list.append(head(features, mask))

        # Learned projection heads
        for head in self.proj_heads:
            attn_list.append(head(features, mask))

        # Standard heads
        if self.n_std > 0:
            Q = self.std_q(x).view(B, N, self.n_std, self.d_k).transpose(1, 2)
            K = self.std_k(x).view(B, N, self.n_std, self.d_k).transpose(1, 2)
            std_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                std_scores = std_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            for h in range(self.n_std):
                attn_list.append(F.softmax(std_scores[:, h], dim=-1))

        attn = torch.stack(attn_list, dim=1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class WaveGatedFFN(nn.Module):
    """FFN with wave amplitude gating."""
    def __init__(self, d_model, d_ff, feature_dim, dropout=0.1):
        super().__init__()
        self.ff_up = nn.Linear(d_model, d_ff)
        self.ff_gate = nn.Linear(feature_dim, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, features):
        up = F.gelu(self.ff_up(x))
        gate = torch.sigmoid(self.ff_gate(features))
        return self.dropout(self.ff_down(up * gate))


class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, features=None):
        return self.net(x)


class WaveBlock(nn.Module):
    """Transformer block with wave function attention."""
    def __init__(self, d_model, n_wave, n_proj, n_std, feature_dim,
                 d_k, d_ff, dropout=0.1, use_wave_gate=True,
                 use_direct_overlap=False):
        super().__init__()
        self.attn = WaveAttention(d_model, n_wave, n_proj, n_std,
                                   feature_dim, d_k, use_direct_overlap)
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


class WaveTransformerLM(nn.Module):
    """Wave Function Language Model.

    The architecture from the article: tokens as wave functions,
    attention as wave overlap. Structure IS computation.

    Args:
        n_wave: number of pure wave overlap heads (0 Q/K params each)
        n_proj: number of learned projection heads (small Q/K params)
        n_std: number of standard heads (full Q/K params)
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6,
                 n_wave=2, n_proj=2, n_std=4, feature_dim=28,
                 d_k=32, d_ff=512, max_seq=256, dropout=0.1,
                 use_wave_gate=True, use_direct_overlap=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            WaveBlock(d_model, n_wave, n_proj, n_std, feature_dim,
                     d_k, d_ff, dropout, use_wave_gate, use_direct_overlap)
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
        wave_params = 0
        proj_params = 0
        std_qk = 0
        wave_gate = 0

        for b in self.blocks:
            # Wave heads have no learnable params (buffers don't count)
            for h in b.attn.wave_heads:
                wave_params += sum(p.numel() for p in h.parameters())
            # Projection heads
            for h in b.attn.proj_heads:
                proj_params += sum(p.numel() for p in h.parameters())
            # Standard Q/K
            if hasattr(b.attn, 'std_q'):
                std_qk += sum(p.numel() for p in b.attn.std_q.parameters())
                std_qk += sum(p.numel() for p in b.attn.std_k.parameters())
            # Wave gate
            if hasattr(b.ffn, 'ff_gate'):
                wave_gate += b.ffn.ff_gate.weight.numel()

        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'wave_heads': wave_params,
            'proj_heads': proj_params,
            'standard_qk': std_qk,
            'wave_gate': wave_gate,
            'v_out_ffn_emb': total - wave_params - proj_params - std_qk - wave_gate,
        }


class StandardTransformerLM(nn.Module):
    """Standard transformer baseline."""
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8,
                 d_k=32, d_ff=512, max_seq=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            WaveBlock(d_model, 0, 0, n_heads, 28, d_k, d_ff, dropout,
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
    V = 20000

    configs = [
        ("Standard 8std",          dict(n_wave=0, n_proj=0, n_std=8, use_wave_gate=False)),
        ("Combined 1proj+7std",    dict(n_wave=0, n_proj=1, n_std=7, use_wave_gate=True)),
        ("Wave 2wave+2proj+4std",  dict(n_wave=2, n_proj=2, n_std=4, use_wave_gate=True)),
        ("Wave 4wave+4std",        dict(n_wave=4, n_proj=0, n_std=4, use_wave_gate=True)),
        ("Wave 4wave+2proj+2std",  dict(n_wave=4, n_proj=2, n_std=2, use_wave_gate=True)),
        ("Wave 8wave (pure)",      dict(n_wave=8, n_proj=0, n_std=0, use_wave_gate=True)),
        ("Direct 4wave+4std",      dict(n_wave=4, n_proj=0, n_std=4, use_wave_gate=True,
                                        use_direct_overlap=True)),
    ]

    print("WAVE FUNCTION TRANSFORMER — PARAM COMPARISON")
    print(f"d=256, 6L, dk=32, ff=512, vocab=20K\n")
    print(f"{'Config':<30s} {'Total':>10s} {'Wave':>8s} {'Proj':>8s} {'StdQK':>8s} {'Gate':>8s} {'Other':>8s}")
    print("─" * 85)

    for name, cfg in configs:
        m = WaveTransformerLM(V, d_model=256, n_layers=6, d_k=32, d_ff=512, **cfg)
        p = m.param_breakdown()
        print(f"{name:<30s} {p['total']:>10,} {p['wave_heads']:>8,} {p['proj_heads']:>8,} "
              f"{p['standard_qk']:>8,} {p['wave_gate']:>8,} {p['v_out_ffn_emb']:>8,}")

    # Forward test
    print("\nForward pass test:")
    x = torch.randint(0, V, (2, 64))
    f = torch.randn(2, 64, 28)
    y = torch.randint(0, V, (2, 64))

    for name, cfg in configs:
        m = WaveTransformerLM(V, d_model=256, n_layers=6, d_k=32, d_ff=512, **cfg)
        out = m(x, f, y)
        print(f"  {name}: loss={out['loss'].item():.3f} OK")
