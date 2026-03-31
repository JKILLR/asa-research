"""
SMD-1.3B: Semantic Molecular Dynamics Language Model at Scale.

Architecture: LLaMA-style transformer with SMD attention heads.
- RMSNorm (not LayerNorm)
- RoPE positional encoding
- SwiGLU FFN (or wave-gated FFN)
- SMD attention heads (energy minimization) + standard heads
- BPE tokenization with subword feature assignment

Target: 1.3B params on single A100 80GB with:
- fp16 mixed precision
- Gradient checkpointing
- Gradient accumulation

The subword feature problem:
  BPE splits "unhappiness" → ["un", "happ", "iness"]
  Solution: learn a lightweight feature predictor that maps
  token embeddings → periodic features. This is a small MLP
  (embed_dim → 28) trained jointly. The features are PREDICTED
  not fixed, but the prediction is constrained to the periodic
  table's structure by the MLP architecture.

  This is a compromise: features aren't fully predetermined,
  but the MLP is tiny (embed_dim × 28 = ~100K params) and
  learns to map subword tokens to their linguistic properties.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, d_k, max_seq=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq = max_seq

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class FeaturePredictor(nn.Module):
    """Predicts periodic table features from token embeddings.

    Solves the subword problem: BPE tokens don't have POS tags,
    but their embeddings encode enough information to predict
    periodic features. This is a tiny MLP (~100K params) that
    learns the mapping from embedding space to feature space.

    The features are structured: the output is constrained to
    [0, 1] range and has the same semantic layout as the periodic table.
    """
    def __init__(self, d_model, feature_dim=28, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),  # constrain to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class SMDHead(nn.Module):
    """Energy minimization attention head with charge depletion."""
    def __init__(self, feature_dim, d_k, n_steps=4):
        super().__init__()
        self.d_k = d_k
        self.n_steps = n_steps
        self.W_q = nn.Linear(feature_dim, d_k, bias=False)
        self.W_k = nn.Linear(feature_dim, d_k, bias=False)
        self.charge_proj = nn.Linear(feature_dim, 1, bias=True)
        self.step_size = nn.Parameter(torch.tensor(0.5))
        self.charge_decay = nn.Parameter(torch.tensor(0.3))
        self.locality_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, cos, sin, mask=None):
        B, N, _ = features.shape
        Q = self.W_q(features)
        K = self.W_k(features)
        # Apply RoPE to Q/K for position awareness
        Q, K = apply_rope(Q.unsqueeze(1), K.unsqueeze(1),
                          cos.unsqueeze(0).unsqueeze(0),
                          sin.unsqueeze(0).unsqueeze(0))
        Q, K = Q.squeeze(1), K.squeeze(1)

        compat = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        charge = torch.sigmoid(self.charge_proj(features)).squeeze(-1)

        logits = compat
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))

        for step in range(self.n_steps):
            attn = F.softmax(logits, dim=-1)
            received = attn.sum(dim=-2)
            charge = charge * (1.0 - self.charge_decay * torch.sigmoid(received - 1.0))
            energy = -compat * charge.unsqueeze(-1) * charge.unsqueeze(-2)
            logits = logits - self.step_size * energy
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))

        return F.softmax(logits, dim=-1)


class SMDScaleAttention(nn.Module):
    """Multi-head attention with SMD + standard heads at scale."""
    def __init__(self, d_model, n_smd, n_std, feature_dim, d_k, n_steps=4):
        super().__init__()
        self.n_smd = n_smd
        self.n_std = n_std
        self.n_heads = n_smd + n_std
        self.d_k = d_k

        self.smd_heads = nn.ModuleList([
            SMDHead(feature_dim, d_k, n_steps) for _ in range(n_smd)
        ])

        if n_std > 0:
            self.std_q = nn.Linear(d_model, n_std * d_k, bias=False)
            self.std_k = nn.Linear(d_model, n_std * d_k, bias=False)

        self.v_proj = nn.Linear(d_model, self.n_heads * d_k, bias=False)
        self.out_proj = nn.Linear(self.n_heads * d_k, d_model, bias=False)

        self.rotary = RotaryEmbedding(d_k)

    def forward(self, x, features, mask=None):
        B, N, D = x.shape
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        cos, sin = self.rotary(N, x.device)

        attn_list = []
        for head in self.smd_heads:
            attn_list.append(head(features, cos, sin, mask))

        if self.n_std > 0:
            Q = self.std_q(x).view(B, N, self.n_std, self.d_k).transpose(1, 2)
            K = self.std_k(x).view(B, N, self.n_std, self.d_k).transpose(1, 2)
            cos_4d = cos.unsqueeze(0).unsqueeze(0)
            sin_4d = sin.unsqueeze(0).unsqueeze(0)
            Q, K = apply_rope(Q, K, cos_4d, sin_4d)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            for h in range(self.n_std):
                attn_list.append(F.softmax(scores[:, h], dim=-1))

        attn = torch.stack(attn_list, dim=1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class WaveGatedSwiGLU(nn.Module):
    """SwiGLU FFN with optional wave gating from features."""
    def __init__(self, d_model, d_ff, feature_dim=None, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # SwiGLU gate
        if feature_dim:
            self.wave_gate = nn.Linear(feature_dim, d_ff, bias=False)
        else:
            self.wave_gate = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, features=None):
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        out = gate * up
        if self.wave_gate is not None and features is not None:
            wave = torch.sigmoid(self.wave_gate(features))
            out = out * wave
        return self.dropout(self.w2(out))


class SMDBlock(nn.Module):
    def __init__(self, d_model, n_smd, n_std, feature_dim, d_k, d_ff,
                 dropout=0.0, n_steps=4, use_wave_gate=True):
        super().__init__()
        self.attn = SMDScaleAttention(d_model, n_smd, n_std, feature_dim, d_k, n_steps)
        self.ffn = WaveGatedSwiGLU(d_model, d_ff,
                                     feature_dim if use_wave_gate else None, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, features, mask=None):
        x = x + self.attn(self.norm1(x), features, mask)
        x = x + self.ffn(self.norm2(x), features)
        return x


class SMD1B(nn.Module):
    """SMD Language Model at 1.3B scale.

    Architecture:
        - d_model=2048, n_layers=24, n_heads=16 (d_k=128)
        - 2 SMD heads + 14 standard heads per layer
        - SwiGLU FFN with wave gating (d_ff=5504)
        - RoPE, RMSNorm
        - Learned feature predictor (embed → 28-dim periodic features)
        - Gradient checkpointing for memory efficiency

    ~1.3B params total.
    """
    def __init__(self, vocab_size=32000, d_model=2048, n_layers=24,
                 n_smd=2, n_std=14, feature_dim=28,
                 d_k=128, d_ff=5504, max_seq=2048, dropout=0.0,
                 n_steps=4, use_wave_gate=True, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.feature_predictor = FeaturePredictor(d_model, feature_dim, hidden_dim=256)

        self.blocks = nn.ModuleList([
            SMDBlock(d_model, n_smd, n_std, feature_dim, d_k, d_ff,
                    dropout, n_steps, use_wave_gate)
            for _ in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, targets=None):
        B, N = ids.shape
        x = self.tok_emb(ids)

        # Predict periodic features from embeddings
        features = self.feature_predictor(x.detach())  # detach: features don't backprop into embeddings

        mask = torch.tril(torch.ones(N, N, device=ids.device, dtype=torch.bool)).unsqueeze(0)

        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, features, mask, use_reentrant=False)
            else:
                x = block(x, features, mask)

        logits = self.lm_head(self.norm_f(x))
        out = {'logits': logits}
        if targets is not None:
            out['loss'] = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                          targets.view(-1), ignore_index=-1)
        return out


# Smaller configs for testing
CONFIGS = {
    '125M': dict(vocab_size=32000, d_model=768, n_layers=12, n_smd=2, n_std=10,
                 d_k=64, d_ff=2048, n_steps=4),
    '350M': dict(vocab_size=32000, d_model=1024, n_layers=24, n_smd=2, n_std=14,
                 d_k=64, d_ff=2816, n_steps=4),
    '1.3B': dict(vocab_size=32000, d_model=2048, n_layers=24, n_smd=2, n_std=14,
                 d_k=128, d_ff=5504, n_steps=4),
}


if __name__ == "__main__":
    print("SMD-1B ARCHITECTURE")
    print()

    for name, cfg in CONFIGS.items():
        m = SMD1B(**cfg, use_checkpointing=False)
        p = sum(p.numel() for p in m.parameters())

        # Count SMD vs standard params
        smd_p = sum(p.numel() for b in m.blocks for h in b.attn.smd_heads for p in h.parameters())
        feat_p = sum(p.numel() for p in m.feature_predictor.parameters())
        wave_p = sum(b.ffn.wave_gate.weight.numel() for b in m.blocks if b.ffn.wave_gate is not None)

        print(f"  {name}: {p/1e6:.0f}M params")
        print(f"    SMD heads: {smd_p/1e6:.1f}M ({smd_p/p*100:.1f}%)")
        print(f"    Feature predictor: {feat_p/1e6:.2f}M ({feat_p/p*100:.1f}%)")
        print(f"    Wave gate: {wave_p/1e6:.1f}M ({wave_p/p*100:.1f}%)")
        print()

    # Forward test with 125M
    print("Forward pass test (125M):")
    m = SMD1B(**CONFIGS['125M'], max_seq=512, use_checkpointing=False)
    x = torch.randint(0, 32000, (1, 64))
    y = torch.randint(0, 32000, (1, 64))
    out = m(x, y)
    print(f"  loss={out['loss'].item():.3f} OK")
    print(f"  logits shape: {out['logits'].shape}")
