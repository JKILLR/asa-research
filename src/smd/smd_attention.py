"""
Semantic Molecular Dynamics (SMD) Attention Layer.

The article's Level 3 vision: attention through energy minimization.
Tokens are particles with charge and typed bonding sites.
They attract/repel based on feature compatibility.
The equilibrium configuration IS the attention pattern.

Instead of:  A = softmax(QK^T/√d) · V
We compute:  A = equilibrium(E(features)) · V

Where E(features) is the energy function from the periodic table.
The energy minimization is differentiable (unrolled gradient descent),
so we can backprop through it and learn the initial conditions + step sizes.

Key differences from standard attention:
1. Attention pattern is determined by ENERGY MINIMIZATION, not a single matmul
2. The energy function encodes LINGUISTIC CONSTRAINTS (valence, saturation, locality)
3. Multiple "relaxation steps" allow iterative refinement (like SA in the parser)
4. Charge system: satisfied bonds reduce a token's "need" (natural stopping)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunction(nn.Module):
    """Computes pairwise interaction energy from periodic features.

    E(i,j) = compatibility(i,j) × locality(i,j) × charge(i) × charge(j)

    Where:
    - compatibility = learned projection of features (like our validated approach)
    - locality = 1/distance bias (structural constraint)
    - charge = remaining "bonding capacity" (decreases as bonds form)
    """
    def __init__(self, feature_dim, d_energy, use_locality=True):
        super().__init__()
        self.d_energy = d_energy
        self.use_locality = use_locality

        # Learned compatibility projection (validated: this is what works)
        self.W_q = nn.Linear(feature_dim, d_energy, bias=False)
        self.W_k = nn.Linear(feature_dim, d_energy, bias=False)

        # Learned charge projection: features → initial charge (scalar per token)
        self.charge_proj = nn.Linear(feature_dim, 1, bias=True)

        # Locality bias (learnable scale)
        if use_locality:
            self.locality_scale = nn.Parameter(torch.tensor(1.0))

    def compatibility(self, features):
        """Pairwise compatibility scores from periodic features."""
        Q = self.W_q(features)
        K = self.W_k(features)
        return torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_energy)

    def locality_bias(self, seq_len, device):
        """Distance-based locality bias: nearby tokens interact more."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().clamp(min=1)
        return self.locality_scale / dist

    def initial_charge(self, features):
        """Initial bonding charge from features. High charge = seeks bonds."""
        return torch.sigmoid(self.charge_proj(features)).squeeze(-1)

    def energy(self, scores, charge_i, charge_j):
        """Total energy: lower = better bonding configuration."""
        # Energy is negative compatibility × charge product
        # Tokens with depleted charge stop attracting
        return -scores * charge_i.unsqueeze(-1) * charge_j.unsqueeze(-2)


class SMDAttentionHead(nn.Module):
    """Single SMD attention head.

    Runs K steps of energy minimization to find equilibrium attention pattern.
    Each step:
    1. Compute energy between all token pairs
    2. Update attention scores toward lower energy
    3. Reduce charge of "satisfied" tokens (charge depletion)

    The result is an attention pattern shaped by physics, not just a single matmul.
    """
    def __init__(self, feature_dim, d_energy, n_steps=4, step_size=0.5):
        super().__init__()
        self.energy_fn = EnergyFunction(feature_dim, d_energy)
        self.n_steps = n_steps
        self.step_size = nn.Parameter(torch.tensor(step_size))
        self.charge_decay = nn.Parameter(torch.tensor(0.3))

    def forward(self, features, mask=None):
        B, N, _ = features.shape

        # Initial state
        compat = self.energy_fn.compatibility(features)  # [B, N, N]
        locality = self.energy_fn.locality_bias(N, features.device)  # [N, N]
        charge = self.energy_fn.initial_charge(features)  # [B, N]

        # Initialize attention logits from compatibility + locality
        logits = compat + locality.unsqueeze(0)

        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))

        # Energy minimization loop
        for step in range(self.n_steps):
            # Current attention distribution
            attn = F.softmax(logits, dim=-1)

            # Charge depletion: tokens that receive lots of attention lose charge
            # (their bonding slots get filled)
            received = attn.sum(dim=-2)  # how much attention each token receives
            charge = charge * (1.0 - self.charge_decay * torch.sigmoid(received - 1.0))

            # Update logits based on charge-modulated energy
            energy = self.energy_fn.energy(compat + locality.unsqueeze(0),
                                            charge, charge)
            logits = logits - self.step_size * energy  # gradient descent on energy

            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))

        return F.softmax(logits, dim=-1)


class SMDAttention(nn.Module):
    """Multi-head SMD attention: some heads use energy minimization,
    others use standard learned attention."""
    def __init__(self, d_model, n_smd, n_std, feature_dim, d_k,
                 n_steps=4):
        super().__init__()
        self.n_smd = n_smd
        self.n_std = n_std
        self.n_heads = n_smd + n_std
        self.d_k = d_k

        self.smd_heads = nn.ModuleList([
            SMDAttentionHead(feature_dim, d_k, n_steps=n_steps)
            for _ in range(n_smd)
        ])

        if n_std > 0:
            self.std_q = nn.Linear(d_model, n_std * d_k)
            self.std_k = nn.Linear(d_model, n_std * d_k)

        self.v_proj = nn.Linear(d_model, self.n_heads * d_k)
        self.out_proj = nn.Linear(self.n_heads * d_k, d_model)

    def forward(self, x, features, mask=None):
        B, N, D = x.shape
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_list = []
        for head in self.smd_heads:
            attn_list.append(head(features, mask))

        if self.n_std > 0:
            Q = self.std_q(x).view(B, N, self.n_std, self.d_k).transpose(1, 2)
            K = self.std_k(x).view(B, N, self.n_std, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            for h in range(self.n_std):
                attn_list.append(F.softmax(scores[:, h], dim=-1))

        attn = torch.stack(attn_list, dim=1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class SMDBlock(nn.Module):
    """Transformer block with SMD attention."""
    def __init__(self, d_model, n_smd, n_std, feature_dim, d_k, d_ff,
                 dropout=0.1, n_steps=4, use_wave_gate=True):
        super().__init__()
        self.attn = SMDAttention(d_model, n_smd, n_std, feature_dim, d_k, n_steps)
        if use_wave_gate:
            from wave_transformer import WaveGatedFFN
            self.ffn = WaveGatedFFN(d_model, d_ff, feature_dim, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
            self._ffn_standard = True
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, features, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), features, mask))
        if hasattr(self, '_ffn_standard'):
            x = x + self.ffn(self.ln2(x))
        else:
            x = x + self.ffn(self.ln2(x), features)
        return x


class SMDLM(nn.Module):
    """Semantic Molecular Dynamics Language Model.

    Attention through energy minimization + standard heads for world knowledge.
    The first trainable implementation of the article's Level 3 vision.

    Args:
        n_smd: number of SMD heads (energy minimization attention)
        n_std: number of standard heads (learned Q/K)
        n_steps: relaxation steps per SMD head (more = more refined)
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6,
                 n_smd=2, n_std=6, feature_dim=34,
                 d_k=32, d_ff=512, max_seq=256, dropout=0.1,
                 n_steps=4, use_wave_gate=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SMDBlock(d_model, n_smd, n_std, feature_dim, d_k, d_ff,
                    dropout, n_steps, use_wave_gate)
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
    print("SMD ATTENTION — ARCHITECTURE TEST")
    print()

    configs = [
        ("SMD 2smd+6std (4 steps)", dict(n_smd=2, n_std=6, n_steps=4)),
        ("SMD 4smd+4std (4 steps)", dict(n_smd=4, n_std=4, n_steps=4)),
        ("SMD 2smd+6std (8 steps)", dict(n_smd=2, n_std=6, n_steps=8)),
        ("SMD 8smd pure (4 steps)", dict(n_smd=8, n_std=0, n_steps=4)),
    ]

    for name, cfg in configs:
        m = SMDLM(V, d_model=256, n_layers=6, d_k=32, d_ff=512,
                  feature_dim=34, **cfg)
        p = sum(p.numel() for p in m.parameters())
        x = torch.randint(0, V, (2, 64))
        f = torch.randn(2, 64, 34)
        y = torch.randint(0, V, (2, 64))
        out = m(x, f, y)
        print(f"  {name:<30s}: {p:>10,} params, loss={out['loss'].item():.3f} OK")
