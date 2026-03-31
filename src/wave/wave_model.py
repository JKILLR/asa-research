"""
ASA Wave Function Attention (Phase 2 — True Wave Architecture)

From the article:
  "Each token becomes a function over a set of relational bases.
   Attention between two tokens simply becomes their wave function overlap.
   If two words share no relational notes, their overlap is zero by construction."

The wave function for each token is built from TWO sources:
  1. Syntactic amplitudes (from POS tag) — which structural roles can this token fill?
  2. Semantic amplitudes (from features/requirements) — what specific types does it match?

KEY INSIGHT: features and requirements live in the SAME 15-dim semantic space.
A noun's ANIMATE feature (features[0]=1) and a verb's ANIMATE requirement
(requirements[0]=1) occupy the same dimension. Their dot product naturally
captures compatibility:

  ⟨examine | doctor⟩ = syntactic_overlap + req_examine · feat_doctor = HIGH
  ⟨examine | rock⟩   = syntactic_overlap + req_examine · feat_rock   = LOW

Because features + requirements is just features for nouns (requirements=0)
and requirements for verbs (features=0), this preserves IS/NEEDS separation
while putting everything in one wave function.

Wave heads compute attention via this overlap — NO Q/K projections needed.
Standard heads compute normal Q·K^T/√d. The hybrid saves parameters on
wave heads while keeping flexibility on standard heads.

"Encode what we know, learn what we don't."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model import POS_IDS, NUM_POS, ASA_FEATURE_DIM, POS_COMPAT_MATRIX, compute_asa_bias

# ═══════════════════════════════════════════════════════════════
# WAVE FUNCTION BASES
# ═══════════════════════════════════════════════════════════════

WAVE_SYNTACTIC_DIM = 9

# Syntactic bases (from POS):
#   0: DET_NOUN    1: ADJ_NOUN    2: SUBJ_VERB   3: OBJ_VERB
#   4: ADV_VERB    5: PREP_NOUN   6: PREP_VERB    7: NOUN_NOUN
#   8: ADV_ADJ
#
# Each POS gets amplitudes showing which relationships it participates in.
# Orthogonal pairs get zero overlap by construction.

SYNTACTIC_AMPLITUDES = torch.tensor([
    # POS:      DET  ADJ  SUBJ OBJ  ADV  PRN  PRV  N_N  A_A
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # 0: Noun
    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # 1: Verb
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2: Det
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 3: Adj
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # 4: Adv
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # 5: Prep
    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 6: Pron
    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # 7: Other
], dtype=torch.float32)

# Total wave dimension = syntactic (9) + semantic (15) = 24
WAVE_DIM = WAVE_SYNTACTIC_DIM + ASA_FEATURE_DIM


def build_contextual_wave(pos_ids: torch.Tensor,
                          features: Optional[torch.Tensor] = None,
                          requirements: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Context-modulated wave functions: amplitudes depend on local POS context.

    For each token i, the wave function is modulated by the POS of neighbors i-1, i+1.
    This makes the wave function context-dependent WITHOUT learned parameters.

    A noun after a determiner gets stronger DET_NOUN amplitude.
    A verb before a noun gets stronger SUBJ_VERB amplitude.
    Deterministic, no learning.
    """
    device = pos_ids.device
    B, S = pos_ids.shape

    # Start with base wave amplitudes
    base = build_wave_amplitudes(pos_ids, features, requirements)

    # POS context modulation: look at neighbors
    # Create shifted POS arrays
    pos_left = torch.zeros_like(pos_ids)
    pos_left[:, 1:] = pos_ids[:, :-1]
    pos_left[:, 0] = POS_IDS["Other"]  # BOS

    pos_right = torch.zeros_like(pos_ids)
    pos_right[:, :-1] = pos_ids[:, 1:]
    pos_right[:, -1] = POS_IDS["Other"]  # EOS

    # Modulation rules (boost specific wave dims based on local POS context)
    # These are deterministic: if my left neighbor is POS X, boost dim Y
    modulation = torch.zeros_like(base)

    # Noun after Det → boost DET_NOUN (dim 0)
    noun_mask = (pos_ids == POS_IDS["Noun"]) | (pos_ids == POS_IDS["Pron"])
    det_left = (pos_left == POS_IDS["Det"])
    modulation[:, :, 0] += (noun_mask & det_left).float() * 0.5

    # Noun after Adj → boost ADJ_NOUN (dim 1)
    adj_left = (pos_left == POS_IDS["Adj"])
    modulation[:, :, 1] += (noun_mask & adj_left).float() * 0.5

    # Noun after Verb → boost OBJ_VERB (dim 3) — likely an object
    verb_left = (pos_left == POS_IDS["Verb"])
    modulation[:, :, 3] += (noun_mask & verb_left).float() * 0.5

    # Noun before Verb → boost SUBJ_VERB (dim 2) — likely a subject
    verb_right = (pos_right == POS_IDS["Verb"])
    modulation[:, :, 2] += (noun_mask & verb_right).float() * 0.5

    # Verb before Noun → boost OBJ_VERB (dim 3)
    verb_mask = (pos_ids == POS_IDS["Verb"])
    noun_right = (pos_right == POS_IDS["Noun"]) | (pos_right == POS_IDS["Pron"])
    modulation[:, :, 3] += (verb_mask & noun_right).float() * 0.5

    # Verb after Noun → boost SUBJ_VERB (dim 2)
    noun_left = (pos_left == POS_IDS["Noun"]) | (pos_left == POS_IDS["Pron"])
    modulation[:, :, 2] += (verb_mask & noun_left).float() * 0.5

    # Adv before Verb → boost ADV_VERB (dim 4)
    adv_mask = (pos_ids == POS_IDS["Adv"])
    modulation[:, :, 4] += (adv_mask & verb_right).float() * 0.5

    # Prep before Noun → boost PREP_NOUN (dim 5)
    prep_mask = (pos_ids == POS_IDS["Prep"])
    modulation[:, :, 5] += (prep_mask & noun_right).float() * 0.5

    return base + modulation


def build_wave_amplitudes(pos_ids: torch.Tensor,
                          features: Optional[torch.Tensor] = None,
                          requirements: Optional[torch.Tensor] = None,
                          normalize: bool = False) -> torch.Tensor:
    """Build per-token wave functions from linguistic properties.

    Each token's wave function = [syntactic_amps, semantic_amps]
    where semantic_amps = features + requirements (they occupy the same
    15-dim space, and IS/NEEDS separation means only one is nonzero per token).

    Args:
        pos_ids: [batch, seq_len] integer POS IDs
        features: [batch, seq_len, 15] — what each token IS (nonzero for nouns/pronouns)
        requirements: [batch, seq_len, 15] — what each token NEEDS (nonzero for verbs)
        normalize: If True, L2-normalize each token's wave function so
                   overlap becomes cosine similarity in [-1, 1].

    Returns:
        wave_amp: [batch, seq_len, WAVE_DIM] — the full wave function per token
    """
    device = pos_ids.device
    syn_amps = SYNTACTIC_AMPLITUDES.to(device)[pos_ids]  # [B, S, 9]

    if features is not None and requirements is not None:
        sem_amps = features + requirements  # [B, S, 15]
    elif features is not None:
        sem_amps = features
    elif requirements is not None:
        sem_amps = requirements
    else:
        sem_amps = torch.zeros(*pos_ids.shape, ASA_FEATURE_DIM, device=device)

    wave = torch.cat([syn_amps, sem_amps], dim=-1)  # [B, S, 24]

    if normalize:
        wave = F.normalize(wave, p=2, dim=-1)

    return wave


# Asymmetric syntactic amplitudes: what each POS IS (can fill) vs NEEDS (seeks)
# IS: what structural roles this token can FILL for other tokens
# NEEDS: what structural roles this token SEEKS from other tokens
SYNTACTIC_IS = torch.tensor([
    #          DET  ADJ  SUBJ OBJ  ADV  PRN  PRV  N_N  A_A
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # 0: Noun — IS a subject/object/det-target/etc
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # 1: Verb — IS an adverb-target/prep-target
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2: Det — IS nothing (only seeks)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 3: Adj — IS nothing (only modifies)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 4: Adv — IS nothing (only modifies)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 5: Prep — IS nothing
    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 6: Pron — IS a subject/object
    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # 7: Other
], dtype=torch.float32)

SYNTACTIC_NEEDS = torch.tensor([
    #          DET  ADJ  SUBJ OBJ  ADV  PRN  PRV  N_N  A_A
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0: Noun — NEEDS nothing (is a filler)
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 1: Verb — NEEDS subject + object
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2: Det — NEEDS a noun
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 3: Adj — NEEDS a noun to modify
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # 4: Adv — NEEDS a verb/adj to modify
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # 5: Prep — NEEDS a noun complement + verb/noun head
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6: Pron — NEEDS nothing (is a filler)
    [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # 7: Other
], dtype=torch.float32)


def build_wave_asymmetric(pos_ids: torch.Tensor,
                          features: Optional[torch.Tensor] = None,
                          requirements: Optional[torch.Tensor] = None):
    """Build asymmetric wave amplitudes: IS (what token offers) and NEEDS (what token seeks).

    Returns:
        wave_is: [batch, seq_len, WAVE_DIM] — what each token IS/offers
        wave_needs: [batch, seq_len, WAVE_DIM] — what each token NEEDS/seeks
    """
    device = pos_ids.device

    syn_is = SYNTACTIC_IS.to(device)[pos_ids]      # [B, S, 9]
    syn_needs = SYNTACTIC_NEEDS.to(device)[pos_ids]  # [B, S, 9]

    if features is not None:
        sem_is = features  # what token IS
    else:
        sem_is = torch.zeros(*pos_ids.shape, ASA_FEATURE_DIM, device=device)

    if requirements is not None:
        sem_needs = requirements  # what token NEEDS
    else:
        sem_needs = torch.zeros(*pos_ids.shape, ASA_FEATURE_DIM, device=device)

    wave_is = torch.cat([syn_is, sem_is], dim=-1)       # [B, S, 24]
    wave_needs = torch.cat([syn_needs, sem_needs], dim=-1)  # [B, S, 24]

    return wave_is, wave_needs


# ═══════════════════════════════════════════════════════════════
# WAVE HYBRID ATTENTION
# ═══════════════════════════════════════════════════════════════

class WaveHybridAttention(nn.Module):
    """Hybrid attention: wave function heads + standard Q·K heads.

    Wave heads: attention score = ⟨ψ_i | ψ_j⟩ / √wave_dim
      - No Q/K projections → parameter savings
      - Sparsity from orthogonal wave functions (natural zeros)
      - DIFFERENTIATED: each wave head sees a different slice of the wave function
        (syntactic bases, semantic bases, or mixed) for unique attention patterns

    Standard heads: attention score = Q·K^T / √d_k
      - Normal learned attention for world knowledge, context
      - These learn what linguistics doesn't predetermine

    Both share V projection and output projection.
    """

    def __init__(self, d_model: int, n_wave_heads: int, n_std_heads: int,
                 dropout: float = 0.0, differentiated: bool = True,
                 wave_rank: int = 0, wave_boost: bool = False,
                 asa_alpha: float = 0.0):
        super().__init__()
        self.n_wave = n_wave_heads
        self.n_std = n_std_heads
        self.n_total = n_wave_heads + n_std_heads
        self.differentiated = differentiated
        self.wave_rank = wave_rank
        self.wave_boost = wave_boost
        self.asa_alpha = asa_alpha
        self.wave_pos_decay = False  # set externally for positional decay
        assert self.n_total > 0
        assert d_model % self.n_total == 0

        self.d_model = d_model
        self.d_k = d_model // self.n_total

        # Standard heads need Q/K projections
        if n_std_heads > 0:
            std_dim = n_std_heads * self.d_k
            self.q_proj = nn.Linear(d_model, std_dim)
            self.k_proj = nn.Linear(d_model, std_dim)

        # Wave-boosted heads: half-size Q/K + wave overlap as additive bias
        # This is cheaper than full Q/K but allows context-dependent attention
        # modulated by predetermined linguistic structure.
        # Params: d_model × (d_k//2) × 2 per head (half of standard Q/K)
        if n_wave_heads > 0 and wave_boost:
            self.wave_qk_dim = max(self.d_k // 2, 4)  # at least 4
            wave_qk_total = n_wave_heads * self.wave_qk_dim
            self.wave_q_proj = nn.Linear(d_model, wave_qk_total)
            self.wave_k_proj = nn.Linear(d_model, wave_qk_total)

        # ALL heads share V and output projections
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Low-rank learned projection for wave heads (when not boosted)
        if n_wave_heads > 0 and wave_rank > 0 and not wave_boost:
            self.wave_proj = nn.ModuleList([
                nn.Linear(WAVE_DIM, wave_rank, bias=False)
                for _ in range(n_wave_heads)
            ])

        # Build wave head dimension assignments for differentiated mode
        # wave_dim will be set at runtime from the actual wave_amp tensor shape
        if n_wave_heads > 0 and differentiated and wave_rank == 0 and not wave_boost:
            self._wave_slices_built = False  # defer to forward when we know wave_dim

        self.last_weights: Optional[torch.Tensor] = None

    def _build_wave_slices(self, n_wave_heads: int, wave_dim: int = None):
        """Assign different wave dimensions to each wave head.

        Strategy depends on wave dimensionality:
        - 24-dim (original): syntactic (0-8) vs semantic (9-23)
        - 128-dim (expanded): equal partitions of ~32 dims each
        """
        if wave_dim is None:
            wave_dim = WAVE_DIM  # 24

        if n_wave_heads == 1:
            self.wave_slices = [list(range(wave_dim))]
        elif wave_dim <= 24 and n_wave_heads == 2:
            # Original 24-dim: syntactic vs semantic
            self.wave_slices = [
                list(range(WAVE_SYNTACTIC_DIM)),  # 0-8: syntactic
                list(range(WAVE_SYNTACTIC_DIM, wave_dim)),  # 9-23: semantic
            ]
        else:
            # General case: equal partition with stride for any dim/head count
            chunk_size = wave_dim // n_wave_heads
            self.wave_slices = []
            for h in range(n_wave_heads):
                start = h * chunk_size
                end = start + chunk_size if h < n_wave_heads - 1 else wave_dim
                self.wave_slices.append(list(range(start, end)))

    def forward(self, x: torch.Tensor,
                wave_amp: Optional[torch.Tensor] = None,
                wave_needs: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            wave_amp: [batch, seq_len, WAVE_DIM] — symmetric wave functions (IS+NEEDS combined)
            wave_needs: [batch, seq_len, WAVE_DIM] — if provided, wave_amp is IS, this is NEEDS (asymmetric mode)
            causal_mask: [seq_len, seq_len] bool
            pos_ids, features, requirements: for ASA additive bias on standard heads

        Returns:
            output: [batch, seq_len, d_model]
        """
        B, S, _ = x.shape

        # V for all heads
        V = self.v_proj(x).view(B, S, self.n_total, self.d_k).transpose(1, 2)

        score_parts = []

        # ── Wave heads: attention from wave function overlap ──
        if self.n_wave > 0 and wave_amp is not None:
            # Determine wave_q (what i needs) and wave_k (what j is)
            # Asymmetric mode: wave_needs_i · wave_is_j (directional)
            # Symmetric mode: wave_amp_i · wave_amp_j (same for both)
            w_q = wave_needs if wave_needs is not None else wave_amp  # query = NEEDS
            w_k = wave_amp  # key = IS (wave_amp is IS in asymmetric mode)

            # Build wave slices on first call (deferred to know actual wave_dim)
            if self.differentiated and not getattr(self, '_wave_slices_built', True):
                self._build_wave_slices(self.n_wave, w_k.shape[-1])
                self._wave_slices_built = True

            if self.wave_boost and hasattr(self, 'wave_q_proj'):
                wQ = self.wave_q_proj(x).view(B, S, self.n_wave, self.wave_qk_dim).transpose(1, 2)
                wK = self.wave_k_proj(x).view(B, S, self.n_wave, self.wave_qk_dim).transpose(1, 2)
                qk_scores = torch.matmul(wQ, wK.transpose(-2, -1)) / math.sqrt(self.wave_qk_dim)
                overlap = torch.bmm(w_q, w_k.transpose(1, 2))
                overlap = overlap / math.sqrt(w_k.shape[-1])
                wave_bias = overlap.unsqueeze(1).expand(-1, self.n_wave, -1, -1)
                combined = qk_scores + wave_bias
                score_parts.append(combined)
            elif self.wave_rank > 0 and hasattr(self, 'wave_proj'):
                for head_idx in range(self.n_wave):
                    proj_q = self.wave_proj[head_idx](w_q)
                    proj_k = self.wave_proj[head_idx](w_k)
                    overlap = torch.bmm(proj_q, proj_k.transpose(1, 2))
                    overlap = overlap / math.sqrt(self.wave_rank)
                    score_parts.append(overlap.unsqueeze(1))
            elif self.differentiated and hasattr(self, 'wave_slices'):
                for head_idx in range(self.n_wave):
                    dims = self.wave_slices[head_idx]
                    q_slice = w_q[:, :, dims]
                    k_slice = w_k[:, :, dims]
                    overlap = torch.bmm(q_slice, k_slice.transpose(1, 2))
                    overlap = overlap / math.sqrt(len(dims))
                    score_parts.append(overlap.unsqueeze(1))
            else:
                overlap = torch.bmm(w_q, w_k.transpose(1, 2))
                overlap = overlap / math.sqrt(w_k.shape[-1])
                for _ in range(self.n_wave):
                    score_parts.append(overlap.unsqueeze(1))
            # Apply positional decay if enabled: overlap * 1/sqrt(1 + |i-j|/k)
            if self.wave_pos_decay:
                positions = torch.arange(S, device=x.device).float()
                dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
                decay = 1.0 / torch.sqrt(1.0 + dist / 8.0)  # k=8 characteristic length
                # Apply decay to all wave head scores
                for idx in range(len(score_parts)):
                    score_parts[idx] = score_parts[idx] * decay.unsqueeze(0).unsqueeze(0)

        elif self.n_wave > 0:
            score_parts.append(torch.zeros(B, self.n_wave, S, S, device=x.device))

        # ── Standard heads: learned Q·K attention ──
        if self.n_std > 0:
            Q = self.q_proj(x).view(B, S, self.n_std, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, S, self.n_std, self.d_k).transpose(1, 2)
            std_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Add ASA additive bias to standard heads (from Phase 2b)
            if self.asa_alpha > 0 and pos_ids is not None:
                asa_bias = compute_asa_bias(pos_ids, features, requirements, mode="full")
                std_scores = std_scores + self.asa_alpha * asa_bias.unsqueeze(1)

            score_parts.append(std_scores)

        # Combine all heads
        scores = torch.cat(score_parts, dim=1)  # [B, n_total, S, S]

        # Causal mask
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = weights.nan_to_num(0.0)
        self.last_weights = weights.detach()
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


class WaveTransformerBlock(nn.Module):
    """Pre-norm transformer block with wave hybrid attention."""

    def __init__(self, d_model: int, n_wave_heads: int, n_std_heads: int,
                 d_ff: int, dropout: float = 0.1, differentiated: bool = True,
                 wave_rank: int = 0, wave_boost: bool = False,
                 asa_alpha: float = 0.0, wave_ffn_gate: bool = False,
                 wave_dim_for_gate: int = 24):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = WaveHybridAttention(d_model, n_wave_heads, n_std_heads,
                                             dropout, differentiated, wave_rank,
                                             wave_boost, asa_alpha)
        self.norm2 = nn.LayerNorm(d_model)
        self.wave_ffn_gate = wave_ffn_gate

        if wave_ffn_gate:
            # Gated FFN: output = W2(GELU(W1(x)) * sigmoid(W_gate(wave_amp)))
            self.ff_up = nn.Linear(d_model, d_ff)
            self.ff_gate = nn.Linear(wave_dim_for_gate, d_ff, bias=False)
            self.ff_down = nn.Linear(d_ff, d_model)
            self.ff_drop = nn.Dropout(dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor,
                wave_amp: Optional[torch.Tensor] = None,
                wave_needs: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), wave_amp, wave_needs, causal_mask,
                               pos_ids, features, requirements)
        if self.wave_ffn_gate and wave_amp is not None:
            h = self.norm2(x)
            up = F.gelu(self.ff_up(h))
            gate = torch.sigmoid(self.ff_gate(wave_amp))  # [B, S, d_ff]
            x = x + self.ff_drop(self.ff_down(up * gate))
        else:
            x = x + self.ff(self.norm2(x))
        return x


class WaveLanguageModel(nn.Module):
    """Causal language model with wave function hybrid attention.

    The article's vision: "encode what we know, learn what we don't."
    Wave heads encode known linguistic structure (no learned Q/K).
    Standard heads learn everything else.
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 16,
                 n_wave_heads: int = 8, n_layers: int = 6, d_ff: int = 1024,
                 max_seq_len: int = 256, dropout: float = 0.05,
                 differentiated: bool = True, wave_rank: int = 0,
                 wave_layers: str = "all", wave_boost: bool = False,
                 wave_normalize: bool = False, wave_asymmetric: bool = False,
                 asa_alpha: float = 0.0, wave_pos_decay: bool = False,
                 wave_residual_eps: float = 0.0,
                 wave_dim: int = 24, wave_contextual: bool = False,
                 wave_ffn_gate: bool = False):
        """
        Args:
            wave_layers: Which layers get wave heads.
                "all" — every layer gets n_wave_heads wave heads
                "early" — first half of layers get wave heads, rest are all standard
                "late" — last half of layers get wave heads
                "0,1,2" — specific layer indices (comma-separated)
            wave_boost: If True, wave heads get half-size Q/K + wave overlap as bias.
            wave_normalize: If True, L2-normalize wave amplitudes (cosine overlap).
            wave_asymmetric: If True, use separate IS/NEEDS wave functions (directional overlap).
            wave_residual_eps: If >0, add learned residual correction to wave amplitudes:
                amp = predetermined + eps * learned_correction(token_id)
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_wave_heads = n_wave_heads
        self.wave_normalize = wave_normalize
        self.wave_asymmetric = wave_asymmetric
        self.wave_residual_eps = wave_residual_eps
        self.wave_dim = wave_dim
        self.wave_contextual = wave_contextual

        n_std_heads = n_heads - n_wave_heads
        assert n_std_heads >= 0

        # Determine which layers get wave heads
        if wave_layers == "all":
            wave_layer_set = set(range(n_layers))
        elif wave_layers == "early":
            wave_layer_set = set(range(n_layers // 2))
        elif wave_layers == "late":
            wave_layer_set = set(range(n_layers // 2, n_layers))
        else:
            wave_layer_set = set(int(x) for x in wave_layers.split(","))

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Learned residual correction for wave amplitudes
        # amp = predetermined + eps * wave_residual_emb(token_id)
        # Adds vocab_size × WAVE_DIM params (e.g., 10K × 24 = 240K)
        if wave_residual_eps > 0 and n_wave_heads > 0:
            self.wave_residual_emb = nn.Embedding(vocab_size, wave_dim)
            nn.init.zeros_(self.wave_residual_emb.weight)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in wave_layer_set:
                self.layers.append(
                    WaveTransformerBlock(d_model, n_wave_heads, n_std_heads, d_ff,
                                        dropout, differentiated, wave_rank,
                                        wave_boost, asa_alpha, wave_ffn_gate,
                                        wave_dim)
                )
            else:
                self.layers.append(
                    WaveTransformerBlock(d_model, 0, n_heads, d_ff, dropout,
                                        asa_alpha=asa_alpha)
                )

        # Set positional decay on wave heads
        if wave_pos_decay:
            for layer in self.layers:
                layer.attention.wave_pos_decay = True

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,
                pos_ids: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                requirements: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=input_ids.device))

        # Build wave amplitudes from linguistic properties
        wave_amp = None
        wave_needs = None
        if self.n_wave_heads > 0 and pos_ids is not None:
            if self.wave_dim > 24:
                wave_amp = kwargs.get('wave_amp_expanded', None)
                if wave_amp is None:
                    wave_amp = build_wave_amplitudes(pos_ids, features, requirements)
            elif self.wave_contextual:
                wave_amp = build_contextual_wave(pos_ids, features, requirements)
            elif self.wave_asymmetric:
                wave_amp, wave_needs = build_wave_asymmetric(pos_ids, features, requirements)
            else:
                wave_amp = build_wave_amplitudes(pos_ids, features, requirements,
                                                 normalize=self.wave_normalize)

            # Apply learned residual correction
            if self.wave_residual_eps > 0 and hasattr(self, 'wave_residual_emb'):
                correction = self.wave_residual_emb(input_ids)  # [B, S, WAVE_DIM]
                wave_amp = wave_amp + self.wave_residual_eps * correction
                if wave_needs is not None:
                    wave_needs = wave_needs + self.wave_residual_eps * correction

        for layer in self.layers:
            x = layer(x, wave_amp, wave_needs, causal, pos_ids, features, requirements)

        return self.output(self.norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
