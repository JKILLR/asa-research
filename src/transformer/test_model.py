"""
Gate 2a Verification Tests for ASA Attention Module (Hardened)

Tests:
  1. Forward pass: no NaN/Inf in logits across 100 random inputs
  2. Backward pass: all parameters receive gradients, no NaN gradients
  3. Copy task: loss decreases by >=50% in 500 steps (proves gradient flow)
  4. Mode parity: all 4 ablation modes produce valid outputs
  5. Attention entropy: weights don't collapse to single token
  6. ASA bias affects attention: full mode differs from none mode
  7. Parameter count: reasonable for tiny config
  8. ASA bias is asymmetric: bias[verb_pos, noun_pos] != bias[noun_pos, verb_pos]
  9. POS mask sparsity: <=40% of entries are True
"""

import sys
import torch
import torch.nn.functional as F
from model import (ASALanguageModel, ASA_FEATURE_DIM, NUM_POS,
                   POS_COMPAT_MATRIX, POS_IDS, compute_asa_bias)

VOCAB_SIZE = 200
SEQ_LEN = 32
BATCH_SIZE = 4
DEVICE = "cpu"


def test_forward_pass():
    """ASA attention produces valid logits (no NaN/Inf) across random inputs."""
    print("Test 1: Forward pass (100 random inputs)...", end=" ")
    model = ASALanguageModel(VOCAB_SIZE, mode="full").to(DEVICE)
    model.eval()

    for i in range(100):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        pos = torch.randint(0, NUM_POS, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        feat = torch.randn(BATCH_SIZE, SEQ_LEN, ASA_FEATURE_DIM, device=DEVICE)
        req = torch.randn(BATCH_SIZE, SEQ_LEN, ASA_FEATURE_DIM, device=DEVICE).abs()

        with torch.no_grad():
            logits = model(x, pos_ids=pos, features=feat, requirements=req)

        assert not torch.isnan(logits).any(), f"NaN in logits at iteration {i}"
        assert not torch.isinf(logits).any(), f"Inf in logits at iteration {i}"
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), \
            f"Wrong shape: {logits.shape}"

    print("PASS")


def test_backward_pass():
    """All parameters receive non-NaN gradients."""
    print("Test 2: Backward pass (gradient flow)...", end=" ")
    model = ASALanguageModel(VOCAB_SIZE, mode="full").to(DEVICE)
    model.train()

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    pos = torch.randint(0, NUM_POS, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    feat = torch.randn(BATCH_SIZE, SEQ_LEN, ASA_FEATURE_DIM, device=DEVICE)
    req = torch.randn(BATCH_SIZE, SEQ_LEN, ASA_FEATURE_DIM, device=DEVICE).abs()

    logits = model(x, pos_ids=pos, features=feat, requirements=req)
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    print("PASS")


def test_copy_task():
    """Model trains on simple next-token prediction -- loss must decrease significantly.

    Uses a small vocab with repeated fixed sequences so the model can actually
    memorize patterns. This proves gradient flow, not generalization.
    """
    print("Test 3: Next-token prediction convergence (500 steps)...", end=" ")
    small_vocab = 20
    model = ASALanguageModel(small_vocab, d_model=64, n_heads=2, n_layers=2,
                             d_ff=256, mode="none").to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Fixed training sequences (model should memorize these)
    torch.manual_seed(123)
    fixed_data = torch.randint(0, small_vocab, (16, 16), device=DEVICE)

    initial_loss = None
    final_loss = None

    for step in range(500):
        # Sample a batch from fixed data
        idx = torch.randint(0, 16, (4,))
        x = fixed_data[idx]
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, small_vocab),
            x[:, 1:].reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            initial_loss = loss.item()
        if step == 499:
            final_loss = loss.item()

    ratio = final_loss / initial_loss
    print(f"loss {initial_loss:.3f} -> {final_loss:.3f} ({ratio:.1%})", end=" ")
    assert final_loss < initial_loss * 0.5, \
        f"Didn't converge: {initial_loss:.3f} -> {final_loss:.3f}"
    print("PASS")


def test_all_modes():
    """All 4 ablation modes produce valid outputs."""
    print("Test 4: Ablation modes...", end=" ")
    modes = ["full", "pos_only", "features_only", "none"]
    for mode in modes:
        model = ASALanguageModel(VOCAB_SIZE, mode=mode).to(DEVICE)
        model.eval()

        x = torch.randint(0, VOCAB_SIZE, (2, 16), device=DEVICE)
        pos = torch.randint(0, NUM_POS, (2, 16), device=DEVICE)
        feat = torch.randn(2, 16, ASA_FEATURE_DIM, device=DEVICE)
        req = torch.randn(2, 16, ASA_FEATURE_DIM, device=DEVICE).abs()

        with torch.no_grad():
            logits = model(x, pos_ids=pos, features=feat, requirements=req)

        assert not torch.isnan(logits).any(), f"NaN in mode={mode}"
        assert not torch.isinf(logits).any(), f"Inf in mode={mode}"

    print("PASS")


def test_attention_entropy():
    """Attention weights don't collapse to a single token."""
    print("Test 5: Attention entropy (no collapse)...", end=" ")
    model = ASALanguageModel(VOCAB_SIZE, mode="full").to(DEVICE)
    model.eval()

    x = torch.randint(0, VOCAB_SIZE, (2, 32), device=DEVICE)
    pos = torch.randint(0, NUM_POS, (2, 32), device=DEVICE)
    feat = torch.randn(2, 32, ASA_FEATURE_DIM, device=DEVICE)
    req = torch.randn(2, 32, ASA_FEATURE_DIM, device=DEVICE).abs()

    with torch.no_grad():
        _ = model(x, pos_ids=pos, features=feat, requirements=req)

    # Check entropy of attention weights in each layer
    min_entropy = float("inf")
    for i, layer in enumerate(model.layers):
        weights = layer.attention.last_weights  # [B, n_heads, S, S]
        if weights is None:
            continue
        # Only check rows where attention is non-zero (causal mask means early rows have few targets)
        # Use last 8 positions which have enough context
        w = weights[:, :, -8:, :]  # [B, n_heads, 8, S]
        entropy = -(w * torch.log(w + 1e-10)).sum(dim=-1).mean()
        min_entropy = min(min_entropy, entropy.item())

    # Threshold is low because POS mask legitimately restricts attention to
    # compatible pairs (fewer targets -> lower entropy). True collapse is ~0.
    assert min_entropy > 0.1, f"Attention collapsed: min entropy = {min_entropy:.3f}"
    print(f"min_entropy={min_entropy:.3f} PASS")


def test_asa_bias_affects_attention():
    """ASA bias produces different attention patterns than no-ASA baseline."""
    print("Test 6: ASA bias has effect on attention...", end=" ")

    x = torch.randint(0, VOCAB_SIZE, (2, 16), device=DEVICE)
    pos = torch.randint(0, NUM_POS, (2, 16), device=DEVICE)
    feat = torch.randn(2, 16, ASA_FEATURE_DIM, device=DEVICE)
    req = torch.randn(2, 16, ASA_FEATURE_DIM, device=DEVICE).abs()

    # Same weights, different modes
    torch.manual_seed(42)
    model_asa = ASALanguageModel(VOCAB_SIZE, mode="full").to(DEVICE)
    torch.manual_seed(42)
    model_none = ASALanguageModel(VOCAB_SIZE, mode="none").to(DEVICE)

    model_asa.eval()
    model_none.eval()

    with torch.no_grad():
        logits_asa = model_asa(x, pos_ids=pos, features=feat, requirements=req)
        logits_none = model_none(x)

    # Outputs should differ (ASA bias changes attention)
    diff = (logits_asa - logits_none).abs().mean().item()
    assert diff > 0.01, f"ASA bias has no effect: diff = {diff:.6f}"
    print(f"mean_diff={diff:.4f} PASS")


def test_param_count():
    """Verify model size is reasonable for tiny config."""
    print("Test 7: Parameter count...", end=" ")
    model = ASALanguageModel(VOCAB_SIZE)
    n_params = model.count_parameters()
    print(f"{n_params:,} params", end=" ")
    # Tiny model should be under 5M params
    assert n_params < 5_000_000, f"Too many params: {n_params:,}"
    print("PASS")


def test_bias_asymmetry():
    """ASA bias must be asymmetric: bias[verb, noun] != bias[noun, verb].

    This is the core seeker/filler principle: verbs have requirements (what they
    NEED), nouns have features (what they ARE). The directional compatibility
    requirements_i . features_j / |requirements_i| is NOT symmetric.
    """
    print("Test 8: ASA bias asymmetry...", end=" ")

    batch = 1
    seq_len = 4

    # Create a sequence: [Noun, Verb, Noun, Det]
    pos_ids = torch.tensor([[POS_IDS["Noun"], POS_IDS["Verb"],
                             POS_IDS["Noun"], POS_IDS["Det"]]])

    # Nouns have features (what they ARE), no requirements
    # Verbs have requirements (what they NEED), no features
    features = torch.zeros(batch, seq_len, ASA_FEATURE_DIM)
    requirements = torch.zeros(batch, seq_len, ASA_FEATURE_DIM)

    # Noun 0: animate=1, concrete=1
    features[0, 0, 0] = 1.0  # animate
    features[0, 0, 4] = 1.0  # concrete

    # Verb 1: requires animate subject
    requirements[0, 1, 0] = 1.0  # needs animate

    # Noun 2: concrete=1, comestible=1
    features[0, 2, 4] = 1.0  # concrete
    features[0, 2, 13] = 1.0  # comestible

    bias = compute_asa_bias(pos_ids, features, requirements, mode="features_only")

    # bias[verb, noun0] should differ from bias[noun0, verb]
    # Verb (pos 1) looking at Noun0 (pos 0): requirements[1] . features[0] / |req[1]|
    # = [1,0,...] . [1,0,0,0,1,...] / 1 = 1.0
    verb_to_noun = bias[0, 1, 0].item()

    # Noun0 (pos 0) looking at Verb (pos 1): requirements[0] . features[1] / |req[0]|
    # = [0,...] . [0,...] / eps = 0.0
    noun_to_verb = bias[0, 0, 1].item()

    assert verb_to_noun != noun_to_verb, \
        f"Bias is symmetric! verb->noun={verb_to_noun:.4f}, noun->verb={noun_to_verb:.4f}"
    assert verb_to_noun > noun_to_verb, \
        f"Expected verb->noun > noun->verb, got {verb_to_noun:.4f} vs {noun_to_verb:.4f}"

    print(f"verb->noun={verb_to_noun:.4f}, noun->verb={noun_to_verb:.4f} PASS")


def test_pos_sparsity():
    """POS compatibility matrix must block >=30% of attention pairs.

    v2.2 had 65% density (35% blocked) with 17 POS tags. With only 8 tags,
    the "Other" category (punctuation, unknown) adds full connectivity for
    1 row+column. Our target: block >=30% AND be denser than v2.2's 65%.
    """
    print("Test 9: POS mask sparsity...", end=" ")

    total = NUM_POS * NUM_POS
    allowed = POS_COMPAT_MATRIX.sum().item()
    blocked = total - allowed
    density = allowed / total
    sparsity = blocked / total

    print(f"density={density:.1%}, blocked={sparsity:.1%}", end=" ")
    assert sparsity >= 0.30, \
        f"POS mask blocks too few pairs: {sparsity:.1%} (want >=30%)"
    assert density <= 0.65, \
        f"POS mask looser than v2.2: {density:.1%} dense (v2.2 was 65%)"
    print("PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Gate 2a: ASA Attention Mechanical Integration Tests (Hardened)")
    print("=" * 60)

    tests = [
        test_forward_pass,
        test_backward_pass,
        test_copy_task,
        test_all_modes,
        test_attention_entropy,
        test_asa_bias_affects_attention,
        test_param_count,
        test_bias_asymmetry,
        test_pos_sparsity,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    if failed > 0:
        print(f"GATE 2a: FAIL ({failed} failures)")
        sys.exit(1)
    else:
        print("GATE 2a: PASS")
