---
tags: [reference]
---

# ASA (Atomic Semantic Architecture) — Article Reference Document

**Author:** JKILLR
**Development Period:** December 2025 – January 2026
**Purpose:** This document crystallizes the concepts, evolution, and findings from the ASA research project to serve as a comprehensive brief for writing a conceptual article.

---

## 1. The Central Thesis

**Current transformers learn structure that already exists in language. ASA proposes encoding that known structure directly, so the model doesn't have to rediscover it from billions of parameters.**

The core insight: attention sparsity — which tokens should attend to which — can be *predetermined* from semantic and syntactic properties rather than learned. This is analogous to how chemistry predicts which atoms can bond based on their electron configurations, without needing to "learn" bonding from data.

---

## 2. The Chemistry Metaphor (and why it's more than a metaphor)

ASA maps linguistic properties onto atomic physics concepts:

| Atomic Property | Linguistic Analogue | What It Encodes |
|----------------|---------------------|-----------------|
| **Nucleus** (proton count) | POS tag + WordNet supersense | Core identity: "this is a noun about a person" |
| **Mass** (neutron count) | Word frequency/stability | Resistance to meaning shift |
| **Electron shells** | Selectional features (VerbNet) | Semantic type: [+animate], [+concrete], [+human] |
| **Valence electrons** | Thematic role potential | Bonding capacity: what roles a word can fill or assign |
| **Charge** (+/-) | Activation state | Seeking bonds (-) vs. stable (+); flips with negation |
| **Energy level** | Working memory activation | Ground state vs. actively attended |

This isn't arbitrary analogy — it maps directly onto established linguistic frameworks:
- **VerbNet** selectional restrictions → shell configurations
- **Universal Dependencies** syntactic relations → bonding rules
- **Binding Theory** (Chomsky) → coreference compatibility
- **WordNet** taxonomy → hypernym-based feature assignment

---

## 3. The Three Key Innovations

### 3.1 Predetermined Sparsity (Bonding Rules)

Standard transformers compute attention between *all* token pairs — O(n²). Sparse variants like Longformer use *positional* heuristics (local windows + global tokens). ASA uses *semantic* rules:

- **Level 1 — POS Compatibility:** A binary filter from Universal Dependencies. Adverbs don't modify nouns. Determiners seek nouns. Verbs seek nominal arguments. This alone eliminates many unnecessary attention computations.

- **Level 2 — Feature Compatibility:** VerbNet selectional restrictions. "examined" requires an [+animate] subject. "doctor" is [+animate, +human, +concrete]. "rock" is [+concrete] only. So "She" (pronoun seeking [+animate, +human] antecedent) can bond with "doctor" but not "rock" — *regardless of distance*.

The key advantage over positional sparsity: semantic bonding rules don't decay with distance. A pronoun 2,000 tokens away from its antecedent still has a bonding pathway if the semantic types are compatible.

### 3.2 Thermodynamic Attention Scoring

Instead of the standard dot-product attention score (Q·K^T / √d), ASA uses a Gibbs Free Energy formulation:

```
Standard:  Score = Q·K^T / √d              (pure similarity)
ASA:       Score = Q·K^T / √d - λ_H·ΔH + λ_C·(-q_i × q_j)
```

Where:
- **ΔH (Enthalpy):** Measures feature alignment — how well the semantic shells of two tokens match. Lower ΔH = stronger bond.
- **Charge Interaction (-q_i × q_j):** Opposite charges attract. Verbs (charge: -0.5, "seeking arguments") attract nouns (charge: +0.5, "stable"). This gives structure to attention beyond mere embedding similarity.
- **Entropy Regularization (-T · Σ w·log(w)):** Prevents attention collapse — a known problem in standard transformers where attention concentrates on few tokens.
- **Temperature (T):** A learnable parameter enabling runtime switching between focused retrieval (low T) and creative/exploratory attention (high T).

The entropy penalty is particularly novel: it *structurally prevents* the attention collapse problem that standard transformers suffer from, where attention degenerates into near-one-hot distributions.

### 3.3 Charge System for Negation

Standard embeddings notoriously fail at negation: "I love this" ≈ "I don't love this" (similar vectors). ASA's charge system explicitly models polarity:

- Each token has a charge value (-1 to +1)
- Negation words ("not", "never", "no") *flip the charge* of governed tokens
- "succeed" has positive charge (+0.7); "not succeed" has negative charge (-0.7)
- These become opposite in the embedding space, making negation semantically meaningful

---

## 4. Evolution of the Project (4 Phases)

### Phase 1: "ASA new" / ASA 7.0 (Late December 2025)
**The theoretical foundation and first implementations.**

- Wrote the full specification (ASA_SPEC_1_0.md, ~2500 lines)
- Had it reviewed by Gemini, which identified the hardware bottleneck as the critical challenge
- Built the first complete implementation mapping all 5 linguistic properties (POS, selectional features, thematic roles, dependency relations, WordNet supersenses) → atomic structure → bonding rules → thermodynamic attention
- Designed a benchmark suite against Longformer to test the fundamental question: *does semantic sparsity outperform positional sparsity?*
- Prototyped block-sparse adaptation for GPU compatibility
- Key finding: the concept was sound but element-wise sparsity was GPU-hostile

### Phase 2: "Unity" (Late December 2025 – Early January 2026)
**Grounding the extraction pipeline in real linguistic resources.**

- Built a rigorous property extraction system with proper WordNet hypernym chain walking
- Developed two-level bonding: POS compatibility (coarse) + feature compatibility (fine)
- Created VerbNet-derived verb frames with proper argument structure (perception verbs need [+animate] experiencers, communication verbs need [+human] agents, etc.)
- Expanded verb coverage to ~300 verbs from VerbNet
- Key insight crystallized: "Properties actually determine bonding" — the bonding rules weren't just heuristics, they were linguistically grounded predictions about which tokens *should* attend to each other

### Phase 3: "ASA v2.2" (January 2026)
**The trainable implementation with real experiments.**

- Built a complete, trainable transformer with ASA attention as a drop-in replacement
- Added ablation modes: full ASA, POS-only, features-only, and standard baseline (mode="none")
- Implemented subword alignment for BPE/WordPiece tokenizers (handling the mismatch between linguistic analysis at the word level and model tokenization at the subword level)
- Trained on WikiText-2 with comparison between ASA modes and standard transformer baseline
- Training configs preserved show experiments at both "tiny" and "small" model sizes
- Included preprocessing/caching utilities for practical training workflows

### Phase 4: "ASA Wave" (January 2026)
**The radical evolution — from attention modification to physics simulation.**

This is where the project took its most ambitious turn.

**v1-v2: Wave Functions.** Instead of discrete categorical properties, represent each token as a 20-dimensional wave function over relational bases (DET_NOUN, ADJ_NOUN, SUBJ_PRED, AGENT_ACTION, ANIMACY_FIELD, etc.). The amplitude in each dimension encodes how strongly a token participates in that relation. Wave function overlap naturally encodes compatibility — high overlap means tokens want to bond, orthogonal waves mean no attraction.

**v3: Semantic Molecular Dynamics (SMD).** The biggest conceptual leap — *bonding IS the computation*. Instead of using linguistic structure to *modify* attention, use it to *replace* attention entirely:

- Words become particles with mass, charge, and valency frames
- Grammatical relations are *forces* (selectional attraction, structural constraints)
- Parsing is *energy minimization* — structure emerges from dynamics, not from a learned attention matrix
- Based on Natural Semantic Metalanguage (NSM) — 65 universal semantic primes that organize into 7 orbital levels
- 7 physically-motivated energy terms: valency (electron shells → required arguments), selectional (Coulomb → type compatibility), saturation (steric repulsion → role limits), locality (bond strain → prefer local dependencies), exclusivity (Pauli exclusion → one role = one filler), coherence (van der Waals → semantic fields), projection (hybridization → head-phrase typing)
- Complexity: O(n×k) where k is average bonds per word (~2-4), vs O(n²) for transformers

**v3 continued: Syntactic Electrostatics.** Discovered through debate with Gemini, this addresses the long-range dependency problem. Local molecular forces decay with distance, but wh-movement ("What did John think Mary believed Bill said Sarah saw ___?") requires long-range binding:

- Wh-words carry positive charge (+Q)
- Gaps (unfilled theta-roles) carry negative charge (-Q)
- Island constraints become regions of high dielectric constant (ε >> 1) — they *screen* the electrostatic field
- Implemented as a Poisson equation solver: ∇·(ε(x)∇φ) = -ρ
- Achievable in O(n) via Fast Multipole Method
- Key predictions: Ross's island constraints emerge as dielectric screening, intervention effects as field distortion, superiority effects as Coulomb repulsion between wh-words, parasitic gaps as multi-pole equilibrium

---

## 5. The Big Ideas Worth Articulating

### 5.1 "Encode what we know, learn what we don't"

The philosophical core. Linguistics has centuries of scholarship on how language works — Universal Dependencies, VerbNet selectional restrictions, Binding Theory, semantic primes. Current ML rediscovers this from scratch using billions of parameters. ASA proposes encoding known structure directly and only learning what we genuinely don't know (token embeddings, fine-grained semantic relationships).

### 5.2 The Democratization Argument

If structure is in the physics (the rules, the energy function) rather than in the weights (billions of learned parameters), language understanding becomes dramatically cheaper:

- GPT-4: ~1.8T parameters (rumored)
- ASA/SMD core structure: ~5K-16K parameters (plus frozen pre-trained embeddings)
- 100,000,000× reduction

This means potentially running sophisticated language models on smartphones, embedded devices, edge computing, and developing-world infrastructure.

### 5.3 Attention Sparsity: Semantic vs. Positional

The deepest empirical question ASA poses: when you skip attention computation, *which* computations should you skip?

- **Longformer/BigBird:** Skip based on *position* — attend locally + a few global tokens
- **Learned sparsity:** Skip based on what the model *learned* to ignore
- **ASA:** Skip based on *semantic incompatibility* — adverbs shouldn't attend to nouns, pronouns should always attend to compatible antecedents

The hypothesis: at equivalent sparsity levels, semantic selection should outperform positional selection for tasks requiring long-range understanding (coreference, argument structure, negation scope).

### 5.4 Language as Physics

The SMD framework proposes that parsing is energy minimization, not pattern matching. Grammatical structure *emerges* from the interaction of particles with known properties — the same way molecular structure emerges from atomic forces. This is a fundamentally different computational paradigm from transformers.

### 5.5 The Negation Problem as a Canary

That "I love this" ≈ "I don't love this" in standard embeddings is a symptom of a deeper issue: current architectures have no principled mechanism for polarity. ASA's charge system — where negation flips charge — is a simple, interpretable solution that directly addresses a known failure mode.

### 5.6 Resonance States for Ambiguity

ASA borrows quantum mechanics' superposition concept for lexical ambiguity. A word like "bank" exists in a resonance state (multiple weighted interpretations: financial institution vs. river bank) until sufficient context "collapses" it to a single interpretation. This is more principled than either pure symbolic disambiguation or purely learned contextual embeddings.

---

## 6. Honest Assessment: What Works, What Doesn't, What's Unproven

### What's Strong
- The theoretical framework is internally consistent and grounded in established linguistics
- The chemistry/physics metaphor maps precisely (not loosely) onto linguistic concepts
- The negation/charge system solves a real, known problem
- The benchmark design against Longformer asks the right question (semantic vs. positional sparsity)
- The entropy penalty addresses a real problem (attention collapse)

### What's Challenging
- **Hardware reality:** Element-wise sparsity is GPU-hostile. Block-sparse adaptation was developed but loses fine-grained semantic distinctions
- **Rigidity of rules:** Hard bonding rules break on creative language, metaphor, poetry. Soft bonding (attention bias) was proposed but adds complexity
- **POS tagger dependency:** 3-5% POS error rate propagates into bonding rules. End-to-end learning of categories was proposed but not fully implemented
- **Double memory bandwidth:** Thermodynamic scoring requires computing both enthalpy and entropy per attention score

### What's Unproven
- Large-scale training results comparing ASA vs. standard transformers aren't in the files (training infrastructure exists, small experiments were run, but no definitive results)
- The SMD/physics simulation approach is a prototype — impressive conceptual framework but not validated at scale
- The syntactic electrostatics predictions (island constraints, etc.) are simulated but not tested against real linguistic data
- Whether semantic sparsity actually outperforms positional sparsity in practice remains an open question

---

## 7. Key References from the Project

### Linguistic Foundations
- Universal Dependencies v2 (Nivre et al.) — POS compatibility rules
- VerbNet 3.4 (Kipper-Schuler et al.) — Selectional restrictions and verb argument frames
- WordNet 3.0 (Miller) — Semantic taxonomy and hypernym chains
- Binding Theory (Chomsky) — Coreference constraints
- Natural Semantic Metalanguage (Wierzbicka, 1996) — 65 semantic primes
- Ross (1967) — Island Constraints
- Pesetsky (1987) — Superiority Effects

### ML/AI Context
- FlashAttention (Dao et al., 2022) — IO-aware attention
- Block-Sparse FlashAttention (Ohayon et al., Dec 2025) — 50% sparsity, 1.1-1.24x speedup
- SchNet, DimeNet, GemNet — Molecular GNNs that validate encoding atomic structure into embeddings
- Nickel & Kiela (2017) — Poincaré Embeddings for hierarchical representations
- Greengard & Rokhlin (1987) — Fast Multipole Method for O(N) n-body simulation

### Hardware Landscape
- TPU v7 SparseCore — Fine-grained sparse operations
- Graphcore IPU — MIMD architecture, 5-10x throughput for sparse models
- FlashAttention-3 — 230 TFLOPs/s on H100

---

## 8. Suggested Article Angles

1. **"What If Attention Already Knew Where to Look?"** — Focus on the semantic vs. positional sparsity insight. Accessible, practical, ties to efficiency concerns.

2. **"The Periodic Table of Language"** — Lead with the chemistry metaphor. Explain how linguistic properties map to atomic structure and predict bonding. More conceptual/visionary.

3. **"Language Processing as Physics Simulation"** — Lead with the SMD evolution. Most ambitious framing — parsing as energy minimization, grammatical forces, Coulombic syntax.

4. **"The 100-Million-X Parameter Reduction"** — Lead with the democratization argument. If structure is in the physics, not the weights, what does that mean for access and cost?

5. **"What Transformers Rediscover (and What They Miss)"** — Focus on negation, long-range coreference, and attention collapse. Problems ASA solves that transformers struggle with.

---

## 9. File Map for Reference

```
ASA/
├── ASA new/ASA 7.0/          # Phase 1: Theory + first implementation
│   ├── ASA_IMPLEMENTATION_ROADMAP.md    # Synthesized analysis + path forward
│   ├── ASA_BENCHMARK_SUITE.md           # Benchmark design vs Longformer
│   ├── asa_proper_implementation.py     # Full pipeline: extraction → bonding → thermo attention
│   ├── asa_attention.py                 # Standalone attention layer with demos
│   ├── asa_block_sparse_prototype.py    # GPU-compatible block sparse adaptation
│   └── asa_benchmark_fixed.py           # Coreference benchmark (fixed methodology)
│
├── Unity/                     # Phase 2: Rigorous linguistic grounding
│   ├── asa_extraction_v2.py             # Property extraction with full WordNet/VerbNet
│   └── asa_unified_system.py            # Two-level bonding: POS + features
│
├── ASA v2.2/                  # Phase 3: Trainable implementation
│   ├── asa_v2_2.py                      # Complete model with ablation modes
│   ├── train_asa.py                     # Training script (WikiText-2)
│   ├── quick_start.py                   # Quick validation script
│   └── asa_output/                      # Training checkpoints (full/none × tiny/small)
│
└── ASA Wave/                  # Phase 4: Wave functions → physics simulation
    ├── v2/asa_wave_extraction.py        # Wave functions (21-dim)
    ├── v3/asa_wave_extraction_v3.py     # Wave functions v3 (20-dim, L2 normalized)
    ├── v3/semantic_molecular_dynamics.py # SMD: bonding IS the computation
    ├── v3/syntactic_electrostatics.py   # Coulombic syntax (long-range forces)
    ├── v3/unified_smd.py                # Local + long-range unified
    └── v3/SEMANTIC_MOLECULAR_DYNAMICS.md # SMD overview document
```
