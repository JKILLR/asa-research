---
tags: [reference]
---

# References

Key papers, frameworks, and external knowledge relevant to ASA research.

**Comprehensive project history:** See [[references/ASA_ARTICLE_REFERENCE|ASA Article Reference]] for the full evolution across all 4 phases with detailed findings.

## Linguistic Frameworks

| Framework | Relevance to ASA | Resource |
|-----------|-----------------|----------|
| **VerbNet 3.4** (Kipper-Schuler) | Selectional restrictions, thematic roles, verb classification. ~300 verbs extracted in Phase 2. | https://verbs.colorado.edu/verbnet/ |
| **Universal Dependencies v2** (Nivre et al.) | POS compatibility rules, syntactic relation types | https://universaldependencies.org/ |
| **Binding Theory** (Chomsky 1981) | Coreference constraints (pronouns, anaphora) | Lectures on Government and Binding |
| **WordNet 3.0** (Miller) | Semantic taxonomy, hypernym chain walking for feature assignment | https://wordnet.princeton.edu/ |
| **Natural Semantic Metalanguage** (Wierzbicka 1996) | 65 semantic primes → 7 orbital levels in SMD | |
| **X-bar Theory** | Hierarchical phrase structure, headedness | |
| **Construction Grammar** | Form-meaning pairings as primary units | |
| **Ross (1967)** | Island Constraints — modeled as dielectric screening in syntactic electrostatics | |
| **Pesetsky (1987)** | Superiority Effects — modeled as Coulomb repulsion between wh-words | |

## ML/AI Research

| Paper/Project | Relevance | Notes |
|---------------|-----------|-------|
| **Poincaré Embeddings** (Nickel & Kiela, 2017) | Hyperbolic space for tree structures | Foundation for non-projectivity solution |
| **FlashAttention** (Dao et al., 2022) | IO-aware attention computation | Baseline for performance comparison |
| **FlashAttention-3** | 230 TFLOPs/s on H100 | Current state-of-art dense attention speed |
| **Block-Sparse FlashAttention** (Ohayon et al., Dec 2025) | 50% sparsity → 1.1-1.24x speedup | Relevant to ASA's block-sparse adaptation |
| **SchNet, DimeNet, GemNet** | Molecular GNNs encoding atomic structure into embeddings | Validates the "encode structure" approach in chemistry domain |
| **Greengard & Rokhlin (1987)** | Fast Multipole Method for O(N) n-body simulation | Enables O(n) syntactic electrostatics |

## Hardware Landscape

| Platform | Relevance |
|----------|-----------|
| **TPU v7 SparseCore** | Fine-grained sparse operations — best fit for ASA's element-wise sparsity |
| **Graphcore IPU** | MIMD architecture, 5-10x throughput for sparse models |
| **H100 GPU** | Dense-optimized; requires block-sparse adaptation for ASA |

## Prior ASA Code (sibling directories under `ASA/`)

| Directory | Phase | Key Files |
|-----------|-------|-----------|
| `ASA new/ASA 7.0/` | Phase 1 | `asa_proper_implementation.py`, `asa_attention.py`, `asa_block_sparse_prototype.py`, `ASA_BENCHMARK_SUITE.md` |
| `Unity/` | Phase 2 | `asa_extraction_v2.py`, `asa_unified_system.py` |
| `ASA v2.2/` | Phase 3 | `asa_v2_2.py`, `train_asa.py`, `quick_start.py` |
| `ASA Wave/` | Phase 4 | `v2/asa_wave_extraction.py`, `v3/semantic_molecular_dynamics.py`, `v3/syntactic_electrostatics.py`, `v3/unified_smd.py` |

---

## How to Use This Directory

- [[references/ASA_ARTICLE_REFERENCE|ASA Article Reference]] — comprehensive project history (read for full context)
- Add `.md` summaries of important papers with key takeaways
- Link to external resources, don't copy large texts
- Use naming: `[topic]-[author-or-source].md`
