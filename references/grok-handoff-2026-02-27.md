---
tags: [reference]
---

# Grok Handoff — Phase 0 Roadmap

**Date:** 2026-02-27
**From:** Grok (xAI), based on initial discussions
**To:** Claude Code

## Summary

Grok was used to discuss the ASA concept and build the initial [[asa_toy.py]] prototype with greedy bonding, and defined a 4-phase roadmap. The prototype had a bonding failure (only formed 1 bond out of expected 3-4) due to nouns being excluded from the unsaturated token search.

## Roadmap Defined

| Phase | Goal | Timeline | Resources |
|-------|------|----------|-----------|
| 0 | Fix toy, get multiple bonds forming, add visualization | 1-3 days | Laptop only |
| 1 | Expand vocab 50-100, implement SMD with scipy, add temperature, test on tiny datasets | 1-2 weeks | Free Colab |
| 2 | Scale to 10M-50M params, pre-train on small corpus, evaluate on GLUE/ARC | 2-4 weeks | Rented GPU ~$20-50 |
| 3 | Hyperbolic embeddings, Context Synthesizer, open-source, ArXiv | Ongoing | GPU as needed |

## Phase 0 Issues Identified by Grok

1. Greedy selection locks on first bond, remaining pairs don't score high enough
2. Valence penalty was crushing scores
3. Directional scoring not capturing verb-initiated pulls
4. No thermodynamic randomness — pure greedy is myopic
5. No visualization
6. No generation (just parsing)

## Resolution (by Claude Code)

Root cause was different than suspected: nouns had valence 0, so `unsaturated_indices()` never returned them. The bonding loop only saw seekers (verbs, dets, adj, adv), never fillers (nouns). Fixed by redesigning as seeker→filler paradigm with POS compatibility filter and locality bias.
