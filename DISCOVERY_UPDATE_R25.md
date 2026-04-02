# Discovery Agent Update — Rounds 22-25 (for Periodic Agent)

## Summary
Discovery agent went from F1=0.592 (round 21, your snapshot) to F1=0.636 (round 25).
Key: P=0.728 R=0.572. 7787/13701 gold bonds. +71.9% total improvement from baseline.

## NEW PROPERTIES (transformer-portable)

### 1. Morphological Frame (dim 1 on nouns)
Deverbal noun suffixes predict PP complement attachment.
Words ending in -tion/-sion (0.9), -ment (0.8), -ance/-ence (0.6) → attract PP bonds.
This is a pure NUMERIC PROPERTY: every noun gets a value 0-1 based on suffix.
Portable: in PropertyExtractor, check noun suffix → set feature dim.

### 2. WordNet Verb Frames (per-verb properties)
get_verb_frames() extracts 4 continuous properties per verb from WordNet frame IDs:
- intransitive: 0-1 (purely intrans=1.0, mixed=0.5, trans=0.0)
- clausal: 0-1 (takes clause complement)
- animate_subj: 0-1 (prefers animate subject)
- pp_complement: 0-1 (takes PP complement)
These override valence and set dim 2 = animate_subj preference.
Portable: WordNet is already in train.py. Map frame IDs → 4 float dims.

### 3. Copula as distinct verb class
Copula verbs (is/was/are/were/be/been/seem/seemed/become/became/remain/appeared)
get separate features: weaker head, seeks predicate (adj/noun) not object.
Portable: word-level check in PropertyExtractor.

## STRUCTURAL FINDINGS (harder to port as features)
- Projectivity: 79% of crossing bonds are spurious → projectivity constraint helps
- NP chunking: Det-Adj-Noun sequences bonded as units improves recall
- Long-range V→V chains: verbs bond to verbs at distance (auxiliary chains)
- These are POST-PROCESSING rules, not per-word features. May not help as attention bias.

## RECOMMENDATION
The ablation showed pure POS mask ≈ real features. These new properties are MORE SPECIFIC
than the broad orbital categories — they might be the kind of fine-grained signal that
actually differentiates from what POS mask alone provides.

Test hypothesis: POS mask + verb frames + morph frame > pure POS mask alone.

## CODE REFERENCE
See eval_treebank.py:
- get_verb_frames() at line 394 — extracts 4 WordNet frame properties
- DEVERBAL_SUFFIXES at line 522 — morphological frame values
- COPULA_WORDS at line 260 — copula detection set
