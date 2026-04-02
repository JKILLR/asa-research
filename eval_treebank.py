"""
Evaluate ASA toy on NLTK dependency treebank.

This is the MICROSCOPE for discovering new structural elements.
Each failure = a missing element in the periodic table of language.

Speed 1: seconds per test, no GPU, no learning. Pure structure.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import nltk

nltk.download('dependency_treebank', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import dependency_treebank, wordnet as wn
from nltk.stem import WordNetLemmatizer

# Import toy bonding system
from asa_toy import (
    Token, POS_RULES, pos_compatible,
    wave_overlap, LAMBDA_VAL, LAMBDA_SAT, LAMBDA_EXCL,
)
import asa_toy

# Extend feature space: dims 0-9 = KEY (provides), dims 10-19 = QUERY (seeks)
# Plus dims 20-23 for structural properties
# Expand to 28 dims: +4 for selectional preference vectors
FEATURE_DIM = 28
asa_toy.FEATURE_DIM = FEATURE_DIM

# Load selectional preference vectors
import os as _os2
_sel_path = _os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), 'selectional_vectors_combined.npz')
if _os2.path.exists(_sel_path):
    _sel_data = np.load(_sel_path, allow_pickle=True)
    _sel_verb_vecs = _sel_data['verb_vecs'][:, :4]  # top 4 SVD components
    _sel_noun_vecs = _sel_data['noun_vecs'][:, :4]
    _sel_v2idx = {str(v): i for i, v in enumerate(_sel_data['top_verbs'])}
    _sel_n2idx = {str(n): i for i, n in enumerate(_sel_data['top_nouns'])}
    print(f"  Selectional vectors: {len(_sel_v2idx)} verbs, {len(_sel_n2idx)} nouns, 4d")
else:
    _sel_verb_vecs = np.zeros((1, 4)); _sel_noun_vecs = np.zeros((1, 4))
    _sel_v2idx = {}; _sel_n2idx = {}
    print("  No selectional vectors found")

# ── KEY-QUERY LAYOUT ──
# KEY (what I provide):     QUERY (what I seek):
#   0: NP_provider            10: NP_seeker (seeks NP things)
#   1: VP_provider            11: VP_seeker (seeks VP things)
#   2: ARG_provider           12: ARG_seeker (seeks arguments)
#   3: HEAD_provider          13: HEAD_seeker (seeks heads)
#   4-9: semantic keys        14-19: semantic queries
#
# cross_score = sum(seeker[10+i] * filler[i] for i in range(10))
# This is ASYMMETRIC: Det(query NP=1) × Noun(key NP=1) → 1
#                      Noun(query NP=0) × Det(key NP=0) → 0  (nouns don't seek dets)

# EXPAND POS_RULES for treebank evaluation
# These represent NEW STRUCTURAL ELEMENTS we're discovering:
asa_toy.POS_RULES["Verb"] = ["Noun", "Pron", "Verb", "Prep"]  # + Prep for V→PP bonds
# Noun compounds handled in post-bonding (directional, left-to-right only)

# ── Round 4: ARGUMENT FRAME SATURATION ──────────────────────
# Problem: 61 spurious Noun↔Verb bonds. Verbs grab nouns that belong to other verbs.
# Fix 1: Modals/auxiliaries are a new POS "Aux" — valence 1, bonds ONLY to Verb.
#         This removes ~half the spurious N↔V bonds (modals were competing for nouns).
# Fix 2: Copula/linking verbs bond to Adj/Noun predicates, not grab extra arguments.
# Fix 3: Increase exclusivity penalty so nouns resist serving two verbs.
# Round 12: Aux chains — Aux bonds to Verb AND other Aux ("will be paid")
# Expanded POS_RULES based on zero-recall analysis
asa_toy.POS_RULES["Aux"] = ["Verb", "Aux"]
asa_toy.POS_RULES["Adj"] = ["Noun"]  # adj modifies nouns only (Adj→Prep moved to post-processing)
asa_toy.POS_RULES["Adv"] = ["Verb", "Aux"]   # adverbs modify verbs + modals
asa_toy.LAMBDA_EXCL = 8.0  # optimal for SA
asa_toy.LAMBDA_SAT = 0.0   # no pressure — post-processing handles most noun bonds
asa_toy.LAMBDA_VAL = 3.0   # optimal


# Fix 4: Distance-dependent verb-noun boost.
#   Spurious N↔V bonds cluster at distance 4+ where gold bonds rarely exist.
#   The 5x boost in wave_overlap is distance-independent and overwhelms locality.
#   Patch: decay the boost so distant nouns are much less attractive.

# ── Round 10: CLAUSE BOUNDARY markers ──
# Commas and subordinators (that, which, because, if...) signal clause edges.
# Gold V↔N bonds almost NEVER cross these boundaries (0% in 9/10 sentences).
# This is a fundamental structural element: SCOPE LIMITERS.
SUBORDINATORS = {'that', 'which', 'who', 'whom', 'whose', 'where', 'when',
                 'because', 'although', 'though', 'if', 'unless', 'while',
                 'since', 'before', 'after', 'whether', 'as'}
SUBORDINATOR_TAGS = {'IN', 'WDT', 'WP', 'WRB'}
_current_tokens = None  # set per sentence for property-based scoring

# ── DISTRIBUTIONAL SELECTIONAL PREFERENCES (from Brown corpus) ──
# Pre-compute verb-noun PPMI for selectional preference scoring.
_wnl = WordNetLemmatizer()

# ── DISTRIBUTIONAL EMBEDDINGS (GloVe 6B, 100d) ──
# 400K words trained on 6 BILLION tokens. Per-word selectional preferences.
# Cosine similarity between word embeddings = compatibility.
import os as _os
_glove_path = _os.path.join(_os.path.dirname(__file__), 'glove.6B.50d.txt')
_glove = {}
if _os.path.exists(_glove_path):
    print("Loading GloVe 100d embeddings...")
    with open(_glove_path, 'r') as _f:
        for _line in _f:
            _parts = _line.strip().split()
            _glove[_parts[0]] = np.array([float(x) for x in _parts[1:]])
    print(f"  GloVe loaded: {len(_glove)} words, 100d")
else:
    print("  No GloVe file found")

# Pre-compute norms for fast cosine
_glove_norms = {w: np.linalg.norm(v) for w, v in _glove.items()}

def glove_similarity(word1: str, word2: str) -> float:
    """Cosine similarity between two words from GloVe embeddings."""
    w1 = word1.lower()
    w2 = word2.lower()
    if w1 in _glove and w2 in _glove:
        v1, v2 = _glove[w1], _glove[w2]
        n1, n2 = _glove_norms[w1], _glove_norms[w2]
        if n1 > 0 and n2 > 0:
            return float(np.dot(v1, v2) / (n1 * n2))
    return 0.0

# Pre-compute average noun similarity for each verb (for relative scoring)
# Use a sample of common nouns as the reference set.
_NOUN_SAMPLE = ['stock', 'company', 'share', 'market', 'price', 'year', 'rate',
                'board', 'plan', 'cent', 'income', 'loss', 'profit', 'debt',
                'trade', 'deal', 'cost', 'tax', 'bond', 'fund', 'bank', 'firm',
                'group', 'unit', 'system', 'issue', 'report', 'program',
                'meeting', 'agreement', 'president', 'director', 'money',
                'time', 'way', 'people', 'state', 'work', 'day', 'man']
_NOUN_SAMPLE_VECS = [_glove[n] / _glove_norms[n] for n in _NOUN_SAMPLE if n in _glove]
_NOUN_SAMPLE_MATRIX = np.array(_NOUN_SAMPLE_VECS) if _NOUN_SAMPLE_VECS else np.zeros((1, 50))
_verb_avg_noun_sim = {}  # cache

def glove_relative_similarity(verb: str, noun: str) -> float:
    """How much BETTER is this V-N similarity than the verb's average with nouns?
    Positive = better than average (good match), negative = worse (bad match)."""
    v = verb.lower()
    n = noun.lower()
    abs_sim = glove_similarity(v, n)
    if abs_sim == 0.0:
        return 0.0
    # Compute verb's average noun similarity (cached)
    if v not in _verb_avg_noun_sim:
        if v in _glove:
            v_norm = _glove[v] / _glove_norms[v]
            sims = _NOUN_SAMPLE_MATRIX @ v_norm
            _verb_avg_noun_sim[v] = float(np.mean(sims))
        else:
            _verb_avg_noun_sim[v] = 0.5
    return abs_sim - _verb_avg_noun_sim[v]

_original_wave_overlap = asa_toy.wave_overlap

def _patched_wave_overlap(seeker, filler, slot_idx=None,
                           seeker_idx=None, filler_idx=None,
                           state=None, complement=None):
    """Patched wave_overlap with clause boundary awareness."""
    score = _original_wave_overlap(seeker, filler, slot_idx=slot_idx,
                                    seeker_idx=seeker_idx, filler_idx=filler_idx,
                                    state=state, complement=complement)


    # Round 9: Extra locality bonus (stacks on top of base 1/d)
    if seeker_idx is not None and filler_idx is not None:
        dist = abs(seeker_idx - filler_idx)
        if dist > 0:
            score += 2.0 / dist  # 3.0/d total locality

    # ── PROPERTY-BASED scope barrier (replaces IF-based clause boundary rule) ──
    # Instead of checking if tokens are commas/subordinators by category,
    # use dim 9 (scope_barrier) — a continuous property every token carries.
    # Bonds crossing high-barrier tokens are penalized proportionally.
    if (seeker_idx is not None and filler_idx is not None and _current_tokens is not None):
        lo, hi = min(seeker_idx, filler_idx), max(seeker_idx, filler_idx)
        if hi - lo > 1:  # only check if there are tokens between
            max_barrier = 0.0
            for mid in range(lo + 1, hi):
                if mid < len(_current_tokens):
                    barrier = _current_tokens[mid].features[9]
                    if barrier > max_barrier:
                        max_barrier = barrier
            if max_barrier > 0.1:
                # Damping = (1 - barrier)^2 for V↔N, (1 - barrier*0.5)^1 for others
                is_vn = ((seeker.pos == "Verb" and filler.pos in ("Noun", "Pron"))
                          or (seeker.pos in ("Noun", "Pron") and filler.pos == "Verb"))
                if is_vn:
                    score *= (1.0 - max_barrier) ** 2  # e.g. comma(0.9) → 0.01
                else:
                    score *= (1.0 - max_barrier * 0.5)  # lighter for non-V↔N

    # Distance-dependent damping now handled by reach property (dim 13)
    # The old V→N specific IF-rule has been replaced by the generic reach penalty above.

    # ── PROPERTY: bond_absorber (dim 14) ──
    # How much does a token between seeker and filler "absorb" the filler?
    # Preps absorb nouns (claim them as complements), modals partially absorb.
    # This replaces the POS-specific Prep/Aux blocking IF-rule.
    if (seeker_idx is not None and filler_idx is not None
            and _current_tokens is not None):
        lo, hi = min(seeker_idx, filler_idx), max(seeker_idx, filler_idx)
        for mid in range(lo + 1, hi):
            if mid < len(_current_tokens):
                absorber = _current_tokens[mid].features[14]
                if absorber > 0.1:
                    score *= (1.0 - absorber)  # prep(0.85) → score *= 0.15


    # ── PROPERTY-BASED directionality + reach (replaces Det/Adj adjacency IF-rules) ──
    # dim 12 = directionality (+1=rightward, -1=leftward)
    # dim 13 = argument_reach (0.1=adjacent only, 1.0=whole clause)
    # Instead of POS-specific IF statements, use the seeker's own properties.
    if seeker_idx is not None and filler_idx is not None:
        signed_dist = filler_idx - seeker_idx  # positive = filler is right of seeker
        directionality = seeker.features[12]    # +1=right-seeking, -1=left-seeking
        reach = seeker.features[13]             # 0.0-1.0

        if reach > 0.01:  # only for seekers (reach > 0)
            abs_dist = abs(signed_dist)
            # Directional bonus: reward bonds in preferred direction, decay with distance
            if directionality > 0.3 and signed_dist > 0:
                score += directionality * 3.0 / abs_dist  # decays: dist=1→3.0, dist=2→1.5
            elif directionality > 0.3 and signed_dist < 0:
                score *= max(0.1, 1.0 - directionality)  # wrong direction penalty
            elif directionality < -0.3 and signed_dist < 0:
                score += abs(directionality) * 2.0 / abs_dist
            elif directionality < -0.3 and signed_dist > 0:
                pass  # verbs need both directions

            # Reach-based distance penalty: bonds beyond reach are penalized
            max_reach_tokens = max(1, int(reach * 10))  # reach 0.3 → 3 tokens
            abs_dist = abs(signed_dist)
            if abs_dist > max_reach_tokens:
                overshoot = abs_dist - max_reach_tokens
                score *= 0.5 ** overshoot  # halve per token beyond reach

    # Prep dual directionality: complement RIGHT (slot 0), head LEFT (slot 1).
    # This is a PROPERTY of prepositions (Janus-faced words), but our single-vector
    # architecture can't encode per-slot directionality. Kept as scoring rule for now.
    if (seeker.pos == "Prep" and filler.pos in ("Noun", "Pron")
            and seeker_idx is not None and filler_idx is not None):
        dist = filler_idx - seeker_idx
        if slot_idx == 0:  # complement — RIGHT
            if dist <= 0:
                score *= 0.1
            elif dist == 1:
                score += 3.0
            elif dist == 2:
                score += 1.0
            else:
                score *= 0.5
        elif slot_idx == 1:  # head — LEFT
            if dist > 0:
                score *= 0.3
            elif dist == -1:
                score += 2.0
            elif dist == -2:
                score += 1.0

    # ── INFORMATION STRUCTURE: GIVENNESS (dim 10) ──
    # Given nouns (dim10 high) prefer SUBJECT slot; new nouns prefer OBJECT slot.
    # 43pp gap: definite→63% subject vs indefinite→20%.
    if (seeker.pos == "Verb" and filler.pos in ("Noun", "Pron")
            and slot_idx is not None and filler_idx is not None):
        givenness = filler.features[10]  # 0=new, 1=given
        if givenness > 0.1:  # has a givenness value
            if slot_idx == 0:  # subject slot
                score *= 1.0 + (givenness - 0.5) * 0.4  # given → boost subject

    # ── NP HEAD PREFERENCE ──
    # Det and Adj should prefer NP-HEAD nouns (last noun before non-noun)
    # over NP-MODIFIER nouns (followed by another noun).
    if (seeker.pos in ("Det", "Adj") and filler.pos == "Noun"
            and filler_idx is not None and _current_tokens is not None):
        # Is this noun an NP head? (not followed by another noun)
        next_idx = filler_idx + 1
        is_np_head = (next_idx >= len(_current_tokens) or
                      _current_tokens[next_idx].pos != "Noun")
        if not is_np_head:
            score *= 0.6  # reduce attraction to non-head nouns

    # ── PREP HEAD PREFERENCE ──
    # Each preposition has an intrinsic verb-vs-noun head preference.
    # "of" almost always modifies a noun (85%); "to" almost always attaches to a verb (88%).
    # dim 11 on Prep tokens encodes this: 1.0 = verb-preferring, 0.0 = noun-preferring
    if (seeker.pos == "Prep" and slot_idx == 1  # head attachment slot
            and seeker_idx is not None and filler_idx is not None):
        verb_pref = seeker.features[11]  # 0=noun-pref, 1=verb-pref
        if filler.pos == "Verb":
            score += verb_pref * 3.0      # bonus proportional to verb preference
        elif filler.pos in ("Noun", "Pron"):
            score += (1.0 - verb_pref) * 3.0  # bonus proportional to noun preference

    # PP-taking verb boost tested — no gain (Verb→Prep already captured by POS rules)

    return max(0.0, score)

asa_toy.wave_overlap = _patched_wave_overlap

# Common auxiliaries and modals (lemmatized forms + inflections)
AUX_WORDS = {
    # Modals (PTB tag MD)
    'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must',
    # Auxiliary "have" forms
    'have', 'has', 'had', 'having',
    # Auxiliary "do" forms
    'do', 'does', 'did',
    # Auxiliary "be" forms (when followed by verb — VBG/VBN)
    'be', 'been', 'being',
}

# Copula/linking verbs: "is", "was", "are", "were", "am" + "'s" (contracted is)
# These bond to predicate (Adj or Noun), not to extra arguments
COPULA_WORDS = {
    'is', 'was', 'are', 'were', 'am', "'s", "'re", "'m",
    'become', 'became', 'seem', 'seemed', 'remain', 'remained',
    'appear', 'appeared',
}

# ── Round 5: Expand verb classification for common WSJ verbs ──
# These are frequent in financial text but missing from VERB_TO_CLASS.
WSJ_VERB_CLASSES = {
    # Intransitive (motion/change of state — no direct object NP)
    'rise': 'motion', 'fell': 'motion', 'fall': 'motion', 'rose': 'motion',
    'drop': 'motion', 'climb': 'motion', 'decline': 'motion', 'slip': 'motion',
    'surge': 'motion', 'soar': 'motion', 'plunge': 'motion', 'tumble': 'motion',
    'rally': 'motion', 'rebound': 'motion', 'dip': 'motion',
    # Transitive (takes NP object)
    'raise': 'transfer', 'add': 'transfer', 'seek': 'transfer',
    'collect': 'transfer', 'impose': 'transfer', 'replace': 'transfer',
    'cost': 'transfer', 'order': 'communication', 'note': 'communication',
    'publish': 'communication', 'rule': 'communication', 'appeal': 'communication',
    'face': 'perception', 'track': 'perception', 'anticipate': 'cognition',
    'elect': 'social', 'record': 'creation', 'regulate': 'social',
    # Stative (typically intransitive in WSJ context)
    'accord': 'stative', 'base': 'stative', 'relate': 'stative',
    'expose': 'change', 'pour': 'motion', 'classify': 'cognition',
    # "said" takes clausal complement, not NP object — valence 1
    'say': 'stative', 'said': 'stative',
}

_wnl = WordNetLemmatizer()

# ═══════════════════════════════════════════════════════════════
# POS MAPPING: Penn Treebank → ASA toy POS
# ═══════════════════════════════════════════════════════════════

PTB_TO_ASA = {
    "NN": "Noun", "NNS": "Noun", "NNP": "Noun", "NNPS": "Noun",
    "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb",
    "VBP": "Verb", "VBZ": "Verb",
    "MD": "Aux",  # Round 4: modals are Aux, not Verb
    "DT": "Det", "PDT": "Det", "WDT": "Det",
    "JJ": "Adj", "JJR": "Adj", "JJS": "Adj",
    "RB": "Adv", "RBR": "Adv", "RBS": "Adv", "WRB": "Adv",
    "IN": "Prep", "TO": "Prep",
    "PRP": "Pron", "PRP$": "Det", "WP": "Pron", "WP$": "Det",  # possessive pronouns act like dets
    # NEW STRUCTURAL CATEGORIES (was all "Other"):
    "CC": "Coord",   # Coordinator: and, or, but — LINKS same-type elements
    "CD": "Num",     # Number: quantifies nouns — MODIFIES like adjective
    "POS": "Poss",   # Possessive 's — BINDS possessor to possessed
    "RP": "Part",    # Particle: up, off, out — ATTACHES to verb
    "EX": "Other",   # Existential there
    # Punctuation stays Other
    ",": "Other", ".": "Other", ":": "Other",
    "``": "Other", "''": "Other", "-LRB-": "Other",
    "-RRB-": "Other", "#": "Other", "$": "Other",
    "FW": "Other", "LS": "Other", "SYM": "Other", "UH": "Other",
}

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (from WordNet, matching toy's 12-dim space)
# ═══════════════════════════════════════════════════════════════

ANIMACY_SYNSETS = {'person.n.01', 'animal.n.01', 'organism.n.01',
                   'living_thing.n.01', 'human.n.01'}
EDIBLE_SYNSETS = {'food.n.01', 'food.n.02', 'fruit.n.01'}
INSTRUMENT_SYNSETS = {'tool.n.01', 'device.n.01', 'artifact.n.01',
                      'instrument.n.01', 'utensil.n.01'}

def get_noun_features(word: str) -> np.ndarray:
    """Extract 12-dim features for a noun from WordNet."""
    features = np.zeros(FEATURE_DIM, dtype=float)
    lower = word.lower()
    synsets = wn.synsets(lower, pos=wn.NOUN)
    if not synsets:
        features[3] = 1.0  # default: physical
        return features

    # Walk hypernym chain
    visited = set()
    queue = [synsets[0]]
    is_animate = False
    is_edible = False
    is_instrument = False

    while queue:
        s = queue.pop(0)
        name = s.name()
        if name in visited:
            continue
        visited.add(name)
        if name in ANIMACY_SYNSETS:
            is_animate = True
        if name in EDIBLE_SYNSETS:
            is_edible = True
        if name in INSTRUMENT_SYNSETS:
            is_instrument = True
        if len(visited) < 15:
            queue.extend(s.hypernyms())

    if is_animate:
        features[0] = 1.0  # animate
        features[3] = 1.0  # physical
    else:
        features[1] = 1.0  # inanimate
        features[3] = 1.0  # physical
    if is_edible:
        features[2] = 1.0  # edible
    if is_instrument:
        features[10] = 1.0  # instrument

    # ── NEW FEATURE DIMENSIONS ──


    return features


# Verb class → requirements (simplified from train.py)
VERB_CLASSES = {
    'perception': {4: 1.0, 5: 1.0},  # needs_subject, needs_object
    'cognition': {4: 1.0, 5: 1.0, 8: 1.0},  # + abstract
    'communication': {4: 1.0, 5: 1.0},
    'motion': {4: 1.0},  # intransitive
    'consumption': {4: 1.0, 5: 1.0},
    'transfer': {4: 1.0, 5: 1.0},
    'emotion': {4: 1.0, 5: 1.0},
    'contact': {4: 1.0, 5: 1.0},
    'creation': {4: 1.0, 5: 1.0},
    'stative': {4: 1.0},
    'change': {4: 1.0, 5: 1.0},
    'social': {4: 1.0, 5: 1.0},
}

# Common verb → class (subset)
from train import VERB_TO_CLASS

def get_verb_frames(word: str) -> Dict[str, float]:
    """Extract subcategorization properties from WordNet verb frames.

    Returns dict with:
      'intransitive': 0-1 (how intransitive is this verb)
      'clausal': 0-1 (takes clausal complement)
      'animate_subj': 0-1 (prefers animate subject)
      'pp_complement': 0-1 (takes PP complement)
    """
    lemma = _wnl.lemmatize(word.lower(), pos='v')
    synsets = wn.synsets(lemma, pos=wn.VERB)
    if not synsets:
        synsets = wn.synsets(word.lower(), pos=wn.VERB)
    if not synsets:
        return {'intransitive': 0.0, 'clausal': 0.0, 'animate_subj': 0.5, 'pp_complement': 0.0}

    # Collect frame IDs from all senses (weighted by frequency)
    all_frames = set()
    for s in synsets[:3]:  # top 3 senses
        for lem in s.lemmas():
            if lem.name().lower() in (lemma, word.lower()):
                all_frames.update(lem.frame_ids())

    intrans_frames = {1, 2, 3, 4, 6, 7, 22, 23}  # "Something/Somebody V", "V PP"
    trans_frames = {8, 9, 10, 11}  # "Somebody V something/somebody"
    ditrans_frames = {14, 15, 16, 17, 18, 19}
    clausal_frames = {26, 28, 29, 34}  # "that CLAUSE", "to INFINITIVE", "whether"
    pp_frames = {4, 13, 15, 16, 17, 18, 19, 20, 21, 22, 27, 30, 31}
    animate_subj_frames = {2, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33}
    inanimate_subj_frames = {1, 3, 4, 5, 6, 10, 11, 34, 35}

    has_intrans = bool(all_frames & intrans_frames)
    has_trans = bool(all_frames & trans_frames)
    has_clausal = bool(all_frames & clausal_frames)
    has_pp = bool(all_frames & pp_frames)
    animate_count = len(all_frames & animate_subj_frames)
    inanimate_count = len(all_frames & inanimate_subj_frames)

    intransitivity = 1.0 if (has_intrans and not has_trans) else (0.5 if has_intrans else 0.0)
    clausal = 1.0 if has_clausal else 0.0
    animate_subj = animate_count / max(animate_count + inanimate_count, 1)
    pp_comp = 1.0 if has_pp else 0.0

    return {
        'intransitive': intransitivity,
        'clausal': clausal,
        'animate_subj': animate_subj,
        'pp_complement': pp_comp,
    }


def get_verb_features(word: str) -> Tuple[np.ndarray, int]:
    """Extract features and valence for a verb."""
    features = np.zeros(FEATURE_DIM, dtype=float)
    lemma = _wnl.lemmatize(word.lower(), pos='v')

    # WSJ_VERB_CLASSES takes priority (Round 9 overrides for reporting/thinking verbs)
    vclass = (WSJ_VERB_CLASSES.get(lemma) or WSJ_VERB_CLASSES.get(word.lower())
              or VERB_TO_CLASS.get(lemma) or VERB_TO_CLASS.get(word.lower()))
    if vclass and vclass in VERB_CLASSES:
        for dim, val in VERB_CLASSES[vclass].items():
            features[dim] = val
        valence = sum(1 for d in [4, 5] if features[d] > 0)
    else:
        # Default: transitive
        features[4] = 1.0  # needs_subject
        features[5] = 1.0  # needs_object
        valence = 2

    return features, valence


def make_token(word: str, ptb_tag: str) -> Token:
    """Convert a word+POS into a toy Token."""
    asa_pos = PTB_TO_ASA.get(ptb_tag, "Other")
    lower = word.lower()

    # ── Round 4: Auxiliary/modal detection ──
    # MD tags are already mapped to "Aux" in PTB_TO_ASA.
    # But verb-tagged auxiliaries (VBP "have", VBZ "is") need word-level detection.
    if asa_pos == "Verb" and lower in AUX_WORDS:
        asa_pos = "Aux"
    # Copula verbs: bond to predicate (Noun/Adj), not grab subject+object
    is_copula = (asa_pos == "Verb" and lower in COPULA_WORDS)

    # Complementizer check moved below SCOPE_BARRIER definition

    # ── PROPERTY: scope_barrier (dim 9) ──
    # Every word carries a barrier value. Bonds crossing high-barrier tokens are penalized.
    # This replaces the IF-based clause boundary rule.
    SCOPE_BARRIER = {
        ',': 0.9, '.': 1.0, ':': 0.8, ';': 0.9,
        'that': 0.7, 'which': 0.7, 'who': 0.7, 'whom': 0.7, 'whose': 0.7,
        'because': 0.8, 'although': 0.8, 'though': 0.8,
        'if': 0.8, 'unless': 0.8, 'while': 0.7, 'since': 0.7,
        'before': 0.6, 'after': 0.6, 'whether': 0.7, 'where': 0.6, 'when': 0.6,
        'as': 0.5, 'and': 0.2, 'or': 0.2, 'but': 0.4,
    }
    # Set dim 9 = scope_barrier for ALL tokens (below, after POS dispatch)

    # ── PROPERTIES (dims 12-15) ──
    # dim 12: DIRECTIONALITY — +1=seeks right, -1=seeks left, 0=both
    # dim 13: ARGUMENT_REACH — how far bonds can extend
    # dim 14: BOND_ABSORBER — how much this token absorbs pass-through bonds
    # dim 15: STRUCTURAL_ROLE — continuous encoding of what this word IS in a sentence
    #   1.0 = entity/referent (nouns, pronouns) — provides arguments
    #   0.5 = predicate (verbs) — both provides and needs
    #   0.3 = modifier (adj, adv) — modifies other words
    #   0.1 = function (det, prep, aux) — structural glue
    #   0.0 = boundary (punctuation, coordinators)

    # ── BONDING ORBITALS (dims 16-19) ──
    # dim 16: NP_ORBITAL — how much this word participates in noun phrases
    # dim 17: VP_ORBITAL — how much this word participates in verb phrases
    # dim 18: HEAD_ORBITAL — how much this word governs structure
    # dim 19: ARG_ORBITAL — how much this word provides/seeks arguments
    # Compatible pairs share high values on the SAME orbital → high dot product.

    # Complementizer reclassification (after SCOPE_BARRIER is defined)
    if asa_pos == "Prep" and lower in ('that', 'because', 'although', 'though'):
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[9] = SCOPE_BARRIER.get(lower, 0.7)
        features[14] = 0.5
        return Token(word, "Other", 0, features)

    # dim 1: MORPHOLOGICAL_FRAME — does this word's shape predict argument structure?
    # Deverbal suffixes predict PP complement attachment.
    DEVERBAL_SUFFIXES = {
        'tion': 0.9, 'sion': 0.9, 'ment': 0.8,
        'ance': 0.6, 'ence': 0.6,
    }
    morph_frame = 0.0
    if asa_pos == "Noun" and len(lower) > 4:
        for suffix, val in DEVERBAL_SUFFIXES.items():
            if lower.endswith(suffix):
                morph_frame = val
                break

    if asa_pos == "Noun":
        features = get_noun_features(word)
        features[9] = SCOPE_BARRIER.get(lower, 0.0)
        features[7] = 1.0   # can_be_modified
        features[1] = morph_frame  # morphological frame
        features[16] = 1.5  # NP participant
        features[17] = 0.0  # not VP
        features[19] = 0.0  # not a head
        features[18] = 1.5  # provides arguments
        features[12] = 0.0
        features[13] = 0.0
        features[15] = 1.0
        # Query dims: nouns don't seek
        features[20] = 0.0
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        # dims 24-27: SELECTABILITY VECTOR (what verbs this noun serves)
        if lower in _sel_n2idx:
            features[24:28] = _sel_noun_vecs[_sel_n2idx[lower]] * 0.0  # disabled — used in patched scoring
        return Token(word, asa_pos, 0, features)
    elif asa_pos == "Pron":
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[0] = 1.0
        features[3] = 1.0
        features[12] = 0.0
        features[13] = 0.0
        features[15] = 0.9
        features[16] = 0.5  # weak NP
        features[18] = 1.5  # provides arguments
        # Query dims: pronouns don't seek
        features[20] = 0.0
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, asa_pos, 0, features)
    elif asa_pos == "Aux":
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[12] = 0.8
        features[13] = 0.15  # aux bonds to adjacent verb
        features[14] = 0.7
        features[15] = 0.1
        features[17] = 0.5  # VP participant
        features[19] = 0.5  # weak head
        # Query dims: Aux seeks verb
        features[20] = 0.0   # NP_QUERY
        features[21] = 1.0   # VP_QUERY - seeks verb
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, "Aux", 1, features)
    elif is_copula:
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[4] = 1.0
        features[12] = 0.0
        features[13] = 0.3
        features[15] = 0.4
        features[17] = 0.5  # VP
        features[19] = 1.0  # head
        features[18] = 1.0  # argument-seeker (weaker)
        # Query dims: Copula seeks arguments
        features[20] = 0.0
        features[21] = 0.0
        features[22] = 0.8   # ARG_QUERY
        features[23] = 0.0
        return Token(word, "Verb", 2, features)
    elif asa_pos == "Verb":
        features, valence = get_verb_features(word)
        # ── VERB FRAME PROPERTIES (from WordNet, per-word) ──
        frames = get_verb_frames(word)
        # Override valence with frame-based intransitivity
        if frames['intransitive'] > 0.8 and valence > 1:
            valence = 1  # WordNet says this verb is intransitive
        # Only reduce to intrans if clausal AND already stative class
        if frames['clausal'] > 0.5 and valence == 1:
            pass  # already intransitive, good
        features[12] = -0.3
        if valence == 1:
            features[13] = 0.15
        else:
            features[13] = 0.2
        features[15] = 0.5
        features[17] = 0.5  # VP orbital
        features[19] = 1.0  # clause head
        features[18] = 0.5  # argument seeker
        features[2] = frames['animate_subj']  # animate_subj_pref from frames
        # dim 8: FINITENESS — finite verbs are clause heads, non-finite are dependents
        # VBD/VBZ/VBP = 1.0 (finite), VB/VBG/VBN = 0.0 (non-finite)
        FINITE_TAGS = {'VBD': 1.0, 'VBZ': 1.0, 'VBP': 1.0,
                       'VB': 0.0, 'VBG': 0.2, 'VBN': 0.1}
        features[8] = FINITE_TAGS.get(ptb_tag, 0.5)
        # Query dims: Verb seeks arguments, weak V→V chains
        features[20] = 0.0
        features[21] = 0.3   # VP_QUERY - weak, for V→V chains
        features[22] = 1.0   # ARG_QUERY - seeks arguments
        features[23] = 0.0
        # dims 24-27: SELECTIONAL PREFERENCE VECTOR (what nouns this verb wants)
        vl = _wnl.lemmatize(lower, pos='v')
        if vl in _sel_v2idx:
            features[24:28] = _sel_verb_vecs[_sel_v2idx[vl]] * 0.0  # disabled in dot product — used in patched scoring instead
        return Token(word, asa_pos, valence, features)
    elif asa_pos == "Det":
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[5] = 1.0
        features[12] = 1.0
        features[13] = 0.25  # slightly reduced: dist 4+ is 33% precision
        features[15] = 0.1
        features[16] = 0.5  # NP orbital
        # Query dims: Det seeks NP
        features[20] = 1.0   # NP_QUERY - seeks NP
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, asa_pos, 1, features)
    elif asa_pos == "Adj":
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[7] = 1.0
        features[12] = 0.8
        features[13] = 0.15  # REDUCED: adj bonds should be dist ≤ 2 (P drops to 39% at dist=2)
        features[15] = 0.3
        features[16] = 0.5  # NP orbital
        # Query dims: Adj seeks NP
        features[20] = 1.0   # NP_QUERY - seeks NP
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, asa_pos, 1, features)
    elif asa_pos == "Adv":
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[6] = 1.0
        features[12] = 0.0
        features[13] = 0.2
        features[15] = 0.3
        features[17] = 0.5  # VP orbital
        # Query dims: Adv seeks verb
        features[20] = 0.0
        features[21] = 1.0   # VP_QUERY - seeks verb
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, asa_pos, 1, features)
    elif asa_pos == "Prep":
        features = np.zeros(FEATURE_DIM, dtype=float)
        # dim 11: PREP HEAD PREFERENCE — verb vs noun head tendency
        # Derived from gold treebank statistics:
        #   1.0 = strongly verb-preferring ("to" 88%, "because" 100%)
        #   0.0 = strongly noun-preferring ("of" 3%)
        PREP_VERB_PREF = {
            'of': 0.03, 'to': 0.88, 'in': 0.61, 'for': 0.47,
            'on': 0.44, 'from': 0.55, 'by': 0.69, 'with': 0.50,
            'at': 0.68, 'as': 0.53, 'about': 0.07, 'than': 0.00,
            'into': 0.80, 'through': 0.60, 'between': 0.30,
            'under': 0.50, 'after': 0.70, 'before': 0.70,
            'since': 0.80, 'during': 0.50, 'against': 0.60,
            'without': 0.60, 'until': 0.80, 'toward': 0.80,
            'among': 0.40, 'because': 1.0, 'although': 1.0,
            'while': 1.0, 'if': 1.0, 'whether': 1.0,
        }
        features[11] = PREP_VERB_PREF.get(lower, 0.5)
        features[9] = SCOPE_BARRIER.get(lower, 0.0)
        features[12] = 0.5
        features[13] = 0.3
        features[14] = 0.85
        features[15] = 0.1
        features[16] = 0.8  # weak NP (bonds to noun complement)
        features[19] = 0.5  # weak head
        features[18] = 0.5  # argument-ish (connects to args)
        # Query dims: Prep seeks arguments and heads
        features[20] = 0.0
        features[21] = 0.0
        features[22] = 0.8   # ARG_QUERY
        features[23] = 0.8   # HEAD_QUERY
        return Token(word, asa_pos, 2, features)
    elif asa_pos == "Coord":
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[9] = SCOPE_BARRIER.get(lower, 0.0)
        features[12] = 0.0   # Coord looks both ways
        features[13] = 0.1   # short reach (adjacent conjuncts)
        # Query dims: Coord all 0
        features[20] = 0.0
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, "Coord", 2, features)
    elif asa_pos == "Num":
        # Numbers modify nouns (like adjectives)
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[7] = 1.0  # can_modify_noun
        # Query dims: Num treated as Adj, seeks NP
        features[20] = 1.0   # NP_QUERY - seeks NP
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, "Adj", 1, features)  # treat as Adj for bonding
    elif asa_pos == "Poss":
        # Possessive 's binds to head noun
        features = np.zeros(FEATURE_DIM, dtype=float)
        # Query dims: Poss treated as Det, seeks NP
        features[20] = 1.0   # NP_QUERY - seeks NP
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, "Det", 1, features)  # treat as Det for bonding
    elif asa_pos == "Part":
        # Particles attach to verbs (like adverbs)
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[6] = 1.0  # can_modify_verb
        # Query dims: Part treated as Adv, seeks verb
        features[20] = 0.0
        features[21] = 1.0   # VP_QUERY - seeks verb
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, "Adv", 1, features)  # treat as Adv for bonding
    else:
        features = np.zeros(FEATURE_DIM, dtype=float)
        features[9] = SCOPE_BARRIER.get(lower, 0.0)
        features[12] = 0.0
        features[13] = 0.0
        # Query dims: Other all 0
        features[20] = 0.0
        features[21] = 0.0
        features[22] = 0.0
        features[23] = 0.0
        return Token(word, "Other", 0, features)


# ═══════════════════════════════════════════════════════════════
# EVALUATION: Compare predicted bonds to gold dependencies
# ═══════════════════════════════════════════════════════════════

def extract_gold_bonds(dep_graph) -> List[Tuple[int, int, str]]:
    """Extract (head_idx, dep_idx, relation) from NLTK DependencyGraph.

    Returns 0-indexed token positions (excluding ROOT).
    """
    bonds = []
    for idx, node in sorted(dep_graph.nodes.items()):
        if idx == 0:
            continue  # skip ROOT
        head = node['head']
        if head == 0:
            continue  # skip root attachment
        # Convert to 0-indexed
        bonds.append((head - 1, idx - 1, node.get('rel', '')))
    return bonds


def run_bonding(tokens: List[Token]) -> Dict[Tuple[int, int], float]:
    """Run energy-minimization bonding on token list.

    Uses simulated annealing for >12 tokens, exhaustive for smaller.
    Returns dict of (i, j) -> score where i < j.
    """
    from asa_toy import MoleculeState, simulated_annealing_bonding, exhaustive_bonding, EXHAUSTIVE_THRESHOLD

    # Construct MoleculeState manually (bypass word lookup)
    state = MoleculeState.__new__(MoleculeState)
    state.tokens = tokens
    state.bonds = {}
    state.bond_labels = {}
    state.remaining_valence = [t.valence for t in tokens]

    result = simulated_annealing_bonding(state)

    bonds = dict(result.bonds)

    # Post-bonding: NOUN COMPOUND detection
    #
    # Proper noun sequences: FLAT structure — all tokens bond to the LAST token.
    # "James A. Talcott" → James→Talcott, A.→Talcott (not James→A.)
    # This matches gold parse: 46% precision on NNP bonds → needs fixing.
    #
    # Common noun compounds: CHAIN structure — adjacent pairs.
    # "stock market" → stock→market
    i = 0
    while i < len(tokens):
        if tokens[i].pos == "Noun" and i < len(_ptb_tags) and _ptb_tags[i].startswith("NNP"):
            # Start of proper noun sequence — find the run
            j = i
            while j < len(tokens) and tokens[j].pos == "Noun" and j < len(_ptb_tags) and _ptb_tags[j].startswith("NNP"):
                j += 1
            # Check if NNP run is followed by a common noun (mixed compound)
            # "Kent cigarettes", "Treasury bills" — NNP + NN
            if j < len(tokens) and tokens[j].pos == "Noun" and j < len(_ptb_tags) and not _ptb_tags[j].startswith("NNP"):
                # The NN is the compound head, all NNPs bond to it
                head = j
                j += 1  # include the NN in the span
            else:
                head = j - 1  # last NNP is head
            # Bond all tokens to head (flat structure)
            for k in range(i, head):
                key = (k, head)
                if key not in bonds:
                    bonds[key] = 1.5
            i = j
        else:
            # Common noun: chain to next adjacent noun
            if (tokens[i].pos == "Noun" and i + 1 < len(tokens)
                    and tokens[i+1].pos == "Noun"):
                key = (i, i+1)
                if key not in bonds:
                    bonds[key] = 1.5
            i += 1

    # Gap-2 and gap-3 compounds: N-X-N and N-X-X-N
    for i in range(len(tokens) - 2):
        if (tokens[i].pos == "Noun" and tokens[i+2].pos == "Noun"
                and tokens[i+1].pos in ("Adj", "Noun")):
            key = (i, i+2)
            if key not in bonds:
                bonds[key] = 0.8
    # Gap-3: N-X-X-N where X is Adj/Noun (within same NP)
    for i in range(len(tokens) - 3):
        if (tokens[i].pos == "Noun" and tokens[i+3].pos == "Noun"
                and tokens[i+1].pos in ("Adj", "Noun")
                and tokens[i+2].pos in ("Adj", "Noun")):
            key = (i, i+3)
            if key not in bonds:
                bonds[key] = 0.6

    # Post-bonding: VERB CHAIN detection
    # Verbs at dist 1-3 that share a clause should chain.
    # "said ... would", "expected to rise", "began trading"
    verb_idxs = [i for i, t in enumerate(tokens) if t.pos in ("Verb",)]
    for vi in range(len(verb_idxs)):
        for vj in range(vi + 1, len(verb_idxs)):
            i, j = verb_idxs[vi], verb_idxs[vj]
            if j - i <= 3:  # within 3 tokens
                key = (i, j)
                if key not in bonds:
                    # Check no barrier AND no noun/pron between (clause boundary signal)
                    between_pos = [tokens[m].pos for m in range(i+1, j)]
                    max_barrier = max((tokens[m].features[9] for m in range(i+1, j)), default=0)
                    has_np = any(p in ('Noun', 'Pron') for p in between_pos)
                    if max_barrier < 0.6 and not (has_np and j - i == 3):
                        bonds[key] = 1.0

    # Post-bonding: ADJ→PREP — only adjacent ("afraid of", "based in")
    for i, tok in enumerate(tokens):
        if tok.pos == "Adj" and i + 1 < len(tokens) and tokens[i + 1].pos == "Prep":
            key = (i, i + 1)
            if key not in bonds:
                bonds[key] = 0.8

    # ── PROJECTIVITY FILTER ──
    # Remove bonds that CROSS other bonds. 79% of crossing bonds are spurious.
    # Keep the higher-scoring bond when two cross.
    bond_list = sorted(bonds.items(), key=lambda x: -x[1])  # highest score first
    kept = {}
    for (i, j), score in bond_list:
        a, b = min(i, j), max(i, j)
        crosses = False
        for (ki, kj) in kept:
            ka, kb = min(ki, kj), max(ki, kj)
            if ka < a < kb < b or a < ka < b < kb:
                crosses = True
                break
        if not crosses:
            kept[(i, j)] = score
    bonds = kept

    # Post-bonding: COORDINATION — right conjunct, only if left has matching POS
    for i, tok in enumerate(tokens):
        if tok.pos == "Coord" and i + 1 < len(tokens) and i > 0:
            right_pos = tokens[i+1].pos
            left_pos = tokens[i-1].pos
            # Only bond if right conjunct matches something on the left
            if right_pos in ("Noun", "Verb", "Adj", "Pron"):
                # Check: is there a same-POS word within 3 tokens left?
                has_match = False
                for j in range(i-1, max(-1, i-4), -1):
                    if tokens[j].pos == right_pos:
                        has_match = True
                        break
                if has_match:
                    key = (i, i+1)
                    if key not in bonds:
                        bonds[key] = 1.0

    # Post-bonding: RELATIVE PRONOUN ATTACHMENT
    # WDT/WP (which/that/who) bonds to nearest left Noun AND nearest right Verb.
    for i, tok in enumerate(tokens):
        if i < len(_ptb_tags) and _ptb_tags[i] in ("WDT", "WP"):
            # Bond to nearest left noun (the antecedent)
            for j in range(i - 1, max(-1, i - 4), -1):
                if tokens[j].pos == "Noun":
                    key = (j, i)
                    if key not in bonds:
                        bonds[key] = 1.2
                    break
            # Bond to nearest right verb (the relative clause predicate)
            for j in range(i + 1, min(i + 5, len(tokens))):
                if tokens[j].pos in ("Verb", "Aux"):
                    key = (i, j)
                    if key not in bonds:
                        bonds[key] = 1.0
                    break

    # Post-bonding: POSSESSIVE ATTACHMENT
    # 's bonds to the nearest left Noun (its possessor).
    # Gold: 's → preceding noun, not following noun.
    for i, tok in enumerate(tokens):
        if i < len(_ptb_tags) and _ptb_tags[i] == "POS":
            # Remove any SA-assigned bonds for this token
            bonds_to_remove = [(a, b) for (a, b) in bonds if a == i or b == i]
            for key in bonds_to_remove:
                del bonds[key]
            # Bond to nearest left noun
            for j in range(i - 1, max(-1, i - 5), -1):
                if tokens[j].pos == "Noun":
                    bonds[(j, i)] = 1.5
                    break

    # Post-bonding: SUBJECT ATTACHMENT for Aux
    # Aux tokens need a subject (nearest left Noun/Pron).
    for i, tok in enumerate(tokens):
        if tok.pos == "Aux":
            for j in range(i - 1, max(-1, i - 4), -1):
                if tokens[j].pos in ("Noun", "Pron"):
                    key = (j, i)
                    if key not in bonds:
                        max_barrier = max((tokens[m].features[9] for m in range(j+1, i)), default=0)
                        if max_barrier < 0.5:
                            bonds[key] = 1.0
                    break

    # ── COPULA→ADJ PREDICATE ATTACHMENT ──
    # Copula/aux (is, was, are, were, be) + adjacent Adj = predicate adjective
    COPULA_SET = {'is', 'was', 'are', 'were', 'be', 'been', "'s", 'seem', 'seemed',
                  'become', 'became', 'remain', 'remained', 'appear', 'appeared'}
    for i, tok in enumerate(tokens):
        if tok.word.lower() in COPULA_SET and tok.pos in ("Verb", "Aux"):
            # Look for Adj within 2 tokens to the right
            for j in range(i + 1, min(i + 3, len(tokens))):
                if tokens[j].pos == "Adj":
                    key = (i, j)
                    if key not in bonds:
                        bonds[key] = 1.5
                    break

    # ── NEGATION ATTACHMENT ──
    # "n't" always bonds to immediately preceding verb/aux
    for i, tok in enumerate(tokens):
        if tok.word.lower() in ("n't", "not") and tok.pos == "Adv" and i > 0:
            if tokens[i-1].pos in ("Verb", "Aux"):
                # Remove any existing bond for this token
                to_remove = [k for k in bonds if i in k]
                for k in to_remove:
                    del bonds[k]
                bonds[(i-1, i)] = 3.0  # very strong — negation is almost always correct

    # ── COMPARATIVE PREP→ADJ ──
    # "more than", "as such", "at least", "at earliest" — prep + comparative adj
    COMP_PREPS = {'than', 'as', 'at', 'among', 'like', 'for'}
    for i in range(len(tokens) - 1):
        if (tokens[i].pos == "Prep" and tokens[i+1].pos == "Adj"
                and tokens[i].word.lower() in COMP_PREPS):
            key = (i, i+1)
            if key not in bonds:
                bonds[key] = 1.0

    # ── ADV MODIFICATION ──
    # Adv→Adj: "very large", "most important"
    # Adv→Prep: "just in", "even for", "right after"
    for i in range(len(tokens) - 1):
        if tokens[i].pos == "Adv":
            if tokens[i+1].pos == "Adj":
                key = (i, i+1)
                if key not in bonds:
                    bonds[key] = 1.2
            elif tokens[i+1].pos in ("Prep", "Adv"):
                key = (i, i+1)
                if key not in bonds:
                    bonds[key] = 0.8

    # ── INFINITIVAL TO → reclassify as Aux and bond to following VB ──
    # "to rise", "to trade" — TO followed by VB is an infinitive marker.
    # Bond TO to the VB. This captures ~300 TO→VB gold bonds.
    for i in range(len(tokens) - 1):
        if (i < len(_ptb_tags) and _ptb_tags[i] == 'TO'
                and tokens[i].pos == "Prep"
                and i + 1 < len(_ptb_tags) and _ptb_tags[i+1] in ('VB', 'VBG')):
            # Remove any existing Prep bonds for this TO
            to_remove = [k for k in bonds if i in k]
            for k in to_remove:
                del bonds[k]
            # Bond TO to following verb
            bonds[(i, i+1)] = 2.0

    # ── LONG-RANGE V→V CHAINS ──
    # For reporting/thinking verbs, find the next verb at dist 4-8
    # if separated only by Noun/Pron/Aux/Adv (argument + modal, no clause break)
    CLAUSAL_VERBS = {'said', 'say', 'says', 'think', 'thought', 'believe',
                     'believed', 'expect', 'expected', 'report', 'reported',
                     'note', 'noted', 'add', 'added', 'announce', 'announced',
                     'warn', 'warned', 'indicate', 'indicated', 'suggest',
                     'suggested', 'contend', 'contended', 'estimate', 'estimated'}
    for i, tok in enumerate(tokens):
        if tok.pos == "Verb" and tok.word.lower() in CLAUSAL_VERBS:
            for j in range(i + 3, min(i + 10, len(tokens))):
                if tokens[j].pos in ("Verb", "Aux"):
                    # Check no strong barrier between
                    max_barrier = max((tokens[m].features[9] for m in range(i+1, j)), default=0)
                    if max_barrier < 0.8:  # allow commas but not strong subordinators
                        key = (i, j)
                        if key not in bonds:
                            bonds[key] = 0.8
                    break  # only bond to first verb found

    # ── NP CHUNKING (greedy left-to-right) ──
    # Override Det/Adj bonds: find [Det? Adj* Noun+] spans
    # and bond all modifiers to the LAST noun (NP head).
    i = 0
    while i < len(tokens):
        if tokens[i].pos in ("Det", "Adj") and i + 1 < len(tokens):
            np_start = i
            j = i
            while j < len(tokens) and tokens[j].pos in ("Det", "Adj", "Noun", "Num", "Adv"):
                j += 1
            np_end = j
            # Find last noun in span (NP head)
            head_idx = None
            for k in range(np_end - 1, np_start - 1, -1):
                if tokens[k].pos == "Noun":
                    head_idx = k
                    break
            if head_idx is not None and head_idx > np_start:
                for k in range(np_start, head_idx):
                    if tokens[k].pos in ("Det", "Adj"):
                        # Remove existing bonds for this modifier
                        to_remove = [key for key in bonds if k in key]
                        for key in to_remove:
                            a, b = key
                            if tokens[min(a,b)].pos in ("Det","Adj") and tokens[max(a,b)].pos == "Noun":
                                del bonds[key]
                        bonds[(k, head_idx)] = 5.0  # bond to NP head
            i = max(j, i + 1)
        else:
            i += 1

    # ── COMPOUND HEAD REDIRECTION ──
    # Redirect Verb→compound_modifier bonds to Verb→compound_head
    redirected = {}
    for (i, j), score in list(bonds.items()):
        for tok_idx in (i, j):
            other_idx = j if tok_idx == i else i
            if (tok_idx < len(tokens) and other_idx < len(tokens)
                    and tokens[tok_idx].pos == "Noun" and tokens[other_idx].pos == "Verb"):
                next_idx = tok_idx + 1
                if next_idx < len(tokens) and tokens[next_idx].pos == "Noun":
                    head = next_idx
                    while head + 1 < len(tokens) and tokens[head + 1].pos == "Noun":
                        head += 1
                    if head != tok_idx:
                        new_key = (min(other_idx, head), max(other_idx, head))
                        if new_key not in bonds:
                            redirected[(i, j)] = (new_key, score)
    for old_key, (new_key, score) in redirected.items():
        if old_key in bonds:
            del bonds[old_key]
        bonds[new_key] = score

    # ── PROJECTIVITY FILTER ──
    bond_list = sorted(bonds.items(), key=lambda x: -x[1])
    kept = {}
    for (i, j), score in bond_list:
        a, b = min(i, j), max(i, j)
        crosses = False
        for (ki, kj) in kept:
            ka, kb = min(ki, kj), max(ki, kj)
            if ka < a < kb < b or a < ka < b < kb:
                crosses = True
                break
        if not crosses:
            kept[(i, j)] = score
    bonds = kept

    return bonds


def evaluate_sentence(dep_graph, verbose=False) -> Dict:
    """Evaluate one sentence: predict bonds and compare to gold."""
    global _ptb_tags, _current_tokens

    # Extract tokens
    tokens = []
    ptb_tags = []
    for idx in range(1, len(dep_graph.nodes)):
        node = dep_graph.nodes[idx]
        word = node['word']
        tag = node['tag']
        if word is None:
            continue
        tokens.append(make_token(word, tag))
        ptb_tags.append(tag)

    _ptb_tags = ptb_tags
    _current_tokens = tokens

    # ── INFORMATION STRUCTURE: GIVENNESS (dim 10) ──
    # Definite NPs (after "the") are given → tend to be subjects.
    # Indefinite NPs (after "a") are new → tend to be objects.
    # 43pp gap: definite→63% subject vs indefinite→20% subject.
    for idx in range(len(tokens)):
        if tokens[idx].pos == "Noun":
            if idx > 0 and tokens[idx-1].word.lower() == 'the':
                tokens[idx].features[10] = 0.9  # definite → given
            elif idx > 0 and tokens[idx-1].word.lower() in ('a', 'an'):
                tokens[idx].features[10] = 0.2  # indefinite → new
            elif idx < len(ptb_tags) and ptb_tags[idx].startswith('NNP'):
                tokens[idx].features[10] = 0.85  # proper noun → given
            else:
                tokens[idx].features[10] = 0.5  # ambiguous
        elif tokens[idx].pos == "Pron":
            tokens[idx].features[10] = 0.95  # pronouns are always given

    if len(tokens) < 3 or len(tokens) > 30:
        return None  # skip very short or long sentences

    # Get gold bonds
    gold_bonds = extract_gold_bonds(dep_graph)
    gold_pairs = set()
    for h, d, rel in gold_bonds:
        if h < len(tokens) and d < len(tokens):
            gold_pairs.add((min(h, d), max(h, d)))

    # Run bonding
    try:
        predicted_bonds = run_bonding(tokens)
    except Exception as e:
        return None

    pred_pairs = set()
    for (i, j) in predicted_bonds:
        pred_pairs.add((min(i, j), max(i, j)))

    # Compute metrics
    correct = pred_pairs & gold_pairs
    precision = len(correct) / max(len(pred_pairs), 1)
    recall = len(correct) / max(len(gold_pairs), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Analyze errors
    missed = gold_pairs - pred_pairs  # gold bonds we didn't predict
    spurious = pred_pairs - gold_pairs  # bonds we predicted that aren't gold

    result = {
        'n_tokens': len(tokens),
        'n_gold': len(gold_pairs),
        'n_pred': len(pred_pairs),
        'n_correct': len(correct),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'missed': [],
        'spurious': [],
    }

    # Annotate missed bonds with token info
    for (i, j) in missed:
        if i < len(tokens) and j < len(tokens):
            result['missed'].append({
                'i': i, 'j': j,
                'word_i': tokens[i].word, 'pos_i': tokens[i].pos,
                'word_j': tokens[j].word, 'pos_j': tokens[j].pos,
            })

    for (i, j) in spurious:
        if i < len(tokens) and j < len(tokens):
            result['spurious'].append({
                'i': i, 'j': j,
                'word_i': tokens[i].word, 'pos_i': tokens[i].pos,
                'word_j': tokens[j].word, 'pos_j': tokens[j].pos,
            })

    if verbose:
        words = [t.word for t in tokens]
        print(f"  Sentence: {' '.join(words)}")
        print(f"  Gold: {len(gold_pairs)}, Pred: {len(pred_pairs)}, Correct: {len(correct)}")
        print(f"  P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
        if missed:
            print(f"  MISSED: {[(m['word_i']+'→'+m['word_j']) for m in result['missed'][:5]]}")
        if spurious:
            print(f"  SPURIOUS: {[(s['word_i']+'→'+s['word_j']) for s in result['spurious'][:5]]}")

    return result


def evaluate_treebank(max_sents=500, verbose_every=50):
    """Evaluate on dependency treebank. THE MAIN LOOP."""
    sents = dependency_treebank.parsed_sents()

    all_results = []
    pos_pair_errors = Counter()  # (pos_i, pos_j) → count of missed bonds
    pos_pair_spurious = Counter()

    for i, sent in enumerate(sents[:max_sents]):
        result = evaluate_sentence(sent, verbose=(i % verbose_every == 0))
        if result is None:
            continue
        all_results.append(result)

        # Aggregate error patterns
        for m in result['missed']:
            pos_pair_errors[(m['pos_i'], m['pos_j'])] += 1
        for s in result['spurious']:
            pos_pair_spurious[(s['pos_i'], s['pos_j'])] += 1

    # Summary
    n = len(all_results)
    avg_p = sum(r['precision'] for r in all_results) / max(n, 1)
    avg_r = sum(r['recall'] for r in all_results) / max(n, 1)
    avg_f1 = sum(r['f1'] for r in all_results) / max(n, 1)
    total_gold = sum(r['n_gold'] for r in all_results)
    total_pred = sum(r['n_pred'] for r in all_results)
    total_correct = sum(r['n_correct'] for r in all_results)

    print(f"\n{'='*60}")
    print(f"TREEBANK EVALUATION: {n} sentences")
    print(f"{'='*60}")
    print(f"  Micro: P={total_correct/max(total_pred,1):.3f} R={total_correct/max(total_gold,1):.3f}")
    print(f"  Macro: P={avg_p:.3f} R={avg_r:.3f} F1={avg_f1:.3f}")
    print(f"  Total bonds: gold={total_gold} pred={total_pred} correct={total_correct}")

    print(f"\n  TOP MISSED POS PAIRS (what bonds we fail to predict):")
    for (p1, p2), count in pos_pair_errors.most_common(15):
        print(f"    {p1:>6} → {p2:<6}: {count} missed")

    print(f"\n  TOP SPURIOUS POS PAIRS (wrong bonds we predict):")
    for (p1, p2), count in pos_pair_spurious.most_common(10):
        print(f"    {p1:>6} → {p2:<6}: {count} spurious")

    # ── Diagnostic: distance distribution of spurious N↔V bonds ──
    spurious_nv_dists = Counter()
    gold_nv_dists = Counter()
    for r in all_results:
        for s in r['spurious']:
            poses = {s['pos_i'], s['pos_j']}
            if 'Noun' in poses and 'Verb' in poses:
                spurious_nv_dists[abs(s['i'] - s['j'])] += 1
        for m in r['missed']:
            poses = {m['pos_i'], m['pos_j']}
            if 'Noun' in poses and 'Verb' in poses:
                gold_nv_dists[abs(m['i'] - m['j'])] += 1

    if spurious_nv_dists:
        print(f"\n  SPURIOUS N↔V bond distances:")
        for d in sorted(spurious_nv_dists):
            print(f"    dist={d}: {spurious_nv_dists[d]} spurious, {gold_nv_dists.get(d,0)} missed-gold")

    # ── Diagnostic: Prep attachment analysis ──
    prep_comp_spurious = Counter()  # dist for spurious prep→complement
    prep_head_spurious = Counter()  # dist for spurious prep→head
    for r in all_results:
        for s in r['spurious']:
            if s['pos_i'] == 'Prep' and s['pos_j'] == 'Noun':
                prep_comp_spurious[s['j'] - s['i']] += 1
            elif s['pos_i'] == 'Noun' and s['pos_j'] == 'Prep':
                prep_head_spurious[s['i'] - s['j']] += 1
    if prep_comp_spurious or prep_head_spurious:
        print(f"\n  PREP ATTACHMENT: complement (Prep→Noun) signed distances:")
        for d in sorted(prep_comp_spurious):
            print(f"    dist={d}: {prep_comp_spurious[d]} spurious")
        print(f"  PREP ATTACHMENT: head (Noun→Prep) signed distances:")
        for d in sorted(prep_head_spurious):
            print(f"    dist={d}: {prep_head_spurious[d]} spurious")

    return all_results, pos_pair_errors, pos_pair_spurious


if __name__ == "__main__":
    results, missed_patterns, spurious_patterns = evaluate_treebank(max_sents=1000)
