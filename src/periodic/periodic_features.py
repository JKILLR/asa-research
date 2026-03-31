"""
Periodic Table Feature Extraction for ASA v3.

Maps every token to a 34-dim feature vector encoding its structural properties.
These features are FIXED (not learned) — they come from the periodic table.

Feature dimensions:
  0-8: Semantic (animate, inanimate, edible, physical, needs_subj, needs_obj,
       can_modify_verb, can_modify_noun, finiteness)
  9: scope_barrier
  10: givenness/transitivity
  11: prep_head_pref
  12: directionality
  13: argument_reach
  14: bond_absorber
  15: structural_role
  16-19: bonding orbitals (NP, VP, ARG, HEAD)
  20-23: query dims
  24-27: selectional preference (SVD) [legacy, now overwritten by new elements]

  --- NEW ELEMENTS (discovered 2026-03-29) ---
  28: clause_complement_pref  (verb: P(takes SBAR complement), 0-1)
  29: semantic_weight          (verb: avg dependents per token, normalized 0-1)
  30: prep_selectivity         (prep: complement entropy, normalized 0-1)
  31: noun_governability       (noun: P(is dependent of verb), 0-1)
  32: frame_complexity         (verb: number of argument frames, normalized 0-1)
  33: embedding_depth          (all: typical syntactic depth, normalized 0-1)
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Optional
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

FEATURE_DIM = 34

_wnl = WordNetLemmatizer()

# Load new elements data
_NEW_ELEMENTS = {}
_new_elements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_elements.json')
if os.path.exists(_new_elements_path):
    with open(_new_elements_path) as _f:
        _NEW_ELEMENTS = json.load(_f)

# POS tag mapping
PTB_TO_ASA = {
    'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Noun', 'NNPS': 'Noun',
    'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb',
    'VBP': 'Verb', 'VBZ': 'Verb',
    'MD': 'Aux',
    'DT': 'Det', 'PDT': 'Det', 'WDT': 'Det',
    'JJ': 'Adj', 'JJR': 'Adj', 'JJS': 'Adj',
    'RB': 'Adv', 'RBR': 'Adv', 'RBS': 'Adv', 'WRB': 'Adv',
    'IN': 'Prep', 'TO': 'Prep',
    'PRP': 'Pron', 'PRP$': 'Det', 'WP': 'Pron', 'WP$': 'Det',
    'CC': 'Coord', 'CD': 'Num',
}

# Feature templates per ASA POS
POS_FEATURES = {
    'Noun':  {'12': 0.0, '13': 0.0, '15': 1.0, '16': 1.5, '18': 1.5, '7': 1.0},
    'Verb':  {'12': -0.3, '13': 0.2, '15': 0.5, '17': 0.5, '19': 1.0, '18': 0.5},
    'Det':   {'5': 1.0, '12': 1.0, '13': 0.25, '15': 0.1, '16': 0.5, '20': 1.0},
    'Adj':   {'7': 1.0, '12': 0.8, '13': 0.15, '15': 0.3, '16': 0.5, '20': 1.0},
    'Adv':   {'6': 1.0, '12': 0.0, '13': 0.2, '15': 0.3, '17': 0.5, '21': 1.0},
    'Prep':  {'12': 0.5, '13': 0.3, '14': 0.85, '15': 0.1, '16': 0.8, '18': 0.5, '19': 0.5},
    'Pron':  {'0': 1.0, '3': 1.0, '15': 0.9, '16': 0.5, '18': 1.5},
    'Aux':   {'12': 0.8, '13': 0.15, '14': 0.7, '15': 0.1, '17': 0.5, '19': 0.5, '21': 1.0},
    'Coord': {'15': 0.0},
    'Num':   {'7': 1.0, '12': 0.8, '13': 0.15, '15': 0.3, '16': 0.5},
    'Other': {},
}

# Scope barrier values
SCOPE_BARRIERS = {
    ',': 0.9, '.': 1.0, ':': 0.8, ';': 0.9,
    'that': 0.7, 'which': 0.7, 'who': 0.7, 'whom': 0.7,
    'because': 0.8, 'although': 0.8, 'though': 0.8,
    'if': 0.8, 'unless': 0.8, 'while': 0.7, 'since': 0.7,
    'before': 0.6, 'after': 0.6, 'whether': 0.7,
    'and': 0.2, 'or': 0.2, 'but': 0.4,
}

# Prep head preferences
PREP_VERB_PREF = {
    'of': 0.03, 'to': 0.88, 'in': 0.61, 'for': 0.47,
    'on': 0.44, 'from': 0.55, 'by': 0.69, 'with': 0.50,
    'at': 0.68, 'as': 0.53, 'about': 0.07, 'than': 0.00,
    'into': 0.80, 'through': 0.60, 'between': 0.30,
    'under': 0.50, 'after': 0.70, 'before': 0.70,
    'since': 0.80, 'during': 0.50, 'against': 0.60,
}

# Normalization constants for new elements
_SEMANTIC_WEIGHT_MAX = 8.0   # max deps/token
_PREP_ENTROPY_MAX = 4.0      # max entropy
_FRAME_COMPLEXITY_MAX = 10.0  # max frame count
_EMBED_DEPTH_MAX = 12.0      # max depth (clip outliers)


def word_to_features(word: str, ptb_tag: str) -> np.ndarray:
    """Convert a word + POS tag to a 34-dim periodic table vector."""
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    lower = word.lower()
    asa_pos = PTB_TO_ASA.get(ptb_tag, 'Other')

    # Apply POS template (dims 0-23)
    template = POS_FEATURES.get(asa_pos, {})
    for dim_str, val in template.items():
        features[int(dim_str)] = val

    # Scope barrier (dim 9)
    features[9] = SCOPE_BARRIERS.get(lower, 0.0)

    # Prep head preference (dim 11)
    if asa_pos == 'Prep':
        features[11] = PREP_VERB_PREF.get(lower, 0.5)

    # Finiteness for verbs (dim 8)
    if asa_pos == 'Verb':
        finite_map = {'VBD': 1.0, 'VBZ': 1.0, 'VBP': 1.0,
                      'VB': 0.0, 'VBG': 0.2, 'VBN': 0.1}
        features[8] = finite_map.get(ptb_tag, 0.5)

    # ─── NEW ELEMENTS (dims 28-33) ───

    # Lemmatize for lookup
    if asa_pos == 'Verb':
        lemma = _wnl.lemmatize(lower, pos='v')
    elif asa_pos == 'Noun':
        lemma = _wnl.lemmatize(lower, pos='n')
    else:
        lemma = lower

    clause_comp = _NEW_ELEMENTS.get('clause_complement_pref', {})
    sem_weight = _NEW_ELEMENTS.get('semantic_weight', {})
    prep_sel = _NEW_ELEMENTS.get('prep_selectivity', {})
    noun_gov = _NEW_ELEMENTS.get('noun_governability', {})
    frame_comp = _NEW_ELEMENTS.get('frame_complexity', {})
    embed_depth = _NEW_ELEMENTS.get('embedding_depth', {})

    # Dim 28: clause_complement_pref (verbs only, 0-1)
    if asa_pos == 'Verb':
        features[28] = clause_comp.get(lemma, clause_comp.get(lower, 0.2))

    # Dim 29: semantic_weight (verbs only, normalized 0-1)
    if asa_pos == 'Verb':
        raw = sem_weight.get(lemma, sem_weight.get(lower, 3.0))
        features[29] = min(raw / _SEMANTIC_WEIGHT_MAX, 1.0)

    # Dim 30: prep_selectivity (preps only, normalized 0-1)
    if asa_pos == 'Prep':
        raw = prep_sel.get(lower, 2.0)  # default moderate entropy
        features[30] = min(raw / _PREP_ENTROPY_MAX, 1.0)

    # Dim 31: noun_governability (nouns/pronouns, 0-1)
    if asa_pos in ('Noun', 'Pron'):
        features[31] = noun_gov.get(lemma, noun_gov.get(lower, 0.4))

    # Dim 32: frame_complexity (verbs only, normalized 0-1)
    if asa_pos == 'Verb':
        raw = frame_comp.get(lemma, frame_comp.get(lower, 3.0))
        features[32] = min(raw / _FRAME_COMPLEXITY_MAX, 1.0)

    # Dim 33: embedding_depth (all words, normalized 0-1)
    raw = embed_depth.get(lower, 5.0)  # default moderate depth
    features[33] = min(raw / _EMBED_DEPTH_MAX, 1.0)

    return features


def sentence_to_features(words: List[str], tags: Optional[List[str]] = None) -> np.ndarray:
    """Convert a sentence to (seq_len, 34) feature matrix."""
    if tags is None:
        tagged = pos_tag(words)
        tags = [t for _, t in tagged]

    features = np.stack([word_to_features(w, t) for w, t in zip(words, tags)])
    return features


def batch_to_features(token_ids: torch.Tensor, tokenizer,
                      device: torch.device = None) -> torch.Tensor:
    """Convert a batch of token IDs to periodic table features."""
    B, N = token_ids.shape
    all_features = []

    for b in range(B):
        ids = token_ids[b].tolist()
        words = [tokenizer.decode([id_]) for id_ in ids]
        try:
            tagged = pos_tag(words)
            tags = [t for _, t in tagged]
        except:
            tags = ['NN'] * len(words)
        feats = sentence_to_features(words, tags)
        all_features.append(feats)

    features = torch.tensor(np.stack(all_features), dtype=torch.float32)
    if device is not None:
        features = features.to(device)
    return features


if __name__ == "__main__":
    words = ['The', 'big', 'dog', 'said', 'that', 'he', 'runs', 'quickly', 'in', 'the', 'park', '.']
    tags = ['DT', 'JJ', 'NN', 'VBD', 'IN', 'PRP', 'VBZ', 'RB', 'IN', 'DT', 'NN', '.']

    features = sentence_to_features(words, tags)
    print(f"Feature matrix shape: {features.shape}")

    dim_names = {
        0: 'animate', 1: 'inanimate', 2: 'anim_subj', 3: 'physical',
        4: 'needs_subj', 5: 'needs_obj', 6: 'mod_verb', 7: 'mod_noun',
        8: 'finiteness', 9: 'scope_barrier', 10: 'givenness', 11: 'prep_pref',
        12: 'direction', 13: 'reach', 14: 'absorber', 15: 'struct_role',
        16: 'NP_orb', 17: 'VP_orb', 18: 'HEAD_orb', 19: 'ARG_orb',
        20: 'NP_q', 21: 'VP_q', 22: 'ARG_q', 23: 'HEAD_q',
        24: 'sel0', 25: 'sel1', 26: 'sel2', 27: 'sel3',
        28: 'clause_comp', 29: 'sem_weight', 30: 'prep_sel',
        31: 'noun_gov', 32: 'frame_comp', 33: 'embed_depth',
    }

    for i, (w, t) in enumerate(zip(words, tags)):
        nonzero = [(dim_names.get(d, f'd{d}'), f'{features[i, d]:.2f}')
                   for d in range(FEATURE_DIM) if abs(features[i, d]) > 0.01]
        print(f"  {w:10s} ({t:4s}): {nonzero}")
