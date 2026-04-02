"""
Periodic Table Feature Extraction for ASA v3.

Maps every token to a 28-dim feature vector encoding its structural properties.
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
  24-27: selectional preference (SVD)
"""

import numpy as np
import torch
from typing import List, Dict, Optional
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

FEATURE_DIM = 28

_wnl = WordNetLemmatizer()

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


def word_to_features(word: str, ptb_tag: str) -> np.ndarray:
    """Convert a word + POS tag to a 28-dim periodic table vector."""
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    lower = word.lower()
    asa_pos = PTB_TO_ASA.get(ptb_tag, 'Other')

    # Apply POS template
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

    return features


def sentence_to_features(words: List[str], tags: Optional[List[str]] = None) -> np.ndarray:
    """Convert a sentence to (seq_len, 28) feature matrix.

    If tags not provided, uses NLTK pos_tag.
    """
    if tags is None:
        tagged = pos_tag(words)
        tags = [t for _, t in tagged]

    features = np.stack([word_to_features(w, t) for w, t in zip(words, tags)])
    return features


def batch_to_features(token_ids: torch.Tensor, tokenizer,
                      device: torch.device = None) -> torch.Tensor:
    """Convert a batch of token IDs to periodic table features.

    Args:
        token_ids: (batch, seq_len) tensor of token IDs
        tokenizer: tokenizer with decode() method
        device: target device

    Returns:
        features: (batch, seq_len, 28) tensor
    """
    B, N = token_ids.shape
    all_features = []

    for b in range(B):
        # Decode tokens to words
        ids = token_ids[b].tolist()
        words = [tokenizer.decode([id_]) for id_ in ids]

        # POS tag
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
    # Test feature extraction
    words = ['The', 'big', 'dog', 'runs', 'quickly', 'in', 'the', 'park', '.']
    tags = ['DT', 'JJ', 'NN', 'VBZ', 'RB', 'IN', 'DT', 'NN', '.']

    features = sentence_to_features(words, tags)
    print(f"Feature matrix shape: {features.shape}")
    for i, (w, t) in enumerate(zip(words, tags)):
        nonzero = [(d, features[i, d]) for d in range(FEATURE_DIM) if abs(features[i, d]) > 0.01]
        print(f"  {w:10s} ({t:4s}): {nonzero}")
