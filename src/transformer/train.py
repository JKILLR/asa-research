"""
Phase 2a Training Pipeline: Word-Level WikiText-2 (Hardened)

Trains ASALanguageModel on WikiText-2 using word-level tokenization with
NLTK POS tagging and WordNet/VerbNet feature extraction.

Feature extraction:
  - Nouns: WordNet hypernym chain walking → semantic features (animate, concrete, etc.)
  - Verbs: VerbNet class → requirement vectors (needs animate subject, etc.)
  - Pronouns: hand-curated features (he/she → animate+human, it → empty)
  - Other POS: empty features/requirements

Gate 2d criteria:
  1. Loss decreases over >=1000 steps (with noise)
  2. No attention collapse (entropy monitoring)
  3. No mode collapse (diverse top-10 predictions)

Usage:
    python train.py --mode full --epochs 3
    python train.py --mode none --epochs 3   # baseline
    python train.py --compare                 # run both and compare
    python train.py --alpha-sweep             # sweep alpha values
"""

import argparse
import math
import time
import json
from pathlib import Path
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk import pos_tag

from model import ASALanguageModel, POS_IDS, ASA_FEATURE_DIM, NUM_POS

# ── NLTK setup ────────────────────────────────────────────────
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

_wnl = WordNetLemmatizer()

# ── POS mapping: Penn Treebank → ASA ─────────────────────────
PTB_TO_ASA = {
    # Nouns
    "NN": "Noun", "NNS": "Noun", "NNP": "Noun", "NNPS": "Noun",
    # Verbs
    "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb",
    "VBP": "Verb", "VBZ": "Verb", "MD": "Verb",
    # Determiners
    "DT": "Det", "PDT": "Det", "WDT": "Det",
    # Adjectives
    "JJ": "Adj", "JJR": "Adj", "JJS": "Adj",
    # Adverbs
    "RB": "Adv", "RBR": "Adv", "RBS": "Adv", "WRB": "Adv",
    # Prepositions
    "IN": "Prep", "TO": "Prep",
    # Pronouns
    "PRP": "Pron", "PRP$": "Pron", "WP": "Pron", "WP$": "Pron",
}


def ptb_to_asa_id(ptb_tag: str) -> int:
    """Convert Penn Treebank tag to ASA POS ID."""
    asa_pos = PTB_TO_ASA.get(ptb_tag, "Other")
    return POS_IDS.get(asa_pos, POS_IDS["Other"])


# =============================================================================
# WORDNET SYNSET → FEATURE MAPPINGS (ported from v2.2)
# =============================================================================
# Feature indices match model.py ASA_FEATURE_DIM=24:
# KEY (0-10): 0=ANIMATE, 1=HUMAN/INANIMATE, 2=ANIMAL/EDIBLE, 3=ORGANIZATION/PHYSICAL,
#   4=CONCRETE/NEEDS_SUBJ, 5=LOCATION/NEEDS_OBJ, 6=ABSTRACT/CAN_MOD_VERB,
#   7=COMMUNICATION/CAN_MOD_NOUN, 8=COGNITION, 9=SCOPE_BARRIER, 10=EVENT/INSTRUMENT
# STRUCTURAL (11-19): 11=PREP_HEAD_PREF, 12=DIRECTIONALITY, 13=ARG_REACH,
#   14=BOND_ABSORBER, 15=STRUCTURAL_ROLE, 16=NP_ORBITAL, 17=VP_ORBITAL,
#   18=HEAD_ORBITAL, 19=ARG_ORBITAL
# QUERY (20-23): 20=NP_QUERY, 21=VP_QUERY, 22=ARG_QUERY, 23=HEAD_QUERY

SYNSET_TO_FEATURES = {
    # Animate
    'person.n.01': {0, 1, 4},       # ANIMATE, HUMAN, CONCRETE
    'human.n.01': {0, 1, 4},
    'animal.n.01': {0, 2, 4},       # ANIMATE, ANIMAL, CONCRETE
    'organism.n.01': {0, 4},         # ANIMATE, CONCRETE
    'living_thing.n.01': {0, 4},

    # Organizations
    'organization.n.01': {3},        # ORGANIZATION
    'social_group.n.01': {3},
    'institution.n.01': {3},
    'company.n.01': {3},

    # Concrete
    'object.n.01': {4},              # CONCRETE
    'physical_entity.n.01': {4},
    'artifact.n.01': {4, 12},        # CONCRETE, INSTRUMENT
    'tool.n.01': {4, 12},
    'device.n.01': {4, 12},
    'container.n.01': {4},
    'substance.n.01': {4},
    'whole.n.02': {4},

    # Locations
    'location.n.01': {5},            # LOCATION
    'region.n.01': {5},
    'area.n.01': {5},
    'place.n.02': {5},
    'structure.n.01': {4, 5},        # CONCRETE, LOCATION
    'building.n.01': {4, 5},

    # Abstract
    'abstraction.n.06': {6},         # ABSTRACT
    'psychological_feature.n.01': {6},
    'attribute.n.02': {6},
    'relation.n.01': {6},
    'measure.n.02': {6},
    'group.n.01': {6},
    'state.n.02': {6},

    # Communication
    'communication.n.02': {6, 7},    # ABSTRACT, COMMUNICATION
    'message.n.02': {6, 7},
    'document.n.01': {4, 7},         # CONCRETE, COMMUNICATION

    # Cognition
    'cognition.n.01': {6, 8},        # ABSTRACT, COGNITION
    'idea.n.01': {6, 8},
    'concept.n.01': {6, 8},

    # Emotion
    'feeling.n.01': {6, 9},          # ABSTRACT, EMOTION
    'emotion.n.01': {6, 9},

    # Events
    'event.n.01': {6, 10},           # ABSTRACT, EVENT
    'act.n.02': {6, 10},
    'activity.n.01': {6, 10},

    # Time
    'time.n.01': {6, 11},            # ABSTRACT, TIME
    'time_period.n.01': {6, 11},

    # Food
    'food.n.01': {4, 13},            # CONCRETE, COMESTIBLE

    # Body
    'body_part.n.01': {4, 14},       # CONCRETE, BODY_PART
}


# =============================================================================
# VERB REQUIREMENTS (ported from v2.2 — ~300 verbs)
# =============================================================================

VERB_CLASS_REQUIREMENTS = {
    'perception': {0},       # ANIMATE
    'cognition': {0},        # ANIMATE
    'communication': {1},    # HUMAN
    'motion': {4},           # CONCRETE
    'motion_directed': {4},  # CONCRETE
    'change': {4},           # CONCRETE
    'change_state': {4},     # CONCRETE
    'creation': {0},         # ANIMATE
    'consumption': {0},      # ANIMATE
    'transfer': {0},         # ANIMATE
    'possession': {0},       # ANIMATE
    'contact': {4},          # CONCRETE
    'contact_impact': {4},   # CONCRETE
    'emotion': {0},          # ANIMATE
    'emotion_cause': {0},    # ANIMATE
    'social': {1},           # HUMAN
    'judgment': {0},         # ANIMATE
    'assessment': {0},       # ANIMATE
    'body_action': {0},      # ANIMATE
    'sound': {4},            # CONCRETE
    'light': {4},            # CONCRETE
    'weather': set(),
    'stative': set(),
    'existence': set(),
    'spatial': {4},          # CONCRETE
}

VERB_TO_CLASS = {
    # === Perception ===
    'see': 'perception', 'hear': 'perception', 'watch': 'perception',
    'observe': 'perception', 'examine': 'perception', 'notice': 'perception',
    'look': 'perception', 'listen': 'perception', 'feel': 'perception',
    'smell': 'perception', 'taste': 'perception', 'view': 'perception',
    'spot': 'perception', 'witness': 'perception', 'perceive': 'perception',
    'detect': 'perception', 'sense': 'perception', 'glimpse': 'perception',
    'overhear': 'perception', 'sight': 'perception', 'eye': 'perception',
    'scan': 'perception', 'inspect': 'perception', 'survey': 'perception',
    'study': 'perception', 'read': 'perception',

    # === Cognition ===
    'think': 'cognition', 'believe': 'cognition', 'know': 'cognition',
    'understand': 'cognition', 'consider': 'cognition', 'realize': 'cognition',
    'remember': 'cognition', 'forget': 'cognition', 'learn': 'cognition',
    'imagine': 'cognition', 'suppose': 'cognition', 'assume': 'cognition',
    'expect': 'cognition', 'hope': 'cognition', 'wish': 'cognition',
    'doubt': 'cognition', 'suspect': 'cognition', 'recognize': 'cognition',
    'recall': 'cognition', 'recollect': 'cognition', 'comprehend': 'cognition',
    'grasp': 'cognition', 'fathom': 'cognition', 'deduce': 'cognition',
    'infer': 'cognition', 'conclude': 'cognition', 'decide': 'cognition',
    'determine': 'cognition', 'figure': 'cognition', 'guess': 'cognition',
    'estimate': 'cognition', 'calculate': 'cognition', 'reckon': 'cognition',
    'ponder': 'cognition', 'contemplate': 'cognition', 'reflect': 'cognition',
    'meditate': 'cognition', 'speculate': 'cognition', 'wonder': 'cognition',

    # === Communication ===
    'say': 'communication', 'tell': 'communication', 'speak': 'communication',
    'talk': 'communication', 'write': 'communication', 'announce': 'communication',
    'report': 'communication', 'ask': 'communication', 'answer': 'communication',
    'explain': 'communication', 'describe': 'communication', 'mention': 'communication',
    'state': 'communication', 'claim': 'communication', 'argue': 'communication',
    'suggest': 'communication', 'propose': 'communication', 'recommend': 'communication',
    'advise': 'communication', 'warn': 'communication', 'inform': 'communication',
    'notify': 'communication', 'declare': 'communication', 'proclaim': 'communication',
    'assert': 'communication', 'insist': 'communication', 'maintain': 'communication',
    'deny': 'communication', 'admit': 'communication', 'confess': 'communication',
    'acknowledge': 'communication', 'confirm': 'communication', 'promise': 'communication',
    'swear': 'communication', 'vow': 'communication', 'whisper': 'communication',
    'shout': 'communication', 'yell': 'communication', 'scream': 'communication',
    'murmur': 'communication', 'mutter': 'communication', 'mumble': 'communication',
    'chat': 'communication', 'converse': 'communication', 'discuss': 'communication',
    'debate': 'communication', 'negotiate': 'communication', 'communicate': 'communication',

    # === Motion ===
    'go': 'motion', 'come': 'motion', 'run': 'motion', 'walk': 'motion',
    'move': 'motion', 'travel': 'motion', 'fly': 'motion', 'swim': 'motion',
    'jump': 'motion', 'leap': 'motion', 'hop': 'motion', 'skip': 'motion',
    'crawl': 'motion', 'creep': 'motion', 'slide': 'motion', 'glide': 'motion',
    'roll': 'motion', 'spin': 'motion', 'turn': 'motion', 'rotate': 'motion',
    'rush': 'motion', 'hurry': 'motion', 'race': 'motion', 'dash': 'motion',
    'sprint': 'motion', 'jog': 'motion', 'stroll': 'motion', 'wander': 'motion',
    'roam': 'motion', 'drift': 'motion', 'float': 'motion', 'sink': 'motion',
    'rise': 'motion', 'climb': 'motion', 'descend': 'motion', 'drop': 'motion',
    'fall': 'motion', 'tumble': 'motion', 'stumble': 'motion', 'trip': 'motion',
    'drive': 'motion', 'ride': 'motion', 'sail': 'motion', 'cruise': 'motion',

    # === Motion Directed ===
    'arrive': 'motion_directed', 'leave': 'motion_directed', 'depart': 'motion_directed',
    'enter': 'motion_directed', 'exit': 'motion_directed', 'return': 'motion_directed',
    'approach': 'motion_directed', 'reach': 'motion_directed', 'escape': 'motion_directed',
    'flee': 'motion_directed', 'retreat': 'motion_directed', 'advance': 'motion_directed',

    # === Change ===
    'break': 'change', 'destroy': 'change', 'damage': 'change',
    'fix': 'change', 'repair': 'change', 'restore': 'change',
    'change': 'change', 'transform': 'change', 'alter': 'change',
    'modify': 'change', 'adjust': 'change', 'adapt': 'change',
    'improve': 'change', 'worsen': 'change', 'increase': 'change',
    'decrease': 'change', 'reduce': 'change', 'expand': 'change',
    'shrink': 'change', 'grow': 'change', 'stretch': 'change',
    'bend': 'change', 'fold': 'change', 'twist': 'change',
    'crush': 'change', 'smash': 'change', 'shatter': 'change',
    'tear': 'change', 'rip': 'change', 'split': 'change',
    'cut': 'change', 'slice': 'change', 'chop': 'change',
    'cook': 'change', 'bake': 'change', 'fry': 'change',
    'boil': 'change', 'roast': 'change', 'grill': 'change',
    'melt': 'change', 'freeze': 'change', 'burn': 'change',

    # === Creation ===
    'make': 'creation', 'build': 'creation', 'create': 'creation',
    'produce': 'creation', 'develop': 'creation', 'design': 'creation',
    'construct': 'creation', 'assemble': 'creation', 'manufacture': 'creation',
    'generate': 'creation', 'form': 'creation', 'shape': 'creation',
    'craft': 'creation', 'forge': 'creation', 'compose': 'creation',
    'draw': 'creation', 'paint': 'creation',
    'sculpt': 'creation', 'carve': 'creation', 'weave': 'creation',
    'knit': 'creation', 'sew': 'creation',
    'invent': 'creation', 'devise': 'creation', 'originate': 'creation',

    # === Consumption ===
    'eat': 'consumption', 'drink': 'consumption', 'consume': 'consumption',
    'devour': 'consumption', 'swallow': 'consumption', 'ingest': 'consumption',
    'chew': 'consumption', 'bite': 'consumption', 'sip': 'consumption',
    'gulp': 'consumption', 'gobble': 'consumption', 'nibble': 'consumption',

    # === Transfer ===
    'give': 'transfer', 'send': 'transfer', 'bring': 'transfer',
    'take': 'transfer', 'receive': 'transfer', 'get': 'transfer',
    'pass': 'transfer', 'hand': 'transfer', 'deliver': 'transfer',
    'provide': 'transfer', 'supply': 'transfer', 'offer': 'transfer',
    'present': 'transfer', 'donate': 'transfer', 'contribute': 'transfer',
    'lend': 'transfer', 'borrow': 'transfer',
    'mail': 'transfer', 'ship': 'transfer', 'transport': 'transfer',

    # === Possession ===
    'own': 'possession', 'possess': 'possession', 'have': 'possession',
    'keep': 'possession', 'retain': 'possession',
    'acquire': 'possession', 'obtain': 'possession', 'gain': 'possession',
    'lose': 'possession', 'lack': 'possession',

    # === Contact ===
    'hit': 'contact', 'touch': 'contact', 'kick': 'contact',
    'push': 'contact', 'pull': 'contact', 'strike': 'contact',
    'grab': 'contact', 'hold': 'contact', 'catch': 'contact',
    'throw': 'contact', 'toss': 'contact', 'hurl': 'contact',
    'pat': 'contact', 'tap': 'contact', 'slap': 'contact',
    'punch': 'contact', 'poke': 'contact', 'prod': 'contact',
    'rub': 'contact', 'stroke': 'contact', 'scratch': 'contact',
    'squeeze': 'contact', 'press': 'contact', 'pinch': 'contact',

    # === Emotion ===
    'love': 'emotion', 'hate': 'emotion', 'like': 'emotion',
    'dislike': 'emotion', 'fear': 'emotion', 'dread': 'emotion',
    'enjoy': 'emotion', 'prefer': 'emotion', 'want': 'emotion',
    'desire': 'emotion', 'crave': 'emotion',
    'admire': 'emotion', 'respect': 'emotion', 'appreciate': 'emotion',
    'value': 'emotion', 'cherish': 'emotion', 'treasure': 'emotion',
    'miss': 'emotion', 'regret': 'emotion', 'resent': 'emotion',
    'envy': 'emotion', 'pity': 'emotion', 'trust': 'emotion',

    # === Emotion Cause ===
    'amuse': 'emotion_cause', 'please': 'emotion_cause', 'delight': 'emotion_cause',
    'satisfy': 'emotion_cause', 'annoy': 'emotion_cause', 'irritate': 'emotion_cause',
    'anger': 'emotion_cause', 'upset': 'emotion_cause', 'frighten': 'emotion_cause',
    'scare': 'emotion_cause', 'terrify': 'emotion_cause', 'surprise': 'emotion_cause',
    'shock': 'emotion_cause', 'amaze': 'emotion_cause', 'astonish': 'emotion_cause',
    'bore': 'emotion_cause', 'tire': 'emotion_cause', 'exhaust': 'emotion_cause',
    'excite': 'emotion_cause', 'thrill': 'emotion_cause', 'inspire': 'emotion_cause',
    'motivate': 'emotion_cause', 'encourage': 'emotion_cause', 'discourage': 'emotion_cause',
    'comfort': 'emotion_cause', 'calm': 'emotion_cause', 'relax': 'emotion_cause',
    'worry': 'emotion_cause', 'concern': 'emotion_cause', 'trouble': 'emotion_cause',
    'confuse': 'emotion_cause', 'puzzle': 'emotion_cause', 'perplex': 'emotion_cause',

    # === Social ===
    'meet': 'social', 'marry': 'social', 'divorce': 'social',
    'befriend': 'social', 'date': 'social', 'visit': 'social',
    'greet': 'social', 'welcome': 'social', 'introduce': 'social',
    'invite': 'social', 'host': 'social', 'accompany': 'social',
    'join': 'social', 'follow': 'social', 'lead': 'social',
    'help': 'social', 'assist': 'social', 'support': 'social',
    'serve': 'social', 'hire': 'social', 'fire': 'social',
    'employ': 'social', 'manage': 'social', 'supervise': 'social',

    # === Judgment ===
    'judge': 'judgment', 'evaluate': 'judgment', 'assess': 'judgment',
    'rate': 'judgment', 'rank': 'judgment', 'grade': 'judgment',
    'criticize': 'judgment', 'praise': 'judgment', 'blame': 'judgment',
    'accuse': 'judgment', 'forgive': 'judgment', 'excuse': 'judgment',
    'approve': 'judgment', 'reject': 'judgment', 'accept': 'judgment',

    # === Stative/Existence ===
    'be': 'stative', 'exist': 'stative', 'remain': 'stative',
    'stay': 'stative', 'seem': 'stative', 'appear': 'stative',
    'become': 'stative', 'sound': 'stative',
    'belong': 'stative', 'consist': 'stative', 'contain': 'stative',
    'include': 'stative', 'involve': 'stative', 'require': 'stative',
    'need': 'stative', 'deserve': 'stative', 'merit': 'stative',
    'equal': 'stative', 'resemble': 'stative', 'differ': 'stative',
    'matter': 'stative', 'count': 'stative', 'depend': 'stative',
    'fit': 'stative', 'suit': 'stative', 'match': 'stative',
    'last': 'stative', 'continue': 'stative', 'persist': 'stative',

    # === Body Actions ===
    'breathe': 'body_action', 'blink': 'body_action', 'wink': 'body_action',
    'nod': 'body_action', 'shrug': 'body_action', 'yawn': 'body_action',
    'sneeze': 'body_action', 'cough': 'body_action', 'hiccup': 'body_action',
    'smile': 'body_action', 'frown': 'body_action', 'laugh': 'body_action',
    'cry': 'body_action', 'weep': 'body_action', 'sigh': 'body_action',
    'sleep': 'body_action', 'wake': 'body_action', 'rest': 'body_action',

    # === Spatial ===
    'put': 'spatial', 'place': 'spatial', 'set': 'spatial',
    'lay': 'spatial', 'position': 'spatial', 'locate': 'spatial',
    'remove': 'spatial', 'withdraw': 'spatial', 'extract': 'spatial',
    'insert': 'spatial', 'attach': 'spatial', 'connect': 'spatial',
    'separate': 'spatial', 'divide': 'spatial', 'combine': 'spatial',
    'fill': 'spatial', 'empty': 'spatial', 'cover': 'spatial',
    'wrap': 'spatial', 'surround': 'spatial', 'enclose': 'spatial',
    'hang': 'spatial', 'mount': 'spatial', 'install': 'spatial',

    # === Additional common verbs (gap-fill from WikiText coverage check) ===
    'find': 'perception', 'discover': 'perception',
    'call': 'communication', 'name': 'communication',
    'show': 'perception', 'display': 'perception', 'reveal': 'perception',
    'try': 'cognition', 'attempt': 'cognition',
    'begin': 'change_state', 'start': 'change_state', 'end': 'change_state',
    'stop': 'change_state', 'finish': 'change_state', 'complete': 'change_state',
    'play': 'body_action', 'sing': 'body_action', 'dance': 'body_action',
    'sit': 'body_action', 'stand': 'body_action', 'lie': 'body_action',
    'live': 'existence', 'die': 'existence', 'survive': 'existence',
    'happen': 'existence', 'occur': 'existence',
    'pay': 'transfer', 'spend': 'transfer', 'buy': 'transfer', 'sell': 'transfer',
    'use': 'contact', 'handle': 'contact', 'operate': 'contact',
    'do': 'stative', 'work': 'body_action', 'act': 'body_action',
    'open': 'change', 'close': 'change', 'shut': 'change',
    'carry': 'contact', 'lift': 'contact', 'pick': 'contact', 'drop': 'motion',
    'wait': 'stative', 'allow': 'stative', 'let': 'stative', 'prevent': 'stative',
    'would': 'stative', 'could': 'stative', 'should': 'stative',
    'might': 'stative', 'must': 'stative', 'shall': 'stative',
    'may': 'stative', 'can': 'stative', 'will': 'stative',
    'cause': 'emotion_cause', 'force': 'emotion_cause',
    'win': 'possession', 'earn': 'possession',
    'teach': 'communication', 'train': 'communication', 'educate': 'communication',
    'fight': 'contact', 'attack': 'contact', 'defend': 'contact',
    'save': 'possession', 'protect': 'social', 'guard': 'social',
    'fail': 'stative', 'succeed': 'stative', 'manage': 'social',
    'wonder': 'cognition', 'agree': 'communication', 'disagree': 'communication',
    'prove': 'cognition', 'test': 'cognition', 'check': 'perception',
    'plan': 'cognition', 'prepare': 'creation', 'arrange': 'spatial',
    'wear': 'contact', 'dress': 'change',
    'eat': 'consumption',  # already present but ensures coverage
}

# Irregular past tenses that WordNetLemmatizer doesn't handle
# (because the past form is also a valid base verb — e.g., "saw" = to saw wood)
IRREGULAR_PAST_TO_BASE = {
    'saw': 'see',
    'bore': 'bear',
    'wound': 'wind',
    'bound': 'bind',
    'ground': 'grind',
    'found': 'find',  # WNL gets this, but 'found' is also a verb (to found a company)
    'lay': 'lie',     # WNL gets this, but 'lay' is also a verb (to lay something down)
    'led': 'lead',
    'read': 'read',   # past tense pronounced differently but spelled same
    'shed': 'shed',
    'sped': 'speed',
    'wed': 'wed',
}


# =============================================================================
# PRONOUN FEATURES (ported from v2.2)
# =============================================================================

PRONOUN_FEATURES = {
    # Third person — features encode what the pronoun IS
    'he': {0, 1, 4},    # ANIMATE, HUMAN, CONCRETE
    'she': {0, 1, 4},
    'him': {0, 1, 4},
    'her': {0, 1, 4},
    'his': {0, 1, 4},
    'hers': {0, 1, 4},
    'they': {0},         # ANIMATE (could be non-human)
    'them': {0},
    'their': {0},
    'theirs': {0},
    'it': {4},           # CONCRETE (default — could be abstract)
    'its': {4},
    # First/second person
    'i': {0, 1, 4},
    'me': {0, 1, 4},
    'my': {0, 1, 4},
    'mine': {0, 1, 4},
    'we': {0, 1, 4},
    'us': {0, 1, 4},
    'our': {0, 1, 4},
    'ours': {0, 1, 4},
    'you': {0, 1, 4},
    'your': {0, 1, 4},
    'yours': {0, 1, 4},
    'myself': {0, 1, 4},
    'yourself': {0, 1, 4},
    'himself': {0, 1, 4},
    'herself': {0, 1, 4},
    'themselves': {0},
    'ourselves': {0, 1, 4},
}


# =============================================================================
# WORDNET FEATURE EXTRACTION
# =============================================================================

@lru_cache(maxsize=50000)
def get_noun_features(word: str) -> set:
    """Walk WordNet hypernym chain to extract semantic features.

    Returns set of feature indices from SYNSET_TO_FEATURES.
    Uses first synset (most frequent sense). Walks up the hypernym tree
    until a mapped synset is found or the chain is exhausted.
    """
    synsets = wn.synsets(word.lower(), pos=wn.NOUN)
    if not synsets:
        return set()

    # Try first sense (most frequent)
    synset = synsets[0]
    features = set()

    # Check direct match
    if synset.name() in SYNSET_TO_FEATURES:
        features.update(SYNSET_TO_FEATURES[synset.name()])
        return features

    # Walk hypernym chain (up to 10 levels)
    for hyper in synset.closure(lambda s: s.hypernyms(), depth=10):
        name = hyper.name()
        if name in SYNSET_TO_FEATURES:
            features.update(SYNSET_TO_FEATURES[name])
            break  # stop at first match (most specific)

    return features


@lru_cache(maxsize=50000)
def get_verb_lemma(word: str) -> str:
    """Get base form of a verb for VerbNet lookup.

    Uses WordNetLemmatizer with pos='v' which correctly handles irregular
    past tense: saw→see, went→go, sat→sit, found→find, etc.
    The previous approach (wn.synsets first lemma) failed on these because
    it picked wrong senses (saw→saw.n.01 "cutting tool", found→establish).
    """
    return _wnl.lemmatize(word.lower(), pos='v')


@lru_cache(maxsize=50000)
def get_verb_frames(word: str) -> dict:
    """Extract subcategorization from WordNet verb frames (discovery R25).

    Returns 4 continuous properties:
      intransitive: 0-1, clausal: 0-1, animate_subj: 0-1, pp_complement: 0-1
    """
    lemma = _wnl.lemmatize(word.lower(), pos='v')
    synsets = wn.synsets(lemma, pos=wn.VERB)
    if not synsets:
        synsets = wn.synsets(word.lower(), pos=wn.VERB)
    if not synsets:
        return {'intransitive': 0.0, 'clausal': 0.0, 'animate_subj': 0.5, 'pp_complement': 0.0}

    all_frames = set()
    for s in synsets[:3]:
        for lem in s.lemmas():
            if lem.name().lower() in (lemma, word.lower()):
                all_frames.update(lem.frame_ids())

    intrans_frames = {1, 2, 3, 4, 6, 7, 22, 23}
    trans_frames = {8, 9, 10, 11}
    clausal_frames = {26, 28, 29, 34}
    pp_frames = {4, 13, 15, 16, 17, 18, 19, 20, 21, 22, 27, 30, 31}
    animate_subj_frames = {2, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33}
    inanimate_subj_frames = {1, 3, 4, 5, 6, 10, 11, 34, 35}

    has_intrans = bool(all_frames & intrans_frames)
    has_trans = bool(all_frames & trans_frames)
    intransitivity = 1.0 if (has_intrans and not has_trans) else (0.5 if has_intrans else 0.0)
    clausal = 1.0 if bool(all_frames & clausal_frames) else 0.0
    animate_count = len(all_frames & animate_subj_frames)
    inanimate_count = len(all_frames & inanimate_subj_frames)
    animate_subj = animate_count / max(animate_count + inanimate_count, 1)
    pp_comp = 1.0 if bool(all_frames & pp_frames) else 0.0

    return {'intransitive': intransitivity, 'clausal': clausal,
            'animate_subj': animate_subj, 'pp_complement': pp_comp}


# Deverbal noun suffixes (discovery R25) — predict PP attachment
DEVERBAL_SUFFIXES = {
    'tion': 0.9, 'sion': 0.9, 'ment': 0.8,
    'ance': 0.6, 'ence': 0.6,
}

# Copula verbs (discovery R25) — distinct from transitive verbs
COPULA_WORDS = {
    'is', 'was', 'are', 'were', 'am', "'s", "'re", "'m",
    'be', 'been', 'being',
    'become', 'became', 'seem', 'seemed', 'remain', 'remained',
    'appear', 'appeared',
}


def get_verb_requirements(word: str) -> set:
    """Look up verb class requirements from VerbNet mapping.

    Returns set of feature indices that the verb's subject must have.
    Tries: direct lookup → WordNetLemmatizer → irregular past table.
    """
    lower = word.lower()

    # 1. Direct lookup
    if lower in VERB_TO_CLASS:
        vclass = VERB_TO_CLASS[lower]
        return VERB_CLASS_REQUIREMENTS.get(vclass, set())

    # 2. WordNetLemmatizer (handles regular inflection: running→run, played→play)
    lemma = get_verb_lemma(lower)
    if lemma in VERB_TO_CLASS:
        vclass = VERB_TO_CLASS[lemma]
        return VERB_CLASS_REQUIREMENTS.get(vclass, set())

    # 3. Irregular past tense table (saw→see, bore→bear)
    if lower in IRREGULAR_PAST_TO_BASE:
        base = IRREGULAR_PAST_TO_BASE[lower]
        if base in VERB_TO_CLASS:
            vclass = VERB_TO_CLASS[base]
            return VERB_CLASS_REQUIREMENTS.get(vclass, set())

    return set()


# =============================================================================
# PROPERTY EXTRACTION (features + requirements)
# =============================================================================

def extract_properties(word: str, ptb_tag: str):
    """Extract 24-dim features (what token IS) and requirements (what token NEEDS).

    24-dim layout (key-query asymmetric):
      KEY (what I provide): 0-9 semantic, 10 instrument
      QUERY (what I seek) + structural: 11-23
      See model.py for full dim map.

    IS/NEEDS separation: nouns/pronouns have KEY features only, verbs/function
    words have QUERY features (what they SEEK). This is the sacred invariant.

    Returns:
        features: np.ndarray [ASA_FEATURE_DIM=24] — filler properties
        requirements: np.ndarray [ASA_FEATURE_DIM=24] — seeker needs
    """
    features = np.zeros(ASA_FEATURE_DIM, dtype=np.float32)
    requirements = np.zeros(ASA_FEATURE_DIM, dtype=np.float32)

    asa_pos = PTB_TO_ASA.get(ptb_tag, "Other")
    lower = word.lower()

    if asa_pos == "Noun":
        # Nouns have KEY features (what they ARE), no requirements
        feat_set = get_noun_features(word)
        if feat_set:
            for idx in feat_set:
                if idx < ASA_FEATURE_DIM:
                    features[idx] = 1.0
        else:
            if ptb_tag in ("NNP", "NNPS"):
                features[0] = 1.0; features[1] = 1.0; features[4] = 1.0
            else:
                features[4] = 1.0
        # Structural properties
        features[7] = 1.0    # can_be_modified
        features[15] = 1.0   # structural_role: entity
        features[16] = 1.5   # NP_orbital
        features[18] = 1.5   # ARG_orbital
        # R25: Morphological frame — deverbal nouns attract PP bonds
        if len(lower) > 4:
            for suffix, val in DEVERBAL_SUFFIXES.items():
                if lower.endswith(suffix):
                    features[8] = val  # morph_frame: PP-attracting
                    break

    elif asa_pos == "Pron":
        # Pronouns have KEY features (what they ARE), no requirements
        if lower in PRONOUN_FEATURES:
            for idx in PRONOUN_FEATURES[lower]:
                if idx < ASA_FEATURE_DIM:
                    features[idx] = 1.0
        else:
            features[0] = 1.0  # ANIMATE default
        # Structural properties
        features[3] = 1.0    # physical
        features[15] = 0.9   # structural_role: entity (slightly less than noun)
        features[16] = 0.5   # NP_orbital: weak NP
        features[18] = 1.5   # ARG_orbital: provides arguments
        # Query dims all 0 — pronouns don't seek

    elif asa_pos == "Verb":
        # R25: Copula detection — distinct verb class
        is_copula = lower in COPULA_WORDS

        # Verbs have QUERY features (what they NEED)
        req_set = get_verb_requirements(word)
        for idx in req_set:
            if idx < ASA_FEATURE_DIM:
                requirements[idx] = 1.0

        # R25: WordNet verb frames — per-verb subcategorization
        vframes = get_verb_frames(word)
        features[8] = vframes['intransitive']    # intransitivity (0=trans, 1=intrans)
        features[9] = vframes['pp_complement']   # takes PP complement
        features[10] = vframes['animate_subj']   # prefers animate subject
        # Use clausal to adjust requirements
        if vframes['clausal'] > 0.5:
            features[10] = max(features[10], 0.7)  # clausal verbs need animate subjects

        if is_copula:
            # Copula: weaker head, seeks predicate not object
            features[12] = 0.0    # no directional preference
            features[13] = 0.3    # moderate reach
            features[15] = 0.4    # weaker structural role
            features[17] = 0.5    # VP_orbital
            features[18] = 0.3    # weak HEAD (copula is linking, not governing)
            features[19] = 0.5    # weak ARG (seeks predicate, not full argument)
            requirements[17] = 0.3
            requirements[19] = 0.5  # weaker ARG_QUERY for copula
        else:
            # Regular verb
            features[12] = -0.3   # directionality: slightly leftward
            if vframes['intransitive'] > 0.8:
                features[13] = 0.1   # intransitive: very short reach
            elif requirements[0] > 0 or len(req_set) > 1:
                features[13] = 0.2   # transitive: short reach
            else:
                features[13] = 0.15
            features[15] = 0.5   # structural_role: predicate
            features[17] = 0.5   # VP_orbital
            features[18] = 0.5   # HEAD_orbital
            features[19] = 1.0   # ARG_orbital: clause head
            requirements[17] = 0.3   # VP_QUERY
            requirements[19] = 1.0   # ARG_QUERY

    elif asa_pos == "Det":
        # Determiners seek nouns (QUERY features)
        features[5] = 1.0    # needs_object (legacy compat)
        features[12] = 1.0   # directionality: rightward
        features[13] = 0.25  # argument_reach
        features[15] = 0.1   # structural_role: function word
        features[16] = 0.5   # NP_orbital
        # Query: Det seeks NP — aligned to NP_orbital dim
        requirements[16] = 1.0   # NP_QUERY → NP_orbital (dim 16)

    elif asa_pos == "Adj":
        # Adjectives modify nouns (QUERY features)
        features[7] = 1.0    # can_modify_noun
        features[12] = 0.8   # directionality: rightward
        features[13] = 0.15  # argument_reach: adjacent
        features[15] = 0.3   # structural_role: modifier
        features[16] = 0.5   # NP_orbital
        # Query: Adj seeks NP — aligned to NP_orbital dim
        requirements[16] = 1.0   # NP_QUERY → NP_orbital (dim 16)

    elif asa_pos == "Adv":
        # Adverbs modify verbs (QUERY features)
        features[6] = 1.0    # can_modify_verb
        features[13] = 0.2   # argument_reach
        features[15] = 0.3   # structural_role: modifier
        features[17] = 0.5   # VP_orbital
        # Query: Adv seeks VP — aligned to VP_orbital dim
        requirements[17] = 1.0   # VP_QUERY → VP_orbital (dim 17)

    elif asa_pos == "Prep":
        # Prepositions: dual role — seek arguments AND heads
        # Prep head preference lookup
        PREP_VERB_PREF = {
            'of': 0.03, 'to': 0.88, 'in': 0.61, 'for': 0.47,
            'on': 0.44, 'from': 0.55, 'by': 0.69, 'with': 0.50,
            'at': 0.68, 'as': 0.53, 'about': 0.07, 'than': 0.00,
            'into': 0.80, 'through': 0.60, 'between': 0.30,
            'under': 0.50, 'after': 0.70, 'before': 0.70,
            'since': 0.80, 'during': 0.50, 'against': 0.60,
            'without': 0.60, 'until': 0.80, 'toward': 0.80,
            'among': 0.40,
        }
        features[11] = PREP_VERB_PREF.get(lower, 0.5)  # prep_head_pref
        features[12] = 0.5   # directionality: dual
        features[13] = 0.3   # argument_reach
        features[14] = 0.85  # bond_absorber
        features[15] = 0.1   # structural_role: function word
        features[16] = 0.8   # NP_orbital: bonds to noun complement
        features[18] = 0.5   # ARG_orbital
        features[19] = 0.5   # HEAD_orbital
        # Query: Prep seeks arguments and heads — aligned to orbital dims
        requirements[18] = 0.8   # HEAD_QUERY → HEAD_orbital (dim 18)
        requirements[19] = 0.8   # ARG_QUERY → ARG_orbital (dim 19)

    # Other POS types get empty vectors (attention controlled by POS mask)

    return features, requirements


# ── Dataset ───────────────────────────────────────────────────

class WikiTextWordLevel(Dataset):
    """WikiText-2 with word-level tokenization, NLTK POS tagging,
    and WordNet/VerbNet feature extraction.

    Caches preprocessed data to disk (.pt files) to avoid re-running
    POS tagging and feature extraction on every run (~2 min saved).
    """

    CACHE_DIR = Path("cache")

    def __init__(self, split: str = "train", seq_len: int = 128,
                 min_vocab_freq: int = 3, max_vocab: int = 20000):

        cache_path = self.CACHE_DIR / f"wikitext2_{split}.pt"

        if cache_path.exists():
            self._load_from_cache(cache_path, seq_len)
        else:
            self._build_from_scratch(split, seq_len, min_vocab_freq, max_vocab)
            self._save_to_cache(cache_path)

    def _load_from_cache(self, cache_path: Path, seq_len: int):
        """Load preprocessed data from disk cache."""
        print(f"Loading cached WikiText-2 from {cache_path}...")
        data = torch.load(cache_path, weights_only=False)
        self.token_ids = data["token_ids"]
        self.pos_ids = data["pos_ids"]
        self.features = data["features"]
        self.requirements = data["requirements"]
        self.word2id = data["word2id"]
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        self.seq_len = seq_len
        self.n_sequences = (len(self.token_ids) - 1) // seq_len
        print(f"  Vocab size: {self.vocab_size:,}, Sequences: {self.n_sequences:,}")

    def _save_to_cache(self, cache_path: Path):
        """Save preprocessed data to disk cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "token_ids": self.token_ids,
            "pos_ids": self.pos_ids,
            "features": self.features,
            "requirements": self.requirements,
            "word2id": self.word2id,
        }, cache_path)
        print(f"  Cached to {cache_path}")

    def _build_from_scratch(self, split: str, seq_len: int,
                            min_vocab_freq: int, max_vocab: int):
        """Full preprocessing: tokenize, POS tag, extract features."""
        from datasets import load_dataset

        print(f"Loading WikiText-2 ({split})...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenize: simple whitespace + lowercase
        all_words = []
        for line in ds["text"]:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            words = line.split()
            all_words.extend(words)

        print(f"  Total words: {len(all_words):,}")

        # Build vocab
        counts = Counter(all_words)
        vocab_words = [w for w, c in counts.most_common(max_vocab) if c >= min_vocab_freq]
        self.word2id = {"<pad>": 0, "<unk>": 1}
        for w in vocab_words:
            self.word2id[w] = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        print(f"  Vocab size: {self.vocab_size:,}")

        # Encode full corpus
        self.token_ids = [self.word2id.get(w, 1) for w in all_words]

        # POS tag the corpus in chunks (NLTK is slow on huge lists)
        print("  POS tagging + feature extraction...")
        self.pos_ids = []
        self.features = []
        self.requirements = []
        chunk_size = 5000
        n_verb_hits = 0
        n_noun_hits = 0
        n_verbs = 0
        n_nouns = 0

        for i in range(0, len(all_words), chunk_size):
            chunk = all_words[i:i + chunk_size]
            tagged = pos_tag(chunk)
            for word, ptb_tag in tagged:
                self.pos_ids.append(ptb_to_asa_id(ptb_tag))
                feat, req = extract_properties(word, ptb_tag)
                self.features.append(feat)
                self.requirements.append(req)

                # Track coverage statistics
                asa_pos = PTB_TO_ASA.get(ptb_tag, "Other")
                if asa_pos == "Verb":
                    n_verbs += 1
                    if req.any():
                        n_verb_hits += 1
                elif asa_pos == "Noun":
                    n_nouns += 1
                    if feat.any():
                        n_noun_hits += 1

        self.seq_len = seq_len
        self.n_sequences = (len(self.token_ids) - 1) // seq_len

        # Report coverage
        verb_cov = n_verb_hits / max(n_verbs, 1) * 100
        noun_cov = n_noun_hits / max(n_nouns, 1) * 100
        print(f"  VerbNet coverage: {n_verb_hits}/{n_verbs} verbs ({verb_cov:.1f}%)")
        print(f"  WordNet coverage: {n_noun_hits}/{n_nouns} nouns ({noun_cov:.1f}%)")
        print(f"  Sequences (len={seq_len}): {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target shift

        ids = self.token_ids[start:end]
        pos = self.pos_ids[start:end]
        feat = self.features[start:end]
        req = self.requirements[start:end]

        # Pad if needed
        while len(ids) < self.seq_len + 1:
            ids.append(0)
            pos.append(POS_IDS["Other"])
            feat.append(np.zeros(ASA_FEATURE_DIM, dtype=np.float32))
            req.append(np.zeros(ASA_FEATURE_DIM, dtype=np.float32))

        return {
            "input_ids": torch.tensor(ids[:self.seq_len], dtype=torch.long),
            "labels": torch.tensor(ids[1:self.seq_len + 1], dtype=torch.long),
            "pos_ids": torch.tensor(pos[:self.seq_len], dtype=torch.long),
            "features": torch.tensor(np.stack(feat[:self.seq_len]), dtype=torch.float32),
            "requirements": torch.tensor(np.stack(req[:self.seq_len]), dtype=torch.float32),
        }


# ── Training ──────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, mode, epoch,
                scheduler=None, log_interval=100):
    """Train for one epoch, return average loss and per-step losses."""
    model.train()
    total_loss = 0.0
    step_losses = []
    n_batches = 0

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        pos_ids = batch["pos_ids"].to(device) if mode != "none" else None
        features = batch["features"].to(device) if mode in ("full", "features_only") else None
        requirements = batch["requirements"].to(device) if mode in ("full", "features_only") else None

        logits = model(input_ids, pos_ids=pos_ids, features=features,
                       requirements=requirements)
        loss = F.cross_entropy(
            logits.reshape(-1, model.output.out_features),
            labels.reshape(-1),
            ignore_index=0  # pad
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        step_losses.append(loss.item())
        n_batches += 1

        if (i + 1) % log_interval == 0:
            avg = total_loss / n_batches
            ppl = math.exp(min(avg, 20))  # cap to avoid overflow
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch} step {i+1}/{len(dataloader)} "
                  f"loss={avg:.4f} ppl={ppl:.2f} lr={cur_lr:.2e}")

    return total_loss / max(n_batches, 1), step_losses


def evaluate(model, dataloader, device, mode):
    """Evaluate perplexity on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            pos_ids = batch["pos_ids"].to(device) if mode != "none" else None
            features = batch["features"].to(device) if mode in ("full", "features_only") else None
            requirements = batch["requirements"].to(device) if mode in ("full", "features_only") else None

            logits = model(input_ids, pos_ids=pos_ids, features=features,
                           requirements=requirements)
            loss = F.cross_entropy(
                logits.reshape(-1, model.output.out_features),
                labels.reshape(-1),
                ignore_index=0
            )
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def check_attention_health(model):
    """Gate 2d: check attention entropy and diversity."""
    issues = []
    for i, layer in enumerate(model.layers):
        weights = layer.attention.last_weights
        if weights is None:
            continue
        # Entropy
        w = weights[:, :, -8:, :]
        entropy = -(w * torch.log(w + 1e-10)).sum(dim=-1).mean().item()
        if entropy < 0.1:
            issues.append(f"Layer {i}: attention entropy {entropy:.4f} (collapsed)")
    return issues


def train(mode: str, epochs: int, device: str, seq_len: int = 128,
          batch_size: int = 8, lr: float = 3e-4, alpha: float = 1.0,
          d_model: int = 128, n_heads: int = 2, n_layers: int = 5,
          d_ff: int = 256):
    """Full training run."""
    print(f"\n{'='*60}")
    print(f"Training ASA Language Model -- mode={mode}, alpha={alpha}")
    print(f"{'='*60}")

    # Load data
    train_ds = WikiTextWordLevel("train", seq_len=seq_len)
    val_ds = WikiTextWordLevel("validation", seq_len=seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    vocab_size = train_ds.vocab_size

    # Build model
    model = ASALanguageModel(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, max_seq_len=seq_len,
        mode=mode, alpha=alpha
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # LR schedule: linear warmup (100 steps) + cosine decay to 0
    # Decay over 1 epoch (not all epochs) to prevent overfitting in later epochs
    total_steps = len(train_dl) * epochs
    decay_steps = len(train_dl)  # 1 epoch worth of steps
    warmup_steps = min(100, decay_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
        progress = min(progress, 1.0)  # clamp after decay_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"LR schedule: {warmup_steps} warmup steps, cosine decay over {decay_steps} steps (1 epoch), {total_steps} total steps")

    # Training loop
    all_step_losses = []
    results = {"mode": mode, "alpha": alpha, "epochs": [], "step_losses": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, step_losses = train_epoch(
            model, train_dl, optimizer, device, mode, epoch, scheduler=scheduler
        )
        all_step_losses.extend(step_losses)
        elapsed = time.time() - t0

        val_loss, val_ppl = evaluate(model, val_dl, device, mode)

        # Attention health check
        health_issues = check_attention_health(model)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
              f"time={elapsed:.1f}s")
        if health_issues:
            print(f"  WARNINGS: {health_issues}")

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "time": elapsed,
            "health_issues": health_issues,
        })

    results["step_losses"] = all_step_losses
    results["n_params"] = n_params
    results["vocab_size"] = vocab_size

    # Gate 2d checks
    print(f"\n{'~'*40}")
    print("Gate 2d Checks:")

    # 1. Loss decreased
    first_100 = np.mean(all_step_losses[:100])
    last_100 = np.mean(all_step_losses[-100:])
    loss_decreased = last_100 < first_100
    print(f"  Loss decrease: {first_100:.4f} -> {last_100:.4f} "
          f"({'PASS' if loss_decreased else 'FAIL'})")

    # 2. No attention collapse
    no_collapse = len(results["epochs"][-1]["health_issues"]) == 0
    print(f"  Attention health: {'PASS' if no_collapse else 'FAIL'}")

    # 3. Perplexity is finite
    final_ppl = results["epochs"][-1]["val_ppl"]
    ppl_ok = not math.isinf(final_ppl) and not math.isnan(final_ppl) and final_ppl > 0
    print(f"  Perplexity finite: {final_ppl:.2f} ({'PASS' if ppl_ok else 'FAIL'})")

    gate_2d = loss_decreased and no_collapse and ppl_ok
    print(f"\nGate 2d: {'PASS' if gate_2d else 'FAIL'}")

    # Save results
    out_path = Path(f"results_{mode}_a{alpha}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    return results


def alpha_sweep(epochs: int, device: str):
    """Sweep alpha values to find optimal ASA bias strength."""
    print("\n" + "=" * 60)
    print("Alpha Sweep: finding optimal ASA bias strength")
    print("=" * 60)

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for alpha in alphas:
        print(f"\n--- alpha={alpha} ---")
        r = train("full", epochs, device, alpha=alpha)
        final = r["epochs"][-1]
        results[alpha] = {
            "train_loss": final["train_loss"],
            "val_loss": final["val_loss"],
            "val_ppl": final["val_ppl"],
        }

    # Also run baseline
    print(f"\n--- baseline (mode=none) ---")
    r = train("none", epochs, device)
    final = r["epochs"][-1]
    results["baseline"] = {
        "train_loss": final["train_loss"],
        "val_loss": final["val_loss"],
        "val_ppl": final["val_ppl"],
    }

    # Summary
    print("\n" + "=" * 60)
    print("Alpha Sweep Summary")
    print("=" * 60)
    print(f"{'Config':>12}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 50)

    best_alpha = None
    best_val = float("inf")
    for key in alphas + ["baseline"]:
        r = results[key]
        label = f"a={key}" if isinstance(key, float) else key
        print(f"{label:>12}  {r['train_loss']:>10.4f}  {r['val_loss']:>10.4f}  {r['val_ppl']:>10.2f}")
        if isinstance(key, float) and r["val_loss"] < best_val:
            best_val = r["val_loss"]
            best_alpha = key

    baseline_val = results["baseline"]["val_loss"]
    print(f"\nBest alpha: {best_alpha} (val_loss={best_val:.4f})")
    print(f"Baseline val_loss: {baseline_val:.4f}")
    if best_val < baseline_val:
        print(f"ASA WINS by {(baseline_val - best_val):.4f}")
    else:
        print(f"Baseline wins by {(best_val - baseline_val):.4f}")

    # Save sweep results
    with open("results_alpha_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def compare_modes(epochs: int, device: str, alpha: float = 1.0, lr: float = 3e-4,
                   batch_size: int = 8, d_model: int = 128, n_heads: int = 2,
                   n_layers: int = 5, d_ff: int = 256):
    """Run full vs none and compare."""
    print("\n" + "=" * 60)
    print(f"A/B Comparison: ASA (full, alpha={alpha}) vs Baseline (none)")
    print("=" * 60)

    results_full = train("full", epochs, device, alpha=alpha, lr=lr, batch_size=batch_size,
                         d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    results_none = train("none", epochs, device, lr=lr, batch_size=batch_size,
                         d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    for mode, results in [("full", results_full), ("none", results_none)]:
        final = results["epochs"][-1]
        print(f"  {mode:>12}: val_ppl={final['val_ppl']:.2f} "
              f"val_loss={final['val_loss']:.4f} "
              f"train_loss={final['train_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASA Language Model")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "pos_only", "features_only", "none"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--compare", action="store_true",
                        help="Run full vs none comparison")
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Sweep alpha values (0.1, 0.5, 1.0, 2.0, 5.0)")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        args.device = "mps"
        print(f"Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        args.device = "cuda"
        print(f"Using CUDA")
    else:
        print(f"Using CPU")

    if args.alpha_sweep:
        alpha_sweep(args.epochs, args.device)
    elif args.compare:
        compare_modes(args.epochs, args.device, alpha=args.alpha, lr=args.lr,
                      batch_size=args.batch_size, d_model=args.d_model,
                      n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff)
    else:
        train(args.mode, args.epochs, args.device,
              seq_len=args.seq_len, batch_size=args.batch_size,
              lr=args.lr, alpha=args.alpha, d_model=args.d_model,
              n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff)
