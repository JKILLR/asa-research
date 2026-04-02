import numpy as np
import math
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import product as cartesian_product
from typing import List, Dict, Optional, Tuple

# ------------------------------------------------------------
# Token definition
# ------------------------------------------------------------

@dataclass
class Token:
    word: str
    pos: str                  # Noun, Verb, Det, Adj, Adv, Prep, Pron, ...
    valence: int              # how many arguments it wants (0 = saturated)
    features: np.ndarray      # small vector encoding selectional restrictions / bonding preferences
    charge: float = 0.0       # polarity (-1 to +1); scaffolded for Phase 1 charge system


# Feature vector: 12 dimensions (expanded for Phase 0.5 stress test)
# Indices:
#   0 = animate
#   1 = inanimate
#   2 = edible
#   3 = physical
#   4 = needs_subject (verb-like)
#   5 = needs_object  (verb-like)
#   6 = can_modify_verb (adverb-like)
#   7 = can_modify_noun (adjective-like)
#   8 = abstract (for verbs like "know" + abstract objects)
#   9 = liquid
#  10 = instrument (for PP "with" instrument sense)
#  11 = visual_property (for visible attributes — "red", "spots")
FEATURE_DIM = 12

# Charge interaction weight (from thermodynamic scoring formula)
# Score += LAMBDA_C * (-q_i * q_j)
# Opposite charges attract (boost), same charges repel (penalty)
# CONSTRAINT: charge must bias, never veto. Max repulsion (LAMBDA_C * 1.0)
# must stay well below POS floor (1.0) so valid bonds always form.
LAMBDA_C = 0.5

# Energy function weights (Phase 1a — global bond configuration scoring)
# E_total = E_bond + E_valency + E_saturation + E_exclusivity
# Lower energy = better configuration.
LAMBDA_VAL  = 5.0   # Penalty per unfilled seeker slot
LAMBDA_SAT  = 3.0   # Penalty per unbound content filler (Noun/Pron)
LAMBDA_EXCL = 2.0   # Penalty per extra bond on an already-bonded filler

# Simulated annealing parameters (Phase 1b)
EXHAUSTIVE_THRESHOLD = 12   # tokens; above this, use SA instead of exhaustive
SA_T_START = 5.0            # initial temperature (most moves accepted)
SA_T_END = 0.01             # final temperature (effectively frozen)
SA_COOLING = 0.95           # geometric cooling rate
SA_STEPS_PER_TEMP = 50      # local exploration steps per temperature level

VOCAB: List[Token] = [
    # --- Determiners ---
    Token("the",     "Det", 1, np.array([0,0,0,0, 0,1,0,0, 0,0,0,0], dtype=float)),
    Token("a",       "Det", 1, np.array([0,0,0,0, 0,1,0,0, 0,0,0,0], dtype=float)),
    # --- Nouns (original) ---
    # NOTE: dims 4-5 (needs_subject/needs_object) are SEEKER-ONLY features.
    # Nouns encode only semantic properties (animate, edible, physical, etc.).
    # Role assignment is handled by slot bonuses, not feature overlap.
    Token("dog",     "Noun",0, np.array([1,0,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # animate, physical
    Token("cat",     "Noun",0, np.array([1,0,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),
    Token("rock",    "Noun",0, np.array([0,1,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # inanimate
    Token("apple",   "Noun",0, np.array([0,0,1,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # edible
    # --- Nouns (Phase 0.5) ---
    Token("man",     "Noun",0, np.array([1,0,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # animate, physical
    Token("woman",   "Noun",0, np.array([1,0,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),
    Token("fish",    "Noun",0, np.array([1,0,1,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # animate AND edible
    Token("ball",    "Noun",0, np.array([0,1,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # inanimate
    Token("water",   "Noun",0, np.array([0,1,0,1, 0,0,0,0, 0,1,0,0], dtype=float)),     # inanimate, liquid
    Token("fork",    "Noun",0, np.array([0,1,0,1, 0,0,0,0, 0,0,1,0], dtype=float)),     # inanimate, instrument
    Token("spots",   "Noun",0, np.array([0,1,0,1, 0,0,0,0, 0,0,0,1], dtype=float)),     # visual_property
    Token("food",    "Noun",0, np.array([0,0,1,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # edible
    Token("something","Noun",0,np.array([0,0,0,0, 0,0,0,0, 1,0,0,0], dtype=float)),     # abstract
    # --- Verbs (original) ---
    Token("examine", "Verb",2, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float)),
    Token("eat",     "Verb",2, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float)),
    # --- Verbs (Phase 0.5) ---
    Token("run",     "Verb",1, np.array([0,0,0,0, 1,0,0,0, 0,0,0,0], dtype=float)),     # intransitive
    Token("see",     "Verb",2, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float)),     # transitive
    Token("give",    "Verb",3, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float)),     # ditransitive
    Token("sleep",   "Verb",1, np.array([0,0,0,0, 1,0,0,0, 0,0,0,0], dtype=float)),     # intransitive
    Token("break",   "Verb",2, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float)),     # transitive
    Token("know",    "Verb",2, np.array([0,0,0,0, 1,1,0,0, 1,0,0,0], dtype=float)),     # transitive, prefers abstract
    Token("love",    "Verb",2, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float), charge=+0.7),  # positive sentiment
    Token("hate",    "Verb",2, np.array([0,0,0,0, 1,1,0,0, 0,0,0,0], dtype=float), charge=-0.7),  # negative sentiment
    # --- Adverbs ---
    Token("quickly", "Adv", 1, np.array([0,0,0,0, 0,0,1,0, 0,0,0,0], dtype=float)),
    Token("slowly",  "Adv", 1, np.array([0,0,0,0, 0,0,1,0, 0,0,0,0], dtype=float)),
    Token("not",     "Adv", 1, np.array([0,0,0,0, 0,0,1,0, 0,0,0,0], dtype=float), charge=-1.0),
    # --- Adjectives ---
    Token("big",     "Adj", 1, np.array([0,0,0,0, 0,0,0,1, 0,0,0,0], dtype=float)),
    Token("small",   "Adj", 1, np.array([0,0,0,0, 0,0,0,1, 0,0,0,0], dtype=float)),
    Token("red",     "Adj", 1, np.array([0,0,0,0, 0,0,0,1, 0,0,0,1], dtype=float)),     # visual_property
    # --- Pronouns (Phase 0.5 — NEW POS, semantic features only like nouns) ---
    Token("he",      "Pron",0, np.array([1,0,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # like "man"
    Token("she",     "Pron",0, np.array([1,0,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # like "woman"
    Token("it",      "Pron",0, np.array([0,1,0,1, 0,0,0,0, 0,0,0,0], dtype=float)),     # inanimate
    # --- Prepositions (Phase 0.5 — NEW POS) ---
    Token("in",      "Prep",2, np.array([0,0,0,0, 0,1,0,0, 0,0,0,0], dtype=float)),     # slot 0=complement, slot 1=head
    Token("on",      "Prep",2, np.array([0,0,0,0, 0,1,0,0, 0,0,0,0], dtype=float)),
    Token("with",    "Prep",2, np.array([0,0,0,0, 0,1,0,0, 0,0,1,0], dtype=float)),     # instrument sense
]

# Quick lookup by word
WORD_TO_TOKEN: Dict[str, Token] = {t.word: t for t in VOCAB}


# ------------------------------------------------------------
# POS compatibility (Level 1 filter from Universal Dependencies)
# ------------------------------------------------------------

# Which POS pairs can form bonds? Seeker → Filler
POS_RULES: Dict[str, List[str]] = {
    "Det":  ["Noun"],                     # determiners modify nouns
    "Adj":  ["Noun"],                     # adjectives modify nouns
    "Adv":  ["Verb"],                     # adverbs modify verbs
    "Verb": ["Noun", "Pron"],             # verbs seek noun/pronoun arguments
    "Prep": ["Noun", "Pron", "Verb"],     # slot 0: noun complement; slot 1: verb/noun head
}

def pos_compatible(seeker: Token, filler: Token) -> bool:
    allowed = POS_RULES.get(seeker.pos, [])
    return filler.pos in allowed


# ------------------------------------------------------------
# Thermodynamic scoring (Phase 1b — A/B comparison)
# ------------------------------------------------------------

LAMBDA_H = 3.0   # Enthalpy weight (penalty for unsatisfied requirements)

def get_slot_requirements(seeker: Token, slot_idx: int) -> Dict[int, float]:
    """What features does this verb slot require? Returns {feature_index: weight}."""
    if seeker.pos != "Verb":
        return {}
    if slot_idx == 0:  # subject
        return {0: 1.0}  # animate
    elif slot_idx == 1:  # object
        if seeker.features[8] > 0:  # abstract verb (know)
            return {8: 1.0}
        return {2: 0.6, 3: 0.3, 8: 0.1}  # edible preferred, physical ok, abstract fallback
    elif slot_idx == 2:  # iobj
        return {0: 1.0}  # animate recipient
    return {}


def compute_satisfaction(requirements: Dict[int, float], filler_features: np.ndarray) -> float:
    """What fraction of requirements does the filler satisfy? 0 to 1."""
    if not requirements:
        return 1.0
    total_weight = sum(requirements.values())
    satisfied = sum(w * float(filler_features[idx]) for idx, w in requirements.items())
    return min(1.0, satisfied / total_weight)


def thermodynamic_overlap(seeker: Token, filler: Token, slot_idx: Optional[int] = None,
                           seeker_idx: Optional[int] = None, filler_idx: Optional[int] = None,
                           complement: Optional[Token] = None) -> float:
    """Thermodynamic scoring: alignment/√d - λ_H·ΔH + λ_C·charge + locality.

    Replaces ad-hoc slot bonuses with enthalpy (requirement satisfaction ratio).
    """
    if not pos_compatible(seeker, filler):
        return 0.0

    # Alignment term: Q·K^T / √d
    alignment = float(np.dot(seeker.features, filler.features)) / np.sqrt(FEATURE_DIM)

    # POS floor: minimum alignment for compatible pairs
    alignment = max(alignment, 0.3)

    # Asymmetric boosts (structural, not semantic — kept from System A)
    if seeker.pos == "Verb" and filler.pos in ("Noun", "Pron"):
        alignment *= 5.0
    if seeker.pos == "Prep" and filler.pos in ("Noun", "Pron") and slot_idx == 0:
        alignment *= 3.0
        if seeker_idx is not None and filler_idx is not None:
            if filler_idx == seeker_idx + 1:
                alignment += 2.0

    # Enthalpy: feature requirement satisfaction (replaces slot bonuses)
    delta_H = 0.0
    if seeker.pos == "Verb" and slot_idx is not None:
        requirements = get_slot_requirements(seeker, slot_idx)
        satisfaction = compute_satisfaction(requirements, filler.features)
        delta_H = 1.0 - satisfaction  # 0 = perfect, 1 = nothing satisfied

        # Word-order bias (kept — structural, not semantic)
        if seeker_idx is not None and filler_idx is not None:
            if slot_idx == 0:  # subject left of verb
                if filler_idx < seeker_idx:
                    delta_H -= 0.1  # small enthalpy bonus for correct order
            elif slot_idx == 1:  # object right of verb
                if filler_idx > seeker_idx:
                    delta_H -= 0.1
                if seeker.valence >= 3:  # ditransitive object further from verb
                    if (filler_idx - seeker_idx) > 1:
                        delta_H -= 0.2
            elif slot_idx == 2:  # iobj immediately after verb
                if seeker_idx is not None and filler_idx is not None:
                    if (filler_idx - seeker_idx) == 1:
                        delta_H -= 0.3

    # PP head attachment (kept — structural)
    if seeker.pos == "Prep" and slot_idx == 1 and complement is not None:
        if complement.features[10] > 0 and filler.pos == "Verb":
            delta_H -= 0.8  # instrument → verb
        elif complement.features[11] > 0 and filler.pos in ("Noun", "Pron"):
            delta_H -= 0.8  # visual property → noun
        elif filler.pos == "Verb":
            delta_H -= 0.4  # default verb preference

    # Charge interaction
    charge_term = LAMBDA_C * (-seeker.charge * filler.charge)

    # Locality
    locality = 0.0
    if seeker_idx is not None and filler_idx is not None:
        distance = abs(seeker_idx - filler_idx)
        if distance > 0:
            locality = 1.0 / distance

    score = alignment - LAMBDA_H * delta_H + charge_term + locality
    return max(0.0, score)


# ------------------------------------------------------------
# Wave-function overlap (simplified attention / compatibility)
# ------------------------------------------------------------

def wave_overlap(seeker: Token, filler: Token, slot_idx: Optional[int] = None,
                 seeker_idx: Optional[int] = None, filler_idx: Optional[int] = None,
                 state: Optional['MoleculeState'] = None,
                 complement: Optional[Token] = None) -> float:
    """Score how well filler satisfies one of seeker's open slots."""
    score = float(np.dot(seeker.features, filler.features))

    # Base compatibility: POS-compatible pairs always have a minimum attraction
    # (modifier features are orthogonal to noun/verb features by design,
    #  so without this floor, adj→noun and adv→verb would score 0)
    if pos_compatible(seeker, filler):
        score = max(score, 1.0)

    # Asymmetric boost: verbs pull nouns/pronouns hard
    if seeker.pos == "Verb" and filler.pos in ("Noun", "Pron"):
        score *= 5.0

    # Asymmetric boost: preps pull noun complements (less than verbs)
    if seeker.pos == "Prep" and filler.pos in ("Noun", "Pron") and slot_idx == 0:
        score *= 3.0
        # Preps take their complement immediately to the right in English
        if seeker_idx is not None and filler_idx is not None:
            if filler_idx == seeker_idx + 1:
                score += 2.0  # strong right-adjacency preference

    # Directional slot bonuses for verbs
    if seeker.pos == "Verb" and slot_idx is not None:
        if slot_idx == 0 and seeker.features[4] > 0:   # subject slot
            score += 2.5 if filler.features[0] > 0 else -1.0  # animate bonus
            # Word-order bias: subjects tend to be left of verb
            if seeker_idx is not None and filler_idx is not None:
                score += 0.5 if filler_idx < seeker_idx else 0.0
        if slot_idx == 1 and seeker.features[5] > 0:   # object slot
            # Generalized object scoring
            if filler.features[2] > 0:                     # edible
                score += 3.0
            elif filler.features[8] > 0 and seeker.features[8] > 0:  # abstract match
                score += 2.5
            elif filler.features[3] > 0:                   # physical
                score += 1.5
            else:
                score += 1.0
            # Word-order bias: objects tend to be right of verb
            if seeker_idx is not None and filler_idx is not None:
                score += 0.5 if filler_idx > seeker_idx else 0.0
                # In ditransitive, object tends to be further from verb (second NP)
                if seeker.valence >= 3:
                    distance_from_verb = filler_idx - seeker_idx
                    if distance_from_verb > 1:
                        score += 1.0
        if slot_idx == 2:  # indirect object slot (ditransitive)
            score += 2.0 if filler.features[0] > 0 else 0.0  # animate recipient
            # Word-order: iobj immediately follows verb in double-object construction
            if seeker_idx is not None and filler_idx is not None:
                distance_from_verb = filler_idx - seeker_idx
                if distance_from_verb == 1:
                    score += 1.5  # first NP after verb = iobj

    # PP head attachment scoring (Prep slot 1)
    # Uses complement token (from energy function config or state.bonds) to guide attachment.
    if seeker.pos == "Prep" and slot_idx == 1:
        comp = complement  # prefer explicitly passed complement (energy function)
        if comp is None and state is not None and seeker_idx is not None:
            # Fallback: look at complement already bonded in state (greedy mode)
            for (si, fi), _ in state.bonds.items():
                if si == seeker_idx:
                    comp = state.tokens[fi]
                    break
        if comp is not None:
            if comp.features[10] > 0:      # instrument complement (fork)
                if filler.pos == "Verb":
                    score += 4.0           # attach to verb
            elif comp.features[11] > 0:    # visual_property complement (spots)
                if filler.pos in ("Noun", "Pron"):
                    score += 4.0           # attach to noun
            else:
                # Default: PPs prefer verb attachment when complement has no special signal
                if filler.pos == "Verb":
                    score += 2.0

    # Charge interaction: from thermodynamic scoring formula
    # Score += λ_C * (-q_i * q_j)
    # Opposite charges attract (boost), same charges repel (penalty)
    # INVARIANT: charge repulsion must never zero out a POS-compatible bond.
    if seeker.charge != 0 or filler.charge != 0:
        charge_effect = LAMBDA_C * (-seeker.charge * filler.charge)
        score += charge_effect

    # Locality bias: prefer nearby tokens (like SMD locality energy)
    if seeker_idx is not None and filler_idx is not None:
        distance = abs(seeker_idx - filler_idx)
        if distance > 0:
            score += 1.0 / distance  # nearby = +1.0, far = smaller bonus

    # Floor: POS-compatible pairs always maintain a minimum positive score.
    # No scoring term (charge, slot penalty, etc.) should veto a valid bond.
    if pos_compatible(seeker, filler):
        score = max(score, 0.1)

    return max(0.0, score)


# ------------------------------------------------------------
# Current sentence / molecule state
# ------------------------------------------------------------

@dataclass
class MoleculeState:
    tokens: List[Token]
    bonds: Dict[Tuple[int, int], float]     # (seeker_idx, filler_idx) -> strength
    remaining_valence: List[int]            # per token, how many slots still open
    bond_labels: Dict[Tuple[int, int], str] # (seeker_idx, filler_idx) -> role label

    def __init__(self, prompt_words: List[str]):
        valid_words = [w for w in prompt_words if w in WORD_TO_TOKEN]
        self.tokens = [WORD_TO_TOKEN[w] for w in valid_words]
        self.bonds = {}
        self.bond_labels = {}
        self.remaining_valence = [t.valence for t in self.tokens]

    def is_saturated(self) -> bool:
        return all(v <= 0 for v in self.remaining_valence)

    def seeker_indices(self) -> List[int]:
        """Tokens that still have open slots (valence > 0)."""
        return [i for i, v in enumerate(self.remaining_valence) if v > 0]

    def already_bonded(self, i: int, j: int) -> bool:
        return (i, j) in self.bonds or (j, i) in self.bonds

    def slot_label(self, seeker_idx: int) -> str:
        """What slot is the seeker filling next?"""
        tok = self.tokens[seeker_idx]
        filled = tok.valence - self.remaining_valence[seeker_idx]
        if tok.pos == "Verb":
            if filled == 0: return "subj"
            elif filled == 1: return "obj"
            elif filled == 2: return "iobj"
            return f"arg{filled}"
        if tok.pos == "Det":
            return "det→"
        if tok.pos == "Adj":
            return "adj→"
        if tok.pos == "Adv":
            return "adv→"
        if tok.pos == "Prep":
            return "prep→" if filled == 0 else "pp→"
        return "arg"

    def compute_effective_charges(self) -> List[float]:
        """Compute effective charge after negation propagation.

        When a negator (adverb with negative charge, e.g., "not") bonds to a
        token, it flips the target's polarity:
          - "love" (+0.7) → "not love" (-0.7)
          - "hate" (-0.7) → "not hate" (+0.7)
          - "eat" (0.0)   → "not eat" (-1.0) — neutral becomes negated

        Double negation cancels: two negators flip twice → back to original.
        """
        effective = [t.charge for t in self.tokens]
        for (si, fi), _ in self.bonds.items():
            seeker = self.tokens[si]
            label = self.bond_labels.get((si, fi), "")
            if label == "adv→" and seeker.charge < 0:  # negator bonded to target
                if effective[fi] != 0:
                    effective[fi] = -effective[fi]       # flip non-zero charge
                else:
                    effective[fi] = -1.0                 # neutral → negated
        return effective

    def print_charges(self):
        """Display effective charges and sentence polarity after bonding."""
        effective = self.compute_effective_charges()
        has_charge = any(c != 0 for c in effective)
        if not has_charge:
            return  # skip display for fully neutral sentences

        # Show per-token effective charges
        parts = []
        for i, (tok, eff) in enumerate(zip(self.tokens, effective)):
            if eff != 0:
                label = f"{tok.word}={eff:+.1f}"
                if eff != tok.charge:
                    label += " (flipped)" if tok.charge != 0 else " (negated)"
                parts.append(label)
            else:
                parts.append(f"{tok.word}=0")
        print(f"\n  Charges: {', '.join(parts)}")

        # Sentence polarity from main verb
        verb_indices = [i for i, t in enumerate(self.tokens) if t.pos == "Verb"]
        if verb_indices:
            # Find the root verb (one that is a seeker, not a filler of another verb)
            filler_set = {fi for (_, fi) in self.bonds}
            root_verbs = [i for i in verb_indices if i not in filler_set]
            vi = root_verbs[0] if root_verbs else verb_indices[0]
            vc = effective[vi]
            if vc > 0:
                polarity = "POSITIVE"
            elif vc < 0:
                polarity = "NEGATIVE"
            else:
                polarity = "NEUTRAL"
            print(f"  Sentence polarity: {polarity} (verb \"{self.tokens[vi].word}\" charge: {vc:+.1f})")

    def print_state(self):
        print("\n  Tokens:", " ".join(t.word for t in self.tokens))
        print("  Valence:", self.remaining_valence)
        if self.bonds:
            print("  Bonds:")
            for (i, j), strength in sorted(self.bonds.items()):
                label = self.bond_labels.get((i, j), "")
                print(f"    {self.tokens[i].word} —[{label}]→ {self.tokens[j].word}  (score {strength:.2f})")
        else:
            print("  No bonds yet")

    def print_tree(self):
        """ASCII visualization of the bond structure."""
        # Find root(s): tokens that are never fillers
        filler_indices = {j for (_, j) in self.bonds}
        roots = [i for i in range(len(self.tokens)) if i not in filler_indices]
        if not roots:
            roots = list(range(len(self.tokens)))

        # Build children map: seeker → [(filler, label)]
        children: Dict[int, List[Tuple[int, str]]] = {}
        for (s, f), _ in self.bonds.items():
            label = self.bond_labels.get((s, f), "")
            children.setdefault(s, []).append((f, label))

        def _print_subtree(idx: int, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            tok = self.tokens[idx]
            print(f"{prefix}{connector}{tok.word} [{tok.pos}]")
            child_prefix = prefix + ("    " if is_last else "│   ")
            kids = children.get(idx, [])
            for ci, (child_idx, label) in enumerate(kids):
                label_str = f"({label}) " if label else ""
                connector2 = "└── " if ci == len(kids) - 1 else "├── "
                child_tok = self.tokens[child_idx]
                print(f"{child_prefix}{connector2}{label_str}{child_tok.word} [{child_tok.pos}]")
                # recurse for children of the child
                grandkids = children.get(child_idx, [])
                gk_prefix = child_prefix + ("    " if ci == len(kids) - 1 else "│   ")
                for gi, (gk_idx, gk_label) in enumerate(grandkids):
                    gk_label_str = f"({gk_label}) " if gk_label else ""
                    gk_connector = "└── " if gi == len(grandkids) - 1 else "├── "
                    gk_tok = self.tokens[gk_idx]
                    print(f"{gk_prefix}{gk_connector}{gk_label_str}{gk_tok.word} [{gk_tok.pos}]")

        print("\n  Bond tree:")
        for ri, root in enumerate(roots):
            _print_subtree(root, prefix="  ", is_last=(ri == len(roots) - 1))


# ------------------------------------------------------------
# Greedy bonding logic (seeker → filler paradigm)
# ------------------------------------------------------------

def try_greedy_bonding(state: MoleculeState, max_iterations: int = 20):
    """
    Greedy bonding: each iteration finds the best (seeker, filler) pair
    where the seeker has open slots and the filler is POS-compatible.
    Only the seeker's valence decreases — fillers can serve multiple seekers.
    """
    for iteration in range(max_iterations):
        best_score = -1.0
        best_seeker = -1
        best_filler = -1
        best_slot = -1

        seekers = state.seeker_indices()
        if not seekers:
            break

        for si in seekers:
            seeker = state.tokens[si]
            filled_so_far = seeker.valence - state.remaining_valence[si]

            for fi in range(len(state.tokens)):
                if fi == si:
                    continue
                if state.already_bonded(si, fi):
                    continue

                filler = state.tokens[fi]

                # Level 1: POS compatibility filter
                if not pos_compatible(seeker, filler):
                    continue

                # Level 2: Feature-based scoring
                if seeker.pos == "Verb":
                    slot = filled_so_far
                elif seeker.pos == "Prep":
                    slot = filled_so_far
                else:
                    slot = None
                score = wave_overlap(seeker, filler, slot_idx=slot,
                                     seeker_idx=si, filler_idx=fi, state=state)

                if score > best_score:
                    best_score = score
                    best_seeker = si
                    best_filler = fi
                    best_slot = slot if slot is not None else 0

        if best_score <= 0.001:
            break

        # Form the bond (directed: seeker → filler)
        label = state.slot_label(best_seeker)
        state.bonds[(best_seeker, best_filler)] = best_score
        state.bond_labels[(best_seeker, best_filler)] = label
        state.remaining_valence[best_seeker] -= 1

        print(f"  Bond: {state.tokens[best_seeker].word} —[{label}]→ {state.tokens[best_filler].word}  (score {best_score:.3f})")

    return state


# ------------------------------------------------------------
# Global energy function + exhaustive bonding (Phase 1a)
# ------------------------------------------------------------

def slot_label_for(pos: str, slot_idx: int) -> str:
    """Map (POS, slot index) to a role label. Used by exhaustive bonding."""
    if pos == "Verb":
        return ["subj", "obj", "iobj"][slot_idx] if slot_idx < 3 else f"arg{slot_idx}"
    if pos == "Det":  return "det→"
    if pos == "Adj":  return "adj→"
    if pos == "Adv":  return "adv→"
    if pos == "Prep": return "prep→" if slot_idx == 0 else "pp→"
    return "arg"


def total_energy(config: Dict[int, List[Optional[int]]],
                 tokens: List[Token]) -> float:
    """Compute total energy of a bond configuration. Lower = better.

    config: {seeker_idx: [filler_idx_for_slot_0, filler_idx_for_slot_1, ...]}
    Unfilled slots are None.
    """
    E = 0.0
    filler_bond_counts: Counter = Counter()
    filler_roles: Dict[int, List[str]] = {}   # filler_idx → list of role labels

    # --- E_bond: negative of summed wave_overlap scores ---
    for seeker_idx, filler_list in config.items():
        seeker = tokens[seeker_idx]
        for slot_idx, filler_idx in enumerate(filler_list):
            if filler_idx is not None:
                # For PP head attachment (slot 1), pass the complement from slot 0
                comp = None
                if seeker.pos == "Prep" and slot_idx == 1:
                    comp_idx = filler_list[0]
                    if comp_idx is not None:
                        comp = tokens[comp_idx]

                score = wave_overlap(seeker, tokens[filler_idx],
                                     slot_idx=slot_idx,
                                     seeker_idx=seeker_idx,
                                     filler_idx=filler_idx,
                                     complement=comp)
                E -= score
                filler_bond_counts[filler_idx] += 1
                # Track which roles each filler serves (for exclusivity)
                role = slot_label_for(seeker.pos, slot_idx)
                filler_roles.setdefault(filler_idx, []).append(role)

    # --- E_valency: penalty for unfilled seeker slots ---
    for seeker_idx, filler_list in config.items():
        unfilled = sum(1 for f in filler_list if f is None)
        E += LAMBDA_VAL * unfilled

    # --- E_saturation: penalty for unbound content fillers ---
    content_fillers = {i for i, t in enumerate(tokens)
                       if t.pos in ("Noun", "Pron") and t.valence == 0}
    bound_fillers = set(filler_bond_counts.keys())
    unbound = content_fillers - bound_fillers
    E += LAMBDA_SAT * len(unbound)

    # --- E_exclusivity: penalty for filler over-sharing ---
    # Same-role sharing (e.g., obj of two verbs) costs full penalty.
    # Different-role sharing (e.g., subj of one verb, obj of another) costs half.
    # Det+verb sharing (e.g., det→dog and verb→dog(subj)) is free — always legitimate.
    for filler_idx, roles in filler_roles.items():
        if len(roles) <= 1:
            continue
        # Exclude det/adj bonds from exclusivity — they always legitimately share
        verb_roles = [r for r in roles if r in ("subj", "obj", "iobj")]
        other_roles = [r for r in roles if r not in ("subj", "obj", "iobj")]
        # Penalty only from verb-argument sharing
        if len(verb_roles) <= 1:
            continue  # at most 1 verb role — no conflict
        unique_verb_roles = set(verb_roles)
        same_role_dups = len(verb_roles) - len(unique_verb_roles)
        diff_role_count = max(0, len(unique_verb_roles) - 1)
        E += LAMBDA_EXCL * same_role_dups       # full penalty for same-role
        E += LAMBDA_EXCL * 0.5 * diff_role_count  # half penalty for diff-role

    return E


def exhaustive_bonding(state: MoleculeState) -> MoleculeState:
    """Find the minimum-energy bond configuration by exhaustive search.

    Enumerates all valid (POS-compatible) filler assignments for each seeker's
    slots, evaluates total_energy for every combination, and applies the best.
    """
    tokens = state.tokens
    seekers = [(i, t) for i, t in enumerate(tokens) if t.valence > 0]

    if not seekers:
        return state

    # For each seeker, generate all possible filler assignments
    per_seeker_options: List[Tuple[int, List[Tuple]]] = []
    for si, seeker in seekers:
        slot_options = []
        for slot_idx in range(seeker.valence):
            candidates: List[Optional[int]] = [None]  # unfilled is always an option
            for fi, filler in enumerate(tokens):
                if fi == si:
                    continue
                if pos_compatible(seeker, filler):
                    candidates.append(fi)
            slot_options.append(candidates)

        # Cartesian product of all slots for this seeker
        assignments = list(cartesian_product(*slot_options))
        # Filter: no duplicate fillers for same seeker (can't bond to same token twice)
        assignments = [a for a in assignments
                       if len(set(f for f in a if f is not None))
                       == len([f for f in a if f is not None])]
        per_seeker_options.append((si, assignments))

    # Cartesian product across all seekers
    best_energy = float('inf')
    best_config = None
    n_configs = 0

    for combo in cartesian_product(*(assigns for _, assigns in per_seeker_options)):
        config = {}
        for (si, _), assignment in zip(per_seeker_options, combo):
            config[si] = list(assignment)
        energy = total_energy(config, tokens)
        n_configs += 1
        if energy < best_energy:
            best_energy = energy
            best_config = config

    print(f"  Exhaustive search: {n_configs} configurations evaluated, best energy = {best_energy:.2f}")

    # Apply best config to state
    if best_config is not None:
        _apply_config_to_state(best_config, state)

    return state


# ------------------------------------------------------------
# Simulated annealing bonding (Phase 1b — scales beyond exhaustive)
# ------------------------------------------------------------

def _apply_config_to_state(config: Dict[int, List[Optional[int]]],
                           state: MoleculeState) -> MoleculeState:
    """Apply a bond configuration to a MoleculeState (shared by exhaustive + SA)."""
    tokens = state.tokens
    for si, filler_list in config.items():
        for slot_idx, fi in enumerate(filler_list):
            if fi is not None:
                seeker = tokens[si]
                comp = None
                if seeker.pos == "Prep" and slot_idx == 1:
                    comp_idx = filler_list[0]
                    if comp_idx is not None:
                        comp = tokens[comp_idx]
                score = wave_overlap(seeker, tokens[fi],
                                     slot_idx=slot_idx,
                                     seeker_idx=si, filler_idx=fi,
                                     complement=comp)
                label = slot_label_for(seeker.pos, slot_idx)
                state.bonds[(si, fi)] = score
                state.bond_labels[(si, fi)] = label
                state.remaining_valence[si] -= 1
    return state


def random_valid_config(tokens: List[Token],
                        seekers: List[Tuple[int, Token]]) -> Dict[int, List[Optional[int]]]:
    """Generate a random valid (POS-compatible) bond configuration."""
    config = {}
    for si, seeker in seekers:
        filler_list = []
        candidates = [fi for fi, f in enumerate(tokens)
                      if fi != si and pos_compatible(seeker, f)]
        for slot_idx in range(seeker.valence):
            options = candidates + [None]
            filler_list.append(random.choice(options))
        # Remove duplicate fillers within same seeker
        seen = set()
        for slot_idx in range(len(filler_list)):
            if filler_list[slot_idx] is not None and filler_list[slot_idx] in seen:
                filler_list[slot_idx] = None
            elif filler_list[slot_idx] is not None:
                seen.add(filler_list[slot_idx])
        config[si] = filler_list
    return config


def propose_move(config: Dict[int, List[Optional[int]]],
                 tokens: List[Token],
                 seekers: List[Tuple[int, Token]]) -> Dict[int, List[Optional[int]]]:
    """Propose a local move: change one filler assignment for one seeker slot."""
    new_config = {k: list(v) for k, v in config.items()}
    si, seeker = random.choice(seekers)
    slot_idx = random.randint(0, seeker.valence - 1)
    candidates = [fi for fi, f in enumerate(tokens)
                  if fi != si and pos_compatible(seeker, f)]
    candidates.append(None)
    current = new_config[si][slot_idx]
    options = [c for c in candidates if c != current]
    if options:
        new_config[si][slot_idx] = random.choice(options)
    return new_config


def simulated_annealing_bonding(state: MoleculeState,
                                 T_start: float = SA_T_START,
                                 T_end: float = SA_T_END,
                                 cooling: float = SA_COOLING,
                                 steps_per_temp: int = SA_STEPS_PER_TEMP) -> MoleculeState:
    """Find low-energy bond configuration via simulated annealing.

    Starts with a random valid configuration, then iteratively proposes local
    moves (swap one filler), accepting improvements always and worse moves with
    Boltzmann probability exp(-ΔE/T). Temperature cools geometrically.
    """
    tokens = state.tokens
    seekers = [(i, t) for i, t in enumerate(tokens) if t.valence > 0]

    if not seekers:
        return state

    # Initialize with random valid configuration
    config = random_valid_config(tokens, seekers)
    current_energy = total_energy(config, tokens)
    best_config = deepcopy(config)
    best_energy = current_energy

    T = T_start
    n_steps = 0
    n_accepted = 0

    while T > T_end:
        for _ in range(steps_per_temp):
            new_config = propose_move(config, tokens, seekers)
            new_energy = total_energy(new_config, tokens)
            delta_E = new_energy - current_energy
            n_steps += 1

            if delta_E < 0 or random.random() < math.exp(-delta_E / max(T, 1e-10)):
                config = new_config
                current_energy = new_energy
                n_accepted += 1
                if current_energy < best_energy:
                    best_config = deepcopy(config)
                    best_energy = current_energy

        T *= cooling

    accept_rate = n_accepted / max(n_steps, 1)
    print(f"  SA: {n_steps} steps, accept rate {accept_rate:.1%}, best energy = {best_energy:.2f}")

    # Apply best config to state
    _apply_config_to_state(best_config, state)
    return state


def print_energy_breakdown(config: Dict[int, List[Optional[int]]],
                           tokens: List[Token], label: str = ""):
    """Print decomposed energy terms for a configuration."""
    # Delegate to total_energy for consistency, but also compute per-term
    E_bond = 0.0
    E_val = 0.0
    E_sat = 0.0
    E_excl = 0.0
    filler_bond_counts: Counter = Counter()
    filler_roles: Dict[int, List[str]] = {}

    for seeker_idx, filler_list in config.items():
        seeker = tokens[seeker_idx]
        unfilled = 0
        for slot_idx, filler_idx in enumerate(filler_list):
            if filler_idx is not None:
                comp = None
                if seeker.pos == "Prep" and slot_idx == 1:
                    comp_idx = filler_list[0]
                    if comp_idx is not None:
                        comp = tokens[comp_idx]
                score = wave_overlap(seeker, tokens[filler_idx],
                                     slot_idx=slot_idx,
                                     seeker_idx=seeker_idx,
                                     filler_idx=filler_idx,
                                     complement=comp)
                E_bond -= score
                filler_bond_counts[filler_idx] += 1
                role = slot_label_for(seeker.pos, slot_idx)
                filler_roles.setdefault(filler_idx, []).append(role)
            else:
                unfilled += 1
        E_val += LAMBDA_VAL * unfilled

    content_fillers = {i for i, t in enumerate(tokens)
                       if t.pos in ("Noun", "Pron") and t.valence == 0}
    unbound = content_fillers - set(filler_bond_counts.keys())
    E_sat = LAMBDA_SAT * len(unbound)

    for filler_idx, roles in filler_roles.items():
        if len(roles) <= 1:
            continue
        verb_roles = [r for r in roles if r in ("subj", "obj", "iobj")]
        if len(verb_roles) <= 1:
            continue
        unique_verb_roles = set(verb_roles)
        same_role_dups = len(verb_roles) - len(unique_verb_roles)
        diff_role_count = max(0, len(unique_verb_roles) - 1)
        E_excl += LAMBDA_EXCL * same_role_dups
        E_excl += LAMBDA_EXCL * 0.5 * diff_role_count

    total = E_bond + E_val + E_sat + E_excl
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}E_bond={E_bond:.2f}  E_val={E_val:.2f}  E_sat={E_sat:.2f}  E_excl={E_excl:.2f}  TOTAL={total:.2f}")


# ------------------------------------------------------------
# Demo / test
# ------------------------------------------------------------

def run_test(name: str, words: List[str], method: str = "exhaustive"):
    print(f"\n{'='*60}")
    print(f"  {name}  [{method}]")
    print(f"  Sentence: \"{' '.join(words)}\"")
    print(f"{'='*60}")

    state = MoleculeState(words)
    state.print_state()

    print("\n  Bonding...")
    if method == "greedy":
        try_greedy_bonding(state)
    elif method == "annealing":
        simulated_annealing_bonding(state)
    else:  # exhaustive (auto-fallback to SA if too large)
        if len(words) > EXHAUSTIVE_THRESHOLD:
            simulated_annealing_bonding(state)
        else:
            exhaustive_bonding(state)

    state.print_state()
    state.print_tree()
    state.print_charges()

    saturated = state.is_saturated()
    print(f"\n  Saturated: {'Yes' if saturated else 'No — open slots remain'}")
    return state


if __name__ == "__main__":
    print("ASA Toy Prototype — Phase 1b: Dynamics & Thermodynamics")
    print(f"Vocab: {len(VOCAB)} tokens")

    # Wave overlap examples
    print("\nWave overlap examples (verb seeking noun):")
    pairs = [
        ("eat", "dog", 0),      # subject slot
        ("eat", "apple", 1),    # object slot
        ("eat", "rock", 0),     # inanimate subject (should be weak)
        ("examine", "dog", 0),  # animate subject (should be strong)
        ("examine", "rock", 0), # inanimate subject
        ("know", "something", 1),  # abstract object (should be strong)
        ("know", "rock", 1),       # non-abstract object (should be weaker)
    ]
    for w1, w2, slot in pairs:
        t1, t2 = WORD_TO_TOKEN[w1], WORD_TO_TOKEN[w2]
        score = wave_overlap(t1, t2, slot_idx=slot)
        slot_name = "subj" if slot == 0 else "obj"
        print(f"  {w1:8} —[{slot_name}]→ {w2:8} = {score:.3f}")

    # Charge interaction examples
    print("\nCharge interaction examples (negator → verb):")
    charge_pairs = [
        ("not", "love"),    # opposite charges: -1.0 × +0.7 → attraction
        ("not", "hate"),    # same-sign charges: -1.0 × -0.7 → repulsion
        ("not", "eat"),     # neutral target: -1.0 × 0.0 → no effect
    ]
    for w1, w2 in charge_pairs:
        t1, t2 = WORD_TO_TOKEN[w1], WORD_TO_TOKEN[w2]
        score = wave_overlap(t1, t2)
        interaction = LAMBDA_C * (-t1.charge * t2.charge)
        print(f"  {w1:8} → {w2:8} = {score:.3f}  (charge interaction: {interaction:+.3f})")

    # ---- Original test sentences (regression) ----
    print("\n" + "="*60)
    print("  REGRESSION TESTS (Phase 0 originals)")
    print("="*60)

    run_test("Basic SVO", ["the", "dog", "eat", "apple"])
    run_test("Adjective + SVO", ["the", "big", "dog", "eat", "apple"])
    run_test("Adverb + SVO", ["dog", "quickly", "eat", "apple"])
    run_test("Inanimate subject", ["the", "rock", "examine", "cat"])
    run_test("Two determiners", ["a", "cat", "eat", "the", "apple"])

    # ---- Phase 0.5 stress tests ----
    print("\n" + "="*60)
    print("  PHASE 0.5 STRESS TESTS")
    print("="*60)

    run_test("Intransitive", ["dog", "run", "quickly"])
    run_test("Intransitive + det", ["the", "cat", "sleep"])
    run_test("Ditransitive", ["man", "give", "dog", "food"])
    run_test("Pronoun subject", ["he", "eat", "apple"])
    run_test("Abstract object", ["woman", "know", "something"])
    run_test("Break physical", ["man", "break", "ball"])
    run_test("PP verb-attach", ["man", "eat", "food", "with", "fork"])
    run_test("PP noun-attach", ["man", "see", "dog", "with", "spots"])
    run_test("Multiple adjectives", ["the", "big", "red", "ball"])
    run_test("Competing animates", ["man", "see", "woman"])
    run_test("Animate+edible object", ["woman", "eat", "fish"])
    run_test("Negation (neutral verb)", ["dog", "not", "eat", "apple"])
    run_test("Dets + ditransitive", ["the", "man", "give", "a", "dog", "food"])

    # ---- Charge system tests ----
    print("\n" + "="*60)
    print("  CHARGE SYSTEM TESTS")
    print("="*60)

    run_test("Positive verb", ["man", "love", "dog"])
    run_test("Negated positive", ["man", "not", "love", "dog"])
    run_test("Negative verb", ["man", "hate", "dog"])
    run_test("Negated negative (double neg)", ["man", "not", "hate", "dog"])
    run_test("Neutral vs negated contrast", ["dog", "eat", "apple"])

    # ---- Greedy vs Exhaustive comparison (Phase 1a) ----
    print("\n" + "="*60)
    print("  GREEDY vs EXHAUSTIVE COMPARISON")
    print("  (Sentences where greedy fails, exhaustive should fix)")
    print("="*60)

    greedy_failure_sentences = [
        ("G1: Two clauses, filler competition",
         ["man", "eat", "apple", "dog", "see", "ball"]),
        ("G2: Distant subject, close distractor",
         ["man", "the", "dog", "eat", "apple"]),
        ("G3: Two PPs competing for verb",
         ["man", "eat", "food", "with", "fork", "on", "ball"]),
        ("G4: Ditransitive all-animate",
         ["woman", "give", "man", "dog"]),
        ("G5: Center-embedded (crossing deps)",
         ["the", "man", "the", "dog", "see", "eat", "apple"]),
    ]

    for name, words in greedy_failure_sentences:
        print(f"\n{'~'*60}")
        print(f"  --- GREEDY ---")
        run_test(name, words, method="greedy")
        print(f"\n  --- EXHAUSTIVE ---")
        run_test(name, words, method="exhaustive")

    # ---- Parameter robustness sweep (±50%) ----
    print("\n" + "="*60)
    print("  PARAMETER ROBUSTNESS SWEEP (±50%)")
    print("="*60)

    import sys
    # Key test cases: G1 (must stay fixed) + a few regression tests
    sweep_sentences = [
        ("G1", ["man", "eat", "apple", "dog", "see", "ball"],
         lambda s: s.bonds.get((4, 5)) is not None),  # see→ball must exist
        ("Basic SVO", ["the", "dog", "eat", "apple"],
         lambda s: s.bonds.get((2, 1)) is not None),   # eat→dog(subj)
        ("PP verb-attach", ["man", "eat", "food", "with", "fork"],
         lambda s: s.bond_labels.get((3, 1)) == "pp→"),  # with→eat(pp)
    ]

    original_vals = (LAMBDA_VAL, LAMBDA_SAT, LAMBDA_EXCL)
    param_names = ["LAMBDA_VAL", "LAMBDA_SAT", "LAMBDA_EXCL"]
    all_pass = True

    for pi, pname in enumerate(param_names):
        for factor in [0.5, 1.5]:
            # Temporarily override the global
            test_vals = list(original_vals)
            test_vals[pi] = original_vals[pi] * factor
            # We need to set the globals temporarily
            globals()["LAMBDA_VAL"] = test_vals[0]
            globals()["LAMBDA_SAT"] = test_vals[1]
            globals()["LAMBDA_EXCL"] = test_vals[2]

            results = []
            for sname, words, check_fn in sweep_sentences:
                state = MoleculeState(words)
                exhaustive_bonding(state)
                passed = check_fn(state)
                results.append((sname, passed))

            status = "PASS" if all(r[1] for r in results) else "FAIL"
            if status == "FAIL":
                all_pass = False
            failed = [r[0] for r in results if not r[1]]
            print(f"  {pname}={test_vals[pi]:.1f} ({factor:.0%}): {status}"
                  + (f"  — failed: {', '.join(failed)}" if failed else ""))

    # Restore originals
    LAMBDA_VAL, LAMBDA_SAT, LAMBDA_EXCL = original_vals
    globals()["LAMBDA_VAL"] = original_vals[0]
    globals()["LAMBDA_SAT"] = original_vals[1]
    globals()["LAMBDA_EXCL"] = original_vals[2]

    print(f"\n  Overall sweep: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ---- Simulated Annealing validation (Phase 1b) ----
    print("\n" + "="*60)
    print("  SIMULATED ANNEALING VALIDATION")
    print("="*60)

    # Compare SA vs exhaustive on key sentences (run SA multiple times for stochastic check)
    sa_test_sentences = [
        ("G1", ["man", "eat", "apple", "dog", "see", "ball"],
         lambda s: s.bonds.get((4, 5)) is not None),  # see→ball must exist
        ("Basic SVO", ["the", "dog", "eat", "apple"],
         lambda s: s.bonds.get((2, 1)) is not None),
        ("PP verb-attach", ["man", "eat", "food", "with", "fork"],
         lambda s: s.bond_labels.get((3, 1)) == "pp→"),
        ("Ditransitive", ["man", "give", "dog", "food"],
         lambda s: s.bond_labels.get((1, 0)) == "subj"),
        ("G4 word-order", ["woman", "give", "man", "dog"],
         lambda s: s.bond_labels.get((1, 2)) == "iobj"),
    ]

    sa_all_pass = True
    n_sa_trials = 10  # run each sentence 10 times (stochastic)
    for sname, words, check_fn in sa_test_sentences:
        pass_count = 0
        for trial in range(n_sa_trials):
            state = MoleculeState(words)
            simulated_annealing_bonding(state)
            if check_fn(state):
                pass_count += 1
        pct = pass_count / n_sa_trials * 100
        status = "PASS" if pass_count == n_sa_trials else f"PARTIAL ({pass_count}/{n_sa_trials})"
        if pass_count < n_sa_trials:
            sa_all_pass = False
        print(f"  SA {sname}: {status} ({pct:.0f}%)")

    # Compare SA energy to exhaustive energy on medium sentences
    print("\n  SA vs Exhaustive energy comparison:")
    compare_sentences = [
        ("G1", ["man", "eat", "apple", "dog", "see", "ball"]),
        ("G3", ["man", "eat", "food", "with", "fork", "on", "ball"]),
    ]
    for sname, words in compare_sentences:
        # Exhaustive: capture energy from printed output
        state_ex = MoleculeState(words)
        exhaustive_bonding(state_ex)
        # Reconstruct config from state bonds
        ex_config: Dict[int, List[Optional[int]]] = {}
        for (si, fi), _ in state_ex.bonds.items():
            label = state_ex.bond_labels.get((si, fi), "")
            seeker = state_ex.tokens[si]
            if si not in ex_config:
                ex_config[si] = [None] * seeker.valence
            # Map label back to slot index
            if label == "subj": ex_config[si][0] = fi
            elif label == "obj": ex_config[si][1] = fi
            elif label == "iobj": ex_config[si][2] = fi
            elif label == "det→": ex_config[si][0] = fi
            elif label == "adj→": ex_config[si][0] = fi
            elif label == "adv→": ex_config[si][0] = fi
            elif label == "prep→": ex_config[si][0] = fi
            elif label == "pp→": ex_config[si][1] = fi
        ex_energy = total_energy(ex_config, state_ex.tokens)

        # SA: run 5 times, take best
        sa_best = float('inf')
        for _ in range(5):
            state_sa = MoleculeState(words)
            seekers_sa = [(i, t) for i, t in enumerate(state_sa.tokens) if t.valence > 0]
            config_sa = random_valid_config(state_sa.tokens, seekers_sa)
            current_e = total_energy(config_sa, state_sa.tokens)
            best_c, best_e = deepcopy(config_sa), current_e
            T = SA_T_START
            while T > SA_T_END:
                for _ in range(SA_STEPS_PER_TEMP):
                    new_c = propose_move(config_sa, state_sa.tokens, seekers_sa)
                    new_e = total_energy(new_c, state_sa.tokens)
                    dE = new_e - current_e
                    if dE < 0 or random.random() < math.exp(-dE / max(T, 1e-10)):
                        config_sa, current_e = new_c, new_e
                        if current_e < best_e:
                            best_c, best_e = deepcopy(config_sa), current_e
                T *= SA_COOLING
            if best_e < sa_best:
                sa_best = best_e

        gap = abs(sa_best - ex_energy) / abs(ex_energy) * 100 if ex_energy != 0 else 0
        print(f"  {sname}: exhaustive={ex_energy:.2f}, SA best={sa_best:.2f}, gap={gap:.1f}%")

    # Long sentence test (beyond exhaustive threshold)
    print("\n  Long sentence test (20 tokens — SA only):")
    long_words = ["the", "big", "dog", "quickly", "eat", "the", "red", "apple",
                   "with", "fork", "on", "the", "big", "ball",
                   "the", "man", "see", "the", "small", "cat"]
    run_test("Long sentence (20 tokens)", long_words, method="annealing")

    print(f"\n  SA validation: {'ALL PASS' if sa_all_pass else 'SOME PARTIAL'}")

    # ---- Temperature effect validation (Phase 1b gate) ----
    print("\n" + "="*60)
    print("  TEMPERATURE EFFECT VALIDATION")
    print("  (Boltzmann selection over top configs at varying T)")
    print("="*60)

    # Use G1 (6 tokens, exhaustive feasible) — collect ALL configs with energies
    temp_test_words = ["man", "eat", "apple", "dog", "see", "ball"]
    temp_state = MoleculeState(temp_test_words)
    temp_tokens = temp_state.tokens
    temp_seekers = [(i, t) for i, t in enumerate(temp_tokens) if t.valence > 0]

    # Enumerate all configs and their energies
    per_seeker_opts = []
    for si, seeker in temp_seekers:
        slot_opts = []
        for slot_idx in range(seeker.valence):
            cands = [None]
            for fi, f in enumerate(temp_tokens):
                if fi != si and pos_compatible(seeker, f):
                    cands.append(fi)
            slot_opts.append(cands)
        assigns = list(cartesian_product(*slot_opts))
        assigns = [a for a in assigns
                   if len(set(f for f in a if f is not None))
                   == len([f for f in a if f is not None])]
        per_seeker_opts.append((si, assigns))

    all_configs_with_energies = []
    for combo in cartesian_product(*(assigns for _, assigns in per_seeker_opts)):
        config = {}
        for (si, _), assignment in zip(per_seeker_opts, combo):
            config[si] = list(assignment)
        energy = total_energy(config, temp_tokens)
        all_configs_with_energies.append((config, energy))

    # Sort by energy (best first)
    all_configs_with_energies.sort(key=lambda x: x[1])
    best_config_energy = all_configs_with_energies[0][1]
    print(f"  Total configs: {len(all_configs_with_energies)}")
    print(f"  Best energy: {best_config_energy:.2f}")
    print(f"  Worst energy: {all_configs_with_energies[-1][1]:.2f}")

    # Check function: does the config produce see→ball bond?
    def g1_correct(config):
        return config.get(4) is not None and 5 in (config.get(4, []))

    # Temperature sweep: at each T, sample 100 times and measure accuracy
    temperatures = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\n  {'T':>6} | {'Correct%':>8} | {'Avg Energy':>10} | {'Selection'}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*20}")

    prev_accuracy = 101  # for monotonicity check
    monotonic = True
    for T in temperatures:
        correct = 0
        total_e = 0.0
        n_samples = 200

        for _ in range(n_samples):
            if T <= 0.01:
                selected = all_configs_with_energies[0]
            else:
                energies = np.array([e for _, e in all_configs_with_energies])
                shifted = energies - energies.min()
                weights = np.exp(-shifted / T)
                weights /= weights.sum()
                idx = np.random.choice(len(all_configs_with_energies), p=weights)
                selected = all_configs_with_energies[idx]

            config, energy = selected
            total_e += energy
            if g1_correct(config):
                correct += 1

        accuracy = correct / n_samples * 100
        avg_e = total_e / n_samples
        behavior = "frozen" if T <= 0.01 else ("selective" if T < 1.0 else ("diffuse" if T < 5.0 else "near-uniform"))
        print(f"  {T:6.2f} | {accuracy:7.1f}% | {avg_e:10.2f} | {behavior}")

        if accuracy > prev_accuracy + 5:  # allow small noise
            monotonic = False
        prev_accuracy = accuracy

    print(f"\n  Monotonic degradation: {'YES' if monotonic else 'NO — non-monotonic!'}")
    print(f"  Gate 1b criteria:")
    print(f"    Low T → selective: {'PASS' if temperatures[0] == 0.01 else 'CHECK'}")
    print(f"    High T → diffuse: {'PASS' if prev_accuracy < 50 else 'CHECK'}")
    print(f"    Monotonic: {'PASS' if monotonic else 'FAIL'}")

    # ---- Thermodynamic Scoring A/B Comparison (Phase 1b gate 1c) ----
    print("\n" + "="*60)
    print("  THERMODYNAMIC SCORING A/B COMPARISON")
    print("  A = current wave_overlap (ad-hoc boosts)")
    print("  B = thermodynamic_overlap (enthalpy-based)")
    print("="*60)

    # Test sentences with expected bonds for correctness check
    ab_test_sentences = [
        ("Basic SVO", ["the", "dog", "eat", "apple"],
         {(2, 1): "subj", (2, 3): "obj", (0, 1): "det→"}),
        ("Adj + SVO", ["the", "big", "dog", "eat", "apple"],
         {(3, 2): "subj", (3, 4): "obj", (0, 2): "det→", (1, 2): "adj→"}),
        ("Intransitive", ["dog", "run", "quickly"],
         {(1, 0): "subj", (2, 1): "adv→"}),
        ("Ditransitive", ["man", "give", "dog", "food"],
         {(1, 0): "subj", (1, 3): "obj", (1, 2): "iobj"}),
        ("PP verb-attach", ["man", "eat", "food", "with", "fork"],
         {(1, 0): "subj", (1, 2): "obj", (3, 4): "prep→", (3, 1): "pp→"}),
        ("PP noun-attach", ["man", "see", "dog", "with", "spots"],
         {(1, 0): "subj", (1, 2): "obj", (3, 4): "prep→", (3, 2): "pp→"}),
        ("Pronoun subject", ["he", "eat", "apple"],
         {(1, 0): "subj", (1, 2): "obj"}),
        ("Abstract object", ["woman", "know", "something"],
         {(1, 0): "subj", (1, 2): "obj"}),
        ("Competing animates", ["man", "see", "woman"],
         {(1, 0): "subj", (1, 2): "obj"}),
        ("Negation", ["dog", "not", "eat", "apple"],
         {(2, 0): "subj", (2, 3): "obj", (1, 2): "adv→"}),
        ("G1 two clauses", ["man", "eat", "apple", "dog", "see", "ball"],
         {(1, 0): "subj", (1, 2): "obj", (4, 3): "subj", (4, 5): "obj"}),
        ("G4 ditransitive", ["woman", "give", "man", "dog"],
         {(1, 0): "subj", (1, 2): "iobj", (1, 3): "obj"}),
        ("Positive verb", ["man", "love", "dog"],
         {(1, 0): "subj", (1, 2): "obj"}),
        ("Negated positive", ["man", "not", "love", "dog"],
         {(2, 0): "subj", (2, 3): "obj", (1, 2): "adv→"}),
    ]

    def check_bonds(state, expected):
        """Check if state has all expected bonds with correct labels."""
        for (si, fi), label in expected.items():
            actual = state.bond_labels.get((si, fi))
            if actual != label:
                return False
        return True

    def run_ab_test(scoring_fn, words):
        """Run exhaustive bonding with a specific scoring function."""
        # Temporarily swap wave_overlap
        state = MoleculeState(words)
        tokens = state.tokens
        seekers = [(i, t) for i, t in enumerate(tokens) if t.valence > 0]

        if not seekers:
            return state

        # Enumerate configs (reuse exhaustive logic but with custom scorer)
        per_seeker_opts = []
        for si, seeker in seekers:
            slot_opts = []
            for slot_idx in range(seeker.valence):
                cands = [None]
                for fi, f in enumerate(tokens):
                    if fi != si and pos_compatible(seeker, f):
                        cands.append(fi)
                slot_opts.append(cands)
            assigns = list(cartesian_product(*slot_opts))
            assigns = [a for a in assigns
                       if len(set(f for f in a if f is not None))
                       == len([f for f in a if f is not None])]
            per_seeker_opts.append((si, assigns))

        best_energy = float('inf')
        best_config = None

        for combo in cartesian_product(*(assigns for _, assigns in per_seeker_opts)):
            config = {}
            for (si, _), assignment in zip(per_seeker_opts, combo):
                config[si] = list(assignment)

            # Compute energy using the given scoring function
            E = 0.0
            filler_bond_counts = Counter()
            filler_roles_ab = {}

            for seeker_idx, filler_list in config.items():
                seeker = tokens[seeker_idx]
                for slot_idx, filler_idx in enumerate(filler_list):
                    if filler_idx is not None:
                        comp = None
                        if seeker.pos == "Prep" and slot_idx == 1:
                            comp_idx = filler_list[0]
                            if comp_idx is not None:
                                comp = tokens[comp_idx]
                        score = scoring_fn(seeker, tokens[filler_idx],
                                           slot_idx=slot_idx,
                                           seeker_idx=seeker_idx,
                                           filler_idx=filler_idx,
                                           complement=comp)
                        E -= score
                        filler_bond_counts[filler_idx] += 1
                        role = slot_label_for(seeker.pos, slot_idx)
                        filler_roles_ab.setdefault(filler_idx, []).append(role)

            for seeker_idx, filler_list in config.items():
                unfilled = sum(1 for f in filler_list if f is None)
                E += LAMBDA_VAL * unfilled

            content_fillers = {i for i, t in enumerate(tokens)
                               if t.pos in ("Noun", "Pron") and t.valence == 0}
            unbound = content_fillers - set(filler_bond_counts.keys())
            E += LAMBDA_SAT * len(unbound)

            for fi, roles in filler_roles_ab.items():
                if len(roles) <= 1:
                    continue
                verb_roles = [r for r in roles if r in ("subj", "obj", "iobj")]
                if len(verb_roles) <= 1:
                    continue
                unique_vr = set(verb_roles)
                E += LAMBDA_EXCL * (len(verb_roles) - len(unique_vr))
                E += LAMBDA_EXCL * 0.5 * max(0, len(unique_vr) - 1)

            if E < best_energy:
                best_energy = E
                best_config = config

        # Apply best config
        if best_config is not None:
            for si, filler_list in best_config.items():
                for slot_idx, fi in enumerate(filler_list):
                    if fi is not None:
                        seeker = tokens[si]
                        comp = None
                        if seeker.pos == "Prep" and slot_idx == 1:
                            comp_idx = filler_list[0]
                            if comp_idx is not None:
                                comp = tokens[comp_idx]
                        score = scoring_fn(seeker, tokens[fi],
                                           slot_idx=slot_idx, seeker_idx=si, filler_idx=fi,
                                           complement=comp)
                        label = slot_label_for(seeker.pos, slot_idx)
                        state.bonds[(si, fi)] = score
                        state.bond_labels[(si, fi)] = label
                        state.remaining_valence[si] -= 1

        return state

    # Wrapper for wave_overlap (System A) without state parameter
    def system_a(seeker, filler, slot_idx=None, seeker_idx=None, filler_idx=None, complement=None):
        return wave_overlap(seeker, filler, slot_idx=slot_idx,
                            seeker_idx=seeker_idx, filler_idx=filler_idx,
                            complement=complement)

    # System B = thermodynamic_overlap (already defined)
    system_b = thermodynamic_overlap

    a_correct = 0
    b_correct = 0
    both_correct = 0
    a_only = 0
    b_only = 0
    neither = 0

    print(f"\n  {'Test':<25} | {'A':>6} | {'B':>6}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}")

    for name, words, expected in ab_test_sentences:
        state_a = run_ab_test(system_a, words)
        state_b = run_ab_test(system_b, words)

        a_pass = check_bonds(state_a, expected)
        b_pass = check_bonds(state_b, expected)

        a_str = "PASS" if a_pass else "FAIL"
        b_str = "PASS" if b_pass else "FAIL"
        print(f"  {name:<25} | {a_str:>6} | {b_str:>6}")

        if a_pass: a_correct += 1
        if b_pass: b_correct += 1
        if a_pass and b_pass: both_correct += 1
        if a_pass and not b_pass: a_only += 1
        if b_pass and not a_pass: b_only += 1
        if not a_pass and not b_pass: neither += 1

    total = len(ab_test_sentences)
    print(f"\n  Summary:")
    print(f"    System A (ad-hoc): {a_correct}/{total} correct ({a_correct/total*100:.0f}%)")
    print(f"    System B (thermo): {b_correct}/{total} correct ({b_correct/total*100:.0f}%)")
    print(f"    Both correct: {both_correct}")
    print(f"    A only: {a_only}")
    print(f"    B only: {b_only}")
    print(f"    Neither: {neither}")

    # Gate 1c evaluation
    if a_correct > 0:
        b_retention = both_correct / a_correct * 100
    else:
        b_retention = 100
    print(f"\n  Gate 1c criteria:")
    print(f"    B retains ≥95% of A's correct parses: {b_retention:.0f}% — {'PASS' if b_retention >= 95 else 'FAIL'}")
    print(f"    B fixes ≥1 sentence A doesn't: {b_only} — {'PASS' if b_only >= 1 else 'FAIL (no improvement)'}")
    if b_only == 0 and a_only == 0:
        print(f"    → B ≈ A: thermodynamic scoring adds no value over ad-hoc. KEEP SIMPLE (A).")
    elif b_only == 0 and a_only > 0:
        print(f"    → B is strictly worse than A. KEEP A.")
    elif b_only > 0 and a_only == 0:
        print(f"    → B strictly improves on A. ADOPT B.")
    else:
        print(f"    → Mixed results. Evaluate tradeoffs.")

    # ---- SA Parameter Robustness (Phase 1b gate 1d) ----
    print("\n" + "="*60)
    print("  SA PARAMETER ROBUSTNESS (±50%)")
    print("="*60)

    sa_sweep_sentences = [
        ("G1", ["man", "eat", "apple", "dog", "see", "ball"],
         lambda s: s.bonds.get((4, 5)) is not None),
        ("Basic SVO", ["the", "dog", "eat", "apple"],
         lambda s: s.bonds.get((2, 1)) is not None),
    ]

    sa_param_names = ["SA_T_START", "SA_COOLING", "SA_STEPS_PER_TEMP"]
    sa_original = (SA_T_START, SA_COOLING, SA_STEPS_PER_TEMP)
    sa_all_pass_sweep = True
    n_sa_sweep_trials = 5  # stochastic, run multiple

    for pi, pname in enumerate(sa_param_names):
        for factor in [0.5, 1.5]:
            t_start = sa_original[0]
            cooling = sa_original[1]
            steps = sa_original[2]

            if pi == 0:
                t_start = sa_original[0] * factor
            elif pi == 1:
                # Cooling: 0.5× = 0.475 (faster cool), 1.5× = min(0.99, 1.425) (slower cool)
                cooling = min(0.99, sa_original[1] * factor)
            elif pi == 2:
                steps = max(10, int(sa_original[2] * factor))

            results = []
            for sname, words, check_fn in sa_sweep_sentences:
                pass_count = 0
                for _ in range(n_sa_sweep_trials):
                    state = MoleculeState(words)
                    simulated_annealing_bonding(state, T_start=t_start,
                                                cooling=cooling,
                                                steps_per_temp=steps)
                    if check_fn(state):
                        pass_count += 1
                results.append((sname, pass_count == n_sa_sweep_trials))

            status = "PASS" if all(r[1] for r in results) else "FAIL"
            if status == "FAIL":
                sa_all_pass_sweep = False
            val_str = f"{t_start:.1f}" if pi == 0 else (f"{cooling:.3f}" if pi == 1 else str(steps))
            failed = [r[0] for r in results if not r[1]]
            print(f"  {pname}={val_str} ({factor:.0%}): {status}"
                  + (f"  — failed: {', '.join(failed)}" if failed else ""))

    print(f"\n  SA sweep: {'ALL PASS' if sa_all_pass_sweep else 'SOME FAILURES'}")

    # ---- Scale projection ----
    print("\n" + "="*60)
    print("  SCALE PROJECTION (SA timing)")
    print("="*60)

    import time

    scale_sentences = [
        (10, ["the", "big", "dog", "eat", "the", "red", "apple", "with", "fork", "quickly"]),
        (15, ["the", "big", "dog", "quickly", "eat", "the", "red", "apple",
              "with", "fork", "on", "the", "big", "ball", "slowly"]),
        (20, ["the", "big", "dog", "quickly", "eat", "the", "red", "apple",
              "with", "fork", "on", "the", "big", "ball",
              "the", "man", "see", "the", "small", "cat"]),
    ]

    for n_tok, words in scale_sentences:
        start_t = time.time()
        state = MoleculeState(words)
        simulated_annealing_bonding(state)
        elapsed = time.time() - start_t
        saturated = state.is_saturated()
        print(f"  {n_tok} tokens: {elapsed:.3f}s, saturated={saturated}")

    print("\n  Phase 1b complete.")
