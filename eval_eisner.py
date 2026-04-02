"""
Evaluate periodic table features with Eisner optimal parsing instead of SA.
Replaces ONLY the SA step — keeps ALL post-processing (compounds, V-chains, etc.)
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from eisner import eisner_parse
import eval_treebank as et
from eval_treebank import (
    make_token, extract_gold_bonds, _patched_wave_overlap, PTB_TO_ASA,
)
import asa_toy
from nltk.corpus import dependency_treebank
from collections import Counter


def eisner_sa_replacement(tokens):
    """Replace SA with Eisner — same scores, optimal decoding."""
    n = len(tokens)
    size = n + 1  # +1 for ROOT

    scores = np.zeros((size, size))

    # ROOT→Verb bonds
    for j in range(n):
        if tokens[j].pos in ("Verb", "Aux"):
            scores[0][j+1] = 5.0
        else:
            scores[0][j+1] = 0.1

    # All pairwise scores using our scoring function
    for h in range(n):
        for d in range(n):
            if h == d:
                continue
            seeker = tokens[h]
            filler = tokens[d]
            if seeker.valence > 0 and asa_toy.pos_compatible(seeker, filler):
                score = _patched_wave_overlap(
                    seeker, filler, slot_idx=0,
                    seeker_idx=h, filler_idx=d)
                scores[h+1][d+1] = max(scores[h+1][d+1], score)

            seeker2 = tokens[d]
            filler2 = tokens[h]
            if seeker2.valence > 0 and asa_toy.pos_compatible(seeker2, filler2):
                score2 = _patched_wave_overlap(
                    seeker2, filler2, slot_idx=0,
                    seeker_idx=d, filler_idx=h)
                scores[d+1][h+1] = max(scores[d+1][h+1], score2)

    heads = eisner_parse(scores)

    # Convert to MoleculeState-like bond dict
    bonds = {}
    for d in range(1, size):
        h = heads[d]
        if h == 0:
            continue
        hi, di = h - 1, d - 1
        bonds[(hi, di)] = scores[h][d]

    return bonds


# Monkey-patch: replace simulated_annealing_bonding with Eisner
_orig_sa = asa_toy.simulated_annealing_bonding

def _eisner_wrapper(state, **kwargs):
    """Drop-in replacement for SA that uses Eisner."""
    tokens = state.tokens
    bonds = eisner_sa_replacement(tokens)

    # Apply bonds to state
    for (si, fi), score in bonds.items():
        state.bonds[(si, fi)] = score
        state.remaining_valence[si] = max(0, state.remaining_valence[si] - 1)

    return state

# Patch it in
asa_toy.simulated_annealing_bonding = _eisner_wrapper

# Now run the SAME eval_treebank evaluation — it will use Eisner instead of SA
# but keep ALL post-processing (compounds, V-chains, NP chunking, etc.)
if __name__ == "__main__":
    et._current_tokens = None
    et._ptb_tags = []
    results, missed, spurious = et.evaluate_treebank(max_sents=500)
