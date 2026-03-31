#!/usr/bin/env python3
"""
Discovery of Missing Elements — Round 58+

The parser is at F1=0.669, stuck because the 28-dim vector captures
grammatical structure (POS-level) but not lexical selectivity (word-level).

Top errors:
1. Prep attachment (23 spurious + 20 missed Prep→Noun)
2. Verb argument structure (45 missed Verb→Other, 18 spurious Noun→Verb)

Strategy: Find MEASURABLE word-level properties from corpus statistics
that differentiate within POS categories. Properties not rules.

New candidate elements:
1. clause_complement_pref — does this verb take clause complements?
2. semantic_weight — light verb vs heavy verb (corpus frequency of V+clause)
3. prep_selectivity — how specific is this prep's complement? (entropy)
4. noun_governability — how easily does this noun become an argument?
5. embedding_depth — typical syntactic depth of this word
6. argument_frame_complexity — how many different argument frames?
"""

import os, sys
import numpy as np
from collections import Counter, defaultdict

# Get verb subcategorization frames from PTB treebank
import nltk
from nltk.corpus import dependency_treebank, treebank, brown

sys.path.insert(0, '/home/asa/asa')


def analyze_verb_frames():
    """Analyze which verbs take clause complements vs simple NP arguments.

    Measurable property: clause_complement_pref = P(verb takes SBAR/S complement)
    """
    print("=" * 60)
    print("ELEMENT 1: CLAUSE COMPLEMENT PREFERENCE")
    print("=" * 60)

    # Use phrase-structure treebank to find verb subcategorization
    verb_total = Counter()
    verb_sbar = Counter()  # takes SBAR complement
    verb_np = Counter()    # takes NP complement
    verb_pp = Counter()    # takes PP complement
    verb_intrans = Counter()  # no complement

    for fileid in treebank.fileids():
        for tree in treebank.parsed_sents(fileid):
            for subtree in tree.subtrees():
                if subtree.label().startswith('VP'):
                    # Find the head verb
                    verb = None
                    for child in subtree:
                        if hasattr(child, 'label') and child.label().startswith('VB'):
                            verb = child.leaves()[0].lower()
                            break
                    if verb is None:
                        continue

                    verb_total[verb] += 1

                    # Check what follows the verb
                    child_labels = [c.label() if hasattr(c, 'label') else str(c)
                                   for c in subtree]

                    if any(l.startswith('SBAR') or l == 'S' for l in child_labels):
                        verb_sbar[verb] += 1
                    if any(l == 'NP' for l in child_labels):
                        verb_np[verb] += 1
                    if any(l == 'PP' for l in child_labels):
                        verb_pp[verb] += 1
                    if not any(l in ('NP', 'PP', 'SBAR', 'S', 'VP') for l in child_labels):
                        verb_intrans[verb] += 1

    # Compute clause complement preference
    clause_pref = {}
    for v, total in verb_total.most_common(200):
        if total >= 3:
            clause_pref[v] = verb_sbar[v] / total

    print(f"\nVerbs with HIGH clause complement preference:")
    for v, p in sorted(clause_pref.items(), key=lambda x: -x[1])[:20]:
        print(f"  {v:>15s}: {p:.2f} ({verb_sbar[v]}/{verb_total[v]})")

    print(f"\nVerbs with LOW clause complement preference (NP-preferring):")
    for v, p in sorted(clause_pref.items(), key=lambda x: x[1])[:20]:
        if verb_total[v] >= 5:
            print(f"  {v:>15s}: {p:.2f} ({verb_sbar[v]}/{verb_total[v]}) "
                  f"NP={verb_np[v]}/{verb_total[v]}")

    return clause_pref, verb_total, verb_sbar, verb_np, verb_pp


def analyze_semantic_weight():
    """Analyze semantic weight: light verbs vs heavy verbs.

    Light verbs (make, take, have, give, do) are syntactically versatile
    but semantically empty. They should have different bonding behavior.

    Measurable: average number of dependents per verb token.
    """
    print("\n" + "=" * 60)
    print("ELEMENT 2: SEMANTIC WEIGHT (light vs heavy verbs)")
    print("=" * 60)

    verb_dep_count = Counter()  # verb → total deps
    verb_token_count = Counter()  # verb → total tokens

    for sent in dependency_treebank.parsed_sents():
        nodes = list(sent.nodes.values())

        # Count dependents per verb
        dep_count = Counter()
        for node in nodes:
            if node.get('head') is not None:
                head_idx = node['head']
                if head_idx in sent.nodes:
                    head = sent.nodes[head_idx]
                    word = (head.get('word') or '').lower()
                    tag = head.get('tag', '')
                    if tag.startswith('VB'):
                        dep_count[word] += 1

        for word, count in dep_count.items():
            verb_dep_count[word] += count
            verb_token_count[word] += 1

    # Average deps per token
    avg_deps = {}
    for v, total_deps in verb_dep_count.items():
        tokens = verb_token_count[v]
        if tokens >= 3:
            avg_deps[v] = total_deps / tokens

    print(f"\nVerbs with MOST dependents (heavy/complex frames):")
    for v, d in sorted(avg_deps.items(), key=lambda x: -x[1])[:20]:
        print(f"  {v:>15s}: {d:.1f} deps/token ({verb_token_count[v]} tokens)")

    print(f"\nVerbs with FEWEST dependents (light/simple):")
    for v, d in sorted(avg_deps.items(), key=lambda x: x[1])[:20]:
        print(f"  {v:>15s}: {d:.1f} deps/token ({verb_token_count[v]} tokens)")

    return avg_deps


def analyze_prep_selectivity():
    """Analyze how selective each preposition is about its complement.

    "of" is very selective (almost always nouns)
    "that" is very selective (almost always clauses)
    "for" is flexible (nouns, verbs, clauses)

    Measurable: entropy of complement POS distribution.
    """
    print("\n" + "=" * 60)
    print("ELEMENT 3: PREPOSITION SELECTIVITY (complement entropy)")
    print("=" * 60)

    prep_comp_pos = defaultdict(Counter)  # prep → Counter of complement POS

    for sent in dependency_treebank.parsed_sents():
        nodes = list(sent.nodes.values())
        for node in nodes:
            tag = node.get('tag', '')
            if tag == 'IN' or tag == 'TO':
                word = node.get('word', '').lower()
                # Find dependents
                addr = node.get('address')
                for other in nodes:
                    if other.get('head') == addr:
                        dep_tag = other.get('tag', '')
                        prep_comp_pos[word][dep_tag] += 1

    # Compute entropy
    def entropy(counter):
        total = sum(counter.values())
        if total == 0:
            return 0.0
        probs = [c / total for c in counter.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    prep_entropy = {}
    for prep, pos_dist in prep_comp_pos.items():
        total = sum(pos_dist.values())
        if total >= 5:
            prep_entropy[prep] = entropy(pos_dist)

    print(f"\nPreps with LOW entropy (highly selective):")
    for p, e in sorted(prep_entropy.items(), key=lambda x: x[1])[:15]:
        dist = prep_comp_pos[p]
        total = sum(dist.values())
        top = dist.most_common(3)
        print(f"  {p:>10s}: H={e:.2f} ({total} comps) — {', '.join(f'{t}:{c}' for t,c in top)}")

    print(f"\nPreps with HIGH entropy (flexible):")
    for p, e in sorted(prep_entropy.items(), key=lambda x: -x[1])[:15]:
        dist = prep_comp_pos[p]
        total = sum(dist.values())
        top = dist.most_common(3)
        print(f"  {p:>10s}: H={e:.2f} ({total} comps) — {', '.join(f'{t}:{c}' for t,c in top)}")

    return prep_entropy


def analyze_noun_governability():
    """Analyze how easily each noun type becomes a verb argument.

    Some nouns are almost always arguments (proper names, pronouns)
    Others are often modifiers (adjectives nominalized, mass nouns)

    Measurable: P(noun is dependent of a verb) from treebank.
    """
    print("\n" + "=" * 60)
    print("ELEMENT 4: NOUN GOVERNABILITY")
    print("=" * 60)

    noun_as_verb_arg = Counter()
    noun_total = Counter()

    for sent in dependency_treebank.parsed_sents():
        nodes = list(sent.nodes.values())
        for node in nodes:
            tag = node.get('tag', '')
            if tag.startswith('NN') or tag.startswith('PRP'):
                word = node.get('word', '').lower()
                noun_total[word] += 1

                # Check if head is a verb
                head_idx = node.get('head')
                if head_idx and head_idx in sent.nodes:
                    head = sent.nodes[head_idx]
                    head_tag = head.get('tag', '')
                    if head_tag.startswith('VB'):
                        noun_as_verb_arg[word] += 1

    gov = {}
    for n, total in noun_total.items():
        if total >= 3:
            gov[n] = noun_as_verb_arg[n] / total

    print(f"\nNouns with HIGH governability (almost always verb arguments):")
    for n, g in sorted(gov.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:>15s}: {g:.2f} ({noun_as_verb_arg[n]}/{noun_total[n]})")

    print(f"\nNouns with LOW governability (rarely verb arguments):")
    for n, g in sorted(gov.items(), key=lambda x: x[1])[:20]:
        if noun_total[n] >= 3:
            print(f"  {n:>15s}: {g:.2f} ({noun_as_verb_arg[n]}/{noun_total[n]})")

    return gov


def analyze_argument_frame_complexity():
    """Analyze how many different argument frames each verb uses.

    "said" almost always: said + SBAR
    "put" requires: put + NP + PP
    "give" varies: give NP NP, give NP PP, give NP

    Measurable: number of distinct frame patterns per verb.
    """
    print("\n" + "=" * 60)
    print("ELEMENT 5: ARGUMENT FRAME COMPLEXITY")
    print("=" * 60)

    verb_frames = defaultdict(Counter)

    for fileid in treebank.fileids():
        for tree in treebank.parsed_sents(fileid):
            for subtree in tree.subtrees():
                if subtree.label().startswith('VP'):
                    verb = None
                    for child in subtree:
                        if hasattr(child, 'label') and child.label().startswith('VB'):
                            verb = child.leaves()[0].lower()
                            break
                    if verb is None:
                        continue

                    # Build frame signature
                    frame_parts = []
                    for child in subtree:
                        if hasattr(child, 'label'):
                            label = child.label()
                            if label in ('NP', 'PP', 'SBAR', 'S', 'VP', 'ADJP', 'ADVP'):
                                frame_parts.append(label)

                    frame = '+'.join(sorted(frame_parts)) if frame_parts else 'INTRANS'
                    verb_frames[verb][frame] += 1

    complexity = {}
    for v, frames in verb_frames.items():
        total = sum(frames.values())
        if total >= 5:
            complexity[v] = len(frames)

    print(f"\nVerbs with MOST frame types (syntactically flexible):")
    for v, c in sorted(complexity.items(), key=lambda x: -x[1])[:20]:
        frames = verb_frames[v]
        total = sum(frames.values())
        top = frames.most_common(3)
        print(f"  {v:>15s}: {c} frames ({total} tokens) — {', '.join(f'{f}:{n}' for f,n in top)}")

    print(f"\nVerbs with FEWEST frame types (rigid):")
    for v, c in sorted(complexity.items(), key=lambda x: x[1])[:20]:
        frames = verb_frames[v]
        total = sum(frames.values())
        if total >= 5:
            top = frames.most_common(3)
            print(f"  {v:>15s}: {c} frames ({total} tokens) — {', '.join(f'{f}:{n}' for f,n in top)}")

    return complexity


def analyze_embedding_depth():
    """Analyze typical syntactic depth of words.

    Some words tend to appear in deeply embedded clauses (complementizers, relative pronouns)
    Others are main-clause words (sentence-initial adverbs, discourse markers)

    Measurable: average tree depth from phrase-structure treebank.
    """
    print("\n" + "=" * 60)
    print("ELEMENT 6: EMBEDDING DEPTH PREFERENCE")
    print("=" * 60)

    word_depths = defaultdict(list)

    for fileid in treebank.fileids():
        for tree in treebank.parsed_sents(fileid):
            for pos in tree.treepositions('leaves'):
                word = tree[pos].lower()
                depth = len(pos)
                word_depths[word].append(depth)

    avg_depth = {}
    for w, depths in word_depths.items():
        if len(depths) >= 5:
            avg_depth[w] = np.mean(depths)

    print(f"\nWords with DEEPEST embedding (often in subordinate clauses):")
    for w, d in sorted(avg_depth.items(), key=lambda x: -x[1])[:20]:
        print(f"  {w:>15s}: depth={d:.1f} ({len(word_depths[w])} tokens)")

    print(f"\nWords with SHALLOWEST embedding (main clause):")
    for w, d in sorted(avg_depth.items(), key=lambda x: x[1])[:20]:
        print(f"  {w:>15s}: depth={d:.1f} ({len(word_depths[w])} tokens)")

    return avg_depth


if __name__ == '__main__':
    print("DISCOVERING THE 6 MISSING ELEMENTS")
    print("=" * 60)
    print()

    # Analyze each candidate element
    clause_pref, verb_total, verb_sbar, verb_np, verb_pp = analyze_verb_frames()
    avg_deps = analyze_semantic_weight()
    prep_entropy = analyze_prep_selectivity()
    gov = analyze_noun_governability()
    complexity = analyze_argument_frame_complexity()
    avg_depth = analyze_embedding_depth()

    # Save for integration
    import json
    elements = {
        'clause_complement_pref': {k: float(v) for k, v in clause_pref.items()},
        'semantic_weight': {k: float(v) for k, v in avg_deps.items()},
        'prep_selectivity': {k: float(v) for k, v in prep_entropy.items()},
        'noun_governability': {k: float(v) for k, v in gov.items()},
        'frame_complexity': {k: int(v) for k, v in complexity.items()},
        'embedding_depth': {k: float(v) for k, v in avg_depth.items()},
    }

    with open('/home/asa/asa/new_elements.json', 'w') as f:
        json.dump(elements, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY: 6 CANDIDATE ELEMENTS")
    print(f"{'='*60}")
    print(f"1. clause_complement_pref: {len(clause_pref)} verbs scored")
    print(f"2. semantic_weight: {len(avg_deps)} verbs scored")
    print(f"3. prep_selectivity: {len(prep_entropy)} preps scored")
    print(f"4. noun_governability: {len(gov)} nouns scored")
    print(f"5. frame_complexity: {len(complexity)} verbs scored")
    print(f"6. embedding_depth: {len(avg_depth)} words scored")
    print(f"\nSaved to new_elements.json")
