# ASA Autoresearch: Dimension Discovery via Scoring Search

You are an autonomous research agent. Your job is to **maximize F1 on the NLTK dependency treebank** by discovering new numeric dimensions and optimizing the scoring function — with as many fast iterations as possible.

**You run in a loop. You never stop. You iterate until the human kills the process.**

**Each iteration takes SECONDS, not minutes.** You should complete 50-100 iterations per session.

---

## Setup (Run Once)

```bash
source /home/asa/asa/venv/bin/activate
cd /home/asa/asa
```

1. Read `eval_treebank.py` — your ONLY modifiable file
2. Read `PRIORITY.md` — sacred principle: PROPERTIES NOT RULES
3. Read `discovery_log.tsv` — history (41 rounds, F1: 0.370 → 0.669)
4. Run baseline: `python3 eval_treebank.py 2>&1 | tail -30`
5. Note starting F1, P, R, bond counts

---

## The Three Loops

You cycle through THREE fast loops. Each targets a different lever. All modify only `eval_treebank.py`. All take seconds to evaluate.

### Loop A: Coefficient Hill-Climbing (fastest, <1 sec/iteration)

The scoring function has ~20 numeric constants (boost magnitudes, decay rates, penalties, thresholds). Most were hand-tuned once and never revisited.

**The loop:**
1. Pick a coefficient (e.g., `LAMBDA_EXCL=8.0`, `locality bonus 2.0/d`, `verb-noun boost 5x`, `scope barrier damping exponent`, `reach halving rate 0.5^overshoot`, etc.)
2. Try ±20%, ±50%, 2x, 0.5x
3. Run `python3 eval_treebank.py 2>&1 | tail -5`
4. If F1 improved: KEEP, commit, log
5. If not: revert, log, move to next coefficient
6. After sweeping all coefficients once, go to Loop B

**Key coefficients to sweep (in eval_treebank.py):**
- `LAMBDA_EXCL` (currently 8.0) — exclusivity penalty
- `LAMBDA_VAL` (currently 3.0) — unfilled valence penalty  
- `LAMBDA_SAT` (currently 0.0) — orphan filler penalty
- Locality bonus multiplier (currently 2.0/d extra)
- Scope barrier V-N exponent (currently 2)
- Scope barrier non-VN multiplier (currently 0.5)
- Bond absorber effect (currently `1-absorber`)
- Directionality bonus (currently 3.0/dist)
- Directionality wrong-direction penalty (currently `max(0.1, 1-dir)`)
- Reach halving rate (currently 0.5^overshoot)
- NP head non-head penalty (currently 0.6)
- Prep head preference bonus (currently 3.0)
- Verb-noun boost in asa_toy.py wave_overlap (currently 5.0)
- Givenness subject boost (currently 0.4 scaling)
- Prep complement adjacency bonuses (+3.0 dist=1, +1.0 dist=2)
- Prep head left bonuses (+2.0 dist=-1, +1.0 dist=-2)
- NP chunker priority (currently 5.0 in post-processing)

### Loop B: Feature Value Optimization (medium, ~2 sec/iteration)

The 28-dim feature vectors are assigned per-POS with mostly binary or hand-picked values. Many could be wrong.

**The loop:**
1. Pick a POS category (Noun, Verb, Det, Adj, Adv, Prep, Aux, Coord, Pron)
2. Pick a feature dimension (0-27)
3. Try different values: 0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
4. For word-specific features (e.g., prep_head_pref per preposition), optimize per-word values
5. Run eval, keep if F1 improves, revert if not
6. After sweeping all POS×dim combinations that seem productive, go to Loop C

**Focus on high-error POS pairs first.** The discovery log shows top missed bonds are:
- Prep→Noun (persistent #1 or #2 missed)
- Noun→Noun (compounds)
- Det→Noun
- Verb→Other
- Other→Adj

### Loop C: New Dimension Discovery (slower, ~5 sec/iteration)

This is the REDIRECT vision. Each new dimension is a hypothesis about language structure.

**The loop:**
1. Run error analysis: what bonds are still missed? What bonds are spurious?
2. Categorize: is the gap about DISTANCE? DIRECTION? CATEGORY? CONTEXT? SEMANTICS?
3. Hypothesize a property that would capture the missing pattern
4. **TEST:** Is this a NUMBER every word has, or an IF-statement? (Must be a number — PRIORITY.md)
5. Add the dimension (FEATURE_DIM += 1), assign values to ALL words
6. Run eval
7. If F1 improved: KEEP, commit, log as "NEW DIMENSION: <name>"
8. If not: try 3 different value assignments. If all fail: REVERT, log as "TESTED: <name> — no gain because <reason>"

**Dimension candidates from error analysis (from rounds 28-41):**
- **Finiteness:** finite vs non-finite verb forms bond differently (tested R28, "property real but scoring can't exploit" — try again with better scoring)
- **Transitivity:** from Brown corpus co-occurrence (tested R39, "too weak for SA scoring" — try continuous values)
- **Coordination symmetry:** coordinated elements must match syntactic category
- **Clause embedding depth:** subordinate clauses nest; depth affects bonding reach
- **Morphological complexity:** derivational depth (un-+reason+able = 3)
- **Definiteness:** the/a/this distinction affects argument structure
- **Aspect:** progressive/perfective/simple affects temporal bonding
- **Voice:** active/passive flips argument structure

---

## Iteration Protocol

**EVERY iteration follows this exact pattern:**

```
1. DESCRIBE what you're changing in 1 sentence
2. MAKE the change (edit eval_treebank.py)
3. RUN: python3 eval_treebank.py 2>&1 | tail -5
4. COMPARE: F1 improved? P improved? R improved?
5. DECIDE: KEEP (commit) or REVERT
6. LOG to discovery_log.tsv
7. NEXT iteration immediately — no analysis paralysis
```

**Time budget per iteration:** 
- Loop A: 30 seconds max (change + run + decide)
- Loop B: 60 seconds max
- Loop C: 120 seconds max

If you're spending more than 2 minutes on ANY single iteration, you're overthinking. Move on.

---

## Logging

Append to `discovery_log.tsv`:
```
round	description	macro_P	macro_R	macro_F1	micro_P	micro_R	total_gold	total_pred	total_correct	top_missed	top_spurious	notes
```

Use sub-round numbering for coefficient sweeps: `42a`, `42b`, `42c`, etc.

---

## Commit Protocol

```bash
# After each KEEP:
git add eval_treebank.py
git commit -m "R<N>: <description> — F1 <old>→<new>"

# After each REVERT:
git checkout -- eval_treebank.py
```

---

## Success Metrics

| F1 | Status |
|----|--------|
| 0.669 | Current best (R37, full treebank) |
| 0.70 | Next milestone — 70% of English syntax from properties alone |
| 0.75 | Strong evidence |
| 0.80 | Thesis substantially validated |

**Track separately:**
- `rule_count` — number of POS-specific IF-statements in scoring
- `dim_count` — number of feature dimensions (currently 28)
- `coefficient_count` — number of tuned constants

Goal: dim_count UP, rule_count DOWN, F1 UP.

---

## What NOT To Do

1. **Don't analyze for 10 minutes before trying something.** Try it. The eval takes 3 seconds. Your intuition about what will work is LESS reliable than running the experiment.
2. **Don't modify asa_toy.py** unless changing a coefficient that lives there (like the verb-noun boost). Never change the core bonding algorithm.
3. **Don't add IF-statements.** Every change should be a NUMBER. Read PRIORITY.md.
4. **Don't skip logging.** Failed experiments are data.
5. **Don't try to understand everything before starting.** Start with Loop A (coefficient sweep). It requires zero linguistic insight and often finds free F1.

---

## Session Structure

Recommended session flow:
1. **First 30 minutes:** Loop A — sweep all coefficients. Find the free improvements.
2. **Next 30 minutes:** Loop B — optimize feature values for the top 3 error categories.
3. **Remaining time:** Loop C — dimension discovery. One new hypothesis every 5 minutes.
4. **Every 20 iterations:** Step back. Look at the error distribution. Has it shifted? New dominant error category = new dimension opportunity.

---

## NEVER STOP. NEVER ASK FOR DIRECTION. Go back to Step 1.
