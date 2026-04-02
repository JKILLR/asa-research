# PRIORITY: Properties, Not Rules

**READ THIS BEFORE EVERY ROUND.**

## The Distinction

A **property** is a measurable dimension that EVERY word has a value for.
A **rule** is a conditional that fires for specific categories.

Properties (GOOD):
- Prep head preference: "of"=0.03, "to"=0.88 (continuous, per-word)
- Valence: "eat"=2, "sleep"=1, "the"=1 (a number every word has)
- Scope radius: "not"=1, "because"=5 (how far governance reaches)
- Abstractness: "idea"=0.9, "rock"=0.1 (continuous spectrum)
- Referentiality: "the"=1.0, "very"=0.0 (does it point to something?)

Rules (BAD):
- "If Noun adjacent to Noun, bond them"
- "LAMBDA_EXCL = 8.0"
- "Clause boundary = 0.10 penalty for V-N"
- "If distance > 3, damp verb-noun boost"
- "Aux bonds ONLY to Verb"

## The Test

For every change you make, ask: "Did I add a NUMBER that every word has, or did I add an IF statement?"

- If you added a number -> you discovered a property. Log it as a DIMENSION.
- If you added an IF statement -> you wrote a patch. It may improve F1 but it is NOT a discovery.

## What Properties Look Like in Code

GOOD: A new feature dimension = "scope radius"
Every word gets a value. Bonding emerges from the values.

  "not":     scope = 1.0  (governs next word)
  "very":    scope = 1.0
  "because": scope = 5.0  (governs a clause)
  "the":     scope = 0.5  (determiner reach)

BAD: A rule that fires for specific POS

  if seeker.pos == "Verb" and filler.pos == "Noun" and dist > 3:
      score *= 3.0 / dist   <-- this is engineering, not discovery

## The Goal

Find the MINIMAL SET OF DIMENSIONS such that, given every word's values on those dimensions, correct parses emerge from energy minimization WITHOUT category-specific rules.

The periodic table of elements doesn't say "carbon bonds to oxygen." It gives carbon and oxygen NUMBERS (electronegativity, electron shells, atomic radius) and bonding FALLS OUT of those numbers.

Same here. Don't say "verbs bond to nouns." Give verbs and nouns NUMBERS and let bonding fall out.

## What To Do Now

1. Review eval_treebank.py. Identify every IF-statement that encodes a category-specific rule.
2. For each rule, ask: what PROPERTY would make this rule unnecessary? What number could each word carry such that the rule emerges from dot products / energy minimization?
3. Add that property as a new feature dimension. Give every word in the vocabulary a value.
4. Remove the rule. See if the property alone recovers the F1.
5. Log in discovery_log.tsv: round, dimension name, dimension index, F1 before, F1 after, whether the rule was successfully REPLACED by a dimension.

## Tracking

For each round, log:
- How many NEW DIMENSIONS were added
- How many RULES were replaced by dimensions
- The current dimension count
- The current rule count

The goal is: DIMENSION COUNT UP, RULE COUNT DOWN, F1 UP.

## Examples of Rules to Replace

1. POS_RULES["Aux"] = ["Verb", "Aux"] -> What property makes Aux seek Verb? Maybe a "structural_target" dimension where Aux and Verb have matching values.

2. Clause boundary penalty (0.10 for V-N crossing comma) -> What property do commas/subordinators carry? A "scope_barrier" dimension. Words with high scope_barrier block bonds from crossing them.

3. Distance damping for V-N at dist>3 -> Why? Because verbs have limited argument reach. That IS a property: "argument_reach" = how far a verb can grab. "eat"=2, "said"=4 (takes a clause).

4. Noun compound adjacency rule -> What property makes nouns compound? A "compound_affinity" dimension. "stock"=0.8, "market"=0.8, "the"=0.0.

5. The 5x verb-noun boost -> Why does this exist as a multiplier? Because verbs NEED nouns. That need is already partially encoded in valence, but the 5x is a hack. The need should emerge from feature dot products being naturally high for verb-noun pairs with matching selectional properties.

Every rule you replace with a property is a step toward the periodic table.
