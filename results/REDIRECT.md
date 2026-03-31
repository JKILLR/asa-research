# REDIRECT: Discovering the Structural Code of Language

Stop. Read this completely. The research has fundamentally changed.

## What We Are Actually Doing

We are not optimizing a transformer. We are trying to decode the fundamental structural code that gives language its meaning -- the same way chemistry decoded the structural code that gives matter its properties.

Carbon does not need a neural network to know it forms 4 bonds. Its electron configuration DETERMINES that. The question is: do words have an equivalent configuration? And if so, what are the properties that define it?

This has never been done. There is no paper to follow. We are discovering something.

## What We Have Proven

The architecture works. Wave function overlap as attention, wave-gated FFN, 1.19x compression, t=9.05, p<<0.001. The MECHANISM for testing the thesis is validated. Do not touch it again.

What we have NOT proven: that we know the right properties. Our current "periodic table" has 24 dimensions -- 9 syntactic (POS pair rules) and 15 semantic (WordNet categories). This is like chemistry knowing about 24 elements. The real table might have 200. The 1.19x compression we get with 24 dims is a hint of what is possible, not the destination.

## The New Research Question

What are the fundamental properties of words that predetermine how they combine?

This is not "what NLP features help a language model." That is the wrong frame. The question is deeper: what is the MINIMAL set of properties such that, if you know them for every word in a sentence, the parse structure is DETERMINED -- no learning required?

Chemistry answers this for matter: atomic number + electron configuration + electronegativity -> bonding behavior is determined. What is the equivalent for language?

## How to Test This: Three Speeds

### Speed 1: The Toy (seconds, no learning)

asa_toy.py is the purest test. It does energy-minimization parsing with ZERO learned parameters. If a linguistic property helps the toy produce correct parses, it is a real structural element. If it does not help, it is noise.

The toy currently has 37 tokens and 12-dim features. EXPAND IT:
- Add 200+ words covering diverse categories
- Implement new feature dimensions for each hypothesis
- Parse 500+ sentences from a dependency treebank (NLTK has parsed corpora)
- Measure: what percentage of gold-standard dependency bonds does energy minimization recover?
- Each feature set = a hypothesis about the periodic table
- Test 50 hypotheses per hour

This is the inner loop. Fast, no GPU, pure structure.

### Speed 2: Attention Correlation (seconds, no training)

Take the trained standard transformer. Extract where it actually attends. For each candidate linguistic feature, compute: does knowing this feature predict where attention goes?

High correlation = the model spent parameters learning this structure = encode it.
Zero correlation = either noise, or something the model cannot learn (interesting either way).

### Speed 3: Full Validation (minutes, GPU)

Only for feature sets that passed Speed 1 and 2. Run the wave-gated comparison. This is confirmation, not exploration.

## What to Explore

Think of these not as "NLP features" but as candidate ELEMENTS of a structural code. The question for each: does this property predetermine bonding behavior?

RELATIONAL GEOMETRY:
- How many things can this word connect to? (valence -- we have this)
- What ROLES can it assign or fill? (thematic roles -- we have basic ones)
- What is its DIRECTIONALITY? (does it look left, right, or both for its arguments?)
- What is its SCOPE? (does it govern one word, a phrase, a clause, the whole sentence?)

CATEGORICAL IDENTITY:
- What TYPE of thing is it? (we have this via WordNet, but coarsely)
- What LEVEL of abstraction? (concrete to abstract spectrum, not binary)
- Is it REFERENTIAL (points to something in the world) or FUNCTIONAL (organizes structure)?
- Is it CONTENTFUL (carries meaning) or GRAMMATICAL (carries structure)?

COMPOSITIONAL BEHAVIOR:
- Does it CHANGE the category of what it attaches to? (derivational morphology)
- Does it RESTRICT or EXPAND the meaning of what it attaches to?
- Does it create a NEW referent or modify an EXISTING one?
- Is it ENDOCENTRIC (same type as its head) or EXOCENTRIC (different type)?

INFORMATION DYNAMICS:
- Does it introduce NEW information or refer to OLD?
- Is it the TOPIC (what we are talking about) or COMMENT (what we say about it)?
- Is it PRESUPPOSED (taken for granted) or ASSERTED (the new claim)?
- Does it create an EXPECTATION for what comes next?

TEMPORAL/CAUSAL STRUCTURE:
- Does it place something in TIME? (tense, aspect)
- Does it establish CAUSATION? (because, therefore, so)
- Does it create CONDITIONALITY? (if, unless, whether)
- Does it express MODALITY? (can, must, might -- possibility vs necessity)

BOUNDARY MARKERS:
- Does it OPEN a new structural unit? (subordinators, relative pronouns)
- Does it CLOSE one? (punctuation, certain adverbs)
- Does it LINK two units? (coordinators, discourse connectives)

Do not think of these as a checklist. Think of them as HYPOTHESES about what the structural code contains. Test them empirically. Some will be redundant. Some will be transformative. Some we have not thought of yet -- and the gaps in our results will point to them, just as gaps in Mendeleev's table predicted undiscovered elements.

## The Meta-Pattern

Each round of discovery should follow this pattern:

1. Run the current best feature set through the toy on hundreds of sentences
2. Look at WHERE IT FAILS -- which bonds does it get wrong?
3. Ask: what property of language, if I knew it, would let me predict the correct bond?
4. That property is a candidate element
5. Encode it, test it, keep or discard
6. The failures AFTER adding it point to the NEXT element

The errors are the map. They tell you what is missing from the table.

## Success Criteria

- 24 dims, 1.19x compression (current)
- 50 dims, ??? (if 1.5x, we are on the right track)
- 100 dims, ??? (if 2x+, the thesis is vindicated)
- If compression plateaus no matter what we add, the thesis has a ceiling, document it honestly

## Remember

This is not incremental ML research. This is an attempt to discover something fundamental about how language -- and by extension, structured thought -- works. The periodic table of chemistry was not found by optimizing furnace temperatures. It was found by noticing patterns in the properties of elements. We are looking for the equivalent patterns in the properties of words.

Think creatively. Question assumptions. Follow the errors.
