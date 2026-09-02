# CLAUDE.md — 04-data-cleaning-dedup

Component notes. Repo-wide conventions: root `AGENTS.md`. The decision record — why the answer is
8, why these corpora, what may be published — is `DECISIONS.md`, and it is the thing to read first.
`REQUIREMENTS.md` is the requirements, and is gitignored: requirement documents are input for whoever builds an exercise,
not part of the published work.

## The rules this exercise adds

- **Count tokens; never estimate them.** Fertility is a property of a *tokenizer*, not of a corpus:
  Manipuri swings 7.6× across the five tokenizers exercise 03 measured. `tokens.py` counts with our
  own Exercise 02 vocabulary, and any figure without a named tokenizer is not a measurement.

- **A count that is mostly `[UNK]` is not a count.** `TokenCount.usable` gates publication at 5%
  `[UNK]`. Above it, `as_figure()` returns `value=None` with provenance `unknown` and the reason in
  `source`. This is not a display preference — it is what selected the corpus. Bengali script
  measures 82–84% `[UNK]`, which is why the Indic corpus is Devanagari and Telugu.

- **Two publication invariants, in `tests/test_publication_invariants.py`.** No personal information in any
  published artifact, and no corpus text beyond a bounded window (12 excerpts × 300 characters).
  Both scan **every byte** of `web/` and the notebook. Both have twins that plant a leak. **Do not
  merge a change to `web/` with either twin green for the wrong reason.**

- **Declare stand-ins; never publish their accuracy.** The classifier gate has no model behind it
  and is off by default. The PII name layer is a gazetteer, not NER, and publishes `precision: null`
  with `provenance: "unknown"`. Inventing a number for either is the same error.

- **UNCHECKED is not clean.** When the gated benchmark index is absent, decontamination reports
  `coverage: "none"` and the headline says UNCHECKED. A stage that reported "0 contaminated" there
  would read as a clean bill of health.

## One rule, one implementation — except where it cannot be

`chapters.js` duplicates six rules from Python: `cleanText`, `unescapeFully`, `lshThreshold`,
`pCandidate`, `jaccard`/`shingle`, and both PII layers. The page recomputes them live as a reader
drags a slider, and a round-trip per keystroke is not an interaction.

**`tests/test_agreement.py` is what makes that safe.** It rewrites `chapters.js` into a harness
*beside itself* (relative imports resolve against the importing file) and diffs both
implementations over shared fixtures. It caught a real divergence on its first run: the JS sentinel
used for protecting Indic joiners was itself inside the noise class, so the joiner came back as a
stray letter. If you touch either side of a duplicated rule, this test is the reason you can.

## Things that bit, so they do not bite again

- **Python's `hash()` on strings is randomised per process.** Using it for shingles made which
  documents deduplication deleted drift between runs. `dedup._stable_hash` uses blake2b and a test
  pins a hard-coded digest.
- **Python's `\w` and `isalnum` skip Devanagari vowel signs** (a matra is category `Mn`), so
  `mean_word_length` measured every Devanagari word short and deleted well-formed Hindi at 2.24
  against a floor of 3.0. `quality._word_length` counts letters *and* marks. Same family as
  exercise 03's correction X16.
- **Stage 2 destroys stage 2b's evidence.** Whitespace collapse erases the blank lines between
  conversation turns, so the turn count is captured on `Document.turns` at load time. Recovering it
  downstream silently returned 1 and made the format overhead look like nothing.
- **A canary shorter than the scanning window is invisible.** Five-word canaries against a
  thirteen-word scan recovered 0 of 24 while the note said the scanner worked. `canary_strings`
  takes the width and builds `width + 2` words.
- **Row groups are not a budget.** Sangraha's Telugu shard has row groups of tens of thousands of
  documents, so checking the token budget only *between* them overshot twentyfold. `corpus.py`
  checks inside each row group.
- **Absolutely-positioned tooltips contribute to scroll width even while invisible.** One term near
  the right edge pushed the page 312px sideways. They are `position: fixed`, placed by script, and
  clamped to the viewport.

## This deduplication does not scale past this exercise

`dedup.py` holds a **full shingle set for every document** at once, and only ever sees one run's
documents — so shard N is never compared with shard N-1. Both are fine here and neither reaches
Exercise 01's one-billion-token gate.

Measured on real prose: a 500-word document's shingle set is **73×** the size of its MinHash
signature, and the gap widens with length (1,199× at 10,000 words). This exercise's full run holds
~2.4 GB of shingle sets resident; at ~616k documents it would need **40.5 GB**.

**The scalable path is `05-datamixtures-and-curriculum/src/mixture/accumulate.py`**, which persists
signatures only, streams them from disk, and deduplicates each new shard against every earlier one.
It buys that with accuracy: cross-shard pairs are judged by the MinHash *estimate* rather than exact
Jaccard, so its threshold is widened by one standard error — a false keep leaves a duplicate, a
false drop deletes text that never comes back. Within a shard it still uses this module's exact
check.

Nothing here is deprecated. Exercise 04's published numbers were produced by this code and stand;
the store is the continuation, not a replacement.

## Layout

`config.py` holds every threshold in one frozen dataclass, and its `fingerprint()` lands in the
manifest so a threshold change is visible as a different run. `sources.py` holds the corpora with
their verified shard byte sizes — a mismatch means upstream replaced the file. `pipeline.py`
composes the stages; `IMPLEMENTED` is the map, and anything absent from it falls back to a counting
pass-through that declares itself with `real: false`.

`extract` is permanently a pass-through: every corpus here ships extracted text, so claiming a
yield for it would be inventing one.

## Running it

```bash
uv run python -m datacleaning.fetch --profile lite   # reachability, seconds
uv run python -m datacleaning --profile lite         # smoke run, ~2 minutes
uv run python -m datacleaning --profile full         # the published corpus
uv run pytest src/exercises/04-data-cleaning-dedup   # 160 tests
uv run pytest -m integration                         # browser suite, needs chromium
```

Local network note: Python verifies TLS against `certifi/cacert.pem`, which Claude Code's sandbox
denies by default. `.claude/settings.local.json` carries a narrow read-allow for that one file.
Colab is unaffected.
