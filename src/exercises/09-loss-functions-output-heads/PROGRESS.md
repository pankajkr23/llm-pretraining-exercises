# PROGRESS — 09 · Loss functions and output heads

The plan, the state, and the evidence. `REQUIREMENTS.md` (local only) holds the requirement text;
this file holds what we decided to build from it and how far it has got.

**Deliverable:** a public GitHub README link, incognito-testable, backed by the repo's own runnable
code and its logged results.

**Contract, in one line: seven numbers from Part 1, two losses from Part 2**, each one printed by
code that runs top to bottom rather than typed into prose.

---

## Status

| stage | what it delivers | state |
| --- | --- | --- |
| **1 · Scaffold** | generator skeleton, CI shard, root README row | **done** |
| **2 · Requirement** | `REQUIREMENTS.md` written from the requirements; conflicts named | **done** |
| **3 · The trunk** | a real 4-block transformer, untrained, laptop-sized | **done** |
| **4 · The tokenizer** | exercise 02's frozen BPE loaded, so targets print as *strings* | **done** |
| **5 · Shift** | `t+1` and `t+k` slices, the string table, the deliberate off-by-one | **done** |
| **6 · Masks** | padding, packed-document boundaries, contributing-token counts | **done** |
| **7 · Losses** | masked cross-entropy, perplexity, chunked, z-loss, smoothing | **done** |
| **8 · Heads** | tied vs untied parameter counts; the `t+2` head for Part 2 | in progress |
| **9 · Memory** | peak bytes, materialised against chunked, measured not estimated | not started |
| **10 · Harness** | one run producing all seven numbers into `results/` | not started |
| **11 · Training** | the short run Part 2 needs, and only Part 2 needs | not started |
| **12 · Tests** | every claim twinned — no-op setting *and* away from it | partial |
| **13 · Notebook** | the Colab notebook, importing the package, never re-implementing | not started |
| **14 · README** | the write-up, rendering `results/` rather than quoting it | drafted, stale |
| **15 · Web page** | the deployable explainer, to the twelve-part spine | not started |
| **15b · Register** | `SPINE_ENFORCED` + the landing card — both fail in two directions | not started |
| **16 · Submit** | PK's action, after production is live | blocked on 1–15 |

---

## The seven numbers, and where each one comes from

Part 1 of the requirements is seven separate asks. This is what each one means here, what it is
graded on, and the trap in it.

### 1 · Every tensor shape, with what each dimension is

`[batch, seq_len]` ids in; `[batch, seq_len, d_model]` hidden; `[batch, seq_len, vocab]` logits;
`[positions]` targets after the shift and the reshape. **Graded on the printing, not the
arithmetic** — the point is that the shapes are stated somewhere a reader can check them against
what the code did.

*The trap:* the logits tensor is the one worth stopping on. It is `vocab / d_model` times larger
than the hidden states that produced it, which is where stage 9's whole argument comes from.

### 2 · The shift, verified in token **strings**

`shift.shift_table` prints input beside target, one position per line, as text. **This is the item
the requirements warn about.** A shift in the wrong direction produces a *better* loss curve
and raises nothing.

*Done, and the deliberate bug is kept as `shift.shift_wrong_way`* so a test can assert the loss
falls when it is introduced. A warning nobody has watched come true is a warning.

### 3 · Padding masked, with the contributing count

`masks.keep_non_padding` drops a position if **either** its input or its target is padding, and
returns a `MaskReport` carrying the count. **The count is the deliverable**, not the mask.

*The trap, and it is ours:* exercise 02's tokenizer has 10,000 entries and **no padding token**, so
`[PAD]` is this exercise's own addition at id 10,000 and `Config.vocab_size` is 10,001. Reusing
`[UNK]` (id 0) would have conflated "padding" with a token the model may legitimately predict.

### 4 · Two documents packed, the boundary masked, loss before and after

`masks.pack_documents` concatenates and records which document owns each position;
`masks.keep_within_document` drops any position whose target belongs to a different document.
Report **both** losses and explain the difference.

*The trap:* the boundary mask must be built for the horizon it is used at. A `t+2` head crosses a
join two positions early, so a mask built for `t+1` leaves exactly one crossing pair per join —
near-miss, survives review. `keep_within_document` takes `horizon` for this reason.

### 5 · Perplexity, and the untrained anchor

`exp(mean loss)`. An untrained model over `V` classes must sit near `V`, because uniform is what
"knows nothing" means. **If it does not, stop and fix the alignment before anything else.**

*Two numbers get reported:* `ln(10,001) = 9.2105` for our vocabulary, and the course model's
`ln(131,072) = 11.7778` alongside — the second is arithmetic, not a measurement of ours, and gets
labelled that way.

*The caveat that must be in the open text:* perplexity is **not comparable across tokenizers**. A
tokenizer that splits more finely is asked an easier question per step and scores better while
being no better. Exercise 02 measured a real case of this.

### 6 · Tied against untied head parameters

Untied is `d_model × vocab_size` on top of the embedding. Tied is **zero** — not "fewer" — because
the embedding already holds every number the head would buy.

*The trap, and it is the interesting part:* tying is the standard escape and **it is not always
available**. It needs an input embedding table with one row per token to tie *to*. An architecture
whose input side is a codec plus a projection has no rows, so the saving is simply closed off. The
table should therefore have three rows, not two: untied, tied, and tying-unavailable.

### 7 · Peak memory, ordinary against chunked, with the ratio

Measured, not estimated. Both numbers and the ratio.

*The trap:* the two must produce the **same loss to floating point**, and the report is worthless
without that stated. Chunking is not an approximation; if the numbers differ, something else broke.

*Second trap, already found:* chunked cross-entropy must divide by the **contributing** count, not
the row count. Every test written on unmasked input passes either way, which is how that ships.

---

## Part 2 — the `t+2` head

Add a second head predicting two positions ahead. Report both losses separately, their sum, and
what happens to head 2 over training relative to head 1.

- **The trunk is shared; only the slice changes.** `shift.shift_for_horizon(tokens, 2)` is the
  entire difference on the data side, which is worth showing directly.
- **The losses add.** That is the whole of multi-token prediction as an objective.
- **The expected finding: head 2's loss sits above head 1's, and stays there.** Predicting further
  ahead is genuinely harder, and this is the one item in the requirements that needs an actual
  training run to answer — a single forward pass cannot say what happens "over training".
- **State the expectation before the result.** It costs nothing and it is the only way a reader can
  tell a finding from a story told backwards.
- **The honest cost:** `k` heads means `k ×` the head parameters. On a large vocabulary that is the
  argument against dense extra heads, and it belongs next to the result.

---

## What is left, in order

1. **Stage 8 · heads** — extend `heads.py` with a `MultiTokenHead` over `Config.horizons`, and make
   the parameter table carry the third row (tying unavailable).
2. **Stage 9 · memory** — a `memory.py` that measures peak allocation for both paths on the same
   input, and asserts the two losses agree before reporting any ratio.
3. **Stage 10 · harness** — one entry point writing `results/harness.json`. Every number the README
   renders comes from that file; none is typed.
4. **Stage 11 · training** — a short run over a real corpus slice, logging both heads' losses per
   step to `results/`. **Test the last line first:** exercise 05 lost fifteen trained models to a
   `json` encode failure in a final statement, so the write path gets exercised on a two-step run
   before any longer one.
5. **Stage 12 · tests** — twins for every claim, plus the off-by-one watched making the loss fall.
6. **Stage 13 · notebook** — imports the package, `lite` under ten minutes, MPS and CUDA, outputs
   stripped, built by the local builder.
7. **Stage 14 · README** — the three-reader path, the seven numbers rendered from `results/`, and
   what this cannot establish.
8. **Stage 15 · web page** — the twelve-part spine, a mechanism figure and not only results, one
   failure in the opening tiles, six themes.
9. **Stage 16 · submit** — PK's, once production is live and the link resolves anonymously.

---

## Decisions taken (2026-09-03)

- **The notebook stays local; `results/` is tracked.** The wording offers *"the ipynb file **or**
  training logs"*, and training logs are our own output rather than course material — so tracked
  `results/` plus a README rendering them satisfies it with no exception to the convention.
  Exercise 10's requirement offers no such alternative, and **its** notebook is tracked under a
  written exception. Goes in `DECISIONS.md`.
- **This exercise ships a `web/` page**, built to `docs/DESIGN.md` with the twelve-part spine in
  order, and joins `SPINE_ENFORCED` and the landing card. PK's call, overriding a recommendation to
  skip it on deadline grounds. **It is built after stage 10**, because a page whose figures are not
  yet measured is a page that gets rebuilt.
- **09 is finished before 10 starts**, timeboxed — 10 reuses this exercise's config, trunk,
  tokenizer, shift, masks and losses unchanged, so finishing here hands 10 a foundation rather than
  a blank directory. Exercise 10 is due 5 September and is the only live deadline.

## Still open

- **How much training for Part 2?** Enough to see the ordering hold, not enough to claim a curve.
  The number of steps gets stated next to the result rather than chosen quietly.

## What this exercise cannot establish

Recorded here as it accrues, so the README's section is derived from work rather than written at
the end:

- Nothing here says which loss trains a *better* model. Every equivalence is a statement about
  arithmetic being right.
- The memory numbers are CPU-measured at laptop shapes. The scale where the logits tensor stops
  fitting on any accelerator is arithmetic quoted from the course material, not something this
  exercise ran.
- One tokenizer, one vocabulary. Perplexity moves with both.
