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
| **7 · Losses** | masked cross-entropy, perplexity, both chunkings, z-loss, smoothing | **done** |
| **8 · Heads** | tied / untied / tying-unavailable, and the `t+2` head | **done** |
| **9 · Memory** | peak RSS in isolated child processes, measured not estimated | **done** |
| **10 · Harness** | one run producing all seven numbers into `results/harness.json` | **done** |
| **11 · Training** | 300 steps, both findings, into `results/training.json` | **done** |
| **11b · Sensitivity** | the same at 60/150/300 steps + five memory repeats | **done** |
| **12 · Tests** | 54 tests; every claim twinned, the off-by-one watched falling | **done** |
| **13 · Review** | auditor, engineer and reader run over the finished work | **done** |
| **14 · Fixes** | three blockers and eleven lesser findings from that review | **done** |
| **15 · Notebook** | the Colab notebook, importing the package, never re-implementing | **done** |
| **16 · Web page** | the deployable explainer, to the twelve-part spine | **done** |
| **16b · Register** | `SPINE_ENFORCED` + the landing card — both fail in two directions | **done** |
| **17 · Submit** | PK's action, once production serves the page | blocked on the deploy |

## What the review found, and what it cost

Three reviewers read the finished work. **Three blockers, and every one was a claim that read as
checked and was not** — which is the shape this repository keeps paying for:

1. **`keep_within_document` kept every pad-to-pad pair.** `-1 == -1` is `True`, so a mask written
   as `source == destination` dropped nothing but the two transitions. 68 of 125 "contributing"
   positions were padding predicting padding, in the exercise whose item 3 exists to forbid exactly
   that. **The guard agreed with the bug**, because it asserted the dropped count equalled a count
   of transitions — the same expression the implementation used, so it held for any input at all.
2. **`RESULTS.md` claimed every figure in it was generated, and fifteen were typed** — the
   sensitivity sweep and the memory repetitions, which are the two blocks the document leans on
   hardest. One printed the 300-step correct shift as `4.15` where the generated table above read
   `4.1447`. The byte-equality test could not see them: they lived *inside* the template it compared
   against. They are `results/sensitivity.json` now.
3. **A `pytest.importorskip` turned a repo-wide guard red**, and I had run only this exercise's
   directory. It was spurious too — `sys.path` was set two lines above it, so it could never fire.

Eleven lesser findings were fixed alongside: an `assert ... or True`, the harness computing item 2's
comparison over unpadded positions, "the loss an untrained model **must** show" printed beside a
different measured number, the memory child re-implementing the function it was measuring,
`check=True` burying a child's traceback, and four documents shipped as unfilled templates.

**One finding was not a defect and is now stated rather than fixed: the corpus is read 8.55 times
over.** Every loss here is a memorisation number. Both findings survive it — each compares two
models trained identically on the same repeated text — but the absolute values do not transfer, and
`RESULTS.md` now says so.

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

1. **The notebook** — imports the package, `lite` under ten minutes, MPS and CUDA, outputs stripped,
   built by the local builder. Gitignored; `results/` is the tracked evidence instead.
2. **The web page** — the twelve-part spine in order, a mechanism figure and not only results, one
   failure in the opening tiles, six themes, built to `docs/DESIGN.md`.
3. **Both registrations** — the landing card in `deploy/vercel/index.html` and the `SPINE_ENFORCED`
   entry. Neither goes in before the page exists; both guards fail in two directions.
4. **Submit** — PK's, once production is live and the link resolves anonymously.

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
