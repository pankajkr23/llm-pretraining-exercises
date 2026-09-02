# CLAUDE.md — 08-modern-attention-variants

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is a public web app plus
a sourced chronology; the reasoning is `DECISIONS.md`, the running log is `PROGRESS.md`, and
`BRIEF.md` is the assignment (local only, gitignored).

**Status: shipped.** `config.py`, `cache.py`, `sources.py`, `catalogue.py`, `timeline.py`,
`results/mechanisms.json`, and the page at `web/` — twelve spine sections, the two-object mechanism
figure, and the timeline. Registered in the `rest` CI shard, the landing card, `SPINE_ENFORCED`, and
`OPTIONAL_DEPENDENCY_GATES` (the render test gates on playwright).

**The page derives nothing.** `tools/build_web_data.py` reads the catalogue and the same functions
the tests exercise, and emits `web/data.js`; `chapters.js` renders only what is in it. After changing
a date or a trade-off:

```bash
uv run python src/exercises/08-modern-attention-variants/tools/build_web_data.py
bash deploy/vercel/build.sh
uv run pytest src/exercises/08-modern-attention-variants -m integration
```

## What makes this exercise different

Every previous exercise measured something it ran. This one's central claim is a **chronology**, and
the instructor grades on it:

> "Your job is to be right about the dates, right about the trade-offs, and clear about the story."
>
> "Your agent will happily invent a launch date and describe a technique it has half remembered.
> Check every date against the actual paper or release."

He also says plainly that a missing mechanism scores zero, and invites us to catch errors in his own
material. So the rules below are all about evidence, not about code.

## The rules this exercise adds

- **No date without a primary source, enforced at construction.** `sources.Source` refuses to build
  a `verified` citation with no `url` or no `quoted_date`. `quoted_date` holds the source's own
  string — for arXiv, the **v1** submission-history line — so a reader compares two fields rather
  than trusting one number. `catalogue.unverified()` lists anything a reader could not check.

- **Use the arXiv `v1` date, and record which version you read.** Later versions move by months and
  sometimes years: Bahdanau's v1 is Sep 2014 and its v7 is May 2016, a twenty-month spread. Quoting
  a conference date instead of v1 changes the order of the timeline.

- **`confidence: "unverified"` is a legitimate value. Use it rather than guessing.** A catalogue
  that cannot express doubt will express confidence it has not earned.

- **A mechanism with no stated cost is rejected.** `catalogue.Mechanism.__post_init__` raises when
  `new_tradeoff`, `gives_up` or `when_to_choose` is empty. The assignment: *"If you write down a
  technique with only pros, you have not understood it yet."*

- **`MANDATED` is the instructor's own list, quoted, mapped to our keys.** The test reads his
  phrases, so a rename on our side can never silently drop one of his items. Do not reword the left
  side of that dict.

- **Reproduce the session's numbers; never copy them into prose.** `cache.kv_cache_bytes` recomputes
  6.44 GB at one user and 51.54 GB at eight, and GQA at two KV heads is exactly a quarter of MHA.
  Tests pin all three, so editing the yardstick breaks the documents that cite it.

- **The claimed arc is derived, not repeated.** The brief says the field went "exactness → memory →
  length → memory again". `timeline.pressure_by_period` counts which bill each window addressed, and
  `Period.dominant` returns `None` on a tie instead of picking a winner. If the arc is not in the
  data, say so.

## Two errors in the course material, both verified

Recorded because the assignment explicitly invites it — *"if you catch me in another one, tell me"* —
and because a reader deserves to know which claims we checked.

- **The transformer is mis-dated in the transcript.** It says Vaswani "invented in 2018 and 17";
  *Attention Is All You Need* is `arXiv:1706.03762`, v1 **Mon, 12 Jun 2017**, read from the abstract
  page. June 2017, not 2018.

- **DroPE is two different papers in the source, and the transcript quotes the wrong one's title.**
  The technique the session describes — pretrain with positional embeddings, drop them, recalibrate
  briefly — is *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings*,
  `arXiv:2512.12167` (Sakana AI), v1 **13 Dec 2025**. The transcript's garbled "rotate position
  emitting for efficient" maps instead onto **DRoPE** (capital R), `arXiv:2503.15029`, *Directional
  Rotary Position Embedding for Efficient Agent Interaction Modeling* — an autonomous-driving
  trajectory paper with no relation to the technique. Two papers whose names differ by one
  capital letter. Cite the first; footnote the second so nobody re-finds it and "corrects" us.

## One number that does not reproduce

The transcript says eight users at a 1M-token context need "about 1 TB". The session's **own
formula**, at the session's own yardstick, gives **1.57 TB**:

    2 x 48 x 8 x 128 x 1,000,000 x 8 x 2 = 1,572,864,000,000 bytes

Both are recorded. Do not publish either alone, and do not quietly adopt the rounder one — say which
inputs would reconcile them (a smaller model, fewer KV heads, or fp8 would each do it).

## Where the material actually comes from

`docs/sessions/s8.md` teaches ten of the eighteen mandated mechanisms. **Eight are named in the
coverage list and never taught**: sinusoidal, learned absolute positions, ALiBi, sliding window,
attention sinks, NTK-aware scaling, YaRN and MLA. Those are sourced entirely from outside the course
material, and `taught_in_session` on each entry records which is which — so a reader can see where
our evidence came from rather than assuming it all came from class.

## The page is a monograph, and four rules keep it one

It was rebuilt after a review found the previous version was text and tables: no explainers, no
graphics, misaligned, "a ten year old boy's project". The rebuild is six numbered plates and six
chapters. If you touch it, these are the rules that produced it.

- **A text card per mechanism is not a design, it is a list with a scrollbar.** The catalogue is
  **one object entered once per mechanism, shown three ways** — the plate (where each sits in time),
  the reading spread (what one traded, in depth), the index plate (all of them, same six fields in
  the same six places). Two of the three need no interaction at all, because a grader must not click
  two dozen times and a printed page must still carry the evidence.
- **Never type a count into page prose.** Every reader-facing count goes through `spell()` in
  `chapters.js`, which reads `M.counts.total`. The page said "twenty-three" in six places, and
  adding one mechanism made all six wrong at once while every table beside them stayed right —
  the failure `AGENTS.md` calls the most expensive in this repo, because only the sentences are
  wrong and a reader believes the sentences.
  `tests/test_attention_docs.py::test_no_count_is_typed_into_the_page_as_a_word` is lexical, because
  a runtime check cannot tell a derived "twenty-four" from a typed one. A spelled number that is
  genuinely fixed — a duration, the 6×6 grid — carries a `// count-literal-ok` marker rather than
  the guard being loosened until it stops catching anything.
- **A glyph is derived, never drawn.** Four generators in `web/glyphs.js` read the `pattern` block
  each catalogue entry carries. Adding a mechanism means adding a pattern, not drawing a picture.
  Two glyphs are load-bearing: **FlashAttention's field is byte-identical to standard attention's**
  because it is exact attention — a different shape there would be the worst factual error on the
  page — and **linear attention gets a state box, never a thin diagonal**, which would imply the
  opposite of what a fixed-size state does. `tests/test_attention_catalogue.py` pins both.
- **No shell commands on the page.** Commands go in the README, and
  `test_the_page_shows_no_shell_commands` enforces it.
- **`--accent` has exactly one job:** the current selection, the playhead, or the line being
  crossed. A second job and the plate stops reading at a glance under six themes.

## Six defects this page shipped with a green suite

Every one was found by **looking at a rendered screenshot**, and each now has a guard in
`tests/test_attention_render.py` named after it. This is the list to re-read before believing the
suite about a change under `web/`.

- **The verdict grid was handed `glyph()`** — which returns an SVG `<g>`. Appended into an HTML
  `<div>` a bare `<g>` renders nothing, so the grid drew frames and TIE stamps over a full set of
  invisible chips. Use `glyphSvg()` anywhere outside an existing `<svg>`.
- **`onFirstView` observed detached nodes.** Every figure asks for it before `chapters.js` appends
  it, and an `IntersectionObserver` on a detached node never fires, silently. Three plates never
  animated. It now defers a frame and checks `isConnected`.
- **The invoice cut line was invisible, and the guard passed it** because the guard scrolled the row
  into view before measuring — triggering the behaviour it was testing for. The reveal is deleted
  rather than fixed: an element invisible until a scroll is invisible to a screenshot, a print and
  an anchor landing. **Prefer a painted terminal state to an animated one when the motion buys
  nothing.**
- **Plate labels laddered to a fixed 48px** while a label is up to 200px wide, printing five staves
  on top of each other. Laddering measures the label now, across three tiers.
- **Every glyph escaped its viewBox.** The schema tilde sits at `x = size + 2`, and a square viewBox
  clipped almost all of them on two sides. An earlier mark at a negative `y` rendered on the caption of
  the glyph in the row *above* — SVG does not clip by default, so an escaping mark is present,
  legible, and attributed to the wrong mechanism.
- **The masthead field is `preserveAspectRatio="slice"`**, so it is wider than its box by design and
  scrolled a 320px screen sideways by 86px until it was given `overflow: hidden`.

## The plate is two plates, and the phone gets the other one

`figPlate` — all six plates live in `web/figures.js` — is a 1440-unit landscape SVG. Scaled into a 342px column every label is sub-pixel and the
page's centrepiece carries no information at all, so `figPlateTall` runs the same argument down the
page below 720px — same lanes, same to-scale gaps, same ties — and **drops the names**, because at
that width there is no honest way to fit them. Dropping a label is a decision; shrinking it to four
pixels is a pretence. A tap loads the entry into the reading spread, and the index plate below
prints every name with no interaction at all.

Both are built, both are selected together, and CSS shows exactly one. **The selector must be
`svg.plate-tall`, not `.plate-tall`**: `.plate svg { display: block }` is (0,1,1) and out-specifies
a bare class at (0,1,0), so the first version rendered both and the phone got the smear stacked on
top of the fix. `test_exactly_one_plate_is_visible_at_each_width` exists for that.

**The sweep is the only motion here that teaches something no static arrangement can.** The field's
trajectory is a *rate*, and a rate needs time to be shown in — it visibly races through 2023 and
stalls through 2018. It stops on any key or pointer event, because a reader who has started reading
an entry must not have the page move under them. Under reduced motion the control is **not built at
all**: a sweep has no terminal state, and offering a control that would do nothing is worse than
withholding it. The evidence is never withheld, only the motion.

## The timeline runs to 31 August 2026, and keeping it there is work

The catalogue is **30 mechanisms, 2014-09-01 to 2026-08-30**. Six were added in one pass after a
sweep of everything published since DroPE, and the rules that pass established:

- **Every 2026 date was verified by opening the arXiv abstract page and copying the
  submission-history line.** Research agents proposed candidates; not one date was accepted on an
  agent's word. That is not distrust of the tooling — it is that a plausible-looking arXiv id which
  resolves to a different paper is the exact failure mode here, and it has already happened once in
  this exercise's source material.
- **One date disagrees with its own identifier and we say so rather than choosing.**
  `deepseek_csa` has identifier `2606.19348` — normally June 2026 — and a submission line reading
  26 April 2026. We record the v1 line, because that is the convention everywhere else here, and
  the discrepancy is written into `source.note`.
- **A model release is not a mechanism.** GLM-5, Qwen, Gemma, ERNIE and Kimi K3 all describe their
  attention using mechanisms already on the plate. They are evidence about *adoption*, which this
  page cannot see and does not claim to.
- **The negative results are on the page.** OpenAI, Anthropic and Meta published no architecture at
  all in the window. JEPA and world models change the objective, not the attention. Both are stated
  in `limits` because a reader should know the recent end of the plate is drawn almost entirely
  from labs that publish papers.
- **Adding an entry can overturn a published claim, and twice now it has.** Top-k broke the 2018-19
  tie; the 2026 entries added a seventh pressure window and moved Well IV's end past DroPE. Every
  count and every plural on the page is derived for this reason — but a *headline* is not derived
  unless you make it so, and Well IV's headline had to be rewritten because it stated a day count
  the data no longer supported.

## The diagrams: one predicate, two resolutions, and a citation as the price of a number

`web/diagrams.js` draws all thirty from the same `pattern` block the glyphs use, through
`web/support.js` — extracted so a glyph at T=12 and a diagram at its own T cannot disagree about
what a mechanism does. Four scenes (`field`, `stack`, `state`, `bands`), no numeric literal in the
module describing any mechanism, and `web/field-guide/` is a second route over the same catalogue.

- **A size may enter the catalogue only with a citation attached.** `GLYPH_SCALES` says why there
  were none: *a glyph drawn to specific numbers would be inventing them.* `Glyph._check_sizes`
  keeps that guarantee by pricing entry — a `stated` size quotes the sentence and names where, an
  `ours` size says why, and **the quote must contain the number it is evidence for**. That last rule
  caught 512 attributed to Longformer on a quote that never says 512.
- **A silent fallback produced a plausible wrong number.** MLA is not a point on the cache sharing
  ladder, so the lookup fell through to its first row and drew MLA at "192 KiB, 1× less than keeping
  every head" — directly beneath its own credit line claiming a large cache reduction. It draws
  93.3% now, quoting its own abstract, with no fallback.
- **Form carries the semantics; colour carries only the parts.** Under `high-contrast`, `--muted`
  and `--ink` are the same `#000000`. And colour only works while there are more colours than
  meanings: four `--part-*` tokens were asked to separate six update steps and two of them collided,
  so the steps are **numbered** now — which is also more informative, because they happen in order.
- **Sourcing at scale: agents propose, the paper's own bytes dispose.** All 30 carry sizes now (80,
  78 quoted). Download every paper *first*, have agents read the local files, then check each quote
  as a contiguous run of that file's characters — 82 proposed, 82 verbatim, 0 fabrications. **Test
  the checker before trusting it**: three normalisation bugs (arXiv double-renders equations, hides
  `U+200B` inside numbers, and writes `1 M` as well as `1M`) each made it report a hand-verified
  quote as absent, and a gate with false negatives silently converts sourced numbers into "ours".
- **Verbatim is not correct.** "Figure 4: The KV cache of StreamingLLM" passes every authenticity
  check and is not evidence for four attention sinks. Ask separately whether the quote talks about
  the quantity claimed.
- **An absent number is not a zero, and a percentage that rounds to zero is not a measurement.** MSA
  published "16 of 7,813 blocks plus a **0-token window** — about **0%** of a 1,000,000-token
  context". It has a window we never sourced, and its share is 0.2% — the paper's entire claim.
- **Four defects, all found by reading a rendered screenshot with the suite green**, each now with a
  guard named after it in `tests/test_attention_diagrams.py`: a lookup table drawn as frequency
  bands, DroPE keeping two bands under a caption saying it removes them all, a figure printing its
  summary twice, and a legend whose two classes resolved to one token.
- **Two guards asked for a phrasing instead of a property** and failed correct work — one demanded
  "drawn to scale" from a figure that quotes its paper verbatim, one demanded a `THE MARKS` heading
  from eleven figures keyed by other means. Both ask the underlying question now.

## The readability pass, and the two defects it turned up that were not readability

The page was audited section by section against Sebastian Raschka's *A Visual Guide to Attention
Variants in Modern LLMs* (local-only, `docs/sessions/`) and against `AGENTS.md`'s ladder of readers.
75 findings, 37 edits. Most were wording. Two were not, and both are the kind this exercise exists
to catch.

- **A derived number can answer the wrong question, and that is harder to spot than a wrong number.**
  The verdict published *"the claimed arc holds in 6 of these 7 windows"*. `held` counted windows
  with a clear winner; the claim under test is whether the winners come in the predicted order. Six
  do decide and the order is not the claimed one, so the published verdict was the opposite of the
  truth — and it was convincing precisely because the number was real. `timeline.arc_verdict`
  compares sequences now, and its test refuses to pass if `matches` ever stops doing that.
- **The bucket edges are an arbitrary choice and nothing had varied them.** The section asserted
  its count was "not noise" with no evidence. `arc_robustness` shifts the edges by a year: the arc
  fails under both slicings and cache wins no window under either — but the claim that the field
  settles on both bills from 2020 **does not survive**, and it had been published an hour earlier.
  Corrected in place, per `AGENTS.md`: a quietly amended number is worse than the original error.

Two rules for anyone editing the page after this:

- **Every count the page presents as a partition must add up, and the guard now checks it.** The
  required list names **18 phrases but 19 mechanisms** — "sparse and top-k attention" is two
  techniques — so `counts.mandatedPhrases` and `counts.mandatedMechanisms` are both emitted and the
  page uses the right one in each place. A first draft used the phrase count against the bonus
  count and printed 29 of 30.
- **An interaction must never be the only route to a lesson, and the centrefold was breaking it.**
  All five bays had good prose in `STAGES`, and `go()` rewrote the note on every tab change, so one
  was visible at a time and four were unreachable without clicking. They are a static block now;
  the tab keeps only the arithmetic.

## Two claims that live in Python because a test must reach them

- **`story.py` holds the six chapters.** The grouping is an editorial claim, so it is tracked data
  with a guard rather than prose inside `chapters.js`. `story.check()` refuses a partition that does
  not cover the catalogue exactly once, and `build_web_data.py` calls it before emitting — both ways
  it rots are invisible on the rendered page: a mechanism in no well is simply never drawn, and one
  in two wells reads as an editing slip. Its **pull-quote guard** asserts every quote the page sets
  large is a phrase the catalogue already contains, and caught two of the six on the day it was
  written — one differing from its source only by an em dash, one invented outright.
- **`cache.tokens_before_wall()` holds the race's three crossings** — 406,901 / 1,627,604 /
  3,255,208 tokens — as the invoice's own arithmetic solved for the context instead of the bytes, so
  the figure and the table cannot disagree. `ACCELERATOR_BYTES` is decimal and the page says so;
  binary units would move every crossing by 7.4%.

## The centrefold has five stages, and the fifth is not optional

`figCentrefold` runs `Q·K → ÷√d → mask → softmax → ×V` on six real tokens with live arithmetic. An
earlier version stopped at softmax, which is precisely the step where a reader concludes attention
outputs weights. It outputs a **vector**; bay five is where that happens. `V` is deliberately not
`K` — a figure that reuses the keys as the values teaches that attention returns its own keys, the
commonest misreading of the formula there is.

**Do not carry a derived figure by searching a list.** The "days nobody touched the cost" tile once
looked its gap up out of the top-five list, which meant a new mechanism displacing it would have made
the tile silently show a *different* gap under the same label. `data.js` carries `quietStretch`
explicitly now.

## Running it

```bash
uv sync --all-packages                                    # no extras: this exercise needs no torch
uv run pytest src/exercises/08-modern-attention-variants
```

Test modules are prefixed `test_attention_*`. pytest imports by **basename**, so a second
`test_cache.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.
