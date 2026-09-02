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
  in two wells reads as an editing slip.

  **What that guard could not see, and now can.** `check()` asserts the partition and says nothing
  about whether a chapter's headline is *true of its members*. Chapter VI was headed "keep a
  fixed-size state" and promised "every one of them pays in the same single way" while holding NSA
  and DeepSeek CSA — both of which build a score grid and keep a KV cache. They select from the
  cache; they do not replace it. The chapter's own headline was false of a fifth of its members and
  nothing was red. They are in Chapter III now, which leaves VI as exactly the eight STATE entries —
  the same eight the key counts as refusing to build a grid, so the chapter and the shape are one
  object. `test_one_chapter_is_exactly_the_mechanisms_that_refuse_to_build_a_grid` asserts that
  property, with a twin that puts a FIELD mechanism back and watches it go red.

  **The pull-quote guard is gone with the pull quotes.** Six quotes set in the page's largest type,
  each attributed to "this page's own catalogue" — the visual grammar of a citation with none of its
  function, since a page quoting itself corroborates nothing. The field, the guard and its twin were
  deleted rather than left as data nothing reads; `DECISIONS.md` records why.
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

## The page was rebuilt around six readers, and two of them found factual defects

Six personas read the page end to end — a fifteen-year-old, a practising engineer, a frontier
researcher, an adversarial sceptic, an assignment grader, and a reader who had just come from
Raschka's *Visual Guide to Attention Variants*. What they changed, and the rules that came out of it.

**The page's headline claim was false, and the key's own counts had been right all along.** The
glossary ended *"Only 13 of the 30 build a score grid at all. That is the finding the rest of the
page is built on."* Thirteen is the **FIELD** count — the mechanisms that edit *which cells survive*.
RoPE and ALiBi build a grid and change what goes into it; MQA, GQA and MLA build one and change what
is kept from it. Only the eight STATE entries refuse. So the sentence conflated *edits the grid* with
*builds one*, and it was found by a reader adding up the four counts printed directly above it. It
is derived from `M.counts.glyphKinds.state` now, with a guard refusing the field count there.

**A definition belongs where the reader first meets the thing, not where the page finds it
convenient.** The glossary carried an alphabet of four shapes, a sorting into five labels and a
reference model shape — four thousand words before the first glyph is used at size and five thousand
before the first byte figure it governs. Every reader stalled in it; the teenager stopped there.
`figKey` is `figKeyShapes` (above the chronology, where thirty glyphs have to be read at once) and
`figKeyYardstick` (above the invoice, the first number it decides). The glossary is ninety-five words
defining the score grid.

**`whenToChoose` was on all thirty entries and rendered exactly once.** It lived inside the reading
spread, which shows one mechanism at a time and only after a click — so the field a reader arriving
with a decision needs was the least reachable thing on the page and no two could be compared. The
**at-a-glance table** (`chapterGlance`) is thirty rows of data the page already held. It declares no
`data-role` on purpose: the spine is twelve roles in a fixed order and 05, 06 and 07 read the same
tuple, so a thirteenth would be a repo-wide change to publish one table. It carries `data-nav`, and
`buildRail`'s selector matches both.

**A short read has to be *correct*, not merely comforted.** Every finding sat between word 6,000 and
word 8,000, so stopping early was a partial read by construction. Four tiles in the opening carry
them now — one of them a failure, as `AGENTS.md` requires — and an exit line after the chronology
says the argument ends there. `test_a_reader_who_stops_at_the_exit_line_has_read_a_complete_argument`
asserts the property rather than the wording. **It is 2,770 words of prose, not the 800 the review
estimated**; the estimate omitted the key, the plate caption, the colophon and the reading spread,
and the ceiling is set against the measured number because a ceiling set to a wrong estimate fails
honest work.

**Two numbers were unsound rather than long.** The correction figure printed `+57.3%` — three
significant figures of a difference computed against a source stating "about 1 TB". One significant
figure in, one out. And Figure 4's caption invited the reader to *"decide whether that much head
diversity was worth it"* from a figure with one axis, bytes against tokens, and no quality axis at
all.

**A caption may not claim what the drawing cannot show.** Figure 5's said the animation showed cause
"where two static curves would only show correlation". Animating a schematic does not make it causal;
the dial illustrates an assumed mechanism and this page has no measurement of a deployed model.

**Once is orientation; three times is a template readers skip.** The three figure `brief()` blocks
ran 217, 265 and 279 words in the same five-heading shape, and two readers said they were skipping
the good sentences with the boilerplate. They are 52, 50 and 59 now — and both lessons that lived
only inside a deleted block moved into a caption **first**, because `AGENTS.md` forbids leaving a
lesson only where a reader skips.

**The lexical count guard starts at *eleven*, and the defect was at four.** `next` was headed "Three
things this opens" above four items, with the rail agreeing — the repo's most expensive documented
failure, live, green. Widening the pattern would have meant marking thirty-six legitimate fixed
quantities (`six words`, `five stages`, `eight readers`) with `count-literal-ok`, and a marker on
thirty-six lines is noise nobody reads. `test_no_heading_or_rail_label_types_a_count` narrows the
scope instead: inside a heading or a rail label a spelled number is *always* a count of that
section's contents, so there it must be derived. "one" is excluded and only "one" — it is a
determiner far more often than a count.

**Four defects were found by looking at the page with the whole suite green**, which is the pattern
this repo keeps re-learning:

- The invoice's cut line was `white-space: nowrap` inside `overflow: hidden`, so the sentence
  carrying the argument read *"…needs a second ma"* at every width. `test_the_invoice_cut_line_is_visible`
  passed throughout — *visible* and *legible* are different assertions and only one was being made.
  `test_no_sentence_on_the_page_is_silently_cut_off` now asserts the general property at four widths.
- The masthead's decorative field painted its one accent mark at full opacity, and the body text sits
  on that field at every width from 1440 down by design. The bar ran through the words "every one of"
  in the opening sentence and read as a strikethrough.
- The at-a-glance head row's `display: none` was written *above* the `display: flex` it had to beat,
  at equal specificity, so it lost on source order and every phone opened the table with five
  orphaned column labels.
- The key's ~ note carried `.lab`, which is only styled *inside* `.key-alpha`; as a sibling it
  rendered at full body size, larger than every label it explained.

**On the plate/well vocabulary, the readers partly disagreed with the complaint.** All six said
"well" hindered — read as a hole in the ground, or as a section number. All six said the *numbering*
helped: one wrote *"I have already written 'Plate I' and 'Plate IV' three times in this response
without thinking about it, which is the test."* So the referent stayed and the jargon went: figures
are `Figure 1`–`Figure 6`, chapters are titled by subject, and the 83-word paragraph teaching the
vocabulary is deleted.

## Two decisions were settled by an A/B, and the harness is gone

Two things about this page could reasonably have gone either way, so rather than decide on the
reader's behalf both shipped behind a labelled switch with a tool that measured all four
combinations. PK read them on the deployed preview and chose. **The harness, its CSS, its guards,
`web/variants.js` and `tools/compare_variants.py` were then deleted** — a temporary switch with no
end date is a permanent one, and this one had its end date written into it from the first commit.

**What won, and why it is worth knowing.**

*The index carries the thirty; the chapters name them.* The alternative gave each chapter its own
full entries and made the index a receipt. It read well and it cost ~3,900px more, because it kept
both a chapter body and a source table. What the chapters get instead is a **strip**: every entry,
with its year, in date order, linked to the index. The year is the part that earns its place —
every chapter's claim is about sequence ("the three repairs", "each fixes the last one's way of
forgetting", "until the last two, which stop choosing"), and a bare list of names is no evidence
for a claim about order.

*The type is fluid, 19→22px.* The complaint was "why do you narrow too much", and the lever turned
out to be type size rather than measure: **77 characters a line at every width**, with the prose
going from 36% of a 1920px viewport to 50%, and 58% at 1440.

**The mechanism is subtler than it looks and is the reason the first attempt did nothing.** `ch`
resolves against the element's *own* computed font-size, so `#main .say { font-size: 16px;
max-width: 68ch }` pins the paragraph at 685px whatever `#main` does — growing the grid track alone
changes nothing, because the paragraph is the binding constraint. `.say` has to **inherit** the
fluid size, and then its own `ch` cap scales with it for free. The same trap bit twice more in
blocks written *after* that was understood: `.limitlist`'s `80ch` sat on the `ul`, which inherits
`#main`'s size, so the limits read at **111 characters a line** in one variant and 80 in the other.
Put the measure on the element carrying the type, every time.

**And a variant nobody measures is a variant that ships broken.** `test_attention_measures.py` drove
only the default, so the 111-character line existed for two commits with the suite green. While a
harness lives, every guard that can differ between variants has to run against both.

## Running it## Running it

```bash
uv sync --all-packages                                    # no extras: this exercise needs no torch
uv run pytest src/exercises/08-modern-attention-variants
```

Test modules are prefixed `test_attention_*`. pytest imports by **basename**, so a second
`test_cache.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.
