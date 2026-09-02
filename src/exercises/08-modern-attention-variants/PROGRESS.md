# PROGRESS — Exercise 08

A running log of what was built, what was verified, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on a branch, not yet merged. This file does not name branch or PR numbers
— `git log` and `gh pr list` answer that correctly and a markdown file goes stale.

**Deliverable shape — read this before calling the source material done.** The platform asks for a **live
app link** and the **GitHub repo**, and the README must say which sources the dates came from.
Question 1 is 1000 points for the link and repo; Question 2 is a written answer about what the
timeline shows, worth a further 1000 if it also names a mechanism the instructor missed, with a date
and a primary source; Question 3 is an optional 250 for sharing publicly. The submission field is
labelled "Netlify Link" but the requirements say "Netlify or Vercel or wherever you like" — our Vercel
pipeline is fine, and the link must resolve for a logged-out stranger.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **The catalogue** | **done** | 30 mechanisms, 2014 to Aug 2026, every date read from the primary source and cross-checked against the source's own wording. 19 required by the coverage list + 5 beyond it. |
| O2 | **The arithmetic** | **done** | The source material's 6.44 GB / 51.54 GB / 4× GQA all reproduce exactly from `cache.py`. |
| O3 | **The page** | **done** | Twelve spine sections, the two-object mechanism figure and the timeline, at `/08-modern-attention-variants/`. Registered in the landing card, `SPINE_ENFORCED` and `OPTIONAL_DEPENDENCY_GATES` in the same change. (The browser-test count this row used to carry went stale four times over; `uv run pytest src/exercises/08-modern-attention-variants -m integration` answers it correctly.) |
| O4 | **Question 2's written answer** | **ready to submit** | `artifacts/q2_answer.txt` (gitignored) is generated from `catalogue.py` and `timeline.py`, so every count, date and citation in it is derived rather than typed — regenerate it rather than editing it. **The link is live**: v0.13.0 was tagged on 2026-09-02, PK approved the production gate, and `https://llm-pretraining-demos.vercel.app/08-modern-attention-variants/` returns **200** to an anonymous request with no redirect and no login wall — `chapters.js`, `data.js`, `page-extra.css` and `/_shared/tokens.css` all 200 as well, which is the check that matters, because a page that loads while its data file 404s renders empty. Submitting is PK's: the platform takes the app link, the GitHub repo, and the written answer. |
| O5 | **A mechanism figure** | **done** | Figure 1: the causal score triangle beside the KV-cache column, with eight variants as predicates rather than pictures. Three browser tests make it falsifiable — switching must change the drawing, GQA must touch no score, linear attention must leave no per-position square. |
| O10 | **Sourced sizes for every mechanism** | **done** | 80 sizes, 78 quoted verbatim from the primary paper. Agents proposed, a mechanical substring check against the downloaded text disposed: 82 proposed, 82 verbatim, 0 fabrications. |
| O11 | **Readability pass to a named benchmark** | **done** | Six-agent audit against Raschka's visual guide and the ladder-of-readers rubric: 75 findings, 37 edits, all applied. Six themes × two widths screenshotted, clean console throughout. |
| O12 | **Adoption — which models ship which mechanism** | **done** | 21 records across 8 models, every arXiv id found via the search API rather than recalled, every quote gated as a substring of the paper. 22 of 30 deliberately empty. |
| O13 | **The arc verdict, and its noise floor** | **done** | The published claim measured the wrong thing; `arc_verdict` compares sequences and `arc_robustness` varies the bucket edges. One finding was lost to the noise floor and corrected in place. |
| O7 | **A diagram per mechanism** | **done** | Thirty, four scenes, generated from the `pattern` block each catalogue entry already carried. Sourced sizes carry a citation as the price of entry. |
| O8 | **The field guide** | **done** | `/08-modern-attention-variants/field-guide/` — all thirty at once, filters derived from the data, deep links both ways. No build change needed. |
| O9 | **A theme test** | **done** | Six themes × render, tokens, text contrast, mark separation. The first in the repo; it closed a gap that predates this exercise. |
| O17 | **The A/B: two readings of the same page** | **decided, harness removed** | A reader rejected the page for duplication, apparatus dressed as argument, and a prose column using a third of the screen. Two of the fixes could reasonably go either way, so both shipped behind a labelled switch. PK read them on the preview and chose **the index, and the large type**. The losing branch, the switch, `variants.js`, `compare_variants.py` and the variant guards are deleted. |
| O18 | **The rail marks where you are** | **done** | "I just get lost on the page without knowing where I am reading from." The vendored stylesheet has styled `.rail-link.on` since before this page existed and this page never set the class — nor do 05, 06 or 07. Exercise 03's logic, copied. Plus a derived read-time. |
| O19 | **Four blocks written to be wide that silently were not** | **done** | The invoice carried a `.bleed` class no rule ever matched and rendered at 685px at every width since it was written; the colophon's `min(1025px, 100%)` never applied because nothing put it in the `wide` track; the reading spread's ledger read at 36 characters at 1920 — the page's narrowest prose produced by its widest screen; Q/K/V read at 23. |
| O14 | **Six-persona rebuild for readability** | **done** | A teenager, an engineer, a researcher, a sceptic, a grader and a Raschka reader read the page end to end. Two found factual defects. Five changes applied: the borrowed plate/well vocabulary removed, ~900 words cut and 190 moved to `docs/METHOD.md`, the key split to where the glyphs and the byte figures are first used, an at-a-glance table of all thirty, and an exit line after the chronology. |
| O15 | **The state chapter held two mechanisms that keep a cache** | **done** | Chapter VI promised "a fixed-size state" and "every one of them pays in the same single way" while holding NSA and DeepSeek CSA, both of which build a score grid and keep a KV cache. Moved to Chapter III; VI is now exactly the eight STATE entries. A guard asserts that property with a broken twin. |
| O16 | **Five reader-facing defects with a green suite** | **done** | The invoice's cut line truncated mid-word at every width; the masthead's accent bar struck through the opening sentence; the table's column heads survived on phones because `display:none` lost on source order; the key's ~ note rendered at body size; and the page claimed "almost every mechanism" attacks a bill when ten of thirty attack neither. Each found by looking; three new guards, each watched failing. |
| O6 | **The notebook** | **done** | `notebooks/S08-modern-attention-variants.ipynb`, 24 cells, built by a 314-line builder that imports `attention.*` in six code cells rather than re-implementing anything. Outputs stripped. `tests/test_notebook_builders.py` passes locally — the only place it can, since both files are gitignored. |

---

## Findings

**The instructor's tidy arc is not what the data shows, and that is the interesting part.** The
requirements document predicts "exactness → memory → length → memory again". Deriving the dominant pressure per
two-year window gives something messier: **two of the six windows have no single dominant pressure
at all** (2018–19 and 2022–23). In those periods the field was attacking compute, cache and position
simultaneously. `timeline.Period.dominant` returns `None` on a tie rather than picking a winner, and
a test fails if the ties ever disappear — so the finding cannot quietly relax into the tidy story.

**Attention is three years older than the Transformer.** Bahdanau's soft alignment is 2014-09-01;
*Attention Is All You Need* is 2017-06-12. The 2017 paper removed the recurrence around attention
rather than inventing it. Ordering by date makes this obvious; the teaching order hides it.

**Learned absolute positions predate the Transformer by five weeks.** 2017-05-08 against 2017-06-12
— and the ConvS2S paper is the source *Attention Is All You Need* itself cites for them.

**Nobody attacked the cost for 680 days.** Between the Transformer and Sparse Transformers there is
a stretch of nearly two years in which the field used attention without trying to make it cheaper.
The longest gap on the whole timeline is longer still: 980 days, from Bahdanau to learned positions.

**Attention sinks predate Mistral 7B by eleven days.** 2023-09-29 against 2023-10-10, which reverses
the usual telling of that period.

**NTK-aware scaling has no paper.** It is a Reddit post by `bloc97`, dated by the platform's own
timestamp. reddit.com refused our requests, so the field was read from a Wayback capture — recorded
in the entry, because a reader who needs the live page needs a browser.

---

## Corrections — errors found in the course material

The requirements invites these: *"if you catch me in another one, tell me."*

**The transformer is mis-dated in the source.** It says Vaswani "invented in 2018 and 17".
*Attention Is All You Need* is `arXiv:1706.03762`, v1 **Mon, 12 Jun 2017**, read from the abstract
page.

**DroPE is two papers, and the source quotes the wrong one's title.** The technique taught —
pretrain with positional embeddings, drop them, recalibrate briefly — is *Extending the Context of
Pretrained LLMs by Dropping Their Positional Embeddings*, `arXiv:2512.12167` (Sakana AI, v1 13 Dec
2025). The source's garbled *"rotate position emitting for efficient"* maps instead onto
**DRoPE** with a capital R, `arXiv:2503.15029`, *Directional Rotary Position Embedding for Efficient
Agent Interaction Modeling* — an autonomous-driving trajectory paper. Two papers, one capital
letter apart. Both are recorded so nobody "corrects" us back to the wrong one.

**A cache figure does not reproduce.** The source says eight users at 1M tokens need about
1 TB; the source material's own formula at the source material's own yardstick gives **1.57 TB**. Both are recorded.
A smaller model, fewer KV heads or fp8 would each reconcile them and the source does not say
which was meant — so neither number is published alone.

---

## Change log

### 2026-09-02 (the A/B decided, and the harness retired)

PK read both variants on the deployed preview and chose **the index, and the large type**. The
harness came out in the same pass — `web/variants.js`, the head bootstrap, the switch, the
`story = b` branch, `tools/compare_variants.py` and `tests/test_attention_variants.py`. A temporary
switch with no stated end date is a permanent one; this one carried its end date from its first
commit and it was honoured.

**What the chapters get instead of full entries.** Three of the six were a heading and nothing else,
so each now carries a strip: every entry, with its **year**, in date order, linked to the index. PK
proposed names alone; the year is the addition, because every chapter's claim is about sequence and
a bare list is no evidence for a claim about order.

**Where the page ended up**, measured, against the 29,999px / 33.3-screen baseline this pass began
from — and note the type is 19% to 38% larger at every one of these widths:

| viewport | height | prose | share | body type |
| --- | --- | --- | --- | --- |
| 2560 | 30,321px | 951px | 37% | 22px |
| 1920 | 29,911px | 951px | **50%** | 22px |
| 1440 | 28,508px | 835px | **58%** | 19px |
| 1180 | 29,943px | 835px | 71% | 19px |

The page is shorter than it started **while carrying larger type, six chapter strips and a new
four-families figure** — the duplication that came out paid for all three. It is not the 18,000px
the plan hoped for, and the reason is stated plainly: thirty entries of catalogue prose have a
floor, and PK's instruction was to keep the facts.

### 2026-09-02 (the readability rebuild, and an A/B)

PK rejected the deployed page: duplication ("Every mechanism, one line each" restated "The
index"), apparatus given display headings, repetitive chapter openers, no sense of place,
"why do you narrow too much", "the length of the page is too much", and — the one that
mattered — *"you are eating my time fixing just the UI not the actual content, storytelling
and experience."*

**Measured before anything changed.** 29,999px, 33.3 screens, prose 27–36% of the viewport
above 1600px. The index was 30 rows at 306px, and widening its plate from 720px to 1,676px
moved a row only to 292px — because a row was six stacked bands on a four-column grid, so the
extra width shortened lines that were already short. **The narrowness was the length.**

**Three chapters had no bodies.** I, II and VI were a heading and nothing else; III, IV and V
a heading plus one figure. The thirty mechanisms they are chapters *about* were named in none
of them — the duplication one level below the table PK named.

**Two decisions ship as an A/B**, because PK asked for evidence rather than my judgement.
`uv run python tools/compare_variants.py` prints this, read off the rendered page:

```

                                  story a / type a  story a / type b  story b / type a  story b / type b
  ------------------------------------------------------------------------------------------------------

  at 2560px
  page height                             27,948px          28,437px          28,193px          28,490px
  screens                                     31.1              31.6              31.3              31.7
  prose width                          685px (27%)       951px (37%)       685px (27%)       951px (37%)
  characters a line                             74                77                74                77
  body type                                   16px              22px              16px              22px

  at 1920px
  page height                             27,749px          28,027px          27,932px          28,035px
  screens                                     30.8              31.1              31.0              31.1
  prose width                          685px (36%)       951px (50%)       685px (36%)       951px (50%)
  characters a line                             74                77                74                77
  body type                                   16px              22px              16px              22px

  at 1440px
  page height                             27,819px          26,042px          28,081px          27,200px
  screens                                     30.9              28.9              31.2              30.2
  prose width                          685px (48%)       769px (53%)       685px (48%)       769px (53%)
  characters a line                             74                77                74                77
  body type                                   16px           17.46px              16px           17.46px

  structure (width-independent)
  rendered words                             8,946             8,946             9,147             9,147
  entries rendered                              30                30                30                30
  containers holding them                        1                 1                 6                 6
  chapters with a body                      0 of 6            0 of 6            6 of 6            6 of 6
  transferred                               460 KB            460 KB            460 KB            460 KB

  where the height is, at 1920px
    thesis                                 1,601px           1,574px           1,601px           1,574px
    glossary                                 484px             441px             484px             441px
    problem                                1,511px           1,479px           1,511px           1,479px
    mechanism                              2,567px           2,608px           2,567px           2,608px
    method                                   312px             324px             312px             324px
    expected                                 529px             600px             529px             600px
    results                                9,361px           9,510px          13,999px          13,953px
    negatives                                872px             872px             872px             872px
    conclusion                             1,390px           1,438px           1,390px           1,438px
    limits                                   441px             398px             441px             398px
    next                                     493px             580px             493px             580px
    reproduce                              7,878px           7,892px           3,423px           3,457px

  Baseline before this pass: 29,999px / 33.3 screens, prose 685px (36% at 1920),
  74 characters a line, thirty entries in TWO containers — the duplication a reader
  found — and 3 of 6 chapters with a body.
```

The one thing the table does not measure is the thing PK will judge: whether meeting a
mechanism inside the chapter that argues for it beats meeting it in one table at the back.
That is why it is a switch and not a commit.


### 2026-09-02 (the six-persona rebuild)

Six readers with different stakes read the page end to end. Two of them found factual defects, which
is the part that matters more than any styling.

- **The page's central claim was false.** The key ended *"Only 13 of the 30 build a score grid at
  all. That is the finding the rest of the page is built on."* Thirteen is the FIELD count — the
  mechanisms that edit *which cells survive*. Position schemes build a grid and change what goes
  into it; head-sharing schemes build one and change what is kept from it. Only the eight STATE
  entries refuse. Found by a reader adding up the four counts printed directly above the sentence.
- **The state chapter held two mechanisms that keep a cache.** NSA and DeepSeek CSA are `field`, not
  `state`; the chapter's headline was false of a fifth of its members. They moved to Chapter III,
  leaving Chapter VI as exactly the eight the key counts.
- **A caption had gone stale against the data beside it** — *"1,698 days of repair work, and the
  last repair was to delete it"*, when that chapter's last entry is HD-RoPE, 260 days later, arguing
  the opposite. Both numbers derived now.
- **`whenToChoose` was on all thirty entries and rendered exactly once.** The at-a-glance table is
  thirty rows of data the page already held, arranged so a reader can compare. It declares no
  `data-role`, so the twelve-part spine stays twelve.
- **The key split to where the glyphs and the byte figures are first used**; the glossary is
  ninety-five words. Every reader stalled in the old one; the teenager stopped there.
- **Four finding tiles and an exit line** — every finding used to sit between word 6,000 and word
  8,000, so stopping early was a partial read by construction. Prose to the exit line is 2,770
  words, not the 800 the review estimated; the estimate omitted the key, the plate caption, the
  colophon and the reading spread.
- **~900 words cut** and 190 moved to `docs/METHOD.md`; the correction that was true and obvious
  moved to `DECISIONS.md`. The three figure orientation blocks went from 217/265/279 words to
  52/50/59, with both transferable lessons moved into captions *before* their blocks were deleted.
- **Two numbers were unsound rather than long** — `+57.3%` computed against a source stating "about
  1 TB", and a caption inviting a quality judgement from a figure with no quality axis.
- **Five reader-facing defects, all with a green suite**, all found by looking: the invoice's cut
  line truncated mid-word; the masthead's accent bar struck through the opening sentence; the
  table's column heads survived on phones because `display: none` lost on source order; the key's ~
  note rendered at body size; and three separate sentences claimed "almost every mechanism" attacks
  a bill when ten of thirty attack neither.
- **Five new guards, each watched failing**: one chapter is exactly the STATE family · no sentence
  is clipped by its own box · no heading or rail label types a count · the glance table carries
  every mechanism with `whenToChoose` reaching the reader · a reader stopping at the exit line has
  the complete argument.

### 2026-09-01 (the page)

- **Twelve spine sections and the timeline**, at `/08-modern-attention-variants/`. Every date,
  trade-off and citation is rendered from `web/data.js`, which `tools/build_web_data.py` derives from
  the catalogue and from the same functions the tests exercise — so the page cannot disagree with the
  evidence, and the derived findings cannot disagree with the code.
- **Figure 1 draws the framing the source material never states**: attention has exactly two objects that
  cost anything, and every mechanism is a structural edit to one of them. Eight variants, each a
  predicate rather than a drawing.
- **One latent bug caught before it shipped.** The "days nobody touched the cost" tile looked its gap
  up out of the top-five list, so a new mechanism displacing it would have made the tile show a
  *different* gap under the same label — a wrong number reading as a correct sentence. `data.js`
  carries `quietStretch` explicitly now.
- The mechanism figure's first render left a quarter of its box empty below the content; the viewBox
  is tightened to fit.

### 2026-09-01 (scaffold + the chronology)

- Exercise scaffolded to the repo skeleton: `README.md`, `CLAUDE.md`, `NOTICE`, `PROGRESS.md`,
  `DECISIONS.md`, `pyproject.toml`, `src/attention/` (five modules), `tests/` (four modules),
  `results/mechanisms.json`.
- **No torch.** Nothing here trains, so the exercise is fully verified by CI's default sync rather
  than needing the `train` extra and a separate job.
- 30 mechanisms catalogued, every date verified against its primary source and cross-checked
  against the source's own quoted wording.
- Registered in the `rest` integration shard and the root README table. **Not** registered in
  `deploy/vercel/index.html` or `SPINE_ENFORCED` — both guards fail in *both* directions, so an
  entry without a `web/` directory is as red as a missing one.
- Three guards watched failing on a deliberately broken catalogue before being committed: a dropped
  mandated mechanism, a transposed date (`2021-04-20` → `2021-04-02`), and a stripped source URL.

---

## The page, rebuilt (this pass)

The first page shipped every graded item and was rejected on sight: text and tables, no explainers,
no graphics, misaligned. Rebuilt as a monograph feature — six numbered plates, six chapters, and the
23 as one object entered 23 times rather than 23 collapsed cards.

- **Three views, two of which need no interaction.** Plate III (all of them on real time, one stave
  per bill, `both` drawn as ties), the reading spread (one entry in depth), the index plate (all, same
  six fields, same six places). A grader must not click 23 times, and a print must still carry it.
- **A glyph alphabet.** Four generators read a `pattern` block now carried by every catalogue
  entries, so a glyph is derived rather than drawn. Rendering the sheet at 160px is what exposed
  RoPE and ALiBi printing as the same gradient (a `min(.., π/2)` clamp had flattened RoPE's
  oscillation into ALiBi's ramp) and Reformer printing as a sliding window.
- **The centrefold gained its fifth stage.** It stopped at softmax, which is exactly where a reader
  concludes attention outputs weights.
- **No shell commands on the page**, and `method` moved to a colophon — kept in its canonical spine
  position, because a repo-wide order matters more than this page's instinct to put production notes
  last.
- **Six defects found by screenshotting, all with a green suite**, now each with a named guard:
  invisible verdict chips, an invoice cut line revealed by an observer on a detached node, plate
  labels laddered to a fixed 48px, every glyph escaping its viewBox, the masthead scrolling a 320px
  screen sideways, and a guard that scrolled the element into view before measuring whether it was
  visible without scrolling.
- Read in all six themes: no console error, no sideways scroll at 320–1440px, foreground and ground
  resolving as a pair in each.

---

## Completing the source material (this pass)

- **A mandated mechanism was missing and the guard could not see it.** The coverage list says
  *"sparse and top-k attention"*; we had catalogued the sparse half and the Sparse Transformers
  entry additionally claimed "top-k attention" as an alias, so the catalogue asserted the two were
  the same technique. Top-k is now its own entry (2019-12-25, arXiv:1912.11637, v1 date read from
  the abstract page), and `MANDATED` maps a phrase to *every* key it names so a compound
  requirement cannot be satisfied by half of itself again.
- **That overturned a published claim, and it is corrected in the open.** The 2018–19 window had
  been an exact 1–1 tie; a second 2019 compute entry breaks it. Two undecided windows of six became
  one. The page's body text corrected itself — the headline and the rail subtitle did not, because
  they were hand-written. Both are derived now, along with every plural in that section.
- **Every reader-facing count is derived.** The page spelled "twenty-three" in six places; one new
  mechanism made all six wrong at once. A lexical guard now forbids a spelled count as a source
  literal in `web/*.js`.
- **The plate has a portrait form.** The landscape plate is unreadable in a 342px column, which
  meant the page's centrepiece carried nothing on a phone. Time now runs down the page below 720px.
- **The plate can be read as one motion.** A playhead sweeps the whole chronology, lighting each
  entry as it passes and advancing the reading spread — visibly racing through 2023 and stalling
  through 2018. Withheld entirely under reduced motion, because a sweep has no terminal state.
- **The notebook covers the shipped code again**, including `story.py`, `tokens_before_wall` and the
  glyph alphabet. All eleven code cells were executed against the package rather than assumed to run.
- **The Q2 deliverable is written down**: five findings visible only on a date axis, the five
  mechanisms beyond the coverage list with their dates and sources, and the one we had wrong.

---

## Carried to the frontier (this pass)

- **Six mechanisms added, taking the timeline to 31 August 2026**: Kimi Delta Attention
  (2025-10-30), Mamba-3, DeepSeek-V4's compressed sparse attention, Gated DeltaNet-2, MiniMax
  sparse attention, and higher-dimensional RoPE (2026-08-30). Every arXiv abstract page opened and
  its submission-history line copied by hand; no date taken on a research agent's word.
- **KDA was a gap, not an extension.** Kimi Linear is dated October 2025 — it predates DroPE and
  should always have been on the plate.
- **The position lane now ends on a contradiction.** DroPE concludes positional embeddings should
  be deleted; HD-RoPE, eight months later, concludes they should be made richer. Both report gains.
  Well IV's headline was rewritten because it stated a day count that the new entry invalidated.
- **Negative results recorded as results.** OpenAI, Anthropic and Meta published no architecture in
  the window. GLM-5, Qwen, Gemma, ERNIE and Kimi K3 use mechanisms already on the plate. JEPA and
  world models change the objective, not the attention. Gnani.ai has published no mechanism at all.
- **Two page bugs found by PK that no test caught**: the sweep control threw on every click because
  the plate wrapper forwarded `select` and not `sweep`, and Plate V had no replay control.
- **Neutral voice.** Every word tying the page to a particular class or requirement is gone from the
  page, the served NOTICE and the meta description.

---

## Verification

```bash
uv sync --all-packages
uv run pytest src/exercises/08-modern-attention-variants
uv run ruff check . && uv run ruff format --check .
```
