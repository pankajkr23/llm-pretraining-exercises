# PROGRESS — Session 8

A running log of what was built, what was verified, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on a branch, not yet merged. This file does not name branch or PR numbers
— `git log` and `gh pr list` answer that correctly and a markdown file goes stale.

**Deliverable shape — read this before calling the session done.** The platform asks for a **live
app link** and the **GitHub repo**, and the README must say which sources the dates came from.
Question 1 is 1000 points for the link and repo; Question 2 is a written answer about what the
timeline shows, worth a further 1000 if it also names a mechanism the instructor missed, with a date
and a primary source; Question 3 is an optional 250 for sharing publicly. The submission field is
labelled "Netlify Link" but the brief says "Netlify or Vercel or wherever you like" — our Vercel
pipeline is fine, and the link must resolve for a logged-out stranger.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **The catalogue** | **done** | 30 mechanisms, 2014 to Aug 2026, every date read from the primary source and cross-checked against the source's own wording. 19 required by the coverage list + 5 beyond it. |
| O2 | **The arithmetic** | **done** | The session's 6.44 GB / 51.54 GB / 4× GQA all reproduce exactly from `cache.py`. |
| O3 | **The page** | **done** | Twelve spine sections, the two-object mechanism figure and the timeline, at `/08-modern-attention-variants/`. Registered in the landing card, `SPINE_ENFORCED` and `OPTIONAL_DEPENDENCY_GATES` in the same change. 20 browser tests. |
| O4 | **Question 2's written answer** | **drafted, blocked on the deploy** | `artifacts/q2_answer.txt` (gitignored) is generated from `catalogue.py` and `timeline.py`, so every count, date and citation in it is derived rather than typed — regenerate it rather than editing it. **It cannot be submitted yet**: the app link `/08-modern-attention-variants/` returned **404** on 2026-09-02, because the page is on an unmerged branch and production is gated. Question 1 asks for a public URL and a 404 fails it outright, so the order is merge → run the production workflow → re-check the link logged out → submit. |
| O5 | **A mechanism figure** | **done** | Figure 1: the causal score triangle beside the KV-cache column, with eight variants as predicates rather than pictures. Three browser tests make it falsifiable — switching must change the drawing, GQA must touch no score, linear attention must leave no per-position square. |
| O10 | **Sourced sizes for every mechanism** | **done** | 80 sizes, 78 quoted verbatim from the primary paper. Agents proposed, a mechanical substring check against the downloaded text disposed: 82 proposed, 82 verbatim, 0 fabrications. |
| O11 | **Readability pass to a named benchmark** | **done** | Six-agent audit against Raschka's visual guide and the ladder-of-readers rubric: 75 findings, 37 edits, all applied. Six themes × two widths screenshotted, clean console throughout. |
| O12 | **Adoption — which models ship which mechanism** | **done** | 21 records across 8 models, every arXiv id found via the search API rather than recalled, every quote gated as a substring of the paper. 22 of 30 deliberately empty. |
| O13 | **The arc verdict, and its noise floor** | **done** | The published claim measured the wrong thing; `arc_verdict` compares sequences and `arc_robustness` varies the bucket edges. One finding was lost to the noise floor and corrected in place. |
| O7 | **A diagram per mechanism** | **done** | Thirty, four scenes, generated from the `pattern` block each catalogue entry already carried. Sourced sizes carry a citation as the price of entry. |
| O8 | **The field guide** | **done** | `/08-modern-attention-variants/field-guide/` — all thirty at once, filters derived from the data, deep links both ways. No build change needed. |
| O9 | **A theme test** | **done** | Six themes × render, tokens, text contrast, mark separation. The first in the repo; it closed a gap that predates this exercise. |
| O6 | **The notebook** | **done** | `notebooks/S08-modern-attention-variants.ipynb`, 24 cells, built by a 314-line builder that imports `attention.*` in six code cells rather than re-implementing anything. Outputs stripped. `tests/test_notebook_builders.py` passes locally — the only place it can, since both files are gitignored. |

---

## Findings

**The instructor's tidy arc is not what the data shows, and that is the interesting part.** The
brief predicts "exactness → memory → length → memory again". Deriving the dominant pressure per
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

The assignment invites these: *"if you catch me in another one, tell me."*

**The transformer is mis-dated in the transcript.** It says Vaswani "invented in 2018 and 17".
*Attention Is All You Need* is `arXiv:1706.03762`, v1 **Mon, 12 Jun 2017**, read from the abstract
page.

**DroPE is two papers, and the transcript quotes the wrong one's title.** The technique taught —
pretrain with positional embeddings, drop them, recalibrate briefly — is *Extending the Context of
Pretrained LLMs by Dropping Their Positional Embeddings*, `arXiv:2512.12167` (Sakana AI, v1 13 Dec
2025). The transcript's garbled *"rotate position emitting for efficient"* maps instead onto
**DRoPE** with a capital R, `arXiv:2503.15029`, *Directional Rotary Position Embedding for Efficient
Agent Interaction Modeling* — an autonomous-driving trajectory paper. Two papers, one capital
letter apart. Both are recorded so nobody "corrects" us back to the wrong one.

**A cache figure does not reproduce.** The transcript says eight users at 1M tokens need about
1 TB; the session's own formula at the session's own yardstick gives **1.57 TB**. Both are recorded.
A smaller model, fewer KV heads or fp8 would each reconcile them and the transcript does not say
which was meant — so neither number is published alone.

---

## Change log

### 2026-09-01 (the page)

- **Twelve spine sections and the timeline**, at `/08-modern-attention-variants/`. Every date,
  trade-off and citation is rendered from `web/data.js`, which `tools/build_web_data.py` derives from
  the catalogue and from the same functions the tests exercise — so the page cannot disagree with the
  evidence, and the derived findings cannot disagree with the code.
- **Figure 1 draws the framing the session never states**: attention has exactly two objects that
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

## Completing the session (this pass)

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
- **Neutral voice.** Every word tying the page to a particular class or assignment is gone from the
  page, the served NOTICE and the meta description.

---

## Verification

```bash
uv sync --all-packages
uv run pytest src/exercises/08-modern-attention-variants
uv run ruff check . && uv run ruff format --check .
```
