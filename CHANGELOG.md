# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Record user-facing changes under `[Unreleased]` as they land; on release, rename that
section to the new version with a date and open a fresh `[Unreleased]`.

## [Unreleased]

### Removed

- **The programme's schedule and internal authoring specs no longer ship on the remote.**
  `docs/BRIEF.md` (course structure, 20-class syllabus, capstone staffing) and the two explainer
  authoring specs — `docs/EXPLAINER_PROMPT.md`, `docs/EXPLAINER_PATTERN.md` — are untracked and
  gitignored, and the root README's "About the program" section — duration, session times, format,
  capstone — is gone with them. The files are unchanged on a working checkout; they simply stop
  being published. This repo carries the engineering work; the course calendar is not part of it.
  Every tracked link to them has been removed, because a link to an untracked file is dead for
  everyone but us.
- **`DESIGN_CRITIQUE.md` is local too.** Exercise 03's self-critique of its own first build is a
  useful record for whoever works on it next, but it is not part of the published work — untracked,
  gitignored, unchanged on disk, and no longer linked from the root README or the exercise brief.

## [0.3.0] - 2026-08-07

Exercise 02 — the multilingual tokenizer, measured properly: the reference recipe reproduced to
the last digit and then beaten on both evenness and compression, with a one-page explainer that
lets a reader work out for themselves why the biggest number on the page is not the submitted
one. The original word-denominated experiments are retained in full as a second profile rather
than overwritten, and the widget can now tokenize text you paste into it.

### Added

- **Exercise 02 now carries two evaluation profiles, both retained and neither deprecated.**
  **v1** is the original work — clipped article prose, scored in tokens per whitespace word, no
  Hindi penalty, en/hi/te/ta — and **v2** is the measurement the assignment grades — wiki-faithful
  Markdown, faithful units, Hindi penalty, en/hi/te/mai. Their scores can never be ranked against
  each other: the same tokenizer reads ≈ 2.13 under v1 and ≈ 0.60 under v2. `ablate.sweep` raises
  if handed rows from both, and the widget renders one labelled section per profile with the
  non-comparability stated in the copy.
- **v1 is committed and still runnable, not remembered.** Its corpus ships in `corpus/v1/`, and
  `tests/test_v1_retained.py` regenerates its four published scores — 2077.90 / 1300.12 / 1228.34
  / 189.59 — on every run. That guard earns its keep because the two profiles share an engine:
  training from files, `[UNK]`, `min_frequency=1` and Metaspace `prepend_scheme="never"` are all
  v2 decisions, and every one of them moves v1's numbers, so `ablate._v1` pins all four rather
  than inheriting them.

- **A one-page explainer at [`/02-tokenization/how-it-works.html`](src/exercises/02-tokenization/web/how-it-works.html)**, so the tokenizer page stays a tool and the argument gets its own room. Three figures. Fig. 0
  shows the corpus: English's article is 32× Maithili's, and a weight of `×3` means that article
  is fed to the trainer three times — taking Maithili from 1.1% to 1.6% of what it reads. Fig. 1
  is a dial over ten real training runs: the score peaks and falls while total tokens climb the
  whole way, so a reader can watch evenness being bought with compression. Fig. 2 lets them change
  which fifth of the corpus is held back and see the ranking fail to hold still. It closes with
  what was tested and how each result was checked. Every point is a measured run — 45 of them
  behind the figures — never an interpolation, and the landing page links to it from the top.
- **A submitted tokenizer that beats the reference on both axes**: score 6,503 → **11,250.51** with
  *fewer* total tokens (191,266 → 189,785). Two independent changes — train on documents, and raise
  Maithili's weight from 2 to 3 (it is 1.8% of the corpus and shares Devanagari with Hindi, so it
  won almost no merges of its own and sat at the worst fertility).
- **A held-out check that reports its own failure.** `tokenization.holdout` trains on 80% of each
  article and scores the 20% never seen — across all five possible slices. Held out five different
  ways, one recipe's score swings **9,421 points** while the three recipes' averages sit **648
  apart**: the noise is more than ten times the difference being measured, so this test cannot rank
  them. That is reported as the finding, including the inconvenient half — on those averages the
  rejected recipe is slightly ahead. What rules that recipe out is total tokens (192,713 against
  189,785), measured on the whole corpus, which does not move.
- **Corpus-wide fertility reported beside every score.** `1000 / (X_max − X_min)` is maximised by
  making every language equally mediocre, and the published Hindi penalty only fires above X = 1.2
  while everything on this corpus sits near 0.6 — so the anti-exploit device is inert. Total
  tokens / total units is the counterweight that makes flattening visible.
- **The widget can now actually tokenize.** `web/data.json` carries the **ordered merges**, not
  just the vocabulary, and `web/encoder.js` replays them in the browser: paste any text and watch
  it split, with out-of-vocabulary characters rendered as a visible `[UNK]` chip instead of being
  silently dropped. Plus a download button for the vocab + merges. A vocabulary list on its own
  cannot reproduce a score.
- **Python↔JavaScript parity is tested, not hoped for.** `tests/test_js_encoder.py` runs corpus
  lines through both implementations under `node` and requires identical token streams — including
  a line containing a literal `_`, which must never be confused with the `▁` metaspace marker.
- **The faithfulness rule as executable checks**, run against all four real articles: round-trip
  every visible character (baseline post-NFKC, since NFKC genuinely rewrites `″`→`′′` and friends),
  assert zero `[UNK]`, and assert the corpus contains no raw `U+2581`. Each invariant is also run
  against a deliberately broken tokenizer, so every guard is known to be able to fail.
- **A fourth-language comparison.** Tamil fetched with the same pipeline and compared against
  Maithili. The scores are not comparable — different corpora — but the structure is the finding:
  Maithili is 5,808 units in a script Hindi already pays for, Tamil is 188,367 units (larger than
  English) in a script nothing else uses, and swapping them moves which language is starved.
- **The rejected experiment is on the page, labelled as rejected.** One configuration scores
  35,604 — more than three times the submission — and the widget was quietly showing only the
  submission. A reader had no way to know a bigger number was found and turned down, which is the
  most interesting decision in the exercise. It now appears badged `rejected`, saying in its own
  words that it reaches its evenness by making English and Hindi *worse*, and needs 192,713 tokens
  for the same corpus against the submission's 189,785.
- **Every tokenizer on the page explains itself** in three lines — what was changed, why it was
  worth trying, what came of it — with a badge marking it as the reference, the submission, a
  rejected experiment or an ablation. The render test fails if a config reaches the page without
  an explanation: a row of numbers with no story is not a finding.
- **The submitted tokenizer ships in the bundle**, in HuggingFace's own format at
  `web/tokenizer.json` — `artifacts/` is gitignored by design, and the assignment asks for the exact
  file. `tests/test_submission.py` trains nothing: it loads that file, scores it on the committed
  corpus, and asserts the figures the README prints, so the artifact and the documentation cannot
  drift apart unnoticed. It also pins the corpus unit counts, making a re-fetched snapshot loud.
- **The widget is loaded in a browser, not just parsed** (`tests/test_widget_render.py`,
  integration-marked, Playwright). `node --check` cannot tell you the page imports its encoder, that
  the import resolves where it is served from, or that a handler calls a function that exists — all
  valid syntax, all a blank panel.

### Changed

- **Exercise 02's graded numbers now come from the corpus and denominator the assignment
  specifies.** The submission is trained and measured on committed **wiki-faithful Markdown** —
  Wikipedia's REST HTML with its links, URLs, tables and categories intact — against a
  **faithful-unit** denominator, with the Hindi penalty and adjusted score computed throughout.
  Both corpora ship in the repo, so a fresh clone reproduces every published figure with the
  network switched off.
- **The reference recipe is now a correctness gate, reproduced to the last digit** — tokens
  111,390 / 51,190 / 24,428 / 4,258, spread 0.153786, score 6502.56 — and it runs as the first row
  of the ablation suite. Reaching it turned up a detail invisible in any config: HuggingFace splits
  *files* into lines, so training from files means no merge may span a newline. Handing the same
  trainer whole documents lowers every token count by ~0.6% and lifts the score to 6771. Same
  recipe, different number; it is now an explicit `Spec` field rather than an accident, and there
  is deliberately only one trainer in the package.

### Fixed

- **Exercise 02's README and BRIEF no longer describe a denominator the code stopped using**, and
  the stale "connecting Vercel is the remaining one-time step" note is gone from both the exercise
  and the root README — the site has been live for some time.
- **The tokenizer page no longer scrolls sideways on a phone.** Its two-column grid used a bare
  `1fr` track, which refuses to shrink below its content's min-content width, so one long
  unbreakable string pushed the whole page 18px wider than a 390px viewport. Tables now scroll
  inside their own container instead of widening the page.
- **The fertility table no longer reshuffles between tokenizers.** It was sorted by fertility, so
  the four languages appeared in a different order on every tab and the `X1…X4` tags attached to
  whichever language happened to sit in that slot. Two tokenizers measured on identical languages
  looked like they had been measured on different ones. Languages now appear in a fixed order on
  every tab, with best and worst flagged instead of implied by position, and each non-benchmark
  row carries a **vs benchmark** column showing exactly which languages it improved and which it
  paid for — the rejected configuration's evenness is visibly bought by making English and Hindi
  worse. A test asserts every tab in a section lists the same languages in the same order.
- **Internal jargon removed from the public page.** Experiments were labelled with the codes they
  carry in the sweep — `reference recipe (gate)`, `E2b · te ×6 · mai ×7` — which mean nothing to a
  reader arriving cold. They now read `the reference solution (benchmark)` and `more Telugu +
  Maithili (rejected)`, and the README table matches the labels the code actually emits.

## [0.2.0] - 2026-08-06

Exercise 03 — the data-collection framework: a graded catalogue of 145 datasets behind one
interactive page that works out what an India-first 40B model would actually train on, and finds
that four datasets are committable today.

### Added

- **CI now renders the page, not just parses it.** Every check in the pipeline read the bundle or
  the syntax; nothing loaded the site, and the two worst bugs this project shipped both lived in
  exactly that gap — a containment subtraction that was correct in `data.json` and silently never
  fired in the browser, and a headline reading "0 of 55" that was true of a question nobody meant
  to ask. `node --check` caught neither, because both files parsed perfectly. A Playwright suite
  (integration-marked, skipped when no browser is present) now loads the built site and asserts
  what a reader would see: no JS error takes out the chapters, no headline figure reads as nothing,
  the body never scrolls sideways at 1500/900/390px, and no chart label is silently cut off.

- **Exercise 03 — data collection framework.** A graded catalogue of 145 datasets and 31 benchmarks
  behind one public page that works out what an India-first, 40-billion-parameter model would train
  on — how much text, what kind, which datasets, how to clean it, how to tokenise it, and how you
  would know it worked. Every figure carries `{value, unit, provenance, source}` and the renderer
  refuses to print a bare one; where a quantity has never been measured, the page says so rather
  than showing a plausible number.
- **Five data-handling invariants enforced in CI.** Training never touches eval data · nothing
  excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a
  measurement must name what produced it · no source content is silently dropped. Each ships with a
  test proving it fails when broken.
- **The tokenizer tax, measured rather than cited.** All twenty-two scheduled languages now carry a
  real fertility number with a run id behind it — five tokenizers over IN22-Gen (source-original
  Indian text) and IN22-Conv (conversational), reported apart because register changes the answer.
  Our measurement puts the mean Indic tax at ×7.45 under cl100k and finds a better tokenizer removes
  78% of it, independently corroborating the published figures the atlas cites (8.0× and 73%); we
  measure Malayalam at ×13.0 on our own text, which is the paper's own headline example.
- **Gemma 4 is the worst Indic tokenizer of the three serious candidates.** ×2.49 mean tax against
  Sarvam-105B's ×1.81 and XLM-R's ×1.66, and ×8.84 worst-case against ×2.90. It is the assignment's
  named target, so `docs/DECISIONS.md` now prices the "continue-pretrain from Gemma-4-31B" fork with
  that number attached: continue-pretraining inherits the tokenizer, and the tokenizer is not free.
- **Contamination coverage is no longer `none`.** MILU's validation split is indexed — 126,044
  shingles from 8,923 items across 11 languages — so the gate guards something. 1,090 of those items
  fall under the 13-word window and would have been undetectable before the short-item fix.
- **Interactive explainers** rather than static tables and charts: the contamination
  gate you can try to defeat with your own sentence, a vocabulary optimum that moves as you change
  the model width, a quality filter that deletes twelve of twenty-two languages until the protected
  lane restores them, and a confidence ledger that narrows to the nine claims that would survive
  checking. Conventions recorded in `docs/EXPLAINER_PROMPT.md` and `docs/EXPLAINER_PATTERN.md`.


- **Every dataset now says how it relates to the others.** FineWeb's 15T sat beside FineWeb-Edu's
  1.3T with nothing to say the second is *inside* the first — likewise FinePDFs/FinePDFs-Edu and
  Nemotron-CC/v2 — so a reader adding any of those pairs was double-counting and the page gave them
  no way to know. Fifteen datasets now carry a published relationship as a badge beside their name,
  in four kinds that are deliberately not interchangeable: **contained_by** (3 pairs, stated by
  their publishers, subtracted from any sum holding the parent), **additional_to** (1 — recorded
  because Nemotron-CC-v2.1 genuinely *is* additive and shouldn't be "fixed" into a containment),
  **shares_source** (7, real overlap of unpublished size), and **independent** (2, because "these do
  not overlap" is as much a finding as the reverse). Each carries the concrete thing that *is* known
  — crawl counts, date ranges, modality — plus its citation, on hover. **No per-pair overlap
  coefficient is invented**: nobody has measured one, and a made-up fraction wearing the authority
  of a computation would be worse than the honest band. What the callouts say instead: FinePDFs
  draws on the same crawls as FineWeb but takes the PDFs rather than the HTML, so its overlap is far
  smaller than the shared source implies; CulturaX is built not from the crawl but from mC4 and
  OSCAR, which both were. Correction X26.
- **Supply is no longer a raw sum of overlapping corpora.** The page reported "61T reachable · 363%
  of budget" by adding datasets that are differently-filtered views of the same crawls — FineWeb is
  96 Common Crawl dumps, FinePDFs 106 of them, Nemotron-CC is Common Crawl, HPLT v3.0 is 45% Common
  Crawl by volume. Risk **R01** has said so throughout, severity high, *"the single most likely
  schedule-breaker"*: 60–80% cross-corpus duplication. Every sum over catalogue tokens now reports
  the deduplicated range alongside the raw figure, through **one shared helper** — there were ten
  sum sites, and fixing one is how this kind of error survives. Two mechanisms, deliberately not
  interchangeable: **exact containment is subtracted** (NVIDIA states Nemotron-CC-v2 *is* v1 plus
  eight snapshots, so counting both double-counted ~6.3T), while **unknown overlap gets a band** —
  per-pair coefficients are not invented to make the arithmetic look more precise than the evidence.
  Stage 3 now reads **55.4T raw · 11.1T–22.2T after dedup · 66–132% of budget**: a possible
  shortfall where it read a comfortable surplus. The Indic pool gets its own treatment, since R01's
  range is about Common Crawl and Sangraha's *verified* portion is scraped, OCR'd and transcribed
  rather than crawled — what is unmeasured there is that it and IndicCorp v2 are both AI4Bharat
  scrapes of Indian sites with no published cross-deduplication, so **84.9B is a ceiling, not a
  count**. Also corrected: the code row recorded 377M tokens, which is NVIDIA's count of GitHub
  *files*; its card states 747.4B. Corrections X24.
- **Chart labels wrap instead of being cut off.** Fourteen were losing text, up to 82px of it —
  "natural Indic we h…", and every tool name in the cleaning chapter, which appears nowhere else on
  the page. An ellipsis is honest only when the full text is reachable another way: a catalogue row
  is a button that opens the dataset's gates, so it keeps its ellipsis; a chart label has no such
  escape. The deeper bug was grid sizing — a fixed or `auto` track beside a flexible one left the
  flexible column absorbing the entire shortfall, down to **4px at 900px and 28px at 1180px**, which
  stacks letters vertically and is worse than truncating. Rows now collapse on a **container query**
  against their own column rather than the viewport, because the figure column is narrow in two
  disjoint viewport bands and breakpoints fixed one while leaving the other. Verified clean from
  390px to 1600px. Correction X25.
- **Costs are shown as an order of magnitude, never as a figure.** Every price on this page sits on
  arithmetic that is solid — 6ND, or a share of it — multiplied by two assumptions that are not: a
  sustained throughput that moves ±30% between real runs, and a list rate the project's own cost
  record calls *"negotiated well below it"* at reservation scale, and below that again on spot.
  Those compound to a band several times wide, so `$5,600,000` claimed seven significant figures of
  precision nobody has. One scale now runs through the whole page — `$` thousands, `$$` tens of
  thousands, `$$$` hundreds of thousands, `$$$$` millions — and it makes the comparison the cost
  chapter is actually about legible at a glance: the three forks read `$$$$`, `$$$`, `$$$$`. The
  figures stay in the records, so the arithmetic remains checkable; it is the display that stops
  overclaiming. No raw money figure renders anywhere on the page now.
- **The rephrasing route is costed, and the answer is that it does not close the gap.** Correction
  X15 established that repetition is the weakest answer to a scarce pool and that the frontier
  rephrases instead — and the page then went on planning around passes, because nobody had priced
  the alternative. Now it is priced. Restating the 84.9B committable Indic pool **twice** makes it
  170B and lifts what the tier is worth by about **25% at identical seen tokens**, for a generation
  cost that stays **under 1% of the run** however the throughput is estimated. Chapter 2 says so
  where it used to end on how many times to read the pool; chapter 13 prices it beside the speech
  route. The finding is that it is worth doing *and is not the answer*: at two variants — the only
  depth Kimi K2 uses in production — the tier is still **13% unique text and 87% repetition**, and
  filling it outright would take sixteen variants, past anything anybody has reported. Whether a
  rephrasing counts as a distinct document for the repetition curve has never been measured; the
  25% assumes it does. Cost is shown as a **band rather than a figure**, with the arithmetic stated,
  because the throughput term spans an order of magnitude — decode is bound by weight reads rather
  than FLOPs, so a point estimate would claim a precision this project does not have. That makes it
  the strongest argument for funding collection: the one that survives having tried the alternative.
  Recorded as correction X23.
- **Chapter 2 was stating the mixture's requirement as if it were the corpus.** It took its pool
  from a tier field called `unique_tokens` and said of it: *"Read once, this is the entire natural
  Indian-language pool — every verified corpus anyone has assembled, added together."* That field is
  not a measurement. `milestones.py` computes it as `share × budget / epochs`, so the 8% Indic tier
  over 4 passes produced **336B — the pool the mixture would need for the share it wants**. What the
  catalogue can actually commit is **84.9B**: Sangraha's verified portion plus IndicCorp v2. A demand
  figure wearing the clothes of a supply figure, and the chapter's whole argument rested on it. The
  field is renamed `unique_tokens_required` throughout, the pool is now summed from the same
  catalogue the datasets chapter reads, and the corrected chapter is the stronger one: four passes
  on the real pool are worth 316B — **24% of what the tier is allocated** — and filling that
  allocation takes about **16 passes, the half-life the published fit names**, at which the tokens
  are worth 66% of what they cost. "Nearly free" was true of a pool four times larger than the one
  anybody holds. Two things had hidden it: the growth chapter had it right all along ("336B… asks
  for, 4.0× SHORT"), so the page contradicted itself across two chapters with one number; and risk
  R19 records a 250–500B estimate that 336B sits inside, making it look corroborated — while R19
  itself calls that estimate *"the single most important unverified number in the entire document"*.
  Recorded as correction X22.
- **Each guardrail says where its number came from, and how strong that is.** `mix.py` held eight
  of them in one voice: the repetition constants cited a fitted curve to three significant figures,
  and the composition constants cited nothing. `MAX_SYNTHETIC_SHARE_OF_INDIC = 0.50` carried the
  comment *"past this, the Indic tier is mostly manufactured text"* — which restates what 0.50 means
  and says nothing about why fifty. Traced: it comes from one row of the risk register's mitigation
  column, phrased as *"cap synthetic at ~50% of the Indic tier"*, with no citation and no
  experiment. Now every guardrail is classified — **four published or fitted** from the repetition
  literature, **one adopted** (the protected lane's 8% floor is what LightningLM reserved on its own
  corpus, not a measured optimum for this one), and **three asserted** with no measurement behind
  them at all. The strength maps onto provenance, so a measured guardrail wears the mark meaning
  somebody ran it and an asserted one does not. This matters because **both lines the mixture
  crosses are in the asserted group** — which does not excuse the breach, but changes it from
  "we exceeded a measured limit" to "we exceeded a line we drew ourselves and have never tested".
  One further gap surfaced while tracing it: the risk that 50% answers prescribes **four**
  mitigations and only the cap was implemented — the KenLM-perplexity floor, the n-gram diversity
  floor and the per-language entropy monitor are recorded as prose and checked by nothing. The page
  says that too. Recorded as correction X21.
- **A grade now says how much was asked, not only how it answered.** `UNKNOWN` and `FAIL` both
  score zero — deliberately, so ignorance costs what a poor result costs — but that made "scored 5
  with every gate measured" and "scored 5 with three gates never looked at" the same letter. Grade A
  requires all five gates scored, B requires at least three, and every dataset ships a `gates_scored`
  count beside its grade. **The distribution is unchanged** (B 14, C 116, X 15), which is the point:
  the rule states what the letters already meant instead of reshuffling them. And the page now says
  why no dataset holds the top grade, which was previously left for a reader to notice. Gate by gate,
  out of 145: provenance 144 scored, composition 6, **contamination 1**, yield 15, evidence 145.
  Evidence passes for every dataset because every catalogued dataset has been used by somebody, so
  it is two free points that discriminate nothing, and 125 of 145 records have two gates scored or
  fewer — for most of the catalogue the only real signal is where the text came from. A is empty not
  because nothing is good enough but because **nothing has been fully checked**, and the sharpest
  case is now stated plainly: the contamination gate is scored on one dataset in the entire
  catalogue and on none of the four the plan commits to. Those four clear licence, provenance and
  size; they are not datasets somebody finished checking. Recorded as correction X20.
- **Prose that quotes a count now reads it.** The opening paragraph called the corpus **17
  trillion** tokens — `.toFixed(0)` on 16.8 — while every other mention said 16.8T. Chapter 4 said
  the mixture had **eight** tiers and **ten** tiers on one screen, beside a figure drawing ten bars a
  reader can count. The glossary defined a rung as one of "5, 10, 15 or 20 trillion tokens", a ladder
  replaced by 3T/8T/16.8T/30T, and defined a pass as contributing "four times its size" — the exact
  arithmetic correction X15 exists to retract, sitting in the band readers are told to read first.
  Chapter 13 claimed all four licence-blocked datasets were English corpora when the largest is
  **HPLT v3.0 at 198 languages**, and planned against 35,000 hours of open Indic speech while this
  page's own appendix put the pool "well north of 100,000". The cost chapter priced a 15T run: 2.50M
  H100-hours where the recommendation implies **2.80M**, and $5.0M where it implies **$5.6M**. Two
  figures were both numbered Fig. 9. Sarvam-30B was 30B in one register and 32B in another (the card
  says 32B), and both cited the model cards for token counts **the cards do not state**. All fixed —
  and fixed at the cause: every count quoted in prose is now read from the record that holds it, a
  new `run_cost.py` recomputes the run price the way `vocab_trade.py` already recomputes the
  vocabulary trade, and a test asserts the two model registers agree about any model they both
  describe. Recorded as correction X19.
- **Three of the guards enforcing the five invariants could not fail.** INV-1b's mutation proof
  built a string from an eval item and asserted the item was in it — it touched no project code and
  could not fail under any change to the repository. INV-2's check asked
  `is_commercially_usable(grade)` inside `if grade == "X"`, and that function was `grade != "X"`, so
  it reduced to asserting that `"X"` equals `"X"`. And the leak scan looked only for synthetic
  fixtures that were never within reach of the pipeline, never for a real benchmark item. The guards
  now test the claims: INV-2 reads the shipped plan and asserts no excluded dataset reaches a
  committed tier, and `is_commercially_usable` takes the record and requires an **established**
  licence — because *unknown is not permission*, which the rest of the framework says and this one
  function contradicted. Alongside them: `score_gates` counts only the five named gates, so a
  duplicated key can no longer score 14 out of a stated maximum of 10 and buy a grade A;
  `coverage.py`'s exclusion clause, which could only ever fire for one capability, now runs for the
  first time and stops counting **IndicMMLU-Pro** as an instrument for Indian worldview when its own
  notes say it was translated; and `mix.check` reports a tier it cannot assess instead of skipping
  it in silence, errors on a negative schedule, and no longer raises on a null one. The page also
  now discloses **both** guardrails its mixture breaches — the synthetic-Indic share at 52.4% was
  going unmentioned beside the protected lane at 21.0%, which reads as candour while showing half
  the picture. Recorded as correction X18.
- **Nothing derived from an estimate calls itself measured any more.** The `sourcing`, `lifecycle`
  and `orphan_tiers` blocks each declared `provenance: "measured"` over every number inside them —
  including `committed_tokens` of 6.39T, which is a sum over the catalogue's own sizes, and **not
  one of the 145 records carries a measured size** (24 estimated, 121 unknown). The page tells
  readers the green underline means "somebody ran it". Nobody had. A derived number is now no more
  measured than its least-measured input, so those blocks are estimated — while the exact counts
  keep the mark they earned, because counting records in a catalogue we hold really is a
  measurement. Three figures the browser labelled by hand are corrected too, one of them a
  counterfactual showing what the corpus *would* hold if the reader resolved blockers that have not
  been resolved. Separately, 115 fertility values shipped claiming measurement against a run id
  literally prefixed `pending-`: the substitution walked one block and missed `conversational`. It
  now walks all of them, and `protocol_gaps` — a hardcoded string still insisting "three of the six
  tokenizers are unavailable" long after the run measured five with one unavailable — is computed
  from the run it describes. Recorded as correction X17.
- **The contamination gate no longer shatters Indic text, or deletes clean documents.** It
  tokenised with `\w+`, and Python's `\w` matches letters and digits but **not Unicode combining
  marks** — which is what every Indic vowel sign, virama, anusvara and nukta is. So every Indic word
  was split at every vowel sign and the sign discarded: five Hindi words became eleven consonant
  fragments. 91% of the indexed items are in Indic scripts, inflated by a mean factor of 2.58, so a
  "thirteen-word fingerprint" was about five real words of consonant skeleton there. It produced
  **false positives that would have deleted clean training text**: measured against 203,388 held-out
  FLORES-200 sentences the old tokeniser collided with 5, all Indic, all ordinary news prose — one
  of them the Malayalam for "the attack greatly affected relations between India and Pakistan",
  three ordinary words that normalised to a full thirteen-token window. The corrected tokeniser
  collides with none of them. The same corpus now indexes to 126,044 fingerprints rather than
  411,442, across eight window widths rather than five, and **1,090 of 8,923 items are genuinely
  shorter than the window where the old count said 56**. The refusal floor rose from 5 words to 6,
  because the example the code's own comment gives as too generic to index is five words long. The
  browser demo carried the identical defect and now mirrors the pipeline, including the floor — it
  used to tell readers any question under thirteen words was unprotectable, contradicting both the
  pipeline and the paragraph beneath it. Recorded as correction X16.
- **The tier shares in chapter 11 no longer overrun the tier names.** The register's first column
  was 18px, sized for the single-digit job numbers it was originally built for, and the evaluation
  chapter reused the same row for percentages — so `15.0%` printed straight over `english-web-hq`.
  Nine of the ten rows were affected. The share variant now gets its own column width rather than
  padding the numbered list to fit a string it never contains.
- **Each growth stage now answers "why this number" twice — from supply, then from analogy.** The
  supply half is measured on this page and had never been placed next to a budget: for every stage,
  what clears every bar today and what is blocked on nothing but an unanswered licence, each as a
  percentage of that stage's own budget. The analogy half names the comparators the budget was set
  by (Gemma 3 4B's 4T, Llama 3.1 8B's 15T, Qwen3's 36T) and states plainly what an analogy proves
  about this corpus, which is nothing — it assumes corpus size follows parameter count, the
  assumption the same chapter disproves. Reading them together produces a finding neither gives
  alone: three of the four budgets are comfortably reachable, but only once four licence letters are
  answered, and the seed — which admits no web text at all — can reach **2.8% of its own budget**.
  The binding constraint on the ladder was never the size of the numbers; it is the web-data policy
  and four emails.
- **The growth stages say which of their numbers were reasoned and which were assumed.** Four token
  budgets printed in the same typeface read as four equally solid figures, and only one is. 16.8T is
  derived, and the page now shows the derivation: the plan is written against Gemma 4 31B, which
  publishes no token count, so it takes the 14T its predecessor Gemma 3 27B does publish and adds
  20% for one generation — with that 20% named as the single free parameter. 3T, 8T and 30T are
  analogies to other labs' models, each labelled as what it is: a rule of thumb, a figure chosen
  below its own comparators, and parity with published frontier counts. The method cannot produce
  them because it anchors to a *model* rather than to a size, and only one stage has a comparator —
  nobody publishes the corpus for an intermediate stage of a lineage they grew. The parameter counts
  are marked illustrative throughout, since no scaling strategy has been chosen. A tokens-per-
  parameter row makes the seam visible: the two rule-of-thumb stages sit on exactly 1,000, which is
  the signature of a rule applied rather than a budget set — and it is the same tokens-per-parameter
  reasoning the rest of the page argues against. Two stage descriptions that had drifted from the
  record are now computed from it (the seed's text called a 3B model "a dense 40B"; the third stage
  claimed to add 80B parameters where it adds 32B).
- **Repeated tokens are no longer counted as though they were fresh ones.** The mix engine computed
  `effective tokens = unique pool × epochs`, and chapter 2 printed that product as the value of a
  repetition schedule. The multiplication is right for what compute is billed on and wrong for what
  the passes are worth, and the paper the page cites gives the second as a decaying sum. Seen and
  worth are now separate quantities everywhere: four passes cost 4× and are worth 3.73×, sixteen
  cost 16× and are worth 10.6×, forty cost 40× and are worth 15.2×, and **no schedule exceeds 16.4×
  the unique pool**. The page used to display 6.72T from a 336B pool read twenty times, badged
  "unevidenced"; it was not unevidenced but unreachable — the ceiling for that pool is 5.51T at any
  number of passes — and a guardrail now errors rather than rendering such a figure. Two claims
  beside it were false and are corrected: 16 epochs is the half-life at which a repeated token has
  lost 1/e of its value, not where published work stops (the same paper reports 44-epoch runs and
  labels 40 epochs worthless), and repetition **has** been measured on Indian-language text — ATLAS
  (ICLR 2026, 774 runs over 400+ languages) finds Hindi's curve bends upward sooner than English's,
  which makes these constants the optimistic end for an Indic pool rather than the neutral one.
  Recorded as correction X15, with what the frontier does instead: Kimi K2 measured ten epochs of
  raw repetition at ~23.8% against ~28.9% for ten rephrasings read once, and Kimi K3's pre-training
  section never uses the word "epoch".
- **A contents rail that stays with the reader, and only one contents list.** The one-page report
  runs thirteen chapters and an appendix, and the only way to see where you were was to scroll back
  to the top. There is now a single contents with two presentations: a block in the flow on narrow
  screens, carrying a line on what each chapter answers, and on wide ones a rail pinned to the left
  margin, vertically centred, marking the chapter you are in and collapsing to one button when you
  want the width back — a choice it remembers. The two-column block that used to sit under the lede
  is gone, along with the screen of scrolling it put between the opening and the first chapter.
- **The growth plan says what each stage would actually train on.** The four stages — 3T, 8T, 16.8T
  and 30T — now each carry the sequence length, whether web data is admitted, whether the script
  quarantine is absolute or enforced, and how much noise passes, with a table comparing all four
  down the page. Each stage then names how many catalogued datasets its own rule admits and what
  they carry, so the corpus a stage needs is checked against the corpus that exists rather than
  asserted beside it. A stage now summarises itself as its window size and its corpus rather than
  its parameter count: what to read, and in how long a window, is the decision being made — the
  parameter count follows from it.
- **Each growth stage names the datasets it would read.** Saying a stage "admits 4 datasets carrying
  6.39T" is a count standing in for a shopping list. Every stage now prints the list, in two groups,
  because they are blocked by different things: clear today (3 datasets and 85.3B at the seed, which
  forbids web text; 4 and 6.39T after it) and one letter away — blocked on an unanswered licence and
  nothing else, which is four datasets holding 54.6T. The Indic sizes quoted are the verified counts
  rather than the announced ones, so a stage is never planned against 187B that nobody has checked.
- **One content width, with prose held to a reading measure inside it.** The page used to cap
  everything at 860px, which cramped a seven-column dataset table and a two-column figure while a
  thousand pixels sat empty either side. The container is now a single 1240px and never moves, so
  every left edge lines up; prose stops at its own line-length measure and leaves the right ragged,
  while tables, figures and registers fill the width. The explainer's chart column grows with the
  window rather than the prose lines getting longer.
- **Deploys no longer leave readers on a stale page.** Assets were referenced by bare name and
  served with default caching, so after a release a returning reader could hold a fresh
  `index.html` alongside a cached `chapters.js` — and every chapter title, number and figure on the
  data-collection page comes out of that one file. The build now appends each asset's content hash
  to every reference, walking the whole `index.html → chapters.js → explainer.js → num.js` graph to
  a fixpoint so a change to a leaf reaches the top. Sources keep bare names; only the built output
  carries hashes.
- **Six themes, site-wide, every one contrast-checked.** Tokens moved out of the individual pages
  into one `/_shared/tokens.css` that the landing page and all three exercises link, so a colour
  decision is made once. The system light/dark pair stays the default, joined by soft light (warm
  off-white), tinted dark (deep navy, not pure black), high contrast (monochrome) and neon
  (near-black with luminous accents). The choice persists across the whole site and is applied
  before first paint, so there is no flash of the wrong theme. Promoting the tokens also fixed two
  WCAG failures that were still live on the landing page and exercises 01 and 02 — `--faint` at
  3.33:1 and `--accent` at 4.31:1, both already corrected in 03 and never propagated. Verified in
  the browser: five pages x four themes plus both system modes, all contrast-pass.
- **Five themes, and every one of them contrast-checked.** The system light/dark pair stays the
  default, joined by soft light (warm off-white rather than cool grey), tinted dark (deep navy, not
  pure black), high contrast (monochrome), and neon (near-black with luminous accents). The choice
  persists and is applied before first paint, so there is no flash of the wrong theme. Two rules
  make it safe rather than decorative: the `prefers-color-scheme` block is scoped so a chosen theme
  always wins, and each theme defines the whole token set instead of inheriting half of it. Every
  text token in every theme was generated from a palette that a contrast checker had already
  passed — four contrast failures had shipped in this exercise when values were picked by eye — and
  all six themes measure accessibility 100 in the browser.
- **Every number now carries provenance declared in the record it comes from.** Figures extracted
  from `docs/DECISIONS.md` used to be typed at the point of render with a source string written in
  the chapter — putting the claim about a number somewhere no reviewer of the record would look,
  and several of them circular ("the vocabulary design" as the source for a figure *in* the
  vocabulary design). All six records now declare their own `provenance.fields`, distinguishing a
  proposal from a derivation from a published figure, and a number with no declaration throws
  rather than quietly borrowing the nearest string. The 21 figures sourced to "the proposed tier
  shape" now name the document and module that decided it.
- **The growth lineage is 3B, 8B, 40B, 200B.** The 40B is stage three, not the start. Each stage
  reserves 8% of its batch for natural Indian-language text, so the requirement scales with the
  corpus while the supply does not: 60B unique needed at 3B against 84.9B committable, then 160B,
  336B and 600B. The seed is the only stage this catalogue can supply, and its text is inherited by
  every model above it.
- **A growth chapter.** The 40B is a seed, and the chapter follows it to the largest Indic model in
  four stages — dense, grow sparse, grow deep, frontier parity — using the state-preserving growth
  method of arXiv:2606.07404. The shape of the answer is the lesson: parameters rise 7.5×, the
  corpus rises 1.8×, and the natural Indian-language pool does not rise at all. One stage adds 80
  billion parameters and reads no additional text.
- **Two India-first tiers no frontier lab has a reason to build.** Indian knowledge systems
  (Ayurveda, Siddha, Jyotish, and the NDLI/DLI scanned archives) and Indian civilizational
  literature (Vedic corpus, classical Sanskrit, Upanishads, Dharmashastra). Both are OCR problems
  before they are data problems. The mixture is zero-sum, so their 5% comes off general web and
  synthetic Indic. One of them has **zero catalogued candidates** — not one of the 145 datasets
  supplies it.
- **A second lens on the mixture.** Ten tiers is more than a reader holds in mind, so each is also
  labelled *skills* or *knowledge*. The mixture now runs 40% skills / 60% knowledge, raised from 28%
  because the assignment names coding and agentic work as primary capabilities.
- **A comparison table for the corpus budget** (`records/scaling_reference.json`): thirteen models
  with their parameters, architecture and pre-training tokens, every row carrying its source. The
  finding that matters is that a corpus is not sized by the model reading it — Llama 3.1 trained 8B,
  70B and 405B on about the same 15T, and DeepSeek-V3 at 671B total read less than Gemma 3 27B.
- **The shopping list is filterable.** Chapter 5 was ten stacked tables and 109 rows; it now filters
  by tier and by what is blocking each dataset, with live counts on every chip and nothing hidden.
- **The catalogue view is grouped and searchable.** 145 identical marks in one wall carried their
  meaning entirely in hover. They are now grouped into labelled grade bands — each naming its own
  colour — with a search box, and an empty band says so rather than being silently absent.
- **Web design system** (`docs/DESIGN.md`): a shared Apple-style visual language — palette tokens,
  typography, components, interaction, and copy/tone rules — that every exercise's `web/` bundle
  follows.

### Changed

- **Exercise 03 is one page instead of three.** The decision and the evidence behind it were split
  across two dense pages of 25 sections; a reader could not find what to actually train on. It is now
  a single page of thirteen chapters and an appendix, each answering one question in the order a
  reader asks it — what we are building, how much text, how it grows, what goes into it, **which
  datasets**, what we may legally use, how to clean it, keeping the exam out of the training data,
  how to teach behaviour, how to tokenise it, how we would know it worked, what it costs, and what
  to do first. Old links still resolve: every retired anchor redirects to the chapter that absorbed
  it.
- **Every chapter has three layers**, so one page serves a reader meeting the subject today and one
  who trains models for a living: a plain headline and a single number, the interaction that proves
  it, and a closed *"The arithmetic"* holding the derivation, sources and caveats. Stepped content
  fell from 45 states (~21 screens) to 29 (~8.7), because states are now justified by the argument
  rather than filled to a quota.
- **The 40B is a seed, and its corpus is derived rather than asserted.** An earlier draft presented
  15 trillion tokens as the model's budget; 15T was the research's target for a 300B mixture of
  experts, so a growth ladder had been mislabelled as one model's options. The seed now sits at
  16.8T — Gemma 3's officially stated 14T for its 27B, plus 20% for one generation — and it is
  labelled an estimate with a stated method everywhere it appears, because Gemma 4 discloses no
  token count in either its technical report or its model card.
- **Four questions the site never answered now have chapters**: what may legally be used, what to
  train on after pre-training, what the whole thing costs and whether to build from scratch at all,
  and what to do first.

### Fixed

- **One rule, one implementation — and a guard that runs both.** The containment bug (X28) happened
  because the same rule lived in Python for the bundle and in JavaScript for the page, and only one
  got fixed. Auditing for the same shape found `tierOf` written out **four times** in `chapters.js`
  and the natural-Indic token rule twice; all are now single module-scope functions. The
  cross-implementation guard was extended from blockers alone to also compare tier assignment and
  countable tokens per dataset, by running the real browser code against the real bundle, plus a
  check that the deduplication band's fallback literal still matches `DEDUP_SURVIVAL`.
- **Index headroom, not just a pass mark.** 99.2KB against a 100KB budget left no room for a
  sentence. Dropped the per-tier `worth_tokens` and its two totals, and `feasibility` — nothing on
  the page reads any of them, and the worth-vs-seen distinction stays computable with its own test.
  **96.4KB.**

- **The index is under its 100KB budget again — 137.5KB to 99.1KB.** It had been over since before
  this branch and the modality work made it worse, so: the per-dataset relationship notes, the 17
  priors and the curriculum prose moved to `records.json` (both surfaces already load it, so this is
  off the first-paint parse, not hidden); repeated source strings are stored once and referenced by
  index, since 145 datasets carried the *same* sentence under `gates_scored.source` for 17KB — the
  exact source still reaches the reader, resolved by the renderer instead of by duplication; and
  fields nothing renders were dropped, `slug` having zero references on the page and a benchmark
  shipping eleven fields for a table with five columns. No figure and no citation was shortened.
- **"0 of 55 post-training datasets state a size" now reads "40 of 55", because it was measuring the
  wrong thing.** The size parser only understands tokens, so every size these datasets *do* state
  was discarded — SWE-Gym's "2,438 executable tasks", SWE-RL's "273K seed tasks", AutoTool's "200K
  tool-use trajectories", Bhashini's "thousands of audio hours". A headline reading `0` says "we
  have nothing" whatever its caption says, and that was false. The catalogue now records whether a
  row states a size in *any* unit: 40 of the 55 post-training sets do, 15 state nothing at all, and
  none states one in tokens — which is the real finding, since it means the post-training budget
  cannot be checked against supply the way the pre-training one can. Counted from the full records
  at build time, so the index carries no extra bytes for it.

- **The mix now says what each token is *for*, and whether we can actually source it.** Tiers named
  a provenance and `kind` split them into skills and knowledge, but nothing said what a token
  *teaches* or what it is *about* — so the growth ladder described four stages without saying what
  each one teaches, and "145 datasets" could not be asked whether it covers news, or literature, or
  agriculture. Two lenses added over the same tokens, deliberately not merged with the tier: a
  **modality** (what kind of thinking) and a **domain** (what it is about). The seven modalities and
  the code language list are the specification's own, carried verbatim — including that
  `agentic_traces` is owned by Team 17 with `format_pending`, because a modality nobody has agreed a
  format for cannot be collected and the plan should say whose decision that is. Each stage of the
  ladder shows the modality mix it is taught in, in the order a person is taught: language, then the
  world, then symbols, then the things that need all three. **General text falls 62% → 29% across
  the four stages while code rises 6% → 26%.**
- **And the coverage register, which is the counted half.** Of 16 domains the curriculum names, 12
  have a dataset that isolates them; **social and qa exist only as an unseparated slice of a web
  crawl** — trainable, but impossible to weight, measure or hold out; and **agriculture and health
  have nothing in the catalogue at all**, which for an India-first model makes agriculture the
  uncomfortable one, being the sector most of the country works in. Counting only what could be
  committed today, **2 of the 16 domains have even one dataset clear of every blocker** — the
  curriculum is not short of candidates, it is short of permission. Every row ships the pattern it
  was matched by, so the count is checkable rather than asserted; the tier weights and curriculum
  emphases are typed `estimated` and named as a proposal, because nobody has classified a crawl by
  modality and a plan dressed as a measurement is the exact failure this register exists to catch.
  Correction X30.

- **The Dataset Card shows its sources again.** The card kept the five gates through the one-page
  rebuild and dropped the facts they were judging, so a reader could see a verdict and not reach the
  evidence — the links were in the catalogue the whole time (79 of the 145 rows carry one) and had
  nowhere to be shown. Restored as the things the table it opens from cannot say: the
  verified/unverified/synthetic split behind a headline token count, languages, prior use, how the
  data is actually distributed and when that was last checked, and every recorded source as a link,
  with `arXiv:` identifiers resolved to URLs. Tokens, stage, kind and commercial use stay in the
  table rather than being repeated on the card.
- **The catalogue could not see the difference between "open" and "ask permission".** Availability
  came from prose in a seed cell, so corpora gated behind NVIDIA's manual approval sat in the same
  band as ones you can fetch anonymously — while the page's whole dividing line is whether anybody's
  permission is needed. Each dataset's distribution point is now read from the publisher, carrying
  HuggingFace's three-way gating value and the date it was checked. Manual gating is a blocker in its
  own right; click-through gating is recorded but does not block, because accepting terms is
  something you do unilaterally. Nemotron-CC-v2 moves from one letter away to two, which is the
  honest answer to why the older v1 is the committable one. Two figures were wrong as a result and
  are corrected: the Nemotron code row counted 747.4B for v1/v2/v3 when v1 and v2 need approval and
  v3 — the only ungated one, and explicitly incremental to them — holds 173B; and "Dolma" was
  catalogued with no version and no size while AI2 had shipped Dolma 3, a 6T ungated ODC-By mix, so
  a corpus larger than the whole committable band counted as nothing. Correction X29.
- **A web corpus's licence covers the curation, not the copyright of the text inside it.** The legal
  chapter now says so: Common Crawl's terms license use of the service, require you to respect
  third-party copyright in the crawled material, demand indemnity, and advise counsel before
  commercial use. A permitted mark on FineWeb, Nemotron-CC or HPLT means its curator allows you to
  redistribute their package — not that anyone may train on the text.
- **A subset was being summed beside the set that contains it, on screen.** The stage registers
  reported every stage's reachable supply as a plain sum: Nemotron-CC v1's 6.30T counted once alone
  and again inside Nemotron-CC-v2's 6.60T. The containment map was correct in the bundle and the
  browser ignored it — `contained_by` maps a child to a *list* of parents, and the filter tested
  that list for membership in a set of ids, which is false for every list. The reachable total at
  the recommended budget drops from 61.7T raw to 55.4T. Correction X28.
- **Provenance is declared where the figure is made, not one block up.** The lifecycle block marked
  all seven of its fields `estimated` to cover the one that is a sum over catalogue sizes, so
  "4 of 24 post-training datasets state a size" — a count — rendered under the same hedge as a
  projection. It now declares `measured` and the sum carries its own mark inline; the orphan-tier
  block keeps `estimated` for its costs and states its match count as measured. Correction X27.
- **Nine dataset-relationship notes had lost a space at a line seam**, rendering `notcollected`,
  `Englishand` and `v2,not` in the hover callouts.

- **Machine translation was being counted as natural Indian text.** Sangraha ships 251B tokens of
  which 64B are verified human-origin; the rest is roughly 90B machine-translated from Wikimedia and
  72B romanised transliteration. The catalogue card records that split and the budget used the
  headline — while the mixture chapter argued, in the same page, that counting synthetic as natural
  is "the commonest way to overstate a corpus". The natural-Indic tiers now count verified
  human-origin tokens only, which takes that tier from 80.9% covered to **25.3%**. Both figures are
  kept: every tier reports what its headline totals would have claimed, because the gap between them
  is the finding.
- **The vocabulary trade was a stale hand calculation.** `docs/DECISIONS.md` works it once against a
  15T budget with a 25.3% India slice; both have since moved. It is now recomputed from the mixture
  on every export — 1.05T tokens, 175,048 H100-hours, about ₹3.0 crore — and the one input that
  cannot be measured is named on the page rather than buried, since it describes a candidate
  tokenizer nobody has trained.
- **The post-training budget was three tables of uncited numbers.** `docs/DECISIONS.md` states them
  with a tilde and cites nothing, and the page typed them as estimates whose source was *"the
  post-training plan"* — the document containing the number. They now say what they are, and a new
  state compares them against the two labs that published theirs. That comparison finds a real
  defect: Tulu 3 pairs preference data at 36% of its SFT size and this plan proposes 6.7%.
- **Fast scrolling skipped explainer states.** The active state was chosen by intersection with a
  band 10% of the viewport tall, so several steps could be inside it at once with the last one
  processed winning, and a step could pass through entirely between callbacks and never fire. It is
  now whichever step is nearest the viewport centre, which has neither failure.
- **White text on the dark-mode accent measured 3.01:1.** Wrong since the accent was darkened, and
  invisible because every previous audit happened to run in light mode. One `--on-accent` token now
  carries white on light and near-black on dark; both themes score 100 for accessibility.
- **Contamination gate missed short evaluation items.** An item shorter than the 13-word shingle
  window hashed to a single whole-text gram, which can never match a 13-gram drawn from a longer
  training shard — so a short benchmark question pasted verbatim into a shard was reported clean.
  The index now records the window width each item was hashed at and checks the document at each
  width; items too short to identify anything are refused and counted rather than silently
  accepted.
- **The framework answered a third of its own assignment.** The class brief scopes this exercise as
  the full lifecycle — pre-training, SFT, preference, safety and evaluation — and the exercise brief
  narrowed it to "the pre-training data mix". Every one of the 145 catalogue records already carried
  a training-stage tag (101 pre-training, 47 SFT, 17 RL, 15 evaluation, 1 safety); it was rendered in
  one place and read by no module, while the tier mapping silently dropped 36 records including every
  preference, RL-only and safety dataset. The catalogue is now grouped by stage and nothing is
  dropped without being reported. Recorded in full in `docs/DESIGN_CRITIQUE.md`.
- **Seventeen answers that were already written had never been published** — the SFT/DPO/RLVR
  budgets, the twelve script blocks that sum to the chosen vocabulary, the per-language fertility
  targets, the FLOP comparison, the vocabulary trade, six objective-specific cleaning rules, the
  safety and PII gate, the curriculum phases, and the evaluation trust matrix. All are extracted into
  data and rendered.
- **Seven statements that were false as shipped**, most introduced when the vocabulary sweep was
  anchored on a measurement and the consequence was not propagated: a section claiming the curve was
  unanchored while the next said it was anchored, "three tokenizers" beside a chart showing five, a
  corpus described as translated when it is source-original, and a landing-page honesty line wrong in
  all three of its parts.
- **Two interactions that defeated themselves**: the landing page printed the answer to the page's
  only predict-before-reveal question, and using that question's input suppressed the most concrete
  fact on the page.
- **Contrast failures across the palette.** Seven semantic tokens passed as fills and failed as text
  — `--accent` at 4.31:1, `--grade-b` at 2.99:1. Re-measured and darkened; the page now scores 100
  for accessibility and 100 for performance.
- **Every page claimed a build time it did not have.** `generated_at` was exported as `None` with a
  comment saying the caller would stamp it; no caller ever did. It is stamped at the I/O boundary
  now, leaving the bundle builder pure.
- **Accessibility failures found by an actual Lighthouse run**, not by inspection: 145 interactive
  marks below the minimum touch-target size, a `--faint` token at 3.33:1, inactive scrollytelling
  steps dimmed to 1.57:1, and white ribbon labels at 3.25:1 on amber. All four fixed; all three
  pages now score 100 on accessibility and 95–100 on performance.

### Changed

- **Unified the site's look.** The landing page and both exercises now share one design language
  (cool-gray/blue, system sans, soft-shadow panels), replacing exercise 01's prior warm-paper/serif
  theme. Added a consistent `← Back` link across pages.
- **Rewrote public-page copy for a general audience.** Page footers and copy now read as standalone,
  blog-style demos, so first-time visitors can follow them without any course context.
- **Depth demo (`s2.html`):** the linear↔ReLU toggle now animates as a smooth grid morph with stable
  framing, instead of an instant redraw that also resized the panels.

## [0.1.0] - 2026-07-10

First tagged release: two interactive exercises live on Vercel with a gated deploy pipeline.

### Added

- **Session 1 — Introductions:** four live, in-browser, dependency-free proofs of why neural
  networks work (nonlinearity, depth, learned embeddings, and the role of data).
- **Session 2 — Tokenization:** a single 10,000-token multilingual BPE shared across four
  languages, scored by cross-language fertility spread.
  - An ablation harness sweeping algorithm × representation × normalization × vocab size ×
    corpus weighting.
  - A **from-scratch BPE** (no tokenizer library), competitive with the HuggingFace baseline.
  - A zero-dependency widget: per-language ratios, the score calculation, and a searchable
    view of the full vocabulary.
- **Web hosting on Vercel:** one project serving every exercise's static bundle under its slug
  (`/01-introductions/`, `/02-tokenization/`) behind a minimal landing page, assembled by
  `deploy/vercel/build.sh`.
- **Continuous delivery:** automatic preview deployments per pull request; **on-demand,
  approval-gated** production deploys via the `Deploy to production` GitHub Actions workflow.
- **Tooling & conventions:** uv workspace (Python 3.12), ruff lint/format, pytest (unit +
  integration split), GitHub Actions CI, and a PR-only workflow documented in `AGENTS.md`.

[Unreleased]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pankajkr23/llm-pretraining-exercises/releases/tag/v0.1.0
