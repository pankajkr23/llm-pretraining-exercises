# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Record user-facing changes under `[Unreleased]` as they land; on release, rename that
section to the new version with a date and open a fresh `[Unreleased]`.

## [Unreleased]

### Added

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
- **Contamination coverage is no longer `none`.** MILU's validation split is indexed — 411,442
  shingles from 8,923 items across 11 languages — so the gate guards something. 56 of those items
  fall under the 13-word window and would have been undetectable before the short-item fix.
- **Interactive explainers** rather than static tables and charts: the contamination
  gate you can try to defeat with your own sentence, a vocabulary optimum that moves as you change
  the model width, a quality filter that deletes twelve of twenty-two languages until the protected
  lane restores them, and a confidence ledger that narrows to the nine claims that would survive
  checking. Conventions recorded in `docs/EXPLAINER_PROMPT.md` and `docs/EXPLAINER_PATTERN.md`.


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

[Unreleased]: https://github.com/pankajkr23/llm-pretraining-exercises/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pankajkr23/llm-pretraining-exercises/releases/tag/v0.1.0
