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


- **Web design system** (`docs/DESIGN.md`): a shared Apple-style visual language — palette tokens,
  typography, components, interaction, and copy/tone rules — that every exercise's `web/` bundle
  follows.

### Changed

- **Exercise 03 is one page instead of three.** The decision and the evidence behind it were split
  across two dense pages of 25 sections; a reader could not find what to actually train on. It is now
  a single page of twelve chapters, each answering one question in the order a reader asks it — what
  we are building, how much text, what goes into it, **which datasets**, what we may legally use, how
  to teach behaviour, how to clean it, keeping the exam out of the training data, how to tokenise it,
  how we would know it worked, what it costs, and what to do first. Old links still resolve: every
  retired anchor redirects to the chapter that absorbed it.
- **Every chapter has three layers**, so one page serves a reader meeting the subject today and one
  who trains models for a living: a plain headline and a single number, the interaction that proves
  it, and a closed *"The arithmetic"* holding the derivation, sources and caveats. Stepped content
  fell from 45 states (~21 screens) to 29 (~8.7), because states are now justified by the argument
  rather than filled to a quota.
- **The corpus defaults to 15 trillion tokens** and names real datasets. The previous version opened
  on the smallest rung, which answers a different question from the one a reader asks.
- **Four questions the site never answered now have chapters**: what may legally be used, what to
  train on after pre-training, what the whole thing costs and whether to build from scratch at all,
  and what to do first.

### Fixed

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
  targets, the FLOP comparison, the ₹2.5 crore trade, six objective-specific cleaning rules, the
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
