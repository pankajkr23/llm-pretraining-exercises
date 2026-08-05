# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Record user-facing changes under `[Unreleased]` as they land; on release, rename that
section to the new version with a date and open a fresh `[Unreleased]`.

## [Unreleased]

### Added

- **Exercise 03 — data collection framework.** A graded catalogue of 145 datasets and 31 benchmarks
  behind two public pages: **the decision** (what to train an India-first model on, in what
  proportion, at what cost) and **the atlas** (the evidence under every number). Every figure
  carries `{value, unit, provenance, source}` and the renderer refuses to print a bare one; where a
  quantity has never been measured, the page says so rather than showing a plausible number.
- **Five data-handling invariants enforced in CI.** Training never touches eval data · nothing
  excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a
  measurement must name what produced it · no source content is silently dropped. Each ships with a
  test proving it fails when broken.
- **The tokenizer tax, measured rather than cited.** Twenty of the twenty-two scheduled languages
  now carry a real fertility number with a run id behind it, from three ungated tokenizers over
  FLORES-200. Our own measurement puts the mean Indic tax at ×7.46 under cl100k and finds XLM-R
  removes 78% of it — independent corroboration of the published figures the atlas cites (8.0× and
  73%). Still partial: three of the protocol's six tokenizers are unavailable, and the candidate
  vocabulary under test has never been trained, so no parity ratio is reported.
- **Interactive explainers** on both pages, replacing static tables and charts: the contamination
  gate you can try to defeat with your own sentence, a vocabulary optimum that moves as you change
  the model width, a quality filter that deletes twelve of twenty-two languages until the protected
  lane restores them, and a confidence ledger that narrows to the nine claims that would survive
  checking. Conventions recorded in `docs/EXPLAINER_PROMPT.md` and `docs/EXPLAINER_PATTERN.md`.


- **Web design system** (`docs/DESIGN.md`): a shared Apple-style visual language — palette tokens,
  typography, components, interaction, and copy/tone rules — that every exercise's `web/` bundle
  follows.

### Fixed

- **Contamination gate missed short evaluation items.** An item shorter than the 13-word shingle
  window hashed to a single whole-text gram, which can never match a 13-gram drawn from a longer
  training shard — so a short benchmark question pasted verbatim into a shard was reported clean.
  The index now records the window width each item was hashed at and checks the document at each
  width; items too short to identify anything are refused and counted rather than silently
  accepted.
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
