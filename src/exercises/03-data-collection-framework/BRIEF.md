# Data Collection & Sourcing — a decision framework

Sourcing data for a frontier LLM is a **decision problem under hard constraints**, not a shopping
list. This exercise turns a large research investigation — **the India LLM Data Atlas**
([`docs/ATLAS.md`](docs/ATLAS.md)) — into a **reusable, mechanically-checkable framework for deciding
the pre-training data mix** of an India-first model, and packages that decision as a self-justifying
interactive site plus a short printable report.

## The core claim

For an India-first frontier model, the binding constraints aren't compute or architecture — they're
**data**: there isn't enough natural Indic text on Earth to pre-train on Indic alone (~250–500B
deduplicated tokens vs 15T for English), Indic scripts are taxed 4–8× by English-centric tokenizers,
India's own educational corpus is non-commercially licensed, and there's no uncontaminated Indic
evaluation set. The framework makes each of those decisions **explicit, evidence-backed, and enforced
in code**.

## What you're building (the deliverables)

1. A **Python pipeline** (`dataframework`) that ingests the seed data spine (145 datasets + 31
   benchmarks) plus the Atlas, grades each source through the framework's five questions, computes the
   token budget/mix, measures tokenizer fertility, and exports a small **typed `web/data.json`** —
   every number carrying `{value, unit, provenance, source}`.
2. **Five invariants enforced in CI** (see [`docs/README.md`](docs/README.md)) — rigor is machine-checked,
   not asserted.
3. A **three-page zero-dependency static site** on the repo's design system: the thesis (`index`),
   **The Decision** (`report/`, ~4 print pages — `Ctrl+P → PDF` is the submission), and **The Reasoning**
   (`reasoning/` — an explorable reference where every number in The Decision links to its justification).
4. A **fertility measurement run** — the site's only `measured` numbers.

## The source of truth

The full spec lives in [`docs/`](docs/) — read **[`docs/README.md`](docs/README.md)** first, then
`TODO.md` (execution order), `FRAMEWORK.md`, `DECISIONS.md`, `DESIGN.md`, `FERTILITY_MEASUREMENT.md`,
and `ATLAS.md` (the research). Open decisions the docs deliberately leave for a human are tracked in
[`docs/OPEN.md`](docs/OPEN.md).

## Status

**Scaffold + spec.** The docs and the seed data spine are in place; the pipeline, invariants, and site
are the phased build (`docs/TODO.md`).
