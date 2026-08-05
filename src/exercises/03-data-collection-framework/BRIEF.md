# Data Collection & Sourcing — a decision framework

Sourcing data for a frontier LLM is a **decision problem under hard constraints**, not a shopping
list. This exercise turns a large research investigation — **the India LLM Data Atlas**
([`docs/ATLAS.md`](docs/ATLAS.md)) — into a framework for deciding what an India-first model trains
on **across the full lifecycle: pre-training corpora, SFT, preference, safety and evaluation**, and
publishes that decision as one interactive page.

> **Scope correction.** An earlier version of this brief said "the **pre-training** data mix". That
> narrowing dropped four of the five stages the class brief names, and everything downstream
> inherited it — the framework's five gates and three rules are all pre-training constructs, and the
> `stage` tag on all 145 catalogue records went unread. See
> [`docs/DESIGN_CRITIQUE.md`](docs/DESIGN_CRITIQUE.md).

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
3. **One zero-dependency page** on the repo's design system, with a chapter per question a reader
   actually asks and three layers in each — a plain headline, the interaction that proves it, and a
   closed "The arithmetic" for anyone who wants the derivation.
4. A **fertility measurement run** — the site's only `measured` numbers.

## The source of truth

The full spec lives in [`docs/`](docs/) — read **[`docs/README.md`](docs/README.md)** first, then
`TODO.md` (execution order), `FRAMEWORK.md`, `DECISIONS.md`, `DESIGN.md`, `FERTILITY_MEASUREMENT.md`,
and `ATLAS.md` (the research). Open decisions the docs deliberately leave for a human are tracked in
[`docs/OPEN.md`](docs/OPEN.md).

## Status

**Scaffold + spec.** The docs and the seed data spine are in place; the pipeline, invariants, and site
are the phased build (`docs/TODO.md`).
