# PROGRESS — 01 · Introductions

The plan, the state, and the evidence. `REQUIREMENTS.md` (local only) holds the requirement text;
this file holds what was built from it and how far it got.

**Written retrospectively on 2026-09-05**, from the commit history, the tracked files and the test
suite — this exercise ran to completion before the convention of keeping a progress ledger existed.
Everything below cites what it was read from. **Where the record does not settle something it says
so** rather than filling the gap: a plausible stage that never happened would be worse than an
admitted blank, because this file is the repository's own account of itself.

**Deliverable:** one public, hosted link. Four interactive in-browser proofs, each a small neural
network the reader trains in their own browser.

**Contract, in one line: nothing is asserted that the page cannot demonstrate live.** Every claim is
something the reader watches happen, not a figure quoted at them.

---

## Status

| stage | what it delivers | state |
| --- | --- | --- |
| **1 · Scaffold** | the uv workspace, the repo conventions, CI, and the first webapp | **done** — `5454b98`, 2026-07-03 |
| **2 · The four proofs** | redesigned as four interactive, dependency-free proofs | **done** — `94dd4e2`, 2026-07-04 |
| **3 · Hosting and registration** | migrated off Netlify to Vercel, production gated behind a workflow, **and** the site root created carrying this exercise's card. `netlify.toml` was *renamed* into `deploy/netlify/`, not deleted | **done** — `29585d4`, `a454113`, 2026-07-09 |
| **4 · Landing page rework** | the site root rewritten and the project copy refined — the page already existed | **done** — `7f7b821`, 2026-07-10 |
| **5 · Design system** | every page onto one Apple-style system, then shared tokens site-wide | **done** — `2a5605e` 2026-07-10, `6969f92` 2026-08-05 |
| **6 · Documents** | README as the guide rather than the map; two stale claims fixed | **done** — `4b9625c`, `7e1a6b0`, 2026-08-24 |
| **7 · Notebook** | a Colab notebook and a builder for it | **done** — `7f3e352`, 2026-08-24. Both **local only**; see below |
| **8 · Confidentiality** | source vocabulary removed from tracked prose | **done** — `09c9619`, `535ab95`, 2026-09-02 |
| **9 · Retro-fix** | dark themes, a page that threw before its first statement, the tab contract | **done** — `40e01cd` (#99), `5232586` (#121), `5835f31` (#114), 2026-09-03/04 |
| **10 · Submit** | PK's action | **no record** — see *Submission* below |

**21 commits touch this exercise**, from `5454b98` (2026-07-03) to `5835f31` (2026-09-04).
**113 tests** collect under `src/exercises/01-introductions`. The page returns **HTTP 200**
anonymously at `https://llm-pretraining-demos.vercel.app/01-introductions/`.

## What is here, and what is deliberately not

Five pages — `web/index.html` plus `s1.html` through `s4.html`, one per proof. **There is no Python
package.** The proofs are hand-written browser JavaScript, and rebuilding them in numpy would be a
second implementation that drifts from the site and then teaches what the site does not do. The
tests are Python, and they check the shipped pages rather than a parallel model.

`notebooks/S01-introductions.ipynb` and `tools/build_notebook.py` both exist on a working checkout
and **neither has ever been tracked on any ref** — verified with `git log --all`. That is the
convention, not an omission: a notebook is course material, and a builder is the same material in
another form. The notebook embeds the shipped pages and runs this exercise's own test suite rather
than re-implementing the proofs.

## What this exercise cannot establish

- **Nothing about scale.** Four networks small enough to train in a browser tab in seconds say
  nothing about what happens at a billion parameters. They demonstrate mechanisms, not behaviours.
- **Nothing about data.** Every proof runs on a hand-made toy set — the s3 grammar is invented for
  the exercise — so no result here transfers to real text.
- **The demonstrations are not measurements.** They are reproducible on the page and that is the
  whole claim; no figure here is offered as evidence about any other system.

## Submission

**No tracked file records a submission**, and this ledger will not invent one. The page has been
live and anonymous-reachable since the Vercel migration, so the deliverable a public link asks for
exists; whether it was submitted, and when, is PK's knowledge and belongs here once PK says.

## What this record does not settle

- **The order within a day.** Several commits share a date, and the history does not say which
  change was decided first.
- **Why the four proofs are those four.** The choice is visible in `94dd4e2`'s result and argued
  nowhere in a tracked file.
- **What was tried and abandoned.** Only what landed is in the history; a rejected approach that
  never reached a commit left no trace.
- **The five-day gap between 2026-07-04 and 2026-07-09**, and the four-week gap to 2026-08-05,
  are not accounted for by anything in this exercise.
