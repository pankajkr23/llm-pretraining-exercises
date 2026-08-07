# CLAUDE.md — 03-data-collection-framework

Component notes. Repo-wide conventions: root `AGENTS.md`.

- **The spec in [`docs/`](docs/) is the source of truth.** Read `docs/README.md` first, then
  `docs/TODO.md` (execution order, phases 0–8). `docs/ATLAS.md` is the research being ingested;
  `docs/DECISIONS.md` holds the resolved numbers; `docs/DESIGN.md` is the site build spec;
  `docs/OPEN.md` tracks the decisions deliberately left open.
- **Python package** (src layout) at `src/dataframework/`, installed editable via the uv workspace
  (matched by `members = ["src/exercises/[0-9][0-9]-*"]`). Add deps with `uv add` (never hand-edit
  `uv.lock`). Pipeline entry point (`python -m dataframework`) arrives in Phase 2.
- **The five invariants (INV-1…INV-5)** are enforced in **CI tests**, not prose (train never touches
  eval · no grade-X in a commercial mix · every judgment has reasoning+citation+confidence · fertility
  measured not annotated · no Atlas content silently dropped). They come **before** any UI
  (`docs/TODO.md` Phase 3). Full table in `docs/README.md`.
- **Data spine:** `data/seed/master_dataset_catalog.csv` (145 rows) + `data/seed/benchmarks.csv`
  (31 rows) are the machine-readable extract of `docs/ATLAS.md`. **Untracked by design** — along with
  `docs/TODO.md` and `docs/NEW_TODO.md`, they're local working files (the root `.gitignore` excludes
  `data/`, `TODO.md` and `NEW_TODO.md`).
  Don't "fix" that by editing `.gitignore`; don't assume they exist on a fresh clone or in CI, and
  never re-derive the catalogue by hand. `artifacts/` is ignored too. The tracked, shipped artifact is
  `web/data.json` — the precomputed, provenance-typed pipeline output the static site reads.
- **`ingest` is a separate stage from the export.** `python -m dataframework` reads `catalog.json`;
  it does not rebuild it. Change anything in `ingest.py` — a size correction, a dataset relationship,
  the distribution register — and you must run `python -m dataframework.ingest` first, or the bundle
  is rebuilt from a stale spine and silently shows the old values.
- **One rule, one implementation.** Several rules live twice, in Python for the bundle and in JS for
  the page: `blockers()`/`blockersOf`, `tier_of`/`tierOfIn`, `_tokens_for`/`countableTokens`, the
  containment filter. That duplication shipped a wrong figure once (correction X28: the bundle was
  right and the browser ignored it). `tests/test_invariants.py` now runs the browser's own functions
  against the real bundle and fails on disagreement — change both halves in the same commit.
- **The browser is tested, not just parsed.** `tests/test_render.py` (integration-marked) loads the
  built site in Playwright and asserts what a reader sees: chapters render, no headline figure reads
  as nothing, no sideways scroll at 1500/900/390px, no label silently cut off. Needs
  `uv run playwright install chromium` once; without it the suite **skips**, so it stops protecting
  you quietly. A guard nobody has watched fail is not a guard — and one that *cannot* fail is worse,
  because it reads as coverage.
- **Modality and domain are a third and fourth lens** on the same tokens, in `modalities.py`, and are
  deliberately not merged with the tier: a tier says where text came from, a modality what kind of
  thinking it teaches, a domain what it is about. The tier weights and curriculum emphases are a
  *proposal* and are typed `estimated`; the domain coverage counts are measured, and each ships the
  pattern it was matched by so the count is checkable.
- **`data.json` is an index under a 100KB budget** (96.4KB now). Prose that is read on hover or inside
  a `<details>` belongs in `records.json`; repeated source strings are stored once and referenced by
  index. The build prints a warning when it goes over — do not silence it, cut something.
- **Every number is provenance-typed** `{value, unit, provenance: "measured"|"estimated"|"unknown", source}` —
  never render a bare number; never invent a figure (unknown → say so). See `docs/DESIGN.md` §6.
- **The pages carry interactive explainers, not widgets.** An explainer makes one claim, and the
  thing the reader does is the *proof* of that claim, not an illustration of it. If a static image
  would teach the same thing, write a paragraph instead. Anything that only draws a shape is
  unfinished. Read `../../../docs/EXPLAINER_PROMPT.md` (what to build, and when **not** to) and
  `../../../docs/EXPLAINER_PATTERN.md` (how — DOM, class names, state-and-render shape, voice)
  before starting one. Both are gitignored — present locally, absent from the remote. **The site is one page**: `web/index.html` is the shell and `web/chapters.js`
  holds **thirteen chapters plus an appendix**, one per reader question, each in three layers —
  headline, interaction, and a closed "The arithmetic". `web/report/` and `web/reasoning/` are redirect stubs from the two-page
  version. Reference implementation: the contamination gate (chapter 8), reader-supplied input,
  verified against `dataframework/shingles.py`. What the first build got wrong, and why, is in
  `docs/DESIGN_CRITIQUE.md` — local and gitignored, like `docs/NEW_TODO.md`; the rebuild has
  shipped, so both are a record rather than a queue.
- **Web = zero runtime dependencies** (hand-written SVG/CSS/vanilla JS), inherits the repo design
  system (`docs/DESIGN.md` at repo root), served per-slug at `/03-data-collection-framework/`. Ship a
  `NOTICE` disclaiming org affiliation. Non-ASCII glyphs: edit with Edit/Write, never byte-mode `perl`/`sed`.
- **Naming:** exercise slug + package are `03-data-collection-framework` / `dataframework` (the docs
  assume this). Keep the docs' counts consistent when you touch them — **thirteen chapters plus an
  appendix**, one per reader question, and **five invariants**. `docs/TODO.md`'s section counts
  describe the retired two-page build, and both it and `docs/NEW_TODO.md` are local, untracked
  planning files. What is genuinely still open lives in `docs/OPEN.md`, which *is* tracked.
