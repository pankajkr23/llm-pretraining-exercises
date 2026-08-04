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
  `docs/TODO.md`, they're local working files (the root `.gitignore` excludes `data/` and `TODO.md`).
  Don't "fix" that by editing `.gitignore`; don't assume they exist on a fresh clone or in CI, and
  never re-derive the catalogue by hand. `artifacts/` is ignored too. The tracked, shipped artifact is
  `web/data.json` — the precomputed, provenance-typed pipeline output the static site reads.
- **Every number is provenance-typed** `{value, unit, provenance: "measured"|"estimated"|"unknown", source}` —
  never render a bare number; never invent a figure (unknown → say so). See `docs/DESIGN.md` §6.
- **The pages carry interactive explainers, not widgets.** An explainer teaches one idea: framing
  line → a title that is a question → prose with the key numbers bold → controls whose presets are
  real alternatives from the data → live stat tiles showing their own arithmetic → a chart with
  named zones and a marker → a callout that interprets the *current* state → a footnote on
  provenance. Anything that only draws a shape is unfinished. Spec and worked examples:
  `docs/DESIGN.md` §4; code: `web/report/` §11, §6, and §3 for the honest-gap variant.
- **Web = zero runtime dependencies** (hand-written SVG/CSS/vanilla JS), inherits the repo design
  system (`docs/DESIGN.md` at repo root), served per-slug at `/03-data-collection-framework/`. Ship a
  `NOTICE` disclaiming org affiliation. Non-ASCII glyphs: edit with Edit/Write, never byte-mode `perl`/`sed`.
- **Naming:** exercise slug + package are `03-data-collection-framework` / `dataframework` (the docs
  assume this). Keep the docs' counts consistent when you touch them — the enumerated lists in
  `TODO.md` (14 explorers + 8 intuitions; 12 Decision sections; five invariants) are canonical.
