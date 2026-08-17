# AGENTS.md

Canonical conventions for this repo, shared across all coding agents (Claude Code reads it
via `CLAUDE.md`'s `@AGENTS.md` import; Cursor/Copilot via the pointer files). Keep it short.

## What this repo is

An **LLM pre-training** project: building a language model from scratch, one topic at a time —
hands-on experiments plus a flagship training run. Each topic's work lives in a numbered exercise
folder under `src/exercises/`.

## Environment

- **uv workspace**, Python **3.12**. One shared root `.venv`, one `uv.lock`.
- `uv sync --all-packages` — install every workspace member + its deps into the shared `.venv` (plain `uv sync` installs only the root). `uv run <cmd>` — run inside the env. `uv add <pkg>` — add a dep (never hand-edit `uv.lock`).
- Each exercise is a workspace member matched by `members = ["src/exercises/[0-9][0-9]-*"]`.

## Repo layout & naming

- **Exercise folders:** `src/exercises/NN-slug/` — numeric, **zero-padded**, slugged (e.g. `01-introductions`). Zero-pad so lexical sort = numeric order.
- **Identical skeleton per exercise:** `BRIEF.md` (assignment — **local only, gitignored**) · `README.md` (what/how) · `pyproject.toml` (member) · code in one place (`src/` or `web/`) · `artifacts/` (gitignored outputs). Long reasoning gets its own tracked `DECISIONS.md`.
- **Shared code:** deferred — add `src/common/` (its own member) only when a 2nd exercise needs to reuse something. No premature abstraction.
- **Notebooks:** top-level `notebooks/`, one per session — see below.

## Every session ships a Colab notebook

`notebooks/SNN-slug.ipynb`, zero-padded session id first (`S04-data-cleaning-dedup.ipynb`), so
lexical sort = session order. **A session's work is not done until its notebook runs the shipped
code end to end.** Four rules keep it from rotting:

- **It imports the exercise's package; it never re-implements it.** A notebook that copies logic
  drifts from the pipeline within a week and then teaches the wrong thing. Importing means it
  cannot disagree with what ships.
- **Colab-first.** Cell one clones the repo and installs the exercise; the badge at the top opens
  it. It must run on a free tier with no local setup, because that is where it gets used.
- **A `lite` profile finishes in under ten minutes**, with the full run one variable away. A
  notebook nobody waits for is a notebook nobody runs.
- **Outputs are stripped before commit.** Committed outputs bloat diffs, and on any exercise that
  touches real data they can bake PII or licensed text into a tracked file.

Write it to be read at two depths: plain what-and-why before each step, the arithmetic and caveats
after it. It is the artifact people learn from and teach from, not a run log.

## Three data concerns — keep them physically separate

- **Briefs → never tracked, at any level.** `BRIEF.md` is gitignored by name everywhere, as is
  programme-level material — the schedule, the class list, the internal authoring specs
  (`docs/BRIEF.md`, `docs/SESSIONS.md`, `docs/EXPLAINER_*.md`). A brief is the course's text and
  is input for whoever builds the exercise; it is not the deliverable. **Never link to one from a
  tracked file** — the link resolves on a working checkout and 404s for everyone else. What we
  *decided*, and why, is published instead: `README.md`, and a tracked `DECISIONS.md` when the
  reasoning needs room (see `04-data-cleaning-dedup/DECISIONS.md`).
- **Datasets** → top-level `data/`, **gitignored** (+ a tracked manifest/download script).
- **Outputs** (plots/checkpoints/logs) → `<exercise>/artifacts/`, **gitignored**.

## Python style (enforced by ruff — see `pyproject.toml`)

- PEP 8 style + naming: `snake_case` funcs/vars, `PascalCase` classes, `UPPER_SNAKE` constants, `_private` prefix.
- PEP 257 google-style docstrings on public modules/classes/functions.
- Modern typing (PEP 585/604): `list[int]`, `X | None` — no `typing.List`/`Optional`, no `from __future__ import annotations` on 3.12. Type all public signatures.
- Idioms: `pathlib` over `os.path`; f-strings; `logging` over `print` in library code; `if __name__ == "__main__":` guards.
- One `config.py` `@dataclass` per exercise; prefer dataclasses/dicts + small pure functions over deep class hierarchies; shallow trees.
- **Run before committing:** `uv run ruff check --fix .` and `uv run ruff format .`.

## Tests

- Each exercise owns `tests/`; `uv run pytest` from root tests everything.
- Split fast **unit** vs slower **integration** (`@pytest.mark.integration`). Run fast only: `uv run pytest -m "not integration"`.
- **A deployable `web/` is tested in a browser, not just parsed.** `node --check` proves a file has no
  syntax error and nothing more; a call to an undefined function, a filter that silently matches
  nothing, and a headline reading `0` all parse perfectly. Exercise 03's `tests/test_render.py` is the
  pattern: Playwright, integration-marked, loading the built site and asserting what a reader sees.
  One-time setup is `uv run playwright install chromium`; without a browser the suite **skips** rather
  than fails, which keeps a fresh checkout working and means it protects you silently or not at all.
- **A guard that cannot fail is worse than no guard**, because it reads as coverage. Every invariant
  is written twice — once against the real spine, once against a deliberately broken fixture — and
  when you add one, break the thing on purpose and watch it go red before you commit.
- The ML-native integration test: **overfit a single batch for a few steps and assert loss collapses** (+ shape/checkpoint round-trip tests).
- **Data-handling invariants are enforced in CI, not in review.** `03-data-collection-framework` defines five that any agent touching a data pipeline should know exist (`tests/test_invariants.py`, full table in that exercise's `docs/README.md`): training never touches eval data · nothing excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a measurement must name what produced it · no source content is silently dropped. Each is paired with a test proving it *fails* when broken — a guard nobody has watched fail is not a guard.

## Reporting a measurement

Three rules, each learned by getting it wrong in `02-tokenization` (see that exercise's `CLAUDE.md`):

- **Establish the noise floor before you rank anything.** A held-out score there swung 9,421 points across the five possible splits while the recipes it was meant to separate sat 648 apart. One split looked decisive; five showed the test could not rank at all. Before quoting a comparison, re-run it under a different arbitrary choice — a different split, seed, or slice — and check the effect survives.
- **Sweep without gaps.** A weight sweep that went 2 → 5 → 6 confidently named ×6 the optimum; filling in ×3 and ×4 moved it to ×3, which was better on every stable measurement. A coarse sweep does not report "roughly the optimum", it reports the wrong one.
- **Report the number the metric ignores.** Any score that rewards a *ratio* or a *gap* can be improved by making the denominator worse. Print the absolute quantity next to it — there, total tokens beside the fairness score — so buying the metric is visible rather than inferred.

When one of these overturns a published claim, correct it where the claim was made and say what changed. A quietly amended number is worse than the original error.

Two more that cost this repo real defects:

- **A new module is not done until every list that names modules includes it.** `explainer.py` shipped and stayed missing from three places — the README's *Run it*, the README's layout block, and the exercise's `CLAUDE.md` — none of which any test checks. The consequence was not cosmetic: a reader regenerating the site would have run `widget` without `explainer` and published a page whose figures contradicted its own tool.
- **Render a diagram before committing it.** A Mermaid block is not verified by reading it. A semicolon inside a `Note over` is a statement separator, so the note terminated mid-sentence and GitHub would have rendered a parse error where a diagram should be — caught only by running it through `npx @mermaid-js/mermaid-cli`. The same applies to every number inside one: read them back from the code.

## Git workflow

- **Every change lands on `main` via a pull request.** Branch → push → open a PR → merge. **Never push, merge, or force-push directly to `main`** — it's the protected branch that production is promoted from, and the base every PR previews against.
- Keep PRs scoped to one concern; unrelated edits get their own branch/PR.
- **Changelog:** record every user-facing change under `CHANGELOG.md`'s `[Unreleased]` section **in the same PR** (Keep a Changelog + SemVer).
- **Commit & PR messages** carry no AI co-author or session-link trailers — keep the public history clean.

## CI/CD

- CI (`.github/workflows/ci.yml`): `uv sync --all-packages` → `ruff check` → `ruff format --check` → unit → install chromium → integration → `node --check` on every `web/**/*.js`.
- CD: **Vercel**, gated. **Previews auto-deploy per PR**; **production never auto-deploys** (`vercel.json` → `git.deploymentEnabled.main: false`). One project serves every exercise's static `web/` under its slug (`/NN-slug/`) via `deploy/vercel/build.sh` → `public/`. (Netlify was the prior host — deactivated config retained in `deploy/netlify/`, pending decommission.)
- Production deploys go through the reusable `deploy-production.yml` (single source of truth, gated by the `production` environment), invoked two ways: **`deploy.yml`** (`workflow_dispatch`) for an ad-hoc deploy of `main`, and **`release.yml`** for a versioned release.
- **Releasing:** move `CHANGELOG.md`'s `[Unreleased]` → `[X.Y.Z]` (dated) and merge, then `git tag vX.Y.Z && git push origin vX.Y.Z`. `release.yml` creates a GitHub Release from that changelog section and deploys the tagged commit to production.

## Web UI & content

Every deployable exercise's static `web/` bundle shares **one design system** — full reference in `docs/DESIGN.md`. The rules that matter across exercises:

- **Interactive explainers follow two local files.** `docs/EXPLAINER_PROMPT.md` decides *what* one must be (the claim, the interaction that proves it, the topology and family, when **not** to build one). `docs/EXPLAINER_PATTERN.md` records *how* — DOM skeleton, class names, the state-and-render shape, copy voice. Both are gitignored, so they are on a working checkout but not on the remote; read both before building an explainer and don't re-invent the skeleton. Shipped references: `02-tokenization/web/how-it-works.html` and §1 of `03-data-collection-framework/web/chapters.js`.

- **One Apple-style design language** on every page: cool-gray/black surfaces, a single bright-blue accent (`#0071e3` light / `#2997ff` dark), system sans (no serif), soft-shadow rounded panels, and a `← Back` pill to the site root. Style light **and** dark via `prefers-color-scheme`. Reuse the token names in `docs/DESIGN.md` — don't invent a per-exercise palette.
- **Write for a general audience.** The public pages are standalone, blog-style demos of an idea — a first-time visitor should be able to enjoy them without any course context. Favor plain, explanatory copy; the numbered topic eyebrow (`NN · Topic`) makes a nice light section label.
- **Credit the source course in one place.** A single **Credits** section at the bottom of the root `README.md` gives clear, warm credit to the course, instructor, and platform. Keeping it in one prominent spot — rather than repeating it across pages — keeps both the credit and the demos easy to read.
- **Canvas state changes animate** — morph with a short eased transition (≈550ms), not an instant redraw, keeping the framing stable so panels don't resize mid-toggle.
- **Editing non-ASCII HTML** (`—`, `→`, `·`, math glyphs): use the Edit/Write tools. **Never** `perl -0pi`/`sed` with wide-char escapes — byte-mode rewrites double-encode UTF-8 into mojibake.

## Instruction files (this system)

- `AGENTS.md` (this file) is the single source of truth. `CLAUDE.md` = `@AGENTS.md`. `.github/copilot-instructions.md` and `.cursor/rules/conventions.mdc` point here.
- Component-specific notes live in a nested `CLAUDE.md` inside that exercise folder.
- Machine-enforceable rules live in tooling (`pyproject.toml`), not prose — this file references the tooling rather than restating it.
