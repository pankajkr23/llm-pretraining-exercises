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
- **Identical skeleton per exercise:** `BRIEF.md` (assignment) · `README.md` (what/how) · `pyproject.toml` (member) · code in one place (`src/` or `web/`) · `artifacts/` (gitignored outputs).
- **Shared code:** deferred — add `src/common/` (its own member) only when a 2nd exercise needs to reuse something. No premature abstraction.

## Three data concerns — keep them physically separate

- **Briefs/docs** → tracked per exercise (`<exercise>/BRIEF.md`). Programme-level material — the
  schedule, the class list, the internal authoring specs — is **local only** and gitignored
  (`docs/BRIEF.md`, `docs/SESSIONS.md`, `docs/EXPLAINER_*.md`). Never link to them from a tracked
  file: the link is dead for everyone but us.
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
