# AGENTS.md

Canonical conventions for this repo, shared across all coding agents (Claude Code reads it
via `CLAUDE.md`'s `@AGENTS.md` import; Cursor/Copilot via the pointer files). Keep it short.

## What this repo is

An **LLM pre-training** project: building a language model from scratch, one topic at a time —
hands-on experiments plus a flagship training run. Overview: `docs/BRIEF.md`. Each topic's work
lives in a numbered exercise folder under `src/exercises/`.

## Environment

- **uv workspace**, Python **3.12**. One shared root `.venv`, one `uv.lock`.
- `uv sync --all-packages` — install every workspace member + its deps into the shared `.venv` (plain `uv sync` installs only the root). `uv run <cmd>` — run inside the env. `uv add <pkg>` — add a dep (never hand-edit `uv.lock`).
- Each exercise is a workspace member matched by `members = ["src/exercises/[0-9][0-9]-*"]`.

## Repo layout & naming

- **Exercise folders:** `src/exercises/NN-slug/` — numeric, **zero-padded**, slugged (e.g. `01-introductions`). Zero-pad so lexical sort = numeric order.
- **Identical skeleton per exercise:** `BRIEF.md` (assignment) · `README.md` (what/how) · `pyproject.toml` (member) · code in one place (`src/` or `web/`) · `artifacts/` (gitignored outputs).
- **Shared code:** deferred — add `src/common/` (its own member) only when a 2nd exercise needs to reuse something. No premature abstraction.

## Three data concerns — keep them physically separate

- **Briefs/docs** → tracked (`docs/BRIEF.md` program-level, `<exercise>/BRIEF.md` per exercise).
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
- The ML-native integration test: **overfit a single batch for a few steps and assert loss collapses** (+ shape/checkpoint round-trip tests).

## Git workflow

- **Every change lands on `main` via a pull request.** Branch → push → open a PR → merge. **Never push, merge, or force-push directly to `main`** — it's the protected branch that production is promoted from, and the base every PR previews against.
- Keep PRs scoped to one concern; unrelated edits get their own branch/PR.

## CI/CD

- CI (`.github/workflows/ci.yml`): `uv sync --all-packages` → `ruff check` → `ruff format --check` → unit → integration → `node --check` on any `web/js/*.js`.
- CD: **Vercel**, gated. **Previews auto-deploy per PR**; **production is on-demand** — the `Deploy to production` workflow (`.github/workflows/deploy.yml`, `workflow_dispatch`, gated by the `production` environment) runs `vercel build/deploy --prod`. `main` does **not** auto-deploy (`vercel.json` → `git.deploymentEnabled.main: false`). One project serves every exercise's static `web/` under its slug (`/NN-slug/`) via `deploy/vercel/build.sh` → `public/`. (Netlify was the prior host — deactivated config retained in `deploy/netlify/`, pending decommission.)

## Instruction files (this system)

- `AGENTS.md` (this file) is the single source of truth. `CLAUDE.md` = `@AGENTS.md`. `.github/copilot-instructions.md` and `.cursor/rules/conventions.mdc` point here.
- Component-specific notes live in a nested `CLAUDE.md` inside that exercise folder.
- Machine-enforceable rules live in tooling (`pyproject.toml`), not prose — this file references the tooling rather than restating it.
