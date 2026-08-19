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

## Every session builds a Colab notebook — locally, never tracked

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
- **Outputs are stripped, always.** Executed outputs bloat diffs, and on any exercise that touches
  real data they bake PII or licensed text into the file.

Write it to be read at two depths: plain what-and-why before each step, the arithmetic and caveats
after it. It is the artifact people learn from and teach from, not a run log.

**They are gitignored** (`notebooks/S[0-9][0-9]-*.ipynb`) — built from the exercise's
`tools/build_notebook.py`, kept on a working checkout, never versioned.

**An exercise that untracks its notebook needs that builder first.** Exercise 05 has one; exercise
04 does not, so when its notebook left the working tree on a branch switch there was nothing to
rebuild it from and it had to be recovered out of git history (`68abb44^`). Untracking a file whose
only copy is the one in front of you is not a workflow, it is a countdown. Write the builder, then
untrack. A notebook is derived from
the package it imports, so tracking it means versioning a second copy of numbers the modules
already own, and the one that drifts is the one nobody regenerates.

The cost is real and worth naming: every rule above is checked by tests that read the notebook,
and on a fresh clone there is nothing for them to read, so **they all skip**. A rule that only
skips is not a rule. What stops that from adding up to no coverage is `notebooks/hello.ipynb` — a
tracked, stdlib-only sample that CI executes top to bottom. It cannot check that a session notebook
is correct; it checks that a notebook in this repo opens and runs, which is the part CI can still
see. Anything stronger has to be run by whoever has the notebook, before the PR.

## Three data concerns — keep them physically separate

- **Briefs → never tracked, at any level.** `BRIEF.md` is gitignored by name everywhere, as is
  programme-level material — the schedule, the class list, the internal authoring specs
  (`docs/BRIEF.md`, `docs/SESSIONS.md`, `docs/EXPLAINER_*.md`). A brief is the course's text and
  is input for whoever builds the exercise; it is not the deliverable. **Never link to one from a
  tracked file** — the link resolves on a working checkout and 404s for everyone else. What we
  *decided*, and why, is published instead: `README.md`, and a tracked `DECISIONS.md` when the
  reasoning needs room (see `04-data-cleaning-dedup/DECISIONS.md`).
- **Session notebooks** → top-level `notebooks/`, **gitignored** except the tracked
  `hello.ipynb` sample (+ the `tools/build_notebook.py` that regenerates each one).
- **Datasets** → top-level `data/`, **gitignored** (+ a tracked manifest/download script). A
  fetcher **verifies the licence at fetch time from the source itself**, not from our own
  catalogue, and refuses anything that declares none — an unverifiable licence is not a permissive
  one. It also records what each dataset *stands in for* when it is a proxy for something else.
  See `05-datamixtures-and-curriculum/tools/fetch_proxy_corpus.py`.
- **Outputs** (plots/checkpoints/logs) → `<exercise>/artifacts/`, **gitignored**.
- **Measured evidence a document renders** → `<exercise>/results/`, **tracked**. This is the
  exception to the line above and it matters: if a published figure comes from a run, the run's
  output has to survive a clone or the document cannot be rebuilt or checked. Exercise 05 writes
  `results/step0.json` and renders `EXPERIMENTS.md` from it. **A run that writes only to
  `artifacts/` leaves the committed evidence untouched and nothing fails** — the documents keep
  rendering the previous experiment while the terminal shows the new one.

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
- **Test the last line of a long job first.** Three experiments in exercise 05 trained to completion and then died writing their results, because the bundle carried a `torch.device` and `json` cannot encode one — one run lost fifteen trained models to its final statement. A two-step run that exercises `save()` costs seconds and would have caught all three. The same applies to any expensive job: the write, the upload, the commit at the end are the parts least covered and most costly to get wrong.
- **Data-handling invariants are enforced in CI, not in review.** `03-data-collection-framework` defines five that any agent touching a data pipeline should know exist (`tests/test_invariants.py`, full table in that exercise's `docs/README.md`): training never touches eval data · nothing excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a measurement must name what produced it · no source content is silently dropped. Each is paired with a test proving it *fails* when broken — a guard nobody has watched fail is not a guard.

## Reporting a measurement

Three rules, each learned by getting it wrong in `02-tokenization` (see that exercise's `CLAUDE.md`):

- **Establish the noise floor before you rank anything.** A held-out score there swung 9,421 points across the five possible splits while the recipes it was meant to separate sat 648 apart. One split looked decisive; five showed the test could not rank at all. Before quoting a comparison, re-run it under a different arbitrary choice — a different split, seed, or slice — and check the effect survives.
- **Sweep without gaps.** A weight sweep that went 2 → 5 → 6 confidently named ×6 the optimum; filling in ×3 and ×4 moved it to ×3, which was better on every stable measurement. A coarse sweep does not report "roughly the optimum", it reports the wrong one.
- **Report the number the metric ignores.** Any score that rewards a *ratio* or a *gap* can be improved by making the denominator worse. Print the absolute quantity next to it — there, total tokens beside the fairness score — so buying the metric is visible rather than inferred.

When one of these overturns a published claim, correct it where the claim was made and say what changed. A quietly amended number is worse than the original error.

- **Prose that states a number is generated too, or it goes stale while the table beside it stays right.** This is the failure that has cost this repo the most edits. A generated table under a hand-written sentence looks maintained, and only the sentence is wrong — so a reader believes the sentence. Session 5 shipped documents reading "across three lanes", "H3 came back qualified", "Thirteen invariants" and "one verdict did not survive its own noise", every one of them contradicting a correct table directly above or below it, and no test failed. If a sentence contains a count, a verdict or a size, derive it from the same source the table uses. Where prose genuinely must stay hand-written — a row in the root README's exercise table — a test asserts the number in it.

- **An experiment that cannot see a lane is not evidence about that lane.** Exercise 05's proxy dropped the three lanes it had no text for, and one hypothesis read `qualified` for two weeks because the lane its refutation clause tested was absent. Funding the lane flipped it to `refuted` with the effect size essentially unchanged. **A missing input does not make a hypothesis safer, it makes it untestable — and untestable reads as passing.** Before trusting a result, list what the measurement was blind to.

Two more that cost this repo real defects:

- **A new module is not done until every list that names modules includes it.** `explainer.py` shipped and stayed missing from three places — the README's *Run it*, the README's layout block, and the exercise's `CLAUDE.md` — none of which any test checks. The consequence was not cosmetic: a reader regenerating the site would have run `widget` without `explainer` and published a page whose figures contradicted its own tool.
- **Render a diagram before committing it.** A Mermaid block is not verified by reading it. A semicolon inside a `Note over` is a statement separator, so the note terminated mid-sentence and GitHub would have rendered a parse error where a diagram should be — caught only by running it through `npx @mermaid-js/mermaid-cli`. The same applies to every number inside one: read them back from the code.

## Documentation is written for more than one reader

A document that only makes sense to whoever built it is not documentation, it is a note to self.
Exercise 05 shipped every graded item, a proxy run and four experiments, and its own contributor
could not tell from any file what `H1`, `E2`, *arm* or *bits per byte* meant. Everything was
correct and nothing was legible.

**Write for three readers, and say which one each section is for.**

| reader | what they need |
| --- | --- |
| **Meeting it for the first time** | What problem this solves, in plain words, before any table. What the jargon means. What was actually done — not the abstraction, the concrete thing: which model, how big, which data, how measured. |
| **A contributor who has to change it** | How the pieces fit and in what order. Where a number comes from. Which module to edit for which effect. Diagrams, because a pipeline described in prose has to be reassembled in the reader's head every time. |
| **A reviewer deciding whether to believe it** | The measurement, its noise floor, what it could not see, and what would falsify it. Limits stated where the numbers are, not in a closing paragraph. |

The rules that follow from it:

- **Every term used as shorthand is defined in exactly one findable place, and everything else links there.** `SPEC.md` is the decision; `METHOD.md` is the apparatus. Splitting them is deliberate — an adversarially-graded specification cannot carry a glossary and two architecture diagrams without paying for it, and a first-time reader cannot do without them.
- **Explain the metric, not just its name.** "Held-out BPB, lower is better" names a measure. What it measures, what it is divided by, and why *that* denominator, is the part that lets a reader judge the table.
- **State the scale and the limits in the open text.** Not inside a collapsed disclosure. A qualifier a reader has to go looking for is a qualifier the document is hiding — and the scale of a proxy is the most important thing on the page it appears on.
- **The artefact people open first needs the grounding too.** A deployed page is read far more often than a specification. If its vocabulary is only defined in a Markdown file, it is not defined.
- **Render every diagram before committing it, and test that it renders.** A mermaid block is not verified by reading it.

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

- **An interaction must never be the only route to a lesson.** Exercise 05's predict-before-reveal block was written with its transferable point inside the reveal, so a reader who declined to guess never reached it — and neither would any print or reduced-motion reader. The interaction may earn a point more vividly; the point itself belongs in prose that is always visible. The same rule is why a page's limitations sit in the open text and not inside a collapsed `<details>`: **a limitation a reader has to open a drawer to find is a limitation the page is hiding.**
- **Editing non-ASCII HTML** (`—`, `→`, `·`, math glyphs): use the Edit/Write tools. **Never** `perl -0pi`/`sed` with wide-char escapes — byte-mode rewrites double-encode UTF-8 into mojibake.

## Instruction files (this system)

- `AGENTS.md` (this file) is the single source of truth. `CLAUDE.md` = `@AGENTS.md`. `.github/copilot-instructions.md` and `.cursor/rules/conventions.mdc` point here.
- Component-specific notes live in a nested `CLAUDE.md` inside that exercise folder.
- Machine-enforceable rules live in tooling (`pyproject.toml`), not prose — this file references the tooling rather than restating it.
