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

## MANDATORY — never remove anything under `notebooks/` or any `tools/`

**Nothing in `notebooks/` or in any `src/exercises/*/tools/` directory may be deleted, moved,
renamed, or overwritten — locally or on the remote — without PK's explicit prior permission naming
the file.** This is not a style preference and it does not yield to a tidy-up, a "stale" file, or a
rule elsewhere in this document. If something there looks wrong, **say so and stop.**

These are the only files in the repo with **no second copy**. Both are gitignored, so git cannot
restore them: `notebooks/S[0-9][0-9]-*.ipynb` and `src/exercises/*/tools/build_notebook.py`. A
deletion here is permanent in a way that no other deletion in this repo is.

**This now covers `BRIEF.md` too.** All four of exercises 01–04's briefs were destroyed by an
ordinary branch switch after the commit that untracked them, and were recoverable only because
`18015b1^` was still reachable. A brief written *after* the untracking convention has no such
safety net.

**Untracking a file is what makes it fragile, and the mechanism is worth understanding.** `git rm
--cached` plus a `.gitignore` entry leaves the working copy in place — but the *next* `checkout` or
`pull` that crosses the untracking commit sees a file that was tracked at the old HEAD and is not at
the new one, and deletes it. Nobody deleted anything. So:

- **Back up every local-only file outside the repo — there is now a tool, so do not do it by hand:**
  ```bash
  uv run python tools/backup_local_only.py            # snapshot + commit, outside the repo
  uv run python tools/backup_local_only.py --verify   # is the store current? non-zero if not
  ```
  It writes a **git** store at `../.llm-pretraining-exercises-local-only`, so every *version* is
  kept, not just the latest. That matters more than it sounds: these files are regenerated
  constantly, so the likelier loss is a **bad overwrite**, and a plain copy would faithfully replace
  the good version with the broken one. Run it before any branch switch and after any session that
  rebuilds a notebook.

- **The protected set is wider than the three classes named above, and the extra ones were
  unguarded for months.** `docs/sessions/**` is the entire course corpus — every session's notes,
  transcripts and assignments, including sessions this repo has not reached — and
  `docs/EXPLAINER_PROMPT.md` / `docs/EXPLAINER_PATTERN.md` are the two documents any explainer is
  required to be built from. All gitignored, none regenerable, none watched by the tripwire until
  now. **85 files, 12 MB.** A guard that covers the documented cases and misses the largest one
  reads as coverage without being any.

- **After any branch switch, pull, merge, rebase or stash, run the tripwire** —
  `uv run pytest tests/test_local_only_files_present.py`. It fails when *some* of these files are
  present and others gone, and skips when all are absent (a clone, not a loss).
- **Recovery, in the order to try it:**
  ```bash
  uv run python tools/backup_local_only.py --verify              # 1. does the store have it?
  cp ../.llm-pretraining-exercises-local-only/<path> <path>      #    restore the latest
  git -C ../.llm-pretraining-exercises-local-only log -- <path>  #    or an earlier version
  git show <untracking-commit>^:<path> > <path>                  # 2. e.g. 18015b1^ for the briefs
  ```
  Step 2 works only while the removal commit is still reachable, which is why step 1 exists.

**Prohibited without permission, on these paths:** `rm` · `git checkout -- ` · `git restore` ·
`git reset --hard` · moving or renaming · writing over an existing notebook from anything other
than a deliberate builder run — plus the two below, whose **flags** are the part that matters:

- **`git clean -x` or `-X`** deletes every file here in one command. Plain `git clean -fd` does
  not, because these paths are *ignored* rather than merely untracked. Naming the command without
  the flag teaches the wrong lesson: the danger is precisely `-x`/`-X`.
- **`git stash -a` / `--all`** stashes ignored files, which is all of these. Plain `git stash -u`
  is **safe** here for the same reason. This document said the opposite for months.

**And the branch hazard is retired while the tag hazard is live.** The paths are untracked on every
branch now, so a branch switch no longer destroys them — but **eleven tags still carry them**:
`v0.1.0` through `v0.6.2` track between one and five of these files each. `git checkout v0.4.0`
overwrites five live files with stale tagged copies, and switching back deletes them. So
`git checkout <tag>`, `git bisect` and any checkout by SHA are now the remaining live path for the
exact failure that has already happened twice — run the tripwire after each of them.

**It has already happened once, and not by anyone deciding to delete anything.** After the builders
were untracked, an ordinary `git checkout main && git pull` destroyed all five: `checkout` restored
them as tracked files from the pre-merge `main`, then the fast-forward applied the commit that
removed them from the index, so git deleted the working-tree copies. Recovered from the removal
commit's parent (`db9b288^`). **So the danger is routine git operations, not carelessness** — after
any branch switch, pull, merge, rebase or stash, verify:

```bash
ls notebooks/*.ipynb src/exercises/*/tools/build_notebook.py
```

and if anything is missing, restore it before doing anything else:

```bash
git checkout "$(git log --all --diff-filter=D --format=%H -1 -- 'src/exercises/*/tools/build_notebook.py')^" \
  -- 'src/exercises/*/tools/build_notebook.py'
```

That recovery works only while the removal commit is still reachable. **Keep a backup outside the
repo** — it is the real safety net.

**Tests must never write to the real paths.** `tests/test_notebook_builders.py` builds through
`NOTEBOOK_OUT` into a temporary directory for exactly this reason: a test that wrote to
`notebooks/` would destroy the only copy that exists.

## Repo layout & naming

- **Exercise folders:** `src/exercises/NN-slug/` — numeric, **zero-padded**, slugged (e.g. `01-introductions`). Zero-pad so lexical sort = numeric order.
- **Identical skeleton per exercise:** `BRIEF.md` (assignment — **local only, gitignored**) · `README.md` (what/how) · `pyproject.toml` (member) · code in one place (`src/` or `web/`) · `artifacts/` (gitignored outputs). Long reasoning gets its own tracked `DECISIONS.md`.
- **Set the folder up BEFORE writing code.** The skeleton is not paperwork to backfill. Exercise 06
  was scaffolded with `pyproject.toml` and modules but no `CLAUDE.md`, `PROGRESS.md`, `NOTICE` or
  `BRIEF.md`, because a convention that lives only in prose gets skipped under momentum.
  `tests/test_exercise_skeleton.py` now checks the universal ones (`README.md`, `CLAUDE.md`,
  `pyproject.toml`, `tests/`, `tools/`) and asserts **no `BRIEF.md` is ever tracked** — checked with
  `git ls-files`, not by reading `.gitignore`, because a file already in the index stays tracked
  whatever the ignore rules say afterwards.
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

**Both the notebook and its builder are gitignored** — `notebooks/S[0-9][0-9]-*.ipynb` and
`src/exercises/*/tools/build_notebook.py`. A generator is the notebook in another form, so tracking
it would keep the same course material in the repo as Python, which is what untracking the notebook
was for. Every exercise has a builder; they live on a working checkout and are never pushed.

**Name the cost, because it is the whole trade.** The builder used to be the recoverable copy — it
is why exercise 04's notebook could be rebuilt after it left the working tree on a branch switch
(recovered from `68abb44^`). Now nothing tracked can restore either one. **Back up
`tools/build_notebook.py` outside the repo**, and treat losing a working tree as losing the
notebooks.

**CI can no longer see them.** It cannot check that an exercise has a builder, nor that a builder
still runs against the package it imports. `tests/test_notebook_builders.py` skips entirely on a
fresh clone and is a *local* gate — run it on the checkout that has the builders, before opening a
PR:

```bash
uv run pytest tests/test_notebook_builders.py
```

It builds through `NOTEBOOK_OUT` into a temporary path, because a test that wrote to the real
location would destroy the only copy of the notebook that exists.

**An exercise with no Python package still gets a notebook, and it still must not re-implement.**
Exercise 01's proofs are hand-written browser JavaScript; rebuilding them in numpy would be a
second implementation that drifts from the site and then teaches what the site does not do. Its
notebook embeds the shipped pages themselves and runs the exercise's own test suite instead. A notebook is derived from
the package it imports, so tracking it means versioning a second copy of numbers the modules
already own, and the one that drifts is the one nobody regenerates.

The cost is real and worth naming: every rule above is checked by tests that read the notebook,
and on a fresh clone there is nothing for them to read, so **they all skip**. A rule that only
skips is not a rule. What stops that from adding up to no coverage is `notebooks/hello.ipynb` — a
tracked, stdlib-only sample that CI executes top to bottom. It cannot check that a session notebook
is correct; it checks that a notebook in this repo opens and runs, which is the part CI can still
see. Anything stronger has to be run by whoever has the notebook, before the PR.

## Five data concerns — keep them physically separate

- **Briefs → never tracked, at any level.** `BRIEF.md` is gitignored by name everywhere, as is
  programme-level material — the schedule, the class list, the internal authoring specs
  (`docs/BRIEF.md`, `docs/SESSIONS.md`, `docs/EXPLAINER_*.md`). A brief is the course's text and
  is input for whoever builds the exercise; it is not the deliverable. **Never link to one from a
  tracked file** — the link resolves on a working checkout and 404s for everyone else. What we
  *decided*, and why, is published instead: `README.md`, and a tracked `DECISIONS.md` when the
  reasoning needs room (see `04-data-cleaning-dedup/DECISIONS.md`).

  **And `BRIEF.md` is not the authority on what submission requires.** It is the course's text and
  it can be truncated, reformatted or pasted short; the submission platform's own field list is what
  grades. Check the platform before calling a session done, and record the required *shape* — not
  the brief's wording — in the exercise's `PROGRESS.md`. A deliverable specified as a **public URL**
  is not satisfied by a file in the repo, however correct that file is.
- **Session notebooks** → top-level `notebooks/`, **gitignored** except the tracked
  `hello.ipynb` sample. Their generators (`*/tools/build_notebook.py`) are gitignored too — a
  generator is the same material in another form. Keep a backup outside the repo; nothing tracked
  can rebuild either. This does **not** extend to other `tools/` scripts: a dataset fetcher such as
  `05-…/tools/fetch_proxy_corpus.py` stays tracked, because a corpus needs a tracked way to fetch
  and licence-check it.
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

- **A coverage guard must ask whether a test can RUN, not whether it is listed.**
  `tests/test_ci_shards_cover_everything.py` was written for the obvious failure — a file outside
  every shard is never run and CI is green — and was blind to the adjacent one: a file that *is*
  inside a shard and **collects zero tests there**, because a module-level `pytest.importorskip`
  skips the whole file when the dependency is absent. A file that collects nothing is
  indistinguishable from a file with nothing in it, and `ci.yml` treats pytest's exit code 5 as
  success, so **46 of exercise 06's 272 tests and all 20 of its integration tests ran nowhere for a
  week with every gate green** — plus exercise 05's proxy run. The guard is now **lexical**: it
  reads the filesystem and each file's `importorskip` line, which are facts about the source rather
  than about whatever happens to be installed — the property a coverage guard needs, since the
  environment is precisely what it is making claims about. It keeps a tracked
  `OPTIONAL_DEPENDENCY_GATES` ledger that fails in **both** directions, and asserts every gated file
  is reachable by a job that installs what it needs.
- **An optional heavy dependency is a decision, and the wheel size is usually the whole argument.**
  The default Linux torch wheel is **2,722.7 MB** because it bundles CUDA; the CPU-only build is
  **191.8 MB** and drops 19 packages from the lock. Nothing here trains on a GPU, so the CUDA
  payload bought nothing and made "torch in CI" look unaffordable when it was not. Pin it with a
  platform-scoped index (`[[tool.uv.index]]` + `[tool.uv.sources]` with a `sys_platform == 'linux'`
  marker) rather than a global one: macOS arm64 has no CUDA build, so its PyPI wheel is already CPU
  and pinning there only adds a way for the platforms to disagree. `uv sync` has no
  `--torch-backend` flag — that is `uv pip install` only — so the index is the reproducible route.
  Assert the result (`+cpu` in `torch.__version__`) in the job: if the pin regresses, the CUDA wheel
  installs silently and the only symptom is a slower job.
- The ML-native integration test: **overfit a single batch for a few steps and assert loss collapses** (+ shape/checkpoint round-trip tests).
- **Test the last line of a long job first.** Three experiments in exercise 05 trained to completion and then died writing their results, because the bundle carried a `torch.device` and `json` cannot encode one — one run lost fifteen trained models to its final statement. A two-step run that exercises `save()` costs seconds and would have caught all three. The same applies to any expensive job: the write, the upload, the commit at the end are the parts least covered and most costly to get wrong.
- **Data-handling invariants are enforced in CI, not in review.** `03-data-collection-framework` defines five that any agent touching a data pipeline should know exist (`tests/test_invariants.py`, full table in that exercise's `docs/README.md`): training never touches eval data · nothing excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a measurement must name what produced it · no source content is silently dropped. Each is paired with a test proving it *fails* when broken — a guard nobody has watched fail is not a guard.

- **A tested feature with no caller is dead code wearing a test.** `masks.loss_mask(context_spans=...)` in exercise 06 is implemented, documented, covered by two passing tests and taught in the session notebook — and `grep -rn context_spans` finds **zero** callers in the pipeline: `feed.py` builds every microbatch with the default mask. The tests are green, so the capability reads as a behaviour of the run, and the documents describing prompt/tool-observation masking describe something that never happens. The test proves the function works; only a caller proves the system uses it. When you add a keyword-only option to a library function, either wire it through the one path that would exercise it in a real run, or state in the module docstring that it is offered and unused — and put the same sentence wherever the feature is described to a reader.
- **A coverage guard built on `--collect-only` is blind to a file that collects nothing.** `tests/test_ci_shards_cover_everything.py` catches an integration file in no CI shard, and an integration file in two. It cannot catch the third case: **in a shard, and contributing zero tests.** A module-level `pytest.importorskip("torch")` raises during *collection*, so `pytest --collect-only -q` prints no `path: count` line for that file at all — I verified this with a throwaway module importorskipping an absent package: output was `no tests collected`, exit 0. The file is therefore absent from `everything` and from `owners` alike, `missing` is empty, and `covered == sum(everything.values())` holds trivially. The consequence is live: all 20 of exercise 06's integration tests (`crash` 11, `model` 3, `train` 6) sit behind `importorskip("torch")`, CI never installs the `train` extra, and CI's integration step maps exit 5 to success — so the `rest` shard runs **zero** of them, reports green, and the coverage guard agrees. A guard must count what the job was *supposed* to run, from a list it does not derive from the same run it is auditing.

## Reporting a measurement

Three rules, each learned by getting it wrong in `02-tokenization` (see that exercise's `CLAUDE.md`):

- **Establish the noise floor before you rank anything.** A held-out score there swung 9,421 points across the five possible splits while the recipes it was meant to separate sat 648 apart. One split looked decisive; five showed the test could not rank at all. Before quoting a comparison, re-run it under a different arbitrary choice — a different split, seed, or slice — and check the effect survives.
- **Sweep without gaps.** A weight sweep that went 2 → 5 → 6 confidently named ×6 the optimum; filling in ×3 and ×4 moved it to ×3, which was better on every stable measurement. A coarse sweep does not report "roughly the optimum", it reports the wrong one.
- **Report the number the metric ignores.** Any score that rewards a *ratio* or a *gap* can be improved by making the denominator worse. Print the absolute quantity next to it — there, total tokens beside the fairness score — so buying the metric is visible rather than inferred.

When one of these overturns a published claim, correct it where the claim was made and say what changed. A quietly amended number is worse than the original error.

- **Prose that states a number is generated too, or it goes stale while the table beside it stays right.** This is the failure that has cost this repo the most edits. A generated table under a hand-written sentence looks maintained, and only the sentence is wrong — so a reader believes the sentence. Session 5 shipped documents reading "across three lanes", "H3 came back qualified", "Thirteen invariants" and "one verdict did not survive its own noise", every one of them contradicting a correct table directly above or below it, and no test failed. If a sentence contains a count, a verdict or a size, derive it from the same source the table uses. Where prose genuinely must stay hand-written — a row in the root README's exercise table — the number in it went untested long enough for exercise 06's row to read *"Stage 1 of 8"* while the exercise was at stage 7. `tests/test_doc_counts_match.py` now derives that count from the exercise's own stage table. **The prose around the number is still untested**, so a row can carry a correct stage and a wrong description; verify that by hand on every PR that advances an exercise.

- **An experiment that cannot see a lane is not evidence about that lane.** Exercise 05's proxy dropped the three lanes it had no text for, and one hypothesis read `qualified` for two weeks because the lane its refutation clause tested was absent. Funding the lane flipped it to `refuted` with the effect size essentially unchanged. **A missing input does not make a hypothesis safer, it makes it untestable — and untestable reads as passing.** Before trusting a result, list what the measurement was blind to.

- **Size a proxy corpus against the RUN, not against the mixture's ratios.** Getting the
  proportions right and the total wrong does not shrink the experiment, it changes what the
  experiment is: the run stops measuring a mixture and starts measuring repetition. Exercise 06
  consumes `ranks × accumulation × microbatch × sequence_length × steps` =
  `4 × 2 × 8 × 512 × 320` = **10,485,760** token positions (read it from `Config.total_tokens`,
  never from memory). Its first corpus held **2,185,575** tokens — **4.8 epochs flat**, and once
  shaped to session 5's lane weights, **30.2 epochs of the web lane against 0.41 of the agentic
  lane**. The lane the mixture funded most heavily was the one the model saw thirty times over,
  while the lane it funded least was not seen through even once. No mixture claim survives that, and
  nothing in the pipeline failed: the shards read fine and the loss curve looked normal. **It was
  refetched to 10,649,549 tokens at 1.01 epochs**, and `tools/build_corpus.py` now refuses to build
  below one epoch rather than leaving it to be noticed.
  **So before a run, print `total_tokens ÷ corpus_tokens` per lane and put it next to the mixture
  table.** A lane above ~1 epoch is measuring memorisation; a lane below 1.0 was never fully read.
  Fix it by fetching more text or by cutting `steps` — never by leaving the ratio unstated. A
  fetcher records `rows_requested`, so the corpus size is a decision someone made, which means it is
  a decision someone can be shown.

- **A quantity that is pinned to a constant by construction is not a measurement, and recording it as one is worse than omitting it.** Exercise 06's telemetry writes `pack_util` for every microbatch, and it is always exactly `1.0` — not because packing is perfect but because `feed.py` calls `pack.build_window(index, tokens, span.start, span.end)` with no `window=`, so the window size *is* the span length, `packed[: end - start]` fills the array end to end, no `PAD` is ever written, and `masks.utilization` counts `segments >= 0` over the whole row. The number cannot move. A reader sees a per-batch float in a ledger and reads it as evidence the packer is efficient; the arithmetic says only that the code passed one argument and not another. Before publishing a derived number, ask what input would change it — if none of the run's inputs can, either give it a denominator that varies or delete the field. The same test applies to any ratio whose numerator and denominator are computed from the same object.

Two more that cost this repo real defects:

- **Registering a new exercise: three lists, two of them automatic.** `deploy/vercel/build.sh`
  publishes any `src/exercises/*/web/` on its own, and the workspace glob picks up any
  `NN-slug/pyproject.toml` on its own. The two that are **hand-maintained** are the root README's
  exercise table and the cards in `deploy/vercel/index.html` — an exercise can be deployed and
  reachable while being invisible to anyone arriving at the site root. Both are now checked:
  `tests/test_deploy_registration.py`. **The root README row is checked now** —
  `tests/test_doc_counts_match.py` derives the stage count from the exercise's own stage table and
  asserts the row links its README directly. It exists because the row read *"Stage 1 of 8 — in
  progress"* while the exercise was at stage 7, six stages stale, with nothing red. What it still
  does **not** check is the prose around the number, so treat that as hand-verified on every PR that
  moves an exercise forward.
- **A new module is not done until every list that names modules includes it.** `explainer.py` shipped and stayed missing from three places — the README's *Run it*, the README's layout block, and the exercise's `CLAUDE.md`. Exercise 06 now checks two of those three: `tests/test_trainingdata_docs.py::test_every_module_is_named_in_the_documents_that_list_modules` asserts every `src/trainingdata/*.py` is named somewhere in **both** README.md and CLAUDE.md. It was red when written — `replay.py` had shipped and the README never learned about it — and is green now, including `opus.py` and `opus_score.py`. Copy that guard into any exercise that grows past a handful of modules. Note its limit: it checks the *document*, not the *list*, so a module named once in prose satisfies it while the layout block a reader actually follows stays wrong. The consequence was not cosmetic: a reader regenerating the site would have run `widget` without `explainer` and published a page whose figures contradicted its own tool.
- **Render a diagram before committing it.** A Mermaid block is not verified by reading it. A semicolon inside a `Note over` is a statement separator, so the note terminated mid-sentence and GitHub would have rendered a parse error where a diagram should be — caught only by running it through `npx @mermaid-js/mermaid-cli`. The same applies to every number inside one: read them back from the code.

## The root README is a map; each exercise's README is the guide

Two documents, two jobs, and the failure is always the same one: the root grows a deep-dive per
exercise until two thirds of it is detail that belongs one directory down. It reached 307 lines
that way, 211 of them per-exercise sections whose content existed **nowhere else** — so the root
was not summarising the exercises, it was the only place they were described.

- **Root:** what the repo is, how it is laid out, how to run it, and a one-row-per-exercise table.
  **No per-exercise section, not even for the exercise under submission.** High-level, and short
  enough to read in a minute.
- **Exercise:** everything end to end — the argument, the numbers, how to reproduce, what it cannot
  establish. This is where a reader who wants depth is sent, and it must reward the trip.

The root's job is **routing, not retelling**. Where the brief requires the root to reach a
deliverable "without a detour", that is a property of its links, not of how much it repeats — and
the test for it should assert the *link*, since asserting the filename passes against a front door
that names the file and never links it.

**"Without a detour" is satisfied by a link, not by a section.** The brief for the exercise under
submission says the root README *is* the front door — a grader lands there and nowhere else — and
the obvious reading is that the root should therefore carry a summary block for that exercise. It
should not. That block was tried and it grew back into the retelling the split exists to prevent:
what the work is, the rule behind it, three findings, the proxy result, a routing table. All of it
already existed one directory down, and the root became the second place to keep it correct.

What the requirement actually needs is that **the exercise's own table row links `SPEC.md`
directly**. That is one hop from the line the reader is already on, which is what "without a
detour" means. The row is hand-written prose that states counts. `tests/test_readme_links.py` checks the
links resolve; **nothing checks the numbers**, and exercise 06's row has been six stages stale as a
result. Do not rely on a guard here.

Because the row is the *only* per-exercise detail the root carries, **everything else has to be one
directory down**: if the exercise README is not the complete end-to-end guide, nothing is.

Every exercise README therefore carries a **`## How to read this`** reading path naming all three
readers — first time · changing the code · deciding whether to believe it — plus a runnable command
and a section stating what the work *cannot* establish. `tests/test_readme_structure.py` enforces
all three, and `tests/test_readme_links.py` checks that every relative link and in-page anchor
resolves **from the directory of the file containing it** (three links broke silently when prose
moved out of the root, because a path correct from the repo root is wrong two levels down).

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

## Local gates before a commit exists

`.pre-commit-config.yaml` runs the three gates CI fails on — **gitleaks**, `ruff check`, `ruff
format` — plus merge-conflict, private-key, large-file, YAML and TOML checks. Install once per
clone:

```bash
uv sync --all-packages && uv run pre-commit install     # needs gitleaks: brew install gitleaks
uv run pre-commit run --all-files                        # over everything, not just staged
```

- **This is a feedback loop, not the enforcement point.** A hook is skippable with `--no-verify` and
  is absent on a fresh clone, so **CI still decides**. What it buys is finding a problem in two
  seconds on your machine instead of two minutes inside a pull request.
- **The secret scan fails when gitleaks is missing; it never skips.** A scan that quietly does not
  run is worse than none, because it reads as coverage.
- **No hook may rewrite repository content, and this is not a style preference.** `end-of-file-fixer`
  and `trailing-whitespace` were in the config's first draft; run once, they rewrote
  `02-tokenization/web/tokenizer.json` — the **frozen tokenizer whose bytes are hashed and whose
  hash every shard manifest in exercise 06 pins**. A cosmetic newline would have voided that hash
  and invalidated every manifest, and the diff would have read as tidying. They also rewrote the
  tokenization corpus, which is data. `tests/test_precommit_config.py` asserts neither can return.
- **A digest field must not be named like a credential.** gitleaks' `generic-api-key` rule fires on
  an identifier containing *key*, *token*, *secret* or *api* next to a high-entropy value — so a
  field named `plan_key_digest` holding sixteen hex characters reads as a leaked credential, while
  the same value under `plan_digest` does not. Content
  digests are public by construction and a committed ledger is full of them; name them
  `*_digest`/`*_hash` and the scanner stays at full strength. **Never reach for a broad allowlist**
  — `.gitleaksignore` takes `<commit>:<path>:<rule>:<line>` fingerprints, which silence exactly one
  line of one commit and expire when it changes.

## CI/CD

- CI (`.github/workflows/ci.yml`) is **three concurrent jobs, not one chain**. `test`: `uv sync --all-packages` → `ruff check` → `ruff format --check` → `pytest -m "not integration" -n auto --dist loadfile` → `node --check` over `find src/exercises -path '*/web/*' -name '*.js'`. `integration`: a **three-shard matrix** (`tokenization` · `mixtures` · `rest`), each shard syncing, caching and installing chromium, running `deploy/vercel/build.sh` once, then `pytest -m integration`. `security`: gitleaks over the full history. `push` is filtered to `main` — branches are covered by the `pull_request` event, because an unfiltered `push` ran every PR commit twice. `train`: `uv sync --all-packages --extra train` with **CPU-only wheels** (a Linux-scoped
  `pytorch-cpu` index in the root `pyproject.toml` — 191.8 MB instead of 2.7 GB, and 19 fewer
  packages in the lock), running only the files whose module-level `importorskip` would otherwise
  skip them entirely.
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
