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

## The reference material is confidential, and it lives OUTSIDE this repository

Source material this project is built from is **not ours to redistribute**. It is held at a sibling
path resolved by `tools/backup_local_only.py::EXTERNAL_SOURCES` and overridable with `LLM_NOTES_DIR`.

**It used to live inside the working tree, gitignored, and that was not enough.** Gitignoring a
directory protects its bytes and does nothing about a tracked document that *describes* them. While
`.gitignore` worked perfectly, a public branch carried: a table naming two source files with their
line counts and a summary of each one's contents; a source path served to the **live site** in an
exercise's `records.json`; module docstrings citing sources by name; a scaffolder that wrote a source
path into every new exercise's requirements document; and test fixtures whose invented filenames published the real
naming scheme. Moving the material out removes the whole class — there is no path inside the repo to
leak, nothing for `.gitignore` to name, and no way to commit it by accident.

**Three rules, and none of them is about the bytes:**

- **Never name a file in it.** A filename is a disclosure on its own: it says what the directory
  holds. Neither may a tracked file publish a count of them, or their sizes.
- **Never quote it.** Say what *we decided* and why. The published artefact is our reasoning, not
  the source's wording.
- **Never describe its contents or how it is processed.** What kinds of document it holds, and what
  is done to them before an agent reads them, are both confidential.

**What is automatic, and what is not — be precise, because the difference decides what reaches a
public branch:**

| check | where it runs | catches |
| --- | --- | --- |
| `test_forbidden_vocabulary.py` | **CI and pre-commit** | the words themselves |
| `test_no_confidential_leaks.py::…names_a_confidential_source` | **CI and pre-commit** | the naming scheme |
| `…quotes_the_confidential_material` | **pre-commit only** — skips where the material is absent, CI included | verbatim text |

**Three checks, because a leak has three shapes and none of them implies the others.** A document
can name no file and copy no sentence and still describe the source by the kind of thing it is. The
vocabulary check is lexical, needs nothing but the repo, and is therefore the one that actually
stands between a working tree and a public branch — the other two are stronger and narrower.

`FORBIDDEN` in that file is the list, each word with the reason it is banned. Unrelated senses live
in `ALLOWED` and are matched **per line**, never per file: a file-wide exemption is a hole the size
of the file. Both lists fail in the other direction too — an exemption for a sense nothing uses is
removed, so the list can only grow by someone's decision and never as the quick way to clear a red
gate. Two path exemptions exist and both are content that would be *wrong* to edit: the frozen
release snapshots, and the tokenizer's own corpus, whose bytes are a measured input rather than
prose.

Both are gated on commit. The second cannot run in CI, because CI has no copy to compare against,
so **CI can prove no filename leaked and only the hook can prove no sentence did.** If you commit
from a machine without the material, that half silently skips.

**That gap is now closed by a receipt, and the honest limit of it is worth stating.** The quoting
check emits a boolean and two digests — it never needs to reveal corpus content — so its *result*
can be published without publishing anything it read. `.quote-check-receipt.json` records a digest
over the exact tracked prose the check covered, plus a digest of the checker itself, and
`tests/test_quote_check_receipt.py` recomputes both **in CI** from the repository alone:

```bash
uv run python tools/quote_check_receipt.py --write    # after the gate passes, where the material is
uv run python tools/quote_check_receipt.py --verify   # what CI does; non-zero on drift
```

So CI can now prove *a machine holding the material ran this exact checker against exactly this
prose*. It **cannot** prove that machine was honest — anyone who can run the checker can write the
file. The failure this repo actually has is forgetfulness and staleness, and those it does close.

**The consequence is deliberate: changing tracked prose invalidates the receipt**, so it has to be
regenerated on a machine that can run the gate. That means only someone who can actually run the
quoting check can change prose, which is the property being bought. **Regenerate it as the last step
before committing**, after every change to tracked text.

A pre-commit hook says so at the point of commit rather than two minutes later in CI — it **verifies
and never writes**. A hook that regenerated the receipt would make every commit pass by quietly
re-attesting rather than by the gate having run, which is the same shape as the content-rewriting
hooks this repo removed. Regenerating stays a deliberate act. Getting this wrong twice in one
afternoon is why the hook exists at all.

**Name the exercise, not the source's own unit of material.** A topic is referred to by the exercise
that covers it; the material itself is "the source". The banned words came back three times after
being removed by hand, which is why this is a gate rather than a habit — they are cheap to type and
expensive to notice, and a sweep that rewrites five hundred leaves four indistinguishable ones.

**Paraphrase; do not quote.** Every rule this repo takes from the source is stated in our own words,
including where the original was more quotable. The exception is a *functional* overlap — an
identifier the work is graded against, like a required log event name — and those live in
`FUNCTIONAL_OVERLAP` with a reason each, plus a twin that fails when an entry stops being needed.

**A lexical guard cannot see itself until it is tracked.** The first version of this one listed four
real filenames in its own docstring to explain the pattern, passed locally because it was not yet in
`git ls-files`, and was caught by CI flagging its own source. The first CI run after adding a guard
of this shape is the first real run.

**Removing a leak from the working tree does not remove it from history.** Everything scrubbed is
still in earlier commits and in any PR description that quoted it. PR bodies are editable; history
is not, without rewriting published commits.

## Work the queue to the end — a handoff is not a block

`docs/agents/QUEUE.md` is the single source of truth for progress, and **"How to work the queue"**
in it is binding: finishing a unit is not a reason to stop, and opening a pull request is not a
reason to stop. Merging, tagging, the production gate and submission are PK's and always will be —
but they are handoffs. The pull request waits for a person; **the work does not wait for the pull
request.** Open it, record it in the queue, then start the next row on a fresh branch off `main`.

After each unit: update the queue, self-assess against what the unit *actually did* rather than what
it set out to do, name what was left undone and which row now owns it, re-read the order in case the
unit changed it, and begin the next row without pausing for acknowledgement.

**The reasons to stop are enumerated in that section** rather than judged in the moment — a decision
this document marks as a human's, two consecutive review rounds producing new BLOCKERs, a red guard
whose honest fix is out of scope, or a refusal from the sandbox or permission layer. A refusal is a
boundary working, not an obstacle to route around. Everything else — a question, an uncertainty, a
finding — goes in the log and the pull-request body, and the work continues.

## MANDATORY — never remove anything under `notebooks/` or any `tools/`

**Nothing in `notebooks/` or in any `src/exercises/*/tools/` directory may be deleted, moved,
renamed, or overwritten — locally or on the remote — without PK's explicit prior permission naming
the file.** This is not a style preference and it does not yield to a tidy-up, a "stale" file, or a
rule elsewhere in this document. If something there looks wrong, **say so and stop.**

These are the only files in the repo with **no second copy**. Both are gitignored, so git cannot
restore them: `notebooks/S[0-9][0-9]-*.ipynb` and `src/exercises/*/tools/build_notebook.py`. A
deletion here is permanent in a way that no other deletion in this repo is.

**This now covers `REQUIREMENTS.md` too.** All four of exercises 01–04's requirement documents were destroyed by an
ordinary branch switch after the commit that untracked them, and were recoverable only because
`18015b1^` was still reachable. A requirements document written *after* the untracking convention has no such
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
  the good version with the broken one. Run it before any branch switch and after any topic that
  rebuilds a notebook.

- **The protected set is wider than the three classes named above, and the extra ones were
  unguarded for months.** Alongside them,
  `docs/EXPLAINER_PROMPT.md` / `docs/EXPLAINER_PATTERN.md` are the two documents any explainer is
  required to be built from. All gitignored, none regenerable, none watched by the tripwire until
  now. **85 files, 12 MB.** A guard that covers the documented cases and misses the largest one
  reads as coverage without being any.

- **The authoritative list is `tools/backup_local_only.py::PATTERNS`, not this document.** Prose that
  enumerates the set is a second copy of it, and the second copy is the one that drifts — this
  paragraph named three classes while `PATTERNS` protected eleven, so an agent reading only the
  rulebook would have believed `rm TODO.md` was recoverable. **Read `PATTERNS` before touching
  anything gitignored.** Each entry there carries a comment saying why it cannot be regenerated, and
  five are named nowhere else: `TODO.md` · `.claude/settings.local.json` (losing it silently changes
  what agents may run without asking, rather than failing) · `src/exercises/*/docs/*.md` (planning
  and critique notes) · `src/exercises/*/docs/*.html` (saved reference pages, snapshots of things
  that change) · and
  **`src/exercises/*/src/solution/**/*`, the one class with no recovery path at
  all** — it has never been in git on any branch, so the `git show <untracking-commit>^:<path>`
  fallback below is inapplicable by construction, and its `corpus/*.raw.html` inputs pin no revision,
  so re-fetching returns a different article. The backup store is the only copy that exists.

- **After any branch switch, pull, merge, rebase or stash, run the tripwire** —
  `uv run pytest tests/test_local_only_files_present.py`. It fails when *some* of these files are
  present and others gone, and skips when all are absent (a clone, not a loss).
- **The store is a git repo, so it obeys your global gitignore — and that silently un-versions
  files.** `~/.config/git/ignore` applies inside the store like anywhere else. A global rule
  matching `.claude/settings.local.json` meant that file was copied on every run and committed on
  none: present on disk, `--verify` satisfied because the bytes matched, and **no history at all**,
  which is the entire product. `snapshot()` now sets `core.excludesFile` to the null device every
  run and asserts per file that what it copied is what git tracks. A repository-local
  `.git/info/exclude` can still reach this, which is what the twin test plants.

- **Recovery, in the order to try it:**
  ```bash
  uv run python tools/backup_local_only.py --verify              # 1. does the store have it?
  cp ../.llm-pretraining-exercises-local-only/<path> <path>      #    restore the latest
  git -C ../.llm-pretraining-exercises-local-only log -- <path>  #    or an earlier version
  git show <untracking-commit>^:<path> > <path>                  # 2. e.g. 18015b1^ for the requirement documents
  ```
  Step 2 works only while the removal commit is still reachable, which is why step 1 exists.

- **Deleting one of these files on purpose takes a step nobody had written down.** The store is
  **append-only**: `backup_local_only.py` copies in and never removes, so it is a high-water mark,
  and the tripwire's question is *"is anything in the store missing from the working tree?"* That is
  the right question for a loss and the wrong one for an intentional deletion — which therefore
  reds the tripwire **permanently**, and re-running the backup tool does not clear it. After PK has
  named the file, record the removal in the store as its own commit:
  ```bash
  git -C ../.llm-pretraining-exercises-local-only rm <path>
  git -C ../.llm-pretraining-exercises-local-only commit -m "why this was deleted, and who asked"
  ```
  The content stays reachable in the store's history — that is the whole reason the store is a git
  repo rather than a copy — so verify you can still read it back before moving on:
  `git -C <store> show <removal>^:<path> | wc -c`. Never resolve a red tripwire by deleting the
  store, and never by editing the test.

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
- **Identical skeleton per exercise:** `REQUIREMENTS.md` (requirement — **local only, gitignored**) · `README.md` (what/how) · `pyproject.toml` (member) · code in one place (`src/` or `web/`) · `artifacts/` (gitignored outputs). Long reasoning gets its own tracked `DECISIONS.md`.
- **Do not scaffold an exercise by hand. There is a generator.**
  ```bash
  uv run python tools/new_exercise.py 09 loss-functions-output-heads \
      --title "Loss functions and output heads" --package lossheads \
      --summary "One sentence for the root README row." [--dry-run]
  ```
  It writes the whole skeleton, **including the three gitignored files** (`REQUIREMENTS.md`, seeded from the local
  requirement text when one exists; `tools/build_notebook.py`; and the notebook it
  builds), joins the `rest` CI shard, adds the root README row, and prints what is left for you.

  **The sequencing is the reason it exists.** `tests/_exercises.py::exercises_in` only counts a
  directory that has a `pyproject.toml`, so a new exercise is invisible to every guard until that
  file lands — and the moment it does, six test families apply at once, three of them checking for
  gitignored files a fresh clone will never have. Do it by hand and the suite goes red locally with
  a message about files "going missing" that were never there.

  **It deliberately does not touch the two web-gated registrations** — the landing card and the
  `SPINE_ENFORCED` ledger — because both guards assert in *two* directions and a premature entry is
  exactly as red as a missing one. It prints them as deferred instead.

  `tests/test_new_exercise.py` runs the generator for real into a temporary directory and checks the
  result against the **real** guards, importing `REQUIRED`, `REQUIRED_DIRS` and `_READERS` from the
  guard modules rather than restating them. That is the point: a generator whose templates encode
  the conventions is a second copy of them, and a second copy drifts. It has already earned its
  keep — it caught the generator inserting the CI path *after* the shard's trailing `tests` entry.
- **Set the folder up BEFORE writing code.** The skeleton is not paperwork to backfill. Exercise 06
  was scaffolded with `pyproject.toml` and modules but no `CLAUDE.md`, `PROGRESS.md`, `NOTICE` or
  `REQUIREMENTS.md`, because a convention that lives only in prose gets skipped under momentum.
  `tests/test_exercise_skeleton.py` now checks the universal ones (`README.md`, `CLAUDE.md`,
  `pyproject.toml`, `tests/`) — **`tools/` is deliberately not among them**, because the only
  file some exercises keep there is the gitignored `build_notebook.py` and git does not track
  empty directories, so `tools/` exists on a working checkout and not in a clone. Requiring it
  passed locally and failed CI: write the guard for what a clone has, not for what your machine
  has. It also asserts **no `REQUIREMENTS.md` is ever tracked** — checked with
  `git ls-files`, not by reading `.gitignore`, because a file already in the index stays tracked
  whatever the ignore rules say afterwards.
- **Shared code:** deferred — add `src/common/` (its own member) only when a 2nd exercise needs to reuse something. No premature abstraction.
- **Notebooks:** top-level `notebooks/`, one per topic — see below.

## Every topic builds a Colab notebook — locally, never tracked

`notebooks/SNN-slug.ipynb`, zero-padded topic id first (`S04-data-cleaning-dedup.ipynb`), so
lexical sort = topic order. **A source material's work is not done until its notebook runs the shipped
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
tracked, stdlib-only sample that CI executes top to bottom. It cannot check that a topic notebook
is correct; it checks that a notebook in this repo opens and runs, which is the part CI can still
see. Anything stronger has to be run by whoever has the notebook, before the PR.

## Five data concerns — keep them physically separate

- **Requirement documents → never tracked, at any level.** `REQUIREMENTS.md` is gitignored by name everywhere, as is
  programme-level material — the schedule, the class list, the internal authoring specs
  (`docs/REQUIREMENTS.md`, `docs/EXPLAINER_*.md`). A requirements document is the course's text and
  is input for whoever builds the exercise; it is not the deliverable. **Never link to one from a
  tracked file** — the link resolves on a working checkout and 404s for everyone else. What we
  *decided*, and why, is published instead: `README.md`, and a tracked `DECISIONS.md` when the
  reasoning needs room (see `04-data-cleaning-dedup/DECISIONS.md`).

  **And `REQUIREMENTS.md` is not the authority on what submission requires.** It is the course's text and
  it can be truncated, reformatted or pasted short; the submission platform's own field list is what
  grades. Check the platform before calling a topic done, and record the required *shape* — not
  the requirements' wording — in the exercise's `PROGRESS.md`. A deliverable specified as a **public URL**
  is not satisfied by a file in the repo, however correct that file is.
- **Topic notebooks** → top-level `notebooks/`, **gitignored** except the tracked
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

- **Break it in a `finally`, and stage by path — the rule above is what invites this failure.** On
  2026-09-02 an agent auditing this repo did exactly what the previous bullet asks: it backed up
  exercise 05's `checks.py`, injected `return []` into `check_no_orphan_benchmarks` and
  `check_tier_shares` to watch their guards go red, and then restored **one** of the two. A
  `git add -A` swept the mid-experiment tree into a commit, along with the backup file — which was
  itself already mutated, so it could not have restored anything. **Two data-handling invariants
  were dead on this branch for four commits**, returning "no findings" for every input, which is
  indistinguishable from a clean run. Three rules follow. Restore in a `finally`, never on the happy
  path, so an early return or an exception cannot leave the mutation behind. Never write the backup
  inside the working tree — `git stash` or `$TMPDIR`, because a backup in the tree is a file
  `git add -A` will commit. And after any topic in which agents ran near the source tree, **stage
  by path and read `git diff --stat` against `origin/main` before committing**: the only reason this
  was caught is that one mutation happened to break a test that ran, and a mutation to a guard whose
  twin is missing would have shipped in silence.

- **A guard must not trigger the behaviour it is testing for.** Exercise 08's invoice cut line
  starts hidden and is revealed by an `IntersectionObserver`. The guard asserting it was visible
  called `scrollIntoView()` first, then measured — so it fired the observer and then checked the
  result of its own action. It passed for the entire time the cut line was invisible to every reader
  who had not scrolled: a screenshot, a print, a PDF, anyone landing on an in-page anchor. The rule
  generalises past scrolling: if a test clicks, focuses, hovers or scrolls before asserting, ask
  whether the assertion is about the state after that action or about the state the reader actually
  arrives in — and if it is the second, do not perform the action.

- **Prefer a painted terminal state to an animated one wherever the motion buys nothing.** The same
  cut line was a 300ms fade that said nothing the dashed rule did not already say standing still,
  and it cost the plate its entire argument in every non-scrolling context. Reveal-on-scroll is a
  decision to hide something by default; make it deliberately, and never for the one element that
  carries the point.

- **In CI a skip is a failure unless `tests/_skips.py` declares it, and three reasons can never be
  declared.** A skip reports as a pass, so the skip report is the only place a vanished test shows
  up — and nothing read it until the root `conftest.py` landed. The three are `chromium
  unavailable`, `run deploy/vercel/build.sh first` and `{slug} is not published`: each fires inside
  a job that has just installed or built the thing it checks for, so each means **that job's own
  setup broke**, and exempting one turns ~200 browser assertions into a green run. Adding a ledger
  entry is deliberately expensive — a reason with weight, a pattern too narrow to cover a file, a
  pinned count of the skip lines it matches. **Never add an entry to clear a red gate.** And `xfail`
  is refused outright, because an xfail that genuinely fails reports green and no ledger sees it —
  `xfail_strict` catches XPASS and not that.

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

- **A tested feature with no caller is dead code wearing a test.** `masks.loss_mask(context_spans=...)` in exercise 06 is implemented, documented, covered by two passing tests and taught in the topic notebook — and `grep -rn context_spans` finds **zero** callers in the pipeline: `feed.py` builds every microbatch with the default mask. The tests are green, so the capability reads as a behaviour of the run, and the documents describing prompt/tool-observation masking describe something that never happens. The test proves the function works; only a caller proves the system uses it. When you add a keyword-only option to a library function, either wire it through the one path that would exercise it in a real run, or state in the module docstring that it is offered and unused — and put the same sentence wherever the feature is described to a reader.
- **A coverage guard built on `--collect-only` is blind to a file that collects nothing.** `tests/test_ci_shards_cover_everything.py` catches an integration file in no CI shard, and an integration file in two. It cannot catch the third case: **in a shard, and contributing zero tests.** A module-level `pytest.importorskip("torch")` raises during *collection*, so `pytest --collect-only -q` prints no `path: count` line for that file at all — I verified this with a throwaway module importorskipping an absent package: output was `no tests collected`, exit 0. The file is therefore absent from `everything` and from `owners` alike, `missing` is empty, and `covered == sum(everything.values())` holds trivially. The consequence is live: all 20 of exercise 06's integration tests (`crash` 11, `model` 3, `train` 6) sit behind `importorskip("torch")`, CI never installs the `train` extra, and CI's integration step maps exit 5 to success — so the `rest` shard runs **zero** of them, reports green, and the coverage guard agrees. A guard must count what the job was *supposed* to run, from a list it does not derive from the same run it is auditing.

## Reporting a measurement

Three rules, each learned by getting it wrong in `02-tokenization` (see that exercise's `CLAUDE.md`):

- **Establish the noise floor before you rank anything.** A held-out score there swung 9,421 points across the five possible splits while the recipes it was meant to separate sat 648 apart. One split looked decisive; five showed the test could not rank at all. Before quoting a comparison, re-run it under a different arbitrary choice — a different split, seed, or slice — and check the effect survives.
- **Sweep without gaps.** A weight sweep that went 2 → 5 → 6 confidently named ×6 the optimum; filling in ×3 and ×4 moved it to ×3, which was better on every stable measurement. A coarse sweep does not report "roughly the optimum", it reports the wrong one.
- **Report the number the metric ignores.** Any score that rewards a *ratio* or a *gap* can be improved by making the denominator worse. Print the absolute quantity next to it — there, total tokens beside the fairness score — so buying the metric is visible rather than inferred.

When one of these overturns a published claim, correct it where the claim was made and say what changed. A quietly amended number is worse than the original error.

- **Prose that states a number is generated too, or it goes stale while the table beside it stays right.** This is the failure that has cost this repo the most edits. A generated table under a hand-written sentence looks maintained, and only the sentence is wrong — so a reader believes the sentence. Exercise 05 shipped documents reading "across three lanes", "H3 came back qualified", "Thirteen invariants" and "one verdict did not survive its own noise", every one of them contradicting a correct table directly above or below it, and no test failed. If a sentence contains a count, a verdict or a size, derive it from the same source the table uses. Where prose genuinely must stay hand-written — a row in the root README's exercise table — the number in it went untested long enough for exercise 06's row to read *"Stage 1 of 8"* while the exercise was at stage 7. `tests/test_doc_counts_match.py` now derives that count from the exercise's own stage table. **The prose around the number is still untested**, so a row can carry a correct stage and a wrong description; verify that by hand on every PR that advances an exercise.

- **An experiment that cannot see a lane is not evidence about that lane.** Exercise 05's proxy dropped the three lanes it had no text for, and one hypothesis read `qualified` for two weeks because the lane its refutation clause tested was absent. Funding the lane flipped it to `refuted` with the effect size essentially unchanged. **A missing input does not make a hypothesis safer, it makes it untestable — and untestable reads as passing.** Before trusting a result, list what the measurement was blind to.

- **Size a proxy corpus against the RUN, not against the mixture's ratios.** Getting the
  proportions right and the total wrong does not shrink the experiment, it changes what the
  experiment is: the run stops measuring a mixture and starts measuring repetition. Exercise 06
  consumes `ranks × accumulation × microbatch × sequence_length × steps` =
  `4 × 2 × 8 × 512 × 320` = **10,485,760** token positions (read it from `Config.total_tokens`,
  never from memory). Its first corpus held **2,185,575** tokens — **4.8 epochs flat**, and once
  shaped to exercise 05's lane weights, **30.2 epochs of the web lane against 0.41 of the agentic
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

- **Registering a new exercise: five lists, two automatic, and `tools/new_exercise.py` does three
  of the rest.** The generator handles the CI shard and the root README row; the landing card and the
  spine ledger are deferred to whenever `web/` lands, because both fail in two directions. The
  paragraph below is what the generator encodes — read it when it goes wrong, not before.
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
- **`node --check` does NOT parse a `.js` file as an ES module, and CI's syntax gate depended on
  it.** Node parses a lone `.js` with the *script* goal, which means wrapping the source in the
  CommonJS function wrapper first — so a stray `}` merely closes that wrapper early and the file
  passes. It is not theoretical: `node --check` exited 0 on a `diagrams.js` with an unbalanced
  brace and the browser refused the same file with `Unexpected token '}'`. Verified on a
  four-line throwaway module. Every file the gate checks (`find src/exercises -path '*/web/*'
  -name '*.js'`) is an ES module, so the gate was weaker than it read for as long as it has
  existed. Feed the file on stdin instead — `node --input-type=module --check < "$f"` — which
  parses with the module goal, passes valid modules and catches that.

- **A guard must test the property, not one phrasing of it.** Two guards in one topic asked for
  a specific string and failed correct work: one demanded a "drawn to scale" line and red-flagged a
  figure that quotes its own paper verbatim (stronger evidence than the thing being demanded), and
  one demanded a legend headed `THE MARKS` and red-flagged eleven figures keyed by other means — a
  legend headed *"what the update does"*, or marks labelled in place. Both times the honest fix was
  to ask the underlying question: *is this attributed?* and *is this colour explained anywhere on
  the figure?* A guard that names one implementation of a property will fail every other
  implementation, and the pressure is then to reword good work to satisfy the test.

- **Colour can only carry semantics while there are more colours than meanings.** A semantic
  palette of four (`--part-q/k/v/store`) was asked to distinguish up to six update steps, and two
  of them (`dg-local`, `dg-k`) resolved to the same token — so a six-step recipe rendered five
  marks and nobody could see which two had merged. The same bug had already shipped once in the
  same exercise, in a different family, and been fixed locally rather than as a rule. When the
  count of things to distinguish can exceed the count of colours, encode it in **form** — an
  ordinal, a shape, a texture — and let colour keep its one job. Here the ordinal was also *more*
  informative than the colour it replaced: the steps happen in that order.

- **Measure the invariant a design already holds before you change it — and never write a guard
  from a misreading.** Asked to fix a wide-screen layout, I read "shouldn't the rail be centred?"
  as "the rail is too far left", moved the rail inward to sit against the text, and wrote a guard
  demanding a gap of at most 60px. Every railed page (05, 06, 07, 08) already centred the reading
  column in the space the rail leaves — equal air either side, 554px at 2560 and 24px at 1180 — and
  the change destroyed that symmetry, leaving dead space on both sides of the rail and pushing the
  column off centre, which is what the reader had actually been reporting. **The guard was the
  worst part**: green, wrong, and it would have made the misreading permanent by failing anyone who
  restored the correct layout. Two rules follow. Measure what the existing design does across the
  full width range *first*; symmetry, ratios and the relationship between elements are visible in
  numbers and settle what prose cannot. And when a report is ambiguous about which element is
  misplaced, ask — a layout complaint names a symptom, and the element the reader blames is often
  not the one that moved.

- **When a vendored stylesheet centres, reserves or positions something, check the page actually
  builds the element it targets.** `_shared/page.css` centres a pinned rail with
  `.rail-inner { margin-block: auto }`. Exercises 03, 05, 06 and 07 create that wrapper; exercise 08
  did not, so its contents hung at the top of a full-height column while every sibling page sat
  centred — no console error, no failing test, and it took three rounds of feedback to find because
  the symptom ("the rail isn't centred") pointed at a rule that was working. This is the **third**
  time this directory has cost something the same way: it also reserves a 260px gutter only some
  pages fill, and vendors marks whose colours resolve only when the real token file is linked. When
  you copy `web/_shared/`, diff what its rules select against what your page emits.

- **Two rules of equal specificity are decided by source order, and the later one wins.** Two fixes
  in one topic changed nothing at all: `grid-template-columns` set on a flex container, and a
  `max-width` override written above the rule it was meant to beat. Both looked like fixes, moved no
  pixels, and passed every test. Before adding a rule, check what is already computing — then edit
  *that* declaration rather than competing with it. A `margin: 16px 0 0` shorthand will also silently
  cancel a `margin-inline: auto` you added elsewhere.

- **A DERIVED number can answer the wrong question, and that is far harder to catch than a wrong
  one.** Exercise 08 published *"the claimed arc holds in 6 of these 7 two-year windows"*. The
  number was real, generated from the data, and evidence for nothing: it counted windows that
  produced *a* clear winner, not windows whose winner the claim predicted. Six windows do decide,
  the order is not the claimed one, and the verdict was therefore the exact opposite of the truth —
  published confidently **because** the arithmetic was sound. A wrong number gets caught by a
  reader; a right number answering an adjacent question does not. Before quoting a derived figure,
  say out loud what question it answers and check that it is the question you asked.

- **Vary every arbitrary choice before quoting anything that rests on it — and be ready to lose a
  finding.** The same section asserted its count was "not noise" and offered no evidence. Its
  two-year buckets begin in 2014 because attention does, not because the field changed on that
  boundary. Shifting the edges by one year kept two conclusions and destroyed a third — one that
  had been published an hour earlier — so it was corrected in place and demoted to "one reading,
  not a measurement", with a test that fails if it ever becomes robust so the hedge cannot outlive
  its reason. The noise-floor rule already in this file is usually described for a metric; it
  applies just as hard to a *count over buckets you chose*.

- **Naming a real product, model or vendor is a claim, and gets sourced like any other.** Exercise
  08's page named no real model anywhere in its own voice, so a reader could not tell whether it
  described history, a research frontier, or the thing inside the chatbot they used that morning,
  and "almost every open model uses them" asked for trust while offering nothing to check. The fix
  is not to write the names down: find each source through an API or a search rather than from
  memory, quote the sentence, gate the quote against the downloaded document, and **leave the field
  empty where nothing says so**. Twenty-two of thirty ended up empty and that column became the most
  informative one on the page — it separates what the field adopted from what it admired.

- **When agents gather evidence, make a machine the arbiter — and test the machine first.** Exercise
  08 sourced 80 hyperparameters across 29 papers this way: download every source *before* any agent
  runs, have agents read those local files, then check each proposed quote as a contiguous run of
  that file's own characters. 82 proposed, 82 verbatim, zero fabrications — a result worth having
  because the gate was built to catch the opposite. **The gate needed three fixes before it could be
  trusted**, each found by running it against quotes already known to be good: arXiv's HTML prints
  every equation twice (rendered, then LaTeX source), hides `U+200B` inside numbers where Python's
  `\s` will not match it, and papers write `1 M` as often as `1M`. Every one made it report a
  hand-verified quote as absent from its own paper. **A guard with false negatives is not the safe
  direction to err in** — here it silently converts sourced numbers into unsourced ones, which reads
  as caution and is a loss of provenance.

- **Verbatim is not the same as correct, and the second question is the one that catches real
  errors.** A quote can be a genuine sentence from the right paper and still be evidence for
  something else: "Figure 4: The KV cache of StreamingLLM" offered for four attention sinks, "we set
  D = 256" offered as a context length, a *Communications of the ACM* volume number offered as a
  head dimension. All three survive an authenticity check. Ask separately whether the quote talks
  about the quantity being claimed.

- **An absent number is not a zero, and a percentage that rounds to zero is not a measurement.** A
  published figure read "16 of 7,813 blocks plus a **0-token window** — about **0%** of a
  1,000,000-token context." The mechanism has a local window whose size we had not sourced, and its
  true share is 0.2% — which is the entire claim of the paper. Omit the clause when the input is
  missing, and give a small ratio enough significant figures to be sayable out loud.

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

The root's job is **routing, not retelling**. Where the requirements require the root to reach a
deliverable "without a detour", that is a property of its links, not of how much it repeats — and
the test for it should assert the *link*, since asserting the filename passes against a front door
that names the file and never links it.

**"Without a detour" is satisfied by a link, not by a section.** The requirements for the exercise under
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

**Write for a ladder of readers, and let one narrative deepen — never five parallel tracks.** A
reader must be able to stop at any depth and still be *correct*, not merely comforted. Tabs, toggles
and "advanced" drawers split the argument; layering keeps it whole.

| rung | what they need | the test to apply |
| --- | --- | --- |
| **A curious teenager** | the problem in plain words, one concrete analogy, zero notation | could they retell the point to someone else? |
| **A practitioner** | what it is mechanically, how to run it, what it costs | could they use it on Monday? |
| **A researcher** | the method, the noise floor, prior art, what would falsify it | could they attack it? |
| **Product** | what it enables, and when *not* to use it | could they scope it? |
| **A CTO** | the one number that decides, and the risk attached | could they say yes or no? |

**Every exercise page carries the same spine, and it is test-enforced.** Sections declare
`data-role`, so guards check the *structure* while the prose stays free to change:

`thesis` · `glossary` · `problem` · `mechanism` · `method` · `expected` · `results` · `negatives` ·
`conclusion` · `limits` · `next` · `reproduce`

**Enforcement is two halves, and neither is sufficient alone.** `tests/test_page_spine.py` is the
repo-wide, **lexical** half: it reads each `chapters.js` and asserts every enforced page constructs a
section for every role. It runs in the plain `test` job with no browser and no assembled site,
because a structural rule that only runs when chromium happens to be installed is one that can
silently stop running — this repo has already lost 46 tests exactly that way. What it cannot see is
**order**, since source order is not DOM order; that is the per-exercise browser test's job
(`test_embeddings_render.py::test_the_page_has_the_required_spine_in_order`), and the lexical guard
asserts every enforced exercise *has* such a test so the two halves cannot drift apart.

**The ledger fails in both directions.** `SPINE_ENFORCED` names the exercises held to the standard;
`SPINE_EXEMPT` names the deployable pages deliberately outside it **with a reason each**, and the
deployable set is read from the filesystem rather than listed. A new exercise ships a `web/` bundle,
lands in neither, and the guard goes red — which is the whole point, because the previous version of
this rule lived only in prose and applied to whoever remembered it. 01–04 are exempt: the spine
describes an exercise that ran an experiment and reports a result, and those four do not.

**Exercise 08 is the reference implementation, and `docs/DESIGN.md` is the canonical standard** —
grid, type scale, components, what enforces each rule, and a numbered retro-fit checklist, every
number in it measured on 08. This section carries the short version; where the two disagree,
`docs/DESIGN.md` wins and this file is the one to correct.

**Exercise 07 is where the rules below were learned**, which is why they are stated in its terms. It
was rebuilt after an audit found the previous page was **nine tables, one button and no diagram of
any kind** — ~1,300 words that never said what an embedding is, never stated the question being
answered, never explained the method, and had no summary, conclusion or next step. The rewrite runs
~3,300 words with six figures, and the shared `web/_shared/` helpers it needed had been sitting
vendored and unused the whole time. 08 then took the same rules further — fluid type, the chapter
strip, a rail that marks position — and 07 is itself queued for the retro-fix.

The rules that follow from it:

- **A mechanism figure is not a results chart, and a page needs both.** Results say *what happened*;
  mechanism says *why it must*. A page with only results can be believed but not understood — and
  mechanism is the half that survives five years. Draw the central object: exercise 07 spent weeks
  on a 256×32 grid its own page never once showed.
- **A caption argues; it does not label.** State what to conclude, and where useful what would
  falsify it — *"one hidden state where that sum is meaningfully non-zero would refute this
  section."* A figure whose caption is its title has made the reader do the interpreting.
- **Say what you expected before what you found.** It is the only way a reader can tell a finding
  from a story told backwards, and it costs nothing when the prediction was wrong — which is when it
  is worth the most.
- **Define every term where the reader first meets it**, and give each definition a real number from
  your own run rather than a textbook gloss.
- **Put a failure in the opening tiles.** A page that shows only its wins has not earned the ones it
  shows.
- **Screenshot every section you build. A green suite is not a rendered page.** Retrofitting 05 and
  06 to the spine produced four real defects and **every one was found by looking at the page**,
  with the whole suite passing each time: a raw `<b>` tag shown as literal text, stray `*` markers
  from a bold that cannot nest an italic, two rail entries with the same title, and a figure whose
  caption pointed at two boxes that sat off-screen behind a horizontal scroll. The existing markup
  guard could not see the first two — it looks for `[[`, `**` and backticks, and neither string
  contains any. Render the section, read it, *then* write the guard for what you found.

- **A guard that asserts an element is VISIBLE has not asserted it is LEGIBLE.** Exercise 08's
  invoice cut line — the sentence the whole figure exists to deliver — was `white-space: nowrap`
  inside `overflow: hidden`, which truncates with no ellipsis and no warning. It read *"…the cache
  alone needs a second ma"* at every width narrower than the sentence, for as long as the figure had
  existed, and `test_the_invoice_cut_line_is_visible` passed the entire time. The general property is
  cheap to assert and catches the whole class: no element whose `scrollWidth` exceeds its
  `clientWidth`, at several widths, allowing 1px for sub-pixel rounding.

- **A count in a heading or a navigation label is always a count of that section's own contents, so
  it must be derived — and the lexical guards for this start too high to see it.** Exercise 08's
  `next` section was headed *"Three things this opens"* above **four** items, with its rail entry
  agreeing, live and green: `test_no_count_is_typed_into_the_page_as_a_word` scans for *eleven* and
  up, deliberately, since these pages say "two bills" and "six words" constantly and those are fixed
  quantities. Widening that pattern would have meant marking **thirty-six** legitimate lines with
  `count-literal-ok`, and a marker on thirty-six lines is noise nobody reads. Narrow the *scope*
  instead of widening the pattern: inside a heading or a rail label the small numbers can be
  forbidden as literals with no false positives at all. Exclude `one` and only `one` — it is a
  determiner far more often than a count ("One step, taken apart").

- **A `display: none` in a media query loses to a `display: flex` written below it at the same
  specificity.** This is ordinary cascade and it is worth naming because the symptom is invisible on
  the machine you are working on: exercise 08's at-a-glance table hid its column heads below 900px,
  the rule was written above the one that re-laid the row, and every phone opened the table with five
  orphaned column labels. `AGENTS.md` already records a `max-width` lost the same way. When a media
  query both re-lays an element and hides part of it, put the hide *after* the re-lay, or raise its
  specificity, and screenshot the narrow width.

- **A decorative background is only decorative if it stays decorative at every width.** Exercise 08's
  masthead field is 7–13% ink and the body text sits on it by design; one accent rule inside it
  painted at full opacity, and at 1440px it ran straight through the words "every one of" in the
  opening sentence and read as a strikethrough. Where the text falls across a background is not
  something the graphic can know, so the graphic cannot own a mark that would be a defect anywhere
  the text might land.

- **A cross-reference to something you decided not to write is worse than no cross-reference.**
  Promoting a finding to the top of exercise 08's page left a clause in its limits section reading
  "it is stated at the top of the page" — but the tile that actually went up carried a *different*
  finding. The pointer survived the edit that invalidated it, which is the normal way this happens:
  the sentence you edit and the sentence that refers to it are rarely on the same screen. After
  moving anything, grep for the words that pointed at it.

- **Do not delete a feature and leave its guard behind, or leave the data the guard reads.** Exercise
  08's six pull quotes were removed for a good reason — each was set in the page's largest type and
  attributed to "this page's own catalogue", which is the visual grammar of a citation with none of
  its function. The `pull_quote` field, its sourced-from-the-catalogue guard and that guard's broken
  twin went with them, because a tested field with no renderer is `AGENTS.md`'s own "dead code
  wearing a test" one level up: the guard passes, so the capability reads as a behaviour of the page.

- **A partition guard does not check that a group's headline is true of its members.** Exercise 08's
  `story.check()` refuses a chapter grouping that does not cover the catalogue exactly once, and it
  was green while a chapter headed "keep a fixed-size state" — promising "every one of them pays in
  the same single way" — held two mechanisms that build a score grid and keep a KV cache. Coverage
  and truth are different properties. Where a group's title makes a claim about its members, assert
  that claim: the fix here was that one chapter must be *exactly* the set the page's key counts, so
  a reader counting the chapter and a reader counting the key land on the same number.


- **Every term used as shorthand is defined in exactly one findable place, and everything else links there.** `SPEC.md` is the decision; `METHOD.md` is the apparatus. Splitting them is deliberate — an adversarially-graded specification cannot carry a glossary and two architecture diagrams without paying for it, and a first-time reader cannot do without them.
- **Explain the metric, not just its name.** "Held-out BPB, lower is better" names a measure. What it measures, what it is divided by, and why *that* denominator, is the part that lets a reader judge the table.
- **State the scale and the limits in the open text.** Not inside a collapsed disclosure. A qualifier a reader has to go looking for is a qualifier the document is hiding — and the scale of a proxy is the most important thing on the page it appears on.
- **The artefact people open first needs the grounding too.** A deployed page is read far more often than a specification. If its vocabulary is only defined in a Markdown file, it is not defined.
- **Render every diagram before committing it, and test that it renders — and this is now
  enforced in CI rather than merely stated.** A mermaid block is not verified by reading it.
  The render test existed and had **never once run in CI**, because `mermaid-cli` drives
  puppeteer and puppeteer insists by default on a chromium it downloaded itself. The
  `mixtures` shard the test lives in had already installed playwright's chromium two steps
  earlier, so the fix was one environment variable — `PUPPETEER_EXECUTABLE_PATH` — and the
  rule went from decorative to real at no cost. Resolve that path in a **subprocess**:
  `sync_playwright()` called inside a live pytest session raises `TargetClosedError` from
  its own teardown.

## The agent fleet

**`docs/AGENT_FLEET.md` is the architecture**: the unit lifecycle end to end, the five mechanisms
and what each is for, the enforcement points, what is deliberately *not* built and on what evidence,
setup, and the growth path. Read it before changing how agents run here.

The one distinction to carry away from it, because confusing these is how a rule becomes decorative:

| kind | example | can an agent ignore it? |
| --- | --- | --- |
| **Enforcement** | `PreToolUse` exit 2 · `permissions.deny` · a failing test · a GitHub ruleset | **No** |
| **Feedback** | pre-commit hooks | Yes — `--no-verify`, and absent on a fresh clone |
| **Request** | this file's prose · a prompt · `CLAUDE.md` | Yes, silently |

**A rule that must hold every time belongs in a hook or a test, never in prose** — and a rule that
lives only in prose should say so, rather than reading as a guarantee.

## Git workflow

- **One commit, one decision — at most 20 files and 5,000 changed lines, or say why.** Gated at the
  `commit-msg` stage by `tools/check_commit_scope.py`. `CHANGELOG.md` and `uv.lock` are not counted:
  the conventions already require the first in the same change, and the second is generated.

  **The limit is a prompt to ask "is this one decision or several?", not a measurement.** Where the
  honest answer is *one decision, five files*, record it and it is allowed:

  ```
  Wide-change: the hook, the ledger it imports and its guard cannot land separately
  ```

  **A hard cap with no escape would fight the property it protects.** Landing a `PreToolUse` hook
  means shipping the hook, the module it imports and its test together; split across three commits,
  the first two do not import, so `git bisect` lands on a tree that fails for a reason unrelated to
  what is being bisected — which is exactly what atomic commits exist to prevent. **Small batch is
  the means; independent revertibility is the end.** A merge or a revert is not judged, because its
  breadth is a property of the branches rather than a decision anyone is making now.

- **Every change lands on `main` via a pull request.** Branch → push → open a PR → merge. **Never push, merge, or force-push directly to `main`** — it's the protected branch that production is promoted from, and the base every PR previews against.
- Keep PRs scoped to one concern; unrelated edits get their own branch/PR.
- **Changelog:** record every user-facing change under `CHANGELOG.md`'s `[Unreleased]` section **in the same PR** (Keep a Changelog + SemVer).
- **Commit messages carry a `Co-Authored-By` trailer for the agent that wrote them, and nothing
  else.** No links back to an agent conversation, no run ids, no tool banners — those point at
  something nobody outside this machine can open, and they date badly. Attribution is useful;
  a dead link in the permanent history is not.
- **Neither a branch name nor a PR title names the source material.** Say what the change does
  (`refactor: rename the reference-material folder`), not which numbered topic it came from. The
  public history is the engineering work; the course's own structure stays out of it.

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

- **Before changing code that runs where you cannot watch it, read that environment's own log —
  and confirm the inputs your change depends on exist there.** `should-build.sh` was rewritten
  around a comparison with `origin/main`, shipped green, merged, and the very next build printed
  `origin/main could not be resolved (shallow clone?)`. **Vercel checks out a single-branch shallow
  clone**, so `origin/main` is absent and `git merge-base` has nothing to walk: the mechanism could
  never execute. It was inert in production and *unsound* wherever the ref did resolve — it skipped
  the build after a branch reverted its page, leaving the live preview serving content the branch
  no longer had. Hermetic tests prove a predicate is internally consistent and say **nothing** about
  whether its inputs exist in production, so a green suite over a mechanism that cannot run reads as
  coverage. One `get_deployment_build_logs` call, available the whole time, would have settled it
  before the pull request was opened. **Name the runtime inputs a change depends on — env vars,
  refs, files — and find one real execution that shows each is present.** Where it genuinely cannot
  be observed, say so in the pull request and make the unobservable path degrade to the previous
  behaviour rather than to a new one.

- **`VERCEL_GIT_PREVIOUS_SHA` is the last *successful deployment*, not the previous commit**, and it
  is only exposed while an Ignored Build Step is configured. Treat "no previous deployment" as a
  real state rather than an edge case: the `HEAD^` substitute in `should-build.sh` turns gate 1 into
  "what did the newest commit change", so a branch whose tip commit is not deployable can get **no
  preview at all** — and it is self-reinforcing, because a skipped build never becomes a successful
  deployment, so the variable stays empty. Still live; not yet fixed.

- CI (`.github/workflows/ci.yml`) is **four concurrent jobs, not one chain**. `test`: `uv sync --all-packages` → `ruff check` → `ruff format --check` → `pytest -m "not integration" -n auto --dist loadfile` → `node --check` over `find src/exercises -path '*/web/*' -name '*.js'`. `integration`: a **three-shard matrix** (`tokenization` · `mixtures` · `rest`), each shard syncing, caching and installing chromium, running `deploy/vercel/build.sh` once, then `pytest -m integration`. `security`: gitleaks over the full history. `push` is filtered to `main` — branches are covered by the `pull_request` event, because an unfiltered `push` ran every PR commit twice. `train`: `uv sync --all-packages --extra train` with **CPU-only wheels** (a Linux-scoped
  `pytorch-cpu` index in the root `pyproject.toml` — 191.8 MB instead of 2.7 GB, and 19 fewer
  packages in the lock), running only the files whose module-level `importorskip` would otherwise
  skip them entirely.
- CD: **Vercel**, gated. **Previews auto-deploy per PR**; **production never auto-deploys** (`vercel.json` → `git.deploymentEnabled.main: false`). One project serves every exercise's static `web/` under its slug (`/NN-slug/`) via `deploy/vercel/build.sh` → `public/`. (Netlify was the prior host — deactivated config retained in `deploy/netlify/`, pending decommission.)
- Production deploys go through the reusable `deploy-production.yml` (single source of truth, gated by the `production` environment), invoked two ways: **`deploy.yml`** (`workflow_dispatch`) for an ad-hoc deploy of `main`, and **`release.yml`** for a versioned release.
- **Releasing:** move `CHANGELOG.md`'s `[Unreleased]` → `[X.Y.Z]` (dated) and merge, then `git tag vX.Y.Z && git push origin vX.Y.Z`. `release.yml` creates a GitHub Release from that changelog section and deploys the tagged commit to production. **Then snapshot the standard files** — `uv run python tools/snapshot_standards.py` — so the release's `AGENTS.md`, `DESIGN.md` and configs are diffable from the next rewrite without going through git history. It must run *after* the tag exists, since it reads the tag.

## Web UI & content

Every deployable exercise's static `web/` bundle shares **one design system** — full reference in
`docs/DESIGN.md`, which carries the grid, the type scale, the components, what enforces each rule,
and a **numbered retro-fit checklist** for bringing an older exercise up to standard. Exercise 08 is
the reference implementation and every number in that document was measured on it. Read it before
building or changing a page; the rules that matter across exercises are below.

- **Interactive explainers follow two local files.** `docs/EXPLAINER_PROMPT.md` decides *what* one must be (the claim, the interaction that proves it, the topology and family, when **not** to build one). `docs/EXPLAINER_PATTERN.md` records *how* — DOM skeleton, class names, the state-and-render shape, copy voice. Both are gitignored, so they are on a working checkout but not on the remote; read both before building an explainer and don't re-invent the skeleton. Shipped references: `02-tokenization/web/how-it-works.html` and §1 of `03-data-collection-framework/web/chapters.js`.

- **One Apple-style design language** on every page: cool-gray/black surfaces, a single bright-blue accent (`#0068d1` light / `#2997ff` dark), system sans (no serif), soft-shadow rounded panels, and a `← Back` pill to the site root. **Six themes**, not two: the system light/dark pair plus `soft-light`, `tinted-dark`, `high-contrast` and `neon`, each defining the whole token set. A page styled for two of them is unreadable in the other four. Reuse the token names in `docs/DESIGN.md` — don't invent a per-exercise palette.
- **Write for a general audience.** The public pages are standalone, blog-style demos of an idea — a first-time visitor should be able to enjoy them without any course context. Favor plain, explanatory copy; the numbered topic eyebrow (`NN · Topic`) makes a nice light section label.
- **Credit the source course in one place.** A single **Credits** section at the bottom of the root `README.md` gives clear, warm credit to the course, instructor, and platform. Keeping it in one prominent spot — rather than repeating it across pages — keeps both the credit and the demos easy to read.
- **Canvas state changes animate** — morph with a short eased transition (≈550ms), not an instant redraw, keeping the framing stable so panels don't resize mid-toggle.

- **An interaction must never be the only route to a lesson.** Exercise 05's predict-before-reveal block was written with its transferable point inside the reveal, so a reader who declined to guess never reached it — and neither would any print or reduced-motion reader. The interaction may earn a point more vividly; the point itself belongs in prose that is always visible. The same rule is why a page's limitations sit in the open text and not inside a collapsed `<details>`: **a limitation a reader has to open a drawer to find is a limitation the page is hiding.**
- **A shared stylesheet can reserve space for an element each page has to add itself.**
  `_shared/page.css` styles `.rail` and, at 1180px and up, also sets `.wrap { padding-left: 260px }`
  — unconditionally, whether or not that page builds a rail. Only exercise 05 ever had the
  `<aside id="rail">` element and a builder for it, so **06 and 07 rendered a 260px empty gutter on
  every wide screen** and nothing failed. Copying `web/_shared/` into a new exercise copies the
  styles and not the markup they assume. When you vendor that directory, check what it expects the
  page to provide: measure `.wrap`'s computed padding against the rail's rendered width, and assert
  the pairing — gutter reserved **and** gutter filled.

- **Widening a page is two decisions, not one.** The landing page was a fixed 640px column at every
  viewport, using a third of a 1920px screen. The fix is not a bigger `max-width`: a 1200px line of
  prose is unreadable. Split the measures — prose keeps its line length, cards become a responsive
  grid — and **test both halves**, because a naive fix breaks the half nobody guards. Use
  `minmax(min(340px, 100%), 1fr)` and never a bare `340px`: an auto-fill track cannot shrink below
  its own minimum and will push a 320px phone sideways.

- **An `IntersectionObserver` on a detached node never fires, and says nothing.** Every figure
  builder returns its element before the page appends it, so registering the observer inside the
  builder observes a node that is not in the document yet. Three of exercise 08's plates never
  animated and one was invisible outright, with a clean console and a green suite. Defer by one
  frame and check `isConnected`, or register the observer from the code that does the appending.

- **`web/_shared/tokens.css` is NOT the token file, in any exercise, and the name has already cost
  time.** Every deployable exercise (03–08) vendors a byte-identical copy of exercise 03's
  *component* stylesheet under that name — its own first line says so. The real six-theme token file
  is `deploy/vercel/_shared/tokens.css`, served at `/_shared/tokens.css`, and each `index.html`
  links **both**. A scratch harness that linked only the vendored one rendered every glyph mark
  invisible, because `var(--bg)` was undefined and a `stroke: var(--bg)` simply does not paint. When
  you build a test page for an exercise, link `/_shared/tokens.css` the way `index.html` does. (The
  file is misnamed in six places; renaming it is its own change, not a drive-by.)

- **A `ch` or `em` measure resolves against the element that declares it, not the text inside it.**
  A pull quote wrapper at `max-width: 24ch` with `font-size: 16px` is 192px wide however large the
  38px quote inside it is set — one word per line. Put the measure on the element that carries the
  type, or use `rem`.

- **A full-bleed element still needs its own inset.** A `full` grid track runs edge to edge by
  design; that is what makes a plate span the page. Padding belongs on the element, not the track,
  or every full-width figure prints flush against the window on both sides.

- **The narrowness IS the length, and the lever is type size rather than measure.** A page is long
  because its content is narrow far more often than because it has too many words. Exercise 08's
  index was 30 rows at 306px; widening its container from 720px to 1,676px — more than double —
  moved a row to 292px, because a row was **six stacked bands on a four-column grid** and the extra
  width only shortened lines that were already short. Two bands with the prose in columns is 238px.
  Separately, a reader asking why a page "narrows too much" is not asking for longer lines: a
  77-character line at 22px is 951px and at 16px is 685px, so raising the body size gave 39% more
  screen at the *same* words per line. Size the columns by the character floor —
  `minmax(min(315px, 100%), 1fr)` is a 42-character line at 13px — and let the count follow the
  width the page actually has.

- **A variant nobody measures is a variant that ships broken.** While exercise 08 carried an A/B,
  `test_attention_measures.py` drove only the default, and the other variant shipped prose at **111
  characters a line** for two commits with the whole suite green. A guard that measures one of two
  shipped layouts has a hole exactly the size of the other one. While a flag lives, every guard that
  can differ between its values runs against both — and the flag carries a written end date, or a
  temporary switch quietly becomes permanent.

- **Deleting a branch can delete the declaration above it.** Cutting a conditional out of a loop in
  exercise 08 took a `const body = …` with it, because the branch had been inserted directly above
  that line. The page threw `body is not defined` half way through building its index — thirty rows
  became none — and it was caught only because one test fixture listens for `pageerror` on the real
  page rather than asserting solely about its own harness. **Point at least one browser fixture at
  the real page and fail on any console or page error.**

- **A rule with no guard decays, and the dead CSS proves it.** `web/_shared/page.css` has styled
  `.rail-link.on` — the active-section marker — since before most of these pages existed, and only
  exercise 03 ever sets the class. 05, 06 and 07 build a contents rail that never marks where the
  reader is; 06 and 07 reserve a 260px rail gutter they never fill; all six vendor
  `_shared/explainer.css` and only two use it. When you vendor a shared stylesheet, diff what its
  rules select against what your page emits, and write down what you chose not to build.

- **Editing non-ASCII HTML** (`—`, `→`, `·`, math glyphs): use the Edit/Write tools. **Never** `perl -0pi`/`sed` with wide-char escapes — byte-mode rewrites double-encode UTF-8 into mojibake.

## Instruction files (this system)

- `AGENTS.md` (this file) is the single source of truth. `CLAUDE.md` = `@AGENTS.md`. `.github/copilot-instructions.md` and `.cursor/rules/conventions.mdc` point here.
- Component-specific notes live in a nested `CLAUDE.md` inside that exercise folder.
- Machine-enforceable rules live in tooling (`pyproject.toml`), not prose — this file references the tooling rather than restating it.

- **The last two released versions of every standard file are frozen in `docs/standards-history/`,
  and you diff against them before rewriting one.** Git has every version; the problem is that
  finding one means first knowing a rewrite happened, and the rewrites worth comparing are the ones
  nobody remembers making. `docs/DESIGN.md` went **199 → 488 lines in a single commit** and, of its
  30 rules, 19 survived reworded and **nine were dropped with no replacement anywhere in the repo** —
  including "never a chart library", "a glossary must not be hover-only", and "mark pipeline stages
  with an explicit class, never `:nth-child`", each one a lesson from a defect that had already cost
  a page. Nothing went red, because **no guard can cover a rule that used to be written down.**

  ```bash
  diff docs/standards-history/DESIGN.v0.12.0.md docs/DESIGN.md   # what a rewrite actually dropped
  uv run python tools/snapshot_standards.py --check              # is the newest release captured?
  uv run python tools/snapshot_standards.py                      # capture it, after a release
  uv run python tools/snapshot_standards.py --ref v0.11.0        # rebuild an older one, any time
  ```

  The set is `tools/snapshot_standards.py::STANDARDS` — `AGENTS.md`, `docs/DESIGN.md`, `ci.yml`,
  `.pre-commit-config.yaml`, `pyproject.toml`, `.gitignore`, `vercel.json` (13 lines, one of which
  decides whether production deploys itself) and `.gitleaksignore` (where a broad entry silently
  disables the secret scan). **Add the snapshot to the release
  ritual**, alongside moving `[Unreleased]` in the changelog. `tests/test_standards_history.py`
  asserts each copy is byte-identical to the tag it names, carries its `FROZEN COPY — NOT IN FORCE`
  banner (an agent reading an archived `AGENTS.md` as live policy is the obvious failure), and that
  at least two versions are kept, so a rewrite always has something to be compared against. **Rewriting a standard file is not the same as editing one:**
  list what the rewrite drops before you commit it, and put anything you are keeping back.

  **The archive is gitignored, and that is a decision with two consequences.** Tracking it would put
  a second copy of `AGENTS.md` and `DESIGN.md` on the remote — the same argument that untracked the
  notebooks — and it is only ever read on the machine doing the rewriting. So every guard that
  reads it **skips** on a clone and in CI, which means this rule is enforced by whoever has the
  archive or by nobody. Two guards that run everywhere hold the pair together: one fails if a
  snapshot is ever committed, one fails if the ignore rule disappears. It must be exactly one of
  tracked or ignored, never neither.

  **It is deliberately NOT in `PATTERNS`, and the reason matters more than the fact.** It was, on
  the argument that *untracked and unbacked-up* is the class this repo has lost twice — but every
  entry in `PATTERNS` earns its place by being **permanently** lost, and a snapshot is
  `git show <tag>:<file>` plus a banner, so `--ref <tag>` rebuilds any of them byte for byte. What
  the backup actually bought was *history of immutable files*: the byte-identical guard means a
  second version can never legitimately exist, and across the store's whole life the only change
  ever recorded against a snapshot was a reworded banner. The one irrecoverable case is a snapshot
  of a **deleted tag**, so `test_the_archive_is_not_backed_up_because_it_is_rebuildable` asserts
  both halves — not in `PATTERNS`, *and* every tag a snapshot names is still reachable. If tags ever
  start being deleted, that guard goes red and this decision gets revisited rather than quietly
  outlived.
