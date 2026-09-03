# The unit queue

The fleet's source of truth for what to do next, and the record of what happened. **Tracked**, so it
survives a crash, a context reset, a branch switch and a fresh clone.

It replaces `TODO.md` for this purpose for one blunt reason: `TODO.md` is gitignored, so no worktree
gets it from a clone. `TODO.md` stays as the human backlog; this is what an agent reads.

## Why a file rather than a summary

Context degrades **well before** the window fills — Chroma's study measured reliability dropping on
trivial retrieval across 18 models — and Anthropic's harness work names *context anxiety*: models
begin wrapping up prematurely as they approach what they believe is their limit. Their verdict on
the obvious mitigation is blunt: **"Compaction isn't sufficient."**

So the state lives on disk and is re-read in full whenever an agent starts, rather than being
carried in a summary that is lossy by construction.

## The rule for entries

**Append evidence, not prose.**

```
2026-09-03  unit-07-retrofit  pytest -m "not integration" -> 1449 passed, 1 skipped @ b0456e9
2026-09-03  unit-07-retrofit  BLOCKED: rail-inner missing; logged as finding F3, continuing
```

not

```
2026-09-03  unit-07-retrofit  fixed the rail
```

The first can be checked by anyone later. The second is a claim, and this repo has learned what an
agent's unchecked claims are worth: two data-handling invariants once returned "no findings" for
every input across four commits, and nothing in any log said so.

## Unit format

Each unit carries an explicit acceptance contract. Copy this block, fill it in, and write the scope
half into `.claude/UNIT.md` before starting.

```markdown
### unit-NN-slug
- status: QUEUED | IN FLIGHT | BLOCKED | PR OPEN | DONE
- scope: src/exercises/NN-slug/
- scope: CHANGELOG.md
- acceptance:
  - [ ] ruff check + format --check clean
  - [ ] unit suite green
  - [ ] integration shard green after deploy/vercel/build.sh
  - [ ] LOCAL-ONLY gates run on this worktree
  - [ ] every new guard watched failing
  - [ ] screenshots at 2560/1920/1440/1180/768 in four themes, and read
  - [ ] every number in prose derived, not typed
- reviewers: reader, engineer, auditor
- evidence:
```

`- scope:` lines are what `tools/agent_guard.py` reads. Anything outside them is refused, and the
refusal tells the agent to log a finding and continue rather than to stop.

## Stop conditions

The unit is marked BLOCKED and the fleet moves to the next queued one. It never idles and never
guesses.

- A guard goes red and the fix is not obviously inside scope
- A change would touch a protected path, measured data, or a standard file
- **Two consecutive review rounds produce new BLOCKERs** — the unit is not converging
- A decision is needed that `AGENTS.md` marks as a human's
- The audit log shows a write outside scope

---

## Where the work is — read this first

**An entry is written when a pull request is OPENED, not after it merges**, and
`tools/queue_status.py --check` enforces it from pre-commit's `post-merge` stage. Recording after
the fact cannot work: a pull request cannot log its own merge, so the check failed on every `git
pull` and the fix needed recording in turn. If this file disagrees with `git log`, git is right and
this is the bug — say so rather than working around it.

Numbers are deliberately absent here. A count typed into prose goes stale while the thing it counts
moves, which is a failure this repo has already paid for more than once — so read the live ones:

```bash
gh pr list                                  # what is open
git log origin/main --oneline -15           # what landed
uv run pytest -m "not integration" -q       # whether it is green
```

**Every row names the unit that does it**, and those units are defined in full further down. There
is no separate numbering anywhere — if a conversation says "step 2" or "phase 3", it is using a name
this file does not define, and the answer is the unit name instead.

| # | what | unit | state |
| --- | --- | --- | --- |
| 1 | Unblock the pull-request backlog | — | **done.** The batch in the log below is merged; `main` is linear |
| 2 | Track progress in one place | — | **done.** This file, enforced by a checker in CI. `WORKPLAN.md` holds the arc, `TODO.md` the scratch |
| 3 | Arm the fleet | `unit-arm-the-fleet` | **partly done.** The guard, the reviewers and this file are merged — but `tools/install_agent_fleet.py` **has never been run**, so every mechanism is present and none is armed |
| 4 | Live defects on deployed pages | `unit-live-defects` | **NEXT.** Not started. Readers hit these today |
| 5 | The shared `web/_shared/` layer | `unit-shared-layer` | not started, and it **gates running anything in parallel** |
| 6 | Exercise 09 | `unit-09` | **blocked** — see the unit for what on |
| 7 | Exercise 10 | `unit-10` | not started |
| 8 | Retro-fix 07 → 01 | `unit-07-retrofit` … `unit-01-retrofit` | not started. Deliberately **after** 09 and 10, and able to run alongside them once row 5 lands |

**Read the order as a default, not a rule.** Rows 4 and 5 come before 6 because they are cheap and
because row 5 gates parallelism; row 8 comes last because 09 and 10 teach training, which is the
point of the repository, and the retro-fix is polish on work already shipped.

**Waiting on a human, and nothing else moves it:** exercise 08 is finished, released and live, and
has **not been submitted**. It is the only item here that converts completed work into a result, and
it appears in no row above because it is not work — it is a decision.

---

## Queue

Nothing here is IN FLIGHT until a human says so.

**The order changed.** It was the retro-fix order from the workplan; it is now 09 and 10 first,
because those are the two exercises that teach training and the retro-fix is polish on work already
shipped. The workplan's stage numbering is the same decision written the other way round.

### unit-08-notebook — **the pilot. Read this one in full before approving.**

- status: QUEUED — awaiting approval
- scope: `src/exercises/08-modern-attention-variants/`, `CHANGELOG.md`
- reviewers: reader, engineer, auditor

**The problem, measured.** The package is `numpy`-only and holds six modules — `config`, `cache`,
`sources`, `catalogue`, `timeline`, `story` — every one of them chronology machinery. There is **no
attention implementation anywhere in the exercise**. So the notebook's 28 cells import the catalogue
and print what the web page already renders: no `softmax(QKᵀ/√d)V`, nothing that touches a GPU,
nothing whose configuration can be varied. It could not have been otherwise, and the same audit
across all eight notebooks found only two that touch torch at all and **none** that runs a GPU
workload with varying settings.

**What gets built.** A new module of small, readable implementations — in the *package*, with tests,
because a notebook is gitignored and code in cells is invisible to CI and rots silently:

| variant | why it is in the set |
| --- | --- |
| scaled dot-product | the base every other one is a modification of |
| MHA · MQA · GQA | the cache bill, and the whole point of the 6.44 GB → 51.54 GB arithmetic the exercise already computes |
| sliding window · attention sinks | the length bill, and what "streaming" actually means |
| ALiBi · RoPE | position, and why extrapolation breaks |

**The equivalence tests are the lesson.** Each is a fact you can hold, and a test that fails if the
code stops being true:

- GQA at `n_kv == n_heads` **is** MHA, to floating-point tolerance.
- GQA at `n_kv == 1` **is** MQA.
- A sliding window of full width **is** dense attention.
- ALiBi at slope 0 **is** no bias at all.
- Attention sinks with `k=0` and a full window **is** dense attention.

A paragraph claiming these is worth less than five `allclose` assertions that go red when they stop
holding — and the tests double as the map from one variant to the next.

**Plus a benchmark helper** reporting wall time and peak memory across MPS, CUDA and CPU, because the
cost these variants exist to pay down is not vivid from arithmetic alone.

**The notebook then becomes scenarios**, not a results tour: *"serving eight users on one GPU"*
(MHA → GQA → MQA, watch the cache), *"a chat that forgets"* (window vs sinks), *"the model breaks
past its training length"* (RoPE and its scalings). Run, read the numbers, change a setting, watch
what moves. `lite` finishes in under ten minutes; the full run is one variable away.

**Three consequences worth approving deliberately, because none is free:**

1. **This adds `torch` to an exercise that has none.** It has to: MPS and CUDA are the point. It goes
   in as an optional extra with a module-level `importorskip`, which then costs **two** registrations
   — `OPTIONAL_DEPENDENCY_GATES` *and* a CI job that installs it. `AGENTS.md` is explicit that a
   gated file in neither runs **nowhere**, and this repo has already lost 46 tests exactly that way.
2. **The `train` job is the one that installs torch**, so the new tests join it — CPU-only wheels,
   191.8 MB rather than 2.7 GB.
3. **It does not touch the published page.** The chronology, the catalogue and the web bundle are
   out of scope. If the work seems to want them, that is a finding, not a licence.

**Why this is the pilot.** It exercises the whole loop — research, plan, implement, test, review,
iterate, PR — on work small enough to watch in one sitting, and what it produces is the thing that
was actually wanted. The success criterion is **not** the diff: it is that the harness needed no
intervention, and that a reader can say what each guard did and why the run stopped where it did.

### unit-arm-the-fleet — everything is installed and nothing is armed
- status: QUEUED
- scope: `.claude/` (local, gitignored), `docs/agents/`
- what: run `tools/install_agent_fleet.py`, which copies the guard wiring and the reviewers into the
  gitignored `.claude/` tree. Then write the `.claude/UNIT.md` template — scope paths plus acceptance
  checks, every one `false` — and agree where the run log lives.
- why it is its own unit: the guard, the four reviewers and this queue are all **merged**, so the
  machinery reads as present. None of it runs until the installer has been executed on the checkout,
  and nothing in CI can tell the difference, because `.claude/` is gitignored by design.
- verify by observing, not by assuming: a write to `uv.lock` from **inside a worktree** is refused,
  `sed -i` on a guard file is refused, a reviewer cannot write, and `touch AGENT_STOP` halts a run.

### unit-live-defects — readers hit these today
- status: QUEUED
- scope: `src/exercises/01-introductions/`, then `src/exercises/04-data-cleaning-dedup/`
- what: **01** declares its dark-theme diagram tokens with no semicolons across four proof pages, so
  four properties are never declared in the dark blocks and a dark-theme reader gets light diagram
  colours. **04** references seven custom properties that exist in no theme, so the fallback always
  wins. One pull request per exercise, plus one repo-wide guard: no custom property is referenced
  that no theme declares.
- reviewers: reader, engineer

### unit-shared-layer — **the gate on running anything in parallel**
- status: QUEUED
- scope: `src/exercises/*/web/_shared/`, `deploy/vercel/_shared/`
- what: dead code (`anim.js` is vendored six times and imported by nothing; `explainer.css` is
  linked by six pages and used by two), two theme pickers, seven names for two controls, and
  promoting exercise 08's theme and contrast guards into `tests/` for every deployable page.
- why first: exercises 09 and 10 will vendor this directory, so building them first means inheriting
  the breakage and fixing it twice — and until it lands, every retro-fix unit edits the same files
  and three agents collide on all of them.
- reviewers: reader, engineer, auditor

### unit-09 — loss functions and output heads
- status: BLOCKED — on #96, and on deciding what of the explainer documents becomes tracked
- what: scaffolded with `tools/new_exercise.py`, **never by hand**: six test families apply the
  instant `pyproject.toml` lands. Two registrations are deliberately deferred by the generator and
  must be done by a human — the landing card, and the spine ledger entry.
- blocker, stated plainly: the two documents an explainer is *required* to be built from are
  gitignored, so no worktree, no clone and no CI can read them. An agent asked to build 09's
  explainer has no access to the specification it is graded against.

### unit-10 — the training loop
- status: QUEUED
- what: same scaffold, same contract. Two extra rules for the flagship run: exercise `save()` in a
  two-step run **before** any long one, and print tokens-consumed ÷ corpus-tokens per lane next to
  the mixture table before starting.

### unit-07-retrofit … unit-01-retrofit
- status: QUEUED — after 09 and 10, and able to run alongside them once `unit-shared-layer` lands
- scope: one exercise each, `src/exercises/NN-*/`, `CHANGELOG.md`
- what: the twelve-part spine in order, the page rebuilt to `docs/DESIGN.md`, the README's
  three-reader path, and the notebook rebuilt to the rule above. Exercise 07's own `PROGRESS.md`
  already names two defects.
- reviewers: reader, engineer, auditor, **continuity**

---

## Log

**This log had one line while nine pull requests merged past it**, which is worth recording at the
top rather than quietly backfilling. The convention was written here and then not followed, so the
one file built to answer *"where are we?"* could not. The entries below were reconstructed from
`git log` and the pull requests afterwards — which is exactly the re-derivation this file exists to
make unnecessary.

Two things follow. Nothing below carries a `@ sha` unless it was checked, because a fabricated
evidence line is worse than a missing one. And the backlog batch was not run as *units* — it
predates the harness — so it is logged as what it was.

```
2026-09-03  fleet         queue created; no unit has run yet
2026-09-03  backlog       #87 merged: design standard named a CSS class that does not exist
2026-09-03  backlog       #88 merged: backup store obeyed the global gitignore
2026-09-03  backlog       #89 merged: stop backing up the rebuildable standards archive
2026-09-03  backlog       #90 merged: an undeclared CI skip is now a failure
2026-09-03  backlog       #90 CI RED first: test_backup_store_versions_everything skipped
                          undeclared. #88 added that file AFTER this ledger was written. Declared
                          in tests/_skips.py; escalate() returns ALLOWED with the entry and
                          ESCALATED without it, which reproduces the failure
2026-09-03  backlog       #91 merged: commit scope guard, 10 files / 500 lines
2026-09-03  backlog       #91 CAUGHT ITS AUTHOR one PR later: a merge commit given a custom
                          message lost the "Merge " exemption and was refused at 30 files /
                          1754 lines. Split into a pure merge plus a 1-file doc commit
2026-09-03  backlog       #92 merged: fleet architecture. Two defects found while resolving it —
                          STEER.md named twice and read by nothing, and section 9 omitted the one
                          step that arms the system
2026-09-03  backlog       #92 reframed on PK's objection: "simplicity over capability" was the
                          wrong constraint. Rewritten as verifiability, and section 7 split into
                          refused-on-measurement / sequenced / inapplicable-at-one-user
2026-09-03  backlog       #93 merged: the fleet guard. Two bugs fixed first — it failed OPEN
                          inside a worktree (root from __file__), and Bash bypassed it entirely
                          (echo >, sed -i). Verified: main checkout BLOCKED, worktree ALLOWED
2026-09-03  backlog       #95 merged: Tier-1 page invariants. 14 integration tests, reachable via
                          the rest shard, confirmed rather than assumed
2026-09-03  backlog       #94 merged LAST, and held back for two defects of its own: build() took
                          the verdict as a DEFAULT so --write attested PASSED on a machine with no
                          reference material; and the digest covered 474 files while the check
                          reads 293. Both fixed, both regression-tested
2026-09-03  changelog     all 8 conflicted on CHANGELOG.md's [Unreleased] anchor, mutually.
                          Ruleset gained "squash"; each branch took main and kept both sides.
                          Consolidated [Unreleased] to one heading per section on the way
2026-09-03  changelog     resolver was WRONG TWICE before it was right: it handled one conflict
                          region when #93 had two, then kept both sides of an already-shared
                          bullet and produced 39 where 33 was correct. Aborted both times.
                          Final check: result == union of both sides, 0 missing, 0 extra
2026-09-03  ruleset       `update` rule removed by PK, so merges no longer need a bypass.
                          delete_branch_on_merge on. required_status_checks still ABSENT —
                          nothing makes CI green a condition of merging
2026-09-03  prereqs       #96 merged: two prerequisites before exercise 09. The integration shards'
                          check names carried their own exercise list, so a required check pinned
                          to one would have stopped reporting the moment 09 joined the rest shard.
                          And the scaffolder named exercises by a convention none of the eight uses
2026-09-03  tracking      the queue checker found its own author's stale entry: the line above
                          still marked that pull request in flight after it had landed, and the
                          check passed, because naming a pull request counted as recording it.
                          Third bug of one family in that tool — the checker confidently wrong
                          about what it was looking at, each time presenting as a clean pass. It
                          also preferred a local `main`, which is stale the moment it is not
                          pulled; `origin/main` is the authority and now comes first
2026-09-03  tracking      and then the new check flagged the line above THIS one, because that
                          line quoted the marker it searches for. A status and a quotation of a
                          status are lexically the same; only knowledge tells them apart, which is
                          the limit recorded in the tool rather than parsed around
2026-09-03  fleet         this file reconciled with reality; WORKPLAN.md and TODO.md too
2026-09-03  tracking      #97 merged: this file became the single source of truth, WORKPLAN.md and
                          TODO.md became the arc and the scratch, and the checker began enforcing
                          it in CI rather than declaring itself local on an unmeasured cost
2026-09-03  tracking      #98 records itself, which is the convention the checker forced into the
                          open. A pull request cannot log its own merge after the fact, so the
                          check failed on `git pull` after every single merge and the fix — another
                          pull request — needed recording in turn. An entry is now written when a
                          pull request is OPENED, and the regress closes
```
