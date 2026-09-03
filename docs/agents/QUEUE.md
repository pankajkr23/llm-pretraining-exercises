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
| 3 | Arm the fleet | `unit-arm-the-fleet` | **done.** PK ran the installer; the guard is wired and all six behaviours were observed, not assumed — worktree write blocked, `sed -i` on a guard blocked, standard file blocked, ordinary source allowed, reading a guard allowed, `AGENT_STOP` halts and resumes. Reviewers are `Read, Grep, Glob` |
| 4 | Live defects on deployed pages | `unit-live-defects` | **done**, pending review — exercise 01's 26 unterminated declarations and exercise 04's 7 orphan properties, both now guarded |
| 5 | The shared `web/_shared/` layer | `unit-shared-layer` | **partly done**, pending review. 2,578 lines of unreferenced vendored code removed and guarded. **Still open:** `.back:hover` in six identical copies, the two theme pickers, seven names for two controls, and 24 orphan CSS classes |
| 6 | Exercise 09 | `unit-09` | **blocked** — see the unit for what on |
| 7 | Exercise 10 | `unit-10` | not started |
| 8 | Retro-fix 07 → 01 | `unit-07-retrofit` … `unit-01-retrofit` | not started. Deliberately **after** 09 and 10, and able to run alongside them once row 5 lands |
| 9 | Grow the agent roster | `unit-agent-roster` | not started. Read-only agents first, then three writers with **disjoint** scopes |
| 10 | The platform plan, for a parallel workstream | `unit-platform-plan` | **drafted** — `~/.claude/plans/agent-platform.md`. Repository-agnostic by construction |

**Read the order as a default, not a rule.** Rows 4 and 5 come before 6 because they are cheap and
because row 5 gates parallelism; row 8 comes last because 09 and 10 teach training, which is the
point of the repository, and the retro-fix is polish on work already shipped.

**Waiting on a human, and nothing else moves it:** exercise 08 is finished, released and live, and
has **not been submitted**. It is the only item here that converts completed work into a result, and
it appears in no row above because it is not work — it is a decision.

---

## How to work the queue — do not stop between rows

**Finishing a unit is not a reason to stop. Opening a pull request is not a reason to stop.**
Merging, tagging, the production gate and submission are a human's, and they always will be — but
they are **handoffs, not blocks**. The pull request waits for a person; the agent does not wait for
the pull request. Open it, then start the next row on a fresh branch off `main`.

After each unit, before moving on, do all four:

1. **Update this file** — the row's state, and a log entry saying what the unit found and what it
   cost. Write the entry for the *new* pull request when you open it, not after it merges.
2. **Self-assess against what the unit actually did**, not against what it set out to do. Name what
   was deliberately left undone and which row now owns it — that is how row 5 inherited
   `.back:hover` rather than losing it.
3. **Re-read the order.** The rows are a default, not a schedule. If a unit turned up something
   that changes the order, change it here and say why.
4. **Pick the next row and begin.** No pause for acknowledgement.

**The only reasons to stop**, and each is written down elsewhere rather than judged in the moment:

- a decision `AGENTS.md` marks as a human's — deleting a protected path, rewriting history, changing
  a ruleset, submitting work
- **two consecutive review rounds producing new BLOCKERs** — the unit is not converging, and a third
  round is a sunk cost
- a guard goes red and the honest fix is outside the unit's declared scope
- an operation the sandbox or the permission layer refuses. That is a boundary doing its job, not an
  obstacle to route around: report it and take the next row

Anything else — a question, an uncertainty, a finding worth flagging — goes in the log and the pull
request body, and the work continues.

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

### unit-agent-roster — read-only agents first, writers second
- status: QUEUED
- scope: `docs/agents/reviewers/`, `tools/install_agent_fleet.py`, `tests/test_agent_guard.py`
- what: add `research` and `critique` as read-only personas alongside the four reviewers, then
  split the writing role into `coding` (implementation paths), `testing` (test paths **only**) and
  `documentation` (docs and changelog).
- **why coding and testing must be separate agents, and it is a measured result rather than a
  preference:** ImpossibleBench found a frontier model exploited test cases **76%** of the time,
  dropping to **near zero** when test access was made read-only. An agent that writes both the
  implementation and its tests will write tests that pass, and the suite becomes decorative.
- sequencing: read-only agents are pure upside — no writes, no conflicts, no ordering. The three
  writers run **sequentially first**, because the handoff is where information is lost and the
  contract has to carry enough for the next agent to work without re-deriving the unit. Concurrency
  only after the scope guard has been *watched* refusing a cross-scope write.
- what synchronises them: the state file and the finished diff, **not messages between agents**.
  Message passing needs a protocol, ordering guarantees and a deadlock story; a shared artefact
  needs none of those and is readable by a human too.
- explicitly not in scope: an orchestrator that decides which agent runs next. The hub does that.

### unit-platform-plan — the multi-agent platform, for a parallel workstream
- status: DRAFTED — `~/.claude/plans/agent-platform.md`, awaiting PK's read
- scope: none in this repository. The document is **deliberately repository-agnostic**: everything
  project-specific is stripped, because its subject is the platform rather than any codebase.
- what it carries: the topology and the topologies rejected with reasons; the five layers; the
  enforcement / feedback / request distinction; the fully-researched OSS observability stack with
  licences checked and six candidates excluded on licence or maintenance grounds; the scale path
  with a written trigger per step; what is refused **on measurement** with the number to watch for
  each; and the portable lessons ordered by what they cost.
- why it lives outside this repository: tracking it here would put a second copy of a platform
  specification inside a project that is not the platform, and the second copy is the one that
  drifts.

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
2026-09-03  live-defects  #99 opened: exercise 01's four proof pages declared every diagram token
                          inside their dark blocks with NO semicolons — 26 of them, 8 blocks. A
                          value runs to the next `;` or `}`, so each block declared ONE property
                          whose value was the rest of the block. Worse than the audit recorded:
                          the swallowing property is not a colour either, so all five were broken,
                          not four. Verified by parsing the blocks before and after
2026-09-03  live-defects  #99 also: exercise 04 referenced 7 properties declared in no stylesheet,
                          so their hardcoded fallbacks won in all six themes — including a tooltip
                          that was a dark chip on a light page. Mapped onto the tokens DESIGN.md
                          publishes. Read back per theme in a browser: the tip now tracks all six
2026-09-03  live-defects  the `#fff` on `var(--accent)` sweep is PARTLY done and the rest is
                          row 5's: 8 fixed in files one exercise owns, and `.back:hover` left
                          alone because it lives in six byte-identical vendored copies that would
                          drift if one changed
2026-09-03  arm-fleet     row 3 CLOSED. PK ran install_agent_fleet.py; the guard is wired and all
                          six behaviours were observed rather than assumed: a uv.lock write from
                          inside a real worktree BLOCKED (#93's bug 1, live), `sed -i` on a guard
                          file BLOCKED (#93's bug 2, live), a standard file with no unit declared
                          BLOCKED, ordinary source ALLOWED, reading a guard file ALLOWED, and
                          AGENT_STOP halting then resuming (0 -> 2 -> 0). All four reviewers
                          declare `tools: Read, Grep, Glob`
2026-09-03  arm-fleet     one of those checks was mislabelled and the guard was right: a resume
                          test read exit=2 where the label said 0, because UNIT.md still scoped
                          writes elsewhere. Re-run clean. The test UNIT.md was then removed, which
                          restores the documented default — no unit file means scope is inert,
                          while measured data, guards and standards stay refused regardless
2026-09-03  tracking      #102 merged: row 3 closed, rows 9 and 10 added. It did NOT log itself,
                          so the checker refused the next branch that touched this file — the
                          open-time convention was followed for the ROWS and forgotten for the
                          LOG line, which is the same regress in a smaller form
2026-09-03  shared-layer  #101 opened: removed 2,578 lines of vendored code no page referenced —
                          anim.js (167 lines x 6 copies, seven exports, zero importers) plus
                          explainer.js and num.js from the four exercises linking neither. The
                          served site drops 92 -> 78 files. All nine pages render with no console
                          error and no failed request
2026-09-03  shared-layer  THE AUDIT'S OWN NUMBERS WERE WRONG and so were mine. DESIGN.md said
                          explainer.css was "used by two"; it is used by 01, 02, 03, 05, and 03
                          alone emits 36 of its 56 classes. My first extractor said it was used by
                          NOBODY, because it looked for el(tag, class) while 03 calls a local
                          $(tag, class) — deleting on that would have removed a live stylesheet.
                          The counts are now derived by a test, not typed into the standard
2026-09-03  shared-layer  24 orphan CSS classes deliberately NOT removed: a class emitted by a
                          path the extractor cannot see is indistinguishable from a dead one,
                          which is the mistake above. Needs browser verification, not a grep
2026-09-03  scope-limit   commit-scope raised 10 files/500 lines -> 20/5,000 on PK's call. The
                          deletion above was one decision applied ten times and could not be split
                          without a red or unguarded intermediate tree, so the trailer had become
                          the normal path rather than the exception
2026-09-03  mermaid       #100 opened: the diagram-render rule had never once run in CI. mermaid
                          -cli wanted its own puppeteer chromium while the shard had already
                          installed playwright's. One env var; the test now passes rather than
                          skips
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
2026-09-03  fleet         #103 merged: install_agent_fleet.py --drift, wired into the post-merge
                          hook. A reviewer copied into .claude/ and then edited there diverges
                          silently from its tracked source, and the installed copy is the one that
                          runs — so the drift is invisible in review by construction
2026-09-03  exercise-09   #104 opened: the two explainer standards become tracked, and 09 is
                          scaffolded. The generator's own test used 09/lossheads as its fixture —
                          the exact spec 09 would claim — so the collision check found itself the
                          moment 09 existed. Fixture moved to 99/scaffoldprobe
```
