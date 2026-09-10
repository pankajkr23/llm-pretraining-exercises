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

2026-09-04  process       #104 closed and split: it carried three concerns on one branch, against
                          "one pull request per exercise" here and "keep PRs scoped to one
                          concern" in AGENTS.md. #105 is exercise 09, #106 is exercise 10 stacked
                          on it. PK caught it
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
| 4 | Live defects on deployed pages | `unit-live-defects` | **done** — exercise 01's 26 unterminated declarations and exercise 04's 7 orphan properties, both now guarded |
| 5 | The shared `web/_shared/` layer | `unit-shared-layer` | **done** — 2,578 lines of unreferenced vendored code removed and guarded; the theme pickers and control names settled |
| 6 | Exercise 09 | `unit-09` | **done.** Both blockers cleared; 54 tests, the notebook, the page and its registration all shipped |
| 7 | Exercise 10 | `unit-10` | **done, bar the reviewer pass** — stage 14, the one piece of engineering still owed on either exercise |
| 8 | Retro-fix 07 → 01 | `unit-07-retrofit` … `unit-01-retrofit` | **done.** One pull request each, #109-#132; it ran alongside 09 and 10 once row 5 landed |
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
- status: DONE — installed, armed, and all six behaviours observed rather than assumed. Its one remaining gap is that the `PreToolUse` matcher omits `Bash`, tracked separately
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
- status: DONE — exercise 01's 26 unterminated declarations and exercise 04's 7 orphan properties, both fixed and both guarded
- scope: `src/exercises/01-introductions/`, then `src/exercises/04-data-cleaning-dedup/`
- what: **01** declares its dark-theme diagram tokens with no semicolons across four proof pages, so
  four properties are never declared in the dark blocks and a dark-theme reader gets light diagram
  colours. **04** references seven custom properties that exist in no theme, so the fallback always
  wins. One pull request per exercise, plus one repo-wide guard: no custom property is referenced
  that no theme declares.
- reviewers: reader, engineer

### unit-shared-layer — **the gate on running anything in parallel**
- status: DONE — 2,578 lines of unreferenced vendored code removed, the two theme pickers reconciled and the control names settled
- scope: `src/exercises/*/web/_shared/`, `deploy/vercel/_shared/`
- what: dead code (`anim.js` is vendored six times and imported by nothing; `explainer.css` is
  linked by six pages and used by two), two theme pickers, seven names for two controls, and
  promoting exercise 08's theme and contrast guards into `tests/` for every deployable page.
- why first: exercises 09 and 10 will vendor this directory, so building them first means inheriting
  the breakage and fixing it twice — and until it lands, every retro-fix unit edits the same files
  and three agents collide on all of them.
- reviewers: reader, engineer, auditor

### unit-09 — loss functions and output heads
- status: DONE — both blockers cleared (#96 merged; the explainer documents are tracked as of #105). Shipped: 54 tests, the notebook, the page, and its registration
- what: scaffolded with `tools/new_exercise.py`, **never by hand**: six test families apply the
  instant `pyproject.toml` lands. Two registrations are deliberately deferred by the generator and
  must be done by a human — the landing card, and the spine ledger entry.
- blocker, stated plainly: the two documents an explainer is *required* to be built from are
  gitignored, so no worktree, no clone and no CI can read them. An agent asked to build 09's
  explainer has no access to the specification it is graded against.

### unit-10 — the training loop
- status: DONE, bar its reviewer pass — stage 14 in the exercise's own PROGRESS.md is the one piece of engineering still owed on 09 or 10
- what: same scaffold, same contract. Two extra rules for the flagship run: exercise `save()` in a
  two-step run **before** any long one, and print tokens-consumed ÷ corpus-tokens per lane next to
  the mixture table before starting.

### unit-07-retrofit … unit-01-retrofit
- status: DONE — 07 through 01, one pull request each, #109-#132
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

2026-09-03  exercise-09   09 built to its requirements: seven numbers and two findings, all
                          generated into results/ and rendered into both RESULTS.md and the page's
                          data file. A t+2 head sits above a t+1 head on 297 of 300 steps; an
                          off-by-one target shift trains to 0.18 against the correct shift's 4.14 —
                          the bug makes the loss BETTER
2026-09-03  review        three reviewers read 09 and found three blockers, every one a claim that
                          read as checked. The boundary mask kept every pad-to-pad pair (-1 == -1),
                          and its guard asserted the same expression the implementation used, so it
                          held for any input. RESULTS.md claimed every figure was generated and
                          fifteen were typed — inside the template the byte-equality test compared
                          against. And an importorskip turned a repo-wide guard red
2026-09-03  process       #105 opened for exercise 09 alone. #104 had carried three concerns on
                          one branch, against "one pull request per exercise" in this file and
                          "keep PRs scoped to one concern" in AGENTS.md. PK caught it; #104 is
                          closed and every commit in it is reachable from #105 and #106
2026-09-03  tripwire      tracking a file the tripwire watches turns it red on a HEALTHY checkout,
                          which is the worst failure a tripwire has — the fix then looks like
                          weakening it. Two files crossed that line (the explainer standards, and
                          10's notebook), so all four watched lists now drop tracked paths, the
                          way backup_local_only.py::collect already did
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
2026-09-04  shared-layer  #108 opened: _shared/page.css styles 101 classes and 29 appear on no
                          rendered page. NOTHING deleted — #101 nearly removed a live stylesheet on
                          exactly this evidence. Measured from a rendered DOM after driving every
                          input, which is what moved .filter-none and .rail-shut off the list
2026-09-04  ci            #124 opened: `npx --yes @mermaid-js/mermaid-cli` FETCHES the package on
                          first use and it bundles puppeteer, so on a cold runner the download ran
                          past the 180s the render test allows for a RENDER. Failed on three of
                          four consecutive branches and passed on the fourth — a flake that reds
                          pull requests that did not touch it, which teaches people to re-run a
                          gate rather than read it. The fetch moves to a step with its own budget
2026-09-04  ci            the preview-deploy predicate opened: every push deployed a preview,
                          whatever it touched, and about sixty pushes in a day rate-limited the
                          Vercel project for 24 HOURS -- so previews were unavailable for the pull
                          requests that had actually changed a page. PK caught it. `ignoreCommand`
                          now builds only when a deployed path changed: 6 of the last 8 commits on
                          main would not have deployed. THE OBVIOUS PATHSPEC IS SILENTLY WRONG --
                          `src/exercises/*/web` matches nothing, so the predicate would have
                          skipped EVERY deploy; `:(glob)` is what makes it mean what it looks like,
                          and I wrote the broken one first
2026-09-04  ci            #133 opened: every push deployed a Vercel preview, whatever it
                          touched, and about sixty in a day exhausted the quota and rate-limited
                          the project for 24 HOURS -- so previews were unavailable for the pull
                          requests that had actually changed a page. PK caught it. `ignoreCommand`
                          now builds only when a deployed path changed: 6 of the last 8 commits on
                          main would not have deployed. THE OBVIOUS PATHSPEC IS SILENTLY WRONG --
                          `src/exercises/*/web` matches nothing, so the predicate would have
                          skipped EVERY deploy, and `:(glob)` is what makes it mean what it looks
                          like. I wrote the broken one first and caught it only by running the
                          predicate over real history
2026-09-04  tooling       #134 merged: the sync tool anchored a re-applied block on the single
                          line that FOLLOWED it, and QUEUE.md has a dozen code fences -- so #108's
                          entry landed forty lines above the `## Log` heading. It was still in the
                          file, so every count of it looked right; only queue_status.py, which
                          reads the log section and nothing else, noticed, two merges later. The
                          anchor is a neighbour PAIR now, with reported fallbacks. Third defect
                          from that tool in three live runs, each invisible in the diff
2026-09-04  tooling       #135 opened: a pull request that merges without logging itself makes
                          main fail its OWN queue gate, and every branch cut from main then
                          inherits a failure that points nowhere near the cause.
                          Three merge rounds lost to it in one afternoon. The existing check looks
                          BACKWARDS -- does the log record what already merged -- so by the time it
                          fires the damage is on main. This one looks forwards, at the pull request
                          under test, so the branch that would cause the problem is the one that
                          goes red
2026-09-03  fleet         #103 merged: install_agent_fleet.py --drift, wired into the post-merge
                          hook. A reviewer copied into .claude/ and then edited there diverges
                          silently from its tracked source, and the installed copy is the one that
                          runs — so the drift is invisible in review by construction
2026-09-04  tooling       #125 opened: merging one pull request here BREAKS the next, measured
                          rather than assumed. Every open branch touches QUEUE.md, CHANGELOG.md and
                          the quote-check receipt, and the receipt is a digest over ALL tracked
                          prose — so merging two branches produced a QUEUE conflict AND left the
                          receipt full of conflict markers, killing its checker with a
                          JSONDecodeError instead of a clean failure. sync_open_prs.py merges main
                          into every open branch, rebuilds the two logs as main's version plus that
                          branch's own entry, regenerates the receipt and pushes — never rebasing,
                          never force-pushing. `merge=union` was refused: it keeps both sides, and
                          fifteen branches carry a byte-identical #103 line, so the fifteenth merge
                          would land fifteen copies
2026-09-04  retro-fix     #120 opened: `color: #fff` on a background that is bright in half the
                          themes, in three controls — the back pill on hover and the blocking
                          caveat (shared, so all six pages) and 04's toggle in its ON state.
                          1.54:1 on neon; --on-accent already existed and clears AA on all six. The
                          guard is lexical because two of the three are :hover and .on states that
                          no static render enters. It found the third site itself
2026-09-04  retro-fix     #122 opened: the shared step strip squeezed its prose to 29 characters
                          at 768px — an iPad portrait — because the two-column layout collapsed at
                          max-width 760 while keeping a FIXED 296px figure column. Also 41 at
                          exactly 1180px and nowhere else, that being where page.css starts
                          reserving the rail gutter. 90 of 360 and 28 of 112 step paragraphs under
                          the floor before, none after. THE PROMOTED GUARD COULD NOT SEE IT: 08's
                          ROOM_TO_SPARE asks whether a block leaves room inside its own box and a
                          squeezed paragraph fills its box exactly. Found by watching it fail
2026-09-04  retro-fix     #126 opened: exercise 03's print fix -- 0 of 45 scrolly verdict lines survived
                          `emulate_media("print")` because the end-state rule existed only for
                          prefers-reduced-motion. A printed sheet carried 11 figure states and lost
                          34. Fixed by adding `print` to the SAME media query rather than writing a
                          twin block — two blocks drift, and drift is what produced the defect
2026-09-04  retro-fix     #131 opened: the repo-wide link-colour fix. Reported as 7 anchors at the
                          browser default; the real count is TWO. Five compute #0000EE on the
                          anchor while every word sits in a child with its own colour, so that
                          colour paints nothing — the third container-not-text artefact in this
                          sweep, after .rail-link and .badge. The verifier had said the scope was
                          overstated 3.5x and 7/2 is exactly that. The guard's FIRST version was
                          blind: filtering to anchors with their own text excluded the very links
                          whose children inherit the unstyled colour, and it passed the break.
                          Found only by breaking a second page that had a real one
2026-09-04  retro-fix     #114 opened: exercise 01 declared role="tablist" with role="tab" children
                          and had no role="tabpanel", no aria-controls and no arrow-key handling.
                          A wrong announcement is worse than none, because the reader acts on it.
                          The fix REMOVES the claim rather than building the machinery: these
                          redraw a region in place and are not tabs
2026-09-04  retro-fix     #115 opened: exercise 02, same false tab-pattern claim as #114, same
                          answer -- the roles come off rather than the machinery going in
2026-09-04  retro-fix     #109 opened: exercise 04's contents rail never marked the section in
                          view. `.rail-link.on` has been styled in the shared stylesheet since
                          before the page existed and the page never set it, so the rail looked
                          finished and never moved. The rule is the last heading past the first
                          third of the viewport, NOT the nearest one -- sections run several
                          screens, so the nearest heading is often the one ahead of the reader
2026-09-04  retro-fix     #117 opened: exercise 04's widest body paragraph was 1,156px at 16px --
                          about 145 characters a line, roughly double what anyone can track. A
                          measure brings it to 692px, about 83. The `ch` unit goes on the element
                          that CARRIES the type, because it resolves against that element's own
                          font size
2026-09-04  retro-fix     #110 opened: exercise 05's contents rail never marked the section in
                          view. Same defect and same rule as #109
2026-09-04  retro-fix     #116 opened: exercise 05's toggle marked its active option by painting the
                          accent and NOTHING else. A screen reader read two identical buttons on a
                          control whose whole purpose is that the answer changes, and under
                          high-contrast the two fills are nearly the same, so a print or screenshot
                          loses the state outright. Now announced with aria-pressed in a labelled
                          group, and marked by something other than colour
2026-09-04  retro-fix     #118 opened: exercise 05, the same 1,156px at 145 characters. THE FIRST
                          ATTEMPT MOVED NOTHING: the rule selected `main section > .note` and none
                          of this page's long paragraphs are section children, so it matched
                          nothing and the measurement came back unchanged while the rule looked
                          exactly like a fix. Caught only by re-measuring AFTER the change. Matched
                          anywhere under main it lands at about 91
2026-09-04  retro-fix     #128 opened: exercise 05's three fixes -- 14 permalinks with no accessible name;
                          all 17 svg labels between 6.39 and 9.4px at 390 because a viewBox scales
                          its text with the drawing, so legibility is a property of the RENDER and
                          the authored 10px said nothing; and every reproduce command over 72
                          characters cut off at EVERY width, including 2560 where 676px sat empty
                          beside the box. All three guards watched failing: 14 of 14, 17 of 17,
                          2 of 2
2026-09-04  retro-fix     #111 opened: exercise 06's contents rail never marked the section in
                          view. Same defect and same rule as #109
2026-09-04  retro-fix     #112 opened: exercise 07's contents rail never marked the section in
                          view. Same defect and same rule as #109
2026-09-04  retro-fix     #129 opened: exercise 07's two fixes -- 25 table headers plus 2 legend chips at
                          4.15:1 in the DEFAULT theme and nowhere else — the five explicit themes
                          all cleared AA, so a two-theme check would have found nothing — and all
                          81 svg labels between 6.49 and 9.4px at 390px. Both watched failing:
                          25 of 25, 2 of 2
2026-09-04  retro-fix     #132 opened: exercise 08's cut-line fix, and TWO of its three findings were
                          deliberately NOT fixed. The cut line was clipped 87px at 320px with no
                          ellipsis, losing both its sentence and its dashed rule — the same element
                          AGENTS.md already records being truncated once, so the guard asserts
                          geometry rather than a string. Left alone: 178 svg labels below 9.5px
                          (DESIGN.md requires mono labels 9.5-11px FIXED, and holding that needs a
                          1240px plate scrolling 3.2x on a phone) and 29 of 30 timeline markers
                          under the 24px target size (they already touch, so enlarging hit areas
                          changes the timeline's density). Both are design decisions on the
                          reference implementation, not retro-fixes
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
2026-09-04  shared-layer  #107 opened: one theme picker, not eight — 106 lines of duplicated
                          logic removed. One of its four guards was BLIND and the deliberate break
                          is how I found out: it checked the string `bindThemePicker` appeared, so
                          deleting the CALL and leaving the import satisfied it
2026-09-04  guards        #136 opened: two guards scanned wider than they assert. The basename
                          check globbed the repo ROOT, so nine .claude/worktrees/ scratch copies
                          each counted as a second definition of every test file -- 122 invented
                          clashes pytest could never hit, since testpaths is ["src","tests"] and it
                          collects zero files there. Now reads those roots from pyproject rather
                          than restating them. The generator's naming check globbed [0-9][0-9]-* by
                          name instead of exercises_in, so 09/10 -- on disk holding only their
                          gitignored local-only files while the tracked half waits in an unmerged
                          branch -- read as exercises and died on an absent pyproject.toml. Both
                          were GREEN IN CI and red only locally, which is where the notebook,
                          tripwire and quote gates live and CI has no twin
2026-09-04  deploy        #137 opened: a preview was built for pull requests that change no page.
                          VERCEL_GIT_PREVIOUS_SHA is the last successful DEPLOYMENT, not the parent
                          -- the comment said otherwise -- so a sync that merged main in read as
                          deploy-worthy and built a copy of main's own site. A second gate now asks
                          whether the preview would show anything: identical deployed files to main
                          means skip. Two TREES, not two histories, so no common ancestor is needed
                          and it survives a shallow clone. MEASURING REFUTED MY FIRST ATTEMPT: a
                          plain diff against main would have built MORE, since 18 of 19 open pull
                          requests change a deployed file and would then rebuild on test-only
                          pushes. Sized honestly at ONE build per sync round, not the ~20 first
                          claimed; corrected in the changelog where the claim was made
2026-09-04  deploy        #138 opened: REVERTS #137. Its gate compared against origin/main, and
                          Vercel checks out a single-branch SHALLOW clone -- the first build after
                          it merged printed "origin/main could not be resolved". Inert in
                          production; unsound wherever the ref does resolve, because a branch that
                          reverts its page back to main's content skipped its build and left the
                          live preview serving a change the PR no longer made. 24 hermetic cases
                          passed over it: they proved the predicate self-consistent and said
                          nothing about whether its inputs exist in a real build. AGENTS.md now
                          requires reading the target environment's own log first. Also records,
                          unfixed, the pre-existing HEAD^ fallback: with no previous deployment a
                          branch whose TIP commit is not deployable gets no preview at all, self-
                          reinforcingly. Not fixed here -- verifying it needs a real build log,
                          which is the rule being added
2026-09-03  exercise-10   10 built: six items, all generated. Two figures were flattering
                          themselves — MFU divided CPU work by a GPU's peak (39.13%) and priced the
                          embedding tables, which are gathers. The honest figure is 27.69%
2026-09-03  review        three reviewers read 10. The central finding is a real bug in shipped
                          code: decompose() was correct at 0.1 and wrong on 3.7% of bf16 and 30% of
                          E4M3 inputs, returning values exactly twice too large. 0.1 was the whole
                          test. The cross-check now sweeps 2,000 values per format, which found a
                          second limit nobody had reasoned about — subnormals
2026-09-03  exercise-10   10's notebook is TRACKED, under a written exception in AGENTS.md and a
                          .gitignore negation. backup_local_only needs no change: collect already
                          drops tracked files. The tripwire did, because a clone holding one
                          notebook and not the others reads as a partial loss
2026-09-03  process       09 and 10 were split onto their own branches after PK pointed out the
                          convention: one branch and one pull request per exercise. #104 had three
                          concerns stacked on it
2026-09-04  exercise-10   #106 opened: one optimiser step, made to tell the truth about itself.
                          Stacked on feat/09-loss-harness rather than main, so #105 has to merge
                          first. Float decomposition, gradient accumulation that refuses an even
                          micro-batch count, MFU against a MEASURED device peak rather than a
                          datasheet one, and telemetry that requires the loss to follow a leading
                          indicator rather than merely precede it
2026-09-04  retro-fix     #121 opened: exercise 01's s3.html had been DEAD ON ARRIVAL for as long
                          as it has been deployed — `var t` in the theme bootstrap is a global and
                          the page script opens `let ..., t, ...`, so the whole thing threw before
                          its first statement. Every file-level check stayed green because every
                          file was well-formed. Three of the four proof pages also had no LIGHT
                          palette, so their tokens resolved to nothing on the default theme and a
                          chip's white label sat on the white page at 1.00:1
2026-09-04  retro-fix     #121 synced, and a defect was found IN ITS OWN SUBJECT. Each chip's
                          swatch is written into a style attribute from a value colOf() read once
                          at build time, while the label's ink stays a live var(--chip-ink). A
                          reader who picks a theme their OS does not have therefore got near-black
                          ink on the LIGHT swatches: 4.19 / 3.64 / 3.85:1 against 4.5. Now 6.53:1,
                          the figure the stylesheet's own comment already claimed. The CSS was
                          correct throughout -- only the JS froze half the pair. The six-theme
                          guard could not see it because it pairs every dark theme with a dark
                          prefers-color-scheme, the single arrangement in which a frozen colour
                          matches. A cross-scheme guard was added and watched failing while the
                          original stayed green
2026-09-04  tooling       #139 opened: snapshot() ended with `git add -A` INSIDE the backup
                          store -- the same command this repo forbids in the working tree. Three
                          files became tracked in #106, so collect() stopped gathering them and the
                          store's stale claim was reddening the tripwire on every branch that
                          predated them, blocking the sync tool on 17 open PRs. PK removed them
                          with `rm --cached`, which leaves the files on disk; the next snapshot
                          re-added all three, from the post-checkout hook, seconds later, silently.
                          Now stages exactly what it copied. That exposed the commit gate reading
                          `status --porcelain`, which counts UNTRACKED files -- impossible under
                          -A -- so the store announced "NOT safe" over a healthy snapshot. Gated on
                          staged changes now. Both watched failing; the 19 agent_sample_files
                          entries are PK's staging area and deliberately untouched
2026-09-04  tooling       #140 opened: two guards refused SEVENTEEN open PRs on a checkout
                          where nothing was missing. docs/EXPLAINER_*.md and S10's notebook became
                          tracked in #106, so an older branch legitimately lacks them -- and they
                          are gitignored there, so _tracked_paths() cannot see them and the class
                          reads as partly present, the shape of a real loss. The tripwire now asks
                          whether GIT CAN GIVE IT BACK (origin/main too), not whether this commit
                          tracks it. sync_open_prs.py was the other half: it read a non-zero exit
                          from `git checkout` as a failed checkout, but a post-checkout hook cannot
                          abort one -- it judges by where HEAD landed now. Fixing only the tripwire
                          would not help, because the hook runs the OLD copy that lives on the
                          branch being checked out. Still watching 30 local-only files; exactly the
                          three git holds are excluded
2026-09-04  retro-fix     #130 opened: exercise 02's contrast fix -- 5 unselected segmented options at
                          4.15:1 in the DEFAULT theme only. A reported sixth failure was the PROBE,
                          not the page — a badge whose ground is 10% alpha of its own text colour
                          read 1.00:1 until the stack was composited, and is 4.66:1 in fact. The
                          guard blends the painted stack so it cannot repeat that
2026-09-04  retro-fix     #127 opened: exercise 04's permalink fix -- 13 chapter anchors read "#" with no
                          aria-label, so a screen reader announced "number sign" thirteen times for
                          thirteen destinations. 03 and 06 already carried the answer; the wording
                          is copied rather than reinvented, because an identical control should
                          announce identically. The guard asserts a non-empty accessible NAME, not
                          that a particular string is present
2026-09-04  retro-fix     #119 opened: every deployable page is now checked in all six themes, not
                          just 08 — no console error, tokens resolve, body text clears AA, with
                          contrast computed in the browser rather than parsed from CSS. 8 pages x 6
                          themes. Both halves watched failing: one paints the body text its own
                          background, one disables the root token sheet in the live page
2026-09-04  retro-fix     #113 opened: a page that builds a contents rail must mark the section in
                          view. It was RED on purpose while its four subjects sat in other pull
                          requests -- silencing a gate that reports a true defect is what AGENTS.md
                          refuses -- so #109 to #112 were merged INTO it instead, which makes it
                          green honestly and shrinks it to the guard alone once they land. The four
                          merges each conflicted on CHANGELOG.md and resolving by taking a side lost
                          three of the four entries; restored, and it is exactly the failure
                          sync_open_prs.py exists to prevent
2026-09-05  tooling       #146 opened: four guards that were wrong about their own subject,
                          consolidated from #141-#144 after PK asked why one-file pull requests
                          were being opened separately. They were right to: three of the four
                          carried ONE real file each and paid three files of bookkeeping apiece.
                          The sync tool stranded the entries it placed (five times); the MFU guard
                          measured at size=512, which is not a peak, and reddened whichever branch
                          was open; nothing compared the eight vendored copies of web/_shared,
                          which went stale twice; and AGENTS.md gains the rules all three taught.
                          One concern at the right granularity, ~5 real files, inside the 20-file
                          ceiling that says what scale was intended
2026-09-05  refactor      #145 opened: web/_shared/tokens.css was NOT the token file and is now
                          components.css. Every page linked two stylesheets under one name -- the
                          six-theme tokens at /_shared/tokens.css and exercise 03's component
                          styles at ./_shared/tokens.css -- told apart by one character, which is
                          what made the rename mechanical. 8 files renamed, 11 pages relinked
                          (three of them sub-pages a per-exercise sweep would miss), 16 absolute
                          links untouched. Verified in a browser: 10/10 pages load both sheets with
                          every token resolved. The first probe said 0/10 and was wrong -- the
                          build appends a cache-busting query -- which is the third artefact of
                          that shape in this queue
2026-09-05  docs          #147 opened: a blank line had cut the root README's exercise table in
                          two since 023bbd5, so exercises 09 and 10 fell OUTSIDE it and GitHub
                          rendered them as a paragraph of literal pipe characters -- on the front
                          door a grader lands on, for the exercise being submitted. The guard
                          could not see it because it reads a ROW (a line starting `| 09 `) rather
                          than the table; the new one walks from the header, stops where the table
                          stops, and was watched reporting rows 01-08 against ten exercises. Seven
                          documents that called finished work unfinished were corrected with it,
                          including a SECOND orphaned status table in 09's PROGRESS.md and a test
                          count of 44 where the suite collects 54. WORKPLAN.md is now a record
                          rather than a tracker -- every stage complete, pointing at this file --
                          and stays tracked, because untracking is what has destroyed files here
                          three times
2026-09-05  exercise-10   #148 opened: the ONE topic notebook this repo tracks ran on exactly one
                          computer. Its builder interpolated EXERCISE.parents[2] -- the BUILD
                          machine's absolute path -- into the setup cell, so every other machine
                          got a path that does not exist. It also imported matplotlib, which was
                          in no pyproject, absent from uv.lock and absent from the venv; Colab
                          pre-installs it, which is why this was invisible here and certain for
                          anyone else. Root is now found at run time (verified from four cwds),
                          matplotlib joins the dev group beside ipykernel/nbclient, and
                          test_tracked_notebooks_are_portable.py reads the TRACKED notebooks --
                          which a clone has, unlike the builders. Watched failing against the
                          notebook as it stood before the fix. Rebuilt AND executed end to end,
                          26 cells, before being written to its tracked path
2026-09-05  release       #149 opened: v0.14.0 -- [Unreleased] to [0.14.0], 136 entries across 31
                          blocks since v0.13.0. Verified with release.yml's OWN awk extractor
                          rather than by reading the heading, because a heading that does not match
                          the tag makes it publish with --generate-notes and silently replace all
                          of it with commit subjects: have_notes=TRUE, 1,587 lines. Also pins the
                          root pyproject version at 0.0.0 -- it read 0.4.0 while the newest release
                          was v0.13.0, nine minors stale and wrong for eleven releases. Bumping it
                          would have been the worse fix: a number correct once and then rotting
                          reads as maintained. test_root_version_is_pinned.py holds it and was
                          watched failing both ways. PREREQUISITE, PK's: delete the stray v1.0.0
                          and v2.0.0 tags, or snapshot_standards keeps reporting 0/8 at v2.0.0
                          even after the real tag lands
2026-09-05  exercise-02   #150 opened: exercise 02 redistributes CC BY-SA Wikipedia text -- 14
                          tracked corpus files, 1.6M characters, plus a tokenizer derived from it
                          that the deployed page offers as a DOWNLOAD -- and had no NOTICE. The
                          licence appeared in exactly one place in the exercise: a parenthetical in
                          CLAUDE.md, a file addressed to coding agents. That is not attribution and
                          not discoverable by a reader. NOTICE now names each article, its source
                          URL, fetch date and character count, plus the derived artefacts that
                          inherit the licence; CLAUDE.md points at it instead of carrying it.
                          test_tokenization_notice.py fails in both directions -- a language in
                          corpus/v2 and not in NOTICE is a gap, and every character count is read
                          back from its meta.json rather than typed
2026-09-05  exercises     #151 opened: exercises 01-04 get the progress ledger they never had, and
                          PROGRESS.md joins test_exercise_skeleton.py's REQUIRED -- the rule now
                          describes the repository rather than inventing a convention for it.
                          DECISIONS.md and NOTICE stay out on evidence: five of six exercises
                          lacking a decision record would gain a SECOND copy of reasoning they
                          already publish, and 01 ships no third-party content so a required NOTICE
                          would attribute nothing. Each ledger ends with what the record does NOT
                          settle, because a plausible stage that never happened is worse than an
                          admitted blank. A twelve-agent pass verified the reconstruction first and
                          caught a rename recorded as a deletion, diffstat arithmetic that reached
                          the right total by cancelling two errors, a tag list wrong by ten, and a
                          notebook stage citing a file never committed on any ref. Two of MY OWN
                          published claims were refuted by it and corrected in a follow-up commit:
                          a commit subject is not evidence of what a commit did
2026-09-05  fleet         #152 opened: the agent guard could not see a shell command, read mv as
                          a copy, and guarded the backup TOOL rather than the files it protects.
                          Bash was absent from the PreToolUse matcher so decide()'s whole Bash
                          branch was unreachable -- proven live: a merged PR changed uv.lock via
                          `uv sync` and nothing fired. mv sat beside cp, but cp leaves its source
                          and mv destroys it, so `mv uv.lock /tmp/backup` read as a copy and was
                          allowed. notebooks/ and the builders were in no section at all. And the
                          section list was hardcoded, so the new [irreplaceable] section would have
                          enforced nothing while reading as protection. Five reverts, five reds.
                          PK created the UNIT.md scope this needed; writing that permission slip
                          unprompted would have been the incident [guards] exists for. NOT DONE:
                          .claude/settings.local.json still carries the old matcher, so the hole is
                          closed in the repo and open on this machine until the installer re-runs
2026-09-07  fleet         #153 opened: #152 froze the two working notes it was meant to protect.
                          TODO.md and HANDOFF.md went into [irreplaceable], which is deliberately
                          NO_ESCAPE_HATCH -- right for a notebook, whose legitimate rewrite goes
                          through a builder the guard never sees, and wrong for a note whose whole
                          function is being rewritten. It surfaced within the hour, refusing a
                          rewrite of HANDOFF.md PK had asked for. New [working_notes] section that
                          DOES honour the hatch: refused by default, allowed when a unit names it,
                          same bar as editing a guard. notebooks/ and the builders stay no-hatch and
                          a twin proves naming one does not unlock it. The section needed no code
                          change to take effect -- _pattern_sections derives the enforced list,
                          which is #152's own fix paying for itself
2026-09-07  exercise-07   #156 opened: exercise 07's trained arms came from code held outside
                          the repo and now gone, so its central claim was recorded rather than
                          executable. experiment.py + summary.py + tools/run_experiment.py make it
                          runnable: 10 arms, 5 paired seeds, 25 min on a laptop CPU. NOT a
                          reproduction, and that is a property of the record: setup pins the
                          architecture and none of the optimisation -- 13 free parameters against
                          one recorded scalar -- so an experiment aimed at those losses could not
                          be told from one that missed. RunConfig records every knob it turns.
                          RESULT: 9 of 10 arms agree in SIGN with the recorded table. Both n-gram
                          arms supported 5/5 -- wrap+ngram -0.259 vs v1 (record -0.164), tied+ngram
                          -0.232 (record -0.141). Both recorded negatives stay negative; the MLP
                          buys 0.022 against the n-gram's 0.291, which is the finding. The one
                          disagreement is `wrapped positions`, -0.017 here vs +0.248 recorded, and
                          it is below the stated resolution so the summary says inconclusive rather
                          than claiming a reversal. NOTHING PUBLISHED: the runner writes to
                          artifacts/, never results/; no measurement, page or document number moved.
                          09 supplies the body via a new optional embedding= on build_trunk rather
                          than 07 reaching for private trunk.tokens -- 10 already couples to that
                          name and one such coupling is enough. Default path proven bit-identical:
                          SHA-256 over every named parameter matches main. Three traps that give a
                          plausible wrong answer rather than a failure: the dense control at torch's
                          N(0,1) default starts at loss 176 and cripples the baseline; the tie must
                          be ONE object or it silently is not one; the corpus must be the
                          multilingual v2 or the effect is invisible. Fourier is also 23x denser
                          per token and 8x slower to train, which no document here records

2026-09-07  exercise-07   #155 opened: the page reads every number from M so a figure cannot
                          drift from the run that produced it -- except one, typed into the
                          glossary's definition of nats: "beats the published design by 0.141". A
                          measurement that moved would have updated every table and left that
                          sentence contradicting them, inside the prose that explains the unit.
                          Derived from M.attribution now; GLOSSARY became glossaryEntries(M) so it
                          cannot regress. Rendered page proven unchanged: driven in a browser on
                          this branch and on main, body text byte-identical at 19,856 chars.
                          SEPARATELY: nothing checked that web/data.js still matched the
                          measurements it is generated from -- grep of tests/, workflows and deploy/
                          for either filename returned ZERO hits, so editing the JSON and forgetting
                          build_web_data.py served the previous run's numbers with a green suite.
                          New test_embeddings_page_data.py parses both and asserts equal, plus a
                          banner check and a broken twin. Watched red on a one-digit change,
                          restored in a finally, byte-identical after. Pure Python, so it runs in
                          the plain test job rather than behind playwright like the render suite

2026-09-07  exercise-07   #154 opened: S07's notebook taught the analysis and never showed the
                          mechanism. It printed the sparse coordinates but never E = K.W_p, never
                          the tie that IS v2, and never the evidence the recommendation rests on --
                          PK could not answer a question from it, which was the notebook's fault.
                          Rebuilt around input -> process -> output: one real token from text to
                          loss, shape printed at every stage, so a reader watches an embedding be
                          BUILT rather than looked up. Every knob is a named constant with a table
                          saying what it should move and what it must not. Six of the eight
                          withdrawn claims now re-run as live cells -- worth doing because the four
                          documents recording them say three, three, five and six, and a count that
                          is produced cannot drift like one that is typed. The headline statistic is
                          re-derived from the ten raw per-seed losses the record ships: unpaired
                          spread, paired sd, sign test and both arm means all reconstructed and
                          printed beside the recorded values. Section 5 does NOT pretend to
                          reproduce the trained arms -- that code is not in this repo and the setup
                          pins none of what decides a loss -- and says so where a reader meets it.
                          TWO REAL ERRORS FOUND BY READING THE OUTPUT, NOT BY A TEST: the scale cell
                          compared against nn.Embedding's N(0,1) default, whose row norm happens to
                          equal the induced norm, so it reported 1.0x where the effect is 49.5x and
                          hid the entire point; and the recovery cell sampled the first 300 tokens,
                          all short, so its two metrics agreed and demonstrated nothing. Also
                          corrected a claim of mine: 94.67% is absent from measurements.json but IS
                          re-derivable as the vocabulary's fit rate at d_p=32 (9,467/10,000), so it
                          is under-recorded rather than unsupported. lite 8s, full 45s, both
                          executed end to end with nbclient. Nothing tracked changed: no
                          measurement, no page, no package code, so nothing to redeploy
2026-09-08  exercise-07   #156 amended before merge: its changelog headline read "and the
                          published conclusion survives it". That validation rested on a run whose
                          corpus was later found to be 28.6% [UNK] -- exercise 02's tokenizer has no
                          Tamil and 07 was feeding it Tamil. The repo's own gate (04's
                          MAX_UNK_SHARE, reused by 05 and 06) refuses above 5%, and 07 is the only
                          exercise that never measured it. The confound lands on the winning arm:
                          [UNK]'s spelling is fixed, so its hashed byte n-grams are identical every
                          time, and the n-gram arm is the one that wins. Claim withdrawn before
                          merge rather than corrected after -- a quietly amended number is worse
                          than the original error. The machinery claim stands: experiment.py,
                          provenance, the recovered per-seed arrays and the notebook are all sound
                          and PR A depends on them. Invertibility, collisions and the parameter
                          arithmetic are properties of the vocabulary and are unaffected
2026-09-08  exercise-07   #157 opened: the comparison can be checked now, and its conclusion did
                          not survive. Corpus swapped to exercise 06's six licence-manifested lanes
                          (11,781,888 tokens, 0.209% [UNK]); three gates that refuse rather than
                          warn, each watched refusing something real; proportional per-lane
                          sampling, without which 256,000 positions off the front of an 11.8M-token
                          corpus would have read ONE lane of six with every loss curve normal;
                          numbered run directories; and two auditors that import nothing from the
                          package they audit, evidence.py's tests written first because 06's
                          equivalent has none. THE RESULT: the recommendation beats v1 on exercise
                          02's corpus (-0.196 with the unreadable language, -0.551 without) and
                          loses on everything else tried -- +0.158 on the six-lane mixture at 500
                          steps, +0.201 on indic alone, +0.150 on code alone, and a 3/5-seed
                          -0.005 on web that is noise. CORRECTION to the entry above: "the confound
                          lands on the winning arm" was reasoning, not a measurement, and
                          measure_unk_confound.py refutes it -- removing the unreadable language
                          makes the recommendation win by MORE, and every arm's gap grows 2-2.5x in
                          whichever direction it pointed, so [UNK] was a dilution rather than a
                          selective advantage. Three candidate causes tested and refuted ([UNK],
                          step count, script mix); one named and untested -- exercise 02's five
                          files are the same Wikipedia article in five languages, so that corpus is
                          parallel text. Nothing published: results/ untouched, the live page still
                          renders the old numbers. Guards found four defects on their own,
                          including a trace rounded to six decimals that could not re-derive the
                          mean it is the material for. CPU is bit-identical; MPS differs by one
                          float32 ULP (9.537e-07), so the publishable grid is the CPU one
2026-09-09  exercise-07   #157 merged; #158 opened: the documents say what the evidence says.
                          The results narrative was written when the recommendation won, and on a
                          second corpus it loses -- README headline, arm table and NOTICE all
                          corrected to state the finding AND its limit, with both runs published
                          side by side. DECISIONS.md records eleven decisions including the one
                          where our own reason for the corpus fix was refuted by measuring it. PK's
                          two instructions done: no shell commands on the page (the reproduce
                          section argues instead), every command in the README's Run it, and a
                          guard for both. A DECODER DEFECT found while sourcing a claim:
                          decode.recover accepted wrap and could not decode it -- its matched
                          filter argmaxes over unsigned atoms while half the wrap slots carry -1 --
                          scoring 47% on tokens four documents call perfectly recovered, with no
                          test ever driving it that way. Sign-aware it is 100.00%. Two figures the
                          page stated (14.6% / 19.1%) were in no evidence file; the shipped scheme
                          is now measured whole-vocabulary (100.00 / 15.05 / 0.00 by band) and the
                          removed permutation variant is reported UNREPRODUCED after two rebuild
                          attempts produced harness artefacts rather than results. Three guards
                          were wrong about their own subject and are fixed: a hand-rolled JS string
                          parser that desynchronised on an apostrophe and stopped seeing the file,
                          a regex test-counter that would have accepted the stale number it exists
                          to catch, and a hardcoded tool list that called a documented script a
                          deleted module. 1,914 passed, 2 skipped
2026-09-09  exercise-07   #159 opened, and it is a THIRD pull request against a two-PR plan, so
                          the reason is recorded rather than assumed: it is a correctness fix in
                          shipped code, found after #158 was finalised and green, by an agent
                          auditing the codec for the unbounded-length research rather than by the
                          documentation work. It reverts independently. codec.atoms merges
                          duplicate (slot, byte) pairs -- only possible under wrap, where two
                          folded positions share a slot and a byte -- and encode recorded the
                          merged non-zero count where the 1/sqrt(L) scale needed the position
                          count, returning a target multiplied by sqrt(nnz/L). 142 of 10,000
                          tokens, worst case 14.07%. NO published number moves: every published
                          recovery figure was measured under onehot, which cannot merge (positive
                          control: zero), and in the one wrap band where merging is common 105 of
                          465 tokens were mis-scaled while the band reads 15.05% before and after,
                          because those tokens were failing anyway. Latent, and it would have
                          bitten the moment wrapped invertibility mattered -- which is what the
                          research is about. Two guards, both watched going red; the second
                          compares the whole round trip against a hand-built target and would have
                          caught it without knowing the word "merge". 1,908 passed, 2 skipped
2026-09-09  exercise-07   #160 opened, stacked on #158: RESEARCH.md, the three researched
                          problems written so several kinds of reader can use them. PK asked that
                          every experiment carry its literature, hypothesis, approach, rationale,
                          expected result, outcome and configuration in plain language -- the third
                          time he has asked for something to be explained plainly, which is a
                          signal about the writing rather than the reading. EVERY CLAIM CARRIES A
                          MARK: measured here, re-derived by hand, reported and unverified, or an
                          argument. The reason is that one research pass corrected itself twice and
                          named two papers that DO NOT EXIST, so a reader must be able to tell a
                          measurement from a lead without asking. Findings: problem 1 (arithmetic
                          in the embedding) dead as stated -- 0.5% of four-digit integers are
                          single tokens, and z-norm saturates a value coordinate at 90.5152 while
                          inverting the word's own letters above v=4,730 -- with a right-aligned
                          place-value survivor and a scramble control that would settle it; problem
                          2 (images and audio) needs a compression step this repo lacks, since a
                          768-byte patch would need ~9,216 dimensions to reverse; problem 3 (no
                          length limit) is the one to build -- 99.74% whole-vocabulary recovery
                          against today's 94.67% at the SAME code width, provable without training,
                          and it explains problem 4's failure as a side effect (its neighbouring
                          positions point 96% the same way, re-derived by hand). Six guards, each
                          watched going red; one of them counted what it checked after the first
                          version turned out to pass for every possible document. Two agent numbers
                          corrected while writing: 58.3x vs 111.6x is which block you count, and
                          both are true of different things. 1,923 passed, 2 skipped
2026-09-09  repo          #161 opened: a checklist is reconciled against the repository rather
                          than re-read. Three stale artefacts in one afternoon and they are one
                          failure -- a hand-maintained list duplicating something the repository
                          already knows. tools/check_todo.py compares an annotated item with what
                          the tree holds (exists / present / absent), reports BOTH directions, and
                          reports an item with no predicate as UNVERIFIABLE rather than as fine.
                          Its first real run found four entries done and still open, two of them
                          exercises described as empty directories that have been built and merged.
                          The parser had to be built for the file that exists: markers inside
                          backticks, [~] and [!], and seven items on one line of which only the
                          first follows a dash -- the first-marker-only version did not report the
                          other six as unverifiable, it did not see them at all. Anchoring on the
                          bullet is what excludes the status legend with no exemption list. One
                          defect found by using it: an escaped needle made the regex ask whether a
                          string nothing contains was absent -- true of every file -- so an
                          unfinished item read as DONE; fixed with a single-quoted alternative and
                          no escape character at all. FEEDBACK, NOT ENFORCEMENT, and the module
                          says so: TODO.md is gitignored, so only the tool and its 23 tests reach
                          CI. The rule is added to AGENTS.md beside the derived-prose rule it
                          extends
2026-09-09  exercise-07   #162 opened: a position scheme that reaches past d_p without widening
                          the code. onehot discards every byte past d_p and wrap folds them onto
                          slots they then share, so neither can return a token longer than 32
                          bytes -- a limit of the CODE, not of any decoder -- and the exercise's
                          only answer was to raise d_p to 128, which quadruples D. spc gives each
                          position a DIRECTION in one shared d_p-dimensional space instead of its
                          own 256-slot block: unlimited reach at D unchanged, nothing folded,
                          position still identifiable at decode time. Whole vocabulary, whole
                          token: 99.35% at 33-64 bytes and 83.82% at 65-128 where onehot and wrap
                          both read 0.00%, and wherever it misses the truth fits strictly better
                          than the answer returned, so the information survived and only the
                          search was too weak. THE FIRST COMPARISON WAS UNFAIR and this is the
                          reason the tool exists: onehot reads 100% at 49-64 bytes if you ask it
                          about the bytes it KEEPS, spc was being asked about the whole token, and
                          the table said onehot was doing well at a length where it cannot
                          represent the token at all. Cost stated before measuring (Welch bound
                          0.1537 against a repelled 0.2465) and then measured: 189.4 non-zeros per
                          token against onehot's 8.2, which is 23x -- the same factor fourier pays
                          for training runs 8x slower. NEVER TRAINED, and five documents say so
                          rather than leaving a reader to assume. A DEFECT SHIPPED IN THE FIRST
                          DRAFT: the frame was sized to the token and to the batch's longest
                          token, so the same token encoded alone and beside a long one used
                          different directions, with nothing failing; reach is a config field now
                          and a token past it is refused. And the README's byte-recovery numbers
                          are checked against the evidence files for the first time -- red on its
                          first real run, because 94.67%, published as the vocabulary-wide
                          recovery rate at d_p=32, is in no evidence file
2026-09-09  repo          #163 opened: two tools that let through exactly what they exist to
                          prevent. Both answered their question correctly, about a case they
                          never saw. THE PreToolUse GUARD COULD NOT SEE A DESTRUCTIVE GIT
                          COMMAND: every rule in the policy matches a path and
                          bash_write_targets finds paths, so `git clean -fdx` -- which deletes
                          every gitignored file here, meaning every notebook, every builder and
                          every requirements document -- produced an EMPTY target list and a clean
                          pass. The [irreplaceable] section was working; it was never consulted. A
                          tracked [destructive_git] section refuses six shapes BY THEIR FLAG, per
                          shell segment so bundling cannot hide one: clean -x/-X, stash -a,
                          reset --hard, push --force, tag -d, branch -D. The pairs are the point --
                          clean -fd, stash -u and branch -d stay allowed, because the flag is the
                          whole distinction and a guard on the command name would block the safe
                          half and be uninstalled by lunchtime. Clustered short flags read letter
                          by letter, since -fdx is what anyone types. Two limits written into the
                          policy: a tag checkout cannot be told from a branch checkout without
                          asking git, and a shell can build the flag at runtime. SYNC_OPEN_PRS
                          REPLAYED AN EDIT AS AN ADDITION: difflib reports an in-place edit as a
                          replace, the tool collected only the "in" half and only ever inserted, so
                          a reworded line landed beside main's original and both shipped -- nothing
                          failed, because the entry was present and every count of it was right.
                          Edits replay as edits now; where main has since changed those lines the
                          replacement is added and the run REPORTS a possible duplicate rather than
                          deleting a fuzzy match and losing someone's work. Its tests carried their
                          own copy of the diff they tested; both call one changed_blocks now. Six
                          guards, each watched going red against the real defect
2026-09-09  exercise-07   #164 opened: the evidence is graded and guarded, including the parts
                          that were not. THE AUDITOR GRADED A SUBSET AND SAID SO NOWHERE:
                          evidence.py read results/measurements.json and nothing else, so both
                          published byte-recovery tables were graded by NOTHING and a reader
                          running it saw no row for either -- worse than an ungraded claim, because
                          the bundle reads as complete. assess takes every tracked bundle now, read
                          from the filesystem rather than a list beside the files, and two claims
                          are graded from them; an absent bundle grades UNVERIFIABLE, never met.
                          THE COHERENCE TABLE was four rows of hand-typed decimals matching the
                          evidence by nothing but somebody's care -- the block guard could not see
                          it because it matched percentages and these are bare decimals, the same
                          failure in a different notation. It licenses any decimal now and is
                          renamed evidence-numbers. AND I INTRODUCED A BUG IN THIS CHANGE, so its
                          guard ships in it: grading branches referenced claims by list index,
                          inserting two claims re-pointed the last branch at a different claim, and
                          it still ran, still printed a status, and graded the wrong sentence --
                          one claim's verdict under another's id and a third with no row at all.
                          Nothing failed; found by reading the output. References are by id now and
                          a guard asserts every claim is graded exactly once, both directions. Six
                          guards, each watched going red
2026-09-09  exercise-09   #165 opened: the page argues a case, and every number can be
                          regenerated. THE CORPUS WAS A MOVING FILE: training.py read the
                          repository's own AGENTS.md at run time and recorded a 16-character
                          digest prefix that nothing recomputed, so every published loss was a
                          function of a file edited on most pull requests -- and it had already
                          drifted, 92,021 bytes measured against 103,347 live. The revision is
                          frozen in corpus/ now, recovered from history, and re-running against it
                          reproduced every published training figure byte for byte, which is what
                          proves the right revision was frozen. harness.json and sensitivity.json
                          carried NO provenance at all; all three carry six fields now and every
                          writer refuses without them. THREE PUBLISHED NUMBERS WERE WRONG: 37
                          boundary crossings where there is 1 (the harness computed the right
                          number on one line and returned the mask's total drop on the next, which
                          inverted the finding -- one position at 9.51 against a 9.34 mean is the
                          point), 1.9x for chunking a softmax measured by nothing (now 1.80x, from
                          a third measured path, in the row about quoting one technique's figure
                          for another), and a README noise floor of 0.69 against 0.44. THE PAGE
                          opens by asking which of two unlabelled curves you would ship and only
                          then labels them; twelve sections were named for their spine role in
                          English and are named for objects now; the conclusion closed one of four
                          tiles and closes all four; reproduce lost its four shell commands and
                          shows what makes a figure checkable instead. THE NOTEBOOK went from 24
                          cells / 10 code / 0 charts / 0 asserts to 45 / 21 / 8 plots / 11
                          asserts, hands the reader the bug rather than describing it, and sweeps
                          the seed the page called the one thing it never varied. TWO DEFECTS
                          FOUND BY LOOKING with the suite green: the page threw and five of twelve
                          sections never rendered, and the corrections table was 3,594px wide in a
                          994px container. Both guarded, every guard watched failing
2026-09-09  exercise-10   #166 opened: the numbers say where they came from, and three of them
                          were wrong. THREE DOCUMENTS QUOTED THREE DIFFERENT MFU FIGURES -- README
                          27.69%, PROGRESS and CLAUDE 27.64%, against a recorded 27.74%. No single
                          document was obviously wrong; the SET was, and nothing was looking at the
                          set. The guard allowed a full POINT of drift on the ground that MFU's
                          denominator is a wall clock -- true of two runs, irrelevant to a document
                          compared against the one tracked file it renders. Exact now, plus a
                          cross-document guard and a ledger for the historical 39.13% the README
                          narrates on purpose, with a twin that fails if the narration is deleted
                          and the exemption left behind. A RATE THAT LIVED IN A DOCSTRING: the lead
                          tile published "30% of fp8 inputs came back exactly twice too large" from
                          a figure in no result file and recomputed by nothing. It is measured now
                          -- the shipped code replayed against the current one over 200,000 draws
                          in [1,2), the significand's whole space -- and it splits the figure the
                          old one merged: 25.04% doubled, 6.16% raised. THE LEDGER: run.json
                          carried no provenance at all, and the corpus was a sentence because this
                          module imported exercise 09's PRIVATE _corpus rather than its public
                          corpus_facts -- one import choice cost the exercise its whole provenance
                          on the data side. MFU's two halves are bound to one RUN_DEVICE now. THE
                          PAGE claimed a data.js regeneration test twice and it did not exist; the
                          test exists now, which was cheaper than deleting the claim. TWO DEFECTS
                          FOUND BY DRIVING IT with the suite green: hoisting a list above its
                          section took a const ul belonging to a DIFFERENT function, so the page
                          threw and reproduce never rendered; and facts.corpus became a block, so a
                          cell printed [object Object]. RIDING ALONG because they were made after
                          #165 merged: exercise 09's two Figure 1s, and should-build.sh falling
                          back to HEAD^ when VERCEL_GIT_PREVIOUS_SHA is empty -- which meant
                          exercise 09 pushed twice and deployed neither time, self-reinforcingly,
                          and the reviewer opened a cancelled deployment. Verified against Vercel's
                          own record either side of the fix. Also: the reviewer agents' own
                          definitions were in no backup pattern and never have been
2026-09-09  exercise-09   #167 opened: the page uses the display, and the fix was already
                          written down. PK said the text looked squeezed with half the page empty.
                          Four reviewers ran against the deployed page and the answer was that
                          docs/DESIGN.md PUBLISHES the fluid type scale and names THIS page's own
                          declaration -- `.say { font-size: 16px; max-width: 68ch }` -- as the
                          canonical example of getting it wrong. 08 has run the scale since it was
                          written; 04, 07, 09 and 10 never adopted it. 09 is on it now: reading
                          column 685 -> 951px carrying the SAME words per line, air beside it 531
                          -> 265px, pixel-identical to the reference. THE LEVER IS SIZE, NOT
                          MEASURE -- widening the column at 16px would have pushed the line past a
                          hundred characters, which is what the obvious reading of the complaint
                          would have done. NO TEST COULD HAVE CAUGHT IT and none still can:
                          test_prose_measure_repo_wide computes chars as width / ch-width, so an
                          element capped at Nch on ITSELF returns exactly N at every size and
                          viewport. It read 09 as inside its 42-80 band before and after. Tracked
                          as its own follow-up. THE RAIL was fixed at left:0 against a centred
                          1500px wrap, so the gap to the text grew to 554px at 2560 while the 260px
                          gutter stayed reserved -- the page paid for the rail twice and the rail
                          sat in neither space. It travels with the wrap now, 24px at every width,
                          and the reading column does not move at all; AGENTS.md records an earlier
                          attempt that moved the rail INWARD and destroyed the symmetry, and the
                          measurements either side are what tell the two apart. TWO MORE FOUND BY
                          LOOKING: .lede sits outside #main and is sized in rem, so the scale never
                          reached it and the page's thesis rendered smaller than its own captions;
                          and two selectors had lost their block and fused onto the next rule, my
                          own doing earlier today. The new guard keys on the BLANK LINE between
                          selector and rule, because a fused selector IS followed by a selector.
                          THE REVIEWERS CORRECTED MY OWN WORKING TWICE: 08 is 22px/951px with prose
                          centred 483/483, not 24px/762px flush left -- my script had picked its
                          standfirst; and 09's line was 70ch all along, not 84 -- my probe measured
                          lowercase advance. Widening the wrap to 08's 2200px was measured and
                          REJECTED: it grows the void beside the prose from 819 to 989px, because
                          09's tables are two and three columns and its figures are drawn at 700px
2026-09-09  design        #168 opened: 07 and 10 join the type scale, and the measure guard gets
                          the half it was missing. Both carried the identical `.say { max-width:
                          68ch }` at an inherited 16px -- the declaration docs/DESIGN.md names as
                          the canonical example of getting this wrong. Body prose 16px/685px ->
                          22px/951px at 2560, same words per line, with the rail travelling with
                          the centred column and the standfirst on the ramp, as 09 was fixed in
                          #167. THE GUARD COULD NOT HAVE CAUGHT ANY OF IT: it computes characters
                          as width / ch-width, and `ch` IS the advance of `0` at the element's own
                          size, so an element capped at Nch on ITSELF reports exactly N at every
                          font size and viewport -- 09 read 68 at 16px/685px and 70 at 22px/951px,
                          inside the 42-80 band both times. Checked the alternative: measured in
                          REAL characters every page in the repository exceeds 80 somewhere,
                          including 08, so the band is internally consistent and `ch` is the unit
                          the caps are written in. It simply cannot see physical size. The new half
                          measures that instead and its ledger fails in BOTH directions -- a page
                          regressing off the scale, and a page adopting it without being recorded
                          -- both watched failing. It also replaces two skips with an assertion
                          rather than declaring them, because a skip reports as a pass. AND A LONG
                          IDENTIFIER PUSHED THE PAGE SIDEWAYS: inline code holds paths and mono
                          does not hyphenate, so at the scale's 19px floor one <code> made 07's
                          document scroll by 19px at 320px. Caught by the existing guard at the
                          width nobody develops at. 03, 04, 05 and 06 are still on the old scale
                          and each needs measuring first rather than a bulk edit
2026-09-09  exercise-09   #169 opened: the page keeps the rules it states. Six things it claimed
                          about itself that were not true, found by four reviewers reading the
                          DEPLOYED page after #165 merged, every one green in CI throughout. IT
                          STATED ITS OWN PRECISION RULE AND BROKE IT FOUR TIMES: the results
                          section says the memory ratio is quoted "and no finer" than its noise
                          floor allows, and the tile said 9.1x, the glossary 9.1x, the ledger
                          9.09x and the conclusion 9.09x -- one of them fourteen lines above the
                          rule. Each was a toFixed() chosen at its own call site, so the rule was a
                          sentence and the practice was five decisions. Precision is derived from
                          the measured spread now, so the page CANNOT quote finer than it earned.
                          THE TITLE PROMISES THREE LINES and the page showed two; the third is the
                          cross_entropy call, where two of the four failures live, so the headline
                          count was the one number a reader could not check. THE MEMORY FIGURE
                          carried no shape and no baseline on a page about what numbers count --
                          memory.py's own docstring says a report omitting the baseline "would
                          attribute all of it to the loss", and this page was that report. A FOOTER
                          still said "three commands away" one screen below the heading already
                          corrected for it. THREE TILES WERE GREEN under a paragraph saying all
                          four are the same failure, so a ninety-second reader takes the colour and
                          leaves believing two are good news. AND A PROMISE I COULD NOT KEEP: the
                          glossary said every term the tiles use is defined in it, false twice. I
                          tried to guard it and could not -- the tiles emphasise words for stress
                          as often as for terminology, so the check flagged "broken" and
                          "estimated". Watched it fire on correct prose and removed the claim
                          instead, replacing it with one a test does keep: every entry carries a
                          figure from the run. That guard immediately found four entries carrying
                          none, two of which predate today
2026-09-09  exercise-09   #169 resolved against main, and resolving it found the fix itself broken.
                          THE PRECISION RULE I DERIVED WAS STILL A CHOSEN ONE: decimalsFor read
                          `spread >= 0.5 ? 0 : spread >= 0.05 ? 1 : 2`, the recorded spread was
                          0.44, so the page kept printing 9.1x one paragraph under the sentence
                          promising the tenth is not offered. I had written the entry above saying
                          "the digit is not offered anywhere" and it was false when I wrote it.
                          Found by rendering the page and reading the tiles after the merge, not by
                          a test -- every precision guard in this exercise reads the README or
                          results/, and the defect was on the page. The rule is -log10(spread) now:
                          a digit is offered only when the spread is smaller than that digit is
                          worth. No threshold, so nothing to tune and nothing to be lucky about.
                          RE-RAN THE SWEEP, which settled two things. by_steps reproduced BIT FOR
                          BIT on a different commit, so the training half is exactly deterministic
                          and the README's "re-running moves the spread" is now evidence rather
                          than a hedge. The memory spread moved 0.44 -> 0.56, ACROSS the discarded
                          threshold -- the same code would have printed a different digit
                          depending on which run happened to be committed, which is this section's
                          own lesson applied to precision. AND THE SECOND RATIO WAS BEING THROWN
                          AWAY: compare_paths returns the softmax-only ratio on every repeat and
                          the sweep kept only the first, so the page quoted a 1.8 value against a
                          spread of 0.56 measured on a 9. Its own spread is 0.019, thirty times
                          tighter, and it earns the tenth the memory ratio does not. Both recorded,
                          each quoted against its own. NEW GUARD READS THE PAGE: for each repeated
                          ratio, the figure at its earned precision must be present and no finer
                          rendering may appear anywhere. Watched red against the tree exactly as
                          #169 shipped it -- old threshold, old spread -- and against the
                          wrong-spread pairing, mutations held in memory and restored in a finally.
                          Its first version was red for the WRONG reason: "9.1x" is a substring of
                          "39.1x", the logits-to-hidden ratio, so it failed on correct prose until
                          a lookbehind was added
2026-09-09  design        #169 also puts the rail back where 08 has it, on 07, 09 and 10. PK: "you
                          have moved the rail from its original position which does not look
                          correct. The rail position in exercise 8 is good and I think we should
                          keep it standard across all the exercises." He is right and it was mine:
                          #167/#168 added left: max(0px, calc((100vw - 1500px)/2)) to three pages,
                          so above 1440 the rail travelled with the centred wrap and sat 24px from
                          the text -- 24px of air on the column's left against 554px on its right
                          at 2560. MEASURED BEFORE CHANGING ANYTHING, because AGENTS.md records an
                          agent reading this same complaint as "the rail is too far left" and
                          making it worse: 03, 04, 05, 06 and 08 all hold equal air either side at
                          every width, 07, 09 and 10 were off by 530px at 2560 and 210 at 1920.
                          Below 1440 max(0px, ...) clamps, which is why a full screenshot pass and
                          a review round missed it. Override removed; all eight now measure 0px of
                          asymmetry. THE GUARD LIVED IN ONE EXERCISE BOTH TIMES -- 08's centring
                          assertion sweeps widths and is hard-coded to 08 -- so it is now the
                          horizontal half of tests/test_rail_centring.py, which already discovers
                          railed pages from the filesystem for the vertical half. It asserts the
                          PROPERTY and never a distance: 08's wrap is 2200px and everyone else's
                          1500px, so the correct gap is 204px on one page and 554px on another at
                          the same viewport, and a guard naming either number fails the other while
                          both are right -- which is exactly the guard that shipped the first time.
                          Watched red on all three pages with the override re-applied, held in
                          memory, restored in a finally and the restore verified. I FIRST WROTE IT
                          AS A SEPARATE FILE with its own exemption ledger and got 04 wrong in it,
                          claiming its rail is not pinned; the ledger's own both-directions twin
                          caught that, and merging into the existing file removed the ledger
                          entirely because the filesystem answers the question. docs/DESIGN.md now
                          carries the rule and names the misreading that produces it
2026-09-09  ci            #170 opened: a pull request says where its preview actually is, and it
                          costs nothing. Since the build gate landed on 4 September every pull
                          request here ends with `docs: record #NNN in the queue` -- an entry that
                          must name the PR number, so it can only be written after the PR exists,
                          which makes a documentation commit the tip of nearly every branch. It
                          touches no deployed path, the gate correctly skips it, and VERCEL THEN
                          WRITES NO GITHUB DEPLOYMENT for a skipped build -- so the tip has no
                          environment and the PR reads "this branch has not been deployed". THE
                          PREVIEW WAS LIVE THE WHOLE TIME: fetching the branch alias after a cancel
                          returns HTTP 200 serving the earlier READY build, verified against
                          production. Only the report was wrong. THE GATE CANNOT FIX IT and the
                          reason is structural: it runs once per push and cannot know whether
                          another is coming, so it cannot spend one extra deployment on the last
                          one; building every docs push is what rate-limited the account for 24
                          hours in #128. So a workflow finds the deployment that ALREADY EXISTS and
                          comments it -- zero extra deployments in every case, and the comment
                          quotes should-build.sh's own verdict rather than restating its rule so
                          the two cannot drift. THREE OUTCOMES, NEVER TWO: the preview, "no preview
                          exists", and "I could not find out" -- a lookup failure published as an
                          absent preview sends someone to debug a build that worked, and the first
                          draft did exactly that until its twin caught it. ALSO a race that loses
                          the page build entirely: builds take 7 seconds and the two pushes on one
                          branch were 49 seconds apart, so autoJobCancelation could kill the build
                          carrying the page while the docs push skipped. Turned off. UNVERIFIED:
                          whether Vercel's rate limit counts deployments CREATED or builds RUN --
                          if the former, the gate saves build minutes only and the real fix is to
                          disable git deployments and drive previews from a workflow

2026-09-10  a11y          #175 opened: every canvas colour follows the theme. HANDOFF item 9, counted
                          before acting: exactly 12 literals against 16 theme-aware colour writes in
                          s1, s2 and s4. MEASURED BEFORE AND AFTER through the site's real theme
                          mechanism -- s1 and s2 each rendered TWO distinct canvases across six
                          themes and now render six. s4 rendered six either way, because its other
                          colour writes dominate the image while five of its literals were still
                          wrong in the details, and THAT is why the guard is lexical rather than
                          rendered: asking whether a colour CAN move has teeth, asking whether the
                          picture changed is a question a coarse instrument answers yes to. MY
                          FIRST PROBE WAS WRONG and it is worth recording: I set data-theme="light"
                          and data-theme="dark", which match no rule -- the picker offers system
                          plus four named themes, and light/dark come from prefers-color-scheme. So
                          both fell back to :root and I briefly read that as the site failing to
                          distinguish them. Re-measured with color_scheme emulation. CATEGORY
                          COLOURS LEFT ALONE deliberately: they encode a data class, and one is
                          interpolated per pixel into an ImageData buffer where a CSS variable
                          cannot go. The untidiness -- --warm/--cool exist as tokens in the same
                          files and are used for the line charts -- is recorded IN the guard with a
                          twin that fails if it stops being true

2026-09-10  tooling       #176 opened: the queue sync stops claiming work shipped in a release it did
                          not. HANDOFF item 5 named two defects; RE-CHECKED BOTH RATHER THAN
                          TRUSTING THE NOTE and one was already fixed -- _reapply replays an edit as
                          an edit. The other was live and reproduced against the real function
                          before anything changed: an entry written under [Unreleased] landed inside
                          ## [0.15.0], ABOVE that section's own ### Fixed, so it was a false claim
                          about what shipped and malformed too. The only note was "placed by the
                          following line only", which is true of many correct placements and says
                          nothing about a version. Relocates to [Unreleased] now, loudly, and
                          REFUSES in the one case it cannot repair -- no [Unreleased] section at all
                          -- because inventing one would be this tool deciding what a release
                          contains. Four tests: the repair, the refusal, the insertion point being
                          AFTER the section's own heading rather than above it, and the distinction
                          everything rests on (that [Unreleased] is not matched as a released
                          version -- if it were, every block would be "relocated" out of the section
                          it was already in and the first test would still pass). Watched red on the
                          tree as it shipped, restored in a finally. HANDOFF item 5 rewritten:
                          THIRD stale entry found in that file today, after item 7's slider
                          overflow and item 9's count

2026-09-10  tooling       #178 opened: the backup tripwire only cries for files it was protecting.
                          HANDOFF item 6 is 🤝 because removing paths from an append-only store is
                          PK's call -- but the ALARM is mine, and it was wrong. --verify treated any
                          store file with no counterpart in the checkout as a loss, so every run
                          told the reader to restore files PATTERNS never named. Measured: 45, not
                          the 19 the note recorded. 20 were copied into the store by hand and swept
                          in by snapshot()'s git add -A; 25 are docs/standards-history, the residue
                          of a PATTERNS entry that was DELIBERATELY removed while the append-only
                          store kept what it had -- the store working, not failing. A stored path is
                          a loss only if PATTERNS names it, worked out by globbing the STORE with
                          the tool's own patterns so there is no second matcher to drift from
                          collect(). Grouped by directory: four lines instead of forty-five, exit 0,
                          and A PROTECTED FILE THAT VANISHED STILL FAILS -- tested in both
                          directions, because every change that quietens a guard risks quietening
                          what it was for. NOTHING WAS DELETED FROM THE STORE: that needs PK naming
                          the path and its own removal commit, and the tool now prints those exact
                          commands including the read-back check. ALSO: v0.14.0 shipped without its
                          standards snapshot -- the release ritual's last step was skipped -- so
                          snapshot_standards.py was run and 44 archive guards now have something to
                          check instead of skipping

2026-09-10  agentic       #184 opened: the PreToolUse guard failed OPEN on a relative path, which is
                          the one thing its own docstring says it never does. Path(target).resolve()
                          anchors a RELATIVE file_path to the cwd of whatever runs the hook -- not
                          guaranteed to be the root, and under claude --worktree reliably not -- so
                          the resolved path failed relative_to(root) and the except ValueError
                          branch, written for a path genuinely OUTSIDE the repository, returned None
                          and allowed the call. Verified against the guard's own entry point: an
                          absolute out-of-scope path blocked, the identical path sent relative
                          passed, and so did a protected guard file and .claude/UNIT.md itself.
                          bash_write_targets has anchored to the root since it was written, so the
                          two branches of ONE function disagreed -- the same path blocked as a shell
                          redirect and passed as a Write. THIRD bypass in this file of one shape
                          (the guard answering its question correctly about a call it never saw),
                          after taking the root from __file__ and omitting Bash. Watched failing.
                          FOUND BY PROBING RATHER THAN READING, while checking whether an agent
                          could widen its own scope for row 9's writer half -- and MY FIRST PROBE
                          WAS WRONG IN A WAY THAT LOOKED LIKE A MUCH BIGGER FINDING: I sent relative
                          paths, which resolved against the real repo, so every Write appeared to be
                          allowed. Confirming the probe before believing it is what turned "Write
                          bypasses the guard entirely" into the real, narrower defect. CHECKED AND
                          NOT BROKEN: an agent cannot widen its own scope -- .claude/UNIT.md is
                          refused to both Write and Bash
```