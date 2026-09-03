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

## Queue

Order is the retro-fix order from the workplan. Nothing here is IN FLIGHT until a human says so.

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

### unit-07-retrofit
- status: QUEUED
- scope: `src/exercises/07-model-embeddings-internals/`, `CHANGELOG.md`
- what: retro-fit the page to `docs/DESIGN.md`, and its notebook to the rule above. Its own
  `PROGRESS.md` already names two defects.
- reviewers: reader, engineer, auditor, **continuity**

### unit-06-retrofit … unit-01-retrofit
- status: QUEUED
- what: the same, in descending order. Two cross-cutting items get **their own units**, not folded
  in: promoting the theme and contrast guards out of exercise 08 into `tests/`, and consolidating
  the seven names for two controls in `web/_shared/`.

### unit-09, unit-10
- status: QUEUED
- what: new exercises, scaffolded with `tools/new_exercise.py` — never by hand, since six test
  families apply the instant `pyproject.toml` lands.

---

## Log

```
2026-09-03  fleet  queue created; no unit has run yet
```
