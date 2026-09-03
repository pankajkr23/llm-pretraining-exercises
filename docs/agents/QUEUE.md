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

### unit-08-notebook
- status: QUEUED
- scope: `src/exercises/08-modern-attention-variants/`, `CHANGELOG.md`
- what: rebuild the topic notebook to teach the attention variants by **running them**, per
  `AGENTS.md`'s notebook rules. The current one imports the chronology package and prints what the
  web page already renders — it contains no attention implementation at all, nothing that touches a
  GPU, and nothing whose configuration can be varied. The runnable variant implementations belong in
  the exercise package with tests, and the notebook imports those.
- why first: the smallest unit that exercises the whole loop end to end and produces something
  immediately useful.

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
