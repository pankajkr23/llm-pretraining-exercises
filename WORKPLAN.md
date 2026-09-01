# Work plan — the standing backlog

Written down because the queue got long and things were being asked for faster than they were
being finished. Ordered exactly as PK asked for it. **Nothing is pushed; every stage lands as local
commits on its own branch, and PRs are raised only when PK says so — one PR per exercise.**

Status keys: `done` · `in progress` · `queued` · `blocked`

## Stage 1 — Exercise 08, finished to the benchmark  ·  in progress

The benchmark is Sebastian Raschka's *A Visual Guide to Attention Variants in Modern LLMs*
(`docs/sessions/s8_visual_attention_variants_sebastian.html`, local-only, backed up). Logic and data
are sound; readability and design are the work.

| # | item | status |
| --- | --- | --- |
| 1.1 | Six-agent readability audit against the benchmark + the ladder-of-readers rubric | done — 75 findings, 37 ranked edits |
| 1.2 | All 10 blocking edits | done |
| 1.3 | The arc verdict measured the wrong thing; noise floor now measured, one finding lost to it | done |
| 1.4 | Remaining 27 edits (confusing / overclaim / polish tiers) | in progress |
| 1.5 | Name real models per mechanism (`shippedIn`), verified not guessed | queued |
| 1.6 | Full screenshot pass, six themes, 1400px and 390px | queued |
| 1.7 | Notebook re-read to the same standard | queued |
| 1.8 | Docs: exercise README/CLAUDE/PROGRESS/DECISIONS, root README, AGENTS.md, CHANGELOG | queued |
| 1.9 | O4 — Q2 answer drafted from data | blocked: the app link 404s until PR #83 merges and the production gate runs |
| 1.10 | O6 — session notebook | done (the log had said "stub" and was wrong) |

## Stage 2 — Retro-fix readability and design, one exercise at a time  ·  queued

Same treatment as 08, in this order, **each on its own branch with full e2e testing before moving
on**: `07` → `06` → `05` → `04` → `03` → `02` → `01`.

Per exercise: review → plan → fix → test → screenshot → docs → local commits. No PR until asked.

## Stage 3 — Exercise 09  ·  queued

Folder `src/exercises/09-loss-functions-output-heads`; sources `docs/sessions/s9.md`,
`s9_transcript.md`, `s9_assignment.md`. Own branch, own PR. Scaffold with `tools/new_exercise.py`,
never by hand.

## Stage 4 — Exercise 10  ·  queued

Folder `src/exercises/10-training-loop`; sources `docs/sessions/s10.md`, `s10_transcript.md`,
`s10_assignment.md`. Own branch, own PR.

## Standing constraints

- Do not hallucinate. Every date, number and quote from a primary source, checked mechanically.
- No shortcuts. Step by step, thoroughly tested, reviewed before concluding.
- Multiple perspectives; readability is the prime.
- Keep every level of documentation current: exercise, repo, agent instructions, and this file.
- Local commits throughout. PRs on request only.
