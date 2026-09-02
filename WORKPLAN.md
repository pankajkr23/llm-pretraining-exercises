# Work plan — the standing backlog

Written down because the queue got long and things were being asked for faster than they were being
finished. Ordered exactly as PK asked for it. **Exercise 08 is pushed: PR #83 is open, all checks green, awaiting merge and the production
gate — both PK's.** Later stages land as local commits on their own branch and are pushed when
PK asks — one PR per exercise.

Status keys: `done` · `in progress` · `queued` · `blocked`

**Last updated:** 2026-09-02, after the exercise 08 readability rebuild and the A/B decision.

---

## Stage 1 — Exercise 08, finished to the benchmark · complete except O4

The benchmark is Sebastian Raschka's *A Visual Guide to Attention Variants in Modern LLMs*
(`docs/sessions/s8_visual_attention_variants_sebastian.html`, local-only, backed up). Logic and data
were sound from the start; readability and design were the work.

| # | item | status |
| --- | --- | --- |
| 1.1 | Six-agent readability audit against the benchmark and the ladder-of-readers rubric | done — 75 findings, 37 ranked edits |
| 1.2 | All 10 blocking edits | done |
| 1.3 | The arc verdict measured the wrong thing; noise floor now measured, one finding lost to it | done |
| 1.4 | All 37 audit edits | done |
| 1.5 | Name real models per mechanism, verified not guessed | done — 21 claims, 8 models, 0 rejected by the gate |
| 1.6 | Full screenshot pass, six themes, 1400px and 390px | done |
| 1.7 | Notebook re-read to the same standard | done — 28 cells, backed up |
| 1.8 | Docs at every level | done |
| 1.9 | **O4 — Q2 answer** | **blocked**: `artifacts/q2_answer.txt` is generated and correct, but the app link 404s until PR #83 merges and the gated production workflow runs. A preview URL is login-walled and cannot satisfy a public-link requirement. |
| 1.10 | O6 — session notebook | done |

## Stage 1b — The readability rebuild · done

PK read the deployed page and rejected it: duplication, apparatus dressed as argument, repetitive
chapter openers, no sense of place, *"why do you narrow too much"*, *"the length of the page is too
much"*, and the one that mattered — *"you are eating my time fixing just the UI not the actual
content, storytelling and experience."*

| # | item | status |
| --- | --- | --- |
| 1b.1 | Measure before changing: 29,999px, 33 screens, prose 27–36% of the viewport above 1600px | done |
| 1b.2 | One catalogue, not two — the at-a-glance table restated the index | done |
| 1b.3 | Chapters given bodies: three of six were a heading and nothing else | done — chapter strips, with years |
| 1b.4 | Density: a row was 306px because of six stacked bands, not its word count | done — 238px |
| 1b.5 | Four blocks written to be wide that silently were not, incl. the invoice at 685px forever | done |
| 1b.6 | The rail marks the section in view, and says how long the page is | done |
| 1b.7 | Apparatus stops wearing display type — limits and colophon as notices | done |
| 1b.8 | The four families taught as one grid drawn four ways, not four abstract marks | done |
| 1b.9 | **A/B on the two contested decisions**, with a tool measuring all four combinations | done — PK chose **the index, and the large type** |
| 1b.10 | Harness retired; losing branch, switch, `variants.js` and `compare_variants.py` deleted | done |
| 1b.11 | `docs/DESIGN.md` rewritten as the standard, built from what 08 proved | done |

**Where the page landed**, against the 29,999px baseline, with type 19–38% larger at every width:

| viewport | height | prose | share | body type |
| --- | --- | --- | --- | --- |
| 1920 | 29,911px | 951px | **50%** | 22px |
| 1440 | 28,508px | 835px | **58%** | 19px |

Shorter than it started while carrying larger type, six chapter strips and a new four-families
figure. **Not** the ~18,000px the plan hoped for: thirty entries of catalogue prose have a floor,
and the instruction was to keep the facts.

## Stage 2 — Retro-fix readability and design, one exercise at a time · queued

Same treatment as 08, in this order, **each on its own branch with full e2e testing before moving
on**: `07` → `06` → `05` → `04` → `03` → `02` → `01`.

**`docs/DESIGN.md` now carries the standard and a numbered retro-fit checklist**, so this stage is
no longer "do what 08 did and work out what that means" — it is a list. Per exercise: review →
plan → fix → test → screenshot at five widths in four themes → docs → local commits.

A five-lens audit on 2026-09-02 inventoried what actually diverges. The full list is in
[`TODO.md`](TODO.md); the headlines:

- **Exercise 01's dark themes are broken on all four deployed proof pages** — 26 custom-property
  declarations with no semicolons, so four of the five diagram tokens are never declared in the dark
  blocks and a dark-theme reader gets light diagram colours. A live defect, awaiting PK's word on
  whether it jumps the queue.
- **04 names seven custom properties that exist in no theme**, so the fallback always wins.
- **05, 06 and 07 build a contents rail and never mark position**; **06 and 07 reserve a 260px rail
  gutter and never fill it**.
- **`anim.js` is imported by nothing** — 1,002 lines across six vendored copies. `explainer.css` is
  linked by six pages and used by two.
- **Seven names for two controls; five figure treatments; fifteen distinct `ch` measures.**
- **The theme and contrast guards exist only in exercise 08**, while six exercises link the
  six-theme token file.

## Stage 3 — Exercise 09 · queued

`src/exercises/09-loss-functions-output-heads` (currently an empty directory). Sources
`docs/sessions/s9.md`, `s9_transcript.md`, `s9_assignment.md`. Own branch, own PR. **Scaffold with
`tools/new_exercise.py`, never by hand** — six test families apply the moment `pyproject.toml`
lands, three of them checking for gitignored files a fresh clone will never have.

Build it to `docs/DESIGN.md` from the first commit rather than retro-fitting it later.

## Stage 4 — Exercise 10 · queued

`src/exercises/10-training-loop` (currently an empty directory). Sources `docs/sessions/s10.md`,
`s10_transcript.md`, `s10_assignment.md`. Own branch, own PR.

## Release — v0.13.0 · ready when PR #83 merges

Last tag is **v0.12.0**; `origin/main` is at that tag and **59 commits** sit on
`feat/08-attention-timeline-scaffold` (PR #83). `CHANGELOG.md`'s `[Unreleased]` is current.

The sequence, and who does what:

1. **PK merges PR #83.** Merging is his.
2. A release PR moves `[Unreleased]` → `[0.13.0]`, dated. Minor bump: additions and fixes, nothing
   breaking.
3. **PK merges the release PR.**
4. Verify `main` is green, then `git tag v0.13.0 && git push origin v0.13.0`. Tagging is mine and
   needs no prompting.
5. `release.yml` cuts the GitHub Release from that changelog section and deploys the tagged commit
   to production through the gated `production` environment. **The production gate is PK's.**
6. Only then does O4 unblock: re-check the app link logged out, then submit.

---

## Standing constraints

- Do not hallucinate. Every date, number and quote from a primary source, checked mechanically.
- No shortcuts. Step by step, thoroughly tested, reviewed before concluding.
- Multiple perspectives; readability is the prime.
- Keep every level of documentation current: exercise, repo, agent instructions, `docs/DESIGN.md`,
  and this file.
- Local commits throughout. PRs on request only.
- Anything that changes facts, figures, research or the reading experience gets **A/B tested**
  rather than swapped — and the harness carries its own end date.
