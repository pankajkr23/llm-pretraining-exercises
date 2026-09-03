# Work plan — the arc, not the state

> ## ⟶ For *where the work is right now*, read [`docs/agents/QUEUE.md`](docs/agents/QUEUE.md).
>
> **That file is the single source of truth for progress.** This one holds the *arc* — the stages,
> why they are in this order, and what each has to achieve. Two documents because they rot at
> different rates: the arc changes when a decision changes, the state changes several times a day.
>
> This split exists because there were **six** places recording progress and only the changelog was
> accurate. A reader had to reconcile them by hand, which is the re-derivation the queue exists to
> make unnecessary. Anything here that contradicts the queue is stale; fix it here.

**Stage order changed on 2026-09-03.** Exercises 09 and 10 now come **before** the retro-fit of
01–07. They are the two that teach training, which is the point of the repository; the retro-fit is
polish on work already shipped and released. It may run alongside them once the shared `web/_shared/`
layer is fixed, because until then every retro-fit unit edits the same files.

Status keys: `done` · `in progress` · `queued` · `blocked`

---

## Stage 1 — Exercise 08, finished to the benchmark · complete

The benchmark is Sebastian Raschka's *A Visual Guide to Attention Variants in Modern LLMs*
(held locally, not in the repo). Logic and data
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
| 1.9 | **O4 — Q2 answer** | **ready to submit** — v0.13.0 released 2026-09-02, production gate approved, and the app link returns 200 anonymously with every asset resolving. Submitting is PK's. |
| 1.10 | O6 — topic notebook | done |

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

## Confidentiality · complete

The source material this project is built from is confidential. It moved **outside the repository**
on 2026-09-02: keeping it inside and gitignored protected its bytes and nothing else, since a
tracked document could still name its files, publish their sizes, describe them or quote them.

| item | status |
| --- | --- |
| Material moved out; no tracked file names the directory or anything in it | done |
| Paths, filenames, sizes and content summaries scrubbed, including one served to the live site | done |
| Naming guard in CI **and** pre-commit | done |
| Quote guard written; runs only where the material is present | done |
| Paraphrase the passages that quoted the source verbatim (~60, exercises 02–08) | done |
| Both halves of the guard gating on commit | done |
| Five words banned and gated in CI and pre-commit; the per-topic file renamed | done |
| Git history | **left as it is, by decision.** PR descriptions were rewritten and scan clean |

## Stage 2 — Retro-fix readability and design, one exercise at a time · queued, RUNS THIRD

**Reordered on 2026-09-03: this now happens after stages 3 and 4.** Those two teach training,
which is the point of the repository; this is polish on work already shipped and released. It may
run *alongside* them once the shared `web/_shared/` layer is fixed — before that, every unit here
edits the same files and parallel agents collide on all of them.

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

## Stage 3 — Exercise 09 · blocked, RUNS FIRST

**`src/exercises/09-loss-functions-output-heads` no longer exists.** It was an empty directory and
is gone; git never held a file under it on any branch or tag, and the backup store never held one.
Nothing was lost but the placeholder, and neither guard could have caught it — the backup tool
copies *files* matching `PATTERNS` and the tripwire asks whether a stored file is missing, so an
empty directory is invisible to both by construction. A `REQUIREMENTS.md` inside it **would** have
been protected. The generator recreates it, so there is nothing to restore.

**Blocked on two things**, both in [`docs/agents/QUEUE.md`](docs/agents/QUEUE.md): PR #96, and the
decision about the two explainer documents, which are gitignored and therefore unreadable by any
worktree, clone or CI job that would need them to build 09's explainer.

Sources are the local reference material for that topic. Own branch, own PR. **Scaffold with
`tools/new_exercise.py`, never by hand** — six test families apply the moment `pyproject.toml`
lands, three of them checking for gitignored files a fresh clone will never have.

Build it to `docs/DESIGN.md` from the first commit rather than retro-fitting it later.

## Stage 4 — Exercise 10 · queued, RUNS SECOND

`src/exercises/10-training-loop`, likewise gone as an empty directory and likewise recreated by the
generator rather than by hand. Sources are the local reference material for that topic. Own branch,
own PR.

This is the flagship training run, so two rules apply here that do not elsewhere. **Exercise
`save()` in a two-step run before any long one** — three experiments in exercise 05 trained to
completion and then died writing their results, one of them losing fifteen trained models to its
final statement. And **print tokens-consumed ÷ corpus-tokens per lane next to the mixture table
before starting**, so a lane the run never reads through is visible rather than inferred.

## Release — v0.13.0 · shipped

Tagged, released, and deployed to production through the gated environment. The page is live and
returns 200 to an anonymous request.

**One thing did not follow, and it is the only item on this page waiting on a person:** exercise 08
is finished and released and has **not been submitted**. Everything else here is building; that one
converts completed work into a result.

**The release ritual, for the next one.** Recorded as a sequence because it has steps on both sides
of the tag and they are easy to get backwards:

1. A release PR moves `[Unreleased]` → `[X.Y.Z]`, dated. **PK merges it** — merging is always his.
2. Verify `main` is green, then `git tag vX.Y.Z && git push origin vX.Y.Z`. Tagging needs no
   prompting.
3. `release.yml` cuts the GitHub Release from that changelog section and deploys the tagged commit.
   **The production gate is PK's.**
4. **After the tag exists**, `uv run python tools/snapshot_standards.py` — it reads the tag, so it
   cannot run before one.
5. Re-check the app link logged out, then submit.

**A live defect in step 3, found on 2026-09-03 and not yet fixed:** when a tag has no matching
`## [X.Y.Z]` section, `release.yml` writes `have_notes=false`, echoes one line into the log, and
publishes the release with auto-generated notes. Nothing goes red. It should `exit 1`.

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
