# Merge order

**PK merges; nothing here asks you to fix anything.** Where a pull request needs work before it can
go in, that work is mine and this document says whose turn it is. If a row says *waiting on me*,
skip it and take the next one.

> **Kept current.** This file is updated whenever a pull request is opened, merged or changes state.
> If it disagrees with GitHub, GitHub is right and this file is stale — tell me and I will fix it.
>
> *Last updated: 2026-09-04, after the eight-dimension page sweep.*

---

## Read this first — one merge breaks the next, and it is measured

Every open branch touches three files that no two branches can change independently:

| file | why every branch touches it |
| --- | --- |
| `docs/agents/QUEUE.md` | the convention is that a pull request logs itself **when it opens** |
| `CHANGELOG.md` | `AGENTS.md` requires the entry **in the same pull request** |
| `.quote-check-receipt.json` | a digest over **all** tracked prose — any prose change anywhere invalidates every other branch's copy |

Merging one branch into another was tried before this was written. The result:

```
CONFLICT (content): Merge conflict in docs/agents/QUEUE.md
...
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes
```

The receipt did not merely go stale — it filled with conflict markers, so the checker that guards it
**died** rather than failing cleanly. That is not an order anyone can follow by hand.

### So the loop is: merge one → I sync the rest → merge the next

```bash
uv run python tools/sync_open_prs.py            # mine to run, after every merge
uv run python tools/sync_open_prs.py --dry-run  # what it would do
```

It merges `main` into every open branch, puts the two log files back together as *main's version
plus that branch's own entry*, regenerates the receipt and pushes. It **never rebases and never
force-pushes** — the branches are published, `AGENTS.md` forbids rewriting published history, and
this repository's own settings deny `git push --force`. A merge commit is uglier and is the honest
option.

**`merge=union` was considered and refused.** A union driver keeps both sides of every conflict,
which is right for two branches adding different lines and wrong here: fifteen branches carry a
byte-identical `#103` line, so the fifteenth merge would land fifteen copies of it. Both failure
directions — a duplicated line and a silently dropped entry — are covered by
`tests/test_sync_open_prs.py`, and both were watched failing against a deliberately broken copy.

**You do not have to wait for me.** Merging several in a row is fine; GitHub will simply refuse the
ones that conflict, and the next sync clears them. The order below is about *what breaks what*, not
about politeness.

---

## The order

### Round 0 — unblock CI. Merge this first.

| | pull request | why it is first |
| --- | --- | --- |
| **1** | **#124** · *the mermaid render test was timing out on the download* | `npx --yes @mermaid-js/mermaid-cli` fetches the package **inside** the render test's 180-second budget. It failed on three of four consecutive branches and passed on the fourth — a flake that reds pull requests which never touched it. Until this lands, every row below can go red for a reason that has nothing to do with it. |

### Round 1 — the exercises on a clock

| | pull request | notes |
| --- | --- | --- |
| **2** | **#105** · *exercise 09 — the three lines that decide what a model learns* | Must precede #106. |
| **3** | **#106** · *exercise 10 — one optimiser step, made to tell the truth about itself* | **Its base branch is `feat/09-loss-harness`, not `main`**, and it has diverged from it. GitHub will retarget it to `main` when #105 merges; I sync it immediately after. Do not merge it before #105. |

### Round 2 — a live production defect

| | pull request | notes |
| --- | --- | --- |
| **4** | **#121** · *exercise 01's s3 page threw before its first statement* | Confirmed live: the deployed page returns 200 and 18,222 bytes of a page that throws before its first statement and renders nothing. Also fixes three proof pages that have no light palette at all. |

### Round 3 — the shared layer, before anything that builds on it

These touch `web/_shared/`, vendored into six exercises. Landing them first means the per-exercise
rows below are read against a fixed shared layer rather than a moving one.

| | pull request | notes |
| --- | --- | --- |
| **5** | **#120** · *white text on a background that is bright in half the themes* | `_shared/tokens.css` ×6 |
| **6** | **#122** · *the step strip squeezed its prose to 29 characters on a tablet* | `_shared/explainer.css` ×6 |
| **7** | **#107** · *one theme picker, not eight* | `_shared/theme.js` ×6 + each `index.html` |

### Round 4 — per-exercise fixes. Any order within the round.

| | pull request | exercise |
| --- | --- | --- |
| **8** | #114 · *exercise 01 claimed the tab pattern and implemented none of it* | 01 |
| **9** | #115 · *exercise 02 claimed the tab pattern and implemented none of it* | 02 |
| **10** | #109 · *exercise 04's contents rail never said where the reader was* | 04 |
| **11** | #117 · *exercise 04's prose ran 145 characters a line* | 04 |
| **12** | #110 · *exercise 05's contents rail never said where the reader was* | 05 |
| **13** | #116 · *exercise 05's toggle said which option was chosen in colour alone* | 05 |
| **14** | #118 · *exercise 05's prose ran 145 characters a line* | 05 |
| **15** | #111 · *exercise 06's contents rail never said where the reader was* | 06 |
| **16** | #112 · *exercise 07's contents rail never said where the reader was* | 07 |

### Round 5 — a guard that needs its subjects merged first

| | pull request | notes |
| --- | --- | --- |
| **17** | **#113** · *a page that builds a contents rail must mark the section in view* | **Red until #109, #110, #111 and #112 are all in**, and red *correctly* — it is reporting a true defect on `main`. It was deliberately not silenced with an exceptions list, because `AGENTS.md` refuses adding an entry to clear a gate that is reporting something real. Merging the four turns it green with no edit. |

### Round 6 — repo-wide guards

| | pull request | notes |
| --- | --- | --- |
| **18** | #119 · *every deployable page is checked in all six themes, not one* | tests only |
| **19** | #108 · *pin how much of the shared stylesheet no page emits* | tests only |

---

## Before you merge anything

- **Nothing here needs a local checkout.** Every row is green in CI except #113, whose red is
  explained above.
- **Squash or merge commit — your call.** The queue checker reads squash-merged pull requests, and
  both work.
- **`main` is protected and production never auto-deploys** (`vercel.json` →
  `git.deploymentEnabled.main: false`). Merging changes nothing a reader sees until you promote it.

## While you merge

- **One at a time is safest but not required.** If GitHub blocks one as conflicting, leave it — the
  next sync clears it.
- **If CI goes red on a row that was green**, it is almost always one of two things, and both are
  mine: the receipt no longer describing the merged tree, or `QUEUE.md` not recording the pull
  request that just merged. Say the word and I will fix it rather than you resolving anything.

## After the last merge

- `uv run python tools/backup_local_only.py` — the local-only files are not in git.
- `uv run pytest tests/test_local_only_files_present.py` — the tripwire, after any branch switch.
- The changelog is then ready to become a release section whenever you want one.
