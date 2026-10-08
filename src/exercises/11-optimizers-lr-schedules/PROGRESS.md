# PROGRESS — Exercise 11

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** this file does not name branch or PR numbers — `git log` and `gh pr list`
answer that correctly and a markdown file goes stale.

**Deliverable shape:** a public GitHub link to this exercise's README, with the supporting code in
the repository. No notebook is required in the repository for this exercise, so the notebook stays
local like every other topic notebook.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **Scaffold** | **done** | Created by `tools/new_exercise.py`. |
| O2 | **The corpus** | **done** | FineWeb-Edu slice, licence checked on the card at fetch time, disjoint validation and training rows; shared with exercises 13 and 14 (`DECISIONS.md` D1–D3). |
| O3 | **The five experiments** | **done** | `optimizers.experiments`; each tested end to end at the `SMOKE` preset on a synthetic corpus. |
| O4 | **Published run** | **pending** | `FULL` preset on an Apple M4 GPU; `RESULTS.md` rendered from `results/*.json`. |
| O5 | **Notebook** | **staged — PK installs** | `artifacts/staged/build_notebook.py`; the guard forbids agents writing `tools/build_notebook.py`. Built and executed end to end at `LITE`. |
| O6 | **muP checked against the authors' code** | **not done** | The rules are the paper's Table 3; no comparison with the `mup` package (README limits). |
| O7 | **Submission** | **PK's** | The README link, once merged and public. |

---

## Change log

### 2026-10-09 — built

- Five experiments, one per task in the exercise, each answering its question by measurement and
  each setting its effect against the spread between two seeds before naming it.
- The corpus fetcher, the reader and the token format that exercises 13 and 14 reuse.
- **Found while building, before anything was published:** a WSD decay shape that was not refused
  until a run reached its decay phase (now refused at the call); a checkpoint copied as two separate
  `deepcopy` calls, which would have left a branch's optimiser updating parameters its model does
  not hold (now one call, with a test that a branch moves the copied weights).
- **A number worth knowing before reading the results:** without bias correction Adam's step is
  larger, not smaller, for thousands of steps — it rises from 3.16 times the corrected step and comes
  within 1% only after about 3,900. A plot of the first twenty steps cannot show it stop mattering.

### 2026-10-07 — scaffolded

- Exercise created from the skeleton, registered in the `rest` integration shard and the root README
  table.
