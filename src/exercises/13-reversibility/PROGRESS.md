# PROGRESS — Exercise 13

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** this file does not name branch or PR numbers — `git log` and `gh pr list`
answer that correctly and a markdown file goes stale.

**Deliverable shape:** a public link to this exercise's README, and the notebook in the repository. Topic notebooks are gitignored, so tracking this one needs a named `.gitignore` exemption like exercise 10's — `.gitignore` is a standards file, so that change is PK's. The submission platform's own field list has not been checked
against this yet; that is PK's, before submitting.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **Scaffold** | **done** | Created by `tools/new_exercise.py`. |
| O2 | **Reversible stack** | **done** | Midpoint, blend and leapfrog update rules with a custom autograd function that rebuilds each layer's input in the backward pass; matches stored autograd to rtol 1e-9 in float64 (`test_reversible_stack.py`). |
| O3 | **Memory measurement** | **done** | Exact bytes kept for backward by saved-tensor hooks; the largest batch found under a GPU memory cap, in a separate process. |
| O4 | **Published run** | **pending** | `FULL`: the trial over update rules and rates, the 50-million-token runs, and the largest-batch search, on an Apple M4 GPU; `RESULTS.md` rendered from `results/*.json`. |
| O5 | **Notebook** | **staged — PK installs** | `artifacts/staged/build_notebook.py`; the guard forbids agents writing `tools/build_notebook.py`. Executed at `LITE` only once the published run exists. |
| O6 | **Notebook tracked in the repository** | **PK's** | Needs the `.gitignore` exemption above. |
| O7 | **Submission** | **PK's** | The README link and the notebook, once merged and public. |

---

## Change log

### 2026-10-09 — audit fixes

- **The blend's float32 gradients are a few percent off at depth 12** (2.8e-2 relative at h = 0.5,
  against 1e-10 in float64): found by a new float32 check at the published depth. The run now
  records rebuild and gradient agreement for every trial candidate and for the chosen rule before and
  after its long run (D10).
- **The derived largest batch left out the reversible backward pass's working set** — one block
  re-run with autograd on, about five times what the reversible forward keeps per sequence at the
  test shape. Now measured and added (D4). A second, liveness-based instrument in the tests agrees
  with the saved-tensor hooks per sequence for every variant.
- Choices are scored on the first half of the validation split, reported losses on the second; the
  rate check at the largest batch runs a fixed number of steps; picks at a grid's edge are flagged.
- The renderer's direction words now follow the numbers, a batch at the search ceiling is a lower
  bound, and `test_reversible_render.py` checks both and that `RESULTS.md` is a fresh render.

### 2026-10-09 — built

- The stack, the model around it, the memory probes, the three experiments and the tools that run and
  render them, each tested end to end at the `SMOKE` preset on a synthetic corpus.
- **Found while building:** the derived largest batch ignored the search ceiling, so a smoke run
  searched up to 19,170 sequences and took over six minutes (now capped); the renderer stated that
  the measured memory was lower whether or not it was (now conditional on the numbers).
- **Found in a neighbouring exercise and not fixed here:** exercise 09's chunked cross-entropy keeps
  a full set of logits for backward under autograd (at 32,768 positions, 32,768 × 10,001 float32
  logits are 1,250 MiB), which its docstring says it avoids. This exercise uses its own checkpointed chunked loss instead; the 09
  finding is reported, not changed.

### 2026-10-07 — scaffolded

- Exercise created from the skeleton, registered in the `rest` integration shard and the
  root README table.
