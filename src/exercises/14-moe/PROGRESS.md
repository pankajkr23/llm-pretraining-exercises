# PROGRESS — Exercise 14

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** this file does not name branch or PR numbers — `git log` and `gh pr list`
answer that correctly and a markdown file goes stale.

**Deliverable shape:** a public link to this exercise's README, with the training logs in the repository — `submission_artifacts/run.log`, which `.gitignore` already re-includes. The submission platform's own field list has not been checked
against this yet; that is PK's, before submitting.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **Scaffold** | **done** | Created by `tools/new_exercise.py`. |
| O2 | **Upcycling** | **done** | Every expert a copy of the trained feed-forward layer, top-k weights renormalised in the model's dtype; the converted model matches the dense one to 1e-12 in float64 (`test_moe_layer.py`). |
| O3 | **Routing and balancing** | **done** | Float32 router at a tenth of the usual scale; softmax or sigmoid chosen by a short trial; selection bias balancing with no auxiliary loss. |
| O4 | **Published run** | **pending — after exercise 13** | `FULL` starts from exercise 13's trained baseline and its trial-chosen rate, so it runs once 13's has. |
| O5 | **Notebook** | **staged — PK installs** | `artifacts/staged/build_notebook.py`; the guard forbids agents writing `tools/build_notebook.py`. Executed at `LITE` only once the published run exists. |
| O6 | **Submission** | **PK's** | The README link, once merged and public. |

---

## Change log

### 2026-10-09 — independent audit, before the published run

A read-only audit asked of every claim whether a test would fail if it were false. Fixed:
- **The bias never entering the weights was untested** — identical experts make any weights that sum
  to one give the same output. A test now uses differing experts and a non-zero bias; adding the bias
  to the weights turns it red.
- **The router's gradient was untested**; a `.detach()` on the weights passed every test. Now tested
  and watched failing.
- **The router was chosen on the windows the results are reported on.** Selection and reporting now
  use disjoint halves of the validation split.
- **The renderer's verdicts were fixed text** ("the conversion changes nothing" printed whatever the
  difference). Now conditional, with a test that renders both ways. Continuity is measured for both
  routers and the worse is reported.
- **The dense model's token count was typed** (50,000,000; exercise 13 trains a whole number of steps,
  slightly fewer). Now read from exercise 13's results, whose digests join the provenance.
- **Two passages described the reference material's wording.** Rewritten as our own decisions.

### 2026-10-09 — built

- The MoE layer, upcycling, balancing, the three experiments and the tools that run and render them,
  each tested end to end at the `SMOKE` preset on a synthetic corpus.
- **Found while building:** renormalising the top-k weights in float32 and then casting broke exact
  continuity at 1e-12 (now renormalised after the cast); the continuation's rate was a fixed number
  that contradicted the decision it was meant to follow (now `continue_lr_fraction` times the rate
  the dense model trained at, read from exercise 13's results, with a test).

### 2026-10-07 — scaffolded

- Exercise created from the skeleton, registered in the `rest` integration shard and the
  root README table.
