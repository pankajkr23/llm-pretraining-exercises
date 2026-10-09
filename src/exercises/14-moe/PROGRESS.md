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
| O4 | **Published run** | **done** | Commit `6b222e4`, Apple M4 GPU, 19:28–19:54 with no watchdog pause, from exercise 13's published baseline. A test ties the bundle to 13's committed results by digest; `RESULTS.md` is its render; the training log is `submission_artifacts/run.log`. |
| O5 | **Notebook** | **staged — PK installs** | `artifacts/staged/build_notebook.py`; the guard forbids agents writing `tools/build_notebook.py`. Executed at `LITE` after the published run; the executed copy sits beside the builder. |
| O6 | **Submission** | **PK's** | The README link, once merged and public. |

---

## Every process run on this machine for this exercise

Times are IST, 2026-10-09; GPU means the Apple M4 through MPS, outside the sandbox.

| when | process | why | outcome |
| --- | --- | --- | --- |
| ~09:33 | SMOKE run (GPU) | test every path on the device before a full run | **crashed**: the balancing buffers were on the CPU under a GPU model; fixed |
| ~09:35 | SMOKE run (GPU) | confirm the fix | passed |
| ~19:27 | SMOKE run (GPU) | test on the device before the full run | passed |
| 19:28–19:54 | full run (GPU, `caffeinate`) | publish | completed in 1,592 s |
| 19:28–19:54 | vitals watchdog (CPU, one sample a minute) | thermal state, memory pressure, swap, GPU, power; pause if stressed | no pause |
| ~19:58 | notebook at `LITE` (CPU, 33 s) | check it runs end to end against the published results | 9 of 9 code cells, 0 errors; its FULL section reads the published numbers |
| 10-10, ~00:00 | `render_results.py`, the site build, and chromium for the page's tests and screenshots (CPU, no torch) | build and check the page | no training re-run; `RESULTS.md` renders byte-identical |

## The full record — ablations, with the published numbers

From the published run (commit `6b222e4`). `RESULTS.md` is the authority.

| ablation | what changed | held fixed | outcome |
| --- | --- | --- | --- |
| Conversion | the feed-forward layer → 8 copied experts and a router | weights, data, the validation half | validation loss unchanged to the last digit (3.052571 both); largest logit difference 1.6e-5 |
| Router scoring | softmax vs sigmoid, 2M tokens each | the converted model, data, rate | softmax 3.0675, sigmoid 3.0682 — a choice, not a ranking (one run each) |
| MoE vs dense continuation | the converted model vs the unconverted one | the same 10M further tokens, schedule (peak 0.0005 = half the dense model's rate) | both rise first with the re-warmed rate, then fall; the MoE ends at 3.0172, the dense control at 3.0240 (−0.0068, one run each) |
| Load balancing | bias-only, no auxiliary loss | — | final largest violation 0.314; at worst 3 experts idle during the run, none at the end |
| Cost | — | — | 90.3M parameters in total, 31.2M active per token, against 21.3M dense; 14,719 against 35,932 tokens per second in a reference implementation that runs experts in a Python loop |

## Change log

### 2026-10-10 — the page

- `web/` built on exercise 10's template: the conversion as a figure the reader can try to break,
  the router's balance stepped through training, validation against the dense control, and the
  parameters a token pays for against what the model stores.
- `render_results.py` gained `numbers()`, the comparisons `RESULTS.md` states, computed once and
  read by both outputs, and `page_data()`, which writes `web/data.js`. The first block's per-expert
  loads exist only in the training log, so each row is checked against the bundle before use —
  `n_experts` values, summing to `top_k`, the bundle's violation and dead count at that step, and no
  more imbalance than the maximum over every block — with a twin test that breaks each check.
- **Found while building:** the most experts idle at once (3) happened at steps the log does not
  record, where at most 1 shows. The page therefore draws imbalance and idle counts from the bundle
  at every step, and only the loads from the log.

### 2026-10-09 — published

- One full run, from exercise 13's published baseline, with no pause. The conversion changed nothing
  measurable, the MoE kept training below its starting loss, and the page now says why both arms
  rose first (the re-warmed rate, which the unconverted control shows too).
- **Found on the device, not in the tests:** the balancing buffers were created on the CPU while the
  experts and router moved to the GPU, so routing on MPS added a CPU bias to GPU scores and crashed. A
  meta-device test now holds the buffers to the experts' device, watched red against the old code.
- **The confidentiality gate refused a commit**: one sentence in `layer.py` ran twelve words in
  common with the reference material. Rewritten in our own words before it entered any commit.
- `test_moe_results.py` (no torch, runs in CI's plain job) holds the bundle to today's code and
  settings, to exercise 13's committed results by digest, and to its render, and requires the
  training log to cover both continuations.

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
