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

## Every process run on this machine for this exercise

Times are IST, 2026-10-09. GPU means the Apple M4's GPU through MPS, run outside the sandbox (MPS is
hidden inside it). Long runs are wrapped in `caffeinate -i -s`, which stops idle sleep while on AC.

| when | process | why | outcome |
| --- | --- | --- | --- |
| 09:28–09:31 | full run, attempt 1 (GPU) | publish | crashed: float64 on MPS |
| ~09:33 | SMOKE run, both stages (GPU, ~2 min) | test on the device first | passed after the fix |
| 09:34–11:10 | full run, attempt 2 (GPU, `caffeinate`) | publish | trials and both 50M-token runs completed; crashed out of memory at the largest batch |
| ~11:12 | SMOKE run (GPU) | test the probe and margin fix | passed |
| 11:13–12:24 | full run, attempt 3 (GPU, `caffeinate`) | publish | ended when the machine restarted |
| ~15:15 | SMOKE run (GPU) | test on the device after the restart | passed |
| 15:18–16:52 | full run, attempt 4 (GPU, `caffeinate`) | publish | trials, both 50M-token runs and the search completed with the machine calm throughout (thermal state 1, no pauses); crashed out of memory in the rate checks at batch 386 |
| 16:55–17:30 | three diagnostics in one capped process (GPU, minutes each) | find why memory ran out at 85% of a batch that fitted | each run left 0.22–0.37 GiB behind, and a forced garbage collection did not free it: a leak, not a margin |
| ~17:35 | the same diagnostic after the fix (GPU) | confirm the fix on the device | 0.00 GiB left after each of three runs |
| — | full run, attempt 5 | publish | below |
| 15:28– | vitals watchdog (`artifacts/vitals/watchdog.sh`, CPU only, one sample a minute) | record thermal state, memory pressure, swap, GPU use, battery and power; pause the run if the machine is stressed | running |
| throughout | monitors reading the run's log once a minute | report stages and failures | ended with each run |

**The watchdog, and why it may pause a run.** Every minute it appends a row to
`artifacts/vitals/vitals.csv` (gitignored). If the thermal state reaches *serious*, memory pressure
reaches *critical*, or swap passes 2 GB, it stops the run's processes with `SIGSTOP`. It resumes them
with `SIGCONT` after two consecutive calm minutes, and records each pause and its reason in
`pauses.log`. A pause inside a timed stage makes that stage's throughput false, so any pause is
reported beside the results and the speed figures it touched are not used. It was tested on a dummy
process before use: paused, resumed, and exited when the process ended.

## Change log

### 2026-10-09 — attempt 4 found a memory leak in the measurement itself

The fourth attempt ran with the machine calm — thermal state 1, memory pressure normal, no swap and
no watchdog pause in 85 minutes — and still ran out of memory at 85% of a batch the search had just
passed. A diagnostic on the device showed why: every training run in the capped process left about
a third of a gigabyte allocated after it returned, and a forced garbage collection did not free it.

The cause was `memory.saved_bytes`, which measures what a forward pass keeps by handing every saved
tensor through a hook. The hook returned the tensor to autograd. The reversible Function saves its
own outputs, so the graph then held an output, the output held its autograd node, and that node's
`ctx` held the saved states and the blocks: a reference cycle through C++ objects that Python's
collector cannot break. Every run measures its first forward pass this way, so every run leaked one
graph, and three rate checks were enough. The hook now keeps the tensors in a plain list outside
the graph for the length of the forward pass and clears it; unpacking is refused, because a
measured forward is never backwarded. A test fails if any block outlives the measurement — watched
red on exactly the three reversible rules, the only ones whose Function saves its own outputs —
and the rate checks now release the device cache between runs.

### 2026-10-09 — three full-run attempts failed before the published one, each for a new reason

1. **Three minutes in: float64 on the GPU.** Measuring rebuild agreement converted gradients to
   float64, which Apple's MPS does not support. Every test ran on the CPU and could not see it, and
   the first fix (`.to("cpu", torch.float64)`) still converted on the device first. Fixed as
   `.cpu().double()`. **A two-minute SMOKE run on the GPU now precedes every full run**; it would
   have caught this, and it caught exercise 14's device bug the same afternoon.
2. **At the last stage: out of memory at the batch the search had just passed (447).** The search
   probed one bare step; training also clips gradients, and memory outside PyTorch's own tensors
   varies between processes (1.24 GiB of the 8 GiB cap in the failed run). The probe now runs two
   steps through the same operations as training, and the run uses a stated 85% of the batch found.
   The search's answer is still what is reported as the largest batch.
3. **The machine stopped.** A charger that could not keep up with sustained GPU load let the
   battery fall from 14% to 3% during the second attempt. The same baseline model then trained at
   43k, 40k and 31k tokens per second in three trials that differ only in learning rate — the
   machine's own throughput moved by 1.4×. Speed verdicts are now set against that measured spread
   (`throughput_floor`) instead of a fixed threshold. The third attempt ended when the machine was
   restarted.

Bundles from the failed attempts were never published. A test now requires every published bundle to
come from today's code, settings and a single commit, so a page cannot quietly mix two runs.

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
