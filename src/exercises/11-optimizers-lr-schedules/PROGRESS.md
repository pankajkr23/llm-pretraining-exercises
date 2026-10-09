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
| O4 | **Published run** | **done** | `FULL` preset on an Apple M4 GPU; `RESULTS.md` rendered from `results/*.json`. A test fails if any bundle's code digest or settings differ from today's, or if `RESULTS.md` differs from a fresh render. |
| O5 | **Notebook** | **staged — PK installs** | `artifacts/staged/build_notebook.py`; the guard forbids agents writing `tools/build_notebook.py`. Built and executed end to end at `LITE` after the published run; the executed copy sits beside the builder. |
| O6 | **muP checked against the authors' code** | **not done** | The rules are the paper's Table 3; no comparison with the `mup` package (README limits). |
| O7 | **Submission** | **PK's** | The README link, once merged and public. |

---

## The full record — what was tried, what held, what failed

Figures are from the published run (commit `0b101a0`, Apple M4 GPU) unless marked otherwise.
`RESULTS.md` is the authority; this section is the history around it.

### Ablations — each changes one thing

| ablation | what changed | held fixed | outcome |
| --- | --- | --- | --- |
| Adam three ways | the arithmetic: written out by hand, `torch.optim.Adam` on a float64 scalar, the weight inside the model | one real weight and its five real gradients | agree to 1.3e-17 (scalar) and 6.9e-9 (float32 weight in the model) |
| Bias correction off | `SwitchableAdamW`'s one flag | everything else; the flag-on optimiser is held to `torch.optim.AdamW` by a test | the uncorrected step is 3.16× at step 1, peaks at 6.57× (step 12), within 1% only from step 3,925; on the model the loss gap stays outside the seed spread for all 600 steps |
| Warmup off | the 100-step warmup | rate, data, seed | 19 of 19 matrices peak higher without it; with it, the median layer's ratio settles at step 304, three times the warmup's length |
| Cosine vs WSD | the schedule's shape, each at its own tuned peak | 300 planned steps, 30 warmup, data, seeds | stopped at 200 the two are within the noise (0.0181 against 0.0411); run to 300 WSD ends lower |
| Two finished models at step 200 | a WSD branch decayed over steps 170–199 vs a cosine planned for 200 | exactly 200 steps of data each; both tuned | the branch is lower (5.7180 vs 5.8913); the decay itself bought −0.0011 against WSD left at peak — inside the noise |
| SP vs muP across widths | the parametrization (init and per-layer rates, Table 3 of arXiv:2203.03466) | 3 widths × 7 rates × 2 seeds, 300 steps each | SP's optimum moves 3.07× over the 4× width range, muP's 1.23×; muP narrows the drift without removing it |

### Reproduced independently

The width sweep ran twice in full, from different commits (the second after a code fix elsewhere in
the package). First run: SP 3.06×, muP 1.29×, identical-model floor 1.08×, seed floor 1.18×.
Second (published): 3.07×, 1.23×, 1.01×, 1.15×. The conclusion did not move; the floors did, which
is why they are measured each time rather than assumed.

### Failures, in the order they happened, and what each changed

1. **A WSD decay shape was accepted until the decay phase began.** A typo'd shape would have run
   for the whole stable phase, then crashed. Now refused at the call (`_check_shape`), with a test.
2. **A checkpoint was copied as two `deepcopy` calls**, giving the copied optimiser its own copies of
   the parameters — a branch would have trained nothing, silently. Now one call over the pair, and a
   test that a branch moves the copied weights.
3. **The quote gate caught a borrowed phrase** in `schedules.py` and in the notebook builder,
   before commit. Rewritten in our own words.
4. **The schedule comparison was not at an equal budget** (first full run). WSD's branch was taken
   at step 200 and decayed to 230, so it trained 30 steps longer than the cosine it beat (5.5957 vs
   5.8914 in that run); and the cosine planned for 200 borrowed the 300-step cosine's peak instead of
   being tuned. Found by reading the code against the sentence it rendered. Fixed, tested by
   recording every training call's last step, and the run repeated. **The headline survived; the
   margin shrank, and the decay turned out to buy almost nothing at this point in training** —
   which the first version could not have shown.
5. **The SP and muP runs at the base width did not agree** although the models are bit-identical.
   Not a defect: the device's own run-to-run nondeterminism, now published as a floor beside the
   seed spread.
6. **The Mac hibernated mid-run** at 1% battery (02:40–08:39, running on battery overnight). The
   width sweep's wall time would have included six hours of sleep; it was re-run on its own under
   `caffeinate` and AC power.
7. **CI could not collect a test file** because it imported `optimizers.ratios`, which imports torch,
   in the job that has none. Invisible locally, where torch is installed. The two affected tests moved
   to the torch-gated file; the package — and so every bundle's code digest — is unchanged.

### What this exercise did not do

- Compare our muP against the authors' `mup` package (O6).
- Sweep more than two seeds, more than 300 steps, or any width above 1,024: the 4,096 prediction is
  an extrapolation 4× past the data.
- Test any schedule other than cosine and linear-decay WSD.

## Change log

### 2026-10-10 — the page

- **`web/`, on the spine, from the published bundles alone.** No experiment was re-run. Six figures,
  four of them driven by the reader. `tools/render_results.py` gained `render_page_data`, and the
  verdicts its sentences used to compute inline — the schedule comparison, the sweep's floors and
  drifts, the bias peak, the warmup count — moved into `*_numbers` functions that both `RESULTS.md`
  and `web/data.js` call. `RESULTS.md` regenerates byte-identical; a test fails if `data.js` drifts.
- **Curves the bundles do not store are recomputed, and refused if they disagree.** The bias
  curve, the WSD branch and the planned cosine are drawn by `optimizers.adam` and
  `optimizers.schedules`, and the renderer raises unless those functions reproduce the values the
  bundles did store. The bias figure smooths the losses as the experiment did; a torch-gated test
  holds the renderer's copy of that arithmetic to `ratios.smooth`.
- **Found while building it:** the uncorrected run is not merely "outside the noise". It is ahead
  for a dozen steps, then falls behind, and its smoothed gap dips inside the seed gap for stretches
  without ever staying there — the page states that precisely. `RESULTS.md`'s sentence "it never
  falls inside that noise within 600 steps" is looser than the measure it reports, which is "never
  *stays* inside"; left unchanged here, because this change keeps that document byte-identical.

### 2026-10-09 — published, after two corrections

- **The schedule comparison was not at an equal budget.** WSD's branch was taken at step 200 and
  decayed to step 230, so it trained 30 steps longer than the cosine planned for 200 it was compared
  with; and that cosine borrowed the 300-step cosine's peak rather than being tuned. Found by reading
  the code against the sentence it produced, before anything was published. The branch now starts
  at step 170 and ends on step 200, the planned cosine is tuned, and a test records each training
  call's last step. With the budgets equal, the decay itself buys almost nothing this early in
  training — the document now says so, computed.
- **A noise floor came for free.** At the base width SP and muP are the same model bit for bit, yet
  their GPU runs differ: that gap is the device's run-to-run nondeterminism, and every optimum shift
  is now set against it and against the seed spread.
- **The machine hibernated mid-run** (battery at 1%, 02:40 to 08:39). The width sweep was re-run on
  its own so its recorded wall time is real; the other four bundles had finished before it.

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
