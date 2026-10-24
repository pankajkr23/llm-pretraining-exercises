# Re-run, 2026-10-24: the harness, to refresh its provenance

**Why.** The historical [`../../run.json`](../../run.json) was written on commit `3523320`, and its
code digest no longer matches the committed `trainloop` and `lossheads` it covers. This re-run was
made to record a current commit and code digest beside the same figures. The historical file is
unchanged, and the documents still render it.

**How.** `trainloop.harness` and `tools/render_results.py`, on commit `fd18c2e` — the `git_sha` the
bundle records. The bundle does not record the command line. **The working tree had uncommitted
edits**, including a docstring change in exercise 09's `lossheads`, which this exercise's code digest
covers and which has **since been reverted**. So this bundle's `code_digest` (`sha256:f231acb0…`)
describes code that is not committed anywhere, and it differs from today's committed code
(`sha256:9a23b7ea…`). It is kept as it is, because it is the only honest record of what ran. Same
config; the machine's OS moved from macOS 26.6.2 to 27.0.1. **Other jobs were running on the same
CPU during this run.**

| figure | historical | this re-run |
| --- | ---: | ---: |
| every shape, gradient-check, accumulation, grad-norm and float figure | — | identical |
| wall clock, timed steps | 7.663 s | 10.538 s |
| measured device peak | 3.328 TFLOP/s | 3.210 TFLOP/s |
| achieved | 909.99 GFLOP/s | 661.78 GFLOP/s |
| tokens/second | 26,516 | 19,283 |
| **MFU** | **27.34%** | **20.62%** |

Sources: [`RESULTS.md`](RESULTS.md) here against [`../../../RESULTS.md`](../../../RESULTS.md); the
bundle is [`run.json`](run.json), and `web-data.js` is the page data rendered from it. Outside
`provenance`, the two bundles differ only in timing fields: `item_5_mfu`'s time-derived values,
`facts.run_seconds`, and the per-step `trace.seconds`.

**Conclusion: holds for everything the code decides; the utilisation figure is not comparable.**
Every count, loss, gradient and float decomposition reproduced exactly. The MFU of 20.62% was
measured while other jobs shared the CPU, so its wall clock measures the machine's load, not this
code: the timed steps took 38% longer while the separately timed peak fell only 3.6%, which is
what contention during the run looks like. **It is not evidence that the code got slower, and it
does not replace the published 27.34%.** Settling it takes a run on an otherwise idle machine.
