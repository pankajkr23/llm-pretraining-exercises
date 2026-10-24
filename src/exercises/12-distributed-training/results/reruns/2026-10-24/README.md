# Re-run, 2026-10-24: the ZeRO simulation, to refresh its provenance

**Why.** To record a current commit and code digest beside the historical
[`../../zero.json`](../../zero.json), which was written on commit `2b58342`. The historical file is
unchanged, and the documents and the page still render it.

**How.** `tools/run_zero.py` and `tools/render_results.py`, on commit `fd18c2e` — the `git_sha` the
bundle records. The bundle does not record the command line. **The working tree had uncommitted
edits**, including a docstring change in exercise 09's `lossheads`, which this exercise's code digest
covers and which has **since been reverted**. So this bundle's `code_digest` (`sha256:42ace483…`)
describes code that is not committed anywhere. The historical bundle's digest
(`sha256:c38e0377…`) is the one today's committed code produces, so the historical file is the one
that matches the repository. Same config fingerprint, corpus and tokenizer; the machine's OS moved
from macOS 26.6.2 to 27.0.1.

| figure | historical | this re-run |
| --- | ---: | ---: |
| bytes per device, every stage and category | — | identical |
| bytes per weight, DP / ZeRO-1 / -2 / -3 | 16 / 4.375 / 2.4375 / 0.5 | 16 / 4.375 / 2.4375 / 0.5 |
| bytes sent per device per step, DP / ZeRO-3 | 2,582,920 / 3,874,380 | 2,582,920 / 3,874,380 |
| largest weight difference from DP, every stage and precision | 0 | 0 |
| cosine, bf16-mixed against the fp32 single device | 0.999861 | 0.999861 |
| 30B-weight ladder, every cell | — | identical |
| `code_digest` · `git_sha` · OS | `c38e0377…` · `2b58342` · 26.6.2 | `42ace483…` · `fd18c2e` · 27.0.1 |

Sources: [`RESULTS.md`](RESULTS.md) here against [`../../../RESULTS.md`](../../../RESULTS.md); the
bundle is [`zero.json`](zero.json), and `web-data.js` is the page data rendered from it. The two
`RESULTS.md` files differ in two lines, both in the provenance table.

**Conclusion: holds, exactly.** Every measured number is identical; only provenance changed. The
two bundles differ in three fields, all inside `provenance`: the code digest, the commit and the
OS version. The simulation has no timing in it to drift — its times are a model over byte counts —
so a re-run that changes nothing but provenance is what it should produce.
