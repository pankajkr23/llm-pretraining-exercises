# Re-run, 2026-10-24: all three bundles, to refresh their provenance

**Why.** The historical bundles in [`../../`](../../) were written on three different commits, and
only [`sensitivity.json`](../../sensitivity.json)'s code digest matches the code committed today.
This re-run was made to put all three on one commit with a current code digest. The historical files
are unchanged, and the documents still render them.

**How.** The exercise's own producers (`lossheads.harness`, `lossheads.training`, `lossheads.training
--sensitivity`) and `tools/render_results.py`, on commit `fd18c2e` — the `git_sha` every bundle here
records. The bundles do not record the command line. **The working tree had uncommitted edits**,
including a docstring change in `lossheads` that has **since been reverted**. So the `code_digest`
these bundles carry (`sha256:7b1a1007…`) describes code that is not committed anywhere, and it differs
from today's committed `lossheads` (`sha256:dd26c08a…`, the digest the historical
`sensitivity.json` carries). A docstring changes no number, but the digest is the only honest record
of what ran, so it is kept as it is. Same inputs as the historical run: same corpus digest, same
tokenizer digest, same config fingerprint. The machine's OS moved from macOS 26.6.2 to 27.0.1.

| figure | historical | this re-run |
| --- | ---: | ---: |
| broken shift, 300 steps | 0.1784 | 0.1784 |
| correct shift, 300 steps | 4.1447 | 4.1447 |
| `t+2` minus `t+1`, 300 steps | +1.0416 | +1.0416 |
| further head higher | 297/300 | 297/300 |
| every row of the 60 / 150 / 300-step sweep | — | identical |
| memory ratio, plain against chunked | **9.09x** | **8.99x** |
| memory ratio, five repeats | 8.88x to 9.44x (spread 0.56) | 8.96x to 9.25x (spread 0.29) |
| softmax-only ratio | 1.80x (spread 0.019) | 1.79x (spread 0.019) |
| losses on both memory paths | identical | identical |

Sources: [`RESULTS.md`](RESULTS.md) here against [`../../../RESULTS.md`](../../../RESULTS.md); the
bundles are [`harness.json`](harness.json), [`training.json`](training.json) and
[`sensitivity.json`](sensitivity.json). `web-data.js` is the page data rendered from them.

**Conclusion: holds.** Every training number is bit-identical — `training.json` matches its
historical file in every field but `provenance`. Only the memory figures moved, and they moved
inside the exercise's own stated noise: 8.99x lies within the historical five-repeat range of 8.88x
to 9.44x, and the published claim was already "about 9x" and no finer. The re-run's own spread
(0.29) is narrower than the historical one (0.56); two samples of five do not say which is typical.
