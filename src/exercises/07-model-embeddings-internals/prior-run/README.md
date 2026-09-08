# The code that produced `results/measurements.json`

**This is not our code and it is not runnable here.** It is the driver of the *earlier* run — the
one `results/measurements.json` reports — recovered and tracked so that the published numbers have a
parent a reader can open. Its import paths point at a directory that no longer exists; it is
**evidence, not a build step**. The runnable, specified re-run is `tools/run_experiment.py`.

## Why it was nearly lost, which is the whole lesson

The run was driven from a scratch directory inside a coding agent's working area, under `/tmp`. That
directory was cleared. `results/measurements.json` recorded the architecture and none of the
optimisation, so the headline figure could not be aimed at, checked or defended by anybody.

Recovery was luck rather than process: an agent's own recorded tool calls carry the full contents of
every file a `Write` produced, so scanning the recorded calls of past agent runs for writes to these
paths returned seven Python files. **Nine more were never recovered** — see below. This is why `AGENTS.md` now says a
producer of published evidence is tracked code with a tracked entry point, or it is a number with no
parent.

## What is here

| file | lines | `sha256` as recovered |
| --- | ---: | --- |
| `arms.py` | 69 | `9b9c9663532628e0778b05f4649dd699…` |
| `decode.py` | 80 | `cbff76295144f506514cddaf94724bf7…` |
| `experiment.py` | 147 | `9a96825ac4cc61aec2551c8633578262…` |
| `fourier.py` | 65 | `203c24eba70b5a85dcabcaf833b9f547…` |
| `paired.py` | 58 | `6fe3e35756b31637d2a4b1f33519ad66…` |
| `sanity_lb.py` | 28 | `c7637aef2f7de26b10f4dc32d59e1638…` |
| `stress.py` | 104 | `ac19d87113d9e35ed257ec8ea6a52d43…` |

The digests are over the bytes **as recovered**, before the edits recorded below, so anyone can
check what was changed rather than take it on trust.

## What was changed, and why

Two lines of `fourier.py` used vocabulary this repository's lexical gate forbids, because it
describes the confidential source material rather than the engineering. Both are in a docstring and
neither touches the code:

| line | what it was | what it is |
| --- | --- | --- |
| 1 | the first word named the course's own unit of work | `Problem #4` |
| 5 | the subject of *"… calls the sovereign risk"* was that same unit | `the exercise` |

The banned terms are not reprinted here, because this file is tracked and the gate is lexical — it
would refuse the page that documents the fix. The digests above pin the bytes as recovered, so the
change is checkable against them rather than against this description.

**Nothing else was altered** — not the formatting, not the imports, not the dead code. Restyling
recovered evidence would destroy the property that makes it evidence: that this is what ran. The
directory is therefore excluded from `ruff`, and that exclusion is deliberate rather than an
oversight.

**One file was recovered and is NOT tracked here.** `RESULTS.md` quotes the course's own wording
verbatim, which this repository may not redistribute. Its findings are stated in our own words in
`README.md` and `DECISIONS.md`.

## What is missing, and what that costs

**Nine files named as the `source` of a published block were never recovered**, and no agent run on
this machine ever wrote them:

- `summary.py`
- `one_arm.py`
- `lock.py`
- `lock_break.py`
- `ng_sweep.py`
- `trained_w.py`
- `coherence.py`
- `dp128.py`
- `scale_cost.py`

`summary.py` is the important one: it produced the **ten-arm table** that is the exercise's headline.
So the three-arm paired comparison is fully specified by the code here, and the ten-arm table is
not. What survives of it is the per-seed losses, published in `measurements.json::pairing.per_seed`,
from which every gap in that table recomputes — which is a weaker guarantee than having the code and
a much stronger one than having neither.

**Read `results/measurements.json`'s `source` fields with this page open.** A field naming a file in
this directory can be opened. A field naming one of the nine above cannot, and says so.
