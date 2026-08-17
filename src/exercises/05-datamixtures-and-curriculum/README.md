# 05 · Data mixtures and curriculum

The V5 pre-training recipe: how much of each kind of data the model sees, in what order, and what
happens to every share when it is checked against the data that actually exists.

**→ [`SPEC.md`](SPEC.md) is the deliverable.** This file says how it is built and how to rerun it.
[`TOKENIZER.md`](TOKENIZER.md) carries the vocabulary decision; [`PROGRESS.md`](PROGRESS.md) is the
running log of findings, decisions and what would overturn each.

## The one rule everything else follows from

**Lane supply is summed from the datasets named in the Session 5 inventory, never quoted from a
slot headline.** That single choice is what makes the spec defensible under questioning, and it
changed answers on the first run.

| | finding | consequence |
| --- | --- | --- |
| **F1** | STEM's itemised supply is **146B** (D4 STEM 49B + peS2o 42B + proof-pile-2 55B); the session's supply check prices it at **250B**. No dataset carries the missing 104B. | Against a 240B demand: 0.96 epochs and it fits, or 1.64 and it needs repetition. The spec uses the itemised figure. |
| **F2** | The 2% agentic lane asks 40B of a **627M** pool — 63.8 epochs, against a repetition ceiling of 10.3B. The demand is **3.9× more than infinite repetition could ever be worth.** | Survives dropping every correction, so rejecting our supervision estimate still lands on impossible. The share stays at the floor and the gap is priced as a generation bill. |
| **F3** | 60% of the long-context lane is repo-packed code the inventory says is *"packed from code corpora"* — already counted under code. | A 6% share would double-count 60B. It becomes a **sequence-length schedule**, keeping its benchmark and holding no budget. |
| **F4** | Two Indic rows (Samanantar, BPCC) carry no token count at all. The slot headline leaves **5.1B** for them between it and the four rows that do. | Recorded as a residual rather than split into two plausible numbers nobody measured. |

## Everything published is computed

`SPEC.md` is **generated** by `python -m mixture` from the same code the tests pin, and
`tests/test_mixture_spec_render.py` regenerates it and compares byte for byte. Editing it by hand
fails CI — which is the point: exercise 03 shipped a wrong figure because a document and its
pipeline drifted apart and both halves looked plausible.

## Every guard has been watched to fail

`checks.py` holds thirteen invariants. Each is paired with a twin that proves it fails against a
deliberately broken fixture, and `tests/test_mixture_mutation.py` goes further: it rewrites each
guard in turn to return no findings, reruns the suite, and requires the mutant to die. **13 of 13
killed** — so no guard here is decorative.

Two tests found real defects rather than confirming intent. Arm D of the proxy was raising the
agentic share as a side effect of halving Indic, which allocates tokens that do not exist and makes
the arm unable to attribute its own result. And an early version of the roster test asserted
`len(expected) == 13` against a literal it had just built — a constant compared with itself.

## Run it

```bash
uv run python -m mixture              # rebuild SPEC.md and TOKENIZER.md from measured supply
uv run python -m mixture.inventory    # lane supplies, itemised vs the session's two headlines
uv run python -m mixture.checks       # the invariants, with their current state

uv run pytest src/exercises/05-datamixtures-and-curriculum                  # 126 tests
uv run pytest src/exercises/05-datamixtures-and-curriculum -m integration   # mutation testing
```

## Layout

```text
SPEC.md           the deliverable — generated, never hand-edited
TOKENIZER.md      the vocabulary decision, from exercise 03's and 04's measurements
PROGRESS.md       findings, decisions, and what would overturn each
src/mixture/
  config.py       every threshold in one frozen dataclass, with a fingerprint
  inventory.py    the Session 5 inventory as 30 typed rows; lane supply summed from them
  benchmarks.py   benchmark → loss map → training format → lane, across 20 benchmarks
  supply.py       demand against supply, with the repetition ceiling and two corrections
  lanes.py        the mixture: shares, Indic tiers, protected floor, anneal reserve
  curriculum.py   stages, difficulty bands B0–B5, reasoning-length bands, warmup bands
  proxy.py        the 1B/3B experiment, its metric, thresholds fixed before it runs
  checks.py       thirteen invariants
  export.py       renders SPEC.md and TOKENIZER.md
tools/
  build_notebook.py   emits the session notebook; edit this, never the .ipynb
tests/            every invariant twice, plus mutation testing
```

The session notebook is [`notebooks/S05-datamixtures-and-curriculum.ipynb`](../../../notebooks/S05-datamixtures-and-curriculum.ipynb)
— it imports this package rather than re-implementing it, runs in seconds, and ends by breaking
three invariants on purpose so a reader can watch the guards fire.

## Scope

A coursework exercise, not a proposal to anyone — see [`NOTICE`](NOTICE). The dataset figures come
from the Session 5 inventory and are reproduced for analysis; where the session flags them as
approximate, so does `inventory.py`.
