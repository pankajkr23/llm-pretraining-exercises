# PROGRESS — 03 · Data collection framework

The plan, the state, and the evidence. `REQUIREMENTS.md` (local only) holds the requirement text;
`docs/DECISIONS.md` holds the reasoning; this file holds what was built and how far it got.

**Written retrospectively on 2026-09-05**, from the commit history, the tracked data and the test
suite. **Where the record does not settle something it says so** rather than filling the gap.

**Deliverable:** a graded catalogue and a decision made from it — what an India-first model should
train on, argued rather than asserted.

**Contract, in one line: every judgment carries its reasoning and its confidence.** A grade with no
stated basis is an opinion wearing a number.

---

## Status

This exercise was built in numbered phases, and **the commits say so** — a rare case where the
history carries its own stage table.

| stage | what it delivers | state |
| --- | --- | --- |
| **1 · Record spine** | the `Value`/`Gotcha`/`Gate` primitives, the seed CSVs, the twelve reference record arrays | **done** — `b43db9c` … `48c5fb1` *"Phase 1 complete"*, 2026-08-04 |
| **2 · The bundle** | the computed bundle the site renders | **done** — `87b5863` *"Phase 2"*, 2026-08-04 |
| **3 · Invariants in CI** | the five data-handling invariants, enforced rather than reviewed | **done** — `d856d80` *"Phase 3"*, 2026-08-04 |
| **4 · Web foundation** | tokens, motion, and the single path a number takes to the page | **done** — `8ef68c0` *"Phase 4"*, 2026-08-04 |
| **5 · Reasoning surface** | 16 explorers on one anchored page, plus the Dataset Card | **done** — `279b348`, `631d8d8` *"Phase 5 complete"*, 2026-08-04 |
| **6 · The Decision** | twelve sections, argued in order | **done** — `c66bfdb` *"Phase 6"*, 2026-08-04 |
| **7 · Print** | a print stylesheet, because the PDF is what is handed in | **done** — `4ced7b1` *"Phase 7"*, 2026-08-04 |
| **8 · Ship** | `NOTICE`, the root README row, the changelog, an accessibility pass | **done** — `9d7150a`, 2026-08-05 |
| **9 · Answer the question** | which datasets to *actually pull*, which the framework had not yet said | **done** — `8145151`, 2026-08-05 |
| **10 · Phase 0** | stop shipping statements that are false | **done** — `1f93e2b`, 2026-08-05 |
| **11 · Confidentiality** | quoted passages paraphrased; source vocabulary removed | **done** — `0fd6556`, `09c9619`, `535ab95`, 2026-09-02 |
| **12 · Retro-fix** | the shared layer, printing, link colours, one theme picker | **done** — `6207fea` (#101), `a459c25` (#126), `dbf9783` (#131), `3377cc9` (#107), 2026-09-03/04 |
| **13 · Submit** | PK's action | **no record** — see *Submission* below |

**154 commits touch this exercise** — the most of any — from `98dce97` (2026-08-04) onward.
**210 tests** collect under it, the largest suite in the repository. The page returns **HTTP 200**
anonymously.

**Counts read from the tracked data, not typed:** `catalog.json` holds **145** entries and
`benchmarks.json` holds **31**.

## The five invariants are this exercise's real contribution

`tests/test_invariants.py` enforces, in CI, what a data pipeline anywhere in this repository must
hold to: training never touches eval data · nothing excluded may enter a commercial mix · every
judgment carries its reasoning and confidence · a measurement must name what produced it · no source
content is silently dropped.

Each is paired with a test proving it **fails** when broken. That pairing is the point: a guard
nobody has watched fail is not a guard, and these five are the ones later exercises inherit.

## Where the reasoning lives

**`docs/DECISIONS.md` is tracked and is the substance of this exercise** — the mix, the grades and
the argument for each. It sits one directory down rather than at the exercise root, which is where
every other exercise keeps its decision record. That is a real inconsistency and it is left alone
deliberately: moving it would break the links that point at it, for a gain that is purely tidiness.

## What this exercise cannot establish

- **A catalogue is not a corpus.** 145 graded entries say what is worth pulling, not what pulling
  them would yield. No token has been fetched here.
- **The grades are judgments.** They carry their reasoning and a confidence, which is the honest
  form, but a confidence is not a measurement.
- **Licence readings are ours.** Where a source's own terms were ambiguous the reading recorded is
  this author's, and `NOTICE` says so.

## Submission

**No tracked file records a submission**, and this ledger will not invent one. The deliverable was a
PDF — `4ced7b1` built the print stylesheet for exactly that — and whether it was handed in, and
when, is PK's knowledge.

## What this record does not settle

- **Seven phases landed in a single day** (2026-08-04). The history cannot say how much of that was
  planned in advance versus discovered while building.
- **Why 145 datasets and 31 benchmarks** — where the cut-off came from is not argued in a tracked
  file, only the result.
- **What "Phase 0" corrected.** `1f93e2b` says statements were false and fixes them; which
  statements, and how they got published, is not recorded.
- **Task 2.2b**, marked still open at `87b5863`, is never mentioned again. Whether it was closed,
  dropped, or absorbed elsewhere is not in the record.
