# PROGRESS — 04 · Data cleaning and deduplication

The plan, the state, and the evidence. `REQUIREMENTS.md` (local only) holds the requirement text;
`DECISIONS.md` holds the reasoning; this file holds what was built and how far it got.

**Written retrospectively on 2026-09-05**, from the commit history, the tracked files and the test
suite. **Where the record does not settle something it says so** rather than filling the gap.

**Deliverable:** every named cleaning stage run over real, permissively-licensed corpora, with every
token counted by our own tokenizer rather than estimated.

**Contract, in one line: no source content is silently dropped.** Every stage reports what it
removed and why, so the pipeline can be audited rather than trusted.

---

## Status

| stage | what it delivers | state |
| --- | --- | --- |
| **1 · Scaffold** | the exercise skeleton | **done** — `0245b4c`, 2026-08-09 |
| **2 · Pipeline spine** | the stage framework, own-tokenizer counting, and the notebook | **done** — `a5222d6`, 2026-08-16 |
| **3 · Normalize** | normalization, format discipline, language identification | **done** — `79c9dde`, 2026-08-16 |
| **4 · Filter and dedup** | quality filtering, deduplication, decontamination | **done** — `8badad1`, 2026-08-16 |
| **5 · PII and ship** | PII scrubbing, the published page, the deploy | **done** — `7ed331f`, 2026-08-16 |
| **6 · Post-deploy fixes** | a sidebar losing a word from every label; markup leaking; the pinned sidebar | **done** — `8f01d36`, `c2b62ff`, 2026-08-16 |
| **7 · Robustness** | rendering no longer crashes on a checkout without FLORES | **done** — `1f63521`, 2026-08-18 |
| **8 · Narrow screens** | the page stopped scrolling sideways at 320px | **done** — `1969e41`, `c1cd6a8`, 2026-08-19/20 |
| **9 · Documents** | the README becomes the end-to-end guide the root routes to | **done** — `4b9625c`, `be9b24b`, 2026-08-24 |
| **10 · Notebook** | a builder for the notebook, then untracked with the rest | **done** — `e36f162`, `76bb0cb`, `db9b288`, 2026-08-24. Both **local only** |
| **11 · Confidentiality** | quoted passages paraphrased; source vocabulary removed; identifiers renamed | **done** — `f5a93b0`, `0fd6556`, `09c9619`, `535ab95`, 2026-09-02 |
| **12 · Retro-fix** | orphan tokens, the shared layer, white-on-bright text, the rail, prose measure, permalinks | **done** — `40e01cd` (#99), `6207fea` (#101), `92104b5` (#120), `a6b8a5e` (#109), `0ccb016` (#117), `997d277` (#127), 2026-09-03/04 |
| **13 · Submit** | PK's action | **no record** — see *Submission* below |

**37 commits touch this exercise**, from `0245b4c` (2026-08-09) to 2026-09-05. **188 tests** collect
under it. The page returns **HTTP 200** anonymously.

## The three corpora, and why they are real

`NOTICE` attributes **OpenThoughts-114k** (Apache-2.0), **ai4bharat/sangraha** (CC-BY-4.0) and
**HuggingFaceH4/stack-exchange-preferences**, each with what it stands in for. Two of the three
licences require attribution, and that file is the attribution.

**The fetcher verifies each licence at fetch time from the source itself**, not from our own
catalogue, and refuses anything that declares none — an unverifiable licence is not a permissive
one.

Nothing on the page is a hand-made example pretending to be data. Where corpus text appears it has
already passed the PII scrubber, and a test scans every published byte to keep it that way. The
interactive demonstrations run on a synthetic document whose identifiers use RFC 2606 reserved
domains and the RFC 5737 documentation IP range, so they belong to nobody.

## What this exercise cannot establish

- **Three corpora are not the web.** Every rate here — what fraction was dropped, how much was
  duplicated — describes these three sources and does not generalise.
- **Deduplication is measured, not solved.** Near-duplicate detection has a threshold, and a
  threshold is a choice; the numbers move with it.
- **PII scrubbing is a filter, not a guarantee.** It catches the patterns it knows. A test proves
  nothing published matches those patterns, which is a weaker claim than "no PII is present".
- **Language identification is probabilistic**, and the low-resource languages are where it is
  least reliable — which is exactly where this exercise cares most.

## Submission

**No tracked file records a submission**, and this ledger will not invent one. The page has been
live and anonymous-reachable since `7ed331f`; whether it was handed in, and when, is PK's knowledge.

## What this record does not settle

- **The week between the scaffold (2026-08-09) and the pipeline (2026-08-16)** is not accounted for
  by anything in this exercise.
- **Why these three corpora.** `DECISIONS.md` argues what they stand in for; how the shortlist was
  drawn is not recorded.
- **Which quality-filter thresholds were tried and rejected.** Only the surviving values are in the
  code, and a threshold that was tested and dropped left no trace.
- **The 2026-09-02 confidentiality sweep rewrote prose across 25 files in this exercise.** What each
  passage said before is recoverable from git, but *why each one was phrased that way originally* is
  not, and this ledger does not reconstruct it.
