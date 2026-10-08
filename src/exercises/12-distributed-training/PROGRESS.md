# PROGRESS — Exercise 12

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on a branch, not yet merged. This file does not name branch or PR numbers
— `git log` and `gh pr list` answer that correctly and a markdown file goes stale.

**Deliverable shape — read this before calling the source material done.** Check the submission
platform's own field list, not `REQUIREMENTS.md`, which can be truncated. As read on the local
copy, the shape is: **a notebook, and a public link to a README** that shows the concepts are
understood. A public link means the README must be on `main` of the public repository — a branch or a
login-walled preview does not satisfy it. Both submission and merging are PK's.

---

## Stages

| # | stage | status | note |
| --- | --- | --- | --- |
| 1 | Scaffold | **done** | Created by `tools/new_exercise.py`. |
| 2 | Simulated world, ledger, ring collectives | **done** | Byte counts exact at N = 2, 3, 8, 32, padded and unpadded. |
| 3 | Demo model from exercise 09, split into units | **done** | Unit-by-unit forward equals 09's forward bit for bit. |
| 4 | AdamW on flat shards | **done** | Rewritten once — see `DECISIONS.md` D7. |
| 5 | Stages 0–3 | **done** | Bit-identical to each other in both precisions. |
| 6 | Measurements, producer, `RESULTS.md` | **done** | Every persistent memory category and every communication count equals its formula. |
| 7 | README, `DECISIONS.md`, `CLAUDE.md` | **done** | |
| 8 | Topic notebook (local only) | **done** | Built and executed end to end; not tracked, by policy. |
| 9 | CI registration of the torch-gated test files | **done — lead** | Landed in the same pull request as this exercise. |
| 10 | Audit fixes | **done** | See the change log entry below. |
| 11 | Review, merge, submission | **open — PK** | |

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **Scaffold** | **done** | Created by `tools/new_exercise.py`. |
| O2 | Register the torch-gated test files in `.github/workflows/ci.yml` and `tests/test_ci_shards_cover_everything.py::OPTIONAL_DEPENDENCY_GATES` | **done** | By the lead, in the same pull request. |
| O3 | Root README row summary | **done** | By the lead, in the same pull request. |
| O4 | A browser page | **not planned** | No `web/` — the deliverables are a notebook and a README. |

---

## Change log

### 2026-10-09 — audit fixes

- **The ledger charges storage, not views.** It charged `numel × element_size` of what was put, so
  removing a `.clone()` from a shard's placement left a full buffer alive while the ledger reported
  1/N of it, with every test green — watched surviving before the fix. It now charges each
  tensor's storage bytes, once per device, and a test asserts every held tensor owns exactly its
  storage.
- **A per-step leak is now caught.** The no-growth test asserted `current == peak`, which a leak
  satisfies; it now compares the held total step by step and against the formula.
- **The bundle must be fresh.** Tests fail when `results/zero.json`'s config fingerprint or code
  digest differs from today's `Config()` and code. Communication is recorded per step and compared
  step to step, not inferred from a divisible total.
- Broken twins for "memory equals formula" and "bytes equal formula"; the ledger check now runs at
  N = 32 directly; README figures are checked by the quantity they name.
- README: the opening claim says what is counted (persistent buffers, ring traffic) and what is not;
  bit-identity is stated as a property of this simulator's ring all-reduce.

### 2026-10-09 — built

- The simulator: 32 ranks on 4 nodes, a per-rank memory ledger measured from real `nbytes`, and ring
  reduce-scatter / all-gather / all-reduce with exact per-rank and per-link byte counters.
- Stages 0–3 on one lockstep engine, each unit's forward recomputed in the backward so ZeRO-3's
  gathered weights genuinely die (weak-reference test).
- Two findings, both written up in `DECISIONS.md`: torch's fused optimiser kernels are not
  slicing-invariant on this CPU (D7), and the attention key bias is the one place a single-device
  reference drifts, because its true gradient is zero (D8).
- `tools/run_zero.py` → `results/zero.json` with the full provenance block; `tools/render_results.py`
  → `RESULTS.md`; drift and README-figure tests.

### 2026-10-09 — scaffolded

- Exercise created from the skeleton, registered in the `rest` integration shard and the
  root README table.
