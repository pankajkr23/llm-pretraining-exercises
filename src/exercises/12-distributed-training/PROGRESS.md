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

## The full record — what was tried, what held, what failed

Figures are from `results/zero.json`; `RESULTS.md` is the authority and this is the history around
it. Each failure below also became a rule in `CLAUDE.md`.

### Ablations — each changes one thing

| ablation | what changed | held fixed | outcome |
| --- | --- | --- | --- |
| The four stages | what each rank keeps: everything (DP), optimiser state sharded (1), plus gradients (2), plus weights (3) | model, data, world of 32, precision | at N = 32, 16 / 4.375 / 2.4375 / 0.5 bytes per weight, each equal to its formula; DP, ZeRO-1 and ZeRO-2 send 2·P·(N−1)/N per step, ZeRO-3 3·P·(N−1)/N |
| World size | N = 1, 2, 3, 4, 8, 16, 32 | stage, model, one step | every stage equals its formula at every N; N = 3 included because it is the only size here where padding is non-zero |
| Precision | bf16-mixed against fp32 | stage, model, data | both precisions, every stage, end on bit-identical weights to DP; bf16-mixed against an fp32 single device: loss within 5.4e-5 relative, total-update cosine 0.99986 |
| Distributed against one device | 32 ranks against one device on the whole global batch with `torch.optim.AdamW` | fp32, data, steps | loss within 5.7e-8 relative; weights within 6.0e-6, except the 64 attention key-bias weights at 2.0e-5 — their true gradient is zero, so AdamW turns rounding into steps (D8) |

### Guards watched failing

Every guard was broken on purpose and seen to go red before it was trusted: **30** deliberate
breaks in the build pass, **13** more in the audit-fix pass. Each original held in memory, restored in
`finally`, its sha256 verified, under a fresh `PYTHONPYCACHEPREFIX` — one break first looked as if
it had survived, and the cause was a stale `.pyc`, not a weak test.

### Failures, in the order they happened

1. **Fused optimiser kernels are not slicing-invariant on this CPU.** ZeRO-1 disagreed with DP by
   about 1e-6 until AdamW was rebuilt from single-rounding operations (D7).
2. **The key bias drifts against a single-device reference**, for a reason the test now asserts
   rather than widening a tolerance (D8).
3. **A twin probed the property after the event it was meant to observe** and failed for the wrong
   reason; it now holds its reference from inside the all-gather.
4. **An independent audit found four claims that only read as checked**: the ledger charged views,
   not storage; the leak test could not fail; the bundle was checked only against itself; the README
   overclaimed "every byte". All fixed, each with a break watched red (the audit-fixes entry below).
5. **CI's coverage ledger was red** until the six torch-gated test files were registered in the
   `train` job — they would otherwise have run nowhere.

### What this exercise did not do

- Run on real GPUs or a real network: devices and links are simulated in one process, so the time
  column is a model (450 / 50 GB/s, no overlap), not a measurement.
- Count activations or allocator overhead in memory.
- Test any all-reduce other than a ring: with a tree all-reduce, DP and ZeRO would agree only to
  rounding, not to the bit.

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
