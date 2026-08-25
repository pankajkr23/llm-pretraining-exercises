# Decisions — Session 6

What was chosen, what was inherited, and what was invented. Each entry says what would overturn it.

---

## D1 · Replay reads the ledger. It never recomputes.

**Decided.** `replay.py` will re-slice immutable shards from spans recorded in the ledger. Its import
closure will contain no torch and no model code, and a test will assert that.

**Why.** The lecture's own answer to "how is this reproducible without a seed":

> *"I will not run the code… I'm going to run the ledger. I'm going to read and send. I will not
> calculate it."*

It is not a stylistic preference. Once the selector's scores depend on the current checkpoint, the
plan stops being a pure function of position, and re-deriving it can never be bit-identical. Measured
on the prototype: change one line in the planner and recompute — **96/96 slots differ**; read the same
interval from the ledger — **96/96 match**.

**Would overturn it.** Nothing plausible. Even with a fixed selector, replay-by-read is strictly
cheaper and strictly more faithful.

---

## D2 · The data system is torch-free. Only training imports torch.

**Decided.** Shards, manifests, packing, masks, the schedule, the ledgers, replay, fork and audit are
pure Python and numpy. torch is an optional extra (`--extra train`).

**Why.** CI runs `uv sync --all-packages` with no extras, because torch's default Linux wheel bundles
~2.5 GB of CUDA. Keeping the boundary means CI verifies almost the whole system rather than almost
none of it. Exercise 05 already proved the pattern works.

**Would overturn it.** A requirement that the *data* path itself use tensors — there is none.

---

## D3 · The sentinels sit outside the tokenizer's vocabulary.

**Decided.** `EOS = 10000`, `PAD = 10001`, model vocabulary `10_002`. No BOS.

**Why.** The frozen Session 2 tokenizer has **no EOS, no BOS and no PAD** — a contiguous `0..9999`
with no post-processor. Packing needs a document terminator and padding needs a meaningless token,
and neither exists. Adding them to the file would change its bytes and **void the tokenizer hash that
every shard manifest pins**, silently invalidating the provenance of every shard ever built with it.
So they are assigned out of vocabulary and materialised into the shard at tokenize time.

No BOS because it creates an ambiguous "which document owns position 0" case once documents are
packed, and nothing here needs one.

**Would overturn it.** Retraining the tokenizer with real special tokens — which is a Session 2
decision, not ours, and would invalidate every existing count.

---

## D4 · `submission_artifacts/` is tracked; `artifacts/` is not.

**Decided.** The bundle the assignment names lives in `submission_artifacts/`, in git, capped at
2 MiB. Checkpoints, shard arrays and full token traces stay in gitignored `artifacts/`, with their
sha256 recorded in the tracked bundle so integrity is still provable.

**Why not simply call it `artifacts/`.** It cannot work, and it fails *silently*. `**/artifacts/` is
a **directory** pattern, and git's own rule is that *"it is not possible to re-include a file if a
parent directory of that file is excluded"*. A `!` negation there is inert while `git add -A` reports
success and stages nothing. `run.log` is trackable only because `*.log` is a **file** pattern, which
a negation can override — `tests/test_submission_bundle.py` pins both halves.

**Why not track everything.** A checkpoint is ~67 MiB; eight of them is 536 MB against a repository
whose entire history is ~39 MiB.

**Would overturn it.** A grader requiring the checkpoints themselves — the assignment does not; it
regenerates them by running the command.

---

## D5 · `defer` and `floor_override` are ours. We say so.

**Decided.** Four statuses: `accept`, `reject`, `defer`, `floor_override`. The first two are the
selector's; the last two are a governance layer around it.

**Why this needs stating.** Three independent sources were searched and **all three contain zero
occurrences** of either concept:

| source | licence | `defer` | protected floor |
| --- | --- | --- | --- |
| OPUS paper, arXiv:2602.05400v2 | CC BY 4.0 | 0 | none |
| `gszfwsb/OPUS` reference implementation | MIT | 0 across all 25 files | none |
| `The-School-of-AI/LLM` (LightningLM) | Apache-2.0 | 0 | **yes — "AON"**, but with no per-candidate record |

In both implementations the decision is strictly **binary and stateless**: a candidate is in the
selected set or it is not, and a rejected one is never seen again.

**Our definition of `defer`, and why it is principled rather than invented.** Selection uses Gumbel
/ Boltzmann noise, so some accept/reject outcomes are decided by **noise rather than score**. A
candidate is *deferred* when its outcome would flip under resampling — it sits inside the noise band
of the cut. Computable, defensible, and ours.

**Would overturn it.** A course source that does define it. None has been found.

---

## D6 · OPUS is ported, not installed. Attribution is retained.

**Decided.** Reimplement the criterion (~300 lines) against the reference implementation as a spec.
Do not vendor or depend on the repo.

**Why.** Its `train.py` fails at import with `torch.empty(1, device="cuda")` and requires NCCL with no
gloo fallback — 8×A100 or nothing. The algorithm files themselves are plain PyTorch where CUDA appears
only as default keyword arguments.

**Settings, and the reason they differ from the paper's.** `preconditioner='adamw'` · CountSketch
**off** (our largest layer is ~262k dimensions; sketching to 8192 buys nothing) · `score_len=128` ·
buffer 8 · ρ=0.5 · stochastic selection.

**The cost objection is sequence length, not CPU.** OPUS is cheap because `score_len (512) ≪
train_seq_len (6144)` — a 12× lever. Our context *is* 512, so we have none of that discount:
~170% overhead at `score_len=512`, ~42% at 128. We publish our own measured overhead and cite the
paper's 4.7% and LightningLM's 3.2% as theirs, unverified here.

**Licences, retained in any derived file.** OPUS reference: **MIT**, dual copyright — © Keller Jordan
2024 (the harness forks modded-nanogpt) *and* © the OPUS authors 2026. LightningLM: **Apache-2.0**,
© Rohan Shravan and The School of AI.

---

## D7 · The lecture's description of OPUS is wrong, and we build the real thing.

**Decided.** Implement paper Eq. 23 — alignment of the *preconditioned* candidate gradient with a
proxy direction, minus a redundancy penalty against already-selected candidates.

**Why this is a decision at all.** The transcript says the proxy pass records *"which particular
weight is acting bad… store this map"* and selects candidates *"updating those weights"* — a weight
mask. **There is no weight mask in either implementation.** It is a continuous preconditioned inner
product. The lecture also omits the redundancy penalty, which is most of the benefit: greedy top-k
scores 40.49 against full OPUS at 41.75, with random at 40.29.

Building from the lecture alone would have produced the wrong system.

**Would overturn it.** Nothing — this is verified against both the paper and the code.
