# Decisions — Exercise 06

What was chosen, what was inherited, and what was invented. Each entry says what would overturn it.

---

## D1 · Replay reads the ledger. It never recomputes.

**Built.** `replay.py` re-slices immutable shards from spans recorded in the ledger. Its import
closure contains no torch and no planner, asserted transitively by
`test_replay_cannot_reach_the_planner_or_torch`, whose twin fails when the closure walker stops
seeing new imports. Measured over a recorded interval: **32/32 microbatches re-derived**, and one
flipped bit in one shard turns **exactly 1 of the 32** red — the damage is local, and the rest of
the replay stays green as evidence of that.

**Why.** The lecture's own answer to "how is this reproducible without a seed":

> *"I will not run the code… I'm going to run the ledger. I'm going to read and send. I will not
> calculate it."*

It is not a stylistic preference. Once the selector's scores depend on the current checkpoint, the
plan stops being a pure function of position, and re-deriving it can never be bit-identical. Measured
on the prototype: change one line in the planner and recompute — **96/96 slots differ**; read the same
interval from the ledger — **96/96 match**. Measured again on the shipped module: **32/32**
microbatches re-derived from the ledger and the shard bytes alone, and flipping one bit in one shard
turns exactly **1 of 32** red rather than the whole interval — the damage stays local, which is what
makes the check diagnostic rather than merely pass/fail.

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

**Why.** The frozen Exercise 02 tokenizer has **no EOS, no BOS and no PAD** — a contiguous `0..9999`
with no post-processor. Packing needs a document terminator and padding needs a meaningless token,
and neither exists. Adding them to the file would change its bytes and **void the tokenizer hash that
every shard manifest pins**, silently invalidating the provenance of every shard ever built with it.
So they are assigned out of vocabulary and materialised into the shard at tokenize time.

No BOS because it creates an ambiguous "which document owns position 0" case once documents are
packed, and nothing here needs one.

**Would overturn it.** Retraining the tokenizer with real special tokens — which is a Exercise 02
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

**Would overturn it.** A release of the reference implementation that imports and runs without CUDA
and without NCCL. Its licence already permits depending on it; only the hard `torch.empty(1,
device="cuda")` at module load makes a port cheaper than a dependency, and that is a fact about the
code rather than about us.

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

---

## D8 · RoPE rather than a learned position table — a data-system decision, not a modelling one.

**Decided.** `TinyGPT` (5,774,080 parameters — RMSNorm, SwiGLU, tied head) takes a **position id per
token** and rotates queries and keys by it. There is no learned position table anywhere in the model.

**Why this is not an architecture preference.** Packing hands the model positions that `pack.py`
computed, and with continuation offsets (D13) a fragment that continues a document carries its
**true** position. A 5,000-token document chopped into 512-token windows reaches position **4,999**.
A learned table sized to the window has 512 rows and cannot represent that at all; sizing it to the
longest document means knowing that length before the corpus is built, and clamping or wrapping
would corrupt exactly the continuations the offsets exist to fix — silently, since a clamped
position is a valid row. RoPE has no table, and the attention dot product it produces depends on the
**difference** between two positions, which never exceeds the window. An absolute offset of 4,999 is
therefore not a special case; it is arithmetic.

**What is *not* claimed here.** RMSNorm, SwiGLU and the tied head are conventional and are not
defended: the model exists so the ledger has something real to record, and every number it produces
is labelled with the model that produced it. Only the position scheme is forced by the data system,
and it is the only one this entry covers.

**Would overturn it.** Dropping continuation offsets — numbering every fragment from 0 — which would
remove the requirement and reintroduce the window-edge error D13 exists to prevent. Or a window as
long as the longest document, which 512 tokens does not allow.

---

## D9 · The checkpoint's ledger cut is a per-rank vector. Today its values coincide, and we say so.

**Decided.** `Checkpoint.cut` is `dict[rank, ledger length]`, with a matching `segments` map naming
which segment each length refers to. It is collected with `all_gather` and applied per rank by
`resume.plan_resume`.

**Why it must be a vector, structurally.** The cut is *applied* to R separate files, so rank 2's
number truncates rank 2's file whatever the other three hold. And what is already ragged is how much
each rank wrote **after** the checkpoint before it died — 0 / 1 / 2 / 3 discarded events in the
drill — which `plan_resume` computes per rank and which no single number can express.

**The honest bound, because the first draft of this document got it wrong.** At a synchronous
checkpoint the values *coincide*, and in the measured drill they were `{24, 24, 24, 24}`. The drill
does **not** exercise a ragged cut. Mutating `all_gather` to "copy my own number" survives it
undetected, and is killed only by `test_each_rank_is_cut_to_its_own_number`, which runs out of
process with three ranks holding 1 / 2 / 3 events. So the vector is structurally necessary and
forward-looking — per-rank selection breaks uniformity the moment a rank rejects a candidate, and so
does any rank-local retry — but it is not something this drill demonstrates.

**Would overturn it.** Nothing that keeps per-rank ledger files. A single shared ledger would make a
scalar correct, and would reintroduce the shared writer and the lock D10 exists to avoid.

---

## D10 · One event per microbatch, chain-hashed, one file per `(branch, rank, segment)`.

**Decided.** The unit of record is the **microbatch**. Each event carries `prev`, the blake2b digest
of the previous event's canonical JSON. Files are named `<branch>.rank<N>.seg<M>.jsonl` and claimed
with `O_EXCL`; a new segment is created on every process start.

**Why the microbatch and not the step.** It is the smallest thing actually fed to the model. A
per-step event cannot describe a process that died part-way through accumulation, which is precisely
the state the crash drill produces. Measured: 12 steps × 4 ranks × 3 accumulation slots = **144
events**.

**Why one file per rank, and a new segment per attempt.** Four ranks writing one file need a lock,
and a lock is a thing to be holding at the moment a process dies. No shared writer means no lock. A
fresh segment per attempt makes the crash boundary explicit — a resumed process reopening its
predecessor's file would append after a torn line — and `O_EXCL` makes two processes claiming one
segment an error rather than a silent interleave.

**Why the chain, and what it is not.** An append-only file you can edit is not append-only; altering
any line breaks every line after it. It is **not a signature**: anyone who can edit the file can
recompute every hash behind their edit. What exposes them then is `seq`, which is why the sequence
check in `verify_chain` is not redundant with the `prev` check — removing it survived thirty-one
tests before the re-chaining test existed.

**What the event carries, and what it deliberately does not.** Spans as `(shard_id, start, end)`,
addressable at any corpus size; four **separate** array hashes, so a mask bug and a token bug are
distinguishable; `sequence_length`, so the record is self-describing rather than needing the
producer's config to be read; `lane_mix` per microbatch, so a mixture claim is a sum an auditor
recomputes rather than a number it trusts. It does **not** carry document offsets: replay re-derives
those from the shard's own `EOS` positions, which is reading the data the event points at, not
recomputing the plan.

**Would overturn it.** A corpus large enough that a JSONL line per microbatch costs more than the
run it records — at which point the answer is compaction of finished segments, not a coarser unit,
because the coarser unit is the thing that cannot describe a crash.

---

## D11 · An event is written when the model is **fed**, before the optimizer steps.

**Decided.** `consume()` appends to the ledger immediately after the microbatch's `backward()`, and
the enclosing step's `optimizer.step()` happens afterwards. Both the ordinary path and the crash
path go through that one function.

**Why.** "Consumed" means *fed to the model*, not *contributed to a completed update*. A process
that dies mid-accumulation has still shown the model everything up to the point it died, and a
ledger recording only completed steps would say that never happened — losing exactly the events the
recovery has to account for. Whether that work *counts* is then a decision for resume, which owns
the checkpoint's cut, rather than for the writer.

**The cost, published rather than hidden.** After a crash the ledger over-runs the weights by
precisely the microbatches whose update never completed. The cut resolves the disagreement, and the
re-executed events carry `replayed_from` naming the discarded event each one repeats — **6** in the
drill. So "no skipped or repeated batches" is true of the **effective post-cut ledger** and false of
the **device**, and both are stated wherever the claim is made.

**Would overturn it.** Nothing available. Writing after the step would make a mid-accumulation crash
invisible, which is the one failure the record exists for.

---

## D12 · Gradients are reduced as token-weighted sums, never as a mean of means.

**Decided.** Backward on the **summed** loss; all-reduce the gradients **and** the graded-token
counts with `SUM`; divide once by the global count. `cross_entropy` returns
`(summed loss, graded count)` so the caller cannot forget the weight.

**Why.** Packing is ragged: every accumulation slot and every rank holds a different number of
*graded* tokens, because documents end where they end and each document's final token is excluded —
next-token prediction has no target for it. Averaging per-slot means and then averaging those means
weights a slot with 60 graded tokens exactly as heavily as one with 500.

**The failure has no symptom.** The loss curve looks entirely normal while the run systematically
over-weights its shortest sequences. Nothing raises, nothing diverges, and no downstream number
moves in a direction anyone would notice. Only the arithmetic exposes it, which is why it is written
down here rather than left in the code.

**What the reported loss therefore is.** `global_sum / global_count` — the mean **per graded token**
across every rank. It is not the mean of the per-rank means, and the two agree only when every rank
happens to hold the same count.

**Would overturn it.** A batch construction guaranteeing equal graded-token counts per slot, which
packing by design does not produce.

---

## D13 · Packing is concat-and-chop, and a continuation fragment carries its true offset.

**Decided.** `build_span_table` cuts every shard into fixed `sequence_length` spans with no regard
for document boundaries; `pack.build_window` then locates the boundaries from the `EOS` tokens
already in the data and hands `masks.py` per-fragment lengths **and offsets**.

**Why not a document-aware packer.** Best-fit or first-fit bin packing would raise utilisation, and
it would cost the two properties the rest of the system is built on: the span table stops being a
pure function of `(shard set, sequence_length)`, so the plan is no longer an O(1) lookup; and a span
can no longer be named by `(shard_id, start, end)`, which is the address every ledger event, every
replay and every audit uses.

**The part that is easy to get wrong.** A window usually **opens** mid-document. Numbering that
leading fragment from 0 tells the model it is the start of a document when it is not — the same
error restarting positions per document exists to prevent, reintroduced at the window edge. The
fragment therefore carries its true offset, recovered by binary search over the shard's `EOS`
positions. `DocIndex` computes those once per shard; the naive backward scan is billions of
operations at 5M tokens a shard and thousands of windows.

**Two consequences, published rather than left to be discovered.**

- **`pack_utilization` is arithmetically pinned to `1.0`, and `pad_tokens` to `0`.**
  `feed.build_microbatch` never passes `window=`, so the window *is* the span, and every span is
  exactly `sequence_length`. The field measures nothing today; it becomes live only if a window is
  ever larger than the span packed into it. A run reporting 100% packing efficiency here is
  reporting the policy, not a measurement.
- **`masks.loss_mask(context_spans=…)` has zero callers in the pipeline.** It is implemented, tested
  and taught in the notebook, and no run passes it anything: plain pretraining grades everything
  that is not padding and not a document's last token. It exists for the SFT case and is currently
  unexercised by any run.

**Would overturn it.** A curriculum stage needing whole documents in a window — prompt/response
pairs, tool traces — which needs a different packer *and* a span address that is not
`(shard_id, start, end)`.

---

## D14 · A shard's tail is dropped when it is shorter than one sequence.

**Decided.** `build_span_table` takes `count // sequence_length` whole spans per shard and discards
the remainder. `shards.split` does the opposite at its own level — the last *shard* keeps whatever
remains rather than being padded — and the asymmetry is deliberate.

**Why not pad the tail.** Padding would put tokens into the run that nothing put there, and every
count downstream — token totals, lane mix, epochs consumed — would inherit them. A padded tail is
indistinguishable from data in every aggregate the evidence bundle reports.

**Why not carry it into the next shard.** A span would then cross a shard boundary and could no
longer be named by `(shard_id, start, end)`; the ledger's address would need two shards per span,
and replay would need both to verify one microbatch.

**The cost, stated rather than implied.** At most `sequence_length - 1` = **511** tokens per shard
are never trained on — under **0.011%** of a 5,000,000-token shard. The loss is bounded by the
*shard count*, not by corpus size, which is what makes it acceptable rather than merely tolerated.

**Would overturn it.** Shards small enough for the tail to matter: the loss is per shard, so
shrinking `tokens_per_shard` raises it linearly. At 5,000 tokens a shard it would be over 10%.

---

## D15 · Digest fields are named `*_digest`, never `*_key_digest`.

**Decided.** The ledger event's and the checkpoint sidecar's field is `plan_digest`. It was
`plan_key_digest` and was renamed.

**Why this is a real constraint and not a scanner quirk.** gitleaks' `generic-api-key` rule fires on
an identifier containing *key*, *token*, *secret* or *api* sitting beside a high-entropy value.
`plan_key_digest` holding sixteen hex characters reads to the scanner exactly like a leaked
credential; `plan_digest` holding the same value does not. It is not hypothetical — a test
placeholder by that name failed CI's secret scan on PR #67.

**Why not an allowlist.** `.gitleaksignore` takes `<commit>:<path>:<rule>:<line>` fingerprints,
which silence one line of one commit and expire when it changes — right for a one-off, useless for a
field name that appears in every event of every committed ledger. The broad alternative turns the
rule off for exactly the class of finding that matters, and content digests are public by
construction: a committed ledger is full of them, and the scanner has to stay at full strength
around them.

**The name lost nothing.** The value digests a `PlanKey`, and `plan_digest` says so. The rename is
cosmetic in meaning and structural in effect, which is the cheapest kind of constraint to accept.

**Would overturn it.** A gitleaks rule that stopped matching on the identifier — which still would
not make `*_key_digest` a good name for a public content hash.

---

## D16 · The crash drill uses `os._exit`, and an exit marker is what proves it.

**Decided.** `_crash` finishes a per-rank number of that step's microbatches, waits on a barrier, and
calls `os._exit(137)`. Every worker writes an exit marker from a `finally` block.

**Why not `sys.exit`.** It raises `SystemExit`, so `finally` blocks run, `atexit` handlers run,
buffers flush and the process group is torn down politely. That is a shutdown, and a drill built on
a shutdown proves nothing about recovery. `os._exit` returns to the kernel immediately.

**Why the marker exists at all.** Swapping `os._exit(137)` for `raise SystemExit(137)` still exits
137, still leaves a truncated ledger, and passed **every other assertion in the drill** — a
surviving mutant. The marker is written from `finally`, so its *absence* is the record of an abrupt
end, and it is the only assertion that separates the two.

**Why the barrier before the exit.** It makes the drill deterministic, not gentle. Without it the
first rank to die makes the parent `SIGTERM` the survivors, and how much each wrote becomes a race —
measured at 24 / 25 / 25 / 25 events where the offsets asked for **24 / 25 / 26 / 27**. `os._exit`
after a barrier is no less abrupt than before one.

**What the drill therefore does and does not exercise.** It kills a process, not a machine. Every
`append` flushes and `fsync`s, so completed events have reached the filesystem; on macOS that
guarantees survival of *process death*, not of *power loss*, which only `F_FULLFSYNC` buys. The
drill exercises the torn-tail path and never the lost-`fsync` path, and no test in this repo can.

**Would overturn it.** Killing from outside the process — a `SIGKILL` from the parent — which is
strictly more faithful and reintroduces the race the barrier removes, so the ranks would no longer
stop where the test asks them to.

---

## D17 · Four ranks means four OS processes over `gloo`, spawned, with a file rendezvous.

**Decided.** `world_size > 1` starts real processes through `torch.multiprocessing.start_processes`
with `start_method="spawn"`, joined by `init_method="file://…"` inside the run's own directory.
Never `for rank in range(4)`.

**Why.** A loop shares one address space, one RNG, one set of file handles and one failure mode. It
cannot exercise per-rank ledger files, the cut vector, or a crash that kills one process while the
others keep running — which is every one of the failures this exercise is about. `gloo` is the CPU
collective backend and behaves identically on macOS and Linux, so the four processes are genuinely
four without a GPU anywhere.

**Two portability choices, both load-bearing for a grader on another machine.** macOS defaults to
`spawn` and Linux to `fork`; code written under `fork` inherits the parent's memory and works by
accident, and the same code under `spawn` re-imports the module and fails. Forcing `spawn`
everywhere moves that failure onto the author's machine — and is why every entry point needs an
`if __name__ == "__main__":` guard, whose absence raises an error that **never appears on Linux**. A
`tcp://` rendezvous needs a free port, and a port free on a laptop need not be free on a shared
runner; the failure is an intermittent hang, which is the worst kind to debug.

**Measured.** Four `gloo` ranks end a run **bit-identical** — `weight_digest` equal on all four,
recorded per rank by `write_telemetry`. A rank stepping on its own unreduced gradient diverges from
step one with no error and a perfectly healthy loss curve, because each rank is minimising its own
slice correctly; the digest is the only place that becomes visible.

**The cost, stated.** `gloo` binds a loopback socket, so these tests skip in a sandbox that blocks
local networking, and CI installs no torch at all. The multi-rank claims are verified locally before
a PR and nowhere else, which is a limit of the evidence and not of the design.

**Would overturn it.** NCCL and real GPUs, which change the backend and nothing else about the
structure — the code is identical either way, which is the reason `gloo` was chosen.

---

## D18 · The Boltzmann temperature is a multiple of the score spread, not an absolute.

**Decided.** `opus.select` scales scores by `τ · std(contested scores)` before adding Gumbel noise.
`Config.opus_temperature` moves from `0.9` to `0.25`, and the field's meaning changes with it.

**Why.** Gumbel(0, 1) noise has a **fixed** standard deviation of `π/√6 ≈ 1.283`. A utility does
not: it is an inner product of gradients, so it shrinks as the model improves — the instructor's own
control-pool diagnostic shows mean ‖g‖ falling 3.07 → 0.74 over a run. Under an absolute
temperature the ratio between the two drifts, and the selector slides from utility-driven toward
random **over the course of the run with nothing failing**. Batches keep filling, loss keeps
falling, and the audit trail records confident-looking scores beside decisions those scores did not
make.

**Measured, at the old default.** At `τ = 0.9` the noise carried **1.09×** the spread of the signal
it perturbed, and 29 of 32 non-selected candidates flipped under resampling. At `τ = 2.0`, **zero**
rejections survive a redraw. The sweep — `τ` 0.05 / 0.25 / 0.9 / 2.0 → noise-to-signal 0.06 / 0.32 /
1.09 / 2.43 — is in `opus.select`'s docstring and the README.

**Proven scale-free.** Multiply every score by a thousand: identical served set, identical
`noise_dominance`. That is the property an absolute temperature does not have.

**Would overturn it.** A source showing the paper's temperature is already normalised, in which
case this is the paper's design rather than ours and D18 becomes a note rather than a decision.

---

## D19 · `redundancy_weight` exists, and defaults to Eq. 23 unmodified.

**Decided.** Ship the criterion exactly as published (`λ = 1.0`), publish `redundancy_share` on
every pass, and provide `λ` for anyone who wants the diversity term to do something.

**Why not simply rebalance it.** The plan flagged the imbalance as *"a trap to inspect before
letting one term subtract the other"*, and inspecting it confirmed the trap: at `η = 3e-4` the
penalty contributes **0.069%** of the score. Sweeping without gaps — 3e-4 → 0.069%, 1e-3 → 0.27%,
1e-2 → 1.97%, 1e-1 → 24.7%, 1.0 → 85.1% — and with `η` stripped out the penalty's raw inner product
is **4.05× larger** than the alignment's. So nothing is cancelling; one factor of `η` is.

**Two branches, and we do not pick one.** Either the penalty is genuinely inert at any learning
rate a pretraining run uses, or `η` in Eq. 23 is not the raw learning rate and our reading of it is
wrong. **The measurement is certain; the interpretation is not.** Silently rebalancing a published
criterion would hide the question, and silently shipping it without the number would let a reader
believe a diversity term was working. So: faithful by default, measured on every pass, and both
branches stated wherever the number appears.

**Would overturn it.** The paper's definition of `η`, read directly. That settles which branch this
is, and D19 collapses to one sentence either way.

---

## D20 · The held-out split is written to disk, not merely counted.

**Decided.** `corpus.build_lane` materialises the withheld tokens as shards under a `heldout` lane
with `split="heldout"`, which `manifest.admit` refuses.

**Why this is a decision and not a bug fix.** It was both. `heldout_tokens` was computed, stored on
`LaneBuild`, summed into the tracked build report and published — **1,093,019 tokens** — while the
array itself went out of scope one line later. A tenth of the corpus was reported as withheld for
evaluation and existed nowhere. No test failed, because every test asked about the number.

It surfaced when OPUS needed a proxy set: `g_proxy` must come from text the run never trains on, or
selection tunes toward what the model is already being pushed toward. The selector asked for the
held-out split and found an empty lane.

**`split="heldout"`, not `"eval"`, and no `benchmark_ids`.** It is a reference sample, not a
benchmark. Tagging it as one would make the firewall's benchmark clause fire for something that
overlaps no benchmark, which is a true refusal for a false reason.

**Would overturn it.** Nothing. A count with no data behind it is a claim nothing can check.

