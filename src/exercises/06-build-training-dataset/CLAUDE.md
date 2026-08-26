# CLAUDE.md — 06-build-training-dataset

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is the generated
`submission_artifacts/` bundle, the reasoning is `DECISIONS.md`, the running log is `PROGRESS.md`,
and `BRIEF.md` is the assignment (local only, gitignored).

**Status: all eight stages done.** Shipped: `spec.py`, `config.py`, `shards.py`,
`manifest.py`, `firewall.py`, `plan.py`, `masks.py`, `pack.py`, `feed.py`, `ledger.py`, `model.py`,
`train.py`, `runner.py`, `checkpoint.py`, `resume.py`, `replay.py`, `mixture.py`, `corpus.py`,
`fork.py`, `metrics.py`, `evidence.py`, `opus.py` and `opus_score.py`, plus `run_demo.py` and
`verify.py` at the exercise root
and `tools/fetch_corpus.py` + `tools/build_corpus.py`, both **tracked** (unlike the notebook
builder). `results/` is tracked and documents render `corpus_build.json` from it.

**Not shipped, and do not describe the exercise as having them:** any `web/` bundle.

That sentence is now checked. `test_the_not_shipped_paragraph_names_nothing_that_exists` reads the
paragraph above and fails if anything it denies is on disk — because it is the sentence that went
stale, and it went stale in the file whose *next* paragraph warns it would. It denied fork, the
auditor, the demo runner, the metrics module, the evidence writer, the corpus fetcher and a tracked
`results/` while all seven were built. An agent reading it would have rebuilt finished work, or
reported a delivered artefact as missing.

**Stage 7 is proven, not asserted.** Golden run 144 events; ranks stopped at 24/25/26/27; resumed
from `ckpt-main-000007`; 6 microbatches re-executed; every `(step, rank, accum, flat,
microbatch_hash)` after resume equals the golden run.

**Stage 8 landed.** Replay: 32/32 events re-derived, one flipped shard bit turns exactly 1 red.
Fork: lineage recorded rather than inferred. Auditor: 20 of 22 checks, and the 2 failures are it
refusing to bless the OPUS gap.

`tests/test_trainingdata_docs.py` guards **README.md's** status line against **README.md's** stage
table, asserts every module is named in both README.md and CLAUDE.md, and — since this header went
stale twice — asserts the not-shipped paragraph above names nothing that exists. What it still does
**not** read is the rest of this header: the stage claims and their numbers are hand-maintained, and
the first time one of them went stale it denied `replay.py` existed while the module carried 340
lines and 14 tests.

## The rules this exercise adds

- **Replay reads the ledger. It never recomputes.** This is the session's whole thesis, and the
  reason is not stylistic: once OPUS scores depend on the current checkpoint, the plan stops being a
  pure function of position, so re-deriving it can never be bit-identical. `replay.py` keeps an
  import closure with **no torch and no planner**, and `test_replay_cannot_reach_the_planner_or_torch`
  asserts it transitively — with a twin (`test_the_closure_check_would_notice_a_new_import`) that
  goes red when the walker stops seeing new imports. Measured on the prototype: change one planner
  line and recompute → 96/96 slots differ; read the ledger → 96/96 match. Measured on the shipped
  module: 32/32 microbatches re-derived over a recorded interval, and one flipped shard bit turns
  exactly 1 of the 32 red.

- **The data system is torch-free. Only training imports torch.** Shards, manifests, packing, masks,
  the schedule, the ledgers, replay, fork and audit are pure Python and numpy. CI runs
  `uv sync --all-packages` with **no extras**, so this boundary is what lets CI verify almost the
  whole system. Adding a module-level `import torch` outside `train`/`model`/`opus_score` silently
  removes it from CI's reach.

- **`spec.py` is shared with the auditor. Shared facts, never shared logic.** `verify.py` re-derives
  every published claim from artifacts alone; if it imported the producer it would inherit the
  producer's bugs and agree with itself, which is the hardcoded evidence the assignment refuses.
  `tests/test_trainingdata_spec.py` parses `spec.py`'s AST — not `sys.modules`, because an import
  that only fires at call time would not show up there.

- **The sentinels are out of vocabulary, and the tokenizer file is never edited.** The frozen
  Session 2 vocabulary is a contiguous `0..9999` with no EOS, BOS or PAD. Editing the file to add
  them would change its bytes and **void the tokenizer hash every shard manifest pins**. So
  `EOS=10000`, `PAD=10001`, model vocab `10_002`. Checked against the real file, never a remembered
  number.

- **A shard is immutable and content-addressed.** Its id *is* its hash. Modify it and it becomes a
  different shard with a different lineage — never a mutated one. Ledger events record
  `shard_content_hash` and replay re-verifies before reading, so a tampered shard turns exactly the
  batches that used it red.

- **The bundle is `submission_artifacts/`, and it cannot be called `artifacts/`.** `**/artifacts/`
  is a **directory** pattern and git will not re-include a file whose parent is excluded — a
  negation there is inert while `git add -A` reports success. `run.log` is trackable only because
  `*.log` is a **file** pattern. `tests/test_submission_bundle.py` pins both halves; do not "tidy"
  either.

- **The cross-document attention leak has no symptom.** If document B can attend to document A
  nothing crashes and the loss curve looks fine; the model just learns that unrelated text is a
  natural continuation. So the mask invariants are asserted on the **mask itself**, never on a
  downstream number, and `masks.py` stays numpy-only so CI can run them without torch.

- **Position ids restart per document, and that is not cosmetic.** Continuous positions would tell
  the model a document beginning at offset 400 is 400 tokens into something — it is not, and at
  inference it will be at 0. Restarting is what makes packing invisible to the model.

- **The additive mask uses a large finite negative, never `-inf`.** A fully-masked row of `-inf`
  becomes `nan` after softmax, and one `nan` poisons every gradient it touches.

- **Disjointness is asserted on DATA, never on coordinates.** A coordinate bijection is arithmetic
  and says nothing about which tokens a rank reads — two ranks can hold different coordinates that
  point at the same span. `test_no_two_ranks_touch_overlapping_spans_in_a_step` is the only version
  of that claim that could ever fail; keep it that way.

- **`PlanKey.planner_version` is bumped by hand when the planning algorithm changes.** Without it a
  code change silently produces a different plan under an unchanged key, and the ledger becomes the
  only evidence anything moved.

- **The firewall is two-sided on purpose, and both sides must stay.** The manifest carries the
  split *and* the registry is asked independently. The instructor's reason: *"who knows maybe a
  mistake in copying or something may still happen."* Removing either side leaves a single point of
  failure for the one mistake that makes every benchmark score fiction.

- **The firewall stores no evaluation text, ever.** Benchmark items are 8-byte truncated digests of
  13-word shingles. A test greps the written registry for benchmark words; keep it that way.

- **Two of the four OPUS statuses are ours, and the docs must keep saying so.** `accept` and
  `reject` are the selector's. `defer` and `floor_override` appear in **none** of the OPUS paper,
  its reference implementation, or LightningLM — all three were searched. See `DECISIONS.md` D5.

- **The lecture's description of OPUS is wrong; build from the paper.** The transcript describes a
  weight *mask*. There is no weight mask in either implementation — it is a continuous
  preconditioned gradient inner product, minus a redundancy penalty the lecture never mentions.
  `DECISIONS.md` D7.

- **The ledger is a chain, and that is a bounded claim.** Each event carries the previous event's
  hash, so tampering can never be *local*. It is not a signature: anyone who can edit the file can
  recompute every hash after their edit. What exposes them then is `seq`, which is why the sequence
  check in `verify_chain` is **not** redundant with the `prev` check — a mutation removing it
  survived thirty-one tests before the re-chaining test existed.

- **`append` refuses to run before `open`.** `open` claims the segment with `O_EXCL`; `append` uses
  `"a"` mode, which would happily create the file and bypass that claim. Two processes interleaving
  into one segment produce a record no one can read.

- **Only the LAST line of a segment may be repaired.** A torn tail is an interrupted write; an
  unparseable line anywhere earlier is corruption, and repairing it would hide real damage behind a
  routine crash-recovery path. `drop_torn_tail` validates the earlier lines *first* — an early
  return on a healthy tail let mid-file damage read as "nothing to repair".

- **A window usually OPENS mid-document, and numbering it from 0 is the silent error.**
  Concat-and-chop cuts every `sequence_length` tokens with no regard for documents. `pack.py` gives
  a continuation fragment its true offset; `masks.position_ids` takes those offsets. This is why the
  model uses **RoPE and not a learned position table**: a 5,000-token document reaches position
  4,999, which no table sized to the window can hold, and clamping would corrupt exactly the
  continuations the offsets exist to fix.

- **The leak runs from the LATER document to the earlier one.** Causality already stops document A
  from seeing document B, so a test that checks A's logits passes even with the block-diagonal mask
  replaced by plain `is_causal=True`. Assert on B. Both directions are tested; only one can fail.

- **`fork_rng` around module construction is load-bearing.** `nn.Linear.__init__` calls
  `reset_parameters`, which draws from the **global** RNG before our explicit generator runs. Every
  value is overwritten a moment later, so it is invisible in the model and surfaces as the next
  `torch.randn` in the process returning different numbers because a model was built.

- **The reduction is built from sums, never from means of means.** Packing is ragged, so
  accumulation slots and ranks have different graded-token counts. Backward on the summed loss,
  all-reduce the gradients and the counts, divide **once**. Averaging per-slot averages weights a
  60-token slot as heavily as a 500-token one, and the loss curve looks entirely normal.

- **Ranks must end every step bit-identical.** Four gloo ranks were measured to do so. A rank that
  steps on its own unreduced gradient diverges immediately, with no error and a healthy-looking
  loss — the run is then four different models and the checkpoint is whichever one rank 0 held.
  `write_telemetry` records a weight digest per rank so that is checkable rather than assumed.

- **Pin gloo to loopback, and never assume the hostname resolves somewhere sane.** gloo picks its
  interface by resolving the machine's hostname. On a laptop that is a WiFi address, so four
  processes on ONE machine talk over WiFi — and when the network moves, `init_process_group` dies
  with `uv_accept: invalid argument` and `SIGABRT`, naming nothing to do with networking. The same
  drill passed an hour earlier on identical code, which is the signature of an environmental
  dependency rather than a bug. `runner.pin_gloo_to_loopback()` sets `GLOO_SOCKET_IFNAME` with
  `setdefault` semantics, so a real multi-node run can still name its own interface. **Anything
  that creates a process group must call it** — including the out-of-process test probe, which
  failed for exactly this reason until it did.

- **`spawn`, always, and a file rendezvous, never a port.** macOS defaults to `spawn` and Linux to
  `fork`; code written under `fork` works by accident and fails on a grader's machine. A TCP port
  that is free on a laptop may not be on a shared runner, and the failure is an intermittent hang.

- **The torch tests run in CI now, in their own job, and the reason that took a rewrite matters.**
  A module-level `importorskip` skips an ENTIRE file, and a file that collects nothing is
  indistinguishable from a file with nothing in it — so 46 tests and all 20 integration tests ran
  nowhere while every gate reported green. The `train` job installs `--extra train` with CPU-only
  wheels and runs exactly the gated files. Do not delete it and do not drop the extra:
  `tests/test_ci_shards_cover_everything.py` fails on either. `gloo` still needs loopback, so the
  multi-rank tests skip in a sandbox that blocks it.

- **The cut is a vector, and be precise about why.** Today its four values *coincide*, because a
  synchronous checkpoint lands every rank on the same event count. What is already ragged is how
  much each rank wrote **after** the checkpoint before it died, which `resume.plan_resume` computes
  per rank. A scalar would be correct only while every rank writes the same number of events per
  step — per-rank selection breaks that the moment a rank rejects a candidate. Do not "simplify" it
  to a scalar, and do not claim the drill exercises the non-uniform case: it does not, and
  `test_each_rank_is_cut_to_its_own_number` covers that directly.

- **The sidecar is the commit.** `<id>.pt` is renamed into place first, `<id>.json` last, so an
  interrupted save leaves tensors with no sidecar and reads as *absent*. Every write is
  rename-into-place: a reader holding the old file must keep seeing the old file, whole.

- **`latest` orders by step, never by id.** Ids pad to six digits, so below a million steps lexical
  order agrees and a `max` over ids looks right. At a million they diverge —
  `"ckpt-main-1000000"` sorts before `"ckpt-main-999999"` — and a resume would silently redo a
  thousand steps.

- **`os._exit`, never `sys.exit`, and the marker proves which was used.** `sys.exit` raises
  `SystemExit`, so `finally` runs, `atexit` runs and buffers flush; that is a shutdown, and a drill
  built on it proves nothing. Each worker writes an exit marker from `finally`, so its **absence**
  is the record of an abrupt end — swapping the drill to `SystemExit` still exits 137 and still
  leaves a truncated ledger, and only that marker notices.

- **The barrier before the crash makes the drill deterministic, not gentle.** Without it the first
  rank to die makes the parent `SIGTERM` the survivors, and how much each wrote becomes a race —
  measured at 24/25/25/25 events where the offsets asked for 24/25/26/27.

- **Resume checks every rank before cutting any of them.** Half-applied is worse than the crash: a
  run whose ranks disagree about which checkpoint they belong to. `apply_cut` runs a dry pass first.

- **Repair the torn tail before measuring the cut, and use one scan for both.** A torn line is not
  an event; counting it puts the cut one event too far and under-reports what was re-executed.

- **What the resume claim covers, exactly.** *Inputs*: `(step, rank, accum, flat,
  microbatch_hash)` match a run that never crashed. **Not** losses and **not** weights — those move
  with thread count and library version. And "the next batch" means the batch after the
  *checkpoint*, not after the crash; the microbatches between them are re-executed, carry
  `replayed_from` naming the discarded event each repeats, and the count is published.

- **Replay must read the POLICY out of the event, not hardcode the reconstruction.** This is the
  half of "run the ledger, don't calculate it" that `replay.py` has not finished. `rebuild()`
  correctly reads the spans, the shard ids and `sequence_length` from the event — and then calls
  `masks.segment_ids`, `masks.position_ids` and `masks.loss_mask(segments[row], tokens[row])`
  hardcoded, ignoring the three policy fields the event carries for exactly this purpose:
  `attention_policy`, `position_policy`, `pack_policy`. `grep -n "_policy" replay.py` returns
  nothing. Today every run writes the same three strings so nothing diverges; the failure appears
  the first time a run masks a prompt as context or changes the window rule, and it appears in the
  **worst possible disguise** — `loss_mask_hash` mismatches, the verdict goes red, and the report
  blames the shard when the shard is fine. Either dispatch on the recorded policy, or assert it
  equals the single value this build implements and fail loudly with "this replay cannot rebuild
  policy X" — never re-derive silently under an assumption the event contradicts.
  There is also **no `loss_policy` field**, so `context_spans` masking is not recordable at all: add
  the field before adding the caller, or replay is unfixable by construction. And nothing validates
  the strings — `tests/test_trainingdata_ledger.py:39` writes `"restart-per-document"` where
  `train.py:210` writes `"restart-per-document-continue-across-window"`. **A policy string nothing
  reads and nothing validates is a label, not a contract.**

- **`masks.loss_mask(context_spans=...)` has zero callers, and the docs must stop implying
  otherwise.** It is implemented, tested (`test_context_spans_are_excluded_from_loss`) and taught in
  the notebook; `feed.build_microbatch` never passes it. So the SFT-prompt / tool-observation
  masking this exercise explains is a **capability, not a behaviour of the run** — the shipped run
  grades every non-pad token that is not a document's last. The agentic lane is the one that makes
  this concrete, and it is exactly the lane the corpus under-feeds. Either wire it through `feed.py`
  *and* record the spans in the event, or say plainly in the README which of the two it is.

- **`pack_util` in the telemetry is arithmetic, not a measurement — it is always `1.0`.**
  `feed.py:207` calls `pack.build_window(handle.index, handle.tokens, span.start, span.end)` with no
  `window=`, so `size = end - start`, `packed[: end - start]` fills the array end to end, no `PAD`
  is written, and `masks.utilization` can only return 1.0. `train.py:206` records it per microbatch
  and `ledger.ConsumeEvent` carries it into the committed bundle, where a reader will take it as
  evidence the packer is efficient. Fix it by passing `window=cfg.sequence_length` so a short tail
  can show, or delete the field — do not leave a constant in the ledger dressed as a statistic.

- **The corpus is 4.8 epochs, and the mixture claim does not survive that unstated.** The run
  consumes `Config.total_tokens` = 10,485,760 positions; the corpus on disk holds 2,185,575 tokens.
  Shaped to session 5's weights that is **30.2 epochs of web against 0.41 of agentic** — the
  heaviest-funded lane memorised thirty times over, the lightest never read through once. Nothing
  fails; shards read fine and the loss curve looks normal. Print the per-lane epoch count next to
  any mixture-compliance figure this exercise publishes, and treat `mixture_compliance` in
  `spec.REQUIREMENTS` as unmet until it does. `data/proxy/manifest.json` funds four lanes
  (`stem`, `reasoning`, `agentic`, `stem-alt`) and this exercise ships **no fetcher of its own** —
  `tools/` holds only the notebook builder.

- **Three of the deliverables are PUBLIC URLS, not repo files.** The platform totals **1,150** =
  1,000 rubric (the repo link) + 3 × 50 for `run.log`, `evidence.json` and `evidence.md` published
  at **three separate public URLs**. `BRIEF.md` truncates before those fields, so the brief is not
  the authority here — the platform's field list is. `tests/test_submission_bundle.py` proves the
  three files are *committable*; committing them is not publishing them, and the exercise is not
  done until all three resolve.

- **`verify.py` may import `spec` and nothing else, and a test asserts it transitively.** One
  convenient `from trainingdata import metrics` turns every number check into the producer's
  arithmetic checked against the producer's arithmetic — agreeing with itself whatever either got
  wrong — and **the output looks identical**. That is why it is a test rather than a rule. The
  chain hash is re-implemented in `verify.py` with `hashlib` for the same reason: a check that
  called `ledger._digest` would confirm only that it is deterministic.

- **`run_demo.py` never logs an event it did not produce.** Two of the thirteen — `OPUS decisions
  recorded` and `audit completed` — are written `[SKIP]` with the reason, and the auditor reports
  them as NOT PRODUCED. A verdict per line is what lets a reader tell "the run did not do this"
  from "the run did not mention it"; only the second is a hole in the record.

- **A short demo cannot speak for the mixture, and the evidence row says which it measured.** No
  lane's share divides evenly into a 64-sequence step, so a run covering 1.2% of the plan drifts by
  up to 2.1 points. The row reports its coverage, the sample drift, and separately whether the
  *corpus* is compliant — the thing a short run cannot tell you about.

- **A fork inherits its parent's history; it does not copy it.** So `common_prefix` between parent
  and child is legitimately **zero**, which reads as a failure and is the opposite of one. Use
  `fork.verify_fork`: the parent covers the shared steps, the child begins after them, and the
  child re-ran none of them.

## Naming

- **Two shipped helpers currently have no caller, and that is worth knowing before trusting them.**
  `masks.loss_mask(context_spans=...)` is implemented, tested (`test_context_spans_are_excluded_from_loss`)
  and taught in the notebook, but **nothing in the pipeline passes `context_spans`** — the agentic
  loss-mask argument is demonstrated, not exercised. And `pack_utilization` is arithmetically pinned
  to `1.0` because `feed.py` never passes `window=`, so the `pack_util` field `train.py` writes into
  every ledger event is a constant, not a measurement. Neither is a bug; both are claims the evidence
  bundle must not overstate.


Test modules are prefixed `test_trainingdata_*`. pytest imports test modules by **basename**, so a
second `test_config.py` anywhere in the repo aborts *collection* rather than failing a test.
`tests/test_module_names.py` enforces this repo-wide.

## Running it

```bash
uv sync --all-packages                                   # the data system
uv sync --all-packages --extra train                     # ...plus torch, for the training step
uv run pytest src/exercises/06-build-training-dataset    # unit + integration
uv run pytest src/exercises/06-build-training-dataset -m "not integration"
```

Heavy output goes to gitignored `artifacts/`; the tracked bundle is capped at 2 MiB and the cap is
checked **before** a run writes, not after.
