# CLAUDE.md — 06-build-training-dataset

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is the generated
`submission_artifacts/` bundle, the reasoning is `DECISIONS.md`, the running log is `PROGRESS.md`,
and `BRIEF.md` is the assignment (local only, gitignored).

**Status: stage 6 of 8.** `spec.py`, `config.py`, `shards.py`, `manifest.py`, `firewall.py`,
`plan.py`, `masks.py`, `pack.py`, `feed.py`, `ledger.py`, `model.py`, `train.py` and `runner.py`
exist. Crash recovery, resume, replay, fork and the audit do **not**. Do not describe this exercise
as having them until it does — the README carries a stage table for exactly this reason, and
`tests/test_trainingdata_docs.py` now asserts the header agrees with it.

## The rules this exercise adds

- **Replay reads the ledger. It never recomputes.** This is the session's whole thesis, and the
  reason is not stylistic: once OPUS scores depend on the current checkpoint, the plan stops being a
  pure function of position, so re-deriving it can never be bit-identical. `replay.py` must keep an
  import closure with **no torch and no model code**, and a test will assert that. Measured on the
  prototype: change one planner line and recompute → 96/96 slots differ; read the ledger → 96/96
  match.

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

- **`spawn`, always, and a file rendezvous, never a port.** macOS defaults to `spawn` and Linux to
  `fork`; code written under `fork` works by accident and fails on a grader's machine. A TCP port
  that is free on a laptop may not be on a shared runner, and the failure is an intermittent hang.

- **The torch tests skip in CI and that is stated, not hidden.** CI runs `uv sync --all-packages`
  with no extras, so `model`, `train` and `runner` are verified locally before each PR and nowhere
  else. `gloo` additionally needs loopback, so the multi-rank tests skip again inside a sandbox that
  blocks it — `_loopback_available()` probes for that rather than letting it hang.

## Naming

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
