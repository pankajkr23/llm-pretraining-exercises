# CLAUDE.md — 06-build-training-dataset

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is the generated
`submission_artifacts/` bundle, the reasoning is `DECISIONS.md`, the running log is `PROGRESS.md`,
and `BRIEF.md` is the assignment (local only, gitignored).

**Status: stage 3 of 8.** `config.py`, `spec.py`, `shards.py`, `manifest.py` and `firewall.py` exist. Do not
describe this exercise as having packing, ledgers or replay until it does — the README carries a
stage table for exactly this reason.

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

## Naming

Test modules are prefixed `test_trainingdata_*`. pytest imports test modules by **basename**, so a
second `test_config.py` anywhere in the repo aborts *collection* rather than failing a test.
`tests/test_module_names.py` enforces this repo-wide.

## Running it

```bash
uv sync --all-packages                                   # the data system
uv sync --all-packages --extra train                     # ...plus torch, for the training step
uv run pytest src/exercises/06-build-training-dataset
```

Heavy output goes to gitignored `artifacts/`; the tracked bundle is capped at 2 MiB and the cap is
checked **before** a run writes, not after.
