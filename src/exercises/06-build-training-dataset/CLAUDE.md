# CLAUDE.md — 06-build-training-dataset

Component notes. Repo-wide conventions: root `AGENTS.md`. The deliverable is the generated
`submission_artifacts/` bundle, the reasoning is `DECISIONS.md`, the running log is `PROGRESS.md`,
and `BRIEF.md` is the assignment (local only, gitignored).

**Status: all eight stages done.** Shipped: `spec.py`, `config.py`, `shards.py`,
`manifest.py`, `firewall.py`, `plan.py`, `masks.py`, `pack.py`, `feed.py`, `ledger.py`, `model.py`,
`train.py`, `runner.py`, `checkpoint.py`, `resume.py`, `replay.py`, `mixture.py`, `corpus.py`,
`fork.py`, `metrics.py`, `evidence.py`, `opus.py` and `opus_score.py`, plus `run_demo.py` and
`verify.py` at the exercise root, `tools/fetch_corpus.py`, `tools/build_corpus.py` and
`tools/build_web_data.py`, all **tracked** (unlike the notebook builder), and a deployed `web/`
explainer. `results/` is tracked and documents render `corpus_build.json` from it.

**Nothing is outstanding**, so there is no "not shipped" list here any more — and its absence is
deliberate. That sentence was wrong twice.
`test_the_not_shipped_paragraph_names_nothing_that_exists` catches it when there is one: the first
time it denied fork, the auditor, the demo runner, the metrics module, the evidence writer, the
corpus fetcher and a tracked `results/` while all seven were built — in the file whose *next*
paragraph warns it would go stale. The second time it denied the `web/` bundle **while that bundle
was live in production**, and the guard missed it because it only matched `*.py` names: a claim
about a *directory* was invisible to it. It now checks both.

**Stage 7 is proven, not asserted.** Golden run 144 events; ranks stopped at 24/25/26/27; resumed
from `ckpt-main-000007`; 6 microbatches re-executed; every `(step, rank, accum, flat,
microbatch_hash)` after resume equals the golden run.

**Stage 8 landed.** Replay: 32/32 events re-derived, one flipped shard bit turns exactly 1 red.
Fork: lineage recorded rather than inferred. OPUS: 128 candidates over 4 passes, each with a score,
a rank, an outcome and a reason. Auditor: **40 of 40 checks**, re-derived from the bundle alone.

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
  split *and* the registry is asked independently. The stated reason is that a copying slip
  or a missed registration is always possible. Removing either side leaves a single point of
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

- **Replay reads the POLICY out of the event, and refuses one it cannot rebuild.** This is the half
  of "run the ledger, don't calculate it" that is easiest to leave half-done, and it shipped:
  `replay.rebuild` compares `pack_policy`, `position_policy` and `attention_policy` against the
  single values this build implements and raises rather than reconstructing under an assumption the
  event contradicts. `loss_policy` is checked the same way, and a `context-masked` event that
  records no spans is refused outright.

  **Refusing is the whole point, because the alternative failure wears a disguise.** Rebuild the
  wrong window silently and `loss_mask_hash` mismatches — which is the signal reserved for *a shard
  whose bytes moved*, so the report blames the shard when the shard is fine.

  Keep it that way when a policy is added: either dispatch on the recorded value, or extend the
  refusal. Never re-derive silently. `spec.py` owns the four vocabularies
  (`PACK_POLICIES`, `POSITION_POLICIES`, `ATTENTION_POLICIES`, `LOSS_POLICIES`) so the producer, the
  auditor and this guard cannot drift apart.

- **`masks.loss_mask(context_spans=...)` is wired end to end — and it took a caller to make the
  documents true.** It was implemented, tested and taught in the notebook with **zero callers**, so
  every document describing SFT-prompt masking described a capability while the run graded every
  non-pad token. The spans now travel shard manifest → `ShardHandle` → `pack.build_window`, which
  **clips and translates** them into window coordinates → `masks.loss_mask` → the ledger event →
  replay.

  Two details that are load-bearing. The translation matters: handing a shard-relative range
  straight through would mask the wrong positions, and on a window from the middle of a shard would
  usually mask nothing and look like it worked. And the policy is **derived from what the microbatch
  did**, never declared — a run that says `context-masked` and masked nothing is claiming a
  behaviour it lacked, which is exactly how the feature sat unused while the docs said otherwise.

- **`pack_util` in the telemetry is arithmetic, not a measurement — it is always `1.0`.**
  `feed.py:286` calls `pack.build_window(...)` with `context_spans=` but **no** `window=`, so `size = end - start`, `packed[: end - start]` fills the array end to end, no `PAD`
  is written, and `masks.utilization` can only return 1.0. `train.py:206` records it per microbatch
  and `ledger.ConsumeEvent` carries it into the committed bundle, where a reader will take it as
  evidence the packer is efficient. Fix it by passing `window=cfg.sequence_length` so a short tail
  can show, or delete the field — do not leave a constant in the ledger dressed as a statistic.

- **The corpus is sized against the RUN, and that took a refetch.** It was once 2,185,575 tokens
  against `Config.total_tokens` = 10,485,760 positions — **4.8 epochs**, and shaped to session 5's
  weights, **30.2 epochs of web against 0.41 of agentic**: the heaviest-funded lane memorised thirty
  times over, the lightest never read through once. Nothing failed; the shards read fine and the
  loss curve looked normal. It is now **10,649,549 training tokens at 1.01 epochs**, every lane
  compliant and both floors held, plus **1,093,019 held-out tokens** written as `split="heldout"`
  shards the firewall refuses. `tools/fetch_corpus.py` and `tools/build_corpus.py` are tracked, and
  the build refuses outright below one epoch.

  **Still print the per-lane epoch count next to any mixture figure.** The failure it guards against
  is silent by construction: a lane above ~1 epoch is measuring memorisation and a lane below 1.0
  was never fully read, and neither shows up in a loss curve.

- **The platform has FOUR fields, and three of them are direct links to files in this repo.** Read
  from the platform's own submission page, not from `BRIEF.md`, which truncates at the words *"Your
  submission"* and never lists them:

  | field | points | what goes in it |
  | --- | ---: | --- |
  | Github Repo Link | 1000 | the repository URL |
  | Github `run.log` link | 50 | `.../blob/main/src/exercises/06-build-training-dataset/submission_artifacts/run.log` |
  | Github `evidence.json` link | 50 | the same path, `evidence.json` |
  | Github `evidence.md` link | 50 | the same path, `evidence.md` |

  Each field asks you to confirm the link resolves for a logged-out stranger. So **"public" is a property of the repository, not a demand for
  separate hosting** — the repo is already public, and the three files are already tracked. An
  earlier version of this note called them *"PUBLIC URLS, not repo files"* and treated hosting them
  elsewhere as outstanding work. That was wrong, and it was wrong in the expensive direction:
  invented work on a deliverable that was nearly finished.

  What is genuinely required is that the paths **resolve on `main`**, and as of v0.9.0 all four do
  — tested anonymously, HTTP 200. `tests/test_submission_bundle.py` proves the files are
  committable; committing them is not merging them, so re-check the links after any change that
  moves or regenerates the bundle.

- **`verify.py` may import `spec` and nothing else, and a test asserts it transitively.** One
  convenient `from trainingdata import metrics` turns every number check into the producer's
  arithmetic checked against the producer's arithmetic — agreeing with itself whatever either got
  wrong — and **the output looks identical**. That is why it is a test rather than a rule. The
  chain hash is re-implemented in `verify.py` with `hashlib` for the same reason: a check that
  called `ledger._digest` would confirm only that it is deterministic.

- **`run_demo.py` never logs an event it did not produce.** Exactly one of the thirteen —
  `audit completed` — is written `[SKIP]`, and that one is structural rather than a gap: a run that
  certifies its own audit certifies nothing, so `verify.py` produces it. `OPUS decisions recorded`
  was the other `[SKIP]` until OPUS shipped, and it now reads `[PASS]`. A verdict per line is what lets a reader tell "the run did not do this"
  from "the run did not mention it"; only the second is a hole in the record.

- **A short demo cannot speak for the mixture, and the evidence row says which it measured.** No
  lane's share divides evenly into a 64-sequence step, so a run covering 1.9% of the plan drifts by
  up to 2.3 points. The row reports its coverage, the sample drift, and separately whether the
  *corpus* is compliant — the thing a short run cannot tell you about.

- **A fork inherits its parent's history; it does not copy it.** So `common_prefix` between parent
  and child is legitimately **zero**, which reads as a failure and is the opposite of one. Use
  `fork.verify_fork`: the parent covers the shared steps, the child begins after them, and the
  child re-ran none of them.

## The page carries the spine, and two of its sections are not optional

`web/chapters.js` builds the twelve-part narrative `AGENTS.md` requires, declared as `data-role` on
each section. The three explainer chapters (`replay`, `floors`, `chain`) are the `results` block and
are unchanged; everything around them is prose built by `section(id, role, …)`.

- **Roles are literal strings at the construction site, never looked up from a map.**
  `tests/test_page_spine.py` reads this file's *source*, so `sec.dataset.role = ROLES[sec.id]` is
  invisible to it and the guard would go green on a page with no spine at all. That is why the three
  chapters are wrapped in one-line arrow functions in `CHAPTERS` rather than tagged in a loop.
- **The glossary section and the hover tooltips render the same `GLOSSARY` object.** Do not write a
  second set of definitions. The section exists because hover is absent on a touch screen, absent in
  print, and absent for a keyboard reader — the same "drawer a reader has to open" that `AGENTS.md`
  rules out for anything load-bearing. Its heading's count is **derived from the list it heads**, and
  `test_the_glossary_heading_counts_the_terms_it_actually_shows` is what stops someone typing it back
  in.
- **The mechanism figure is the pipeline, and its two accented boxes are marked by an explicit
  `key` class.** Never by `:nth-child` — the arrows are siblings of the boxes, so any positional rule
  counts them too and selects the wrong stages the moment one is added.
- **The limits are a section, not a footer paragraph.** They used to be the last thing in
  `buildFooter`. A caveat a reader reaches only by finishing the page is a caveat the page is hiding,
  which is the same rule that keeps them out of a collapsed `<details>`.
- **One opening tile is a failure on purpose** — three of four selector passes were offered no
  agentic candidate. `AGENTS.md`: a page that shows only its wins has not earned the ones it shows.
  If you replace it, replace it with another honest one.

Two render tests were rescoped when this landed and the distinction matters:
`test_every_chapter_built` now asserts the three interactive chapters are present and in order
rather than pinning the whole section list, and the "a title is a claim, not a topic" rule applies to
`section[data-role="results"]` only — the spine's prose sections are headed by *role*, which is a
different job.

## Naming

- **One shipped helper still has no varying input, and it is worth knowing before trusting it.**
  `pack_utilization` is arithmetically pinned to `1.0` because `feed.py` never passes `window=`, so
  the `pack_util` field `train.py` writes into every ledger event is a constant, not a measurement.
  That is not a bug; it is a claim the evidence bundle must not overstate, and `metrics.py` says so
  in its own docstring. The honest packing number the bundle publishes is **loss utilisation**,
  which does vary — 73.8% on a masked reasoning batch against 99.6% on web.

  `masks.loss_mask(context_spans=...)` used to be the second entry here. It has callers now.

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
