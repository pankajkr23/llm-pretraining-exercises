# 06 · Building the Training Dataset

**A training run eats data for weeks. This is the system that remembers what it ate, why, what the
model learned from it, and how to reconstruct any of it.**

Session 5 produced a *recipe* — how much of each kind of data, in what order. Session 6 builds the
machine that **executes** it and can prove it did:

```text
documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches
          -> training -> consumption ledger -> learning ledger -> checkpoint
          -> crash -> resume -> replay -> audit
```

> **Status: stage 7 of 8.** The system trains on four real processes, writes a chain-hashed
> consumption ledger, survives a genuine crash and resumes onto the same batch ids a run that never
> crashed would have consumed. Stage 8 is part-landed: `replay.py` re-derives a recorded interval
> from the immutable shards alone — 32/32 microbatches re-derived, and one flipped bit in one shard
> turns exactly 1 of those 32 red. Fork, the auditor (`verify.py`) and the evidence bundle are
> **not** built yet, and this file says so rather than describing a system that does not exist. The
> stage table is [below](#the-stages).

## How to read this

| you are | start here | then |
| --- | --- | --- |
| **Meeting this for the first time** | [The problem](#the-problem) — why a training run needs a ledger at all | [The vocabulary](#the-vocabulary), which nine words carry the whole system |
| **Changing the code** | [Layout](#layout) and [Run it](#run-it) | [The producer/auditor wall](#the-producerauditor-wall) — the one structural rule that must not be broken |
| **Deciding whether to believe it** | [What it cannot establish](#what-it-cannot-establish) | [`DECISIONS.md`](DECISIONS.md) — what we chose, what we inherited, and what we invented, each with what would overturn it |

Also here: [`PROGRESS.md`](PROGRESS.md) is the running log — findings, changes, and what is still
open, written so the work can be picked up cold. [`NOTICE`](NOTICE) is the authoritative statement
of scope, affiliation and third-party credit. [`CLAUDE.md`](CLAUDE.md) carries the rules specific to
this exercise, for anyone — human or agent — changing the code.

## The problem

You are 50 days into a training run and something looks wrong. You want to know what the model read
on day 40. You open the folder, find 30 GB of files, and there is no way to answer.

That is the motivation, in the instructor's own words. The deliverable is therefore not a data
loader but a **ledger** — an append-only record written as training happens — so the run can be
interrogated afterwards. The assignment says the system is complete only when it can prove four
things, and each maps onto one subsystem:

| the question | the subsystem |
| --- | --- |
| what did it consume? | consumption ledger |
| why that, and not something else? | OPUS decision records |
| what did the model learn from it? | learning ledger |
| can the run be reconstructed? | replay · fork · audit |

### The one idea everything hangs off

A student asked how any of this is reproducible when there is randomness and no saved seed. The
answer inverts the obvious approach:

> *"I will not run the code… **I'm going to run the ledger.** I'm going to read and send. **I will
> not calculate it."***

You do not make a run reproducible by seeding it. You make it reproducible by **writing down what
actually happened** and replaying that. The record outranks the code — which is the only thing that
works once the selector's decisions depend on the model's current weights.

## The vocabulary

| word | plain meaning | ours |
| --- | --- | --- |
| **token** | one integer; text becomes integers before training | vocab 10,002 |
| **sequence** | a fixed-length window of tokens; the model always eats exactly this many | 512 |
| **shard** | a file of tokens, written once, never changed | 5M tokens |
| **manifest** | the label on a shard: contents, origin, licence, hash, whether training may use it | — |
| **microbatch** | the handful of sequences one worker handles at a time | 8 |
| **global batch** | every sequence contributing to **one** weight update, across all workers | 32,768 tokens |
| **rank** | **one worker process**, owning a slice of every batch. In a cluster, one per GPU; here, four processes on the CPU — *the code is identical either way* | 4 |
| **gloo** | the postal service ranks use to exchange gradients. `gloo` runs on CPU everywhere; `NCCL` needs NVIDIA GPUs | CPU backend |
| **ledger** | the append-only diary, one line per microbatch consumed | the deliverable |

## The stages

Each ends in something you can run and see. Nothing advances until the previous one lands.

| stage | you will be able to | status |
| --- | --- | --- |
| 1 | read the settings, the fingerprint, and why the sentinels sit outside the vocabulary | **done** |
| 2 | build shards, print a manifest, watch a tampered shard get rejected | **done** |
| 3 | offer an evaluation shard to the loader and watch it blocked | **done** |
| 4 | ask "what is slot (step 3, rank 2)?" and get token spans back | **done** |
| 5 | pack a window and **see** the block-diagonal attention mask | **done** |
| 6 | train, then read the consumption ledger back line by line | **done** |
| 7 | crash it on purpose, resume, and watch the batch ids line up | **done** |
| 8 | replay an interval, fork a branch, run the auditor | **part** — replay lands; fork and audit do not |

## Layout

```text
BRIEF.md         # the assignment — LOCAL ONLY, gitignored, never the deliverable
CLAUDE.md        # rules specific to this exercise, for whoever changes the code
DECISIONS.md     # what was chosen, and what would overturn each choice
PROGRESS.md      # the running log — findings, changes, what is still open
NOTICE           # scope, affiliation, third-party credit and licences
src/trainingdata/
  spec.py        # constants the producer AND the auditor share — facts, never logic
  config.py      # one frozen dataclass, every knob, plus the run fingerprint
  shards.py      # immutable uint16 shards, content-addressed, tamper-detecting
  manifest.py    # the 20-field manifest and the admission gate
  firewall.py    # the eval registry — data we know about so we can refuse it
  plan.py        # the odometer: which rank trains on which tokens, uncoordinated
  masks.py       # block-diagonal attention, per-document positions, loss masks
  pack.py        # documents out of a span, and the window edge the naive version gets wrong
  feed.py        # a coordinate becomes a microbatch AND its ledger record, from one object
  ledger.py      # the deliverable: append-only, chain-hashed, one file per rank per segment
  model.py       # TinyGPT — RoPE, SwiGLU, tied head. The only place torch is required
  train.py       # one optimizer step, and the token-weighted reduction across ranks
  runner.py      # real worker processes over gloo, spawned, with a file rendezvous
  checkpoint.py  # weights, optimizer state, and the ledger cut that belongs with them
  resume.py      # bringing a ledger back into agreement with a checkpoint, after a crash
  replay.py      # re-deriving a recorded interval from the shards alone — never from the planner
  mixture.py     # session 5's recipe as data — lane shares, floors, and the token targets
tests/           # discovered by `uv run pytest` from the repo root
tools/
  fetch_corpus.py   # TRACKED — a corpus needs a tracked way to fetch and licence-check it
  build_notebook.py # local-only, gitignored — back it up
artifacts/       # heavy regenerable output (gitignored)
```

## Run it

```bash
uv sync --all-packages                                   # install this member
uv run pytest src/exercises/06-build-training-dataset    # the suite
uv run python -c "from trainingdata.config import Config; c=Config(); print(c.fingerprint(), c.total_tokens)"
```

The training step needs torch, which is an **optional extra** so CI never pulls a multi-gigabyte
wheel to run arithmetic. Everything above — shards, manifests, the firewall, the plan, packing, the
feeder, the ledger and replay — is numpy, which is what lets CI verify almost the whole system:

```bash
uv sync --all-packages --extra train                     # ...plus torch
uv run pytest src/exercises/06-build-training-dataset -m integration
```

**What CI runs, and the gap that used to be invisible.** The torch tests — the model, the training
step, and the four-process `gloo` run — skip without the `train` extra, and `uv sync --all-packages`
does not install it. That silently removed **46 of this exercise's 272 tests** and **every one of
its 20 integration tests**: a module-level `importorskip` skips the whole file, a file that collects
nothing looks exactly like a file with nothing in it, and the shard step treats pytest's exit code 5
as success. Every gate stayed green.

A dedicated **`train` job** now runs those files with CPU-only torch wheels — 191.8 MB rather than
the 2.7 GB CUDA build, pinned by a Linux-scoped index in the root `pyproject.toml` — in parallel
with the 164s tokenization shard, so it costs no wall clock.
`tests/test_ci_shards_cover_everything.py` fails if any gated file stops being reachable by a job
that installs what it needs. **`gloo` additionally needs loopback networking**, so the multi-rank
tests still skip inside a sandbox that blocks it; a GitHub runner allows it.

## The producer/auditor wall

The assignment refuses hardcoded evidence and inspects the code to check the behaviour was not
simulated. So the auditor — `verify.py`, when it lands — will re-derive every published claim from
the artifacts on disk **without importing the code that produced them**. If it imported the
producer it would inherit the producer's bugs and agree with itself.

`spec.py` is the one deliberate exception: shared **facts** (the nine evidence rows, the thirteen
log events, the sentinel ids), never shared **logic**. A test parses its source and fails if it ever
imports from the rest of the package.

**Replay is not the auditor, and the difference is the wall.** `replay.py` is a *producer-side*
tool: it deliberately imports `ledger`, `masks`, `pack` and `shards`, because its job is to rebuild
a microbatch the same way the run built it and check the shards still hold what was fed. What it may
not import is `plan.py` — recomputing the plan instead of reading the record would make the
measurement circular in exactly the way this session is about — and torch.
`test_replay_cannot_reach_the_planner_or_torch` walks the transitive closure for both, and
`test_the_closure_check_would_notice_a_new_import` is the twin that fails when the walker stops
seeing anything. The auditor's wall is stricter and still unbuilt: `verify.py` will import nothing
from `trainingdata` except `spec.py`.

## What it cannot establish

Stated here, and it will be stated again beside the numbers when there are numbers.

- **Data parallelism only.** No tensor, pipeline or sequence parallelism, and no sharded optimizer
  state — those need multiple GPUs, and this runs on one machine.
- **A tiny model on a small corpus.** The design is *about* 100-billion-token mechanics. It was
  never run at that scale, and any frontier figure quoted anywhere is labelled as modelled.
- **Replay proves inputs, never losses.** Batch ids, token spans and input hashes reproduce exactly.
  Losses, selector scores and checkpoint hashes do not — floats differ across devices — and are
  reported with a stated tolerance rather than an equality.
- **Two of the four OPUS statuses are ours.** `defer` and `floor_override` appear nowhere in the
  OPUS paper, its reference implementation, or LightningLM; all three were searched.
  [`DECISIONS.md`](DECISIONS.md) records that we defined them.

## Credits

The selection method is **OPUS — Optimizer-induced Projected Utility Selection**, Wang et al.,
[arXiv:2602.05400](https://arxiv.org/abs/2602.05400) (CC BY 4.0). Where this exercise ports the
criterion it follows the reference implementation at
[`gszfwsb/OPUS`](https://github.com/gszfwsb/OPUS) (MIT, © Keller Jordan 2024 and the OPUS authors
2026 — both retained). The guaranteed-lane and golden-proxy patterns are adapted from
[`The-School-of-AI/LLM`](https://github.com/The-School-of-AI/LLM) (Apache-2.0, © Rohan Shravan and
The School of AI).
