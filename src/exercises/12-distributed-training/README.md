# 12 · Distributed training: ZeRO on 32 simulated GPUs

**Thirty-two simulated devices train one small model four ways — plain data parallelism and ZeRO
stages 1, 2 and 3. On every device, the bytes of every persistent buffer each stage keeps (weights,
gradients, master copy, Adam moments) and every byte the ring collectives send are counted, and
both equal a formula worked out by hand; all four stages train to bit-identical weights.**
Activations and allocator memory are not counted — see
[What this cannot establish](#what-this-cannot-establish).

**The page:** [llm-pretraining-demos.vercel.app/12-distributed-training](https://llm-pretraining-demos.vercel.app/12-distributed-training/)
draws the same argument — which bytes each stage splits, a ring stepped by hand, the ledger beside
its formula, and the ladder at scale — with every number generated from `results/zero.json`.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is), then
  [The sixteen bytes](#the-sixteen-bytes) and [The four stages, by hand](#the-four-stages-by-hand).
  No code needed. [Two things people get wrong](#two-things-people-get-wrong) is worth reading
  even if you think you know ZeRO.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it). `CLAUDE.md` carries the rules this exercise learned the hard way.
- **Deciding whether to believe it** — go to [The evidence](#the-evidence) and
  [`RESULTS.md`](RESULTS.md), then [What this cannot establish](#what-this-cannot-establish), and
  [`DECISIONS.md`](DECISIONS.md) for every choice that could have gone the other way.

## What this is

Training a model on many GPUs usually starts with **data parallelism**: every GPU gets a complete
copy of the model and a different slice of the batch. Each one computes gradients on its own slice,
the GPUs average their gradients, and every GPU applies the same update. They stay in lockstep
because they all start identical and all apply identical updates.

That works, and it is wasteful in one specific way: **every GPU stores exactly the same training
state**. Thirty-two GPUs hold thirty-two identical copies. Think of thirty-two accountants who each
keep a full copy of the same enormous ledger, when each could keep 1/32 of the pages and borrow a
page from a colleague at the moment they need to read it.

**ZeRO** (the Zero Redundancy Optimizer) is that idea, applied in three steps. Each stage stops
replicating one more kind of state:

- **ZeRO-1** shards the optimiser state,
- **ZeRO-2** also shards the gradients,
- **ZeRO-3** also shards the weights themselves.

This exercise builds 32 devices inside one Python process, gives each its own memory and a ledger
of every tensor it holds, connects them with real ring collectives that count every byte they send,
and trains exercise 09's transformer on them under each stage. Then it checks three things: that
the training state each device keeps is what the arithmetic below predicts, that the communication
is what it predicts, and that the stage never changes the result.

## The sixteen bytes

Training one weight with the Adam optimiser in mixed precision costs **16 bytes**, and the whole
argument rests on knowing where they go:

| what | dtype | bytes | why it exists |
| --- | --- | ---: | --- |
| the weight | bf16 | 2 | what the forward and backward pass compute with |
| its gradient | bf16 | 2 | produced by the backward pass |
| an fp32 master copy of the weight | fp32 | 4 | bf16 is too coarse to accumulate many tiny updates into |
| Adam's first moment `m` | fp32 | 4 | a running average of the gradient |
| Adam's second moment `v` | fp32 | 4 | a running average of the squared gradient |
| **total** | | **16** | |

Twelve of the sixteen — the master copy and both moments — exist only for the optimiser. Activations
come on top and depend on the batch; they are a separate bill and ZeRO does not shard them.

Our demo model has **666,560 weights**, so plain data parallelism holds
666,560 × 16 = **10,664,960** bytes on every device. The simulator's ledger measures exactly that.

## The four stages, by hand

At **N** devices, one device holds (bytes per weight; activations excluded):

| stage | weights | gradients | master + m + v | per weight | at N = 32 | N → ∞ |
| --- | --- | --- | --- | --- | ---: | ---: |
| **DP** (stage 0) | 2 | 2 | 12 | 16 | 16 | 16 |
| **ZeRO-1** | 2 | 2 | 12 / N | 4 + 12/N | 4.375 | 4 |
| **ZeRO-2** | 2 | 2 / N | 12 / N | 2 + 14/N | 2.4375 | 2 |
| **ZeRO-3** | 2 / N | 2 / N | 12 / N | 16/N | 0.5 | 0 |

**Worked at N = 32 for our model.** ZeRO-1: 2 + 2 + 12/32 = 4.375 bytes per weight, and
666,560 × 4.375 = 2,916,200 bytes. ZeRO-3: 16/32 = 0.5, and 666,560 × 0.5 = **333,280** bytes —
thirty-two times less than data parallelism. Both are what the ledger measured
([`RESULTS.md` §1](RESULTS.md#1--memory-per-device-measured-by-the-ledger-bf16-mixed)).

**The last column is the floor**: what stays replicated however many devices you add. It is why
the stages differ in kind, not just degree — adding GPUs to data parallelism never reduces its
memory per device at all, and ZeRO-1 can never get below the 4 bytes of replicated weights and
gradients.

### What moves between the devices

Three collective operations do all the work, each run as a **ring** — device *r* only ever sends to
device *r* + 1:

- **reduce-scatter**: sum a buffer across all devices, but each device receives only its own 1/N
  slice of the sum;
- **all-gather**: each device contributes its 1/N slice and every device ends up with the whole
  buffer;
- **all-reduce**: every device ends up with the whole sum — and it *is* a reduce-scatter followed by
  an all-gather. Not "like" one: the ring implementation is literally those two phases.

In a ring, each phase makes every device send N − 1 chunks of size P/N, where **P** is the size of
one full copy of the weights. So each phase costs **P · (N − 1)/N** bytes per device. For our model in
bf16, P = 666,560 × 2 = 1,333,120 bytes, and one phase at N = 32 is 1,333,120 × 31/32 = 1,291,460
bytes.

| stage | collectives per step | bytes sent per device | our model, N = 32 |
| --- | --- | --- | ---: |
| DP | all-reduce the gradients | 2 · P · (N−1)/N | 2,582,920 |
| ZeRO-1 | reduce-scatter the gradients · all-gather the updated weights | 2 · P · (N−1)/N | 2,582,920 |
| ZeRO-2 | the same, the reduce-scatter done layer by layer during backward | 2 · P · (N−1)/N | 2,582,920 |
| ZeRO-3 | all-gather the weights in forward · again in backward · reduce-scatter the gradients | 3 · P · (N−1)/N | 3,874,380 |

**ZeRO-1 and ZeRO-2 cost no extra communication.** Data parallelism's all-reduce already *is* a
reduce-scatter plus an all-gather. ZeRO-1 and -2 run the same two phases, and simply keep the
half-way result — each device's slice of the summed gradient — instead of throwing it away. That is
the whole trick: the slice is exactly what a device needs to update its slice of the optimiser
state. **ZeRO-3 costs 50% more**, because it frees each layer's weights after using them in the
forward pass and must gather them a second time for the backward pass.

### What does not change

**The forward and backward computation per device is identical in every stage.** Every device still
runs every layer, on its own slice of the batch. The simulator counted 21,365,696 forward FLOPs per
device per step under all four stages. Only the optimiser's work shrinks: each device updates
666,560 weights under data parallelism and **20,830** — one thirty-second — under any ZeRO stage.

## Two things people get wrong

**"A batch of 8 means 8 copies of the model."** It does not. Batch size multiplies *activations* —
the intermediate values each example produces on its way through the network — not weights. One
device processing 8 sequences holds one copy of the weights and 8 sets of activations. Data
parallelism has N copies of the weights because it has N *devices*, each with its own copy, and
that is a separate fact from how many examples each one processes.

**"ZeRO-3 splits the model by layers across devices."** That is **pipeline parallelism**, a
different technique: device 0 owns layers 1–4, device 1 owns layers 5–8, and activations flow from
one to the next. ZeRO-3 does the opposite. Every device holds **1/N of every layer** — the simulator's
device 5 holds slice 5 of the embeddings, slice 5 of every block and slice 5 of the head — and every
device still runs every layer on its own data, gathering that layer's full weights for the moment it
needs them. `flat.py` says this in code: a unit's flat buffer is cut into N equal slices, one per
device.

## Pros and cons, per stage

| stage | buys | costs | use it when |
| --- | --- | --- | --- |
| **DP** | simplest; one collective per step; nothing to gather | 16 bytes per weight on every device, forever | the whole training state fits on one device with room for activations |
| **ZeRO-1** | 12 of the 16 bytes shrink by N, **at no extra communication** | weights and gradients still replicated: a 4-byte floor | DP almost fits; it is close to free |
| **ZeRO-2** | gradients shrink too, still at no extra communication | gradients must be reduce-scattered layer by layer during backward (bucketing), and a full layer's gradient exists briefly; a 2-byte floor | the default for large models that fit once weights are the only replicated state |
| **ZeRO-3** | everything shrinks by N; no floor | **+50% communication**; one layer's full weights gathered twice per step, which adds latency unless the next layer is fetched while the current one computes | the model does not fit even at 2 bytes per weight, or you want fewer devices |

## The memory ladder at 30 billion weights

Nothing is simulated at this size; these are the formulas above, multiplied out, against an assumed
80 GB card (74.5 GiB). Data parallelism needs **447.0 GiB** per device at any N. ZeRO-1 never fits
either: its replicated weights and gradients alone are 4 bytes × 30 × 10⁹ = 120 GB, more than the
card, and that figure does not shrink with N. **ZeRO-2 fits from 32 devices** (68.1 GiB) and not at
16. **ZeRO-3 fits from 8** (55.9 GiB). The full table, with the communication each one costs, is
[`RESULTS.md` §7](RESULTS.md#7--the-memory-ladder--a-30b-weight-model-predicted).

So at that size the real choice is ZeRO-2 on many devices or ZeRO-3 on fewer: ZeRO-3 trades a 50%
increase in communication for the ability to fit on a quarter of the hardware.

## Does the stage change the answer?

**No, and here that is shown bit for bit.** After the published run's four steps, the weights
trained under ZeRO-1, -2 and -3 are bit-identical to data parallelism's, in both bf16-mixed and
fp32 (the tests re-check this on a three-step run of their own). ZeRO changes where numbers are
stored and how they travel, not what is computed.

**Bit-identical is a property of this simulator, not of ZeRO in general.** Here data parallelism's
all-reduce *is* a ring reduce-scatter followed by an all-gather, so DP and ZeRO-1/2 add the same
numbers in the same order by construction, and every stage's optimiser update is elementwise. A
real library may all-reduce with a different algorithm — a tree, say — which sums in a different
order; then DP and ZeRO agree only to rounding, as the single-device comparison below shows.

**Against one device trained on the whole global batch** with `torch.optim.AdamW` in fp32, the
losses agree to a relative 5.7e-08 and the step-one gradients to within rounding. The weights agree
to at most 6e-06 everywhere — **except** on the attention **key bias**, where they differ by up to
2e-05. That is not a bug, and finding out why was the most instructive moment of the exercise: the
key bias adds the same number to every attention score in a row, and softmax ignores a constant
added to a whole row, so its true gradient is exactly zero. What a computer produces for it is
rounding noise — and AdamW divides every gradient by its own running size, which turns noise into a
full-sized step. Two runs that add the same numbers in a different order drift apart there and only
there.

**A second finding came from the bit-identical claim itself.** The first optimiser used torch's own
fused kernels, exactly as `torch.optim.AdamW` does, and ZeRO-1 disagreed with data parallelism by
about 1e-6 — in an update that is elementwise and cannot depend on how the buffer is sliced. It
did depend on it. Probed directly on this machine (arm64, torch 2.13), the fused kernels gave the
same bits as the whole buffer when it was cut at multiples of 64 elements, and different bits from
`addcmul_` when it was cut at other offsets — consistent with a vectorised loop and its scalar tail
rounding differently. Rewritten with operations that each round once, the stages agree exactly, and
a test checks that at deliberately unaligned cuts.

bf16-mixed tracks the fp32 single device closely: the cosine between the two runs' total weight
updates is 0.9999 ([`RESULTS.md` §6](RESULTS.md#6--does-the-stage-change-the-answer)).

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | every knob — world size, nodes, model, optimiser, precision — and the **assumed** hardware figures, labelled as such |
| `precision.py` | the 16 bytes, itemised per precision recipe; torch-free |
| `world.py` | the simulated devices (`Rank`), their memory ledgers, the communication counters, and which ring links leave a node |
| `collectives.py` | ring reduce-scatter, all-gather and all-reduce, run for real across the ranks, counting every byte |
| `flat.py` | flat buffers: a unit's weights as one padded 1-D tensor cut into N equal shards |
| `adamw.py` | AdamW on a flat shard, built from single-rounding operations so slicing cannot change a bit |
| `model.py` | exercise 09's trunk and head, split into the units ZeRO-3 gathers one at a time |
| `stages.py` | one engine, four policies: what each stage keeps, gathers, scatters and frees |
| `reference.py` | one device, the whole global batch, `torch.optim.AdamW` — the answer every stage must match |
| `formulas.py` | the hand calculations, as code: bytes per weight, bytes sent, the ladder |
| `timing.py` | the time model — assumed bandwidths and throughput applied to measured counts |
| `provenance.py` | config fingerprint, code digest, and the refusal to write an unprovenanced result |
| `experiment.py` | runs everything and sets each measurement beside its prediction |

`tools/run_zero.py` writes `results/zero.json`; `tools/render_results.py` turns it into
[`RESULTS.md`](RESULTS.md) and into `web/data.js`, the page's only source of numbers.

## Run it

```bash
uv sync --all-packages --extra train

# Every measurement, with its provenance, into results/zero.json (about 20 s on a laptop CPU)
uv run python src/exercises/12-distributed-training/tools/run_zero.py
uv run python src/exercises/12-distributed-training/tools/render_results.py

# The tests: formulas, collectives, ledger, equivalence, provenance, documents
uv run pytest src/exercises/12-distributed-training
```

A Colab notebook (`notebooks/S12-distributed-training.ipynb`, kept local like every topic
notebook here) walks the same path with every hand calculation checked by `assert` against the
ledger.

## The evidence

[`RESULTS.md`](RESULTS.md) is generated from `results/zero.json` and is re-checked against it on
every test run. Every row in it puts a **measured** number beside a **predicted** one. The
measurements are the storage bytes of every persistent and transient buffer a stage puts on each
simulated device (counted once per storage, so a slice of a larger buffer is charged the whole
buffer), the bytes every ring hop sent, torch's own FLOP counter, and the trained weights. The
predictions are `formulas.py`. In the published run, every persistent memory category, every
communication count and every bytes-per-weight figure at N = 1, 2, 3, 4, 8, 16 and 32 matches its
prediction exactly. N = 3 is there on purpose, because it is the one size where padding is not
zero. Activations, temporaries inside one operation and the allocator's own overhead are not
counted; [What this cannot establish](#what-this-cannot-establish) lists what that leaves out.

## What this cannot establish

**Everything here is simulated in a single process on a CPU.** The 32 devices are Python objects;
nothing crosses a network, so no measured number is a time. The communication counts are exact
counts of what a ring algorithm *would* send, and the times in `RESULTS.md` come from **assumed**
bandwidths (450 GB/s inside a node, 50 GB/s between nodes) and an **assumed** device throughput —
change those and every time changes, while the byte counts do not. The time model also assumes a
flat ring with no overlap of compute and communication; real libraries use hierarchical collectives
and overlap heavily, so its absolute times are pessimistic and only its ratios carry meaning.

**Activations are excluded** from every memory figure, and they are often the larger bill. The
ledger records the storage bytes of the training state ZeRO shards, plus the transient buffers it
creates; it does not record the temporaries inside one optimiser update, the ring's in-flight
chunks, or anything a real device's allocator adds — caching, fragmentation, alignment.

**The model is tiny** — 666,560 weights, two blocks — chosen so 32 devices finish in seconds. The
memory and communication formulas are exact at any size and the ladder applies them to 30 billion
weights, but that ladder is arithmetic, not a run. Nothing here measures the latency of gathering
layer by layer, the effect of prefetching, or how ZeRO interacts with tensor or pipeline
parallelism.

**Four training steps prove equivalence, not learning.** Loss barely moves in four steps; the
overfit test in `tests/` shows the ZeRO-3 path can drive a loss down, and nothing more.
