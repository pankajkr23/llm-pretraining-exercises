# DECISIONS — 12-distributed-training

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · One process plays all 32 devices

**Decision.** The devices are `Rank` objects inside one Python process, not 32 processes talking
through `torch.distributed`. Collectives are ring algorithms written here and run across those
objects.

**Why.** What this exercise has to show is *where the bytes are* and *how many move*, exactly. In
one process every tensor a device holds has a storage whose bytes the ledger can add up, and
every send is a function call a counter can see. Real process groups would add a transport whose
memory and traffic we could only estimate, would need 32 processes on a laptop or a free Colab
runtime, and would make the run non-deterministic in ways that hide the bit-identical result. The
cost is that nothing here is a time.

**What would overturn it.** Needing to measure latency, overlap or contention. That needs real
processes on real links, and a CPU simulation cannot supply it.

## D2 · One flat-buffer partitioning for every stage, one buffer per unit

**Decision.** The model is split into units — embeddings, each block, final norm with the head.
Each unit's weights are concatenated into one padded 1-D buffer, and rank `r` owns slice `r` of
**every** unit's buffer. Every stage uses this same layout for weights, gradients, master copy and
both moments.

**Why.** It is how ZeRO shards (each buffer across all ranks), it is the granularity ZeRO-3 gathers
at, and using one layout everywhere means the only difference between the stages is *what each rank
keeps*. That is what makes a bit-identical comparison possible at all.

**What would overturn it.** A model whose units are very unequal in size, where per-unit padding or
the largest unit's transient buffer became the story. Here the largest unit is the embedding table.

## D3 · ZeRO-1 reduce-scatters its gradients rather than all-reducing them

**Decision.** Stage 1 keeps a full gradient buffer (it does not shard gradients) but moves the
gradients with a reduce-scatter, writing each rank's reduced slice into its own part of that buffer.

**Why.** A rank only updates its own slice of the optimiser state, so it only needs its own slice of
the averaged gradient. An all-reduce would hand it the rest too and, with the all-gather of updated
weights that stage 1 needs anyway, would cost 3·P·(N−1)/N instead of 2. With a reduce-scatter,
stage 1 costs exactly what data parallelism costs, which is the point it makes. The memory saving
of stage 2 then comes purely from freeing the gradient that stage 1 kept.

**What would overturn it.** An implementation constraint that needs the full averaged gradient on
every rank — gradient clipping by global norm can be done with one scalar all-reduce instead, so
clipping alone does not.

## D4 · Padding, recorded rather than hidden

**Decision.** A unit's buffer is zero-padded to a multiple of N. Every byte count — measured and
predicted — is computed on padded sizes, and `Layout.padding` reports the padding.

**Why.** Equal shards are what a ring needs, and padding is real memory and real traffic. Zero
weights with zero gradients stay exactly zero under AdamW, so the padding never touches the model.
The collectives **refuse** an indivisible buffer rather than pad silently, so padding happens in one
place.

**What would overturn it.** Nothing in principle; a production system might pad to a larger
alignment for speed, which would change the byte counts and not the conclusions.

## D5 · Two precision modes, bf16-mixed by default

**Decision.** `bf16-mixed` (bf16 weights and gradients, fp32 master and moments) is the measured
mode; `fp32` exists for the tight equivalence check. The loss is computed in fp32 from bf16 logits.
Gradients are reduced in their own dtype, and the loss is scaled by 1/N before backward so the ring
sum is the mean.

**Why.** bf16-mixed is the recipe the sixteen bytes describe. fp32 removes bf16's coarse rounding so
the comparison with a single device can be tight. Scaling by 1/N = 1/32 is exact in binary
floating point.

**What would overturn it.** Wanting fp32 gradient reduction in mixed mode, which some libraries
offer for accuracy. It would double gradient traffic and change the formulas' P for gradients.

## D6 · Lockstep execution, and every unit's forward recomputed in the backward

**Decision.** The step runs unit by unit across all ranks. The forward keeps only each unit's
input; the backward recomputes the unit's forward from that input and backpropagates through it.
This is done identically for all four stages.

**Why.** ZeRO-2 must reduce-scatter a unit's gradient as soon as it exists, and ZeRO-3 must gather a
unit's weights just before it runs — both need every rank at the same unit at once, which lockstep
gives. Recomputation is what makes ZeRO-3's freeing honest: PyTorch's autograd graph keeps a
reference to every weight a forward used, so dropping gathered weights while keeping the graph
would free nothing. A weak-reference test proves the gathered buffers really die. Doing it on every
stage keeps the comparison fair: forward FLOPs double everywhere and stay equal across stages.

**What would overturn it.** Swapping a parameter's storage under autograd, as some libraries do.
That would avoid recomputation but would rely on autograd internals that exercise 09's modules do
not guarantee to respect.

## D7 · AdamW built from single-rounding operations

**Decision.** `adamw.py` writes the update with multiply, add, subtract, divide and square root, not
with torch's fused `lerp_`, `addcmul_`, `addcdiv_`.

**Why.** With the fused kernels, ZeRO-1 disagreed with data parallelism by about 1e-6 although the
update is elementwise. Probed directly on this machine (arm64, torch 2.13), the fused kernels
matched the whole buffer when it was cut at multiples of 64 elements and `addcmul_` did not when it
was cut elsewhere — consistent with a vectorised loop and a scalar tail rounding differently. That
probe is specific to this CPU and is not a test; what *is* tested is that our update is identical at
deliberately unaligned cuts. The price: we match `torch.optim.AdamW` to 1e-6 of the weight scale,
not bit for bit.

**What would overturn it.** A platform where the single-rounding operations themselves were fused
by the compiler. The slicing test would catch it.

## D8 · The key bias is reported separately in the reference comparison

**Decision.** Against the single-device reference, weights are held to 1e-5 (1% of one AdamW step)
everywhere except the attention key bias, which is reported on its own.

**Why.** The key bias's true gradient is exactly zero, because it adds a constant to a whole softmax
row. Its computed gradient is therefore rounding noise, and AdamW normalises noise into steps of
about the learning rate. Two runs whose additions happen in a different order end up with different
noise. The test asserts the reason — the key-bias gradient is below 1e-6 of the largest gradient —
instead of loosening the tolerance for all 666,560 weights to hide 64 of them.

**What would overturn it.** Another parameter whose gradient is structurally zero. The test would
fail on it, which is the intended way to find it.

## D9 · The time model is assumed figures over measured counts

**Decision.** Times come from `timing.py`: 450 GB/s inside a node, 50 GB/s between nodes, 400
TFLOP/s per device, all labelled as assumed in `Config`. A ring collective lasts as long as its
busiest link. Compute and communication are added, never overlapped.

**Why.** A reader thinks in seconds, and the shape of the trade — four slow inter-node links pacing
all 32, ZeRO-3 paying 50% more — is clearer in seconds. Keeping the assumptions in `Config`, the
counts in `results/zero.json` and the times computed separately means the assumptions can change
without touching a measurement.

**What would overturn it.** Real hardware. Hierarchical collectives and overlap would change the
absolute times substantially; nothing here should be quoted as a predicted step time.

## D10 · What the ledger records

**Decision.** The ledger records the five persistent categories ZeRO shards and the two transient
buffers it creates (gathered weights, gradient buckets). It does not record activations, the
temporaries inside one optimiser update, the outputs of a reduce-scatter before they are copied into
place, or the ring's in-flight chunks.

**Why.** The claims are about the training state. Activations are a separate bill that ZeRO does not
shard, and the unrecorded temporaries are at most one unit or one chunk and would make the persistent
figures harder to compare with the formulas.

**It charges storage, not views.** Each held tensor costs its `untyped_storage().nbytes()`, counted
once per device however many held tensors share it. The first version charged
`numel × element_size`, and a shard taken as a view of a full buffer then cost 1/N while keeping
N/N alive: removing a `.clone()` from a sharded placement went unnoticed by every test. Charging
storage makes that mistake show up as a ledger that disagrees with the formula, and a test asserts
that every held tensor owns exactly its storage.

**What would overturn it.** A question about peak device memory in total. That needs activations,
and a real device's allocator.

## D11 · Exercise 09's private `_corpus` is used

**Decision.** `model.batches` calls `lossheads.training._corpus`.

**Why.** It is the one function that turns exercise 09's frozen corpus into fixed-length token
sequences, and its corpus is what `corpus_digest` vouches for. Re-implementing it here would be a
second copy that could drift from the digest. The public `corpus_facts` is used for the reported
corpus statistics.

**What would overturn it.** Exercise 09 renaming or changing `_corpus`. The model tests would fail
at import or on the batch shape.

## D12 · Untied output head; no prefetching in ZeRO-3

**Decision.** The head owns its weights rather than sharing the embedding's. ZeRO-3 gathers one unit
at a time and does not fetch the next unit while the current one computes.

**Why.** A tied weight belongs to two units, which would make ZeRO-3 gather it twice per pass — real,
and not the subject here. Without prefetching, the measured transient peak is exactly one unit,
which is the cleanest thing to assert; real systems prefetch and hold two.

**What would overturn it.** A reader wanting the memory cost of prefetching. It would be a second
transient unit, and a one-line change to the assertion.
