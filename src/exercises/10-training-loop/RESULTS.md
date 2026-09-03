# RESULTS — 10 · The training loop

**Generated from `results/run.json`. Do not edit by hand** — run `tools/render_results.py`. Every
number and every verdict word here, including *higher*, is read from that file.

200 steps, Adam at 0.0003, batch 8 ×
128 tokens, gradient clipped at 1.0, seed 10.

**8,312,832 parameters** — 5,752,576 in the trunk and
2,560,256 in the output head. **2,593,024 of the trunk's
are embedding tables**, and that distinction decides item 5: an embedding lookup is a gather, so
those parameters do no arithmetic and must not be priced.

---

## The six items

### 1 · Every tensor shape in the step

The step's tensors and what each dimension means are printed by the run. The one worth stopping on
is the loss: it has **no dimensions at all**. Everything above it collapses into that scalar and
everything the optimiser does flows back out of it, which is why a mistake anywhere between the
logits and the loss changes training without changing a single shape.

### 2 · One gradient, verified by hand

`head.weight[4884, 0]` — the largest-magnitude gradient in the head, because a near-zero one
would make this a comparison of two zeros.

| epsilon | autograd | central difference | relative error | matching digits |
| --- | --- | --- | --- | --- |
| 1e-01 | 0.007397994 | 0.007397857 | 1.85e-05 | 4.7 |
| 1e-02 | 0.007397994 | 0.007397993 | 1.84e-07 | 6.7 |
| 1e-03 | 0.007397994 | 0.007397994 | 2.05e-09 | 8.7 |
| 1e-04 | 0.007397994 | 0.007397994 | 1.69e-09 | 8.8 |
| 1e-05 | 0.007397994 | 0.007397994 | 7.69e-09 | 8.1 |
| 1e-06 | 0.007397994 | 0.007397995 | 1.36e-07 | 6.9 |
| 1e-07 | 0.007397994 | 0.007398002 | 1.10e-06 | 6.0 |

**Closest agreement at epsilon = 1e-04: 8.8
matching decimal digits.** Autograd said 0.007397994; a central difference said
0.007397994.

**Read the column, not the row.** Agreement improves as epsilon falls — the central difference's
error goes as epsilon squared — and then gets *worse* again, because `loss(w+h)` and `loss(w-h)`
stop differing in bits the float type keeps and the subtraction becomes rounding noise. There is a
window, not a best value, and a check that agrees at exactly one epsilon has not been verified: it
has been fitted.

**This runs in float64, and that is not a convenience.** In fp32 a loss near 9.2 resolves to about
5e-7, so any gradient small enough to move the loss by less than that gives a numeric estimate of
exactly zero at every epsilon. The first version of this did precisely that and reported a relative
error of 1.0 across the whole sweep.

### 3 · Gradient accumulation, broken on purpose

Micro-batches of [4, 4, 2] valid tokens with average losses [2.0, 2.0, 5.0]:

| reduction | value |
| --- | --- |
| correct — total loss over total tokens | **2.6000** |
| wrong — the mean of the means | **3.0000** |
| gap | **+0.4000** (+15.4%) |

The short micro-batch carries half the real tokens and gets exactly the same vote.

**And the same two reductions driving a real run**, 120 steps, everything else
held identical — micro-batch widths [128, 128, 64] tokens:

| reduction | final loss |
| --- | --- |
| correct | 5.2873 |
| wrong | 5.3633 |
| gap | +0.0759 |

The wrong reduction reads **higher**, by a mean of 0.0200 across
the run. Both curves are in `results/run.json` under `item_3_accumulation.curves`.

**The gap is small, and that is exactly why a bug of this shape lived inside every major training
framework until 2024.** The wrong curve does not look wrong. It looks like the right curve. And the
error is *exactly zero* whenever every micro-batch carries the same number of real tokens — which a
hand-built test case almost always does, so the fault was invisible to the checks that would have
caught it.

### 4 · A step where the gradient norm moved before the loss did

step **24**, out of 200 logged. The gradient norm moved 3.1 typical steps while the loss moved 0.1. Gradient norm 0.9321, loss 7.0616.

9 of 200 steps qualify at the default threshold.

**The threshold is arbitrary, so here is what happens when it moves:** threshold=2.0 → 19  ·  threshold=2.5 → 13  ·  threshold=3.0 → 9  ·  threshold=4.0 → 3  ·  threshold=5.0 → 2

**Why the gradient leads.** The loss is an average over a whole batch, so a change in what the model
is doing has to be large enough to move that average before it is visible. The gradient norm
measures how hard the optimiser is pushing *right now*. A run that logs only the loss finds out
about its problems late.

The norm is logged **before** clipping. A trace of the post-clip norm flattens at the clip value,
which hides precisely the spikes the trace exists to show.

### 5 · MFU, computed honestly

| input | value |
| --- | --- |
| parameters priced | 5,719,808 |
| FLOPs per token | 34,318,848 |
| convention | 6 x 5,719,808 NON-EMBEDDING parameters — 2 forward, 4 backward; embedding lookups are gathers and do no arithmetic; attention's quadratic term excluded |
| tokens measured | 203,200 |
| wall clock | 7.504 s |
| achieved | 929.35 GFLOP/s |
| device peak | 3.362 TFLOP/s |
| device | this machine's CPU, 3.362 TFLOP/s sustained on a 2048^3 fp32 matrix multiply — MEASURED here, same device and dtype as the run, not a vendor figure |
| **MFU** | **27.64%** |
| tokens/second | 27,080 |

**Target 40%, achieved 27.64%, short by 12.36% of peak.**

**Two errors were caught in this number before it was published, and both flattered it.** The first
version divided FLOPs achieved on the **CPU** by a **GPU's** advertised peak and reported 39.13% —
a figure that looked excellent and compared two different processors. The peak is now *measured*, on
the same device and dtype as the run, with a large dense matrix multiply. The second counted the
embedding tables in the parameters: an embedding lookup is a gather that does no arithmetic at all,
so those parameters are free inflation, and removing them cut the numerator by 45%.

What costs the distance to 40%, in the order it costs:

1. **The model is too small for the machine.** A `6N` estimate assumes the device spends its time
   inside large matrix multiplies. At this width every multiply finishes before the device is fully
   occupied, so the fixed cost of launching work dominates the work. This is the whole gap, and it
   is a property of the shape rather than of the code.
2. **Nothing is fused.** Every operation reads its inputs from memory and writes its output back.
3. **The step is timed whole**, including data slicing, the optimiser's element-wise work and the
   gradient-norm computation — none of which is in the numerator. Timing only the matrix multiplies
   would report a better number and a less true one.
4. **fp32, not bf16.** The peak quoted is an fp32 peak, so this does not inflate the figure — but
   comparing it to a published bf16 MFU would compare two different quantities.

**The reachable fix is (1).** None of the others is worth doing at this scale, and saying so is more
useful than a list of optimisations nobody should apply here.

### 6 · 0.1 in three formats, bit by bit

One tenth is `0.0001100110011…` repeating in binary, exactly as one third is `0.333…` repeating in
decimal — so no binary format holds it, and the only question is how much each one misses by.

| format | sign · exponent · mantissa | hex | stored value | error |
| --- | --- | --- | --- | --- |
| bf16 | `0 01111011 1001101` | 0x3DCD | 0.10009765625 | +0.0977% |
| fp32 | `0 01111011 10011001100110011001101` | 0x3DCCCCCD | 0.10000000149011612 | +0.0000% |
| fp8 E4M3 | `0 0011 101` | 0x1D | 0.1015625 | +1.5625% |

| format | exponent bits | mantissa bits | smallest normal | largest finite | decimal digits |
| --- | --- | --- | --- | --- | --- |
| bf16 | 8 | 7 | 1.18e-38 | 3.38953e+38 | 2.4 |
| fp32 | 8 | 23 | 1.18e-38 | 3.40282e+38 | 7.2 |
| fp8 E4M3 | 4 | 3 | 0.0156 | 448 | 1.2 |

**Which would I train in? bf16**, and one column decides it. bf16 keeps fp32's **eight** exponent
bits and spends the entire saving out of the mantissa, so it has fp32's *range*: a gradient that
underflowed to zero in fp16 does not underflow here, and no loss-scaling machinery is needed to keep
small values alive. It pays in precision, and gradient descent tolerates imprecision far better than
it tolerates zeros.

**fp8 E4M3 is a different decision, not a further step along the same one.** Four exponent bits give
it a much narrower range, and it does not reserve the all-ones exponent for infinity — which is why
it reaches 448 rather than stopping lower. It is a format for
weights and activations under a scaling scheme that keeps values inside that range, not a drop-in
replacement for bf16.

**These patterns are built from arithmetic here, not read out of the machine** — and then checked
against torch's own casts, because a decomposition that agrees only with itself proves nothing.
