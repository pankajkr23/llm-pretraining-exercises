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

| tensor | shape | what each dimension is |
| --- | --- | --- |
| `tokens` | `(8, 128)` | batch · position — the ids fed in |
| `hidden` | `(8, 128, 256)` | batch · position · width — one vector per position |
| `logits` | `(8, 128, 10001)` | batch · position · vocabulary — one score per token |
| `inputs` | `(8, 127)` | batch · position — last dropped, nothing follows it |
| `targets` | `(8, 127)` | batch · position — first dropped, nothing predicts it |
| `flat logits` | `(1016, 10001)` | position · vocabulary — batch folded away |
| `flat targets` | `(1016,)` | position — one correct id per position |
| `loss` | `()` | a scalar — no dimensions at all, which is the point |
| `head.weight.grad` | `(10001, 256)` | vocabulary · width — one gradient per weight, the weight's own shape |

**The one worth stopping on is the loss: it has no dimensions at all.** Everything above collapses
into that scalar, and everything the optimiser does flows back out of it — which is why a mistake
anywhere between the logits and the loss changes training without changing a single shape.

Two more rows repay a second look. `head.weight.grad` has exactly the shape of the weight it belongs
to, which is what makes the gradient check below possible at all. And `flat targets` is
1,016 rather than 1,024: the shift drops the last position of every
sequence, because nothing follows it.

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
| correct | 5.2658 |
| wrong | 5.3143 |
| gap | +0.0484 |

The wrong reduction reads **higher** at the end. **But that is one endpoint of a curve
whose sign is not constant**, and the run length is an arbitrary choice — so the endpoint alone is
not the finding:

| across all 120 steps | value |
| --- | --- |
| mean **signed** gap | +0.0137 |
| mean **absolute** gap | 0.0185 |
| steps where the wrong reduction read *lower* | 28 |

The signed mean is the one that carries a direction; the absolute mean is non-negative for every
possible run and would read identically if the finding reversed. Both curves and the per-step gap
are in `results/run.json` under `item_3_accumulation.curves`.

**The gap is small, and that is exactly why a bug of this shape lived inside every major training
framework until 2024.** The wrong curve does not look wrong. It looks like the right curve. And the
error is *exactly zero* whenever every micro-batch carries the same number of real tokens — which a
hand-built test case almost always does, so the fault was invisible to the checks that would have
caught it.

### 4 · A step where the gradient norm moved before the loss did

**A "typical step" is the median absolute change from one step to the next**, computed separately
for each trace. It is a unit of *size*, not of time — the loss and the gradient norm are in
different units, so a raw comparison would only measure which number happens to be bigger. Median
rather than mean, because a single large jump is exactly what is being looked for and must not
inflate the yardstick used to find it.

**A step qualifies on three conditions**, and the third is what makes this a claim about *before*:

1. the gradient norm moved at least **3** of its own typical steps,
2. the loss moved at most **1** of its own at that same step,
3. and the loss then made a comparably large move **within the next 5 steps**.

Drop the third and this becomes a same-step magnitude contrast published under a heading that
promises a lead in time. An earlier version of this section did exactly that.

**Step 131** is such a step. The gradient norm moved 4.3 typical steps; the loss moved 0.3 at that same step; the loss then moved 4.0, 4 step(s) later. Gradient norm 0.6089, loss 5.5338.

**1 of 200 steps qualify.** The threshold is arbitrary, so:

| threshold | qualifying steps |
| --- | --- |
| 2.0 | 11 |
| 2.5 | 5 |
| 3.0 | 1 |
| 4.0 | 1 |
| 5.0 | 0 |

**Read that spread before believing the count.** Qualifying steps thin out sharply as the threshold
rises — and vanish entirely at 5 — so this is one reading of an arbitrary cut rather than a stable
measurement.

**Why the gradient leads at all.** The loss is an average over a whole batch, so a change in what
the model is doing has to be large enough to move that average before it is visible. The gradient
norm is not an average over anything: it measures how hard the optimiser is pushing *right now*. A
run that logs only the loss finds out about its problems late.

The norm is logged **before** clipping. A trace of the post-clip norm flattens at the clip value,
which hides precisely the spikes the trace exists to show.

### 5 · MFU, computed honestly

| input | value |
| --- | --- |
| parameters priced | 5,719,808 |
| FLOPs per token | 34,318,848 |
| convention | 6 x 5,719,808 NON-EMBEDDING parameters — 2 forward, 4 backward; embedding lookups are gathers and do no arithmetic; attention's quadratic term excluded |
| tokens measured | 203,200 |
| wall clock | 7.535 s |
| achieved | 925.51 GFLOP/s |
| device peak | 3.336 TFLOP/s |
| device | this machine's CPU, 3.336 TFLOP/s sustained on a 2048^3 fp32 matrix multiply — MEASURED here, same device and dtype as the run, not a vendor figure |
| **MFU** | **27.74%** |
| tokens/second | 26,968 |

**Target 40%, achieved 27.74%, short by 12.26% of peak.**

**Two errors were caught in this number before it was published, and both flattered it.** The first
version divided FLOPs achieved on the **CPU** by a **GPU's** advertised peak and reported 39.13% —
a figure that looked excellent and compared two different processors. The peak is now *measured*, on
the same device and dtype as the run, with a large dense matrix multiply. The second counted the
embedding tables in the parameters: an embedding lookup is a gather that does no arithmetic at all,
so those parameters are free inflation: counting them made the numerator **45% larger than it should
have been**, which is the same thing as saying that removing them cut it by 31%.

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
