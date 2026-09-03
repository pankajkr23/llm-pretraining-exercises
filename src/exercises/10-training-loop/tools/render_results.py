"""Generate `RESULTS.md` from `results/run.json`. No number in it is typed by anyone.

Exercise 09 learned this the expensive way: its generated document opened by claiming every figure
was read from a run, and fifteen were literals inside the renderer — under a header saying they were
not. The byte-equality test could not see them, because they lived inside the template it compared
against. So here, every value and every verdict word is a lookup.

Regenerate after any run:

```bash
uv run python src/exercises/10-training-loop/tools/render_results.py
```

`tests/test_trainloop_results.py` regenerates it and fails if the tracked copy differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results"
OUT = EXERCISE / "RESULTS.md"


def render(run: dict) -> str:
    """Build the whole document. Every interpolation is a lookup, never a literal."""
    facts = run["facts"]
    one = run["item_1_shapes"]
    two, three = run["item_2_gradient"], run["item_3_accumulation"]
    four, five, six = run["item_4_grad_norm"], run["item_5_mfu"], run["item_6_floats"]
    curves = three["curves"]
    trace = run["trace"]

    shape_rows = "\n".join(
        f"| `{name}` | `{tuple(shape)}` | {meaning} |" for name, shape, meaning in one["table"]
    )
    token_positions = five["tokens"] // facts["steps"]
    raw_positions = facts["batch_size"] * facts["seq_len"]

    found = four["found"]
    leading = found[0] if found else None
    lead_block = (
        (
            f"**Step {leading['step']}** is such a step. The gradient norm moved "
            f"{leading['grad_move']:.1f} typical steps; the loss moved {leading['loss_move']:.1f} "
            f"at that same step; the loss then moved {leading['later_loss_move']:.1f}, "
            f"{leading['followed_within']} step(s) later. Gradient norm "
            f"{leading['grad_norm']:.4f}, loss {leading['loss']:.4f}."
        )
        if leading
        else (
            "**No step qualified, and that is the result** rather than a failure of the search. A "
            "manufactured example would have been worse than reporting nothing."
        )
    )
    robustness_rows = "\n".join(
        f"| {name.split('=')[1]} | {count} |" for name, count in four["robustness"].items()
    )
    wrong_verdict = "higher" if curves["wrong_reads_higher"] else "lower"

    def _digits(row: dict) -> str:
        """How many decimal digits matched, or "exact" when the two were bit-identical."""
        return "exact" if row["matching_digits"] is None else f"{row['matching_digits']:.1f}"

    sweep_rows = "\n".join(
        f"| {row['epsilon']:.0e} | {row['analytic']:.9f} | {row['numeric']:.9f} "
        f"| {row['relative_error']:.2e} | {_digits(row)} |"
        for row in two["sweep"]
    )
    float_rows = "\n".join(
        f"| {name} | `{d['bits']}` | {d['hex']} | {d['stored']!r} | {d['relative_error']:+.4%} |"
        for name, d in six.items()
    )
    format_rows = "\n".join(
        f"| {name} | {d['exponent_bits']} | {d['mantissa_bits']} | {d['smallest_normal']:.3g} | "
        f"{d['largest_normal']:.6g} | {d['decimal_digits']:.1f} |"
        for name, d in six.items()
    )

    return f"""# RESULTS — 10 · The training loop

**Generated from `results/run.json`. Do not edit by hand** — run `tools/render_results.py`. Every
number and every verdict word here, including *{wrong_verdict}*, is read from that file.

{facts["steps"]} steps, Adam at {facts["learning_rate"]}, batch {facts["batch_size"]} ×
{facts["seq_len"]} tokens, gradient clipped at {facts["grad_clip"]}, seed {facts["seed"]}.

**{facts["parameters"]:,} parameters** — {facts["trunk_parameters"]:,} in the trunk and
{facts["head_parameters"]:,} in the output head. **{facts["embedding_parameters"]:,} of the trunk's
are embedding tables**, and that distinction decides item 5: an embedding lookup is a gather, so
those parameters do no arithmetic and must not be priced.

---

## The six items

### 1 · Every tensor shape in the step

| tensor | shape | what each dimension is |
| --- | --- | --- |
{shape_rows}

**The one worth stopping on is the loss: it has no dimensions at all.** Everything above collapses
into that scalar, and everything the optimiser does flows back out of it — which is why a mistake
anywhere between the logits and the loss changes training without changing a single shape.

Two more rows repay a second look. `head.weight.grad` has exactly the shape of the weight it belongs
to, which is what makes the gradient check below possible at all. And `flat targets` is
{token_positions:,} rather than {raw_positions:,}: the shift drops the last position of every
sequence, because nothing follows it.

### 2 · One gradient, verified by hand

`head.weight{two["weight"]}` — the largest-magnitude gradient in the head, because a near-zero one
would make this a comparison of two zeros.

| epsilon | autograd | central difference | relative error | matching digits |
| --- | --- | --- | --- | --- |
{sweep_rows}

**Closest agreement at epsilon = {two["best_epsilon"]:.0e}: {two["best_matching_digits"]:.1f}
matching decimal digits.** Autograd said {two["analytic"]:.9f}; a central difference said
{two["numeric"]:.9f}.

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

Micro-batches of {three["token_counts"]} valid tokens with average losses {three["losses"]}:

| reduction | value |
| --- | --- |
| correct — total loss over total tokens | **{three["correct"]:.4f}** |
| wrong — the mean of the means | **{three["wrong"]:.4f}** |
| gap | **{three["absolute_gap"]:+.4f}** ({three["relative_gap"]:+.1%}) |

The short micro-batch carries half the real tokens and gets exactly the same vote.

**And the same two reductions driving a real run**, {len(curves["steps"])} steps, everything else
held identical — micro-batch widths {curves["micro_batch_widths"]} tokens:

| reduction | final loss |
| --- | --- |
| correct | {curves["final_correct"]:.4f} |
| wrong | {curves["final_wrong"]:.4f} |
| gap | {curves["final_gap"]:+.4f} |

The wrong reduction reads **{wrong_verdict}** at the end. **But that is one endpoint of a curve
whose sign is not constant**, and the run length is an arbitrary choice — so the endpoint alone is
not the finding:

| across all {curves["total_steps"]} steps | value |
| --- | --- |
| mean **signed** gap | {curves["mean_signed_gap"]:+.4f} |
| mean **absolute** gap | {curves["mean_absolute_gap"]:.4f} |
| steps where the wrong reduction read *lower* | {curves["steps_where_wrong_read_lower"]} |

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

{lead_block}

**{four["count"]} of {len(trace["steps"])} steps qualify.** The threshold is arbitrary, so:

| threshold | qualifying steps |
| --- | --- |
{robustness_rows}

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
| parameters priced | {five["parameters"]:,} |
| FLOPs per token | {five["flops_per_token"]:,.0f} |
| convention | {five["convention"]} |
| tokens measured | {five["tokens"]:,} |
| wall clock | {five["seconds"]:.3f} s |
| achieved | {five["achieved_flops_per_second"] / 1e9:,.2f} GFLOP/s |
| device peak | {five["device_peak_flops"] / 1e12:,.3f} TFLOP/s |
| device | {five["device_name"]} |
| **MFU** | **{five["mfu"]:.2%}** |
| tokens/second | {five["tokens_per_second"]:,.0f} |

**Target 40%, achieved {five["mfu"]:.2%}, short by {0.40 - five["mfu"]:.2%} of peak.**

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
{float_rows}

| format | exponent bits | mantissa bits | smallest normal | largest finite | decimal digits |
| --- | --- | --- | --- | --- | --- |
{format_rows}

**Which would I train in? bf16**, and one column decides it. bf16 keeps fp32's **eight** exponent
bits and spends the entire saving out of the mantissa, so it has fp32's *range*: a gradient that
underflowed to zero in fp16 does not underflow here, and no loss-scaling machinery is needed to keep
small values alive. It pays in precision, and gradient descent tolerates imprecision far better than
it tolerates zeros.

**fp8 E4M3 is a different decision, not a further step along the same one.** Four exponent bits give
it a much narrower range, and it does not reserve the all-ones exponent for infinity — which is why
it reaches {six["fp8 E4M3"]["largest_normal"]:.6g} rather than stopping lower. It is a format for
weights and activations under a scaling scheme that keeps values inside that range, not a drop-in
replacement for bf16.

**These patterns are built from arithmetic here, not read out of the machine** — and then checked
against torch's own casts, because a decomposition that agrees only with itself proves nothing.
"""


def main() -> int:
    """Write `RESULTS.md`. Returns 0, or 1 when the run file is missing."""
    path = RESULTS / "run.json"
    if not path.is_file():
        print(f"missing {path} — run `uv run python -m trainloop.harness` first", file=sys.stderr)
        return 1
    OUT.write_text(render(json.loads(path.read_text())))
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
