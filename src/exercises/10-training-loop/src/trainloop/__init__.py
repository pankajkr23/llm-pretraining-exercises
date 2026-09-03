"""One optimiser step, made to tell the truth about itself.

Exercise 09 built everything up to the scalar — the target shift, the masks, the loss — and this one
builds the step around it. **Six items, and none of them rewards a low loss:** every one is a
measurement of the loop, or a deliberate breakage of it.

**`step.py`** names every tensor in a step, including the one with no dimensions at all. Everything
collapses into that scalar and everything the optimiser does flows back out of it, which is why a
mistake anywhere in between changes training without changing a single shape.

**`gradcheck.py`** treats `backward()` as a claim and checks it: move a weight, see what the loss
did, divide. The nudge size has a floor as well as a ceiling, so the answer is a window rather than
a value — and a check that agrees at exactly one epsilon has been fitted, not verified.

**`accumulation.py`** holds both reductions of a set of micro-batch losses, one of which is wrong.
It is wrong only when the micro-batches carry *different* numbers of real tokens — which is why a
bug of this shape lived inside every major training framework until 2024, and why `compare()`
refuses an even configuration rather than reporting a reassuring gap of zero.

**`telemetry.py`** logs the gradient norm alongside the loss — before clipping, because a post-clip
trace flattens at the clip value and hides the spikes it exists to show — and searches for a step
where the gradient moved and the loss did not.

**`mfu.py`** computes utilisation with every input named, because the figure is trivially inflated
by a wrong denominator. It was inflated twice here before publication: once by dividing CPU work by
a GPU's peak, once by pricing embedding tables that do no arithmetic.

**`floats.py`** builds 0.1's bit pattern in fp32, bf16 and fp8 E4M3 from field widths and rounding,
then the tests check each against the framework's own cast. It needs no `torch` to run, only to be
checked — so a fresh clone can take a float apart without installing the wheels.

The model is exercise 09's, imported rather than restated, so the two cannot disagree about what a
loss is.
"""
