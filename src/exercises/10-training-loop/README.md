# 10 · The training loop

**A training loop will not tell you it is wrong. It will show you a loss going down.** This exercise
takes a small model and a real loop and makes it report on itself: every shape named, one gradient
checked against arithmetic by hand, gradient accumulation broken on purpose so the gap is visible
rather than asserted, and a utilisation figure whose every input is stated because the number is
trivially inflated.

The headline: **two numbers in this exercise were wrong in the flattering direction before they were
published**, and both were caught by asking what the denominator actually was.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is), which explains the
  problem before any of the machinery.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it).
- **Deciding whether to believe it** — the numbers are in **[RESULTS.md](RESULTS.md)**, generated
  from `results/run.json` and never typed. Then read
  [What this cannot establish](#what-this-cannot-establish).

## What this is

Six things a training step should be able to say about itself, each of which is a measurement or a
deliberate breakage. None of them rewards a low loss.

**The shapes.** Six tensors move through a step and the last has no dimensions at all. Everything
collapses into that scalar and everything the optimiser does flows back out of it — which is why a
mistake anywhere in between changes training without changing a single shape.

**One gradient, checked by hand.** `backward()` reports a derivative, and a derivative is a claim
about what happens to the loss when a weight moves. That is checkable: move it, see what the loss
did, divide. The interesting part is that the nudge size has a floor *and* a ceiling — too large and
curvature shows, too small and the two losses stop differing in bits the float type keeps — so the
honest answer is a window, not a value.

**Gradient accumulation, broken on purpose.** When the batch you want does not fit in memory, you
split it, combine the pieces, and take one step. The combination has to weight each piece by how
many real tokens it holds. Averaging the averages instead gives a short micro-batch the same vote as
a long one — and a bug of exactly this shape lived inside every major training framework until 2024,
because **the error is exactly zero when the micro-batches happen to be equal length**, which in a
hand-built test they almost always are.

**The gradient norm, logged every step.** The loss is an average over a whole batch, so a change in
what the model is doing must be large enough to move that average before it becomes visible. The
gradient norm measures how hard the optimiser is pushing right now, and it moves first.

**MFU.** What fraction of the machine's arithmetic the run actually used. Tokens per second says
nothing without knowing what the hardware could have done.

**And 0.1 in three float formats.** One tenth repeats forever in binary, exactly as one third does in
decimal, so no format holds it and each one misses by a different amount. Which you would train in
follows from one column of the table.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | every dimension a measurement is taken at, including the *uneven* micro-batch sizes |
| `step.py` | one optimiser step, every shape named, and the run that logs a sequence of them |
| `gradcheck.py` | a central difference against `backward()`, swept over the nudge size |
| `accumulation.py` | both reductions — as arithmetic, and as two curves from a real run |
| `telemetry.py` | per-step traces, and the search for a step where the gradient led |
| `mfu.py` | utilisation, with every input named and the device peak *measured* |
| `floats.py` | 0.1 in fp32, bf16 and fp8 E4M3, built from arithmetic |
| `harness.py` | one run producing every item, into `results/run.json` |

**The model is exercise 09's, imported rather than restated.** Its trunk, tokenizer, target shift,
masks and losses all apply unchanged, so the two exercises cannot disagree about what a loss is.

`floats.py` and `accumulation.py`'s arithmetic need no `torch` at all, which is deliberate: the
ordinary CI job runs real assertions about this exercise rather than collecting an empty file.

## Run it

```bash
uv sync --all-packages --extra train

uv run python -m trainloop.harness      # all six items -> results/run.json
uv run python src/exercises/10-training-loop/tools/render_results.py

uv run pytest src/exercises/10-training-loop
uv run pytest                           # and the repo-wide guards, which the line above misses
```

The harness prints as much as it computes — items 1, 3 and 6 are about what a reader can *see* — and
takes about thirty seconds on an M-series laptop.

## The evidence

**Every measured figure is in [RESULTS.md](RESULTS.md)**, generated from `results/run.json`.

**The gradient check agrees to 8.8 decimal digits at its best**, and the shape of the sweep is the
finding rather than that number: agreement improves as the nudge shrinks, then gets *worse* again
when the subtraction becomes rounding noise. A check that agrees at exactly one epsilon has been
fitted, not verified.

**The accumulation gap is 15.4% on the worked arithmetic** and much smaller on a real run — 0.0759
after 120 steps. That smallness is the point: the wrong curve does not look wrong, it looks like the
right curve.

**Two numbers were caught being wrong in the flattering direction**, and both were denominators:

- **MFU was 39.13%.** It divided FLOPs achieved on the **CPU** by a **GPU's** advertised peak — two
  different processors, one ratio. The peak is now measured on the same device and dtype as the run,
  with a large dense matrix multiply.
- **And it counted the embedding tables.** An embedding lookup is a gather that does no arithmetic,
  so those parameters are free inflation. Removing them cut the numerator by 45%. The honest figure
  is **27.89%**, with a real 12-point gap to account for — and accounting for it is what the
  requirement actually asks.

A third was caught before publication: the gradient check was written in fp32, where a loss near 9.2
resolves to about 5e-7, so a small gradient produced a numeric estimate of **exactly zero** at every
nudge size. It runs in float64.

**The float decompositions are built from arithmetic here, not read out of the machine** — and then
checked against torch's own casts for fp32, bfloat16 and float8_e4m3fn. All three match exactly, and
`FP8_E4M3.largest_normal` derives to 448.0 from the field widths alone, which is what the spec says.

## What this cannot establish

**Nothing here says whether the model is any good.** Every item is a measurement of the loop, not of
what the loop produced. The losses are incidental.

**MFU on a laptop CPU is not MFU on a training cluster.** The figure is honest about this machine and
says nothing about how the same code would utilise an accelerator. It is also an fp32 figure, so
comparing it to a published bf16 MFU would compare two different quantities.

**The `6N` FLOPs estimate is a convention, not a measurement.** It excludes attention's quadratic
term, which at this sequence length is small but not nothing, and it treats every non-embedding
parameter as participating in two forward and four backward operations per token.

**The gradient check verifies one weight.** It is evidence that autograd is right *there*, on the
element with the largest gradient in the head, and not a proof about the whole graph.

**The step at which the gradient norm led the loss is one reading of an arbitrary threshold.** The
count at four other thresholds is published beside it for that reason.
