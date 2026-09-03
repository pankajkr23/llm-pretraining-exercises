# 10 · The training loop

**A training loop will not tell you it is wrong. It will show you a loss going down.** This exercise
takes a small model and a real loop and makes it report on itself: every shape named, one gradient
checked against arithmetic by hand, gradient accumulation broken on purpose so the gap is visible
rather than asserted, and a utilisation figure whose every input is stated because the number is
trivially inflated.

The headline: **the utilisation figure was wrong in the flattering direction twice over** — once in
its denominator and once in its numerator — and both were caught by asking what each half of the
ratio was actually counting.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is), which explains the
  problem before any of the machinery.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it).
- **Deciding whether to believe it** — the numbers are in **[RESULTS.md](RESULTS.md)**, generated
  from `results/run.json` and never typed. Then read
  [What this cannot establish](#what-this-cannot-establish).

## What this is

**First, three words this page cannot do without.** A model reads text and guesses what comes next.
The **loss** is one number saying how wrong those guesses were — lower is better, and "training" is
the process of nudging the model's numbers to make it smaller. The model splits into a **trunk**,
which turns text into a vector per position, and a **head**, the final layer turning each vector
into a score for every word it could pick. That split matters here: the trunk and the head are
counted differently in item 5, and the head is where the gradient check does its work.

Six things a training step should be able to say about itself, each of which is a measurement or a
deliberate breakage. None of them rewards a low loss.

**The shapes.** Six tensors move through a step and the last has no dimensions at all. Everything
collapses into that scalar and everything the optimiser does flows back out of it — which is why a
mistake anywhere in between changes training without changing a single shape.

**One gradient, checked by hand.** `backward()` reports a derivative, and a derivative is a claim
about what happens to the loss when a weight moves. That is checkable: move it, see what the loss
did, divide.

The interesting part is that the nudge size can be **too big or too small, and it fails differently
at each end**. Too big, and you are drawing a straight line across a curve: the wider you draw it,
the less it matches the slope at the middle. Too small, and the two losses you are subtracting stop
differing in bits the float type actually keeps, so you are measuring rounding noise rather than the
function. The honest answer is therefore a **window**, not a value.

**Gradient accumulation, broken on purpose.** When the batch you want does not fit in memory, you
split it, combine the pieces, and take one step. The combination has to weight each piece by how
many real tokens it holds. Averaging the averages instead gives a short micro-batch the same vote as
a long one — and a bug of exactly this shape lived inside every major training framework until 2024,
because **the error is exactly zero when the micro-batches happen to be equal length**, which in a
hand-built test they almost always are.

**The gradient norm, logged every step.** The loss is an average over a whole batch, so a change in
what the model is doing must be large enough to move that average before it becomes visible. The
gradient norm measures how hard the optimiser is pushing right now, and it moves first.

**MFU.** What fraction of the machine's arithmetic the run actually used — the work the step needed,
over the work the hardware could have done in the same time. Tokens per second says nothing without
that second number under it. **40% is the target here because it is roughly what a well-tuned large
training run achieves**; it is a rule of thumb rather than a physical limit, and it is quoted as one.

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

**The accumulation gap is 15.4% on the worked arithmetic** and much smaller on a real run: 0.0484
of a loss of 5.2658, so under 1%. That smallness is the point — the wrong curve does not look wrong,
it looks like the right curve, and you only see the difference by subtracting one from the other.

**And the gap's sign is not stable across the run**, which the run's own data shows: the wrong curve
reads higher at the end, and lower at some earlier steps. A single endpoint decides both the
magnitude and the verdict word, so `RESULTS.md` reports the mean absolute gap beside the final one.

**One number was caught being wrong in the flattering direction twice over** — once in its
denominator and once in its numerator:

- **MFU was 39.13%.** It divided FLOPs achieved on the **CPU** by a **GPU's** advertised peak — two
  different processors, one ratio. The peak is now measured on the same device and dtype as the run,
  with a large dense matrix multiply.
- **And it counted the embedding tables.** An embedding lookup is a *gather* — it reads one stored
  row per token and does no arithmetic at all — so those parameters are free inflation. Counting
  them made the numerator **45% larger than it should have been**; equivalently, removing them cut
  it by 31%. Both describe the same correction, and quoting the wrong one of the pair is how a
  right figure ends up answering a different question.

The honest figure is **27.69%**, roughly 12 points short of the 40% rule of thumb — and accounting
for that gap, rather than closing it, is the useful part. It moves by a few tenths of a point between
runs, because the numerator is fixed and the denominator is a wall clock on a shared machine.

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
