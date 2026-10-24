"""Build `notebooks/10-training-loop.ipynb`.

    uv run python src/exercises/10-training-loop/tools/build_notebook.py

**Tracked, like the notebook it writes**; edit this file and rebuild, never the notebook (a test
checks the two are byte-identical).

**The notebook's prose is scanned by `tests/test_forbidden_vocabulary.py` and by the leak checks**,
as every tracked notebook's is. So nothing here may name the source's own unit of material, and
nothing may sit close to its wording. Say what we decided and why.

**The notebook imports the package. It never re-implements it.** A notebook that copies the
gradient check or hard-codes an MFU figure drifts from what ships and then teaches something the
exercise does not claim. Everything below reads `trainloop.*`.
"""

import os
from pathlib import Path

EXERCISE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory -- `tests/test_notebook_builders.py`
#: passes a file, so a test never overwrites the copy a developer has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (EXERCISE.parents[2] / "notebooks" / f"{SLUG}.ipynb"))

REPO = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = (
    "https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/"
    f"blob/main/notebooks/{SLUG}.ipynb"
)

_next_id = 0


def _cell_id() -> str:
    """The next deterministic cell id, so rebuilding does not diff every cell."""
    global _next_id
    _next_id += 1
    return f"s10-{_next_id:02d}"


def md(text: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """A code cell with no stored output — outputs are stripped, always."""
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md(
            f"""# 10 · The training loop, made to tell the truth about itself

[![Open In Colab]({BADGE})]({BADGE})

A training loop will not tell you it is wrong. It will show you a loss going down.

This notebook runs six checks on a real loop, and **not one of them rewards a low loss**. Each is a
measurement of the loop, or a deliberate breakage of it:

1. every tensor shape in a step, and what each dimension means
2. one gradient, verified by hand against arithmetic
3. gradient accumulation, broken on purpose, with both curves plotted
4. the gradient norm logged every step, and a step where it moved before the loss did
5. MFU — and two ways this notebook's own first answer was wrong in the flattering direction
6. the number 0.1 written out in three float formats, bit by bit

Everything here imports the exercise's package. Nothing is re-implemented, so the notebook cannot
disagree with what ships or with `RESULTS.md`.

**Runtime: about a minute on a free Colab CPU.** Nothing needs a GPU.
"""
        ),
        code(
            f"""# Colab: clone the repo and install this exercise. Locally this is a no-op.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises")
    for member in ("09-loss-functions-output-heads", "10-training-loop"):
        subprocess.run(
            ["pip", "-q", "install", "-e", str(root / "src/exercises" / member)], check=True
        )
else:
    # **Found at run time, never baked in at build time.** This line used to interpolate the
    # BUILD machine's own absolute path, so the notebook imported on exactly one computer and
    # raised on every other one -- including for anyone following the submission link, which
    # pointed at this notebook while it was the only topic notebook tracked in the repository.
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit(
            "could not find the repository root by walking up from "
            + str(pathlib.Path.cwd())
            + " -- run this notebook from inside a clone of the repository, or on Colab"
        )
    for member in ("09-loss-functions-output-heads", "10-training-loop"):
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

print("repo root:", root)"""
        ),
        md(
            """## Every knob, in one cell

Nothing below is hard-coded. Change one of these, re-run from here, and watch what moves — and, more
usefully, what does not. Several cells end in `assert`, and those assertions are the lesson rather
than a formality: each states a property the library must hold, and would go red if it stopped
holding."""
        ),
        code(
            """# Every knob below is READ by the cells that follow. That sentence was false until a
# reviewer grepped for these names: STEPS and SEED appeared here and nowhere else, while the
# training cells carried their own literals -- so `LITE = True` printed "steps 40" and then trained
# 200, and the profile shortened one cell of three. Setting a knob nothing reads is worse than
# having no knob, because the print statement makes it look obeyed.
LITE   = True    # False runs the published configuration

# Two step counts, because the two training cells are different experiments and the published run
# used different lengths for them. Naming them separately is what lets `LITE` shorten both while the
# full profile still reproduces what `results/run.json` holds.
STEPS  = 40 if LITE else 200     # section 4: the instrumented run
CURVES = 24 if LITE else 120     # section 3: the two accumulation curves
DRAWS  = 20_000 if LITE else 200_000   # section 6: the float regression sweep
SEED   = 10      # the published seed

import matplotlib.pyplot as plt

from trainloop.config import Config

CONFIG = Config(seed=SEED)
print(f"profile  {'lite' if LITE else 'full'}   steps {STEPS}   curves {CURVES}   "
      f"draws {DRAWS:,}   seed {CONFIG.seed}")
print(f"model    {CONFIG.model.n_layer} blocks x {CONFIG.model.d_model} wide")"""
        ),
        md(
            """## 1 · Every tensor shape, and what each dimension means

Every tensor in one step is printed below, from the token ids to the head's gradient, and one of
them — the loss — has **no dimensions at all**. That is the point of printing them: everything before
it collapses into that scalar, and everything the optimiser does flows
back out of it, so a mistake anywhere in between changes training without changing a single shape.

Read the `head.weight.grad` row too: a gradient has exactly the shape of the weight it belongs to,
which is what makes the next section possible at all."""
        ),
        code(
            """from trainloop.config import Config
from trainloop.step import describe_shapes

config = CONFIG    # built with SEED above; a fresh Config() would ignore that knob
print(describe_shapes(config))"""
        ),
        md(
            """## 2 · Verify one gradient by hand

`backward()` reports a derivative, and a derivative is a **claim**: it says how much the loss would
move if this one weight moved a little. That is checkable without trusting anything. Nudge the
weight up, nudge it down, see how far the loss travelled, divide by how far you nudged.

$$\\frac{\\partial \\mathcal{L}}{\\partial w} \\approx
\\frac{\\mathcal{L}(w + h) - \\mathcal{L}(w - h)}{2h}$$

The **central** form — both directions — matters. A one-sided difference carries error proportional
to $h$; this one cancels that term and leaves error proportional to $h^2$.

**Two things had to be got right before this worked at all, and both are worth knowing:**

- **Choose a weight with a large gradient.** A weight the loss barely depends on moves it by almost
  nothing, so "the two agree" would be a statement about two zeros.
- **Compute in float64.** In float32 a loss near 9.2 resolves to about `5e-7`. A gradient small
  enough to move the loss by less than that gives a numeric estimate of **exactly zero** at every
  nudge size — which reads as a broken implementation and is a broken *instrument*."""
        ),
        code(
            """import torch

from lossheads.losses import cross_entropy
from lossheads.shift import shift_for_next_token
from trainloop import gradcheck
from trainloop.step import build

trunk, head, _ = build(config)
trunk, head = trunk.double(), head.double()          # see above: fp32 cannot answer this
torch.manual_seed(config.seed)
tokens = torch.randint(0, config.model.vocab_size, (config.model.batch_size, config.model.seq_len))


def loss_at():
    with torch.no_grad():
        logits = head(trunk(tokens))
        _, targets = shift_for_next_token(tokens)
        return cross_entropy(
            logits[:, :-1].reshape(-1, config.model.vocab_size), targets.reshape(-1), config.model
        )


logits = head(trunk(tokens))
_, targets = shift_for_next_token(tokens)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, config.model.vocab_size), targets.reshape(-1), config.model
)
head.weight.grad = None
loss.backward()

flat = int(head.weight.grad.abs().argmax())          # the largest gradient, not element [0, 0]
index = (flat // head.weight.shape[1], flat % head.weight.shape[1])

checks = gradcheck.sweep(loss_at, head.weight, index)
print(f"weight under test: head.weight{list(index)}\\n")
print(gradcheck.report(checks))"""
        ),
        md(
            """**Read the column, not the row.**

Agreement improves as the nudge shrinks — the error goes as $h^2$ — and then gets **worse again**.
Once $h$ is small enough, $\\mathcal{L}(w+h)$ and $\\mathcal{L}(w-h)$ stop differing in bits the
float type actually keeps, and the subtraction is measuring rounding noise rather than the function.

So the honest result is a **window**, not a best value. A check that agrees at exactly one nudge
size has not been verified; it has been fitted.

**Try it:** re-run the cell above in float32 (`trunk.float()`, `head.float()`) and watch the *fine*
end of the column degrade while the coarse end stays fine — the loss still moves measurably at
`1e-1`, and only stops doing so once the nudge is small.

To see the total collapse the exercise's write-up describes, pick a weight with a *tiny* gradient
instead of the largest one: `index = (0, 0)` in float32 gives a relative error of 1.0 at every
epsilon, because that weight moves the loss by less than fp32 can represent."""
        ),
        md(
            """## 3 · Break gradient accumulation on purpose

When the batch you want will not fit in memory, you split it into micro-batches, combine their
losses, and take one optimiser step. **The combination has to weight each micro-batch by how many
real tokens it holds.**

Averaging the averages instead gives a micro-batch with two real tokens exactly the same vote as one
with four. Here is what that costs, on three micro-batches:"""
        ),
        code(
            """from trainloop.accumulation import compare

print(compare(config))"""
        ),
        md(
            """**A bug of this shape lived inside every major training framework until 2024, and the
reason it survived is the important part.**

The error is **exactly zero** whenever every micro-batch happens to hold the same number of real
tokens — which, in a hand-built test case, they almost always do. So the fault was invisible to
precisely the checks that would have caught it. Curves looked ordinary. Nothing raised.

That is why `Config.micro_batch_tokens` is uneven **by decision**, and why `compare()` raises rather
than returning a gap of zero on an even configuration. A zero here would say the experiment was
blind, not that the code is right.

**Try it:** `compare(Config(micro_batch_tokens=(4, 4, 4)))` and read the error message."""
        ),
        md(
            """### The same two reductions, driving a real run

Four lines of arithmetic show the bug. Two curves show what it does. Everything is held identical
between them — initialisation, batches, optimiser, order — so the only difference is the reduction.

This takes about twenty seconds."""
        ),
        code(
            """from trainloop.accumulation import two_curves

curves = two_curves(config, steps=CURVES)
print(f"micro-batch widths : {curves['micro_batch_widths']} tokens")
print(f"correct reduction  : final loss {curves['final_correct']:.4f}")
print(f"wrong reduction    : final loss {curves['final_wrong']:.4f}")
print(f"gap                : {curves['final_gap']:+.4f}")
print(f"mean absolute gap  : {curves['mean_absolute_gap']:.4f}")"""
        ),
        code(
            """import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(curves["steps"], curves["correct"], label="correct: total loss / total tokens")
axes[0].plot(curves["steps"], curves["wrong"], label="wrong: mean of the means")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].legend()
axes[0].set_title("Both curves. This is the problem.")

gap = [w - c for c, w in zip(curves["correct"], curves["wrong"])]
axes[1].plot(curves["steps"], gap, color="crimson")
axes[1].axhline(0, color="0.7", linewidth=0.8)
axes[1].set_xlabel("step"); axes[1].set_ylabel("wrong minus correct")
axes[1].set_title("The same data, as a difference.")
plt.tight_layout(); plt.show()"""
        ),
        md(
            """**Look at the left panel first, then the right one.**

On the left the two curves are almost on top of each other. That is not a disappointing result — it
is the finding. The wrong curve does not look wrong. It looks like the right curve, which is exactly
why nobody noticed for years.

The right panel is the same data with the difference plotted directly, and only there does the bug
become obvious. **A defect you can only see by subtracting two things that both look fine is a
defect that ships.**"""
        ),
        md(
            """## 4 · Log the gradient norm, and find where it led

The loss is an average over a whole batch. For a change in what the model is doing to show up in it,
that change has to be big enough to move an average — so the loss is **lagging evidence**.

The gradient norm is not an average over anything. It measures how hard the optimiser is pushing
right now, and it moves first. A run that logs only the loss finds out about its problems late.

The norm below is taken **before** clipping. A trace of the post-clip norm flattens at the clip
value, which hides exactly the spikes the trace exists to reveal."""
        ),
        code(
            """from trainloop.step import run
from trainloop.telemetry import find_leading_steps, robustness

trace, facts = run(config, steps=STEPS)
leading = find_leading_steps(trace)

if leading:
    first = leading[0]
    print(f"step {first.step}: grad norm moved {first.grad_move:.1f} typical steps, "
          f"loss moved {first.loss_move:.1f}")
    print(f"{len(leading)} of {len(trace.steps)} steps qualify\\n")
else:
    print("no step qualified — which is the result, not a failure of the search\\n")

print("the threshold is arbitrary, so:")
for name, count in robustness(trace).items():
    print(f"   {name}: {count}")"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(trace.steps, trace.loss, label="loss", color="steelblue")
ax.set_xlabel("step"); ax.set_ylabel("loss", color="steelblue")

twin = ax.twinx()
twin.plot(trace.steps, trace.grad_norm, label="grad norm", color="darkorange", alpha=0.8)
twin.set_ylabel("gradient norm (pre-clip)", color="darkorange")

for step in [entry.step for entry in leading]:
    ax.axvline(step, color="0.85", zorder=0)

ax.set_title("Loss and gradient norm. Grey lines mark where the gradient moved and the loss did not.")
plt.tight_layout(); plt.show()"""
        ),
        md(
            """## 5 · Compute your own MFU, honestly

**MFU** — model FLOPs utilisation — is the fraction of the machine's arithmetic the run actually
used. Tokens per second says nothing on its own; this puts a denominator under it.

$$\\text{MFU} = \\frac{\\text{FLOPs the step needed}}{\\text{FLOPs the device could have done in
that time}}$$

**It is trivially inflated, and this exercise inflated it twice before catching itself.** Both
mistakes were in the ratio and both made the number look better -- and they were on opposite
sides of it, which is worth separating because the corrections pull in opposite directions:

- **The denominator.** The first version divided FLOPs achieved on the **CPU** by a **GPU's**
  advertised peak. It reported **39.13%**, which looked excellent and compared two different
  processors.
- **The numerator.** It also counted the **embedding tables** in the parameters. An embedding
  lookup is a *gather* — it
  reads one row per token and does no arithmetic at all — so those parameters were free inflation.
  Counting them made the numerator **45% larger than it should have been**, which is the same thing
  as saying that removing them cut it by 31%. Both numbers describe the same correction; quoting the
  wrong one of the pair is how a right figure ends up answering a different question.

So the peak below is **measured**, on this machine, at this dtype, with a large dense matrix
multiply. It is lower than any datasheet figure, and therefore reports a worse MFU. That is the
honest direction to be wrong in."""
        ),
        code(
            """from trainloop import mfu

peak = mfu.measured_peak_flops("cpu")
utilisation = mfu.measure(
    parameters=facts["non_embedding_parameters"],
    tokens=sum(trace.tokens),
    seconds=sum(trace.seconds),
    device_peak_flops=peak,
    device_name=f"this machine's CPU, {peak / 1e12:.3f} TFLOP/s measured on a 2048^3 fp32 matmul",
)
print(utilisation)
print()
print(mfu.distance_to_target(utilisation))"""
        ),
        md(
            """**Try it:** raise `d_model` in the config and re-run. MFU should climb, because the gap
is dominated by the model being too small to keep the machine busy rather than by anything in the
code."""
        ),
        md(
            """## 6 · One tenth, in three float formats

A fixed-width slot has to hold numbers many orders of magnitude apart, and counting in binary cannot
span that. So the bits get three jobs: one **sign** bit, some **exponent** bits fixing the
magnitude, and the remaining **mantissa** bits choosing which value of that magnitude. Scientific
notation does the same thing — in $6.02 \\times 10^{23}$ the exponent picks the scale and the
leading digits pick the value.

**Exponent bits buy range. Mantissa bits buy precision. More of one is always less of the other.**

0.1 is a good number to ask about because **no binary format holds it**: one tenth is
`0.0001100110011…` repeating, exactly as one third is `0.333…` repeating in decimal. Every format
below stores something else, and the only question is how much else.

The patterns are **built from arithmetic** in `floats.py`, not read out of the machine — and then
checked against the framework's own casts, because a decomposition that agrees only with itself
proves nothing."""
        ),
        code(
            """from trainloop.floats import FORMATS, decompose

for fmt in FORMATS:
    taken = decompose(0.1, fmt)
    print(f"\\n{fmt.name}  —  1 sign + {fmt.exponent_bits} exponent + "
          f"{fmt.mantissa_bits} mantissa = {fmt.total_bits} bits")
    print(taken.working())"""
        ),
        md(
            """### The bug this module shipped, measured rather than remembered

`_round_to_nearest_even` used to return an overflow flag computed from *this value's* fraction bit
length where the fixed 23 was meant, and `decompose` then applied a second exponent increment on top
of its own normalisation — returning a number **exactly twice** the right one, or raising where the
shift count went negative.

**It shipped because `0.1` was the only value the tests drove**, and `0.1` is one of the values where
the flag cannot fire. Run it yourself: the cell below replays the shipped code against the current
one over uniform draws in `[1, 2)` — every normal float is a significand in that interval times a
power of two, and the bug lives entirely in the significand, so that range is the whole space rather
than a slice of it."""
        ),
        code(
            """from trainloop.floats import regression_rate

for fmt in FORMATS:
    if fmt.mantissa_bits >= 23:
        continue
    r = regression_rate(fmt, draws=DRAWS)
    print(f"{fmt.name:<10} {r['doubled_rate']:6.2%} returned exactly twice the right value,"
          f" {r['raised_rate']:6.2%} raised   (over {r['draws']:,} draws)")

print()
print("Two different defects. A crash is loud; a value twice too large is not, and an earlier")
print("version of this exercise published a single figure that merged them.")
print()
print("Try it at 0.1 alone -- the value the original test drove:")
from trainloop.floats import _decompose_that_shipped, BF16, FP8_E4M3
for fmt in (BF16, FP8_E4M3):
    shipped = _decompose_that_shipped(0.1, fmt)
    correct = decompose(0.1, fmt).stored
    print(f"  {fmt.name:<10} shipped {shipped!r}  correct {correct!r}  agree {shipped == correct}")
print()
print("Both agree. That is the whole reason it shipped, and the reason a sweep is the check.")"""
        ),
        code(
            """# The independent check: does what we derived match what torch actually stores?
for fmt, dtype in zip(FORMATS, (torch.float32, torch.bfloat16, torch.float8_e4m3fn)):
    mine = decompose(0.1, fmt).stored
    theirs = float(torch.tensor(0.1, dtype=torch.float32).to(dtype).to(torch.float64))
    print(f"{fmt.name:<10} derived {mine!r:<22} torch {theirs!r:<22} "
          f"{'match' if mine == theirs else 'DIFFER'}")"""
        ),
        code(
            """from trainloop.floats import report

print(report(0.1))"""
        ),
        md(
            """**Which would I train in? bf16**, and one column of that table decides it.

bf16 keeps fp32's **eight** exponent bits and spends the entire saving out of the mantissa. So it
has fp32's *range*: a gradient that would underflow to zero in fp16 survives here, and none of the
loss-scaling machinery fp16 needs is required. It pays in precision — and gradient descent tolerates
imprecision far better than it tolerates zeros.

**fp8 E4M3 is a different decision, not a further step along the same one.** Four exponent bits give
it a far narrower range, and it does not reserve the all-ones exponent for infinity — which is why
it reaches 448 rather than stopping lower. It is a format for weights and activations under a
scaling scheme that keeps values inside that range, not a drop-in replacement.

**Try it:** `decompose(0.1, Format("custom", exponent_bits=5, mantissa_bits=2))` and watch the error
move as you trade the bits."""
        ),
        md(
            """## What this notebook cannot tell you

- **Nothing about whether the model is any good.** Every item measures the loop, not what it
  produced. The losses are incidental.
- **MFU here is a laptop CPU in fp32.** It says nothing about how the same code would utilise an
  accelerator, and comparing it to a published bf16 figure would compare two different quantities.
- **The gradient check verifies one weight.** It is evidence about autograd *there*, not a proof
  about the whole graph.
- **The `6N` FLOPs estimate is a convention**, not a measurement: it excludes attention's quadratic
  term and treats every non-embedding parameter as two forward and four backward operations.

The full write-up, with every figure generated rather than typed, is in the exercise's
[`RESULTS.md`](../src/exercises/10-training-loop/RESULTS.md)."""
        ),
    ]


WHERE = """## Where this sits

Notebook 10 of 14 — [the map of all fourteen](README.md). Previous: [`09-loss-functions-output-heads.ipynb`](09-loss-functions-output-heads.ipynb).
Next: [`11-optimizers-lr-schedules.ipynb`](11-optimizers-lr-schedules.ipynb).

**Takes in:** exercise 09's model, shift, masks, loss and corpus — imported, unchanged.
"""

CARRY = """## Carry forward

**What the next notebooks take from here:** the habits of a loop that reports on itself — a
gradient checked by hand, accumulation that weights by token count, the gradient norm logged every
step, and a utilisation figure whose numerator and denominator are counted on the same device.

**Next:** exercise 11 keeps a loop of this shape and changes what happens inside the optimiser
step — bias correction, warmup and the learning-rate schedule.
"""


def linked(body: list[dict]) -> list[dict]:
    """The notebook with its place in the fourteen-notebook path stated at both ends."""
    return [body[0], md(WHERE), *body[1:], md(CARRY)]


def build() -> dict:
    """The whole notebook."""
    return {
        "cells": linked(cells()),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the notebook, outputs stripped."""
    import json

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
