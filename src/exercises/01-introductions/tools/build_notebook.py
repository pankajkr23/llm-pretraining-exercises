"""Build `notebooks/01-introductions.ipynb`.

    uv run python src/exercises/01-introductions/tools/build_notebook.py

The notebook runs the four experiments in this exercise's `foundations` package — a linear model
against one hidden layer, stacked layers with and without a nonlinearity, embeddings learned from
next-token prediction, and one network trained at growing dataset sizes. It imports the package
and never re-implements it; the cells only choose settings, print tables and draw figures.

It does not reproduce the deployed pages in `web/`. Those are hand-written browser JavaScript for a
reader; this is the same four ideas as tested Python code for someone who wants to change a
setting and watch what moves.
"""

import json
import os
from pathlib import Path

#: The exercise folder: the nearest parent holding a `pyproject.toml`, so the builder resolves the
#: same way from `tools/` and from anywhere it is staged inside the exercise.
EXERCISE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SLUG = EXERCISE.name
#: `NOTEBOOK_OUT` is the full output path, not a directory.
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
    return f"01-{_next_id:02d}"


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
            f"""# 01 · Four things a network needs before it can learn language

[![Open In Colab]({BADGE})]({BADGE})

Every later notebook in this repository trains some part of a language model. This one sets up the
four ideas they all lean on, each as a small experiment you can re-run with different settings:

1. **A nonlinearity.** Without one, a network can only draw straight boundaries, however wide it is.
2. **Depth.** Stacking linear layers buys nothing; stacking them *with* a nonlinearity does — and
   at a fixed parameter budget, whether an extra layer beats a wider one is a measurement, not a
   given.
3. **Embeddings.** Give each token a learned vector, train only on predicting the next token, and
   tokens used the same way end up with the same vector.
4. **Data.** One network, held fixed, memorises a small dataset and generalises from a large one.

All the code lives in this exercise's `foundations` package, which has its own tests. The cells
below pick settings, call it, and draw what comes back.

**Runtime:** well under a minute on a laptop CPU at the default `LITE` setting. No GPU needed.
"""
        ),
        md(
            """## Where this sits

**Previous:** none — this is the first notebook.
**Next:** `notebooks/02-tokenization.ipynb`.

**Takes in:** nothing from earlier exercises. Every input here is generated from a seed.

**Hands on:**

- The **hidden-layer network** of section 1 — a linear map, a nonlinearity, another linear map — is
  the feed-forward block inside every block of exercise 09's model, at four times the model width
  with GELU in the middle. Exercise 10 trains that model, and exercises 11–14 train decoders built
  from the same block.
- The **embedding table** of section 3 — one learned row per token id — is exercise 07's subject,
  and the first layer of exercise 09's model. Exercise 02, next, decides what the token ids are.
- The habit of reading a **held-out score** beside a training score, and of running **more than one
  seed** before ranking two settings, is used in every exercise that trains anything."""
        ),
        code(
            f"""# Colab: clone and install this exercise. Locally: import it from this checkout.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises")
    for member in ("01-introductions",):
        # `[train]` is the extra that brings torch; Colab usually has it already.
        target = str(root / "src/exercises" / member) + "[train]"
        subprocess.run(["pip", "-q", "install", "-e", target], check=True)
else:
    # Found at run time, never baked in at build time, so the notebook runs on any checkout.
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit(
            "could not find the repository root by walking up from "
            + str(pathlib.Path.cwd())
            + " -- run this notebook from inside a clone of the repository, or on Colab"
        )
    for member in ("01-introductions",):
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

print("repo root:", root)"""
        ),
        md(
            """## Every knob, in one cell

Each default is the setting this exercise settled on for its deployed demos: a 12-unit hidden layer
on the rings, a 2-D embedding so it can be drawn without any projection, a 2·64·1 network trained
at five dataset sizes from 20 to 2,000. Change any of them and re-run from here down.

`LITE = False` runs more seeds, trains the depth and data experiments for longer, and uses a ten
times larger text stream. It takes a minute or two rather than seconds."""
        ),
        code(
            """LITE = True

# Every experiment that compares settings runs once per seed; the others use the first one.
SEEDS = (1, 2, 3) if LITE else (1, 2, 3, 4, 5, 6, 7, 8)

# 1 · nonlinearity: one hidden layer on two concentric rings
HIDDEN = 12                                   # hidden units; try 2, 3, 4
ACTIVATIONS = ("none", "relu", "tanh", "gelu")  # "none" is the linear model
RING_NOISE = 0.15                             # jitter on the rings; try 0.3

# 2 · depth: a wide shallow network against a narrow deep one, at the same parameter budget
BUDGET = 300                                  # parameters each network may use; try 60
DEEP_LAYERS = 3                               # hidden layers in the deep one; try 2 or 5
DEPTH_STEPS = 1500 if LITE else 3000

# 3 · embeddings: next-token prediction on a toy grammar
EMBED_DIM = 2                                 # 2 can be drawn as-is; try 1 or 8
SENTENCES = 2000 if LITE else 20000

# 4 · data: one fixed network, several dataset sizes
SIZES = (20, 60, 200, 600, 2000)
DATA_NOISE = 0.3                              # how much the two half-moons overlap
DATA_STEPS = 1500 if LITE else 3000

from foundations.train import pick_device

# CUDA, then Apple's MPS, then CPU. These networks are so small that "cpu" is often faster:
# a GPU spends most of each step launching kernels. The device changes rounding, not results.
DEVICE = pick_device()

print(f"profile {'lite' if LITE else 'full'} | seeds {len(SEEDS)} | device {DEVICE}")"""
        ),
        md(
            """## 1 · A straight line cannot separate two rings

Two rings of points, one inside the other, labelled by which ring they are on. We train the same
network once per activation: two inputs, `HIDDEN` hidden units, one output. With activation `none`
the hidden layer is linear, so the whole network collapses to a single linear map and its boundary
is a straight line. With any real nonlinearity, it is not.

The last line is a control: the very same linear network, trained on two blobs that a line *can*
separate. If the linear model only failed because something in the code was broken, it would fail
there too."""
        ),
        code(
            """import matplotlib.pyplot as plt

from foundations import experiments
from foundations.config import NonlinearityConfig

ring_config = NonlinearityConfig(
    hidden=HIDDEN, activations=ACTIVATIONS, noise=RING_NOISE, seed=SEEDS[0]
)
rings = experiments.nonlinearity(ring_config, device=DEVICE)

print(f"{'activation':<11}{'params':>7}{'train acc':>11}{'test acc':>10}{'first loss':>12}"
      f"{'last loss':>11}")
for name, fit in rings.fits.items():
    print(f"{name:<11}{fit.parameters:>7}{fit.train_accuracy:>11.3f}{fit.test_accuracy:>10.3f}"
          f"{fit.losses[0]:>12.4f}{fit.losses[-1]:>11.4f}")
print(f"\\ncontrol — the linear network on two separable blobs: test acc "
      f"{rings.linear_on_blobs:.3f}")"""
        ),
        code(
            """fig, axes = plt.subplots(1, len(rings.fits), figsize=(3.6 * len(rings.fits), 3.6),
                         squeeze=False)
for ax, (name, fit) in zip(axes[0], rings.fits.items()):
    x, y = fit.train_points
    xs, ys, prob = experiments.decision_grid(fit.model, x)
    ax.contourf(xs, ys, prob, levels=20, cmap="RdBu_r", alpha=0.5)
    ax.contour(xs, ys, prob, levels=[0.5], colors="black", linewidths=1)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu_r", s=5)
    ax.set_title(f"{name}: test {fit.test_accuracy:.0%}")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("The black line is where the network switches its answer", y=1.02)
plt.tight_layout(); plt.show()"""
        ),
        md(
            """**Why the linear panel looks the way it does.** A linear model computes
$z = w_1 x_1 + w_2 x_2 + b$ and answers by the sign of $z$. The set where $z = 0$ is a straight
line, and every point on one side gets the same answer. Any line through the rings cuts both of
them, so the best it can do is put a slice of the outer ring on one side — a score barely above
the 50% that guessing gets on two equal-sized rings. Its loss hardly moves from where it started.

**Why one hidden layer is enough.** Each ReLU unit is zero on one side of its own line and rises
linearly on the other. The output adds `HIDDEN` of those ramps, so its surface is folded along
`HIDDEN` lines, and the region where it is positive can be a polygon. Three units are the fewest
that can enclose a region (a triangle), so `HIDDEN = 2` should fail and `HIDDEN = 3` should
sometimes succeed — try both. tanh and GELU bend smoothly instead of folding, and get there too.

**Parameters, by hand:** two inputs into `HIDDEN` units is $2h + h$ weights and biases, and `HIDDEN`
units into one output is $h + 1$; at $h = 12$ that is 49. The linear model has the same 49
parameters and still fails — the count is not what is missing.

**Caveat:** the rings are chosen to be impossible for a line. They show the limitation exists; they
say nothing about how often real data has this shape. One seed is shown here; the tests hold the
same claim over several."""
        ),
        md(
            """**Width does not rescue it.** The same linear model with 256 hidden units instead of
`HIDDEN`. More units, more parameters, the same straight line."""
        ),
        code(
            """from dataclasses import replace

wide = experiments.nonlinearity(replace(ring_config, hidden=256, activations=("none",)),
                                device=DEVICE)
fit = wide.fits["none"]
print(f"linear, 256 hidden units: {fit.parameters} params, test acc {fit.test_accuracy:.3f}")"""
        ),
        md(
            """## 2 · Depth: what an extra layer buys

**First, the collapse.** Two linear layers applied in a row are one linear layer:
$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)\\,x + (W_2 b_1 + b_2)$. So a stack of any depth with no
nonlinearity can be multiplied out into a single matrix and bias. The cell below builds random
stacks in float64, multiplies them out, and measures how far the stack's outputs are from the
single matrix's, as a fraction of the outputs' size. Then it does the same with ReLU between the
layers, where the multiplied-out matrix is a different function.

**Then, a fixed budget.** Two ReLU networks, each the widest that fits `BUDGET` parameters: one
with a single wide hidden layer, one with `DEEP_LAYERS` narrow ones. Both train on two interleaved
spirals, whose boundary winds, once per seed. A third network has the deep shape and no
activation, so it is one linear map wearing many layers.

**What I expected before running it:** the deep network to win. Each ReLU layer folds the space the
next layer sees, so in principle a deep network can cut the plane into more linear pieces per
parameter than a shallow one. Whether it does here is what the verdict line reports."""
        ),
        code(
            """from foundations.config import DepthConfig

depth_config = DepthConfig(
    budget=BUDGET, deep_layers=DEEP_LAYERS, steps=DEPTH_STEPS, seeds=SEEDS
)
depth_run = experiments.depth(depth_config, device=DEVICE)

print("a random stack against its own multiplied-out matrix (largest gap / largest output)")
for (layers, activation), gap in depth_run.collapse.items():
    print(f"   {layers} layers, {activation:<5} {gap:.1e}")"""
        ),
        code(
            """print(f"{'network':<12}{'widths':<20}{'params':>7}   test accuracy, one per seed")
for name, fits in depth_run.fits.items():
    per_seed = "  ".join(f"{acc:.3f}" for acc in depth_run.test_accuracies(name))
    print(f"{name:<12}{str(fits[0].widths):<20}{fits[0].parameters:>7}   {per_seed}")

verdict = experiments.spread_verdict(
    depth_run.test_accuracies("deep"), depth_run.test_accuracies("shallow")
)
readable = {
    "first higher": "deep scores higher, by more than the seed-to-seed spread",
    "second higher": "shallow scores higher, by more than the seed-to-seed spread",
    "within the seed spread": "no ranking: the difference is inside the seed-to-seed spread",
}
deep_wins = sum(
    d > s
    for d, s in zip(depth_run.test_accuracies("deep"), depth_run.test_accuracies("shallow"))
)
print(f"\\nverdict over {len(SEEDS)} seeds — {readable[verdict]}")
print(f"seed by seed, deep scored higher on {deep_wins} of {len(SEEDS)}")"""
        ),
        code(
            """panels = len(depth_run.fits)
fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.6), squeeze=False)
for ax, (name, fits) in zip(axes[0], depth_run.fits.items()):
    fit = fits[0]
    x, y = fit.train_points
    xs, ys, prob = experiments.decision_grid(fit.model, x)
    ax.contourf(xs, ys, prob, levels=20, cmap="RdBu_r", alpha=0.5)
    ax.contour(xs, ys, prob, levels=[0.5], colors="black", linewidths=1)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu_r", s=4)
    ax.set_title(f"{name} {fit.widths[1:-1]}: {fit.test_accuracy:.0%}")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle(f"seed {depth_run.seeds[0]} only — the table above has the rest", y=1.02)
plt.tight_layout(); plt.show()"""
        ),
        md(
            """**Reading the collapse.** float64 carries about 16 significant digits (its machine
epsilon is $2.2 \\times 10^{-16}$), so a linear stack's gap at that order is rounding and nothing
else: the stack *is* the single matrix. With ReLU the gap is a sizeable fraction of the output,
because the folds cannot be multiplied out. The `deep linear` row is the same fact in a training
run: it scores like any linear classifier on the spirals, however many layers it has.

**Reading the budget.** A one-hidden-layer network of width $h$ has $4h + 1$ parameters; one with
$L$ hidden layers of width $w$ has $3w + (L - 1)(w^2 + w) + w + 1$. The squared term is why the
deep network gets so few units per layer for the same budget.

If the verdict says *no ranking*, that is the finding: at this size and with this optimiser, the
extra layers bought nothing that survives a change of seed. The seed-by-seed count underneath is
the second reading — a network can win most seeds and still not clear the spread, when one seed
lands far from the rest. Look at the deep row's spread — a
narrow ReLU layer can lose units that switch off and never switch back on, and one bad seed is how
that shows. One seed alone could have ranked these two networks either way, which is why the
comparison runs on all of them.

**Try:** `BUDGET = 60` (both networks starve), `DEEP_LAYERS = 2` or `5`, `LITE = False` for eight
seeds. **Caveat:** depth's advantage in real models is about learning reusable features from huge
amounts of data, at scales nothing here reaches. This measures one toy problem and says so."""
        ),
        md(
            """## 3 · Embeddings: tokens that keep the same company end up in the same place

A stream of four-token sentences, `animal verb (animal | fruit) .`, with each word drawn at random
from its category. The model is as small as a language model gets: look up the current token's
vector in an embedding table, map it with one linear layer to a score for every possible next
token, train on next-token prediction. Nothing in the objective says which words are animals.

The same model is trained a second time on the **same tokens shuffled**: every word appears exactly
as often, but the order — and with it the company each word keeps — is gone."""
        ),
        code(
            """from foundations import data
from foundations.config import EmbeddingConfig

embed_config = EmbeddingConfig(n_sentences=SENTENCES, dim=EMBED_DIM, seed=SEEDS[0])
embed = experiments.embeddings(embed_config, device=DEVICE)

sample = data.grammar_stream(3, seed=SEEDS[0]).tolist()
print("a sample of the stream:", " ".join(data.VOCAB[i] for i in sample))
print(f"\\nloss: first step {embed.losses[0]:.4f}, last step {embed.losses[-1]:.4f}, "
      f"lowest possible {embed.entropy_floor:.4f}")
print(f"\\n{'table':<10}{'grouping score':>16}{'nearest word in same category':>32}")
for name, (score, purity) in embed.scores().items():
    print(f"{name:<10}{score:>16.3f}{purity:>32.0%}")"""
        ),
        code(
            """import torch

colours = {"stop": "black", "animal": "tab:blue", "fruit": "tab:orange",
           "verb": "tab:green"}
tables = {"before training": embed.before, "after training": embed.after,
          "trained on shuffled tokens": embed.shuffled}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (title, table) in zip(axes, tables.items()):
    # Columns 0 and 1. At EMBED_DIM = 2 that is the whole table; above it, only a slice of it.
    points = table[:, :2] if table.shape[1] > 1 else torch.cat([table, torch.zeros_like(table)], 1)
    for i, word in enumerate(data.VOCAB):
        colour = colours[data.CATEGORY_OF[word]]
        ax.scatter(points[i, 0], points[i, 1], color=colour, s=25)
        ax.annotate(word, (points[i, 0], points[i, 1]), fontsize=8, color=colour,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(title)
plt.tight_layout(); plt.show()"""
        ),
        code(
            """def top(row: torch.Tensor, k: int = 3) -> str:
    values, ids = row.topk(k)
    return ", ".join(f"{data.VOCAB[i]} {v:.2f}" for v, i in zip(values.tolist(), ids.tolist()))

for word in ("cat", "apple", "sees", "."):
    i = data.VOCAB.index(word)
    print(f"after {word!r:8} model: {top(embed.predicted[i])}")
    print(f"{'':15}true:  {top(embed.true[i])}")"""
        ),
        md(
            """**Why the words group.** The head turns a vector into next-token scores. `cat` and
`dog` are followed by exactly the same things, so they need the same scores, and the cheapest way
for both to get them is the same vector. Every gradient step pulls them toward that point.
Fruits are always followed by the full stop; verbs by an animal or a fruit; animals by a verb or
the full stop. Four different futures, so four places in the plane — the grouping is a by-product
of prediction, not a separate objective. The shuffled run shows that word order is the cause: same
words, same counts, no shared futures, no groups.

**The lowest possible loss, by hand.** A model that sees only the current token cannot beat the
grammar's own uncertainty, $H(\\text{next} \\mid \\text{current}) =
\\sum_c P(c)\\,H(\\text{next} \\mid c)$. Per sentence there is one full stop (followed by one of 5
animals: $\\ln 5$), one verb (followed by one of 10 objects: $\\ln 10$), one and a half animals
(two thirds of the time a verb follows, one third the full stop, so $\\tfrac23 \\ln 6 +
\\tfrac13 \\ln 3$) and half a fruit (always the full stop: 0). Weighting by those frequencies gives
about 1.56 nats. The printed floor is the same quantity measured on the sampled stream, and the
training loss should end just above it.

**The grouping score** is the average cosine similarity between words of the same category minus
the average between different categories: near 0 for unrelated vectors, up to 2 when each category
has its own direction. **Try:** `EMBED_DIM = 1` (four futures on a line), `EMBED_DIM = 8` (more room
than needed; the score holds, the picture shows only two of the eight columns), fewer `SENTENCES`.

**Where this goes:** this table is `nn.Embedding(15, 2)`. Exercise 09's model starts with the same
kind of table, one row for each of exercise 02's tokens and hundreds of columns, and exercise 07
asks what that table costs and whether a fixed code can stand in for it."""
        ),
        md(
            """## 4 · Data is the regulariser

One network, never changed: two inputs, 64 ReLU units, one output. It is trained from the same
initial weights, with the same optimiser and the same number of full-batch steps, on two noisy
half-moons at each size in `SIZES`, and every run is scored on the same 2,000 fresh points. The
moons overlap at this noise level, so no boundary can be right about every fresh point.

It runs once per seed, because a gap measured once is a gap measured with its noise included."""
        ),
        code(
            """import statistics

from foundations.config import DataSizeConfig

size_runs = [
    experiments.data_size(
        DataSizeConfig(sizes=SIZES, noise=DATA_NOISE, steps=DATA_STEPS, seed=seed), device=DEVICE
    )
    for seed in SEEDS
]

print(f"{'points':>7}{'train acc':>11}{'test acc':>10}{'gap':>8}{'gap range over seeds':>24}")
for n in SIZES:
    fits = [run.fits[n] for run in size_runs]
    gaps = [fit.gap for fit in fits]
    print(f"{n:>7}{statistics.fmean(f.train_accuracy for f in fits):>11.3f}"
          f"{statistics.fmean(f.test_accuracy for f in fits):>10.3f}"
          f"{statistics.fmean(gaps):>8.3f}{min(gaps):>12.3f} to {max(gaps):.3f}")
print(f"\\nparameters in the network: {size_runs[0].fits[SIZES[0]].parameters}")"""
        ),
        code(
            """first = size_runs[0]
shown = (min(SIZES), max(SIZES))
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, n in zip(axes, shown):
    fit = first.fits[n]
    x, y = fit.train_points
    xs, ys, prob = experiments.decision_grid(fit.model, first.test_points[0])
    ax.contourf(xs, ys, prob, levels=20, cmap="RdBu_r", alpha=0.5)
    ax.contour(xs, ys, prob, levels=[0.5], colors="black", linewidths=1)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu_r", s=12 if n < 100 else 3)
    ax.set_title(f"{n} points: train {fit.train_accuracy:.0%}, test {fit.test_accuracy:.0%}")
    ax.set_xticks([]); ax.set_yticks([])
for run in size_runs:
    axes[2].plot(SIZES, [run.fits[n].gap for n in SIZES], marker="o", alpha=0.7)
axes[2].set_xscale("log"); axes[2].axhline(0, color="0.7", linewidth=0.8)
axes[2].set_xlabel("training points"); axes[2].set_ylabel("train acc minus test acc")
axes[2].set_title("the gap, one line per seed")
plt.tight_layout(); plt.show()"""
        ),
        md(
            """**Reading it.** The network has $2 \\cdot 64 + 64 + 64 + 1 = 257$ parameters. At 20
points that is about thirteen per point — enough to bend the boundary around every one of them,
including the ones the noise put on the wrong side. Training accuracy reaches 100% and means
nothing: it is a record of the points, not of the moons. As the set grows, the same parameters can
no longer bend around each point, the boundary relaxes toward the true shape, and the gap closes
**from both sides**: test accuracy rises, and training accuracy falls toward the same ceiling,
because the overlap that defeats it on fresh points now defeats it on training points too.

Every run is full-batch for the same number of steps, so every point is seen the same number of
times at every size. Only how many points there are differs.

**Try:** `DATA_NOISE = 0.1` (little overlap: the gap closes sooner and the ceiling rises),
`DATA_NOISE = 0.5`, or a larger `SIZES` range. **Caveat:** this is one network on one 2-D problem.
It shows the mechanism — memorising is what a big model does with too little data — not a rule for
how much data any particular model needs."""
        ),
        md(
            """## Carry forward

**Settings this exercise chose, and why:**

- **One hidden layer of 12 units with a nonlinearity** — enough to separate the rings on every
  seed the tests run, small enough that its 49 parameters can be counted by hand. ReLU, tanh and
  GELU all work; exercise 09's model uses GELU in its feed-forward block.
- **A 2-D embedding** — the only width that can be drawn with no projection deciding what you see.
  Real models use hundreds of columns, and the grouping score above is how you would check them.
- **Five sizes from 20 to 2,000 points, one fixed 2·64·1 network** — wide enough to span
  memorising to generalising on one axis.
- **More than one seed before ranking two settings** — the depth comparison is the example: one
  seed could have ranked it either way.

**What the next notebook uses from this:** `notebooks/02-tokenization.ipynb` builds the vocabulary
— the list of token ids that every embedding table after it has one row for. Section 3 here took
its fifteen tokens as given; exercise 02 is where a real text's tokens come from, and why their
choice changes how many rows the table needs and how often each one is trained."""
        ),
    ]


def build() -> dict:
    """The whole notebook."""
    return {
        "cells": cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the notebook, outputs stripped."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
