"""Build `notebooks/09-loss-functions-output-heads.ipynb`.

    uv run python src/exercises/09-loss-functions-output-heads/tools/build_notebook.py

**Tracked, like the notebook it writes.** `tests/test_notebook_builders.py` checks this builder's
output is byte-identical to the tracked notebook, so edit this file and rebuild — never the notebook.

**The notebook imports the package. It never re-implements it.** A notebook that copies the masking
logic or hard-codes a loss drifts from what ships within a week and then teaches something the
exercise does not claim. Everything below reads `lossheads.*`.
"""

import os
from pathlib import Path

EXERCISE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SLUG = EXERCISE.name
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
    return f"s09-{_next_id:02d}"


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
            f"""# 09 · Loss functions and output heads

[![Open In Colab]({BADGE})]({BADGE})

Three lines of tensor code sit between a model's output and the single number it is trained
against. **Four separate things go wrong in them, none of which raises an exception, and two of
which make the loss look better.**

This notebook builds those three lines and makes each of them visible. Then it trains a model with
one of the bugs on purpose, so you can watch the loss *fall* while the model learns nothing.

It also prices the last layer — the one matrix in a transformer sized by the tokenizer rather than
by the model — and computes the same loss two ways to use a ninth of the memory.

**This is a laboratory, not a tour.** Every constant is in one cell below; change one, re-run
from there, and see what moves. Several cells end in `assert`, and those assertions are the lesson —
they state an equivalence the library must hold, and they would go red if it stopped holding.

**Runtime with `LITE = True`: about two minutes on a free Colab CPU.** Set it to `False` and the
runs become the published ones, at roughly ten.
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
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    subprocess.run(
        ["pip", "-q", "install", "-e", str(root / "src/exercises/09-loss-functions-output-heads")],
        check=True,
    )
else:
    # Found at run time, never baked in at build time: a path interpolated here would be the
    # build machine's own, and the notebook would import on that one computer only.
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit(
            "could not find the repository root by walking up from "
            + str(pathlib.Path.cwd())
            + " -- run this notebook from inside a clone of the repository, or on Colab"
        )
    sys.path.insert(0, str(root / "src/exercises/09-loss-functions-output-heads/src"))

print("repo root:", root)"""
        ),
        md(
            """## Every knob, in one cell

Nothing below is hard-coded. These are the only arbitrary choices in the notebook, and the last one
is the interesting one: **the page this exercise publishes says the step count is the only arbitrary
thing in the run, and it is not.** The seed has never been varied. Section 10 varies it.

`LITE` shortens the two training runs and nothing else. Every count, shape and equivalence below is
identical either way — which is worth checking rather than believing, so section 9 checks it."""
        ),
        code(
            """LITE      = True    # False reproduces the published runs. Costs about ten minutes.
STEPS     = 60 if LITE else 300   # optimiser steps in every training cell
SEED      = 9       # the published seed. Section 10 sweeps it
SEEDS     = (7, 8, 9, 10, 11) if not LITE else (8, 9, 10)   # for that sweep
MEM_ROWS  = 1024 if LITE else 4096   # rows the memory comparison scores
MEM_REPEATS = 3 if LITE else 5       # how many times it repeats, for the noise floor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lossheads.config import Config

CONFIG = Config()
print(f"profile   {'lite' if LITE else 'full'}   steps {STEPS}   seed {SEED}")
print(f"model     {CONFIG.n_layer} blocks x {CONFIG.d_model} wide,"
      f" vocabulary {CONFIG.vocab_size:,}")
print(f"memory    {MEM_ROWS:,} rows x {MEM_REPEATS} repeats")"""
        ),
        md(
            """### The corpus, and why every loss here is a memorisation number

The text is this repository's own conventions document, **frozen** at the revision the published run
read. It used to be read live, which made every published loss a function of a file edited on most
changes to the repository — and it had already drifted by the time anyone checked. The cell below
prints the digest that now makes that checkable, and the epoch count that decides how to read every
loss in this notebook."""
        ),
        code(
            """from lossheads.training import corpus_facts

facts = corpus_facts(CONFIG, STEPS * CONFIG.batch_size)
for key, value in facts.items():
    print(f"  {key:<15} {value:,.4f}" if isinstance(value, float) else f"  {key:<15} {value}")

print()
print(f"This run reads the corpus {facts['epochs']:.2f} times.")
print("Above ~1 epoch a loss measures memorisation; below 1.0 the text was never fully read.")"""
        ),
        md(
            """## Where the targets come from

Nobody labels this data. The right answer is simply **the next token**, so the whole training signal
is made by sliding the text against itself by one position:

```python
inputs  = tokens[:, :-1]   # drop the last: nothing follows it
targets = tokens[:,  1:]   # drop the first: nothing predicts it
```

A `T`-token sequence gives `T-1` supervised pairs for free. And the entire supervision is created by
a slice — which is why a slice in the wrong direction is so expensive."""
        ),
        md(
            """## 1 · Shapes

The logits are the interesting row. They are `[batch, position, vocabulary]` while the hidden states
that produced them are `[batch, position, width]` — so the tensor whose only purpose is to collapse
into one number is larger than its own input by the ratio of vocabulary to width. That ratio is
section 7."""
        ),
        code(
            """import torch

from lossheads.config import Config
from lossheads.model import build_trunk, count_parameters
from lossheads.heads import make_multi_token_head
from lossheads.shift import shift_for_next_token

config = Config()
trunk = build_trunk(config)
heads = make_multi_token_head(config)

tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
hidden = trunk(tokens)
logits = heads(hidden)[1]
inputs, targets = shift_for_next_token(tokens)

for name, tensor in [("tokens", tokens), ("hidden", hidden), ("logits", logits),
                     ("inputs", inputs), ("targets", targets)]:
    print(f"  {name:<9} {tuple(tensor.shape)}")

print(f"\\n  logits are {config.vocab_size / config.d_model:.1f}x the hidden states")
print(f"  trunk parameters: {count_parameters(trunk):,} — and it owns NO output head")"""
        ),
        md(
            """## 2 · The shift, read as text

**This is the check that matters, and it is the one nobody runs.** Print the inputs beside the
targets *as strings* and read them. Every row should say "this piece is followed by that piece".

Ids will not do it. An off-by-one in a wall of integers looks exactly like a correct shift."""
        ),
        code(
            """from lossheads.shift import shift_table, shift_wrong_way
from lossheads.tokenizer import load_tokenizer, pieces

tokenizer = load_tokenizer()
text = "The capital of India is New Delhi. It is the seat of the national government."
sample = torch.tensor([tokenizer.encode(text).ids])

print("CORRECT — each input is followed by its target:\\n")
print(shift_table(sample, tokenizer, rows=10, config=config))

print("\\n\\nBROKEN — each input predicts ITSELF:\\n")
bad_in, bad_out = shift_wrong_way(sample)
for i, (a, b) in enumerate(zip(pieces(bad_in[0, :6].tolist(), tokenizer, config),
                               pieces(bad_out[0, :6].tolist(), tokenizer, config))):
    print(f"    {i:>3}  {a!r} -> {b!r}")"""
        ),
        md(
            """## 3 · Padding, and the count that proves it

Batched sequences have different lengths, so short ones are padded. **A padding token is not a
prediction.** Score it and the loss *improves*, because padding is trivially predictable — the number
gets better while the model gets worse.

The mask is not the deliverable. **The count is**: a loss that moved is evidence of nothing until you
know whether the set of positions under it moved too."""
        ),
        code(
            """from lossheads.masks import pad_sequences, keep_non_padding, masked_targets
from lossheads.losses import contributing

sequences = [tokenizer.encode(text).ids, tokenizer.encode("Photosynthesis makes sugar.").ids]
padded = pad_sequences(sequences, config.seq_len, config)
pin, pout = shift_for_next_token(padded)

keep, report = keep_non_padding(pin, pout, config)
print(f"  sequence lengths: {[len(s) for s in sequences]} into {config.seq_len} slots")
print(f"  {report}")
print(f"  contributing after masking: {contributing(masked_targets(pout, keep, config), config):,}")"""
        ),
        md(
            """## 4 · A packed document boundary

Packing several documents into one sequence avoids wasting compute on padding — and creates a join.
The last token of one document sits beside the first of the next, with no relationship at all. The
gradient there asserts a continuation between two texts that have nothing to do with one another.

**One subtlety cost this exercise a real bug.** Padding carries document id `-1`, and `-1 == -1` is
`True` — so a mask written as `source == destination` keeps every pad-to-pad pair while looking
completely correct. On this example it left 68 of 125 "contributing" positions as padding predicting
padding, in the section whose whole point is that padding must never be scored."""
        ),
        code(
            """from lossheads.masks import pack_documents, keep_within_document
from lossheads.losses import cross_entropy

docs = [tokenizer.encode(text).ids, tokenizer.encode("Photosynthesis converts light energy.").ids]
packed, owners = pack_documents(docs, config.seq_len, config)

with torch.no_grad():
    flat = heads(trunk(packed))[1][:, :-1].reshape(-1, config.vocab_size)
_, ptargets = shift_for_next_token(packed)

pad_keep, pad_report = keep_non_padding(packed[:, :-1], ptargets, config)
before = masked_targets(ptargets, pad_keep, config)

bound_keep, bound_report = keep_within_document(owners, horizon=1)
after = masked_targets(ptargets, pad_keep & bound_keep, config)

print(f"  {pad_report}")
print(f"  {bound_report}\\n")
print(f"  loss WITHOUT the boundary mask: {float(cross_entropy(flat, before.reshape(-1), config)):.6f}")
print(f"  loss WITH the boundary mask   : {float(cross_entropy(flat, after.reshape(-1), config)):.6f}")"""
        ),
        md(
            """The difference is tiny, and **that is the finding rather than a disappointment**. One bad
pair barely moves an average, so nothing looks wrong from the number alone — while the gradient at
that pair is wrong on every packed sequence in a real run."""
        ),
        md(
            """## 5 · Perplexity, and where a fresh model starts

Read perplexity — `exp(loss)` — as a **count**: the size of the uniform menu the model behaves as
though it were choosing from. A model that knew nothing at all would show exactly `ln(V)`.

A freshly initialised one shows a little *more*, and that gap is real rather than an error: random
weights are not uniform output, so the model has arbitrary preferences, and being confidently
arbitrary is worse than being uniform.

**If a run starts far *below* `ln(V)`, stop.** The targets are misaligned and no amount of training
fixes that. It is the cheapest correctness check in the whole exercise."""
        ),
        code(
            """import math
from lossheads.losses import perplexity

with torch.no_grad():
    rlogits = heads(trunk(tokens))[1][:, :-1].reshape(-1, config.vocab_size)
loss = float(cross_entropy(rlogits, targets.reshape(-1), config))

print(f"  vocabulary                 : {config.vocab_size:,}")
print(f"  ln(V), a uniform model     : {math.log(config.vocab_size):.4f}")
print(f"  measured                   : {loss:.4f}")
print(f"  perplexity                 : {perplexity(loss):,.1f}")
print(f"  ratio to the vocabulary    : {perplexity(loss) / config.vocab_size:.3f}")"""
        ),
        md(
            """**Perplexity is not comparable across tokenizers.** One that splits text more finely is
asked an easier question at each step and scores better while being no better. Within one tokenizer
it is a good training signal; across them, use bits per byte."""
        ),
        md(
            """## 6 · What the output head costs

Every other matrix in a transformer is about `d_model × d_model`. The head is
`d_model × vocab_size` — the one block sized by the **tokenizer** rather than by the model. So its
share *grows* as the model shrinks.

**Tying** reuses the embedding matrix as the head and removes those parameters entirely. It is not
free: a token's input representation and the direction that scores it become the same vector. And it
is **not always available** — it needs an input table with one row per token to tie *to*, which an
architecture whose input side is a codec plus a projection does not have."""
        ),
        code(
            """from lossheads.heads import head_costs, multi_head_params

for cost in head_costs(config, embedding_has_rows=True)[:2]:
    print(f"  {cost.arrangement:<10} {cost.added_params:>12,} added parameters")
unavailable = head_costs(config, embedding_has_rows=False)[1]
print(f"  {'tied':<10} {'UNAVAILABLE':>12}   {unavailable.note[:60]}...")

print(f"\\n  body (estimated) : {config.body_params:>12,}")
print(f"  untied head      : {config.head_params:>12,}  ({config.head_share:.1%} of the total)")
print(f"  two dense heads  : {multi_head_params(config):>12,}  — the price of section 8")"""
        ),
        md(
            """## 7 · The same loss, a ninth of the memory

The logits tensor exists only to become one scalar, and it is the largest thing in the step. Chunking
computes the loss in row blocks so the full `[rows, vocab]` tensor never exists at once.

**Two kinds of chunking, and only one is what the technique is named for.** Chunking a softmax over
logits that already exist saves the intermediates: about 1.9×. Doing the *projection* inside the
loop, so the tensor is never built: about 9×.

**The loss must be identical.** Chunking is not an approximation — if the numbers differ, something
else broke, and a memory ratio quoted without that check is a comparison of two different things."""
        ),
        code(
            """from lossheads.losses import chunked_projection_cross_entropy

g = torch.Generator().manual_seed(9)
h = torch.randn(2048, config.d_model, generator=g)
w = torch.randn(config.vocab_size, config.d_model, generator=g) * 0.02
t = torch.randint(0, config.vocab_size, (2048,), generator=g)

full = float(cross_entropy(h @ w.T, t, config))
for chunk in (64, 128, 512, 2048):
    got = float(chunked_projection_cross_entropy(h, w, t, chunk, config))
    print(f"  chunk {chunk:>5}: {got:.9f}   {'identical' if abs(got - full) < 1e-5 else 'DIFFERS'}")
print(f"  unchunked   : {full:.9f}")"""
        ),
        code(
            """# The actual memory measurement runs each path in its own process, because tracemalloc is
# blind to torch — it reported 429 bytes for an 81,928,192-byte logits tensor — and because a
# second path in the same process is handed the first one's freed blocks by torch's allocator.
from lossheads.memory import compare_paths

print(compare_paths(rows=4096, config=config))"""
        ),
        md(
            """## 8 · Five equivalences, asserted rather than described

Four knobs in `losses.py` and one in `heads.py` claim to reduce to plain cross-entropy at their
no-op setting. **A claim like that is worth an assertion, not a sentence**, because the failure mode
is silent: a smoothed loss that is *almost* the plain one at epsilon zero is a bug you will never
notice, and a chunked loss that disagrees only on masked input is one this exercise actually
shipped — it divided by the row count instead of the contributing count, and every test written on
unmasked input passed either way.

Each cell below is written **twice**: once at the setting where the two must agree, and once away
from it. A function that ignored its argument entirely would pass the first half and fail the
second, which is the whole reason the second half is there."""
        ),
        code(
            """import torch

from lossheads.losses import (
    chunked_cross_entropy,
    chunked_projection_cross_entropy,
    cross_entropy,
    cross_entropy_with_z_loss,
    label_smoothed_cross_entropy,
)

torch.manual_seed(SEED)
rows, vocab = 256, CONFIG.vocab_size
logits = torch.randn(rows, vocab)
targets = torch.randint(0, vocab, (rows,))
plain = cross_entropy(logits, targets, CONFIG)
print(f"plain cross-entropy: {float(plain):.6f}")
print()

checks = []

# 1 -- label smoothing at epsilon = 0
at_zero = float(label_smoothed_cross_entropy(logits, targets, 0.0, CONFIG))
away = float(label_smoothed_cross_entropy(logits, targets, 0.1, CONFIG))
checks.append(("label smoothing", at_zero, away))
assert abs(at_zero - float(plain)) < 1e-6, "smoothing at 0 must BE plain cross-entropy"
assert abs(away - float(plain)) > 1e-4, "smoothing at 0.1 must move it, or the knob does nothing"

# 2 -- a z-loss weight of 0
at_zero = float(cross_entropy_with_z_loss(logits, targets, 0.0, CONFIG))
away = float(cross_entropy_with_z_loss(logits, targets, 1e-3, CONFIG))
checks.append(("z-loss", at_zero, away))
assert abs(at_zero - float(plain)) < 1e-6
assert abs(away - float(plain)) > 1e-6

# 3 -- chunking the softmax over existing logits
chunked = float(chunked_cross_entropy(logits, targets, CONFIG.chunk_size, CONFIG))
checks.append(("chunked softmax", chunked, float("nan")))
assert abs(chunked - float(plain)) < 1e-5, "chunking is not an approximation"

for name, same, moved in checks:
    tail = "" if moved != moved else f"   moved to {moved:.6f} away from the no-op"
    print(f"  {name:<18} {same:.6f}   equals plain: {abs(same - float(plain)) < 1e-5}{tail}")"""
        ),
        md(
            """**Now the one the equivalences above cannot see.** Every check so far used unmasked
targets, and the bug this exercise shipped only appears once positions are masked out — because the
two versions differ in what they divide by, and with nothing masked the two denominators are equal.

Run it with a mask."""
        ),
        code(
            """masked = targets.clone()
masked[: rows // 3] = CONFIG.ignore_index   # a third of the batch does not count

plain_masked = float(cross_entropy(logits, masked, CONFIG))
chunk_masked = float(chunked_cross_entropy(logits, masked, CONFIG.chunk_size, CONFIG))

print(f"unmasked:  plain {float(plain):.6f}   chunked {chunked:.6f}")
print(f"masked:    plain {plain_masked:.6f}   chunked {chunk_masked:.6f}")
print()
print(f"difference under masking: {abs(plain_masked - chunk_masked):.2e}")
assert abs(plain_masked - chunk_masked) < 1e-5, (
    "the chunked loss disagrees with the plain one on masked input -- which is exactly the "
    "shipped defect: it divided by the row count rather than the contributing count."
)
print()
print("Both denominators are the CONTRIBUTING count, not the row count. Try changing the")
print("fraction masked above: the gap stays at machine precision. If it ever does not, the")
print("bug is back, and no test written on unmasked input would have told you.")"""
        ),
        md(
            """### And the fifth: a tied head is a linear map with the embedding's weights

Tying is not an approximation to a separate head either. It is the *same arithmetic* with the same
numbers — which is why it costs exactly zero extra parameters and why nothing about the loss
changes."""
        ),
        code(
            """from lossheads.heads import make_tied_head, tied_head_params, untied_head_params

torch.manual_seed(SEED)
embedding = torch.nn.Embedding(CONFIG.vocab_size, CONFIG.d_model)
hidden = torch.randn(4, CONFIG.d_model)

tied = make_tied_head(embedding)
by_hand = hidden @ embedding.weight.T

agrees = torch.allclose(tied(hidden), by_hand, atol=1e-6)
assert agrees, "a tied head IS that matmul"
# Equality is not enough: a COPY of the embedding would also be equal here, and would then drift
# apart on the first gradient step. Tying means one tensor, so check the storage.
assert tied.weight.data_ptr() == embedding.weight.data_ptr(), "tied must share storage, not values"
print(f"tied head == hidden @ embedding.weight.T : {agrees}")
print(f"and the same tensor, not a copy          : "
      f"{tied.weight.data_ptr() == embedding.weight.data_ptr()}")
print(f"parameters, tied   : {tied_head_params(CONFIG.d_model, CONFIG.vocab_size):>10,}")
print(f"parameters, untied : {untied_head_params(CONFIG.d_model, CONFIG.vocab_size):>10,}")
print()
print("Zero against a matrix the size of the vocabulary. That is the entire argument for tying,")
print("and the cost is that a token's input vector and its scoring direction become one thing.")"""
        ),
        md(
            """## 9 · A second head, predicting two tokens ahead

The trunk is shared; only the slice changes. `shift_for_horizon(tokens, 2)` is the entire difference
on the data side, and **the losses simply add**.

**Stated before running: the further head should sit above the nearer one**, because predicting two
positions ahead is genuinely harder. Watch whether it does.

The cost is honest: `k` heads is `k ×` the head parameters, and the head is already the expensive
part of a small model."""
        ),
        code(
            """# Trains a two-head model, then one correct model and one with the off-by-one.
# `train`, not `run`: run() prints a report and writes a file, and this notebook must never
# write into the tracked results/ that the exercise's documents render.
from lossheads.training import train

log = train(steps=STEPS, seed=SEED)
summary = log.summary()
print(f"t+1 final {summary['final_by_horizon']['1']:.4f}"
      f"   t+2 final {summary['final_by_horizon']['2']:.4f}"
      f"   gap {summary['gap']:+.4f}")
print(f"the further head was higher on "
      f"{summary['steps_where_further_head_was_higher']} of {summary['total_steps']} steps")"""
        ),
        md(
            """### The two horizons, drawn

Two curves, `STEPS` points each. The gap is the finding, not the height: both curves fall, and a
reader watching either one alone learns nothing about the other."""
        ),
        code(
            """near, far = str(min(CONFIG.horizons)), str(max(CONFIG.horizons))
fig, (ax, gap_ax) = plt.subplots(1, 2, figsize=(11, 3.4))

ax.plot(log.by_horizon[near], lw=1, label=f"head t+{near}")
ax.plot(log.by_horizon[far], lw=1, label=f"head t+{far}")
ax.set_xlabel("step"); ax.set_ylabel("loss (nats)"); ax.legend(frameon=False)
ax.set_title("both heads fall")

gap = np.array(log.by_horizon[far]) - np.array(log.by_horizon[near])
gap_ax.plot(gap, lw=1, color="crimson")
gap_ax.axhline(0, color="0.6", lw=0.8)
gap_ax.set_xlabel("step"); gap_ax.set_ylabel("t+2 minus t+1")
gap_ax.set_title(f"the gap: positive on {int((gap > 0).sum())} of {len(gap)} steps")
fig.tight_layout(); plt.show()

print("Predicted BEFORE running: the further head sits above the nearer one, because predicting")
print("two positions ahead is genuinely harder.")
verdict = "it does" if gap[-1] > 0 else "IT DOES NOT -- that is a finding, write it down"
print(f"Observed: {verdict}.")"""
        ),
        md(
            """## 10 · Find the bug yourself

Now the part this notebook exists for. Two models, identical in every way except the direction of
one slice. **Look at the losses first**, decide which model you would keep, and only then run the
cell after it."""
        ),
        code(
            """print(f"correct shift : final loss {log.correct_shift[-1]:.4f}")
print(f"off-by-one    : final loss {log.broken_shift[-1]:.4f}")
print()
print("Which would you ship?")

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(log.correct_shift, lw=1, label="A")
ax.plot(log.broken_shift, lw=1, ls="--", label="B")
ax.set_xlabel("step"); ax.set_ylabel("loss (nats)"); ax.legend(frameon=False)
ax.set_title("two runs, same everything but one slice")
fig.tight_layout(); plt.show()"""
        ),
        md(
            """**Now read what each model was actually asked.** Nothing above distinguishes them.
This does."""
        ),
        code(
            """from lossheads.shift import shift_for_next_token, shift_wrong_way
from lossheads.tokenizer import load_tokenizer, pieces

tokenizer = load_tokenizer()
tokens = torch.tensor([tokenizer.encode("The capital of France is Paris.").ids])

a_in, a_out = shift_for_next_token(tokens)
b_in, b_out = shift_wrong_way(tokens)
show = 8
cols = [
    pieces(a_in[0, :show].tolist(), tokenizer, CONFIG),
    pieces(a_out[0, :show].tolist(), tokenizer, CONFIG),
    pieces(b_out[0, :show].tolist(), tokenizer, CONFIG),
]

print(f"{'given':>14}  {'A predicts':>14}  {'B predicts':>14}")
print("-" * 50)
for given, a, b in zip(*cols, strict=True):
    flag = "  <- its own input" if b == given else ""
    print(f"{given!r:>14}  {a!r:>14}  {b!r:>14}{flag}")

# Over the WHOLE sequence, not the eight rows shown.
all_in = pieces(a_in[0].tolist(), tokenizer, CONFIG)
all_b = pieces(b_out[0].tolist(), tokenizer, CONFIG)
copies = sum(1 for given, b in zip(all_in, all_b, strict=True) if given == b)
print()
print(f"B is asked to reproduce its own input on {copies} of {len(all_in)} positions.")
assert copies == len(all_in), "the off-by-one must be a pure copy, or this demonstration lies"
print("Copying is easy. That is the whole of the lower curve.")
print()
print("The check works at step ZERO, before any training, which is the point: no loss curve can")
print("tell you this, because at step zero the two models score the same.")"""
        ),
        md(
            """## 11 · The claim the published page could not make

The page says the step count is the only arbitrary choice in the run. **It is not — the seed has
never been varied.** Vary it here and see whether either finding depends on it.

If a verdict flips, that is a finding, and the claim was resting on a seed."""
        ),
        code(
            """rows = []
for seed in SEEDS:
    run_log = train(steps=STEPS, seed=seed)
    s = run_log.summary()
    rows.append({
        "seed": seed,
        "gap": s["gap"],
        "further_harder": s["further_head_is_harder"],
        "broken_lower": s["broken_shift_is_lower"],
        "broken": s["final_broken_shift"],
        "correct": s["final_correct_shift"],
    })
    print(f"  seed {seed:>3}  gap {s['gap']:+.4f}  "
          f"further harder {str(s['further_head_is_harder']):>5}  "
          f"broken lower {str(s['broken_shift_is_lower']):>5}")

gaps = [r["gap"] for r in rows]
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar([str(r["seed"]) for r in rows], gaps, color="steelblue")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlabel("seed"); ax.set_ylabel("t+2 minus t+1 at the end")
ax.set_title(f"the gap across {len(rows)} seeds: {min(gaps):+.3f} to {max(gaps):+.3f}")
fig.tight_layout(); plt.show()

assert all(r["further_harder"] for r in rows), "a seed flipped the two-head verdict"
assert all(r["broken_lower"] for r in rows), "a seed flipped the off-by-one verdict"
print()
print(f"Both verdicts held at every seed tried. The gap itself moved by "
      f"{max(gaps) - min(gaps):.3f} nats, which is the noise floor on the EFFECT SIZE --")
print("so quote the direction confidently and the magnitude with that spread beside it.")"""
        ),
        md(
            """## 12 · The memory ratio, and its noise floor

One measurement is not a measurement. The published figure is quoted as "about 9x and no finer",
and this is why: repeat it and it moves."""
        ),
        code(
            """from lossheads.memory import compare_paths

ratios, softmax_ratios = [], []
for i in range(MEM_REPEATS):
    report = compare_paths(rows=MEM_ROWS, config=CONFIG, seed=SEED + i)
    ratios.append(report.ratio)
    softmax_ratios.append(report.softmax_only_ratio)
    print(f"  repeat {i + 1}: projection inside the loop {report.ratio:.2f}x   "
          f"softmax only {report.softmax_only_ratio:.2f}x   "
          f"losses agree {report.losses_agree}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(1, len(ratios) + 1), ratios, "o-", label="projection inside the loop")
ax.plot(range(1, len(ratios) + 1), softmax_ratios, "s--", label="softmax chunked only")
ax.set_xlabel("repeat"); ax.set_ylabel("times less peak memory")
ax.set_ylim(0, max(ratios) * 1.15); ax.legend(frameon=False)
ax.set_title(f"spread {max(ratios) - min(ratios):.2f} over {len(ratios)} repeats")
fig.tight_layout(); plt.show()

print()
print(f"Chunking the SOFTMAX saves {np.mean(softmax_ratios):.2f}x. Moving the PROJECTION inside")
print(f"the loop saves {np.mean(ratios):.2f}x --"
      f" {np.mean(ratios) / np.mean(softmax_ratios):.1f} times more.")
print("Both are called 'chunked cross-entropy'. Only one of them avoids building the tensor.")
print()
print("Peak RSS of a whole process, so it moves with whatever else the machine is doing. That")
print("is why the figure is quoted to one significant digit, and why repeating it is the point.")"""
        ),
        md("""## What this notebook cannot tell you"""),
        code(
            """print("- Which loss trains a BETTER model. These runs exist to show two specific effects,")
print("  and neither is a claim about which objective to pick.")
print()
print(f"- Anything that GENERALISES. The corpus is {facts['corpus_tokens']:,} tokens and this run")
print(f"  read it {facts['epochs']:.2f} times, so every loss here is a memorisation number. Both")
print("  findings survive that -- each compares two models trained identically on the same")
print("  repeated text, so the repetition cancels -- but the absolute values do not transfer.")
print()
print("- Anything about a GPU. Every figure is CPU, fp32, at laptop shapes.")
print()
print(f"- Anything at {'the published' if not LITE else 'full'} scale.", end=" ")
print(f"You ran {STEPS} steps against the published 300."
      if LITE else "You ran the published configuration.")"""
        ),
        md(
            """The full write-up, with every figure generated rather than typed, is in the
exercise's [`RESULTS.md`](../src/exercises/09-loss-functions-output-heads/RESULTS.md).

**What to change, and what to watch:**

| change | what should move | what must NOT |
| --- | --- | --- |
| `LITE` → `False` | every loss, and the epoch count with it | no verdict in sections 8–11 |
| `SEED` | the gap's magnitude, a little | the direction of either finding |
| `SEEDS` → more of them | the spread section 11 reports | the two assertions in that cell |
| `MEM_ROWS` → smaller | the memory ratio falls toward 1 | that the two losses still agree |
| a third of the batch masked → half, in section 8 | nothing | the plain/chunked agreement |

**The last column is the important one.** A verdict that depends on a knob is not a verdict, and
this notebook is arranged so you can find that out in one edit.
"""
        ),
    ]


WHERE = """## Where this sits

Notebook 9 of 14 — [the map of all fourteen](README.md). Previous: [`08-modern-attention-variants.ipynb`](08-modern-attention-variants.ipynb).
Next: [`10-training-loop.ipynb`](10-training-loop.ipynb).

**Takes in:** exercise 02's frozen tokenizer (10,000 entries; this exercise adds `[PAD]` at id
10,000) and this exercise's own tracked corpus. **This is where the shared model is born:**
`lossheads.model.build_trunk` is the model exercises 10 and 12 train, and the one exercise 07
plugs its embedding into.
"""

CARRY = """## Carry forward

**What the next notebooks take from here:** the target shift by one, the padding and
document-boundary masks, plain masked cross-entropy, and the trunk at its default `Config` —
`d_model` 256, 4 layers, 4 heads, 128-token windows. Exercise 10 drives exactly these; it imports
them rather than restating them, so the two cannot disagree about what a loss is.

**Worth trying before you move on:** label smoothing and z-loss are implemented and tested here
but unused by any run — each reduces to plain cross-entropy at one setting of its own knob.
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
