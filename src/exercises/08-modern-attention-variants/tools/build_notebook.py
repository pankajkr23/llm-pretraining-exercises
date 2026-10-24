"""Build `notebooks/08-modern-attention-variants.ipynb`: an attention laboratory.

Tracked, like the notebook it writes; edit this file and rebuild, never the notebook (a test checks
the two are byte-identical). The notebook imports `attention.lab`
and never re-implements it — every mechanism it runs is the tested code in the package, and every
number it prints with a source comes from the package's verified ledger.

This notebook is for the person learning attention, not for the page's readers: it has no copy of
the page. It runs each variant, draws what the code computes, asserts the identities that define
it, and trains small models to compare them.

    uv run python src/exercises/08-modern-attention-variants/tools/build_notebook.py
    NOTEBOOK_OUT=/tmp/x.ipynb uv run python .../build_notebook.py     # anywhere else
"""

import json
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

_ids = iter(range(1, 10_000))


def _cell(kind: str, text: str) -> dict:
    cell = {
        "cell_type": kind,
        "id": f"s08-{next(_ids):03d}",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def md(text: str) -> dict:
    """A markdown cell."""
    return _cell("markdown", text)


def code(text: str) -> dict:
    """A code cell, outputs stripped."""
    return _cell("code", text)


def cells() -> list[dict]:  # noqa: PLR0915 - one notebook, read top to bottom
    """Every cell, in order."""
    return [
        md(f"""
# Attention, run by hand — from Bahdanau to Kimi K3

[![Open In Colab]({BADGE})]({BADGE})

**This is a laboratory, not a tour.** Every attention variant in the exercise's catalogue is
implemented in `attention.lab`, tested against the equations that define it, and run here. You
change a knob, re-run a cell, and watch what moves.

How the notebook is laid out:

| § | what you do |
| --- | --- |
| 0 | set up, and set every knob in one cell |
| 1 | build attention from Q, K and V, starting from Bahdanau's additive score |
| 2 | give tokens positions: sinusoids, RoPE, ALiBi, NTK, YaRN, DroPE, HD-RoPE |
| 3 | shrink the cache: MHA → MQA → GQA → MLA, measured in bytes |
| 4 | read only some tokens: windows, sinks, strided, top-k, LSH, NSA, CSA, MSA |
| 5 | replace the cache with a fixed-size memory: linear, lightning, delta rules, KDA, Mamba |
| 6 | mix the two: KDA 3:1, lightning 7:1, and Kimi K3's stack |
| 7 | the bake-off: train a small model with each variant and compare |
| 8 | find the bug |
| 9 | what Kimi K3 and MiniMax actually did, from their own reports |
| 10 | what to change, and what should move |

**Every sourced number here is re-found in its paper.** The lab's ledger records, for each
number, the sentence it was read from and the downloaded file it was found in. A number marked
*our choice* is ours, and the documentation says why. See
`src/exercises/08-modern-attention-variants/docs/ATTENTION_LAB.md` for the code-level diff and
configuration table of every variant; `describe(name)` below prints the same section.
"""),
        md("## 0 · Setup"),
        code(f"""
import os, sys, subprocess, pathlib

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    ROOT = pathlib.Path("llm-pretraining-exercises").resolve()
    # Exercise 09 first: exercise 08 depends on it by name, which pip cannot resolve from a path.
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-e",
                    str(ROOT / "src/exercises/09-loss-functions-output-heads")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-e",
                    str(ROOT / "src/exercises/08-modern-attention-variants") + "[train]"], check=True)
else:
    # Find the repository from wherever the notebook was opened, never from a path baked in.
    here = pathlib.Path.cwd().resolve()
    ROOT = next(p for p in (here, *here.parents) if (p / "src/exercises").is_dir())
    for rel in ("src/exercises/08-modern-attention-variants/src",
                "src/exercises/09-loss-functions-output-heads/src"):
        if str(ROOT / rel) not in sys.path:
            sys.path.insert(0, str(ROOT / rel))
print("repository:", ROOT)
"""),
        md("""
**Every knob is in the next cell, and every knob is read below.** `LITE = True` finishes on a
laptop in a few minutes; `LITE = False` is the fuller run for a Colab GPU.
"""),
        code("""
LITE = True
SEED = 0
DEVICE = None          # None picks cuda, then mps, then cpu
D_MODEL = 32           # width of the small models in §7
N_LAYERS = 2 if LITE else 4
SEQ = 32 if LITE else 64
STEPS = 60 if LITE else 300
SEEDS = (0, 1) if LITE else (0, 1, 2)   # two seeds at least: a ranking needs a spread
VARIANTS = None        # None = every registered variant; or a list of names
"""),
        code("""
import math, time
import torch
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from attention.lab import registry, ops, describe, experiments, data, runs, hparams
from attention.lab import core, positions, sparse, linear, delta, ssm

torch.manual_seed(SEED)
device = runs.select_device(DEVICE)
ALL = registry.names()
CHOSEN = ALL if VARIANTS is None else list(VARIANTS)
print(f"{len(ALL)} variants registered; running on {device}")
for family in ("origin", "full", "position", "cache", "sparse", "linear", "recurrent", "hybrid"):
    print(f"  {family:<10}", ", ".join(registry.names(family)))

def show(name):
    \"\"\"Print a variant's documentation: what changed, its numbers and their sources.\"\"\"
    display(Markdown(describe.describe(name)))
"""),
        md("""
## 1 · Attention from scratch

### 1a · Where it started: Bahdanau's additive score

Before the Transformer, a translation decoder asked, for each source word, *how well does this
fit what I am about to write?* — with a small MLP, `v·tanh(W s + U h)`. The weights over the
source sum to one, and the context is their weighted average.
"""),
        code("""
show("bahdanau_attention")
spec = registry.get("bahdanau_attention")
layer = spec.build("lab").eval()
x = torch.randn(1, 6, 32)            # six decoder states
memory = torch.randn(1, 9, 24)       # nine encoder states
with torch.no_grad():
    y, _ = layer(x, memory=memory)
print("output", tuple(y.shape), "— one context vector per decoder step")
"""),
        md("""
### 1b · Scaled dot-product attention, one stage at a time

`ops.attend` writes the Transformer's attention out step by step. Below it runs on six tokens,
and each stage is printed so you can follow the numbers.
"""),
        code("""
torch.manual_seed(SEED)
T, d = 6, 4
q = torch.randn(1, 1, T, d, dtype=torch.float64)
k = torch.randn(1, 1, T, d, dtype=torch.float64)
v = torch.randn(1, 1, T, d, dtype=torch.float64)

scores = q @ k.transpose(-2, -1)
scaled = scores / math.sqrt(d)
mask = ops.causal_mask(T, T)
weights = torch.softmax(scaled.masked_fill(~mask, float("-inf")), dim=-1)
out = weights @ v

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, (title, m) in zip(axes, [("Q·K / √d", scaled[0, 0]), ("causal mask", mask.float()),
                                 ("softmax weights", weights[0, 0])]):
    ax.imshow(m, cmap="viridis"); ax.set_title(title); ax.set_xlabel("key"); ax.set_ylabel("query")
plt.tight_layout(); plt.show()

ours, _ = ops.attend(q, k, v, allowed=mask)
library = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(ours, out) and torch.allclose(ours, library), "the hand-written path must match"
print("rows sum to one:", weights.sum(-1)[0, 0].tolist())
print("attention returns vectors, not weights — output shape", tuple(out.shape))
"""),
        md("""
**Try this:** set `d = 64` and look at the unscaled scores: without `/√d` the softmax saturates to
one-hot rows. That is the whole reason for the scale.
"""),
        md("## 2 · Positions"),
        code("""
table = positions.sinusoidal_table(torch.arange(64), 32, 10000.0)
plt.figure(figsize=(9, 3)); plt.imshow(table.T, aspect="auto", cmap="RdBu")
plt.xlabel("position"); plt.ylabel("dimension"); plt.title("sinusoidal encoding"); plt.show()

# The Transformer paper says PE(p+k) is a linear function of PE(p). The matrix is not fitted:
# each (sin, cos) pair at frequency w is rotated by the angle w*k, whatever p is.
d_model, base, k_offset = 32, 10000.0, 5
w = 1.0 / base ** (torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
M = torch.zeros(d_model, d_model, dtype=torch.float64)
c, s = torch.cos(w * k_offset), torch.sin(w * k_offset)
for i in range(d_model // 2):
    M[2 * i, 2 * i], M[2 * i + 1, 2 * i] = c[i], s[i]          # new sin = sin*c + cos*s
    M[2 * i, 2 * i + 1], M[2 * i + 1, 2 * i + 1] = -s[i], c[i]  # new cos = cos*c - sin*s
p = torch.arange(0, 500)
here = positions.sinusoidal_table(p, d_model, base)
there = positions.sinusoidal_table(p + k_offset, d_model, base)
assert torch.allclose(here @ M, there, atol=1e-9), "offset is not a fixed linear map"
print("PE(p+5) = PE(p)·M for all 500 positions, with one M that never looked at p")
"""),
        code("""
# RoPE: the score between q at m and k at n depends only on n - m.
torch.manual_seed(SEED)
qv, kv = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
def rope_score(m, n):
    qm = ops.apply_rope(qv, ops.rope_angles(torch.tensor([m]), 8, 10000.0))
    kn = ops.apply_rope(kv, ops.rope_angles(torch.tensor([n]), 8, 10000.0))
    return float((qm * kn).sum())
print("score at (3, 7):", rope_score(3, 7), " at (103, 107):", rope_score(103, 107))
assert abs(rope_score(3, 7) - rope_score(103, 107)) < 1e-9
offsets = range(0, 200)
plt.figure(figsize=(8, 2.5)); plt.plot([rope_score(0, o) for o in offsets])
plt.xlabel("distance n - m"); plt.ylabel("q·k"); plt.title("RoPE score against distance"); plt.show()
show("rope")
"""),
        code("""
slopes = positions.alibi_slopes(8)
print("ALiBi slopes for 8 heads:", slopes)
dist = torch.arange(12)[:, None] - torch.arange(12)[None, :]
bias = -torch.tensor(slopes)[:, None, None] * dist.clamp(min=0)
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(bias[0]); axes[0].set_title("head 0 (steepest)")
axes[1].imshow(bias[-1]); axes[1].set_title("head 7 (flattest)")
plt.show()
"""),
        md("""
### Train short, test long

A position scheme is judged by what happens *past* the length it was trained at. The
`extrapolate` task trains each scheme at `SEQ` tokens and evaluates at 2× and 4×.
"""),
        code("""
pos_names = [n for n in registry.names("position") if n in CHOSEN]
spec = experiments.ExperimentSpec.make(
    pos_names, task="extrapolate", lite=LITE, d_model=D_MODEL, n_layers=N_LAYERS,
    seq_len=SEQ, steps=STEPS, seeds=SEEDS, device=DEVICE)
ext = experiments.run(spec)
rows = {name: r["summary"] for name, r in ext["results"].items()}
for name, s in rows.items():
    print(f"{name:<18}", s)
print("corpus report:", ext["corpus"])
"""),
        md("## 3 · The cache: what full attention keeps, and how to keep less"),
        code("""
from attention.config import Yardstick
from attention.cache import kv_cache_bytes

names = ["standard_attention", "gqa", "mqa", "mla"]
lengths = [16, 64, 256]
plt.figure(figsize=(7, 3.5))
for name in names:
    curve = describe.state_curve(registry.get(name), tuple(lengths))
    plt.plot([n for n, _ in curve], [b for _, b in curve], marker="o", label=name)
plt.xlabel("tokens"); plt.ylabel("bytes kept (lab scale)"); plt.legend(); plt.title("the cache"); plt.show()

# The measured cache is exactly the formula the page's invoice uses.
std = registry.get("standard_attention")
kw = {p.name: p.value for p in std.lab}
layer = std.build("lab")
with torch.no_grad():
    _, state = layer(torch.randn(1, 64, kw["d_model"]))
formula = kv_cache_bytes(Yardstick(layers=1, kv_heads=kw["n_heads"], query_heads=kw["n_heads"],
                                   head_dim=kw["head_dim"], dtype="fp32"), context=64, batch=1)
assert layer.state_bytes(state) == formula, (layer.state_bytes(state), formula)
print("measured bytes == kv_cache_bytes formula:", formula)
for name in names:
    show(name)
"""),
        md("## 4 · Sparse attention: which tokens each query reads"),
        code("""
T = 24
qp = torch.arange(T); kp = torch.arange(T)
strided = sparse.factorized_masks("strided", qp, kp, stride=4, c=1)
fixed = sparse.factorized_masks("fixed", qp, kp, stride=4, c=1)
masks = {
    "full causal": ops.causal_mask(T, T),
    "window w=4": sparse.window_mask(qp, kp, 4),
    "strided (both heads)": strided[0] | strided[1],
    "fixed (both heads)": fixed[0] | fixed[1],
}
fig, axes = plt.subplots(1, len(masks), figsize=(14, 3.2))
for ax, (title, m) in zip(axes, masks.items()):
    ax.imshow(m.float(), cmap="Greys"); ax.set_title(title)
plt.show()
for name, m in masks.items():
    print(f"{name:<22} reads {m.float().mean():.0%} of the grid")
"""),
        code("""
# Top-k and LSH depend on the content, so their patterns are drawn from real weights.
for name in ["topk_attention", "reformer", "attention_sinks", "nsa", "deepseek_csa", "msa"]:
    if name in CHOSEN:
        show(name)
"""),
        md("""
### Cost against length

`cost` measures forward time, memory and kept state as the sequence grows. These are reference
implementations: a fast kernel changes the time axis, not the state axis.
"""),
        code("""
cost_names = [n for n in ("standard_attention", "sliding_window", "attention_sinks", "nsa", "msa",
                          "linear_attention", "kda", "mamba") if n in CHOSEN]
spec = experiments.ExperimentSpec.make(cost_names, task="cost", lite=LITE, d_model=D_MODEL,
                                       n_layers=N_LAYERS, seq_len=SEQ, seeds=(SEED,), device=DEVICE)
cost = experiments.run(spec)
for name, r in cost["results"].items():
    print(f"{name:<20}", r["summary"])
"""),
        md("## 5 · A fixed-size memory instead of a cache"),
        code("""
def two_modes(name, a, b, tokens=17):
    \"\"\"Build a variant in two modes with identical weights and compare their outputs.\"\"\"
    spec = registry.get(name)
    torch.manual_seed(SEED)
    first = spec.build("lab", mode=a).double()
    second = spec.build("lab", mode=b).double()
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, tokens, {p.name: p.value for p in spec.lab}["d_model"], dtype=torch.float64)
    with torch.no_grad():
        ya, _ = first(x); yb, _ = second(x)
    gap = float((ya - yb).abs().max())
    assert gap < 1e-8, f"{name}: {a} and {b} disagree by {gap}"
    print(f"{name:<20} {a:>10} == {b:<10} (max gap {gap:.1e})")

two_modes("linear_attention", "parallel", "recurrent")
two_modes("lightning_attention", "tiled", "recurrent")
two_modes("delta_rule", "parallel", "recurrent")
for name in ("deltanet_parallel", "gated_deltanet", "kda", "gated_deltanet2"):
    two_modes(name, "chunkwise", "recurrent")
two_modes("mamba", "scan", "parallel")
two_modes("mamba3", "scan", "parallel")
"""),
        md("""
### Why the delta rule exists: associative recall

Linear attention only *adds* to its memory, so old associations blur into new ones. The delta
rule first *removes* what the memory currently returns for a key, then writes the new value. The
recall task shows the difference: key–value pairs, then queries.
"""),
        code("""
recall_names = [n for n in ("linear_attention", "delta_rule", "deltanet_parallel", "gated_deltanet",
                            "kda", "gated_deltanet2", "mamba", "standard_attention") if n in CHOSEN]
spec = experiments.ExperimentSpec.make(recall_names, task="recall", lite=LITE, d_model=D_MODEL,
                                       n_layers=N_LAYERS, seq_len=SEQ, steps=STEPS, seeds=SEEDS,
                                       device=DEVICE)
recall = experiments.run(spec)
names = list(recall["results"])
means = [recall["results"][n]["summary"]["mean"] for n in names]
plt.figure(figsize=(8, 3)); plt.bar(names, means); plt.xticks(rotation=30, ha="right")
plt.ylabel("recall accuracy"); plt.title("associative recall (mean over seeds)"); plt.show()
for n in names:
    print(f"{n:<20}", recall["results"][n]["summary"])
"""),
        code("""
for name in ("delta_rule", "gated_deltanet", "kda", "mamba"):
    show(name)
"""),
        md("## 6 · Hybrid stacks"),
        code("""
for name in registry.names("hybrid"):
    show(name)
hyb = [n for n in registry.names("hybrid") if n in CHOSEN]
if hyb:
    spec = experiments.ExperimentSpec.make(hyb + ["kda", "standard_attention"], task="cost",
                                           lite=LITE, d_model=D_MODEL, n_layers=N_LAYERS,
                                           seq_len=SEQ, seeds=(SEED,), device=DEVICE)
    for name, r in experiments.run(spec)["results"].items():
        print(f"{name:<20}", r["summary"])
"""),
        md("""
## 7 · The bake-off

The same small model, the same data, one mixer swapped. Read the spread across seeds before
ranking anything: a gap smaller than the spread is not a result.
"""),
        code("""
trainable = [n for n in CHOSEN if not registry.get(n).cross]
spec = experiments.ExperimentSpec.make(trainable, task="lm", lite=LITE, d_model=D_MODEL,
                                       n_layers=N_LAYERS, seq_len=SEQ, steps=STEPS, seeds=SEEDS,
                                       device=DEVICE)
started = time.time()
lm = experiments.run(spec)
print(f"{len(trainable)} variants x {len(SEEDS)} seeds in {time.time() - started:.0f}s")
print("corpus:", lm["corpus"])

fig, ax = plt.subplots(figsize=(10, 4))
for name, r in lm["results"].items():
    curve = r["seeds"][str(SEEDS[0])]["curve"]
    ax.plot(curve, label=name, lw=1)
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("training loss, first seed")
ax.legend(fontsize=6, ncol=3); plt.show()

ranked = sorted(lm["results"].items(), key=lambda kv: kv[1]["summary"]["mean"])
for name, r in ranked:
    s = r["summary"]
    spread = "n/a" if s["spread"] is None else f"{s['spread']:.3f}"
    print(f"{name:<20} mean {s['mean']:.3f}  spread {spread}")
path = runs.save(lm)
print("saved with its provenance to", path)
"""),
        md("""
**Above one epoch the loss measures memorisation too.** The corpus report above prints tokens
consumed ÷ corpus tokens; with `LITE = False` it goes past one, and the report says so.
"""),
        md("""
## 8 · Find the bug

Here is a decoder whose mask is off by one: each query may also read the *next* token. Train it
next to the correct one and look at the losses before reading on.
"""),
        code("""
# Build the leak from the correct layer by widening its mask for one training run.
real_mask = ops.causal_mask
def leaky_mask(queries, keys, offset=0, device=None):
    return real_mask(queries, keys, offset + 1, device)

losses = {}
for label, mask_fn in (("correct", real_mask), ("leaky", leaky_mask)):
    ops.causal_mask = mask_fn
    try:
        spec = experiments.ExperimentSpec.make(["standard_attention"], task="lm", lite=True,
                                               d_model=D_MODEL, n_layers=N_LAYERS, seq_len=SEQ,
                                               steps=STEPS, seeds=(SEED,), device=DEVICE)
        losses[label] = experiments.run(spec)["results"]
    finally:
        ops.causal_mask = real_mask
for label, res in losses.items():
    print(label, {k: round(v["summary"]["mean"], 3) for k, v in res.items()})

# The claim the next cell makes, held rather than stated: a mask that leaks the next token must
# score LOWER than the correct one. If this ever fails, the leak did not reach the model.
correct = losses["correct"]["standard_attention"]["summary"]["mean"]
leaky = losses["leaky"]["standard_attention"]["summary"]["mean"]
assert leaky < correct, f"the leaky model should score lower: {leaky:.3f} against {correct:.3f}"
"""),
        md("""
**Which would you ship?** The leaky model's loss is far lower — because it is reading the answer.
A loss that looks too good is a bug until proven otherwise. The generic test
`test_output_at_t_never_depends_on_a_later_token` is what catches this: it changes the future and
checks the past did not move.
"""),
        code("""
ops.causal_mask = leaky_mask
try:
    layer = registry.get("standard_attention").build("lab").double().eval()
    x = torch.randn(1, 10, 32, dtype=torch.float64)
    changed = x.clone(); changed[:, 6:] = torch.randn_like(changed[:, 6:])
    with torch.no_grad():
        a, _ = layer(x); b, _ = layer(changed)
    leaked = not torch.allclose(a[:, :6], b[:, :6])
finally:
    ops.causal_mask = real_mask
assert leaked, "the leak should be visible"
print("changing token 6 moved the output at position 5: the mask leaks")
"""),
        md("""
## 9 · What Kimi K3 and MiniMax actually did

Read from their own reports and released configs, each sentence re-found in the downloaded file:
"""),
        code("""
ledger = hparams.ledger()
for key in sorted(ledger):
    if any(s in key for s in ("kda.k3", "kimi", "lightning_attention", "kda_hybrid")):
        r = ledger[key]
        print(f"{key}\\n    “{r['quote']}”\\n    {r['where']}  [{r['status']}, {r['match']}]\\n")
"""),
        md("""
- **Kimi K3** stacks three KDA layers per Gated MLA layer (plus a final Gated MLA), bounds KDA's
  decay from below, gives KDA a full-rank output gate, and runs MLA with no position encoding.
  `kimi_k3_stack` is built only from those stated facts. K3's cross-layer *Attention Residuals*
  were not read and are not implemented.
- **MiniMax-01 and M1** place one softmax block after every seven lightning-attention blocks.
  `lightning_hybrid` follows that ratio. **MiniMax-M2** returned to full attention in every layer —
  its report says so — which is itself a finding about where hybrids stand.
"""),
        md("""
## 10 · What to change, and what should move

| change | what should move | what should not |
| --- | --- | --- |
| `d` in §1b to 64 | unscaled softmax rows become one-hot | the scaled rows |
| RoPE base | the score-against-distance curve's period | the offset-only property |
| `SEQ` in §2 | the extrapolation gap at 2× and 4× | the in-length loss ranking |
| `n_kv_heads` of `gqa` | cache bytes, in proportion | the output shape |
| `window` of `sliding_window` | bytes stop growing past it | causality |
| `SEQ` in §5 (the recall task stores `SEQ // 4` pairs) | linear attention falls behind the delta rules as pairs grow | the delta rules' recall at low pair counts |
| `SEEDS` | the spread in §7 | which variants beat the spread |
| `LITE = False` | every curve gets longer; epochs pass one | the identities in §5 |

**No assertion in this notebook may fail after any change above.** They state identities, not
results.
"""),
    ]


WHERE = """## Where this sits

Notebook 8 of 14 — [the map of all fourteen](README.md). Previous: [`07-model-embeddings-internals.ipynb`](07-model-embeddings-internals.ipynb).
Next: [`09-loss-functions-output-heads.ipynb`](09-loss-functions-output-heads.ipynb).

**Takes in:** exercise 09's frozen corpus and exercise 02's tokenizer, through `attention.lab.data`.
Every model here is trained on the same text the rest of the path uses, so a difference between
two mechanisms is a difference between mechanisms, not between datasets.
"""

CARRY = """## Carry forward

**What the rest of the path uses:** exercise 09's model — and every exercise built on it — keeps
plain causal multi-head attention with learned absolute positions. That is a deliberate baseline,
not a verdict on this lab: it is the mechanism everything else is easiest to compare against.

**How to change it:** prove the alternative here first, on 09's corpus, against the same budget.
`attention.lab.model` takes a mixer per layer, so a variant that wins here is one argument away
from a full model.
"""


def linked(body: list[dict]) -> list[dict]:
    """The notebook with its place in the fourteen-notebook path stated at both ends."""
    return [body[0], md(WHERE), *body[1:], md(CARRY)]


def build() -> dict:
    """The notebook, as a dict."""
    return {
        "cells": linked(cells()),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    """Write the notebook."""
    notebook = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in notebook["cells"])
    print(f"wrote {OUT} ({len(notebook['cells'])} cells, {n_code} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
