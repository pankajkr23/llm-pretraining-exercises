"""Build `notebooks/12-distributed-training.ipynb`.

    uv run python src/exercises/12-distributed-training/tools/build_notebook.py

**Tracked, like the notebook it writes**; edit this file and rebuild, never the notebook (a test
checks the two are byte-identical). `NOTEBOOK_OUT` overrides the output path (a full file path, not
a directory) — use it for any trial build, so a test never overwrites the notebook a reader has open.

**The notebook imports the package. It never re-implements it.** The only arithmetic written in a
cell is the hand calculation a reader is meant to do — and every one of those ends in an `assert`
against what the simulator *measured*, so the hand calculation and the code cannot quietly disagree.
"""

import json
import os
from pathlib import Path

EXERCISE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPO_ROOT = EXERCISE.parents[2]
SLUG = EXERCISE.name
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (REPO_ROOT / "notebooks" / f"{SLUG}.ipynb"))

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
    return f"s12-{_next_id:02d}"


def md(text: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """A code cell with no stored output — outputs are stripped, always."""
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md(
            f"""
# 12 · ZeRO on 32 simulated GPUs — where the memory goes, and what it costs to move

[![Open In Colab]({BADGE})]({BADGE})

Data parallelism trains one model on many GPUs by giving each a full copy and a different slice of
the batch. It works, and every GPU stores **exactly the same training state** — thirty-two GPUs,
thirty-two identical copies. ZeRO stops that duplication in three steps.

This notebook builds 32 devices inside one Python process, trains a small transformer on them under
**data parallelism (stage 0)** and **ZeRO stages 1, 2 and 3**, and for every claim does two things:

1. **works the number out by hand**, in a cell, from first principles;
2. **measures it** on the simulator — the real byte size of every tensor each device holds, and every
   byte every ring collective sends — and `assert`s the two are equal.

If a hand calculation is wrong, the cell fails. That is the point.

Everything comes from the exercise's package, `zerosim`. Nothing is re-implemented here.

**Runtime: about a minute on a laptop or a free Colab CPU with `LITE = True`.** Nothing needs a GPU.
"""
        ),
        code(
            f"""
# Colab: clone the repo and install exercises 09 (the model) and 12 (the simulator).
# Locally: find the repository root by walking up from here, and import from source.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    # 09 first: 12 depends on it, and pip cannot resolve a uv workspace source.
    subprocess.run(["pip", "-q", "install", "-e", str(root / "src/exercises/09-loss-functions-output-heads")], check=True)
    subprocess.run(["pip", "-q", "install", "-e", str(root / "src/exercises/12-distributed-training") + "[train]"], check=True)
else:
    # Found at run time, never baked in at build time: a notebook carrying the build machine's
    # absolute path imports on exactly one computer.
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit("run this notebook from inside a clone of the repository, or on Colab")
    for member in ("09-loss-functions-output-heads", "12-distributed-training"):
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))

print("repo root:", root)
"""
        ),
        md(
            """
## Every knob, in one cell

Every value below is **read** by a later cell. Change one, re-run from here, and watch what moves —
and, more usefully, what does not.
"""
        ),
        code(
            """
LITE = True            # False: more steps and a finer sweep of N. Same conclusions, slower.

WORLD = 32             # simulated devices
NODES = 4              # machines they sit on (8 per node); the ring crosses a node boundary 4 times
STEPS = 2 if LITE else 4                                     # optimiser steps per stage
SCALING_NS = (1, 2, 3, 8, 32) if LITE else (1, 2, 3, 4, 8, 16, 32, 64)   # N values to measure at
LADDER_PARAMS = 30 * 10**9                                   # the large hypothetical model
CARD_BYTES = 80 * 10**9                                      # ASSUMED device memory, decimal GB
SEED = 12

from dataclasses import replace

import matplotlib.pyplot as plt
import torch

from zerosim.config import Config

CONFIG = replace(Config(), world_size=WORLD, nodes=NODES, steps=STEPS, seed=SEED,
                 ladder_params=LADDER_PARAMS, card_bytes=CARD_BYTES)
print(f"profile {'lite' if LITE else 'full'} · {CONFIG.world_size} devices on {CONFIG.nodes} nodes "
      f"· {CONFIG.steps} steps · precision {CONFIG.mode} · torch {torch.__version__}")
"""
        ),
        md(
            """
## 1 · The sixteen bytes

Training one weight with Adam in mixed precision costs **16 bytes**. Where they go decides
everything that follows:

| what | dtype | bytes | why |
| --- | --- | ---: | --- |
| the weight | bf16 | 2 | what forward and backward compute with |
| its gradient | bf16 | 2 | produced by backward |
| fp32 master copy | fp32 | 4 | bf16 is too coarse to accumulate many tiny updates into |
| Adam `m` | fp32 | 4 | running average of the gradient |
| Adam `v` | fp32 | 4 | running average of the squared gradient |

Twelve of the sixteen exist only for the optimiser. **Activations are not in this table** — they
depend on the batch, they are a separate bill, and ZeRO does not shard them.

> **A common mistake:** "a batch of 8 means 8 copies of the model". It does not. A batch multiplies
> *activations*. Data parallelism has N copies of the weights because it has N *devices*.

The cell below reads the prices from the package — and asks torch for the real element size of
each dtype, so the table above cannot be wrong without this cell failing.
"""
        ),
        code(
            """
from zerosim.precision import recipe

RECIPE = recipe(CONFIG.mode)
rows = [("weight", RECIPE.param_dtype, RECIPE.param_bytes), ("gradient", RECIPE.grad_dtype, RECIPE.grad_bytes),
        ("master", "float32", RECIPE.master_bytes), ("adam m", "float32", RECIPE.moment_bytes),
        ("adam v", "float32", RECIPE.moment_bytes)]
for name, dtype, nbytes in rows:
    print(f"{name:10s} {dtype:10s} {nbytes} bytes")
print(f"{'total':21s} {RECIPE.total_bytes} bytes")

assert torch.empty((), dtype=RECIPE.torch_dtype("param")).element_size() == RECIPE.param_bytes
assert RECIPE.total_bytes == 2 + 2 + 4 + 4 + 4 == 16
"""
        ),
        md(
            """
## 2 · Thirty-two devices, one ring

Each device is a `Rank` object with its own tensors and a **ledger** that adds up the real `nbytes`
of everything it is given. The ledger knows nothing about ZeRO: it measures, it never predicts.

The devices talk in a **ring**: device *r* only ever sends to device *r* + 1. With 32 devices on 4
nodes, the ring stays inside a node for 28 links and crosses between nodes on 4.
"""
        ),
        code(
            """
from zerosim.world import World

world = World(CONFIG.world_size, CONFIG.devices_per_node)
kinds = [kind for _, _, kind in world.ring_links()]
print(f"{CONFIG.devices_per_node} devices per node; ring links: "
      f"{kinds.count('intra')} intra-node, {kinds.count('inter')} inter-node")
print("the inter-node links:", [(s, d) for s, d, k in world.ring_links() if k == "inter"])
assert kinds.count("inter") == (CONFIG.nodes if CONFIG.nodes > 1 else 0)   # one node: no crossing
"""
        ),
        md(
            """
### The three collectives, run on four devices you can read

A **reduce-scatter** sums a buffer across devices but gives each device only its own slice of the
sum. An **all-gather** does the reverse: each contributes a slice, all end with the whole. An
**all-reduce** gives everyone the whole sum — and it *is* a reduce-scatter followed by an all-gather.

Small integers make the sums easy to check by eye.
"""
        ),
        code(
            """
from zerosim.collectives import all_gather, all_reduce, reduce_scatter

tiny = World(4)
buffers = [torch.arange(8, dtype=torch.float32) * (r + 1) for r in range(4)]
for r, b in enumerate(buffers):
    print(f"device {r} starts with {b.tolist()}")
total = torch.stack(buffers).sum(0)
print("the sum everyone wants:", total.tolist())

shards = reduce_scatter(tiny, buffers)
for r, s in enumerate(shards):
    print(f"after reduce-scatter, device {r} holds chunk {r} of the sum: {s.tolist()}")
gathered = all_gather(tiny, shards)
assert all(torch.equal(g, total) for g in gathered)
print("after all-gather, every device holds the whole sum")

P = buffers[0].numel() * buffers[0].element_size()
print(f"bytes sent per device: {tiny.comm.sent}  —  by hand 2 × P × (N−1)/N = 2 × {P} × 3/4 = {2 * P * 3 // 4}")
assert tiny.comm.sent == [2 * P * 3 // 4] * 4
"""
        ),
        md(
            """
## 3 · The model, cut into units

The model is exercise 09's transformer, imported. ZeRO-3 gathers weights **one unit at a time**, so
the model is cut into units: the embeddings, each block, and the final norm with the output head.
Each unit's weights become one flat buffer, padded so it divides into N equal slices.

> **A common mistake:** "ZeRO-3 splits the model by layers across devices". That is **pipeline
> parallelism**. In ZeRO-3, device 5 holds slice 5 of **every** unit, and every device still runs
> every layer on its own data.
"""
        ),
        code(
            """
from zerosim import flat
from zerosim.model import build_model

model = build_model(CONFIG)
print(f"{'unit':8s} {'weights':>9s} {'padding':>8s} {'shard/device':>13s}")
for unit in model.units:
    spec = flat.layout(unit.name, unit.shapes, CONFIG.world_size)
    print(f"{unit.name:8s} {spec.numel:9,d} {spec.padding:8d} {spec.shard:13,d}")
weights = model.numel
PADDED = sum(flat.layout(u.name, u.shapes, CONFIG.world_size).padded for u in model.units)
print(f"total {weights:,} weights ({PADDED:,} after padding)")
"""
        ),
        md(
            """
## 4 · Memory per device — by hand first, then measured

**By hand.** At N devices, a device holds per weight:

- **DP**: 2 + 2 + 12 = 16, at any N;
- **ZeRO-1**: the 12 optimiser bytes are split N ways → 2 + 2 + 12/N;
- **ZeRO-2**: gradients split too → 2 + 2/N + 12/N = 2 + 14/N;
- **ZeRO-3**: weights split too → 16/N.

The next cell writes those four lines as arithmetic, multiplies by the padded weight count, and
only *then* trains each stage and reads the ledger.
"""
        ),
        code(
            """
N = CONFIG.world_size
by_hand = {0: 2 + 2 + 12, 1: 2 + 2 + 12 / N, 2: 2 + 2 / N + 12 / N, 3: 16 / N}
for stage, bpw in by_hand.items():
    print(f"stage {stage}: {bpw:.4f} bytes/weight × {PADDED:,} = {bpw * PADDED:,.0f} bytes per device")
"""
        ),
        code(
            """
from zerosim.stages import STAGE_NAMES, STAGES, train
from zerosim.world import PERSISTENT

trainers = {s: train(CONFIG, s, model=model, count_flops=True) for s in STAGES}

print(f"{'stage':14s} " + " ".join(f"{c:>10s}" for c in PERSISTENT) + f" {'total':>11s} {'by hand':>11s}")
for s, t in trainers.items():
    peak = t.world.ranks[0].ledger.peak
    measured = sum(peak[c] for c in PERSISTENT)
    print(f"{STAGE_NAMES[s]:14s} " + " ".join(f"{peak[c]:10,d}" for c in PERSISTENT)
          + f" {measured:11,d} {by_hand[s] * PADDED:11,.0f}")
    assert measured == by_hand[s] * PADDED, f"stage {s}: the hand calculation is wrong"
    # every device holds exactly what device 0 holds
    assert all(rk.ledger.peak == t.world.ranks[0].ledger.peak for rk in t.world.ranks)
print("\\nevery hand calculation equals the ledger, on every device")
"""
        ),
        md(
            """
### The price of sharding: short bursts

ZeRO-2 and -3 never hold a full gradient for long — each unit's gradient is reduce-scattered as soon
as backward produces it, then freed. ZeRO-3 never holds a unit's full weights for long either — it
gathers them just before a unit runs, then frees them, and gathers them **again** for the backward.

The plot is device 0's ledger, event by event, for one training step: the total bytes it holds.
DP and ZeRO-1 are flat. ZeRO-3 is a sawtooth whose teeth are one unit tall.
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(9, 3.5))
for s, t in trainers.items():
    held, series = 0, []
    for op, _, _, nbytes in t.world.ranks[0].ledger.events:
        held += nbytes if op == "alloc" else -nbytes
        series.append(held / 1e6)
    ax.plot(series, label=STAGE_NAMES[s])
ax.set_xlabel("ledger event (allocations and frees, in order)")
ax.set_ylabel("MB held by device 0")
ax.set_yscale("log")
ax.legend()
plt.tight_layout(); plt.show()

z3 = trainers[3].world.ranks[0].ledger
largest = max(flat.layout(u.name, u.shapes, N).padded for u in model.units) * RECIPE.param_bytes
print(f"ZeRO-3 gathered-weights peak {z3.peak['gathered_params']:,} bytes = the largest unit, {largest:,}")
assert z3.peak["gathered_params"] == largest
assert all(z3.current[c] == 0 for c in ("gathered_params", "grad_bucket"))
"""
        ),
        md(
            """
## 5 · Communication — by hand, then counted

Let **P** be one full copy of the weights in bf16. In a ring, a reduce-scatter or an all-gather
makes each device send N − 1 chunks of P/N bytes: **P · (N − 1)/N**.

- **DP** all-reduces the gradients: reduce-scatter + all-gather = **2 · P · (N−1)/N**.
- **ZeRO-1, ZeRO-2** reduce-scatter the gradients and all-gather the updated weights: also **2 ·
  P · (N−1)/N**. They run the same two phases DP runs, and keep the half-way result.
- **ZeRO-3** all-gathers the weights in forward, again in backward, and reduce-scatters the
  gradients: **3 · P · (N−1)/N** — 50% more.
"""
        ),
        code(
            """
P = PADDED * RECIPE.grad_bytes
by_hand_comm = {s: (3 if s == 3 else 2) * P * (N - 1) // N for s in STAGES}
print(f"P = {PADDED:,} × {RECIPE.grad_bytes} = {P:,} bytes")
for s, t in trainers.items():
    sent = t.world.comm.sent[0] // t.steps_taken
    print(f"{STAGE_NAMES[s]:14s} sent {sent:12,d} bytes/step   by hand {by_hand_comm[s]:12,d}"
          f"   = {sent / (P * (N - 1) / N):.0f} × P·(N−1)/N")
    assert sent == by_hand_comm[s]
    assert len(set(t.world.comm.sent)) == 1, "every device sends the same"
"""
        ),
        md(
            """
## 6 · Computation — what changes and what does not

Every device still runs every layer on its own micro-batch, so **forward and backward FLOPs per
device are the same in every stage**. (Each unit's forward runs twice here — once forward, once
recomputed inside backward — identically in every stage; see the exercise's `DECISIONS.md`.) What
shrinks is the **optimiser**: from stage 1 on, a device updates only its 1/N of the weights.
"""
        ),
        code(
            """
for s, t in trainers.items():
    f = {k: v[0] // t.steps_taken for k, v in t.flops.items()}
    opt = t.optimizer_elements[0] // t.steps_taken
    print(f"{STAGE_NAMES[s]:14s} forward {f['forward']:,}  recompute {f['recompute']:,}  "
          f"backward {f['backward']:,}  optimiser updates {opt:,} weights")
    assert t.flops == trainers[0].flops
    assert opt == PADDED // (N if s >= 1 else 1)
"""
        ),
        md(
            """
## 7 · Does the stage change the answer?

ZeRO changes where numbers live and how they travel. It must not change what is computed. Here the
four stages are compared **bit for bit**.
"""
        ),
        code(
            """
dp = trainers[0].full_weights()
for s in (1, 2, 3):
    other = trainers[s].full_weights()
    identical = all(torch.equal(dp[k], other[k]) for k in dp)
    print(f"{STAGE_NAMES[s]} vs DP: bit-identical = {identical}")
    assert identical
"""
        ),
        md(
            """
And against **one device trained on the whole global batch** with `torch.optim.AdamW`, in fp32.
The losses agree to rounding. The weights agree closely everywhere except one place — the attention
**key bias** — and the reason is worth a minute: the key bias adds the same number to every score in
a softmax row, and softmax ignores a constant added to a whole row, so its true gradient is **exactly
zero**. What a computer produces is rounding noise, and Adam divides each gradient by its own running
size — so noise becomes a full step.
"""
        ),
        code(
            """
from zerosim.experiment import key_bias_positions
from zerosim.reference import first_gradients, train_reference

fp32 = replace(CONFIG, mode="fp32")
sim = train(fp32, 0)
ref, ref_losses = train_reference(fp32)
masks = key_bias_positions(sim)
w = sim.full_weights()
elsewhere = max(float((w[k] - ref[k])[~masks[k]].abs().max()) for k in w)
key_bias = max(float((w[k] - ref[k])[masks[k]].abs().max()) for k in w if masks[k].any())
sim_losses = [sum(x) / len(x) for x in sim.losses]
print("losses, simulator:", [f"{x:.6f}" for x in sim_losses])
print("losses, one device:", [f"{x:.6f}" for x in ref_losses])
print(f"largest weight difference: {elsewhere:.2e} elsewhere, {key_bias:.2e} on the key bias")

grads = first_gradients(replace(fp32, steps=1))
scale = max(float(g.abs().max()) for g in grads.values())
kb = max(float(grads[k][masks[k]].abs().max()) for k in grads if masks[k].any())
print(f"the key bias's own gradient: {kb:.1e}, against a largest gradient of {scale:.1e}")
assert elsewhere <= CONFIG.learning_rate / 100      # within 1% of one Adam step
assert kb <= 1e-6 * scale                           # the reason, asserted
"""
        ),
        md(
            """
## 8 · Bytes per weight as N grows — measured, not just predicted

A fresh world at each N, one step, ledger read. The **floor** is what no number of devices removes:
16 for DP, 4 for ZeRO-1 (replicated weights + gradients), 2 for ZeRO-2, 0 for ZeRO-3.
"""
        ),
        code(
            """
from zerosim import formulas

measured = {s: [] for s in STAGES}
for n in SCALING_NS:
    cfg = replace(CONFIG, world_size=n, nodes=max(1, n // CONFIG.devices_per_node), steps=1)
    m = build_model(cfg)
    for s in STAGES:
        t = train(cfg, s, model=m)
        padded = sum(spec.padded for spec in t.layouts)
        bpw = sum(t.world.ranks[0].ledger.peak[c] for c in PERSISTENT) / padded
        assert bpw == float(formulas.bytes_per_weight(s, n, CONFIG.mode)["total"])
        measured[s].append(bpw)

fig, ax = plt.subplots(figsize=(7, 3.5))
for s in STAGES:
    ax.plot(SCALING_NS, measured[s], marker="o", label=STAGE_NAMES[s])
    ax.axhline(float(formulas.floor_bytes_per_weight(s, CONFIG.mode)), ls=":", lw=0.8, color="grey")
ax.set_xscale("log", base=2); ax.set_xlabel("devices N"); ax.set_ylabel("bytes per weight")
ax.legend(); plt.tight_layout(); plt.show()
print("every measured point equals the formula; dotted lines are the floors")
"""
        ),
        md(
            """
## 9 · The ladder: a 30-billion-weight model

Nothing is simulated here — this is the formula, multiplied out, against an **assumed** 80 GB card.
DP and ZeRO-1 can never fit, at any N: their replicated weights and gradients alone are
4 bytes × 30 × 10⁹ = 120 GB.
"""
        ),
        code(
            """
rows = formulas.ladder(CONFIG.ladder_params, CONFIG.ladder_world_sizes, CONFIG.mode, CONFIG.card_bytes)
print(f"{'stage':14s} " + " ".join(f"{'N=' + str(n):>12s}" for n in CONFIG.ladder_world_sizes))
for s in STAGES:
    cells = [next(x for x in rows if x["stage"] == s and x["world_size"] == n)
             for n in CONFIG.ladder_world_sizes]
    print(f"{STAGE_NAMES[s]:14s} " + " ".join(f"{c['gib']:7.1f} {'fits' if c['fits'] else '  no'}" for c in cells))
assert 4 * CONFIG.ladder_params > CONFIG.card_bytes
assert all(x["never_fits"] for x in rows if x["stage"] in (0, 1))
"""
        ),
        md(
            """
## 10 · A time model — assumed figures, clearly labelled

Nothing here crossed a real wire, so **no time below is a measurement**. The counts are measured;
the speeds are assumptions in `Config` (450 GB/s inside a node, 50 GB/s between nodes). A ring step
waits for its slowest link, so the 4 inter-node links pace all 32 devices.
"""
        ),
        code(
            """
from zerosim import timing

for s, t in trainers.items():
    seconds = timing.comm_seconds(t.world.comm.calls, CONFIG) / t.steps_taken
    print(f"{STAGE_NAMES[s]:14s} {seconds * 1e6:8.2f} µs of communication per step (assumed links)")
payload = CONFIG.ladder_params * RECIPE.grad_bytes
for s in (2, 3):
    est = formulas.comm_multiple(s) * timing.ring_seconds(payload, 32, CONFIG)
    print(f"at 30B weights on 32 devices, {STAGE_NAMES[s]}: about {est:.2f} s of communication per step")
"""
        ),
        md(
            """
## 11 · What to take away

| stage | buys | costs |
| --- | --- | --- |
| **DP** | simplest; nothing to gather | 16 bytes/weight everywhere, at any N |
| **ZeRO-1** | 12 of 16 bytes shrink by N, **for free** in communication | 4-byte floor |
| **ZeRO-2** | gradients shrink too, still free | per-layer gradient buckets during backward; 2-byte floor |
| **ZeRO-3** | everything shrinks by N, no floor | **+50% communication**, a gather before every layer, twice |

**Try this:** set `WORLD = 8` and `NODES = 1` in the knobs cell. Every assertion still holds, the
memory savings shrink, and in the time model the ring no longer crosses a node.
"""
        ),
    ]


WHERE = """## Where this sits

Notebook 12 of 14 — [the map of all fourteen](README.md). Previous: [`11-optimizers-lr-schedules.ipynb`](11-optimizers-lr-schedules.ipynb).
Next: [`13-reversibility.ipynb`](13-reversibility.ipynb).

**Takes in:** exercise 09's model, loss, tokenizer and corpus, trained across simulated devices.
"""

CARRY = """## Carry forward

**What the next notebook takes from here:** the byte accounting. ZeRO moves optimiser state,
gradients and weights between devices; it does not shrink the activations a step keeps for its
backward pass. Exercise 13 attacks that other half of the memory bill — by rebuilding activations
instead of storing them.
"""


def linked(body: list[dict]) -> list[dict]:
    """The notebook with its place in the fourteen-notebook path stated at both ends."""
    return [body[0], md(WHERE), *body[1:], md(CARRY)]


def build() -> dict:
    """The notebook document."""
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
    """Write the notebook to `NOTEBOOK_OUT`, or to `notebooks/` by default."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
