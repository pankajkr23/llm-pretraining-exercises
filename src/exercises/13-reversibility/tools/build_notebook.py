"""Build `notebooks/13-reversibility.ipynb`.

    uv run python src/exercises/13-reversibility/tools/build_notebook.py
    NOTEBOOK_OUT=/tmp/s13.ipynb uv run python .../build_notebook.py

**Tracked, like the notebook it writes**; edit this file and rebuild, never the notebook. The
notebook imports `reversible` and `optimizers`;
it never re-implements them.
"""

import json
import os
from pathlib import Path

EXERCISE = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if EXERCISE.name == "artifacts":  # staged copy: artifacts/staged/build_notebook.py
    EXERCISE = EXERCISE.parent
REPO = EXERCISE.parents[2]
SLUG = EXERCISE.name
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (REPO / "notebooks" / f"{SLUG}.ipynb"))
REPO_URL = "https://github.com/pankajkr23/llm-pretraining-exercises"
BADGE = f"https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/{SLUG}.ipynb"
_next_id = 0


def _cell_id() -> str:
    global _next_id
    _next_id += 1
    return f"13-{_next_id:02d}"


def md(text: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """A code cell with no stored output."""
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
        md(f"""
# 13 · Reversible training: memory against batch size

[![Open In Colab]({BADGE})]({BADGE})

Training keeps every block's activations until the backward pass needs them, so memory grows with
depth and batch. A **reversible** stack keeps only the top two states and rebuilds the rest during
the backward pass, by running each layer's update backwards. This notebook:

1. measures what an ordinary stack keeps, against depth;
2. builds the three reversible rules and undoes one by hand;
3. **proves** on a small model that rebuilding gives exactly the gradients storing does;
4. runs the exercise's three experiments at a small scale — fixed batch, reversible at the same batch,
   reversible at its largest batch — and plots loss, speed and memory;
5. reads the published full-scale results (21M parameters, 50M tokens).

**Runtime:** `LITE = True` takes minutes on a laptop or a free Colab runtime.
"""),
        code(f"""
# Colab: clone the repo and install exercises 09, 11 and 13. Locally: find the repo from here.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
MEMBERS = ("09-loss-functions-output-heads", "11-optimizers-lr-schedules", "13-reversibility")
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    for member in MEMBERS:
        subprocess.run(["pip", "-q", "install", "-e", str(root / "src/exercises" / member)], check=True)
else:
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit("run this notebook from inside a clone of the repository, or on Colab")
    for member in MEMBERS:
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))
EXERCISE = root / "src/exercises/13-reversibility"
print("repository:", root)
"""),
        md("""
## Every knob, in one cell

`LITE` picks the preset. LITE is a 128-wide, 8-block model on a few hundred thousand tokens; FULL is
the published 320-wide, 12-block model on 50 million tokens (an hour or more on a GPU).
"""),
        code("""
LITE = True
DEVICE = None                 # None picks CUDA, then Apple's MPS, then CPU
FETCH_TOKENS = 2_000_000      # training tokens to fetch if exercise 11's corpus is absent

import matplotlib.pyplot as plt
import numpy as np
import torch

from optimizers.train import select_device
from reversible.config import FULL, LITE as LITE_PRESET

PRESET = LITE_PRESET if LITE else FULL
DEVICE = select_device(DEVICE)
print(f"preset {PRESET.name} | device {DEVICE} | width {PRESET.width} depth {PRESET.depth} "
      f"seq {PRESET.seq_len} batch {PRESET.batch} | {PRESET.tokens:,} tokens per long run")
"""),
        code("""
from optimizers.corpus import CORPUS_DIR, open_corpus

if not (CORPUS_DIR / "manifest.json").is_file():
    fetch = root / "src/exercises/11-optimizers-lr-schedules/tools/fetch_corpus.py"
    subprocess.run([sys.executable, str(fetch), "--train-tokens", str(FETCH_TOKENS),
                    "--val-tokens", str(FETCH_TOKENS // 10)], check=True)
corpus = open_corpus(CORPUS_DIR)
print(f"{corpus.manifest['dataset']}: {corpus.tokens('train'):,} training tokens; "
      f"a long run here reads {corpus.epochs(PRESET.tokens):.3f} epochs")
"""),
        md("""
## 1 · What an ordinary stack keeps

`reversible.memory.saved_bytes` counts every tensor a forward pass keeps for the backward pass, using
PyTorch's saved-tensor hooks. It is exact and works on any device. Below: an ordinary residual stack
and a reversible one, at depths 2 to 12, same width and batch.
"""),
        code("""
from optimizers.model import ModelConfig
from reversible.memory import saved_bytes
from reversible.model import ChainedGPT

ids = torch.randint(0, 10_001, (8, 129), generator=torch.Generator().manual_seed(0))
depths = [2, 4, 6, 8, 10, 12]
kept = {}
for variant in ("standard", "blend"):
    kept[variant] = []
    for depth in depths:
        model = ChainedGPT(ModelConfig(width=128, depth=depth, seq_len=128), variant, h=0.5, loss_chunk=256)
        kept[variant].append(saved_bytes(lambda: model.loss(ids)) / 2**20)
plt.figure(figsize=(7, 3.2))
for variant, values in kept.items():
    plt.plot(depths, values, marker="o", label=variant)
plt.xlabel("depth (blocks)"); plt.ylabel("MiB kept for backward"); plt.legend(); plt.title("activation memory against depth"); plt.show()
growth = {v: (kept[v][-1] - kept[v][0]) / (depths[-1] - depths[0]) for v in kept}
print({v: f"{g:.2f} MiB per extra block" for v, g in growth.items()})
assert growth["blend"] < 0.25 * growth["standard"]
"""),
        md("""
The reversible line still rises a little: every forward pass keeps a reference to each block's
weights, and deeper models have more weights. What it does not keep is per-block **activations** —
which is what grows with batch.

## 2 · The rules, and undoing one by hand

Gal et al. (arXiv:2512.02056) give rules of the form `next = A·previous + B·current + C·f(current)`:

| rule | A | B | C |
| --- | --- | --- | --- |
| midpoint (eq. 4) | 1 | 0 | 2h |
| blend (eq. 15) | a | 1 − a | h |
| leapfrog (eq. 6) | −1 | 2 | h² |

Because `f` is evaluated at `current`, the state below can be recovered:
`previous = (next − B·current − C·f(current)) / A`. The ordinary residual stack, `next = current + f(current)`,
is forward Euler — and has no `previous` in it, so it cannot be undone.
"""),
        code("""
from reversible.stack import Rule

for name in ("midpoint", "blend", "leapfrog"):
    rule = Rule.make(name, h=0.5, a=0.5)
    g = torch.Generator().manual_seed(1)
    previous, current, update = (torch.randn(3, 5, dtype=torch.float64, generator=g) for _ in range(3))
    nxt = rule.step(previous, current, update)
    back = rule.invert(nxt, current, update)
    print(f"{name:9} A={rule.A:+.2f} B={rule.B:+.2f} C={rule.C:.2f}  max |rebuilt - true| = {(back - previous).abs().max().item():.1e}")
    assert torch.allclose(back, previous, atol=1e-12)
"""),
        md("""
## 3 · Rebuilding gives exactly the gradients storing does

The memory-saving backward pass (`memory="reversible"`) is compared with ordinary autograd through the
same recurrence (`memory="stored"`), in float64, for every rule. If these differed, every memory and
speed number later would be a number about the wrong gradients.
"""),
        code("""
cfg = ModelConfig(vocab_size=50, width=64, depth=6, seq_len=10)
small = torch.randint(0, 50, (3, 11), generator=torch.Generator().manual_seed(2))
for name in ("midpoint", "blend", "leapfrog"):
    a = ChainedGPT(cfg, name, h=0.5, memory="reversible", seed=1).double()
    b = ChainedGPT(cfg, name, h=0.5, memory="stored", seed=1).double()
    la, lb = a.loss(small), b.loss(small)
    la.backward(); lb.backward()
    worst = max((p.grad - q.grad).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
    print(f"{name:9} same loss: {la.item() == lb.item()}   largest gradient difference {worst:.1e}")
    assert la.item() == lb.item() and worst < 1e-9
"""),
        md("""
## 4 · Which variant trains

Short runs, same data, scored by validation loss: first the ordinary stack's learning rate, then
every rule and step size `h` at that rate. The best rule is used from here on.
"""),
        code("""
from reversible import experiments

trial = experiments.trials(PRESET, corpus, DEVICE)
print("baseline:", {lr: round(r["final_val"], 4) for lr, r in trial["baseline"].items()}, "-> lr", trial["best_lr"])
for key, r in trial["reversible"].items():
    print(f"  {key:14} {'diverged' if r['diverged'] else round(r['final_val'], 4)}")
choice = {**trial["choice"], "lr": trial["best_lr"]}
print("chosen:", choice)
"""),
        md("""
## 5 · Same batch: the baseline and the reversible model

Same initial weights, same tokens, same learning rate. Compare the loss curves, the tokens per second
and the bytes kept for backward.
"""),
        code("""
fixed = experiments.fixed_batch(PRESET, corpus, DEVICE, choice)
base, rev = fixed["baseline"], fixed["reversible"]
plt.figure(figsize=(8, 3.2))
for label, run in (("standard", base), (f"{rev['variant']} h={rev['h']}", rev)):
    xs = np.arange(len(run["losses"])) * run["curve_every"]
    plt.plot(xs, run["losses"], label=label, lw=0.8)
plt.xlabel("step"); plt.ylabel("training loss"); plt.legend(); plt.show()
for label, run in (("standard  ", base), ("reversible", rev)):
    print(f"{label}  final val {run['final_val']:.4f}  {run['tokens_per_second']:>9,.0f} tok/s  "
          f"kept for backward {run['saved_bytes'] / 2**20:7.1f} MiB")
speed = rev["tokens_per_second"] / base["tokens_per_second"] - 1
kept = rev["saved_bytes"] / base["saved_bytes"]
print(f"reversible runs {abs(speed):.0%} {'faster' if speed > 0 else 'slower'} per token, and keeps "
      f"{1 / kept:.1f}x {'fewer' if kept < 1 else 'MORE'} bytes for backward than the baseline")
"""),
        md("""
## 6 · The largest batch, and training at it

**The search caps GPU memory for the life of the process it runs in**, so — exactly as the published
pipeline does — it runs in a fresh child process, and this kernel is never capped. A cap set here
would also count every tensor the earlier sections left on the GPU against the budget, and the run
would fail for reasons that have nothing to do with the model. On a CPU nothing can be capped, and
the largest batch is derived from the measured bytes instead. The reversible model then trains at a
fraction of the largest batch found — never at the exact edge — after a short learning-rate check.
"""),
        code("""
import json

# **A known defect, stated rather than hidden.** On Apple's GPU the LITE preset's 1 GiB cap is too
# tight for the validation pass inside the run at the largest batch: the search probes training
# steps only, and `train.evaluate` then needs memory the probe never measured. The package's own
# `tools/run_experiments.py --preset lite` fails the same way on MPS (recorded in this exercise's
# PROGRESS.md). Fixing it changes the package, whose code digest the published results pin, so it
# is a separate change. Until then this stage runs on the CPU there, where nothing is capped and
# the largest batch is derived from the measured bytes instead. CUDA is unaffected.
cap_device = "cpu" if DEVICE == "mps" else DEVICE
print(f"capped stage runs on: {cap_device}")

child = f'''
import json, sys
sys.path[:0] = {[str(root / "src/exercises" / m / "src") for m in MEMBERS]!r}
from optimizers.corpus import CORPUS_DIR, open_corpus
from reversible import experiments
from reversible.config import FULL, LITE
preset = {"LITE" if LITE else "FULL"}
corpus = open_corpus(CORPUS_DIR)
choice = {choice!r}
found = experiments.max_batch(preset, corpus, {cap_device!r}, choice)
largest = found["reversible"]["measured_max_batch"] or found["reversible"]["derived_max_batch"]
batch = max(1, int(largest * preset.max_batch_run_fraction))
big = experiments.max_batch_run(preset, corpus, {cap_device!r}, choice, batch)
print(json.dumps({{"found": found, "largest": largest, "batch": batch, "big": big}}, default=str))
'''
done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, cwd=root)
if done.returncode != 0:
    raise RuntimeError("the capped stage failed in its own process:\\n" + done.stderr[-3000:])
result = json.loads(done.stdout.strip().splitlines()[-1])
found, largest, batch, big = result["found"], result["largest"], result["batch"], result["big"]
for label in ("baseline", "reversible"):
    v = found[label]
    print(f"{label:10} measured {v['measured_max_batch']}  derived {v['derived_max_batch']}  "
          f"bytes per sequence {v['saved_bytes_per_sample']:,}")
print(f"training at {batch} = {PRESET.max_batch_run_fraction:.0%} of the largest batch found, {largest}")
r = big["run"]
print(f"reversible at batch {batch}: {r['steps']} steps, final val {r['final_val']:.4f}, "
      + (f"{r['tokens_per_second']:,.0f} tok/s" if r["tokens_per_second"] else "not timed (too few steps)")
      + f" (lr x{big['lr_multiplier']:g})")
"""),
        md("""
## The published results (FULL)

`tools/run_experiments.py` ran all four at the FULL preset; `tools/render_results.py` rendered
`RESULTS.md` from the bundles. The headline numbers, read from them:
"""),
        code("""
import json

results = EXERCISE / "results"
if (results / "fixed_batch.json").is_file():
    full = {p.stem: json.loads(p.read_text())["result"] for p in results.glob("*.json")}
    fb = full["fixed_batch"]
    for label in ("baseline", "reversible"):
        r = fb[label]
        print(f"FULL {label:10} batch {fb['batch']}: val {r['final_val']:.4f}, {r['tokens_per_second']:,.0f} tok/s, "
              f"kept {r['saved_bytes'] / 2**20:,.0f} MiB, {r['parameters']:,} params")
    if "max_batch" in full:
        mb = full["max_batch"]
        print(f"FULL largest batch in {mb['budget_gib']} GiB: baseline {mb['baseline']['measured_max_batch']}, "
              f"reversible {mb['reversible']['measured_max_batch']}")
    if "max_batch_run" in full:
        r = full["max_batch_run"]["run"]
        print(f"FULL reversible at batch {full['max_batch_run']['batch']}: val {r['final_val']:.4f}, {r['tokens_per_second']:,.0f} tok/s")
else:
    print("no FULL results in this checkout yet; see RESULTS.md once they are committed")
"""),
        md("""
## What this cannot tell you

- **One seed per long run**, so a small difference in final loss is not a ranking.
- **Apple's GPU has no peak-memory counter**; the exact measure is the bytes kept for backward, and the
  largest batch is found by trying, under a fixed cap that is our choice.
- **A 12-block model shows less of the saving than a deep one**: the saving grows with depth.
- **`h` and `a` were chosen on short runs**; the rules are only marginally stable, so a long run is the
  real test, which is why its curve is plotted.
"""),
    ]


WHERE = """## Where this sits

Notebook 13 of 14 — [the map of all fourteen](README.md). Previous: [`12-distributed-training.ipynb`](12-distributed-training.ipynb).
Next: [`14-moe.ipynb`](14-moe.ipynb).

**Takes in:** exercise 11's decoder, chained by a reversible rule, and exercise 11's corpus.
"""

CARRY = """## Carry forward

**What the next notebook takes from here:** the dense baseline this exercise trains is the model
exercise 14 converts into a mixture of experts (its full run loads that checkpoint), and the
learning rate and token count 14 uses are read from this exercise's results, never typed.
"""


def linked(body: list[dict]) -> list[dict]:
    """The notebook with its place in the fourteen-notebook path stated at both ends."""
    return [body[0], md(WHERE), *body[1:], md(CARRY)]


def main() -> int:
    """Write the notebook to `OUT`."""
    notebook = {
        "cells": linked(cells()),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
