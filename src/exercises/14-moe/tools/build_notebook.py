"""Build `notebooks/14-moe.ipynb`.

    uv run python src/exercises/14-moe/tools/build_notebook.py
    NOTEBOOK_OUT=/tmp/s14.ipynb uv run python .../build_notebook.py

**Tracked, like the notebook it writes**; edit this file and rebuild, never the notebook. The
notebook imports `moe` and `optimizers`; it
never re-implements them.
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
    return f"14-{_next_id:02d}"


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
# 14 · From a dense model to a mixture of experts

[![Open In Colab]({BADGE})]({BADGE})

1. train a small dense model (or, with `LITE = False`, load exercise 13's 21M-parameter one);
2. **convert** it: copy each block's feed-forward layer into several experts behind a new router;
3. **check** the converted model computes exactly what the dense one did;
4. continue training the MoE and, as a control, the dense model on the same tokens;
5. watch the router balance its experts, and read the training log;
6. read the published full-scale results.

**Runtime:** `LITE = True` takes minutes on a laptop or a free Colab runtime.
"""),
        code(f"""
# Colab: clone the repo and install exercises 09, 11 and 14. Locally: find the repo from here.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
MEMBERS = ("09-loss-functions-output-heads", "11-optimizers-lr-schedules", "14-moe")
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
EXERCISE = root / "src/exercises/14-moe"
print("repository:", root)
"""),
        code("""
LITE = True          # False loads exercise 13's trained model (run exercise 13 first)
DEVICE = None
FETCH_TOKENS = 2_000_000

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

from moe.config import FULL, LITE as LITE_PRESET
from optimizers.train import select_device

PRESET = LITE_PRESET if LITE else FULL
DEVICE = select_device(DEVICE)
print(f"preset {PRESET.name} | device {DEVICE} | width {PRESET.width} depth {PRESET.depth} | "
      f"{PRESET.n_experts} experts, top-{PRESET.top_k}")
"""),
        code("""
from optimizers.corpus import CORPUS_DIR, open_corpus

if not (CORPUS_DIR / "manifest.json").is_file():
    fetch = root / "src/exercises/11-optimizers-lr-schedules/tools/fetch_corpus.py"
    subprocess.run([sys.executable, str(fetch), "--train-tokens", str(FETCH_TOKENS),
                    "--val-tokens", str(FETCH_TOKENS // 10)], check=True)
corpus = open_corpus(CORPUS_DIR)
print(f"{corpus.manifest['dataset']}: {corpus.tokens('train'):,} training tokens")
"""),
        md("""
## 1 · The dense model

A dense model sends every token through the same feed-forward layer in every block. LITE trains a
small one here; FULL loads exercise 13's trained baseline (loaded with `weights_only=True`, so the
file can only ever yield tensors).
"""),
        code("""
from moe import experiments

# FULL starts from exercise 13's run: its published one if the checkpoint is on this machine, else
# its newest re-run. The runner's own `dense_run` decides, so this cell and the tool cannot differ.
ckpt, dense_results = None, None
if not LITE:
    import importlib.util
    spec = importlib.util.spec_from_file_location("run14", root / "src/exercises/14-moe/tools/run_experiments.py")
    run14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(run14)
    ckpt, dense_results = run14.dense_run(PRESET)
    print("exercise 13's run:", dense_results.relative_to(root))
dense, origin = experiments.dense_model(PRESET, corpus, DEVICE, ckpt)
print("dense model:", origin["source"], "|", f"{dense.count_parameters():,} parameters")
if "trace" in origin:
    losses = origin["trace"]["losses"]
    plt.figure(figsize=(7, 3)); plt.plot(losses, lw=0.8); plt.xlabel("step"); plt.ylabel("loss"); plt.title("dense training"); plt.show()
"""),
        md("""
## 2 · Convert, and check nothing changed

Every expert is a copy of the dense layer; the router's chosen scores are renormalised to sum to one.
So `output = Σ weight × expert(x) = dense(x)`: the converted model must have the same loss.
"""),
        code("""
cont = experiments.continuity(PRESET, corpus, DEVICE, dense)
print(f"dense val {cont['val_dense']:.6f} | upcycled val {cont['val_upcycled']:.6f} | "
      f"difference {cont['val_difference']:+.1e} | largest logit difference {cont['largest_logit_difference']:.1e}")
print(f"parameters: dense {cont['parameters_dense']:,} -> MoE {cont['parameters_moe_total']:,} total, "
      f"{cont['parameters_moe_active']:,} active per token")
assert abs(cont["val_difference"]) < 1e-4
"""),
        md("""
**Try it:** look inside one converted layer — the experts start identical, and the router starts
almost uniform. Scores for a few tokens:
"""),
        code("""
from moe.layer import MoEConfig, upcycle, moe_layers

probe = upcycle(copy.deepcopy(dense), MoEConfig(n_experts=PRESET.n_experts, top_k=PRESET.top_k))
layer = moe_layers(probe)[0]
x = torch.randn(5, PRESET.width, device=next(probe.parameters()).device)
print("router scores (5 tokens x experts):\\n", layer.scores(x).detach().cpu().numpy().round(3))
same = all(torch.equal(layer.experts[0].up.weight, e.up.weight) for e in layer.experts)
print("experts identical at conversion:", same)
"""),
        md("""
## 3 · Which router

Softmax or sigmoid scoring before the top-k: one short continuation each, from the same dense model.
"""),
        code("""
trials = None if LITE else dense_results / "trials.json"  # the same 13 run as the weights
lr = PRESET.continue_lr_fraction * experiments.dense_peak(PRESET, trials)
print(f"continuation peak rate {lr:g} ({PRESET.continue_lr_fraction} x the dense model's peak)")
trial = experiments.router_trial(PRESET, corpus, DEVICE, dense, lr)
print("validation loss:", {k: round(v, 4) for k, v in trial["final_val"].items()}, "-> using", trial["choice"])
"""),
        md("""
## 4 · Keep training: the MoE against the dense model

Both start from the same weights and loss, train on the same further tokens, with the same schedule.
The log line for each step records loss, learning rate and — for the MoE — the largest load imbalance
and the experts that got no tokens.
"""),
        code("""
log_path = EXERCISE / "artifacts" / "notebook-run.log"
if log_path.exists():
    log_path.unlink()
result = experiments.continuation(PRESET, corpus, DEVICE, dense, trial["choice"], log_path, lr)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for label in ("moe", "dense"):
    val = sorted((int(k), v) for k, v in result[label]["val"].items())
    ax[0].plot([s + 1 for s, _ in val], [v for _, v in val], marker="o", label=label)
ax[0].set(xlabel="step after conversion", ylabel="validation loss", title="continuing to train"); ax[0].legend()
bal = sorted((int(k), v) for k, v in result["moe"]["balance"].items())
ax[1].plot([s for s, _ in bal], [v["max_violation"] for _, v in bal])
ax[1].set(xlabel="step", ylabel="largest load violation", title="router balance (0 = even)")
plt.tight_layout(); plt.show()
moe_val = sorted((int(k), v) for k, v in result["moe"]["val"].items())
print(f"MoE validation loss {moe_val[0][1]:.4f} -> {moe_val[-1][1]:.4f}")
assert moe_val[-1][1] < moe_val[0][1], "the converted model must keep reducing its loss"
print("".join(open(log_path).readlines()[:6]))
"""),
        md("""
## The published results (FULL)

`tools/run_experiments.py` converted exercise 13's trained model and continued both for 10 million
tokens; `submission_artifacts/run.log` is the full training log.
"""),
        code("""
import json

path = EXERCISE / "results" / "upcycle.json"
if path.is_file():
    r = json.loads(path.read_text())["result"]
    c = r["continuity"]
    print(f"FULL continuity: val difference {c['val_difference']:+.1e}")
    for label in ("moe", "dense"):
        val = sorted((int(k), v) for k, v in r["continuation"][label]["val"].items())
        print(f"FULL {label:5} val {val[0][1]:.4f} -> {val[-1][1]:.4f}")
else:
    print("no FULL results in this checkout yet; see RESULTS.md once they are committed")
"""),
        md("""
## What this cannot tell you

- **One run each**, so a small gap between the MoE and the dense continuation is not a ranking.
- **A short continuation**: the experts start identical and only diverge as routing differentiates
  them; a longer run could change the comparison either way.
- **A reference implementation**: experts run one after another in Python, so the MoE's speed says
  nothing about an optimised kernel.
"""),
    ]


WHERE = """## Where this sits

Notebook 14 of 14 — [the map of all fourteen](README.md). Previous: [`13-reversibility.ipynb`](13-reversibility.ipynb).
This is the last notebook in the path.

**Takes in:** exercise 11's decoder and corpus, and — for the full run — exercise 13's trained
dense checkpoint, which this exercise copies into experts.
"""

CARRY = """## Carry forward

**The path ends here.** Every notebook from 02 onward hands something to a later one, and
[the map](README.md) lists each handoff. The natural next step is a single run that chains the
choices made along the way — 02's tokenizer, 04's cleaning, 05's mix, 06's shards, and exercise
11's decoder with its optimiser and schedule — end to end. That decoder is the one 13 and 14 build
on, and it is 11's own model rather than 09's, because muP needs per-layer learning rates and a
separately initialised output head.
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
