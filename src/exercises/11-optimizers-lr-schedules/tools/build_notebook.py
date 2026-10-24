"""Build `notebooks/11-optimizers-lr-schedules.ipynb`.

    uv run python src/exercises/11-optimizers-lr-schedules/tools/build_notebook.py
    NOTEBOOK_OUT=/tmp/s11.ipynb uv run python .../build_notebook.py     # anywhere else

**Tracked, like the notebook it writes**; edit this file and rebuild, never the notebook (a test
checks the two are byte-identical).

**The notebook imports the package. It never re-implements it.** Every experiment it runs is a
function in `optimizers.experiments`; every number it asserts is computed by the package, so the
notebook cannot disagree with what ships or with `RESULTS.md`.
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
_next_id = 0


def _cell_id() -> str:
    global _next_id
    _next_id += 1
    return f"11-{_next_id:02d}"


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


BADGE = f"https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/{SLUG}.ipynb"


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md(f"""
# 11 · Optimizers and learning-rate schedules, taken apart

[![Open In Colab]({BADGE})]({BADGE})

Five experiments, each run by the package `optimizers` on a real small language model and a real
corpus, each ending in `assert`s that state what it shows:

1. **Adam by hand** — one weight, five real gradients, every quantity written out, checked against PyTorch.
2. **Bias correction off** — how much bigger the steps are without it, and for how long.
3. **The update-to-weight ratio** — per layer, with and without warmup.
4. **Cosine against WSD** — both shaped for 300 steps, both stopped at 200, each tuned first.
5. **The learning rate across widths** — swept at three widths, in the standard parametrization and in muP, and a prediction for a wider model.

**Read it at two depths.** The prose before each cell says what and why; the asserts after it say
what must hold. The last section loads the published full-scale results from `results/`, which
`RESULTS.md` renders.

**Runtime:** `LITE = True` (the default) finishes in a few minutes on a laptop or a free Colab CPU.
`LITE = False` is the published configuration — about an hour on an Apple M4 GPU.
"""),
        code(f"""
# Colab: clone the repo and install exercises 09 and 11. Locally: find the repo from here.
import pathlib
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
MEMBERS = ("09-loss-functions-output-heads", "11-optimizers-lr-schedules")
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
    root = pathlib.Path("llm-pretraining-exercises").resolve()
    for member in MEMBERS:
        subprocess.run(["pip", "-q", "install", "-e", str(root / "src/exercises" / member)], check=True)
else:
    # Found at run time, never baked in: walk up from wherever the notebook is opened.
    root = pathlib.Path.cwd().resolve()
    while not (root / "src" / "exercises").is_dir() and root != root.parent:
        root = root.parent
    if not (root / "src" / "exercises").is_dir():
        raise SystemExit("run this notebook from inside a clone of the repository, or on Colab")
    for member in MEMBERS:
        sys.path.insert(0, str(root / "src/exercises" / member / "src"))
EXERCISE = root / "src/exercises/11-optimizers-lr-schedules"
print("repository:", root)
"""),
        md("""
## Every knob, in one cell

Everything below reads these. `LITE` picks the preset: the same five experiments at a smaller size
and length. **LITE's sweep uses widths 128, 256 and 512**, not the 256, 512 and 1,024 the exercise
specifies, so its prediction answers a smaller question; the published FULL results at the end use
the specified widths.
"""),
        code("""
LITE = True          # False = the published configuration (FULL); slow on a CPU
DEVICE = None        # None picks CUDA, then Apple's MPS, then CPU
FETCH_TOKENS = 3_000_000   # training tokens to fetch if no corpus is present (LITE needs few)

import math
import matplotlib.pyplot as plt
import numpy as np

from optimizers.config import FULL, LITE as LITE_PRESET
from optimizers.train import select_device

PRESET = LITE_PRESET if LITE else FULL
DEVICE = select_device(DEVICE)
print(f"preset {PRESET.name} | device {DEVICE} | model width {PRESET.width}, depth {PRESET.depth}, "
      f"batch {PRESET.batch} x {PRESET.seq_len} tokens | seeds {PRESET.seeds}")
"""),
        md("""
## The corpus

A slice of **FineWeb-Edu** (Hugging Face, sample-10BT), licence **ODC-By**, checked on the dataset's
own card at fetch time. It is tokenized with exercise 02's frozen 10,000-token BPE (through exercise
09), each document followed by a separator id, and split into validation (the first rows) and
training (the rows after) — so validation text is never trained on.

If no corpus is present the next cell fetches a small one: `FETCH_TOKENS` training tokens and a
tenth as many for validation. That needs the network; locally a full corpus fetched earlier is
reused. **Every run here reads a small fraction of one epoch** — the printed figure is the check
`AGENTS.md` asks for, because a run that reads its data many times measures memorisation.
"""),
        code("""
from optimizers.corpus import CORPUS_DIR, open_corpus

if not (CORPUS_DIR / "manifest.json").is_file():
    subprocess.run([sys.executable, str(EXERCISE / "tools/fetch_corpus.py"),
                    "--train-tokens", str(FETCH_TOKENS), "--val-tokens", str(FETCH_TOKENS // 10)],
                   check=True)
corpus = open_corpus(CORPUS_DIR)
m = corpus.manifest
print(f"{m['dataset']} ({m['config']}), licence {m['licence']} checked {m['licence_checked']}")
print(f"train {corpus.tokens('train'):,} tokens (rows {m['splits']['train']['rows']}), "
      f"val {corpus.tokens('val'):,} tokens (rows {m['splits']['val']['rows']})")
longest = (PRESET.schedule_total + PRESET.branch_decay) * PRESET.batch * PRESET.seq_len
print(f"longest single run here reads {longest:,} tokens = {corpus.epochs(longest):.3f} epochs")
assert m["splits"]["val"]["rows"][1] == m["splits"]["train"]["rows"][0], "splits must not overlap"
"""),
        md("""
## 1 · Adam, by hand

For one weight `w` and its gradients `g₁ … g₅`, Adam keeps two running averages and turns them into
a step:

| quantity | update | what it is |
| --- | --- | --- |
| `m` | `β₁·m + (1−β₁)·g` | the direction: an average of recent gradients (β₁ = 0.9, about 10 steps) |
| `v` | `β₂·v + (1−β₂)·g²` | the scale: an average of recent squared gradients (β₂ = 0.999, about 1,000 steps) |
| `m̂`, `v̂` | `m/(1−β₁ᵗ)`, `v/(1−β₂ᵗ)` | bias correction: both averages start at zero, so early on they are too small by exactly this factor |
| step | `−η · m̂/(√v̂ + ε)` | a step of roughly η per weight, whatever the gradient's size |

The cell trains the model for five steps with plain `torch.optim.Adam`, records the five gradients
one weight actually received, and recomputes Adam three ways: written out in Python floats, by
`torch.optim.Adam` on a lone float64 scalar, and as the weight's value inside the model.
"""),
        code("""
from optimizers import experiments
from optimizers.adam import adam_by_hand, torch_adam, largest_difference

r1 = experiments.adam_by_hand(PRESET, corpus, "cpu")
print(f"weight {r1['weight']}, w0 = {r1['w0']:.9f}, eta = {r1['lr']}")
print(f"{'t':>2} {'g':>12} {'m':>12} {'v':>12} {'m_hat':>12} {'v_hat':>12} {'update':>12} {'w':>14}")
for s in r1["by_hand"]:
    print(f"{s['t']:>2} {s['g']:>12.4e} {s['m']:>12.4e} {s['v']:>12.4e} {s['m_hat']:>12.4e} "
          f"{s['v_hat']:>12.4e} {s['update']:>12.4e} {s['w']:>14.9f}")
print("largest difference, by hand vs torch scalar:", max(r1["hand_vs_torch"].values()))
print("largest difference, by hand vs the weight inside the model:", r1["hand_vs_model"])
assert max(r1["hand_vs_torch"].values()) < 1e-12   # the same arithmetic
assert r1["hand_vs_model"] < 1e-6                  # float32 inside the model, float64 by hand
assert abs(r1["by_hand"][0]["update"]) == min(abs(r1["by_hand"][0]["update"]), 1.0001 * r1["lr"])
"""),
        md("""
The first step is **η, almost exactly, whatever the gradient**: at `t = 1` bias correction makes
`m̂ = g` and `v̂ = g²`, so `m̂/√v̂ = ±1`. That is the sense in which Adam "moves every weight by about η".

**Try it:** put your own gradients in the next cell — a tiny one, a huge one, a sign flip — and watch
the step stay near η while `m` and `v` swing by orders of magnitude.
"""),
        code("""
mine = [1e-6, 1e-6, 50.0, -3.0, 0.2]
hand, ref = adam_by_hand(1.0, mine), torch_adam(1.0, mine)
for s in hand:
    print(f"t={s.t}  g={s.g:>8.1e}  m_hat={s.m_hat:>10.3e}  v_hat={s.v_hat:>10.3e}  update={s.update:>11.3e}")
assert max(largest_difference(hand, ref).values()) < 1e-12
"""),
        md("""
## 2 · Bias correction off

Feed Adam a constant gradient `g`. Then `m_t = g(1 − β₁ᵗ)` and `v_t = g²(1 − β₂ᵗ)` exactly, so without
correction the step is `η · (1 − β₁ᵗ)/√(1 − β₂ᵗ)` instead of `η`. At step 1 that is `0.1/√0.001 = 3.16η`.

**It rises before it falls.** The numerator recovers in about 10 steps (β₁ = 0.9); the denominator
in about 1,000 (β₂ = 0.999). So for a while `m` has caught up and `v` has not, and the uncorrected
step is *larger* than at step 1. The first plot is the twenty steps the exercise asks for; the
second is how long it really takes.
"""),
        code("""
from optimizers.adam import uncorrected_over_corrected, steps_until_within

steps20 = range(1, 21)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].plot(list(steps20), [uncorrected_over_corrected(t) for t in steps20], marker="o")
ax[0].axhline(1, color="grey", lw=0.8); ax[0].set(xlabel="step", ylabel="uncorrected / corrected step", title="the first 20 steps")
long = np.unique(np.logspace(0, 4, 200).astype(int))
ax[1].semilogx(long, [uncorrected_over_corrected(t) for t in long])
ax[1].axhline(1, color="grey", lw=0.8); ax[1].set(xlabel="step (log)", title="how long the gap lasts")
plt.tight_layout(); plt.show()
within = {tol: steps_until_within(tol) for tol in (0.10, 0.05, 0.01)}
print("uncorrected step within 10% / 5% / 1% of corrected from step:", within)
assert abs(uncorrected_over_corrected(1) - math.sqrt(1000) / 10) < 1e-12
assert min(uncorrected_over_corrected(t) for t in steps20) > 3   # within 20 steps it always matters
"""),
        md("""
That is the closed form. On a real model the trajectories diverge after step 1, so the honest
measurement is the **loss**: the same seed and data with and without correction, set against how much
two seeds of the corrected run differ for no reason at all. The effect "stops mattering" when it
falls inside that noise.
"""),
        code("""
r2 = experiments.bias_correction(PRESET, corpus, DEVICE)
L = {k: np.asarray(v) for k, v in r2["losses"].items()}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for k in L:
    ax[0].plot(range(1, PRESET.bias_steps + 1), L[k][: PRESET.bias_steps], label=k)
ax[0].set(xlabel="step", ylabel="training loss", title=f"first {PRESET.bias_steps} steps"); ax[0].legend()
ax[1].plot(np.abs(L["uncorrected"] - L["corrected"]), label="|uncorrected - corrected|")
ax[1].plot(np.abs(L["corrected_seed1"] - L["corrected"]), label="|seed 1 - seed 0| (noise)", alpha=0.7)
ax[1].set(xlabel="step", ylabel="loss gap", title="effect against noise"); ax[1].legend()
plt.tight_layout(); plt.show()
print("the effect falls inside the seed noise from step:", r2["settles_at_step"], f"(of {r2['horizon']})")
"""),
        md("""
## 3 · The update-to-weight ratio, per layer

`ratio = ‖ΔW‖ / ‖W‖` for each matrix after each step: the fraction one step changes the layer by.
Adam moves each weight by about η, and a fresh hidden weight is about `1/√fan_in` in size, so at full
η from step one a wide layer can change by a large fraction in a single step. **Warmup holds η down
while that is true.** Two runs: one warms up, one does not; after warmup both hold η constant, so
warmup is the only difference.

"The step at which warmup stops changing it" is **measured**: the first step after which a layer's
smoothed ratio stays within ±10% of its own late level (`optimizers.ratios.settles_at`).
"""),
        code("""
r3 = experiments.update_ratio(PRESET, corpus, DEVICE)
warm, cold = r3["runs"]["warmup"], r3["runs"]["no_warmup"]
names = list(warm["ratios"])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for name in names:
    ax[0].semilogy(warm["ratios"][name], lw=0.8)
    ax[1].semilogy(cold["ratios"][name], lw=0.8)
ax[0].axvline(warm["warmup"], color="k", ls="--", lw=0.8)
ax[0].set(title=f"warmup over {warm['warmup']} steps", xlabel="step", ylabel="‖ΔW‖/‖W‖")
ax[1].set(title="no warmup", xlabel="step")
plt.tight_layout(); plt.show()
print(f"median layer settles at step {warm['median_settle']} (warmup = {warm['warmup']})")
for name in names[:6]:
    print(f"  {name:28} settles {str(warm['settles_at'][name]):>5}   early peak {warm['peak_early'][name]:.2e} "
          f"with warmup vs {cold['peak_early'][name]:.2e} without")
higher = sum(cold["peak_early"][n] > warm["peak_early"][n] for n in names)
print(f"{higher} of {len(names)} matrices peak higher early without warmup")
"""),
        md("""
## 4 · Cosine against WSD, stopped at step 200

**Cosine** takes the total budget as an input to every rate, so it commits to that budget up
front; halt it partway and the rate never got low enough to finish. **WSD** — warmup, stable, decay — holds the peak flat and decays
only at the end, so a checkpoint from the stable phase can be **branched** and decayed on its own.

First, what the two schedules look like over 300 steps, and where step 200 falls.
"""),
        code("""
from optimizers.schedules import cosine, wsd, wsd_branch

T, S, W = PRESET.schedule_total, PRESET.schedule_stop, PRESET.schedule_warmup
steps = range(T)
plt.figure(figsize=(8, 3))
plt.plot(steps, [cosine(s, T, 1.0, warmup=W) for s in steps], label="cosine (shaped for 300)")
plt.plot(steps, [wsd(s, T, 1.0, warmup=W, decay_fraction=PRESET.wsd_decay_fraction) for s in steps], label="WSD (shaped for 300)")
branch = range(S, S + PRESET.branch_decay)
plt.plot(list(branch), [wsd_branch(s, S, PRESET.branch_decay, 1.0) for s in branch], label="WSD branch from 200", lw=2)
plt.axvline(S, color="k", ls="--", lw=0.8); plt.xlabel("step"); plt.ylabel("lr / peak"); plt.legend(); plt.show()
print(f"at step {S}: cosine is at {cosine(S - 1, T, 1.0, warmup=W):.2f} of peak, WSD at "
      f"{wsd(S - 1, T, 1.0, warmup=W, decay_fraction=PRESET.wsd_decay_fraction):.2f}")
"""),
        md("""
Now the experiment. **Each schedule is tuned before they are compared** — several peak rates, two
seeds each, trained to step 200 — because a comparison against a badly tuned baseline is the classic
way an optimiser claim fails to replicate. Then each runs at its own best peak, and the two finished
models at a 200-step budget are trained too: WSD's branch, and a cosine planned for 200 from the start.
"""),
        code("""
r4 = experiments.schedules(PRESET, corpus, DEVICE)
print("tuning, validation loss at step", S, "(mean over seeds):")
for kind in ("cosine", "wsd"):
    print(f"  {kind:6}", {p: round(float(np.mean(v)), 4) for p, v in r4["tuning"][kind].items()})
print("best peaks:", r4["best_peak"])
f = r4["final"]
mean = lambda xs, k: float(np.mean([x[k] for x in xs]))
rows = {
    "cosine (300) stopped at 200": mean(f["cosine"], "at_stop"),
    "WSD (300) stopped at 200": mean(f["wsd"], "at_stop"),
    "cosine run to 300": mean(f["cosine"], "at_end"),
    "WSD run to 300": mean(f["wsd"], "at_end"),
    "WSD branch: 200 + decay": mean(f["wsd_branch"], "at_end"),
    "cosine planned for 200": mean(f["cosine_planned_for_stop"], "at_end"),
}
for k, v in rows.items():
    print(f"  {k:30} {v:.4f}")
noise = max(np.ptp([x["at_stop"] for x in f["cosine"]]), np.ptp([x["at_stop"] for x in f["wsd"]]))
print(f"seed-to-seed spread at step 200: {noise:.4f}")
"""),
        md("""
**Which would you keep?** At step 200 neither run is finished: WSD has not decayed, cosine is part
of the way down. The fair question is which *finished* model you can have for a 200-step budget —
and only WSD gives you one without restarting, by branching. Compare the last two rows, and read every
difference against the seed spread before calling it.
"""),
        md("""
## 5 · The learning rate across widths

In the **standard parametrization (SP)** the best learning rate shrinks as the model widens — roughly
halving when width doubles — so a value tuned on a small model overshoots on a big one. **muP**
(Yang et al., *Tensor Programs V*, arXiv:2203.03466, Table 3) changes three things per layer so the
best rate stays put:

| layer | init variance | Adam learning rate |
| --- | --- | --- |
| embeddings, biases, norm gains | 1/fan_in, both | η, both |
| hidden matrices | 1/fan_in, both | SP η; muP η·base/fan_in |
| output head | SP 1/fan_in; muP base/fan_in² | SP η; muP η·base/fan_in |

with `base` the smallest width, so the two agree exactly there. The sweep trains each width at
learning rates that double, finds each minimum with a parabola on `log₂ η`, fits a power law across
widths, and predicts the next width — with the spread between seeds as the confidence.
"""),
        code("""
from optimizers.sweep import find_minimum

r5 = experiments.width_sweep(PRESET, corpus, DEVICE)
lrs = r5["lrs"]
fig, axes = plt.subplots(1, len(r5["results"]), figsize=(11, 3.4), sharey=True)
for ax, (param, res) in zip(np.atleast_1d(axes), r5["results"].items()):
    for w in r5["widths"]:
        curve = np.mean([res["losses"][str(w)][s] for s in res["losses"][str(w)]], axis=0)
        line, = ax.semilogx(lrs, curve, marker="o", label=f"width {w}")
        best = find_minimum(lrs, list(curve))
        ax.axvline(best.lr, color=line.get_color(), ls=":", lw=1)
    ax.set(title=param.upper(), xlabel="learning rate (log)", ylabel="validation loss"); ax.legend()
plt.tight_layout(); plt.show()
for param, res in r5["results"].items():
    p = res["prediction"]
    print(f"{param.upper():4} exponent {p['exponent']:+.2f} -> eta at width {r5['predict_width']}: {p['prediction']:.2e} "
          f"(per seed {', '.join(f'{x:.2e}' for x in p['per_seed'])}; {p['extrapolation']:.0f}x beyond the widest measured)")
"""),
        md("""
**Reading it.** In SP the dotted minima should step left as width grows and the exponent should sit
near −1; in muP they should line up and the exponent should sit near 0. The prediction is an
extrapolation beyond the widest width measured, and the seed spread is the honest error bar on it.
LITE uses shorter runs and smaller widths, so treat its numbers as a demonstration; the published
answer is below.

## The published results (FULL)

`tools/run_experiments.py` ran all five at the FULL preset and wrote `results/*.json`, each with its
provenance; `tools/render_results.py` turned them into `RESULTS.md`. Here are the headline numbers,
read from those files.
"""),
        code("""
import json

results = EXERCISE / "results"
if (results / "width_sweep.json").is_file():
    full = {p.stem: json.loads(p.read_text()) for p in results.glob("*.json")}
    sweep = full["width_sweep"]["result"]
    for param, res in sweep["results"].items():
        p = res["prediction"]
        print(f"FULL {param.upper():4} widths {sweep['widths']}: exponent {p['exponent']:+.2f}, "
              f"predicted eta at {sweep['predict_width']}: {p['prediction']:.2e}")
    sched = full["schedules"]["result"]["final"]
    for k in ("cosine", "wsd"):
        print(f"FULL {k:6} stopped at 200: {np.mean([x['at_stop'] for x in sched[k]]):.4f}")
    print(f"FULL WSD branch: {np.mean([x['at_end'] for x in sched['wsd_branch']]):.4f}, "
          f"cosine planned for 200: {np.mean([x['at_end'] for x in sched['cosine_planned_for_stop']]):.4f}")
    print("provenance:", {k: (v if isinstance(v, str) else '...') for k, v in full["width_sweep"]["provenance"].items()})
else:
    print("no FULL results in this checkout yet; see RESULTS.md once they are committed")
"""),
        md("""
## What this cannot tell you

- **The models are small and the runs short.** Hundreds of steps on models of 8 to 60 million
  parameters. A schedule's advantage, or the position of a minimum, can move at a hundred times the
  scale; these experiments show the mechanisms, not production settings.
- **The width prediction is an extrapolation.** It reaches four times past the widest width measured,
  from three points; the seed spread bounds the noise but not the model error of assuming a power law.
- **muP here is the paper's Table 3 rules for Adam on this architecture**, with a fixed head size so
  its attention rule is a constant. It is not a full muP implementation of every layer type.
- **Two seeds** set the noise floor. That is enough to refuse a difference smaller than the spread,
  not enough to estimate the spread precisely.
"""),
    ]


WHERE = """## Where this sits

Notebook 11 of 14 — [the map of all fourteen](README.md). Previous: [`10-training-loop.ipynb`](10-training-loop.ipynb).
Next: [`12-distributed-training.ipynb`](12-distributed-training.ipynb).

**Takes in:** a larger corpus than 09's — the token files under `data/fineweb-edu/`, fetched once
by this exercise's `tools/fetch_corpus.py` — and its own decoder in standard or muP
parametrisation.
"""

CARRY = """## Carry forward

**What the later notebooks take from here:** exercises 13 and 14 build on this exercise's decoder
and read the same corpus through `optimizers.corpus`, so its format is a contract: changing the
separator id, the dtype or the split order changes their results too.

**The habit to keep:** measure the noise floor before ranking anything — two identical runs
differ, and a schedule comparison inside that difference is not a comparison.
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
