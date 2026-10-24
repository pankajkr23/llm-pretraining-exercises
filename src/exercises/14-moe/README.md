# 14 · From a dense model to a mixture of experts

**Notebook:** [`notebooks/14-moe.ipynb`](../../../notebooks/14-moe.ipynb) — run this exercise's pipeline yourself, every setting in one cell · [open in Colab](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/14-moe.ipynb) · [all fourteen, in order](../../../notebooks/README.md)

**Exercise 13's dense 21M-parameter model, trained on 50 million tokens, converted into a mixture of
experts by copying each block's feed-forward layer into eight experts behind a router — shown to
start exactly where the dense model stopped, and to keep training and reducing its loss, next to the
dense model trained on the same further tokens.** The numbers are in [`RESULTS.md`](RESULTS.md) and
every training step is in [`submission_artifacts/run.log`](submission_artifacts/run.log). The
argument is also told as [a page](https://llm-pretraining-demos.vercel.app/14-moe/), with the
conversion, the router's balance and the cost drawn from the same files.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is): what an expert is, what
  the router does, and why a converted model can carry on from where the dense one stopped.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then [Run it](#run-it).
  The conversion is one class, `moe.layer.MoE`.
- **Deciding whether to believe it** — go to [The evidence](#the-evidence), then
  [What this cannot establish](#what-this-cannot-establish).
- **Running it yourself** — [`notebooks/14-moe.ipynb`](../../../notebooks/14-moe.ipynb) ([open in Colab](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/14-moe.ipynb)) runs this exercise's own pipeline step by step,
  with every setting in one cell; [the map](../../../notebooks/README.md) shows where it sits
  among the fourteen.

## Best result, and what could beat it

**Best result.** The conversion is exact: the dense model and the converted one, before any update,
both score a validation loss of **3.052571**, with the largest logit difference **1.6e-05**. Trained
on the same 10,000,000 further tokens, **the MoE ends at 3.0172 against the dense control's 3.0240
(-0.0068)** — with 31,159,040 parameters active per token against 21,278,720. One run each, so that
gap measures these two runs and does not rank the designs. All of it is in [`RESULTS.md`](RESULTS.md),
rendered from [`results/upcycle.json`](results/upcycle.json), with every step in
[`submission_artifacts/run.log`](submission_artifacts/run.log).

**Candidates that did or could do better.**

- **The dense control led for part of the run.** It was lower at every measurement from step 300 to
  700 and fell behind only from step 800 ([`RESULTS.md`](RESULTS.md) §3), so reading the curves at
  any of those points reverses the order.
- **Sigmoid routing lost by 0.0007** (3.0682 against softmax's 3.0675) on one short run each — a
  choice, not a ranking ([`DECISIONS.md`](DECISIONS.md) D6).
- **Untested:** the expert count and top-k (D3), the continuation's learning rate (D7), and a second
  seed for either arm.

**The next hypothesis.** Repeat both continuations with a second and third seed: if the MoE's
final-loss advantage is smaller than the seed-to-seed spread of either arm, the -0.0068 is noise and
the published result reduces to "conversion is lossless". Separately, extend both arms past
10,000,000 tokens; if the gap keeps widening as the experts grow apart, the advantage comes from
routing rather than from the extra parameters alone.

## What this is

**A dense model** sends every token through the same feed-forward layer in every block.
**A mixture of experts** keeps several feed-forward layers — the experts — and a small **router** that
scores them for each token; each token goes to its `k` best-scoring experts, and their outputs are
added, weighted by the scores. So the model holds many experts' worth of parameters but each token
only pays for `k` of them.

**Converting rather than starting over.** Each block's trained feed-forward layer is copied into
every expert ("upcycling"), and a new router is added. The router's chosen scores are renormalised to
sum to one, so at the moment of conversion

    output = Σ (weight × expert(x)) = (Σ weights) × dense(x) = dense(x)

— the converted model computes exactly what the dense model did, and training resumes from the same
loss rather than from a jump. After that, different tokens reach different experts, the experts get
different gradients, and they grow apart.

**Keeping the experts busy.** A router left alone tends to send most tokens to a few experts. Each
expert carries a bias added to its score **only when choosing** the top-k; after every step the bias
of an over-used expert is lowered a little and that of an under-used one raised — no extra loss term
pulling against the language model (Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for
Mixture-of-Experts*, arXiv:2408.15664). Every step logs the largest imbalance and the number of
experts that received no tokens.

**What is run:**

1. **Continuity** — the dense model's validation loss and the converted model's, before any update.
2. **Router** — softmax against sigmoid scoring, a short continuation of each; the better one is kept.
3. **Continuation** — the converted model (8 experts, 2 per token) and the dense model, each trained
   on the same 10 million further tokens with the same schedule, validation loss tracked throughout.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | `Preset` and the three scales: `FULL` (from exercise 13's model), `LITE` (trains its own), `SMOKE` (the tests) |
| `layer.py` | `MoE` — experts copied from a dense MLP, a float32 router, top-k, bias balancing; `upcycle` converts a whole model; load-balance statistics; total and active parameter counts |
| `train.py` | one training run to a token budget for a dense or converted model, writing the training log |
| `experiments.py` | the dense starting point, continuity, the router trial and the two continuations |
| `runs.py` | provenance over this package and exercise 11's, including the digest of the dense checkpoint used |

`tools/run_experiments.py` runs everything and writes a new dated set, `results/reruns/<today>/`,
holding `upcycle.json` and its training log `run.log`. The published `results/upcycle.json` and
`submission_artifacts/run.log` are never overwritten. `tools/render_results.py` writes `RESULTS.md`,
or with `--results` a dated set's own. The model architecture, data reader and schedules are
exercise 11's; the dense weights are exercise 13's trained baseline, loaded with `weights_only=True`.

## Run it

```bash
uv sync --all-packages --extra train
uv run pytest src/exercises/14-moe

# FULL starts from exercise 13's trained baseline, so run exercise 13 first. It uses 13's
# published run if its checkpoint is on this machine, else 13's newest re-run; --dense-run picks one.
uv run python src/exercises/14-moe/tools/run_experiments.py              # -> results/reruns/<today>/
uv run python src/exercises/14-moe/tools/render_results.py --results results/reruns/<today>

uv run python src/exercises/14-moe/tools/run_experiments.py --preset lite    # self-contained, minutes
```

### Or run it as a notebook

[`notebooks/14-moe.ipynb`](../../../notebooks/14-moe.ipynb) trains a small dense model, converts it, checks the conversion changes
nothing, watches the router balance itself, and continues the MoE against the dense model, plotting
both curves — then reads the published results.

## The evidence

All of it is in [`RESULTS.md`](RESULTS.md) and the log.

- **Continuity is tested, not assumed.** The converted layer is held to the dense layer in float64 to
  1e-12 for several expert counts and both routers, and a twin test perturbs the experts to prove the
  check can fail. The published run reports the validation loss before and after conversion.
- **The continuation has a control.** The dense model trains on exactly the same further tokens with
  the same schedule, so "the loss kept falling" can be separated from "the experts helped".
- **The log is the raw record.** One line every few steps: loss, learning rate, validation loss when
  measured, and for the MoE the load-balance violation, the dead-expert count and the first layer's
  per-expert load.
- **Total and active parameters are both reported**, because an MoE with more parameters is expected
  to do better per step; the honest comparison names what each token actually used.

## What this cannot establish

- **One seed, one run.** The continuation was trained once for each model. A small difference between
  the MoE and the dense continuation is not a ranking; the curves are reported so its size can be
  judged, not to declare a winner.
- **Ten million tokens is a short continuation.** The experts start identical and diverge only as
  routing differentiates them; a longer run could change the comparison in either direction, and the
  published result says nothing about where it ends up.
- **The router trial is a choice, not a study.** Softmax and sigmoid each got one short run; the better
  one was kept. A different budget or seed could pick the other.
- **The MoE here is a reference implementation.** Experts are run one after another in a Python loop,
  so its tokens per second say nothing about an optimised MoE kernel, and it is slower than the dense
  model for that reason as well as for running two experts per token.
- **The data is the same corpus the dense model trained on**, drawn as new random windows. The
  bundle records how many tokens the continued model has read in total against the corpus size. A
  total under one epoch does not mean no text is seen twice: windows are drawn at random, so as the
  total approaches the corpus size more of them overlap text already read. Both arms are equally
  affected, so the comparison stays fair; the absolute losses are a little optimistic.
