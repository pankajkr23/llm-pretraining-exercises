# 11 · Optimizers and learning-rate schedules

**Five experiments on what an optimiser actually does to a small language model: Adam written out by
hand, bias correction switched off, the per-layer update-to-weight ratio, cosine against WSD stopped
early, and the learning rate swept across widths in the standard parametrization and in muP.** Every
measured number is in [`RESULTS.md`](RESULTS.md), which is generated from `results/*.json` and
carries each run's provenance.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is): each experiment is stated
  as a question before any of the machinery. Then open the notebook, which runs all five.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it). Every experiment is one function in `optimizers.experiments`.
- **Deciding whether to believe it** — go to [The evidence](#the-evidence) for how each number was
  measured and against what noise, then [What this cannot establish](#what-this-cannot-establish).

## What this is

An optimiser turns a gradient into a step. Adam does it with two running averages — one for the
direction, one for the scale — and a correction for the fact that both start at zero. A schedule
then decides how large each step is allowed to be over the course of a run. Each experiment here
asks one question about that machinery and answers it by measurement:

1. **Can Adam be reproduced exactly by hand?** One real weight, the five gradients it actually
   received while training, every intermediate quantity written out, then compared with
   `torch.optim.Adam` and with the weight's own value inside the model.
2. **What does bias correction buy, and for how long?** In closed form, and on the real model with
   the correction switched off — set against the difference between two seeds of the same run, so an
   effect is only called an effect when it is larger than noise.
3. **When does warmup stop mattering to each layer?** The update-to-weight ratio `‖ΔW‖/‖W‖`, logged
   for every matrix at every step, with warmup and without.
4. **Cosine or WSD, if you have to stop at step 200 of a 300-step plan?** Each schedule tuned first,
   then both stopped where the exercise stops them, plus the two ways of having a *finished* model at
   that budget: decaying WSD from its checkpoint, or a cosine planned for 200 from the start.
5. **What learning rate should a wider model use?** A sweep at widths 256, 512 and 1,024, the minimum
   at each, and a power-law prediction for width 4,096 with the seed spread as its error bar — in the
   standard parametrization (SP), where the best rate drifts with width, and in muP, where it should not.

**The corpus** is a licence-checked slice of FineWeb-Edu (ODC-By), tokenized with exercise 02's frozen
10,000-token BPE. This exercise owns it because exercises 13 and 14 need far more text than the
repository held: `tools/fetch_corpus.py` fetches it once, validation and training from disjoint row
ranges, and every run reports the fraction of an epoch it read.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | `Preset` and the three scales — `FULL` (published), `LITE` (the notebook), `SMOKE` (the tests) |
| `corpus.py` | the token files: writing a split with its digest, opening it read-only, verifying it, epochs |
| `data.py` | seeded token windows: the same seed reads the same tokens in the same order |
| `adam.py` | Adam in Python floats, the same steps read back out of `torch.optim.Adam`, and the bias-correction gap in closed form |
| `ablation.py` | `SwitchableAdamW`: PyTorch's AdamW update with bias correction as a flag |
| `schedules.py` | warmup, cosine, WSD and a WSD branch, as pure functions of the step |
| `model.py` | a small decoder in SP or muP, and the per-layer learning-rate groups muP needs |
| `ratios.py` | the per-matrix update-to-weight ratio, and the step at which a curve settles |
| `train.py` | the one training loop: schedules, clipping, validation, ratios, checkpoints for branching |
| `sweep.py` | a sweep's minimum between grid points, the power-law fit, and the prediction with its spread |
| `experiments.py` | the five experiments, each a function from a preset and a corpus to plain data |
| `runs.py` | provenance — settings, code, commit, machine, corpus, tokenizer — and a `save` that refuses without it |

Three tools sit beside them:

- `tools/fetch_corpus.py` downloads the corpus, checking the dataset's licence on its own card
  first, and contacts no host but Hugging Face's two;
- `tools/run_experiments.py` runs the experiments and writes one bundle each;
- `tools/render_results.py` turns those bundles into `RESULTS.md`, sentences included.

The model is this exercise's own rather than exercise 09's: muP needs per-layer learning rates and
an output head initialised separately from the embeddings, and exercises 13 and 14 build on its
blocks. [`DECISIONS.md`](DECISIONS.md) records that choice and every other one.

## Run it

```bash
uv sync --all-packages --extra train
uv run pytest src/exercises/11-optimizers-lr-schedules

# the corpus: 66M training + 1M validation tokens (outside the sandbox: its proxy truncates pages)
uv run python src/exercises/11-optimizers-lr-schedules/tools/fetch_corpus.py

# all five experiments at the published scale, then the document
uv run python src/exercises/11-optimizers-lr-schedules/tools/run_experiments.py
uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py

# one experiment, quickly, written to artifacts/ rather than results/
uv run python src/exercises/11-optimizers-lr-schedules/tools/run_experiments.py --preset lite --task schedules
```

### Or run it as a notebook

`notebooks/S11-optimizers-lr-schedules.ipynb` runs all five experiments at the `LITE` scale with a
plot and the asserts that state each lesson, then reads the published results. On Colab it clones
the repository, installs exercises 09 and 11, and fetches a small slice of the same corpus.

## The evidence

All of it is in [`RESULTS.md`](RESULTS.md). How each number was made, so it can be checked:

- **Adam by hand** is compared quantity by quantity (`m`, `v`, `m̂`, `v̂`, the step, the weight)
  with `torch.optim.Adam` in float64, and with the weight inside the float32 model it came from.
- **Bias correction** is measured twice: the closed-form ratio of uncorrected to corrected step,
  whose tests check it against the hand computation, and three real runs at a constant learning rate
  — corrected, uncorrected and a second seed of corrected — so the correction's effect on the loss
  is reported only where it exceeds the gap between seeds. `SwitchableAdamW` with the flag on matches
  `torch.optim.AdamW` to floating-point rounding, so the ablation changes one thing.
- **The update-to-weight ratio** is measured with the weights snapshotted before every step. "Settled"
  is defined before looking: the first step after which the smoothed curve stays within ±10% of its
  own late level. A layer that never settles is reported as such, not given a step.
- **Cosine against WSD** tunes each schedule's peak rate over the same grid and the same two seeds
  before comparing them. A difference between them is called only when it is larger than the spread
  between seeds, and the document says so either way.
- **The width sweep** places each minimum with a parabola on `log₂ η` through the best grid point and
  its neighbours, flags a minimum at the edge of the grid as a bound rather than a value, fits a power
  law per seed and across seeds, and reports how far beyond the widest measured width it reaches.

Every bundle carries a configuration fingerprint, a digest of every module, the commit, the machine,
and digests of the corpus and the tokenizer; `optimizers.runs.save` raises rather than writing one
without them, and a test fails if `RESULTS.md` differs from a fresh render.

## What this cannot establish

- **The models are small and the runs are short.** The experiments train decoders of roughly 8 to 60
  million parameters for hundreds of steps. That is enough to see each mechanism — the size of an
  uncorrected step, the effect of warmup on a ratio, the shape of a schedule, the drift of a minimum
  with width — and not enough to say what the best settings are for a production run, where a
  schedule's advantage or a minimum's position can move.
- **The width prediction is an extrapolation from three points.** Width 4,096 is four times the widest
  width measured. The seed spread bounds the noise in the minima, but nothing here tests the
  assumption that the best rate follows a power law that far out; the prediction should be read with
  that error in mind, and muP exists precisely because this extrapolation is unreliable in SP.
- **muP is implemented as the published table's rules for Adam on this architecture**, with the head
  size fixed so that its attention rule reduces to a constant. It is not a general muP library, and
  it has not been checked against the authors' code.
- **Two seeds set every noise floor.** That is enough to refuse a difference smaller than the spread
  and not enough to estimate the spread itself precisely; a difference just above it is a weak claim.
- **The corpus is one contiguous slice of one dataset**, the first rows of a published sample. Results
  on other text, other tokenizers or other languages are not measured.
