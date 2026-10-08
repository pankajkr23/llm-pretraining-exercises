# DECISIONS — 11-optimizers-lr-schedules

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · This exercise owns the corpus that 11, 13 and 14 train on

**Decision.** A slice of FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, config `sample-10BT`), fetched by
`tools/fetch_corpus.py` into the gitignored `data/fineweb-edu/`, tokenized with exercise 02's frozen
BPE through exercise 09. 66 million training tokens and 1 million validation tokens by default.

**Why.** Exercise 13 trains on 50 million tokens and exercise 14 continues from there; the largest
corpus in the repository before this held about 11.8 million, so those runs would have read it four
times over, and `AGENTS.md` is explicit that a run past about one epoch measures memorisation. One
fetch serves all three. 11 owns it because it is first in the order; 13 and 14 depend on this
package rather than repeat the fetch. FineWeb-Edu is the corpus exercise 06 already used for its web
lane, it declares a permissive licence (ODC-By) on its own card, and it is ungated.

**What would overturn it.** A shared data package becoming worth its own workspace member (the root
`pyproject.toml` would have to change); or a licence change on the card, which the fetcher would
refuse at the next fetch.

## D2 · Validation first, training after, from disjoint row ranges

**Decision.** The fetcher reads rows from 0 until the validation target is met, then continues from
the next row for training, and records both ranges in the manifest. A test asserts they do not overlap.

**Why.** A held-out loss measured on trained text is not held out. Taking contiguous ranges is
simpler and more auditable than sampling; the cost is that both splits come from the start of the
published sample, which is stated in the README's limits.

**What would overturn it.** Evidence that the sample's row order is not mixed — that the first rows
are systematically different from later ones — which would make a random row sample worth the extra
complexity.

## D3 · The end-of-document marker is exercise 09's padding id

**Decision.** Id 10,000 follows every document. The model's vocabulary is 10,001, as in exercise 09.

**Why.** Exercise 02's BPE has no special tokens. Exercise 09 added id 10,000 as padding; nothing in
this corpus is padded, so reusing the id keeps the vocabulary, and every model's output head, the
same size as 09's. The loss here does not ignore it: predicting where a document ends is part of the
task.

**What would overturn it.** A later exercise needing padding and a separator in the same batch.

## D4 · The model is this exercise's own, not exercise 09's

**Decision.** `optimizers.model.GPT`: pre-norm blocks, learned positions, causal attention through
PyTorch's fused kernel, a GELU MLP four times as wide, an untied output head.

**Why.** muP needs a learning rate per layer type and an output head initialised separately from the
input embedding; exercise 09's model trains under one parameter group and its block classes are
defined inside factory functions, so they cannot be subclassed or extended. Exercises 13 and 14 need
each block's residual update as a function (`Block.delta`) and a replaceable MLP. At width 256 and
depth 4 the parameter count is the same family as 09's model.

**What would overturn it.** 09's model gaining parameter groups and a public block class.

## D5 · muP from the paper's Table 3, with `base_width` as the multiplier

**Decision.** For Adam, per Yang et al., *Tensor Programs V* (arXiv:2203.03466v2), Table 3:

- embeddings, biases and norm gains: initial variance 1/fan_in, learning rate η, in both SP and muP;
- hidden matrices: initial variance 1/fan_in in both; learning rate η in SP, η/fan_in in muP;
- the output head: initial variance 1/fan_in in SP, 1/fan_in² in muP; learning rate η in SP, η/fan_in in muP.

The paper allows a tunable multiplier in front of fan_in; this implementation uses the sweep's
smallest width, so muP equals SP exactly at that width (a test asserts bit-identical heads there).
The table was read from the downloaded paper, not recalled.

**Why.** Multiplying by the base width makes "the best learning rate at width 256" mean the same thing
in both parametrizations, so the comparison is only about how the minimum moves.

**What would overturn it.** A comparison against the authors' `mup` package on this architecture
showing a different per-layer rule; the README states this has not been done.

## D6 · The attention scale stays at 1/√d_head

**Decision.** muP's attention rule (1/d_head instead of 1/√d_head) is not applied.

**Why.** The head size is fixed at 64; width grows by adding heads. The rule's purpose is to keep
attention logits stable as d_head grows, and here d_head does not grow, so the two choices differ by
a constant factor that the learning-rate sweep absorbs.

**What would overturn it.** Scaling the head size with width.

## D7 · Bias correction is ablated with our own AdamW, held to PyTorch's

**Decision.** `ablation.SwitchableAdamW` implements PyTorch's AdamW update with a `bias_correction`
flag. With the flag on, tests require it to match `torch.optim.AdamW` to floating-point rounding, with
and without weight decay; with it off, a twin test requires it to differ.

**Why.** PyTorch's optimiser does not expose the correction. An ablation that changed anything else
— the epsilon placement, the decay order — would measure that instead.

**What would overturn it.** PyTorch exposing the flag; the class would then be deleted in favour of it.

## D8 · "Stops mattering" is decided against seed noise, not by eye

**Decision.** Two definitions, both reported. In closed form: the first step after which the
uncorrected step stays within 10%, 5% or 1% of the corrected one. On the real model: the first step
after which the smoothed loss gap between corrected and uncorrected runs stays at or below the
smoothed gap between two seeds of the corrected run.

**Why.** A threshold chosen after seeing the curve finds whatever it was chosen to find. Seed noise is
the natural yardstick for "matters": a difference two identical runs also produce is not an effect.

**What would overturn it.** More seeds showing the two-seed gap is a poor estimate of the noise.

## D9 · "Warmup stops changing the ratio" is measured as settling, per layer

**Decision.** The first step after which a layer's smoothed ratio stays within ±10% of the median of
its last steps (`ratios.settles_at`), in a run whose rate is constant after warmup. A layer whose
curve is still moving at the end is reported as never settling.

**Why.** "The end of warmup" is a fact about the schedule; when each layer's ratio stops moving is a
fact about the model, and the two need not coincide. Holding the rate constant after warmup means
nothing but warmup moves the ratio early on.

**What would overturn it.** A different band changing the ordering of layers, which would mean the
measure is too sensitive to its parameter.

## D10 · Each schedule is tuned before cosine and WSD are compared

**Decision.** Every peak in `schedule_peaks`, for both schedules and both seeds, trained to the stop
point; each schedule then runs at its own best peak. The comparison also includes two finished
models at the stop point's budget: a decay branched from WSD's checkpoint 30 steps *before* the stop
point, so it ends there, and a cosine planned for the stop point from the start, tuned over the same
peaks.

**Why.** A schedule comparison run at one shared learning rate measures which schedule suits that rate.
And stopping both at step 200 compares two unfinished models; the branch and the planned cosine are
the two ways to actually hold a finished model at that budget, which is the question a practitioner
faces. Both see exactly the stop point's steps of data. A branch taken *at* the stop point and
decayed after it would train longer than the model it is compared with — the first version did
that, and borrowed cosine's 300-step peak for the planned cosine; both were caught reading the code
against its published claim, and fixed before publishing.

**What would overturn it.** The best peak landing at the edge of the grid for either schedule, which
would mean the grid was too narrow — the document prints the whole tuning table so that is visible.

## D11 · Weight decay is off in every experiment

**Decision.** `weight_decay = 0` throughout.

**Why.** With decoupled decay the shrink per step is η·λ, so in muP — where hidden and output rates
scale with width — the decay strength would also change with width, adding a second effect to a sweep
meant to measure one. Off everywhere keeps every experiment about the thing it names.

**What would overturn it.** A sweep that varies λ jointly with η, which is a different experiment.

## D12 · The notebook builder is staged until PK installs it

**Decision.** `artifacts/staged/build_notebook.py` holds the builder; PK copies it to
`tools/build_notebook.py`.

**Why.** The repository's guard forbids agents from writing any `tools/build_notebook.py`, because the
file is gitignored and has no second copy. The staged copy builds and executes the same notebook.

**What would overturn it.** The guard gaining a per-file opening for this path.
