# DECISIONS — 14-moe

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · The model converted is a dense one: exercise 13's baseline

**Decision.** The dense model is exercise 13's standard-residual baseline: exercise 11's architecture,
width 320, depth 12, trained on exercise 13's 50-million-token budget. `FULL` loads it; `LITE` and `SMOKE` train their own
small dense model first so they run anywhere.

**Why.** The model to convert is one whose every token uses the same feed-forward layer — a dense
model. Reusing a model that has already been trained, evaluated and given provenance means the
conversion starts from something known rather than from a model trained only to be converted.

**What would overturn it.** A starting point other than a dense transformer — a model whose
feed-forward layers are already sparse would make "conversion" mean something else.

## D2 · Experts are exact copies; the top-k weights are renormalised to one

**Decision.** Copy-upcycling: every expert is the trained MLP's weights, unchanged. The chosen scores
are divided by their sum. Renormalisation happens after casting to the model's dtype.

**Why.** It makes the converted model compute exactly what the dense one did, so "it kept training" is
measured from the same starting loss. Partitioning the MLP into slices, or perturbing copies, would
start the MoE from a different function and confound the comparison. Normalising in float32 and then
casting left the weights summing to one only to about 1e-7, which a float64 test caught.

**What would overturn it.** Evidence that identical experts fail to diverge — dead experts that
never recover — which the log's dead-expert count would show, and which perturbed or partial copies
are the usual remedy for.

## D3 · Eight experts, two per token

**Decision.** `n_experts = 8`, `top_k = 2`.

**Why.** Small enough that the MoE's total parameters stay within a few times the dense model's on
a laptop, large enough that routing has real choices. Two per token keeps every token's output a
blend rather than a single expert's, which is the common setting and halves the compute cost against
four.

**What would overturn it.** A sweep over the expert count, which this exercise does not run.

## D4 · Bias-only load balancing, no auxiliary loss

**Decision.** Per-expert selection bias, `b_i += γ·sign(mean load − load_i)` after every step,
γ = 0.001; no auxiliary loss term. The bias affects which experts are chosen, never their weights.

**Why.** An auxiliary loss pulls against the language-modelling loss; the bias steers routing without
touching the objective (Wang et al., arXiv:2408.15664, read on arXiv). Balancing is computed over the
whole step's batch.

**What would overturn it.** The log showing dead experts that the bias does not revive.

## D5 · The router is float32 and starts at a tenth of the usual scale

**Decision.** Router logits computed in float32; router weights initialised at `0.1 / √width`.

**Why.** Routing decisions are discrete, so small numerical differences flip them; float32 keeps them
stable whatever the model's dtype. A small initial router makes early choices close to uniform rather
than confident and arbitrary.

**What would overturn it.** Nothing within this exercise; it is a safety setting.

## D6 · Softmax against sigmoid is decided by a short trial

**Decision.** One short continuation each, same data; the better validation loss is used.

**Why.** Both scoring functions are in use, and we had no reason to prefer one. A short trial
chooses on evidence without spending the main budget on it. It chooses on the first half of the
validation split, and every published figure is measured on the second, so the choice cannot flatter
the MoE — the only arm it is applied to.

**What would overturn it.** A longer comparison reversing the choice.

## D7 · The continuation is re-warmed, and the dense model continues on the same tokens

**Decision.** Both continuations: warmup to `continue_lr`, then cosine to 10% of it, on the same batch
stream. The continuation's peak is `continue_lr_fraction` (one half) of the peak the dense model
trained at — exercise 13's trial-chosen rate, read from its results.

**Why.** The dense model ended its training at the bottom of a cosine schedule; continuing at that
rate would barely move it, and jumping straight to a high rate would disturb a converged model. The
dense continuation with the identical schedule is the control.

**What would overturn it.** A sweep showing a different rate changes which model is ahead.

## D8 · The training log is plain text in `submission_artifacts/run.log`

**Decision.** One line per `log_every` steps, written as the run goes; the path is re-included by the
repository's `.gitignore`.

**Why.** The exercise requires training logs in the repository; plain text can be read without any
code, and writing as the run goes means a crash still leaves a log.

**What would overturn it.** Nothing here.

## D9 · The notebook builder is staged until PK installs it

**Decision.** `artifacts/staged/build_notebook.py`; PK copies it to `tools/build_notebook.py`.

**Why.** The guard forbids agents from writing any `tools/build_notebook.py`.

**What would overturn it.** The guard gaining a per-file opening for this path.
