# DECISIONS — 10-training-loop

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · This exercise's notebook IS tracked, and that is a written exception

**Decision.** `notebooks/S10-training-loop.ipynb` is committed, against the repo-wide rule that
gitignores every topic notebook.

**Why.** The submission wording says the ipynb "should also be there", with no alternative offered.
Exercise 09's offered *"the ipynb file **or** training logs"*, and that alternative is why 09's
notebook stays local. There is nothing to fall back to here.

The convention exists to keep the course's own material off a public remote. A notebook of our
training loop is our work: it imports this package and re-implements nothing, so it carries the
course's material no more than the package does.

**The exception is written where the rule is, or it is not a decision.** `AGENTS.md` carries it
alongside the rule it excepts, and `.gitignore` carries a negation naming the one path — which works
because the pattern it excepts is a *file* pattern rather than a directory one.

`tools/backup_local_only.py` needs **no** change, and checking that rather than assuming it is the
point: `collect` ends with `found -= _tracked(root)`, so a tracked file is dropped from the backup
set automatically. An earlier draft of this decision said `PATTERNS` had to be edited. It does not,
and editing it would have been a change with no effect that read as a safeguard.

**What would overturn it.** A grader accepting a rendered export, or the repository going private.

---

## D2 · The micro-batch sizes are uneven by decision, and the comparison refuses even ones

**Decision.** `Config.micro_batch_tokens` is `(4, 4, 2)`, `micro_batches_are_uneven` exists to
assert it, and `accumulation.compare` raises `EvenMicroBatchesError` rather than returning a gap of
zero.

**Why.** Averaging the averages is **exactly correct** when every micro-batch holds the same number
of real tokens. So a demonstration built on even micro-batches reports zero difference and reads as
a clean bill of health — which is precisely how this bug survived inside every major framework until
2024. A zero here must be an error, never a result.

**What would overturn it.** Nothing. This is the single property the item depends on.

---

## D3 · The gradient check runs in float64, and on the largest gradient in the head

**Decision.** The trunk and head are cast to double, and the element under test is
`argmax(|grad|)`.

**Why.** Both were found by the first version failing. A central difference subtracts two losses
that differ by roughly `epsilon x gradient`; in fp32 a loss near 9.2 resolves to about `5e-7`, so a
gradient small enough to move the loss by less than that gives a numeric estimate of **exactly
zero** — and a relative error of 1.0 at every epsilon, which looks like a broken implementation
rather than a broken instrument. Choosing element `[0, 0]` compounded it: that weight's gradient was
`-7.8e-7`, so the comparison was between two numbers that were both, to fp32, nothing.

**What would overturn it.** Nothing about float64. The choice of the largest gradient is worth
revisiting only to add a *second* element with a typical gradient, which would make the point about
the numerical floor directly rather than by anecdote.

---

## D4 · The device peak is measured, not quoted

**Decision.** `mfu.measured_peak_flops` times a 2048³ dense matrix multiply on the same device and
dtype as the run, and that is MFU's denominator.

**Why.** The first version used a configured figure for an Apple GPU while the run executed on the
**CPU**, and reported **39.13%** — a number that looked excellent and divided one processor's
achievement by another's capability. A vendor peak also usually describes a sparse or low-precision
mode the run never touches. A measured GEMM peak is lower, so it reports a *worse* MFU, which is the
honest direction to be wrong in.

**What would overturn it.** Running on an accelerator, where the same function should be pointed at
that device rather than replaced with a datasheet number.

---

## D5 · MFU is priced from non-embedding parameters

**Decision.** `6N` uses `N = parameters - embedding parameters`.

**Why.** An embedding lookup is a gather: it reads one row per token and performs no arithmetic. The
token and position tables here are 2,593,024 parameters, 31% of the total, and counting them
made the numerator 45% larger than it should have been, for free. Every published MFU uses the non-embedding count, so mixing
conventions would also make any comparison meaningless.

**What would overturn it.** A tied output head, where the same matrix is both an embedding and a
matmul — then the convention needs stating explicitly rather than inheriting.

---

## D6 · The gradient norm is logged before clipping

**Decision.** `step.global_grad_norm` runs before `clip_grad_norm_`, and the trace records the
pre-clip value.

**Why.** A trace of the post-clip norm flattens at the clip value. That hides exactly the spikes the
trace exists to reveal, and it does so while looking like a stable run.

**What would overturn it.** Nothing. Logging both would be strictly better and costs one field.

---

## D7 · The model is exercise 09's, imported

**Decision.** `config.Config` holds a `lossheads.Config`; the trunk, tokenizer, shift, masks and
losses are imported.

**Why.** The two exercises then cannot disagree about what a loss is, and the corrections exercise
09 earned — the boundary mask, the contributing-count denominator — apply here automatically.

**What would overturn it.** Nothing foreseeable. The cost is a workspace dependency between two
exercise packages, which the repository already does elsewhere.
