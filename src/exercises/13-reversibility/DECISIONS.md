# DECISIONS — 13-reversibility

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · The rules are the paper's equations 4, 6 and 15, read from the paper

**Decision.** Midpoint (eq. 4), leapfrog (eq. 6) and the blend, the paper's "midpoint (a)" (eq. 15),
from Gal et al., arXiv:2512.02056v1, read from the downloaded paper's own LaTeX. All three are
implemented as one form, `A·p⁽ℓ⁻¹⁾ + B·p⁽ℓ⁾ + C·f(p⁽ℓ⁾)`, so one autograd function serves them all.

**Why.** We needed precise update rules, and the paper is where they are written down as equations.
We take forward Euler to be the ordinary residual stack itself (the paper's eq. 26), so it is the
baseline here rather than a fourth reversible rule; the blend stands nearest to it, since the paper
shows it behaves like forward Euler in expectation (eq. 18) and it becomes forward Euler at `a = 0`.
The Hamiltonian two-stream form (eq. 8) is left out: it carries a second state through the whole
stack and needs a different block, which makes it a different architecture rather than another
rule for this one.

**What would overturn it.** A reversible rule that is a better reading of "Euler" than the
baseline, which would be added as a fourth candidate in the trials.

## D2 · The stack starts from two copies of the embedding output

**Decision.** `p⁽⁻¹⁾ = p⁽⁰⁾`, so the first layer takes an ordinary step.

**Why.** A two-step rule needs two starting states and the paper's text does not say how its first
step is seeded. Duplicating the embedding is the simplest choice that adds no parameters; its
gradient is the sum of both paths, which a test checks.

**What would overturn it.** The paper's code or text specifying another seed.

## D3 · Every variant, the baseline included, uses a chunked loss recomputed in backward

**Decision.** `memory.chunked_loss`: the output projection and cross-entropy computed a chunk of rows
at a time under `torch.utils.checkpoint`.

**Why.** At large batch the logits (`tokens × 10,001` floats) outgrow the stack's activations, and
reversibility cannot shrink them: at 32,768 rows, 32,768 × 10,001 float32 logits are 1,250 MiB,
which a plain cross-entropy keeps for its backward pass. The checkpointed loss keeps none of them —
`test_reversible_stack.py` checks this by which storages are still alive after the forward pass, an
instrument independent of the saved-tensor hooks. Without it the head would cap every variant's
batch and the experiment would measure the loss.

Measuring this also showed that exercise 09's `chunked_projection_cross_entropy` keeps a full set of
logits under autograd (the 1,250 MiB above, at that size), although its docstring says only one
chunk's logits ever exist. That is recorded for exercise 09 and not changed here.

**What would overturn it.** A fused cross-entropy kernel that keeps no logits, which would make this
unnecessary.

## D4 · Memory is reported as bytes kept for backward, plus a measured largest batch

**Decision.** Saved-tensor hooks count the bytes a forward pass keeps for backward, on any device;
the largest batch is found by running real steps under a GPU memory cap. The derived largest batch
adds, for the reversible model, the bytes one block keeps while it is re-run during the backward
pass, measured the same way.

**Why.** Apple's MPS keeps no peak-memory counter, and a sampled allocator figure is not a peak. The
saved bytes are comparable across variants and devices; the largest batch is the quantity the
exercise asks about, and running real steps is the only way to know it fits.

**What the hooks can and cannot see.** They see everything an operation or a custom `Function`'s
`save_for_backward` keeps, and torch 2.13's non-reentrant checkpoint routes its inputs through them
too. They cannot see a tensor kept as a `ctx` attribute or in a closure, nor anything allocated
during backward. The first is closed by a second instrument in the tests — the bytes of storages
still alive after the forward pass, which agree with the hooks per sequence for every variant at
two depths — and by a test that the reversible `Function` keeps no tensor on `ctx`. The second
matters for the reversible model, whose backward pass re-runs one block with autograd on: at the
test shape that block's working set is about five times what the whole reversible forward pass
keeps per sequence, so leaving it out would have overstated the derived batch several-fold. It is
now measured and added. Gradient buffers, allocator slack and kernel workspaces remain uncounted for
both variants, and `RESULTS.md` says so beside the figure.

**What would overturn it.** MPS gaining a peak counter, which would add a third, simpler number.

## D5 · An 8 GiB budget for the largest-batch search

**Decision.** `torch.mps.set_per_process_memory_fraction` set so the process may use 8 GiB.

**Why.** The GPU allows this process several times that (the backend's own figure,
`torch.mps.recommended_max_memory()`, is recorded in `max_batch.json` as `device_limit_bytes`);
searching to it would be slow, would compete with everything else running, and would make the answer
depend on the machine. A fixed cap makes it reproducible and comparable. The cap lasts for the process's life, so the search and the run at its
result run in a separate process.

**What would overturn it.** A comparison at a second budget showing the ratio between variants
changes with the budget.

## D6 · The variant and step size are chosen by short trials at one learning rate

**Decision.** The baseline's learning rate is chosen from a small grid by validation loss after a
short run; every rule × `h` then trains for the same short budget at that rate; the best is used for
both long runs. Choices are scored on the first half of the validation split; every loss reported
is measured on the second.

**Why.** The exercise asks which variant worked. Choosing on equal footing before the long runs
keeps the long runs honest — neither is tuned after seeing its own result. Scoring the choice on
the windows the result is later reported on would flatter the arm the choice was applied to, which
here is the reversible one. A choice at the end of its grid is flagged in the bundle and in
`RESULTS.md`: it is a best-of-grid, not an optimum.

**What would overturn it.** A rule that wins the short trial and diverges in a long run, which the
long run's loss curve would show.

## D7 · The run at the largest batch re-checks the learning rate

**Decision.** Several multiples of the fixed-batch rate, each trained for a fixed number of steps
(`max_batch_check_steps`: 40 at FULL) at the largest batch and scored on the selection half; the
best is used for the full run. A pick at either end of the multiplier grid is flagged.

**Why.** A batch several times larger takes several times fewer steps for the same tokens; at the
same rate it would under-train and the comparison would blame reversibility for it. The check is
counted in steps because a token budget sized for the fixed batch is a handful of steps at a large
one — about ten at a batch of 900 — which cannot tell one rate from another. Forty steps at a large
batch is a sizeable share of the full run's own step count; that cost is accepted.

**What would overturn it.** A pick at the edge of the grid, which says the grid was too narrow.
It is a short check, not a tuned schedule, which the README states.

**Not compared.** The reversible model at its largest batch is compared with the baseline at the
fixed batch only, not with the baseline at the baseline's own largest batch. That third run would
cost a further full 50-million-token run and is left out; `RESULTS.md` says which comparison it
makes.

## D8 · No weight decay and no dropout

**Decision.** AdamW with weight decay 0; the blocks have no dropout.

**Why.** Rebuilding in backward re-runs every block, so the forward pass must be deterministic: a
dropout mask drawn twice would rebuild the wrong state. Weight decay is off for the same reason as in
exercise 11 — one effect per experiment — and so the baseline and the reversible runs are optimised
identically.

**What would overturn it.** Nothing within this exercise.

## D9 · Agents edit a staged copy of the notebook builder; PK installs it

**Decision.** Agents write the builder at `artifacts/staged/build_notebook.py`; PK copies it into
`tools/build_notebook.py`, where the installed builder lives.

**Why.** The guard forbids agents from writing any `tools/build_notebook.py`. The exercise also
requires the notebook in the repository, which needs a `.gitignore` exception like exercise 10's —
a standards change that is PK's to approve.

**What would overturn it.** PK approving both.

## D10 · Rebuilt gradients are checked in float32 at the run's own scale, and the blend's are not exact

**Decision.** Besides the float64 tests, the run measures `model.rebuild_agreement` — the worst
relative error of a rebuilt state and the relative distance between rebuilt and stored gradients —
for every trial candidate at initialisation and for the chosen rule before and after its full run,
in the run's own dtype, depth and device. A candidate whose gradient error at initialisation
exceeds `gradient_tolerance` (1%) is **ineligible**: it is still trained and shown, but never
chosen.

**Why.** Inverting a rule multiplies rounding error at every layer, and the float64 tests cannot
show it. At FULL's width and depth in float32 on a CPU, midpoint and leapfrog gradients stay within
a few tenths of a percent of stored ones at `h ≤ 1`; the blend at `a = 0.5` does not. Its inversion
divides by `a` at every layer, so the error roughly doubles per layer: about 6e-5 at depth 4, 1e-3
at 8 and 3e-2 at 12 (h = 0.5), against 1e-10 for the same model in float64. A run that chooses the
blend would therefore train on gradients a few percent away from the true ones. That might still
train well, but it is not the exact backward pass the method promises, and a published run that
claims "same gradients, less memory" must actually have the same gradients. So the gate is on the
measured error, not on the rule's name: a blend that rebuilds within tolerance at some `h` stays
eligible, and `RESULTS.md` still says so whenever the chosen rule's error exceeds 0.1%. The
tolerance is a judgement — tight enough that the chosen run trains by the gradients it reports,
loose enough that rounding at depth 12 in float32 does not exclude every rule.

**What would overturn it.** A blend weight closer to 1, or float64 accumulation in the inversion,
bringing the measured error below 0.1%; a test pins the current error above that line so this entry
cannot outlive its reason.
