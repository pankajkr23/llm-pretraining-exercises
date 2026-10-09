# CLAUDE.md — 13-reversibility

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, the measured numbers are `RESULTS.md` (generated), and `REQUIREMENTS.md`
is the requirements (local only, gitignored).

**Status: built.** Reversible rules, the rebuilding autograd function, memory measurement and the
four experiments; published by `tools/run_experiments.py` into `results/*.json`.

## Modules

`config.py` (presets) · `stack.py` (the rules, the rebuilding autograd function, the plain-autograd
reference, rebuild error) · `model.py` (`ChainedGPT`, exercise 11's model chained by a rule;
`rebuild_agreement`) · `memory.py` (chunked loss, saved bytes, largest-batch search) · `train.py`
(one run to a token budget) · `experiments.py` (trials, fixed batch, largest batch, its run) ·
`runs.py` (provenance over 13 and 11). Tools: `run_experiments.py`, `render_results.py`.

## The rules this exercise adds

- **The rebuild must match storing, or nothing else matters.** `test_reversible_stack.py` holds the
  memory-saving backward pass to plain autograd through the same recurrence, for every rule, in
  float64. If it fails, every memory and speed number is a number about wrong gradients.
- **float64 agreement is not float32 agreement.** The blend's inversion divides by `a` at every
  layer, so its float32 gradients at depth 12 are a few percent off stored ones (D10). The run
  records `rebuild_agreement`; read it before trusting a run that chose the blend.
- **Choose on one half of the validation split, report on the other** (`experiments.validation_half`).
- **`saved_bytes` cannot see a tensor kept on `ctx` or allocated during backward.** The tests
  cross-check it with a liveness instrument; the reversible backward's re-run block is measured
  separately and added to the derived largest batch (D4).
- **Never put randomness in a block.** The backward pass re-runs each block; a second dropout mask
  rebuilds the wrong state and the gradients silently go wrong.
- **A rule needs `A ≠ 0`.** The blend at `a = 0` is forward Euler and cannot be inverted;
  `Rule.make` refuses it.
- **The memory cap outlives the search.** `memory.cap` lasts for the process's life, which is why the
  largest-batch search and its run are a separate stage, in a separate process.
- **The loss is part of the memory budget.** Every variant uses `memory.chunked_loss`; replacing it
  with a plain cross-entropy makes the logits, not the stack, set the largest batch.
- **Exercise 09's `chunked_projection_cross_entropy` does not save memory under autograd** (measured:
  the same as the full loss). Do not reuse it for memory; it is recorded for 09.
- **MPS is hidden inside the sandbox**, and so is its memory cap; publish from outside it.

## Running it

```bash
uv sync --all-packages --extra train
uv run pytest src/exercises/13-reversibility
uv run python src/exercises/13-reversibility/tools/run_experiments.py     # needs exercise 11's corpus
uv run python src/exercises/13-reversibility/tools/render_results.py
```

Test modules are prefixed `test_reversible_*`. Files that need torch import it with a module-level
`importorskip` and are registered in CI's `train` job and in
`tests/test_ci_shards_cover_everything.py::OPTIONAL_DEPENDENCY_GATES`.
