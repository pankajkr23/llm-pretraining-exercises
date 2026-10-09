# CLAUDE.md — 14-moe

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, the measured numbers are `RESULTS.md` (generated), the training log is
`submission_artifacts/run.log`, and `REQUIREMENTS.md` is the requirements (local only, gitignored).

**Status: built.** Upcycling, routing, balancing and the three experiments.

## Modules

`config.py` (presets) · `layer.py` (`MoE`, `upcycle`, balance statistics, parameter counts) ·
`train.py` (one run to a token budget, with the log) · `experiments.py` (dense start, continuity,
router trial, continuation) · `runs.py` (provenance, with the dense checkpoint's digest). Tools:
`run_experiments.py`, `render_results.py`.

## The rules this exercise adds

- **Continuity is the precondition for every other claim.** If the converted model does not start at
  the dense model's loss, "it kept training" is measured from a different place.
  `test_moe_layer.py` holds it to 1e-12 in float64; keep it that strict.
- **Renormalise the top-k weights in the model's dtype.** Normalising in float32 and casting leaves
  them summing to one only to ~1e-7 — caught by the float64 test.
- **The bias chooses; it never weights.** It is added to scores only for the top-k selection. If it
  leaked into the weights, balancing would change the function the model computes.
- **FULL depends on exercise 13's checkpoint.** `dense_model` refuses with the command to run if it is
  missing, and loads it with `weights_only=True`.
- **The log is written as the run goes**, so a crashed run still leaves its record.
- **Choose on one half of the validation split, report on the other.** `experiments._windows`:
  the router trial selects on `select`, everything published uses `report`. Selecting on the
  reported windows would flatter the MoE, the only arm the choice is applied to.
- **Read exercise 13's numbers; never type them.** The learning rate (`dense_peak`) and the token
  count (`dense_tokens_trained`) come from its results, and their digests go in the provenance.

## Running it

```bash
uv sync --all-packages --extra train
uv run pytest src/exercises/14-moe
uv run python src/exercises/14-moe/tools/run_experiments.py      # after exercise 13
uv run python src/exercises/14-moe/tools/render_results.py
```

Test modules are prefixed `test_moe_*`. Torch-gated files are registered in CI's `train` job and in
`tests/test_ci_shards_cover_everything.py::OPTIONAL_DEPENDENCY_GATES`.
