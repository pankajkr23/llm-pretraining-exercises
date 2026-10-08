# CLAUDE.md — 11-optimizers-lr-schedules

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, the measured numbers are `RESULTS.md` (generated), and `REQUIREMENTS.md`
is the requirements (local only, gitignored).

**Status: built and measured.** Five experiments in `optimizers.experiments`, published by
`tools/run_experiments.py` into `results/*.json` and rendered into `RESULTS.md`.

## Modules

`config.py` (presets `FULL` / `LITE` / `SMOKE`) · `corpus.py` (the FineWeb-Edu token files) ·
`data.py` (seeded windows) · `adam.py` (Adam by hand, read back from PyTorch, the bias gap in closed
form) · `ablation.py` (`SwitchableAdamW`) · `schedules.py` (warmup, cosine, WSD, branch) ·
`model.py` (the decoder in SP or muP, parameter groups) · `ratios.py` (update-to-weight ratio and
settling) · `train.py` (the one loop) · `sweep.py` (minima, power law, prediction) · `experiments.py`
(the five experiments) · `runs.py` (provenance and a refusing `save`). Tools: `fetch_corpus.py`,
`run_experiments.py`, `render_results.py`.

## The rules this exercise adds

- **The corpus is shared, so its format is a contract.** Exercises 13 and 14 read
  `data/fineweb-edu/{train,val}.bin` through `optimizers.corpus`. Changing the separator id, the dtype
  or the split order invalidates their results too; the manifest records all three.
- **Fetch outside the sandbox.** Its egress proxy truncates large responses, and the fetcher only ever
  contacts `huggingface.co` and `datasets-server.huggingface.co` (a test refuses every other host).
  The dataset's licence is read from its card before any row is fetched.
- **Every number in `RESULTS.md` is rendered, including the ones in sentences.** Re-render after any
  results change; `test_optimizers_results.py` fails on a stale document. Findings sentences are
  conditional on the data — "within the noise" is printed when it is.
- **An ablation must change one thing.** `SwitchableAdamW` is held to `torch.optim.AdamW` with the
  flag on; if that test fails, every bias-correction number is suspect.
- **Copy a model and its optimiser in one `deepcopy`.** Separate copies leave the optimiser updating
  parameters the copied model does not hold, and a branch silently never trains. A test catches it.
- **The MPS backend is hidden inside the sandbox.** A run that reports `cpu` when an M-series GPU is
  present was started inside it; publish from outside.
- **muP's rules are the paper's Table 3, read from the downloaded paper** (arXiv:2203.03466v2) —
  never from memory. `DECISIONS.md` D5 records them and the one rule deliberately not applied.

## Running it

```bash
uv sync --all-packages --extra train
uv run pytest src/exercises/11-optimizers-lr-schedules
uv run python src/exercises/11-optimizers-lr-schedules/tools/fetch_corpus.py        # once
uv run python src/exercises/11-optimizers-lr-schedules/tools/run_experiments.py     # FULL
uv run python src/exercises/11-optimizers-lr-schedules/tools/render_results.py
```

Test modules are prefixed `test_optimizers_*`. pytest imports by **basename**, so a second
`test_config.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide. Files that need torch import it with a
module-level `importorskip` and are registered in CI's `train` job and in
`tests/test_ci_shards_cover_everything.py::OPTIONAL_DEPENDENCY_GATES`.
