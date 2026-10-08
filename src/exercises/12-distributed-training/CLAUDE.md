# CLAUDE.md — 12-distributed-training

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, the measured evidence is `RESULTS.md` (generated), and
`REQUIREMENTS.md` is the requirements (local only, gitignored).

**Status: built, awaiting review.** The simulator, all four stages, the measurements, the producer,
the renderer and the tests are done; the topic notebook is built and executed locally. Nothing is
deployed — this exercise has no `web/`.

## The rules this exercise adds, each learned by getting it wrong

- **"Elementwise" does not mean "slicing-invariant" on a CPU.** AdamW written with torch's fused
  kernels made ZeRO-1 disagree with data parallelism by about 1e-6, in an update that reads only
  element `i` to write element `i`. The fused kernels' vectorised loop and scalar tail round
  differently, and a shard boundary moves elements between them. `adamw.py` uses single-rounding
  operations only; `test_updating_n_shards_is_bit_identical_to_updating_the_whole` cuts at
  deliberately unaligned offsets. If you touch `adamw.py`, that test is the one to watch.

- **A gradient that is truly zero is the one AdamW amplifies most.** The attention key bias has a
  true gradient of exactly zero, so what is computed for it is rounding noise, and AdamW normalises
  noise into a full step. Against the single-device reference it is the only place weights drift
  visibly. The equivalence test excludes it *and asserts why* (its gradient is below 1e-6 of the
  largest) rather than widening a tolerance for everything.

- **Freeing a tensor while autograd holds it is not freeing it.** A ZeRO-3 rank that dropped its
  gathered weights from the ledger while keeping the forward's autograd graph would still hold them.
  Each unit's forward is therefore recomputed inside the backward from the unit's saved input, on
  every stage alike, and a weak-reference test proves the gathered buffers die. Do not "optimise"
  this by keeping the graph.

- **A twin must hold its reference from inside the event it probes.** The first twin for that
  weak-reference test grabbed the buffer *after* the step, by which time it was already dead — the
  property under test — and so it failed. It now keeps a strong reference from inside the
  intercepted all-gather.

- **A view costs its whole storage.** The ledger first charged `numel × element_size`, so a shard
  sliced out of a full buffer without `.clone()` cost 1/N in the ledger while keeping the full
  buffer alive — and every test stayed green. It now charges storage bytes, once per device.
  Do not "save a copy" by putting a view; `test_every_held_tensor_owns_exactly_its_storage` exists
  for exactly that.

- **`current == peak` does not catch a leak.** A buffer added every step raises both together.
  The no-growth test compares the held total step by step, and against the formula.

- **The committed bundle must be fresh.** `test_the_bundle_is_from_todays_code` and
  `..._config` go red after any edit to `zerosim` or `lossheads`, or to `config.py`, until
  `tools/run_zero.py` and `tools/render_results.py` are re-run. That is the intended workflow.

- **The corpus is shuffled by its total length.** `batches` calls exercise 09's `_corpus` with
  `steps × global_batch` sequences and that function permutes by count, so a one-step config and a
  four-step config start from different first batches. Any comparison between two runs must build
  both from the *same* config.

- **Padding is invisible at N = 32 with this model.** Every unit happens to divide by 32. N = 3 is in
  the measured scaling and in the ledger tests on purpose — it is where padding is non-zero.

## Running it

```bash
uv sync --all-packages --extra train

uv run python src/exercises/12-distributed-training/tools/run_zero.py        # -> results/zero.json
uv run python src/exercises/12-distributed-training/tools/render_results.py  # -> RESULTS.md

uv run pytest src/exercises/12-distributed-training
uv run pytest                       # and the repo-wide guards, which the line above misses
```

Regenerate `results/zero.json` after any change to `zerosim` or `lossheads` (the bundle's code
digest covers both), then re-render. `test_zerosim_results.py` fails if `RESULTS.md` or a figure
the README quotes has drifted from the bundle.

Test modules are prefixed `test_zerosim_*`. pytest imports by **basename**, so a second
`test_config.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide. Torch-dependent files carry a module-level
`pytest.importorskip("torch")` and must be registered in the CI `train` job. The configuration,
precision, formula, time-model, world and provenance modules import without torch, so their tests
(`world`, `formulas`, `provenance`, `results`, `docs`, `smoke`) run in the ordinary CI job.

## Modules

`config.py` · `precision.py` · `world.py` · `collectives.py` · `flat.py` · `adamw.py` · `model.py` ·
`stages.py` · `reference.py` · `formulas.py` · `timing.py` · `provenance.py` · `experiment.py`,
plus `tools/run_zero.py` and `tools/render_results.py`.

The model is **exercise 09's**, imported — its trunk, head, loss, tokenizer and corpus apply
unchanged.
