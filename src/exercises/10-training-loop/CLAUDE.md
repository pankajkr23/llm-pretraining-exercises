# CLAUDE.md — 10-training-loop

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, the measured evidence is `RESULTS.md` (generated), and the
requirements are `REQUIREMENTS.md` (local only, gitignored).

**Status: all six items run and write `results/run.json`.** Remaining: the notebook and the page.

## The rules this exercise added, each learned by getting it wrong

- **Ask what the denominator is, twice.** Two numbers here were wrong in the flattering direction
  and both were denominators. MFU divided FLOPs achieved on the **CPU** by a **GPU's** advertised
  peak — 39.13%, and meaningless. It also counted the **embedding tables**, which are read by a
  gather and do no arithmetic, inflating the numerator by 45%. The honest figure is 27.89%.

- **A numerical instrument has a floor, and hitting it looks like a broken implementation.** The
  gradient check in fp32 reported a central difference of **exactly 0.0** at every nudge size, for a
  weight whose gradient was `-7.8e-7`: a loss near 9.2 resolves to about `5e-7`, so the two
  perturbed losses were bit-identical. float64, and the largest-magnitude gradient in the head.

- **A demonstration must be able to fail before it can succeed.** Averaging the averages is exactly
  correct when micro-batches hold equal token counts, so an even configuration reports a gap of zero
  — which says the experiment was blind, not that the code is right. `compare()` raises rather than
  returning that zero.

- **Build the bit pattern, then check it against the machine.** `floats.py` derives fp32, bf16 and
  fp8 E4M3 from field widths and rounding; the gated twin asserts each against `torch`'s own cast.
  A decomposition that agrees only with itself proves nothing. `FP8_E4M3.largest_normal` deriving to
  448.0 from `has_infinity=False` is the load-bearing case.

- **`textwrap.fill` for any printed paragraph.** Hand-splitting long f-strings to satisfy the line
  limit produced `"exactly    the same vote"` and `"the    embedding tables"` in shipped output
  twice. The line limit is about source, not about what a reader sees.

## Running it

```bash
uv sync --all-packages --extra train

uv run python -m trainloop.harness      # all six items -> results/run.json
uv run python tools/render_results.py   # RESULTS.md, from that file

uv run pytest src/exercises/10-training-loop
uv run pytest                           # and the repo-wide guards, which the line above misses
```

Test modules are prefixed `test_trainloop_*`. pytest imports by **basename**, so a second
`test_config.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.

## Modules

`config.py` · `step.py` · `gradcheck.py` · `accumulation.py` · `telemetry.py` · `mfu.py` ·
`floats.py` · `harness.py`, plus `tools/render_results.py`.

The model itself is **exercise 09's**, imported: its trunk, tokenizer, target shift, masks and
losses all apply here unchanged, so the two exercises cannot disagree about what a loss is.

`accumulation.accumulate` is offered and unused — the two-curve run inlines its own loop because it
needs the optimiser step between reductions. Say so rather than leaving a tested function that
nothing calls.
