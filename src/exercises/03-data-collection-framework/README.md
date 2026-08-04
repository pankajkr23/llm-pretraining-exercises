# 03 · Data Collection & Sourcing — a decision framework

A reusable framework for **deciding the pre-training data mix** of an India-first LLM: grade every
candidate source through five questions, budget the tokens, and measure the tokenizer tax — then
present the decision as a self-justifying site + a printable report. See [`BRIEF.md`](./BRIEF.md) for
the assignment and [`docs/`](./docs/) for the full spec.

## Layout

```text
docs/                 # ★ the spec + research — start at docs/README.md
  ATLAS.md            #   the source research (the India LLM Data Atlas)
  FRAMEWORK.md        #   the method: 5 questions, 3 mix rules, 8 intuitions
  DECISIONS.md        #   the resolved answers (mix, cleaning, tests, V = 208,896)
  DESIGN.md           #   the build spec for the 3-page site + print
  FERTILITY_MEASUREMENT.md
  TODO.md    ◇        #   execution order, phases 0–8
  OPEN.md             #   decisions deliberately left open
data/seed/ ◇          # the data spine: 145-dataset catalog + 31 benchmarks (CSV)
src/dataframework/    # the Python pipeline (config.py + modules to come)
tests/                # fast, offline unit tests (+ the CI invariants, later)
web/                  # the 3-page static site + precomputed data.json (later)
```

◇ **Local working files — not in git.** `docs/TODO.md` and `data/seed/*.csv` are untracked by design
(the root `.gitignore` excludes `TODO.md` and `data/`), so keep a backup outside the repo: a fresh
clone won't have them, and the pipeline can't rebuild without the seed CSVs. `artifacts/` (generated
outputs) is git-ignored too. What *does* ship is the exported `web/data.json` — the static site reads
only that, so the deployed page works from a clone regardless.

## Run it

```bash
uv sync --all-packages                        # install this member into the shared venv
uv run python -m dataframework.ingest         # seed CSVs -> catalog.json + benchmarks.json
uv run python -m dataframework.catalog --validate   # 355 records, or a loud failure
uv run python -m dataframework                # compute + write web/data.json
```

## Preview it

Preview the **built bundle**, not `web/` directly — the pages fetch `catalog.json`, which
`build.sh` places alongside them rather than duplicating into `web/`:

```bash
bash ../../../deploy/vercel/build.sh          # assemble public/
cd ../../../public/03-data-collection-framework
python3 -m http.server 8000                   # open http://localhost:8000
```

## Tests

```bash
uv run pytest -m "not integration"   # fast unit tests
```

## Status

**Scaffold + reconciled spec.** The docs are the agreed source of truth; the seed spine is in place.
The pipeline, the five CI invariants, and the site are the phased build in
[`docs/TODO.md`](./docs/TODO.md).
