# 03 · Data Collection & Sourcing — a decision framework

A framework for **deciding the pre-training data mix** of an India-first 40B model: grade every
candidate source through five gates, budget the tokens, measure the tokenizer tax, and then try to
actually assemble the corpus from what survives. The answer is uncomfortable, which is the point —
**4 of 145 datasets are committable today**, covering 6.56T of a 16.8T budget, and the blocker is
almost never quality.

See [`docs/`](./docs/) for the full spec.

## What it produces

One interactive page — **thirteen chapters plus an appendix**, one per reader question — built from a
precomputed bundle. Every figure carries `{value, unit, provenance, source}`; the renderer refuses to
print a bare number, and where a quantity has never been measured the page says so instead of showing
something plausible.

| | |
|---|---|
| Catalogue | 145 datasets, 31 benchmarks, 429 records in the spine |
| Grades | 14 B · 116 C · 15 X — nothing reaches A, and the reason is coverage, not quality |
| Committable today | **4**, carrying 6.56T |
| Tokenizer tax | 22 languages × 5 tokenizers, measured on our own runs, not cited |
| Contamination gate | 126,044 fingerprints over 8,923 MILU items |
| Modalities / domains | 7 modalities, 16 domains — with the 4 nothing can currently supply named |
| Corrections | 30, each with a paired test proving it fails when broken |

## Layout

```text
docs/                 # ★ the spec + research — start at docs/README.md
  ATLAS.md            #   the source research (the India LLM Data Atlas)
  FRAMEWORK.md        #   the method: 5 questions, 3 mix rules, 8 intuitions
  DECISIONS.md        #   the resolved answers (mix, cleaning, tests, V = 208,896)
  DESIGN.md           #   the build spec for the site
  OPEN.md             #   decisions deliberately left open
  TODO.md    ◇        #   local planning scratchpad
  NEW_TODO.md ◇       #   local rebuild plan (delivered; kept as the record of how)
data/seed/ ◇          # the data spine: 145-dataset catalog + 31 benchmarks (CSV)
src/dataframework/    # the Python pipeline
tests/                # unit tests, the five CI invariants, and the browser rendering suite
web/                  # the static site + the precomputed bundle it reads
records/              # the reference registers (corrections, risks, priors, legal, …)
catalog.json          # per-dataset detail, served directly and loaded on demand
```

◇ **Local working files — not in git.** `docs/TODO.md`, `docs/NEW_TODO.md` and `data/seed/*.csv` are
untracked by design (the root `.gitignore` excludes `TODO.md`, `NEW_TODO.md` and `data/`), so keep a
backup outside the repo: a fresh clone won't have them, and the pipeline can't rebuild without the
seed CSVs. `artifacts/` is git-ignored too. What *does* ship is `web/data.json` — the site reads only that, so the deployed page
works from a clone regardless.

## Run it

```bash
uv sync --all-packages                              # install this member into the shared venv
uv run python -m dataframework.ingest               # seed CSVs -> catalog.json + benchmarks.json
uv run python -m dataframework.catalog --validate   # "spine is clean", or a loud failure
uv run python -m dataframework                      # compute + write web/data.json
```

`ingest` is a separate stage. Editing anything in `ingest.py` — a size correction, a dataset
relationship, the distribution register — and running only `python -m dataframework` will rebuild the
bundle from a stale `catalog.json` and silently show you the old values.

## Preview it

Preview the **built bundle**, not `web/` directly — the pages fetch `catalog.json`, which `build.sh`
places alongside them rather than duplicating into `web/`:

```bash
bash ../../../deploy/vercel/build.sh          # assemble public/
cd ../../../public/03-data-collection-framework
python3 -m http.server 8000                   # open http://localhost:8000
```

## Tests

```bash
uv run pytest -m "not integration"      # fast: unit tests + the five invariants
uv run playwright install chromium      # once, for the rendering suite
uv run pytest -m integration            # slower: loads the real page in a browser
```

The rendering suite exists because everything else checks the *bundle*. The two worst bugs this
project shipped both lived in that gap — a containment subtraction that was correct in `data.json`
and never fired in the browser, and a headline reading "0 of 55" that was true of a question nobody
meant to ask. `node --check` caught neither, because both files parsed perfectly. Without a browser
installed the suite **skips rather than fails**, so it stops protecting you silently; CI installs one.

## Two rules worth knowing before you change anything

**Every number is provenance-typed.** `{value, unit, provenance, source}`, never a bare figure, and
`unknown` is a real answer. The bare-number invariant enforces that every number is *typed* — it
cannot tell you the type is *right*, so a separate guard checks that no token sum inherits a
`measured` mark.

**One rule, one implementation.** Several rules exist in Python for the bundle and in JavaScript for
the page. That duplication has caused a shipped bug once (correction X28), so the invariant suite
runs the browser's own functions against the real bundle and fails on any disagreement. If you change
`blockers()`, `tier_of()` or the containment filter, change both halves in the same commit.

## Status

**Built and deployed.** The pipeline, the five CI invariants, the rendering suite and the site are
all in place. What remains open is tracked in [`docs/OPEN.md`](./docs/OPEN.md) — deliberately, since
an open decision recorded is worth more than a closed one invented.
