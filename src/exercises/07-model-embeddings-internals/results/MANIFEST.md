# What is in `results/`, and what produced it

**Generated — do not edit.** Rebuild with:

```bash
uv run python src/exercises/07-model-embeddings-internals/tools/publish_rerun.py \
    --manifest-only
```

It reads the tracked bundles in this directory and nothing else, so it rebuilds in
a fresh clone. A manifest generated from the gitignored run directories would be a
tracked document only its author could regenerate.

A dash is not a formatting gap — it is a field the bundle does not carry, and the
contrast is the reason this table exists. `measurements.json` is the inherited record,
and it can say which architecture was trained and **not** which settings, which commit,
which machine or which text. That is why its losses can be read but not aimed at.

| file | settings | commit | device | steps x seeds |
| --- | --- | --- | --- | --- |
| `measurements.json` | `—` | `—` | — | — |
| `rerun.json` | `01e963d37d3b` | `8398f0412e8f` | cpu | 500 x 5 |

| file | corpus | tokens | `[UNK]` | epochs |
| --- | --- | ---: | ---: | ---: |
| `measurements.json` | — | — | — | — |
| `rerun.json` | data/corpus (exercise 06's fetched lanes) | 11,781,888 | 0.209% | 0.0217 |

## Where the material is

Each bundle names a `run_directory` under `artifacts/runs/`, holding the exact id
stream each seed consumed, a weight digest for every model at step zero, a per-step
trace of loss and pre-clip gradient norm, and trained weights with sidecars. It is
**gitignored and regenerable**; what is tracked is the digest of everything in it.

- `rerun.json` → `2026-09-08-01e963d37d3b`

## Checking it yourself

```bash
uv run python src/exercises/07-model-embeddings-internals/verify.py     # a run
uv run python src/exercises/07-model-embeddings-internals/evidence.py   # each claim
```

`verify.py` imports nothing from the package it audits, so it cannot agree with the
producer by construction.
