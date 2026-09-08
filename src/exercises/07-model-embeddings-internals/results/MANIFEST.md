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
| `lane_sensitivity.json` | `552874a44e3c` | `8398f0412e8f` | mps | — |
| `measurements.json` | `—` | `—` | — | — |
| `parallel_text.json` | `64d953fbf8b6` | `b201f1ec5350` | mps | — |
| `rerun.json` | `01e963d37d3b` | `8398f0412e8f` | cpu | 500 x 5 |
| `unk_confound.json` | `de99fd2d2f2f` | `49d599b3e245` | mps | — |

| file | corpus | tokens | `[UNK]` | epochs |
| --- | --- | ---: | ---: | ---: |
| `lane_sensitivity.json` | **3 corpora compared** — indic, web, code | — | — | — |
| `measurements.json` | — | — | — | — |
| `parallel_text.json` | **2 corpora compared** — parallel, ordinary | — | — | — |
| `rerun.json` | data/corpus (exercise 06's fetched lanes) | 11,781,888 | 0.209% | 0.0217 |
| `unk_confound.json` | **2 corpora compared** — confounded, clean | — | — | — |

## What these bundles say they do NOT establish

Carried from each bundle's own `limits` field rather than written here, so a reader
of this index meets the caveat at the same time as the number.

**`lane_sensitivity.json`**
- Lane and DOMAIN are confounded: the indic lane is encyclopedic text and the code lane is source code, so a difference between them is not necessarily a difference about script.
- This says nothing about whether the advantage is a property of comparable text -- exercise 02's corpus being one article in five languages. That is measured separately by tools/measure_parallel_text.py, and its answer is not derivable from these lanes.
- Each lane is read at a different epoch ratio, because the lanes are different sizes. All are far under 1.0, so nothing is seen twice, but the ratios are not equal.

**`parallel_text.json`**
- The two corpora differ in SOURCE and DOMAIN as well as in parallelism -- an encyclopedia article against a web crawl -- so a difference between them is not parallelism alone. It is the closest matched pair this repository holds without fetching new text.
- The overlap measurement counts shared byte n-grams, which is a proxy for shared content and not a measurement of translation. Two unrelated documents about the same subject would score high on it too.
- Both sides are cut into the same number of pieces of the same total size, because the share of shared n-grams falls as a corpus grows for reasons unrelated to translation. The first version of this measurement expressed each corpus as one piece and reported 0.00% for the ordinary one -- true by construction, and indistinguishable from a decisive result.
- The word 'parallel' is loose. These are not translations -- nobody rendered the English article into Telugu sentence by sentence. They are four articles about one subject, written independently in four languages, which is a COMPARABLE corpus rather than a parallel one.


## Where the material is

Each bundle names a `run_directory` under `artifacts/runs/`, holding the exact id
stream each seed consumed, a weight digest for every model at step zero, a per-step
trace of loss and pre-clip gradient norm, and trained weights with sidecars. It is
**gitignored and regenerable**; what is tracked is the digest of everything in it.

- `rerun.json` → `2026-09-08-01e963d37d3b`, whose record is tracked here: `runs/2026-09-08-01e963d37d3b/audit.json`, `runs/2026-09-08-01e963d37d3b/manifest.json`

## Checking it yourself

```bash
uv run python src/exercises/07-model-embeddings-internals/verify.py     # a run
uv run python src/exercises/07-model-embeddings-internals/evidence.py   # each claim
```

`verify.py` imports nothing from the package it audits, so it cannot agree with the
producer by construction.
