# What is in `results/reruns/2026-10-24/`, and what produced it

**Generated — do not edit.** Rebuild with:

```bash
uv run python src/exercises/07-model-embeddings-internals/tools/publish_rerun.py \
    --manifest-only --date 2026-10-24
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
| `codec_facts.json` | `ef9fb3854545` | `fd18c2e190b8` | cpu | — |
| `measurements.restructured.json` | `—` | `—` | — | — |
| `position_schemes.json` | `4eac9fe2598f` | `fd18c2e190b8` | cpu | — |
| `rerun.json` | `01e963d37d3b` | `8398f0412e8f` | cpu | 500 x 5 |
| `wrap_recovery.json` | `0f710d79a76a` | `fd18c2e190b8` | cpu | — |

| file | corpus | tokens | `[UNK]` | epochs |
| --- | --- | ---: | ---: | ---: |
| `codec_facts.json` | — | — | — | — |
| `measurements.restructured.json` | — | — | — | — |
| `position_schemes.json` | — | — | — | — |
| `rerun.json` | data/corpus (exercise 06's fetched lanes) | 11,781,888 | 0.209% | 0.0217 |
| `wrap_recovery.json` | — | — | — | — |

## What these bundles say they do NOT establish

Carried from each bundle's own `limits` field rather than written here, so a reader
of this index meets the caveat at the same time as the number.

**`codec_facts.json`**
- Parameter counts are at d_model 768, GPT-2 124M's width; every trained figure on the page is at 256. Never quote one as evidence at the other.
- The lock samples are taken from a head at initialisation. They show the identity holds for the architecture; the trained-model residual is the earlier run's.
- The scale-cost memory and timing columns are NOT here. They are the earlier run's, measured on an unrecorded machine, and stay in measurements.json marked inherited.

**`position_schemes.json`**
- W is a random Gaussian projection with unit-norm rows at one seed. A different seed moves the percentages; the ORDERING and the by-construction zeros are the claim.
- fourier is not measured. Its code is not block-one-hot, decode.recover refuses it by design, and inventing a decoder for it here would compare two harnesses.
- This is a property of the CODE, not of a trained model. It says nothing about which scheme trains to a lower loss -- on that, wrap still has the only measured win.
- Tokens longer than 128 bytes are outside every band. The longest in this vocabulary is 121 bytes.
- spc's reach is a declared constant, so its cost is paid whether or not any token is that long: the decoder's dictionary is reach x 256 x d_model.

**`wrap_recovery.json`**
- W is a random Gaussian projection with unit-norm rows at one seed, so a different seed moves these percentages. The ORDERING of the two variants is the claim.
- The per-wrap byte-permutation variant is NOT measured. It was tried and removed, and two attempts to reimplement it from the published description produced harness artefacts rather than results. Its 14.6% figure is unreproduced.
- Bands are cut at d_p and 2*d_p, which is where the folding mechanism changes. Any other cut would give different percentages for the same underlying behaviour.


## Where the material is

Each bundle names a `run_directory` under `artifacts/runs/`, holding the exact id
stream each seed consumed, a weight digest for every model at step zero, a per-step
trace of loss and pre-clip gradient norm, and trained weights with sidecars. It is
**gitignored and regenerable**; what is tracked is the digest of everything in it.

- `rerun.json` → `2026-09-08-01e963d37d3b`, whose record is tracked here: —

## Checking it yourself

```bash
uv run python src/exercises/07-model-embeddings-internals/verify.py     # a run
uv run python src/exercises/07-model-embeddings-internals/evidence.py   # each claim
```

`verify.py` imports nothing from the package it audits, so it cannot agree with the
producer by construction.
