# 04 · Data Cleaning & Deduplication


**Raw data is not training data.** Eight named stages stand between the two. This exercise runs all
eight over three real corpora, counts what each one removes, and publishes the result.

**[Read the published page →](https://llm-pretraining-demos.vercel.app/04-data-cleaning-dedup/)**

All eight stages are real. `Extract` is the one permanent pass-through: every corpus here ships
already-extracted text, so claiming a yield for it would be inventing one.

## Data cleaning & deduplication — what survives

A Python pipeline (`src/exercises/04-data-cleaning-dedup/src/datacleaning/`) runs **eight named
cleaning stages** over **three real corpora** — reasoning traces, Indic web text, and Q&A — and
counts what each stage removes. Of **85.7M tokens** in, **69.86%** survive (50,010 documents down
to 36,890); the stages that cut hardest were not the expected ones.

Two things make it more than a filter chain:

- **Tokens are counted, never estimated.** Fertility is a property of a *tokenizer*, not of a
  corpus — Manipuri swings 7.6× across the five tokenizers exercise 03 measured. Every count here
  is produced by our own Session 2 vocabulary, and a count that is more than 5% `[UNK]` is not
  published as a count at all. That rule *selected the corpus*: Bengali script measures **82–84%
  `[UNK]`**, which is why the Indic corpus is Devanagari and Telugu.
- **Three of the nine standard quality rules turned out not to be language-neutral.** Applied
  unchanged to Indic text they delete it rather than filter it. Python's `\w` and `isalnum` skip
  Devanagari vowel signs, so mean-word-length measured every Devanagari word short and scored
  well-formed Hindi at **2.24** against a floor of 3.0. Counting letters *and* marks moves it to
  **3.56**.

```bash
uv run python -m datacleaning --profile lite    # smoke run, ~2 minutes
uv run python -m datacleaning --profile full    # the published corpus
```

> **Hosting:** live at <https://llm-pretraining-demos.vercel.app/04-data-cleaning-dedup/>.
> Ships a
> [decision record](src/exercises/04-data-cleaning-dedup/DECISIONS.md).

## What

| | |
|---|---|
| **The question** | How many cleaning strategies does Session 4 list, and what do they do to a real corpus? |
| **The answer** | **8** — and the session names two *different* eights. See [`DECISIONS.md`](./DECISIONS.md) §D1. |
| **The corpus** | ~90M tokens across three datasets, counted with **our own Session 2 tokenizer** |
| **The deliverable** | A published page and a manifest per shard |

## Why these three corpora

No single corpus exercises eight stages. Format discipline never fires on web crawl; the Indic
joiner branch never fires on English; PII regexes find nothing worth finding in reasoning traces.

| corpus | licence | the stage it alone exercises |
|---|---|---|
| [`open-thoughts/OpenThoughts-114k`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Apache-2.0 | **Format discipline** — the only corpus with chat structure and `<think>` traces |
| [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha) (Devanagari + Telugu) | CC-BY-4.0 | **Joiner preservation**, **language ID**, and the corpus the session names as never deduplicated |
| [`HuggingFaceH4/stack-exchange-preferences`](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) | CC-BY-SA-4.0 | **PII** that is really there — plus false positives that are really wrong |

Plus a deliberate **out-of-vocabulary probe** (Manipuri, Kashmiri) that is *excluded* from the token
budget and exists to produce one number: **84% `[UNK]`**.

## How: three decisions worth knowing

**We count tokens; we never estimate them.** The obvious approach — pick a fertility ratio, multiply
by word count — is wrong, because fertility is a property of *a tokenizer*. Manipuri swings **7.6×**
across the five tokenizers exercise 03 measured. So we tokenize, with our own Session 2 vocabulary,
and publish the spread as a finding.

**A count that is mostly `[UNK]` is not a count.** Our 10k vocabulary was trained on English, Hindi,
Telugu and Maithili, so Bengali script comes back 82–84% `[UNK]`. Rather than report a misleading
number, `Figure` carries `value=None, provenance="unknown"` and the page says so. That gate is why
the corpus is Devanagari and Telugu rather than Assamese.

**Nothing large is downloaded.** Shards are read as parquet **row groups over HTTP range requests**,
so a 344 MB file costs only what we consume.

## Run it

```bash
uv sync --all-packages

uv run python -m datacleaning.fetch --profile lite   # reachability check, seconds
uv run python -m datacleaning --profile lite         # smoke run, ~2 minutes
uv run python -m datacleaning --profile full         # the published corpus

uv run pytest src/exercises/04-data-cleaning-dedup   # the suite
uv run playwright install chromium                   # one-time, for the browser tests
uv run pytest src/exercises/04-data-cleaning-dedup -m integration

cd src/exercises/04-data-cleaning-dedup/web && python3 -m http.server 8000   # preview locally
```

`lite` is deliberately *below* the assignment's 10M-token floor — it exists to surface bugs in
minutes. The published corpus is `full`.

**Prefer the notebook.** `notebooks/S04-data-cleaning-dedup.ipynb` — built locally by
`tools/build_notebook.py` and not tracked, so regenerate it rather than looking for it in a clone
runs this same package step by step, with plain-English explanation before each step and the
arithmetic after it. Upload it to Colab to run it with no local setup.

**Local network note.** Python verifies TLS against `certifi/cacert.pem`, which Claude Code's
sandbox denies by default. `.claude/settings.local.json` carries a narrow read-allow for that one
file. Colab is unaffected.

## Layout

```text
DECISIONS.md          # why the answer is 8, why these corpora, what may be published (read first)
README.md             # this file
pyproject.toml        # workspace member
src/datacleaning/
  config.py           # the one @dataclass — every threshold, hashed into the manifest
  sources.py          # the three corpora + the OOV probe, with verified shard sizes
  records.py          # Document, StageStat, Figure — provenance-typed numbers
  fetch.py            # row-group streaming over HTTP range requests    (CLI)
  tokens.py           # our tokenizer, the [UNK] gate, the spread table
  corpus.py           # shards -> documents, deterministically
  pipeline.py         # the eight stages, composed
  manifest.py         # stage 8 — provenance, hashes, determinism
  export.py           # artifacts/run.json + web/data.json (100 KB budget)
tests/                # discovered by `uv run pytest` from the repo root
artifacts/            # generated outputs (git-ignored)
data/                 # any cached shards (git-ignored)
```

## The eight strategies

| # | stage | what it does |
|---|---|---|
| 1 | Extract | Session 3's topic — every corpus here ships extracted text |
| 2 | Normalize | Unicode, invisibles, entities, whitespace — **joiners preserved**, hash taken after |
| 2b | Format discipline | Ghost tags are *created* by rendering, not inherited from the data |
| 3 | Language ID | Detect the language; never trust the folder it came from |
| 4 | Quality filter | Nine Gopher/C4 rules at the session's thresholds |
| 5 | Deduplicate | Exact hashes, then MinHash/LSH at FineWeb's preset |
| 6 | PII scrub | Structured identifiers by regex; names by a declared stand-in |
| 7 | Decontaminate | Canaries and n-grams against held-out evaluation |
| 8 | Manifest | Provenance, hashes, and a run id derived from content rather than the clock |

Nine rows, eight strategies: stage 1 is inherited from Session 3 and 2b is the one the session's
pipeline map never numbers. Which eight you mean depends on which list you read — that ambiguity is
a real reading result, and [`DECISIONS.md`](./DECISIONS.md) §D1 works through it.

## Licences

Corpus text is redistributed only in bounded, post-scrub excerpts, attributed in `NOTICE`. No raw
PII is published anywhere — interactive demos run on hand-written synthetic documents, and a test
scans every byte of the bundle and the notebook to keep it that way. See
[`DECISIONS.md`](./DECISIONS.md) §D6.
