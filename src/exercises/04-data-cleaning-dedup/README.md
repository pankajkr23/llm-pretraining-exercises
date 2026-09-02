# 04 · Data Cleaning & Deduplication


**Raw data is not training data.** Eight named stages stand between the two. This exercise runs all
eight over three real corpora, counts what each one removes, and publishes the result.

**[Read the published page →](https://llm-pretraining-demos.vercel.app/04-data-cleaning-dedup/)**

All eight stages are real. `Extract` is the one permanent pass-through: every corpus here ships
already-extracted text, so claiming a yield for it would be inventing one.

## What it builds, and what it found

The pipeline lives in `src/exercises/04-data-cleaning-dedup/src/datacleaning/`, and the three
corpora are reasoning traces, Indic web text and Q&A. Of **85.7M tokens** in, **69.86%** survive (50,010 documents down
to 36,890); the stages that cut hardest were not the expected ones.

Two things make it more than a filter chain:

- **Tokens are counted, never estimated.** Fertility is a property of a *tokenizer*, not of a
  corpus — Manipuri swings 7.6× across the five tokenizers exercise 03 measured. Every count here
  is produced by our own Exercise 02 vocabulary, and a count that is more than 5% `[UNK]` is not
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
> [decision record](DECISIONS.md).

## How to read this

| you are | start here | then |
| --- | --- | --- |
| **Meeting this for the first time** | [What](#what) and [Why these three corpora](#why-these-three-corpora) — what was cleaned and why those three | [The eight strategies](#the-eight-strategies), then the [live page](https://llm-pretraining-demos.vercel.app/04-data-cleaning-dedup/) |
| **Changing the code** | [Layout](#layout) — one module per stage, one `config.py` holding every threshold | [Run it](#run-it), then [Tests](#tests) |
| **Deciding whether to believe it** | [How: three decisions worth knowing](#how-three-decisions-worth-knowing) — especially why counts are never estimated | [What it cannot tell you](#what-it-cannot-tell-you), then [`DECISIONS.md`](./DECISIONS.md) |

## What

| | |
|---|---|
| **The question** | How many cleaning strategies does Exercise 04 list, and what do they do to a real corpus? |
| **The answer** | **8** — and the source material names two *different* eights. See [`DECISIONS.md`](./DECISIONS.md) §D1. |
| **The corpus** | ~90M tokens across three datasets, counted with **our own Exercise 02 tokenizer** |
| **The deliverable** | A published page and a manifest per shard |

## Why these three corpora

No single corpus exercises eight stages. Format discipline never fires on web crawl; the Indic
joiner branch never fires on English; PII regexes find nothing worth finding in reasoning traces.

| corpus | licence | the stage it alone exercises |
|---|---|---|
| [`open-thoughts/OpenThoughts-114k`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Apache-2.0 | **Format discipline** — the only corpus with chat structure and `<think>` traces |
| [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha) (Devanagari + Telugu) | CC-BY-4.0 | **Joiner preservation**, **language ID**, and the corpus the source material names as never deduplicated |
| [`HuggingFaceH4/stack-exchange-preferences`](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) | CC-BY-SA-4.0 | **PII** that is really there — plus false positives that are really wrong |

Plus a deliberate **out-of-vocabulary probe** (Manipuri, Kashmiri) that is *excluded* from the token
budget and exists to produce one number: **84% `[UNK]`**.

## How: three decisions worth knowing

**We count tokens; we never estimate them.** The obvious approach — pick a fertility ratio, multiply
by word count — is wrong, because fertility is a property of *a tokenizer*. Manipuri swings **7.6×**
across the five tokenizers exercise 03 measured. So we tokenize, with our own Exercise 02 vocabulary,
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

`lite` is deliberately *below* the requirements' 10M-token floor — it exists to surface bugs in
minutes. The published corpus is `full`.

**Prefer the notebook.** `notebooks/S04-data-cleaning-dedup.ipynb` — **not in a clone.** It and
the `tools/build_notebook.py` that emits it are both local-only, so a fresh checkout has neither
runs this same package step by step, with plain-English explanation before each step and the
arithmetic after it. Upload it to Colab to run it with no local setup.

**Local network note.** Python verifies TLS against `certifi/cacert.pem`, which Claude Code's
sandbox denies by default. `.claude/settings.local.json` carries a narrow read-allow for that one
file. Colab is unaffected.

## Layout

```text
DECISIONS.md          # why the answer is 8, why these corpora, what may be published (read first)
CLAUDE.md             # rules specific to this exercise, for whoever changes the code
NOTICE                # affiliation and licence disclaimer
web/                  # the deployed page — index.html, chapters.js, data.json, page-extra.css, _shared/
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
| 1 | Extract | Exercise 03's topic — every corpus here ships extracted text |
| 2 | Normalize | Unicode, invisibles, entities, whitespace — **joiners preserved**, hash taken after |
| 2b | Format discipline | Ghost tags are *created* by rendering, not inherited from the data |
| 3 | Language ID | Detect the language; never trust the folder it came from |
| 4 | Quality filter | Nine Gopher/C4 rules at the source material's thresholds |
| 5 | Deduplicate | Exact hashes, then MinHash/LSH at FineWeb's preset |
| 6 | PII scrub | Structured identifiers by regex; names by a declared stand-in |
| 7 | Decontaminate | Canaries and n-grams against held-out evaluation |
| 8 | Manifest | Provenance, hashes, and a run id derived from content rather than the clock |

Nine rows, eight strategies: stage 1 is inherited from Exercise 03 and 2b is the one the source material's
pipeline map never numbers. Which eight you mean depends on which list you read — that ambiguity is
a real reading result, and [`DECISIONS.md`](./DECISIONS.md) §D1 works through it.

## Tests

```bash
uv run pytest src/exercises/04-data-cleaning-dedup -m "not integration"   # fast
uv run playwright install chromium                                        # one-time
uv run pytest src/exercises/04-data-cleaning-dedup -m integration         # the browser suite
```

One file per stage — `test_normalize`, `test_langid`, `test_quality`, `test_dedup`, `test_pii`,
`test_decontaminate` — plus three groups that guard the things a stage test cannot see:

- **`test_publication_invariants.py`** is the one to read first. It scans **every byte** of the
  published bundle *and* the notebook for personal information, checks that the PII stage publishes
  counts rather than matches, that corpus excerpts stay inside the declared window, and that no raw
  shard ships. Three of its eight tests exist only to prove the other five *can fail* —
  `test_the_bundle_scan_can_actually_fail` and `test_the_excerpt_bound_can_actually_fail` run the
  same scan against a deliberately poisoned fixture and assert it goes red.
- **`test_agreement.py`** runs the browser's own functions against the real bundle, because several
  rules exist twice — once in Python for the data, once in JavaScript for the page — and a shipped
  bug has come from exactly that gap before.
- **`test_page_render.py`** (integration) loads the built page in Chromium and asserts what a reader
  actually sees. Without a browser installed it **skips rather than fails**, so it protects you
  silently or not at all; CI installs one.

## What it cannot tell you

- **The token counts are our 10k vocabulary's, not a universal measure.** Fertility is a property of
  a tokenizer. Every count here is reproducible and none is portable — a different tokenizer gives
  different numbers for the same text, which is the finding, not a caveat about it.
- **It says nothing about Bengali or Assamese.** Those scripts come back **82–84% `[UNK]`** against
  our vocabulary, so they were excluded rather than reported. The Indic results are **Devanagari and
  Telugu only**, and the out-of-vocabulary probe (Manipuri, Kashmiri) is excluded from the token
  budget entirely — it exists to produce one number.
- **Name detection is a declared stand-in, not a NER model.** Structured identifiers are found by
  regex, which is why the page shows the false positives instead of hiding them. Do not read the
  PII stage as a measure of how much PII a corpus contains.
- **Three corpora is not a sample.** They were chosen because between them they are the smallest set
  that fires all eight stages — no single corpus does. Yields are properties of these three, at
  these thresholds, and generalise no further.
- **`lite` is deliberately below the requirements' 10M-token floor.** It exists to surface bugs in
  minutes. Only `full` produced the published numbers; a `lite` run reproduces the *pipeline*, not
  the results.
- **"Eight strategies" is itself a reading.** The source material names two different eights. Which one you
  mean changes the table — [`DECISIONS.md`](./DECISIONS.md) §D1 works through it rather than
  picking silently.

## Licences

Corpus text is redistributed only in bounded, post-scrub excerpts, attributed in `NOTICE`. No raw
PII is published anywhere — interactive demos run on hand-written synthetic documents, and a test
scans every byte of the bundle and the notebook to keep it that way. See
[`DECISIONS.md`](./DECISIONS.md) §D6.
