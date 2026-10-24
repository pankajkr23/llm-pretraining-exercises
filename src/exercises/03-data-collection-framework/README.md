# 03 · Data Collection & Sourcing — a decision framework

**Notebook:** [`notebooks/03-data-collection-framework.ipynb`](../../../notebooks/03-data-collection-framework.ipynb) — run this exercise's pipeline yourself, every setting in one cell · [open in Colab](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/03-data-collection-framework.ipynb) · [all fourteen, in order](../../../notebooks/README.md)

A framework for **deciding the pre-training data mix** of an India-first 40B model: grade every
candidate source through five gates, budget the tokens, measure the tokenizer tax, and then try to
actually assemble the corpus from what survives. The answer is uncomfortable, which is the point —
**4 of 145 datasets are committable today**, covering 6.56T of a 16.8T budget, and the blocker is
almost never quality.

See [`docs/`](./docs/) for the full spec.

## What it builds, and what it found

A Python pipeline (`src/exercises/03-data-collection-framework/src/dataframework/`) turns a research
atlas into a **graded catalogue of 145 datasets and 31 benchmarks**, then publishes **one page** that
answers, in the order a reader asks them: how much text a 40B model needs, what kind, **which
datasets to actually use**, what may legally be used, what to train on after pre-training, how to
clean it, how to tokenise it, how you would know it worked, what it costs, and what to do first.

Two things make it more than a write-up:

- **Five invariants enforced in CI, not in review.** Training never touches eval data · nothing
  excluded may enter a commercial mix · every judgment carries its reasoning and confidence · a
  measurement must name what produced it · no source content is silently dropped. Each is paired
  with a test proving it *fails* when broken — a guard nobody has watched fail is not a guard.
- **Every number carries its provenance.** `{value, unit, provenance, source}` all the way to the
  DOM, and the renderer throws on a bare number rather than printing it. Where a figure has never
  been measured, the page says so instead of showing a plausible one.

Every chapter is an **interactive explainer** in three layers: a plain headline and one number that
a newcomer can stop at, the interaction that proves the claim, and a closed *"The arithmetic"* with
the derivation for anyone who wants it. The contamination gate is the clearest example — type your
own sentence, try to smuggle it past a thirteen-word fingerprint index, and watch where the method
stops working.

```bash
uv run python -m dataframework          # rebuild web/data.json from the data spine
uv run pytest -m "not integration"      # the invariants, and the proofs they can fail
```

> **Hosting:** deploys via the repo-wide Vercel project at `/03-data-collection-framework/`.
> **Scope:** a coursework exercise, not a proposal to anyone — see
> [`NOTICE`](NOTICE).

## How to read this

| you are | start here | then |
| --- | --- | --- |
| **Meeting this for the first time** | [What it builds, and what it found](#what-it-builds-and-what-it-found), then the [live page](https://llm-pretraining-demos.vercel.app/03-data-collection-framework/) — thirteen chapters in the order a reader asks them | [What it produces](#what-it-produces) for the headline counts |
| **Changing the code** | [Layout](#layout) and [Run it](#run-it) — note `ingest` is a **separate stage**, and skipping it rebuilds from a stale catalogue | [Two rules worth knowing before you change anything](#two-rules-worth-knowing-before-you-change-anything), then [Tests](#tests) |
| **Deciding whether to believe it** | [What it cannot tell you](#what-it-cannot-tell-you) | [Tests](#tests) for which guards exist, and [`NOTICE`](NOTICE) for scope and affiliation |
| **Running it yourself** | [`notebooks/03-data-collection-framework.ipynb`](../../../notebooks/03-data-collection-framework.ipynb) — this exercise's own pipeline, step by step, every setting in one cell | [Open in Colab](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/03-data-collection-framework.ipynb) · [the map of all fourteen](../../../notebooks/README.md) |

The deep material is in [`docs/`](./docs/) — `ATLAS.md` is the source research, `FRAMEWORK.md` the
method, `DECISIONS.md` the resolved answers, `OPEN.md` what was deliberately left open.

## Best result, and what could beat it

**Best result.** This exercise produces a decision rather than a score, and its strongest outcome is
the one it was built to reach honestly: **4 of 145 datasets are committable today, carrying 6.56T of
a 16.8T seen-token budget**. A *seen* token is counted every time training reads it; the mixture's
tiers need **15.372T of *unique* text** to fill that budget, the difference being scarce tiers that
are deliberately read more than once. The grades are 14 B · 116 C · 15 X and nothing at A because
of coverage rather than quality: a grade summarises five gates, **X** means the provenance or
contamination gate failed and the dataset is excluded, **A** needs all five gates answered and a
high score, **B** at least three answered and a passing score, and **C** is everything else —
mostly datasets whose gates nobody has measured ([what it produces](#what-it-produces); the counts
are in [`web/data.json`](web/data.json) under `sourcing` and `grades`). Its one head-to-head
measurement is the Indic **tokenizer tax** — a tokenizer's tokens per word on an Indic language
divided by its tokens per word on English, so ×2 means every Indic word costs twice as much to
train on and to serve — on all 22 scheduled languages, where the best of the five tokenizers
measured has a mean tax of ×1.66 against ×2.49 for the one the inherited-model path would lock in
([`docs/DECISIONS.md`](docs/DECISIONS.md#what-the-fork-now-costs-measured)).

**Candidates that did or could do better.**

- **The fork — whether to train a new model or start from an existing one — is priced, not
  chosen.** Training from scratch, extending an existing model's
  vocabulary and continuing its training, or upcycling to a mixture of experts are all still open;
  the stated way to decide is a head-to-head at ~2B scale on identical data
  ([`docs/OPEN.md`](docs/OPEN.md#open--needs-a-decision)).
- **Nothing here has been deduplicated, and the deduplicated Indic web has never been measured.**
  The committable 6.56T is a sum of publishers' stated sizes; the pages carry 1.31T–2.62T for it
  after cross-corpus deduplication only from an assumed 20–40% survival, and every budget that uses
  it inherits the assumption ([what it cannot tell you](#what-it-cannot-tell-you)).
- **Four domains have no supply.** Agriculture and health have no catalogued dataset at all, and
  two more exist only inside an unseparated web slice ([`docs/OPEN.md`](docs/OPEN.md#open--needs-a-decision)).

**The next hypothesis.** Deduplicating the catalogued Indic web corpora against each other will
leave a survival share inside the assumed 20–40%; a measured share outside that band moves the
committable total and every tier budget downstream of it, which is why it is the measurement worth
making first.

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
| Contamination gate | 126,044 fingerprints over 8,923 items of MILU, an Indic evaluation benchmark built from Indian competitive-exam questions |
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

## What it cannot tell you

State these before quoting any figure above.

- **Most numbers here are other people's, not ours.** The catalogue is a reading of publishers'
  public material, and each figure carries its provenance in the data. `estimated` is the common
  case; only a minority are `measured` on our own runs, the tokenizer tax being the main exception.
- **120 provenance-typed values are `unknown`**, mostly `size_tokens` for datasets whose publisher
  never stated one. The page prints `unknown` rather than a plausible substitute, which is why the
  committable total is a floor and not an estimate.
- **The size of the deduplicated Indic web has not been measured by anyone, including here**, and
  neither has the deduplicated size of the committable supply. What the pages carry for the latter
  is a range — **1.31T–2.62T** of the raw 6.56T, from an assumed **20–40%** cross-corpus survival.
  Every budget that consumes it inherits that assumption, and measuring it would move more of this
  framework than any other single quantity.
- **No dataset reaches grade A, and that is about coverage, not quality.** 14 B · 116 C · 15 X.
  Reading the grades as a quality ranking misreads what the gates test.
- **The licensing material is not legal advice.** It is a summary written by a non-lawyer for a
  coursework exercise. Several catalogued datasets forbid commercial use outright; the catalogue
  records that and does not grant permission to ignore it.
- **This is a study of a decision, not a proposal to anyone**, and not affiliated with or endorsed
  by any organisation named. [`NOTICE`](NOTICE) is the authoritative statement of scope.

## Status

**Built and deployed.** The pipeline, the five CI invariants, the rendering suite and the site are
all in place. What remains open is tracked in [`docs/OPEN.md`](./docs/OPEN.md) — deliberately, since
an open decision recorded is worth more than a closed one invented.
