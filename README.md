# llm-pretraining-exercises

[![CI](https://github.com/pankajkr23/llm-pretraining-exercises/actions/workflows/ci.yml/badge.svg)](https://github.com/pankajkr23/llm-pretraining-exercises/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/packaged%20with-uv-6340ac)

Hands-on exercises and capstone work for **LLM pre-training** — building a large language model
from scratch, from a minimal transformer block all the way to launching and operating a real
training run.

- **Repo conventions (single source of truth for humans & coding agents):** [`AGENTS.md`](AGENTS.md)

## Repository layout

A [uv](https://docs.astral.sh/uv/) **workspace** on Python 3.12. Each topic's work is a member package
under `src/exercises/NN-slug/` (numeric, zero-padded so folders sort correctly), all sharing one root
`.venv` and one `uv.lock`.

```text
docs/DESIGN.md                    # the shared web design system (palette, type, tone)
src/exercises/NN-slug/            # one self-contained exercise per topic (workspace member)
  ├─ README.md                    # what it is + how to run
  ├─ CLAUDE.md                    # rules specific to this exercise (REQUIRED — a test checks for it)
  ├─ DECISIONS.md                 # why it is the way it is (where the reasoning needs room)
  ├─ pyproject.toml               # workspace member
  ├─ src/ | web/                  # the code
  ├─ artifacts/                   # generated outputs (git-ignored)
  ├─ results/                     # measured evidence a document renders (TRACKED — see below)
  └─ tests/                       # exercise tests, discovered from the root
tests/                            # repo-wide guards (README links, anchors, exercise structure)
notebooks/hello.ipynb            # tracked sample; session notebooks are built locally, not versioned
pyproject.toml                    # workspace root + ruff/pytest config
AGENTS.md                         # repo conventions (imported by CLAUDE.md; pointed to by Cursor/Copilot)
.github/workflows/ci.yml          # lint + tests + secret scan
```

**Data conventions** — five concerns kept physically separate: assignment briefs are **never
tracked** (`BRIEF.md` is gitignored everywhere — a brief is the course's text and is input for
whoever builds the exercise, not the deliverable); session notebooks live in `notebooks/`
(git-ignored but for a tracked sample, as do their builders); datasets live in a top-level `data/`
(git-ignored, with a tracked manifest and a fetcher that verifies each licence at fetch time);
per-exercise outputs go to `artifacts/` (git-ignored); and **measured evidence a document renders
goes to `results/`, which is tracked** — the exception that matters, because a published figure whose
run output does not survive a clone cannot be rebuilt or checked. Full rules in
[`AGENTS.md`](AGENTS.md).

## Tech stack

- **Python 3.12**, managed by **uv** (workspace, shared lockfile).
- **[ruff](https://docs.astral.sh/ruff/)** for lint + format, **pytest** for tests (unit / integration split).
- **GitHub Actions** CI (ruff, pytest, `node --check`, gitleaks secret scan).
- Web exercises are **plain HTML/CSS/JS — zero runtime dependencies** — deployed to **Vercel** (one project routes each exercise under `/NN-slug/`; see [`deploy/`](deploy/)). What they render is either hand-written (exercise 01 trains tiny nets live in-browser) or precomputed by a Python pipeline (exercise 02 trains its BPE with HuggingFace `tokenizers`, then the widget renders the exported vocabulary and scores). All web pages share one Apple-style **design system** — see [`docs/DESIGN.md`](docs/DESIGN.md).

## Getting started

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) (installs/manages Python 3.12 for you).

```bash
uv sync --all-packages   # create the shared .venv and install every exercise + dev tools
uv run main.py           # sanity check
uv run pytest            # run every exercise's tests from the root
```

## Exercises

| # | Exercise | Summary |
| --- | --- | --- |
| 01 | [Introductions](src/exercises/01-introductions/) | Four live, in-browser interactive proofs of *why neural nets work*. Static site, zero dependencies, deployed to Vercel. [The page](https://llm-pretraining-demos.vercel.app/01-introductions/). |
| 02 | [Tokenization](src/exercises/02-tokenization/) | One 10k BPE vocabulary balanced across India's Wikipedia in four languages, scored on faithful units — with a live in-browser encoder, and an explainer for why the biggest number on the page is the one we rejected. [The page](https://llm-pretraining-demos.vercel.app/02-tokenization/). |
| 03 | [Data collection framework](src/exercises/03-data-collection-framework/) | How you decide what an India-first 40B model trains on — thirteen interactive chapters, 145 datasets graded on five checks, and five data-handling invariants enforced in CI. [The page](https://llm-pretraining-demos.vercel.app/03-data-collection-framework/). |
| 04 | [Data cleaning & deduplication](src/exercises/04-data-cleaning-dedup/) | Eight cleaning stages over three real corpora, counting tokens with **our own Session 2 tokenizer** — and finding that three of the nine standard quality rules are not language-neutral. [The page](https://llm-pretraining-demos.vercel.app/04-data-cleaning-dedup/). |
| 05 | [Data mixtures & curriculum](src/exercises/05-datamixtures-and-curriculum/) | The V5 training recipe as a **[specification you can argue with](src/exercises/05-datamixtures-and-curriculum/SPEC.md)** — a defended share for every capability lane, sized against the datasets that actually exist, sixteen invariants in CI, and a proxy that costs nothing and returned one **refuted** hypothesis. [The recipe, the evidence and its limits](src/exercises/05-datamixtures-and-curriculum/README.md) · [the page](https://llm-pretraining-demos.vercel.app/05-datamixtures-and-curriculum/). |
| 06 | [Building the training dataset](src/exercises/06-build-training-dataset/) | The **training data execution system**: tokenized shards, manifests, packing, a chain-hashed consumption ledger, a real crash and resume that lands on the same batch ids, and replay that re-derives every microbatch from the record rather than recomputing it. [The stages, the measurements and their limits](src/exercises/06-build-training-dataset/README.md) · [the page](https://llm-pretraining-demos.vercel.app/06-build-training-dataset/). *Stage 8 of 8 — one command builds the bundle at 9 of 9 requirements; a walled-off auditor re-derives every claim from it and passes 40 of 40.* |
| 07 | [Model embeddings internals](src/exercises/07-model-embeddings-internals/) | **Kronecker byte embeddings v2** — the paper's own *Limitations* section says its output head cannot be tied, which makes v1 **1.16× larger** than the baseline it beats on the input side. Tying the *induced* embedding instead, wrapping byte positions rather than truncating them, and adding one hashed byte-n-gram term beats v1 on 5 of 5 seeds with fewer parameters and **no vocabulary-sized parameter anywhere**. The codec also inverts exactly, with a self-certifying decoder. [The argument, the evidence and its limits](src/exercises/07-model-embeddings-internals/README.md) · [the page](https://llm-pretraining-demos.vercel.app/07-model-embeddings-internals/). |
| 08 | [Modern attention variants](src/exercises/08-modern-attention-variants/) | **Attention in the order it was launched** — thirty mechanisms from Bahdanau's 2014 soft alignment to higher-dimensional RoPE in August 2026, each dated from the primary source and framed as an answer to a problem that existed at that moment. Every date carries the URL it was read from and the source's own wording, because the graded axis is the chronology and an invented date is the failure mode. Ordering them turned up what a list hides: attention is three years older than the Transformer, nobody attacked its cost for 680 days after it shipped, two mechanisms sat unusable for over three years each after publication, and the timeline now ends on an open contradiction about whether positional embeddings should be deleted or enriched. The page is a monograph — six numbered plates, six chapters, and the thirty as one object entered thirty times rather than thirty cards. Two claims that would have been easy to write from memory are sourced instead: every hyperparameter it draws carries the sentence it was read from, and every model named as shipping a mechanism is quoted from that model's own paper — with twenty-two of the thirty deliberately naming none, which separates what the field adopted from what it admired. Testing the brief's own claimed arc refuted it, and then varying an arbitrary bucket edge refuted one of our own findings a day after it was published. [The chronology, the trade-offs and its limits](src/exercises/08-modern-attention-variants/README.md) · [the page](https://llm-pretraining-demos.vercel.app/08-modern-attention-variants/). |

More exercises are added each week.

## Development

- **Tests:** `uv run pytest` (fast unit) · `uv run pytest -m integration` (slower end-to-end). Each exercise owns its `tests/`; the root `tests/` holds the repo-wide guards that no single exercise can own — every README's relative links and in-page anchors resolve, every exercise README carries a reading path, a runnable command and a statement of what it cannot establish, and **every published page from exercise 05 onward tells the same story in the same order** (`tests/test_page_spine.py`: a twelve-part spine from `thesis` to `reproduce`, declared as `data-role`, with a ledger that fails in *both* directions so a new exercise cannot skip it by accident).
- **Lint / format:** `uv run ruff check --fix .` and `uv run ruff format .`. The enforceable style spec (PEP 8/257, modern typing, line length 100) lives in `pyproject.toml`.
- **CI** (`.github/workflows/ci.yml`, on every PR and on pushes to `main`): **four concurrent jobs**, not one chain. `test` — `uv sync --all-packages` → `ruff check` → `ruff format --check` → unit tests → `node --check` over every `src/exercises/*/web/**/*.js`. `integration` — a **three-shard matrix** (`tokenization` · `mixtures` · everything else), each shard syncing, installing chromium and assembling the site before running the integration suite. `train` — installs the `train` extra with **CPU-only torch wheels** (191.8 MB rather than 2.7 GB, via a Linux-scoped index) and runs exactly the files whose module-level `importorskip("torch")` would otherwise skip them entirely. `security` — **gitleaks** over the full history.

## Adding a new exercise

Every exercise follows the same skeleton, so the repo stays predictable:

**Do not scaffold one by hand. There is a generator**, and the sequencing is the reason it
exists — `tests/_exercises.py` only counts a directory once it has a `pyproject.toml`, so a new
exercise is invisible to every guard until that file lands and then six test families apply at once.

```bash
uv run python tools/new_exercise.py 09 loss-functions-output-heads \
    --title "Loss functions and output heads" --package lossheads \
    --summary "One sentence for the root README row." [--dry-run]
```

It writes the whole skeleton — including the three gitignored files — joins the CI shard, adds the
row to the table above, and prints what is left for you. It deliberately does **not** touch the two
web-gated registrations (the landing card and the `SPINE_ENFORCED` ledger), because both guards
assert in two directions and a premature entry is exactly as red as a missing one.

Then match the conventions in [`AGENTS.md`](AGENTS.md) and build the page to
[`docs/DESIGN.md`](docs/DESIGN.md) from the first commit rather than retro-fitting it later.
`tests/test_exercise_skeleton.py` requires `README.md`, `CLAUDE.md`, `pyproject.toml` and `tests/`,
and asserts no `BRIEF.md` is ever tracked. Introduce a shared `src/common/` package only once a
second exercise needs to reuse something.

## 💳 Credits

The blog posts and mini-projects here are my own implementations and summaries based on the
curriculum below. Full credit for the assignments, capstone projects, and dataset structures
goes to the course and its instructor.

- **Course:** ERA V5
- **Instructor:** Rohan Shravan
- **Platform:** [The School of AI](https://registrations.theschoolofai.in/courses/era-v5)
