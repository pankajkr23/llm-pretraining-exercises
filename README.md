# llm-pretraining-exercises

[![CI](https://github.com/pankajkr23/llm-pretraining-exercises/actions/workflows/ci.yml/badge.svg)](https://github.com/pankajkr23/llm-pretraining-exercises/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/packaged%20with-uv-6340ac)

Hands-on exercises and capstone work for **LLM pre-training** — building a large language model
from scratch, from a minimal transformer block all the way to launching and operating a real
training run.

- **Overview & full syllabus:** [`docs/BRIEF.md`](docs/BRIEF.md)
- **Repo conventions (single source of truth for humans & coding agents):** [`AGENTS.md`](AGENTS.md)

## About the program

| | |
| --- | --- |
| **Duration** | ~6 months, including the training run that continues past the formal calendar |
| **Sessions** | 20 live classes, up to 3 hours each, Saturdays 7:00 AM IST |
| **Format** | Live coding + weekly assignments + ongoing lab contributions |
| **Capstone** | The actual flagship training run (starts ~week 22); students are staffed into running roles |

The syllabus moves from transformer foundations → tokenization → data (sourcing, cleaning, mixtures,
dataset building) → embeddings & attention variants → losses → training loop → optimizers →
distributed training → MoE → stability & scaling laws → SFT → preference alignment → infra &
quantization → the training-run kickoff. See [`docs/BRIEF.md`](docs/BRIEF.md) for the class-by-class detail.

## Repository layout

A [uv](https://docs.astral.sh/uv/) **workspace** on Python 3.12. Each class's work is a member package
under `src/exercises/NN-slug/` (numeric, zero-padded so folders sort correctly), all sharing one root
`.venv` and one `uv.lock`.

```text
docs/BRIEF.md                     # the program: structure + 20-class syllabus
docs/DESIGN.md                    # the shared web design system (palette, type, tone)
src/exercises/NN-slug/            # one self-contained exercise per class (workspace member)
  ├─ BRIEF.md                     # the assignment
  ├─ README.md                    # what it is + how to run
  ├─ pyproject.toml               # workspace member
  ├─ src/ | web/                  # the code
  ├─ artifacts/                   # generated outputs (git-ignored)
  └─ tests/                       # exercise tests, discovered from the root
pyproject.toml                    # workspace root + ruff/pytest config
AGENTS.md                         # repo conventions (imported by CLAUDE.md; pointed to by Cursor/Copilot)
.github/workflows/ci.yml          # lint + tests + secret scan
```

**Data conventions** — three concerns kept physically separate: briefs/docs are tracked; datasets live
in a top-level `data/` (git-ignored, with a tracked manifest); per-exercise outputs go to `artifacts/`
(git-ignored).

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
| 01 | [Introductions](src/exercises/01-introductions/) | Four live, in-browser interactive proofs of *why neural nets work*. Static site, zero dependencies, deployed to Vercel. |
| 02 | [Tokenization](src/exercises/02-tokenization/) | A single 10k BPE vocabulary balanced across India's Wikipedia article in four languages — scored on faithful units, with a held-out check separating a real gain from in-sample tuning, and a live in-browser encoder you can paste into. |
| 03 | [Data collection framework](src/exercises/03-data-collection-framework/) | How you decide what an India-first 40B model trains on — one interactive page, thirteen chapters: how much text, what kind, **which datasets**, how to clean it, how to tokenise it, and how you would know it worked. 145 datasets graded on five checks, of which **4 are committable today**; five data-handling invariants enforced in CI, plus a browser suite that tests the rendered page. |

More exercises are added each week.

### 01 · Introductions — four live proofs

A small site (`src/exercises/01-introductions/web/`) that **proves** four foundational ML claims by
training tiny models **live in the browser** — no server, no pre-baked figures, no libraries. Each of
the four pages inlines its own CSS + JS; the neural nets (forward pass, backprop, Adam) are hand-written.

| Page | Claim | The interactive |
| --- | --- | --- |
| **The bend** | Activations exist for a reason | Rotate a 3-D neuron surface across none/ReLU/tanh/GELU; then train linear vs a ReLU layer on two rings (~55% vs ~99%). |
| **Five maps, one matrix** | Depth without nonlinearity is a lie | Watch N linear layers collapse into one matrix (gap ≈ 1e-16), then flip on ReLU and it breaks. |
| **Meaning from company** | Embeddings learn similarity from next-token alone | A next-token model on a toy grammar; tokens migrate into animal/fruit/verb clusters. |
| **Memorise, or generalise** | Data closes the generalization gap | Drag the dataset size (20→2000) and watch the memorised boundary smooth out and the gap close. |

Preview locally:

```bash
cd src/exercises/01-introductions/web
python3 -m http.server 8000   # open http://localhost:8000
```

Deploy: handled by the repo-wide **Vercel** project — `deploy/vercel/build.sh` serves this exercise's
`web/` at `/01-introductions/`; previews auto-deploy per PR, production is on-demand. See [`deploy/`](deploy/).
(The prior Netlify config is deactivated in `deploy/netlify/`, pending decommission.)

### 02 · Tokenization — one vocabulary, four languages

A Python pipeline (`src/exercises/02-tokenization/src/tokenization/`) builds **one 10,000-token BPE
vocabulary** shared across India's Wikipedia article in **English, Hindi, Telugu, and Maithili**,
tuned so all four are tokenized about equally efficiently. Fertility X is tokens per *faithful unit*
— one run of letters/marks/digits, or one visible punctuation character — and the score is
`1000 / (X_max − X_min)`, divided by a penalty that fires if Hindi is degraded. The corpus is
committed wiki-faithful Markdown, so every number reproduces offline from a fresh clone.

```bash
uv run python -m tokenization          # train the submission → print + save the report
uv run python -m tokenization.ablate   # the full experiment table
uv run python -m tokenization.holdout  # in-sample vs held-out
```

Two findings worth the detour. **Where the trainer's input is cut matters as much as the recipe:**
HuggingFace splits files into lines, so training from files means no merge may span a newline —
feed it whole documents instead and every token count drops ~0.6%. And **most of a weighting win is
in-sample fit**: a configuration scoring 35,604 on the training corpus *loses* to the submitted one
(10,934) on held-out text, which is why the submission is chosen on held-out numbers and why every
table reports corpus-wide compression next to the score.

A zero-dependency **widget** (`web/index.html`) shows the four fertilities, the score calculation
with its penalty, the full searchable vocabulary, and a **paste-your-own-text encoder** that replays
the real merge list in the browser — a vocabulary list alone cannot reproduce a score, so the
ordered merges, `encoder.js`, and the tokenizer in HuggingFace format all ship with it.

```bash
cd src/exercises/02-tokenization/web
python3 -m http.server 8000   # open http://localhost:8000
```

> **Hosting:** live at <https://llm-pretraining-demos.vercel.app/02-tokenization/>, deployed via the
> repo-wide **Vercel** project (see [`deploy/`](deploy/)) — one project serves every exercise
> under its slug.

### 03 · Data collection framework — deciding the mix

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
stops working. Conventions in [`docs/EXPLAINER_PROMPT.md`](docs/EXPLAINER_PROMPT.md) (what to build)
and [`docs/EXPLAINER_PATTERN.md`](docs/EXPLAINER_PATTERN.md) (how); what the first build got wrong
is in [`DESIGN_CRITIQUE.md`](src/exercises/03-data-collection-framework/docs/DESIGN_CRITIQUE.md).

```bash
uv run python -m dataframework          # rebuild web/data.json from the data spine
uv run pytest -m "not integration"      # the invariants, and the proofs they can fail
```

> **Hosting:** deploys via the repo-wide Vercel project at `/03-data-collection-framework/`.
> **Scope:** a coursework exercise, not a proposal to anyone — see
> [`NOTICE`](src/exercises/03-data-collection-framework/NOTICE).

## Development

- **Tests:** `uv run pytest` (fast unit) · `uv run pytest -m integration` (slower end-to-end). Each exercise owns its `tests/`.
- **Lint / format:** `uv run ruff check --fix .` and `uv run ruff format .`. The enforceable style spec (PEP 8/257, modern typing, line length 100) lives in `pyproject.toml`.
- **CI** (`.github/workflows/ci.yml`, on every push & PR): `uv sync --all-packages` → `ruff check` → `ruff format --check` → unit tests → integration tests → `node --check` on web JS, plus a parallel **gitleaks** secret scan.

## Adding a new exercise

Every exercise follows the same skeleton, so the repo stays predictable:

```bash
mkdir -p src/exercises/03-slug/{src,tests}
# add pyproject.toml (workspace member), BRIEF.md, README.md
uv sync --all-packages   # the members = ["src/exercises/[0-9][0-9]-*"] glob picks it up automatically
```

Match the conventions in [`AGENTS.md`](AGENTS.md): zero-padded `NN-slug` folders, code in one place
(`src/` or `web/`), `artifacts/` for outputs, tests in `tests/`. Introduce a shared `src/common/`
package only once a second exercise needs to reuse something.

## 💳 Credits

The blog posts and mini-projects here are my own implementations and summaries based on the
curriculum below. Full credit for the assignments, capstone projects, and dataset structures
goes to the course and its instructor.

- **Course:** ERA V5
- **Instructor:** Rohan Shravan
- **Platform:** [The School of AI](https://registrations.theschoolofai.in/courses/era-v5)
