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
- Web exercises are **plain HTML/CSS/JS — zero runtime dependencies** — deployed to **Vercel** (one project routes each exercise under `/NN-slug/`; see [`deploy/`](deploy/)). What they render is either hand-written (exercise 01 trains tiny nets live in-browser) or precomputed by a Python pipeline (exercise 02 trains its BPE with HuggingFace `tokenizers`, then the widget renders the exported vocabulary and scores).

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
| 02 | [Tokenization](src/exercises/02-tokenization/) | A single 10k BPE vocabulary balanced across India's Wikipedia article in four languages, with an ablation harness that sweeps algorithm × representation × normalization × vocab × weighting. |

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
vocabulary** shared across India's Wikipedia article in **English, Hindi, Telugu, and Tamil** (the 4th
language is swappable in `config.py`), tuned so all four are tokenized about equally efficiently. The
graded quantity is the **spread** between the best- and worst-served language; the self-score is
`1000 / (max ratio − min ratio)`.

```bash
uv run python -m tokenization          # fetch → train the shared 10k BPE → print + save the report
uv run python -m tokenization.ablate   # sweep algorithm × representation × normalization × vocab × weighting
```

Headline finding from the ablation harness: **representation is the dominant lever**, not corpus
weighting. Byte-level BPE explodes each Indic character into 3 UTF-8 bytes (Tamil ≈ 6.5 tokens/word);
switching to char-level + Unigram + NFKC drops every language to ~1.7–2.2 tokens/word, cutting the
spread ~11× (score 190 → 2078).

A zero-dependency **widget** (`web/index.html`, rendering the exported `web/data.json`) shows the four
ratios, token stats, the score calculation, and the full searchable token list — the reviewer
deliverable. Preview it locally:

```bash
cd src/exercises/02-tokenization/web
python3 -m http.server 8000   # open http://localhost:8000
```

> **Hosting:** the widget deploys via the repo-wide **Vercel** project at `/02-tokenization/` (see
> [`deploy/`](deploy/)) — the single-project + routing setup serves every exercise, so this is no
> longer blocked on multi-exercise publishing. Connecting the Vercel project to the repo is the
> remaining one-time step.

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
