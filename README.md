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
  ├─ DECISIONS.md                 # why it is the way it is (where the reasoning needs room)
  ├─ pyproject.toml               # workspace member
  ├─ src/ | web/                  # the code
  ├─ artifacts/                   # generated outputs (git-ignored)
  └─ tests/                       # exercise tests, discovered from the root
notebooks/hello.ipynb            # tracked sample; session notebooks are built locally, not versioned
pyproject.toml                    # workspace root + ruff/pytest config
AGENTS.md                         # repo conventions (imported by CLAUDE.md; pointed to by Cursor/Copilot)
.github/workflows/ci.yml          # lint + tests + secret scan
```

**Data conventions** — three concerns kept physically separate: assignment briefs are **never
tracked** (`BRIEF.md` is gitignored everywhere — a brief is the course's text and is input for
whoever builds the exercise, not the deliverable); datasets live in a top-level `data/` (git-ignored,
with a tracked manifest); per-exercise outputs go to `artifacts/` (git-ignored).

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
| 02 | [Tokenization](src/exercises/02-tokenization/) | A single 10k BPE vocabulary balanced across India's Wikipedia article in four languages — scored on faithful units, with a one-page explainer showing why the biggest number on the page is the one we rejected, and a live in-browser encoder you can paste into. |
| 03 | [Data collection framework](src/exercises/03-data-collection-framework/) | How you decide what an India-first 40B model trains on — one interactive page, thirteen chapters: how much text, what kind, **which datasets**, how to clean it, how to tokenise it, and how you would know it worked. 145 datasets graded on five checks, of which **4 are committable today**; five data-handling invariants enforced in CI, plus a browser suite that tests the rendered page. |
| 04 | [Data cleaning & deduplication](src/exercises/04-data-cleaning-dedup/) | Eight cleaning stages over three real corpora, counting tokens with **our own Session 2 tokenizer** rather than estimating them. Deduplication by MinHash/LSH, PII masking with its false positives on show, and the finding that **three of the nine standard quality rules are not language-neutral** — applied unchanged to Indic text they delete it rather than filter it. |
| 05 | [Data mixtures & curriculum](src/exercises/05-datamixtures-and-curriculum/) | The V5 training recipe as a **[specification you can argue with](src/exercises/05-datamixtures-and-curriculum/SPEC.md)** — a defended share for every capability lane, sized against the datasets that actually exist. Summing supply from named datasets instead of quoting slot totals found a **104B hole in the STEM lane** and showed the 2% agentic lane asks **3.9× more than infinite repetition could ever be worth**. Thirteen invariants in CI, each disabled on purpose to prove it fails — and **the proxy it commits to has been run**: four arms × five seeds, two hypotheses supported and one qualified by its own declared refutation clause. |

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
uv run python -m tokenization.holdout  # why held-out cannot rank these recipes
```

The earlier round of experiments — clipped prose scored in tokens per *word* — is **retained in
full** as a second profile rather than overwritten, with its own committed corpus and a test that
regenerates its four published scores. Its finding stands on its own: representation is the
dominant lever, and byte-level BPE wastes the budget rebuilding Indic characters from UTF-8 bytes.
The two profiles are never ranked against each other; the same tokenizer reads ≈ 2.13 under one
and ≈ 0.60 under the other.

Two findings worth the detour. **Where the trainer's input is cut matters as much as the recipe:**
HuggingFace splits files into lines, so training from files means no merge may span a newline —
feed it whole documents instead and every token count drops ~0.6%. And **a score that measures only
evenness can be bought by getting worse**: one configuration reaches 35,604 against the submission's
11,251 by making English and Hindi worse until all four languages are equally mediocre, needing
3,000 more tokens for the same corpus. It is on the page, labelled as rejected. Holding text back
does not settle it either — across the five possible splits, one recipe's held-out score swings
9,421 points while the recipes' averages sit 648 apart, so that test is reported for what it cannot
do rather than used to choose.

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
stops working.

```bash
uv run python -m dataframework          # rebuild web/data.json from the data spine
uv run pytest -m "not integration"      # the invariants, and the proofs they can fail
```

> **Hosting:** deploys via the repo-wide Vercel project at `/03-data-collection-framework/`.
> **Scope:** a coursework exercise, not a proposal to anyone — see
> [`NOTICE`](src/exercises/03-data-collection-framework/NOTICE).

### 04 · Data cleaning & deduplication — what survives

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

### 05 · Data mixtures & curriculum — the recipe, and what it costs to defend it

**→ [`SPEC.md`](src/exercises/05-datamixtures-and-curriculum/SPEC.md) is the deliverable.**

The V5 training recipe: how much of each kind of data the model sees, in what order. It is written
to be argued with, so **every number in it is computed rather than typed** — the document is
generated by `uv run python -m mixture` from the same code the tests pin, and a test regenerates it
and compares byte for byte.

One decision produced everything else: **lane supply is summed from the datasets named in the
inventory, never quoted from a slot headline.** Three findings followed immediately.

| | finding | why it changes something |
| --- | --- | --- |
| **1** | The STEM lane's itemised supply is **146B**, not the **250B** quoted. No dataset carries the missing 104B. | Against a 240B demand, the quoted figure says the lane fits in one pass; the itemised figure says it needs repetition. |
| **2** | The 2% agentic lane asks 40B of a **627M** pool — **3.9× more than infinite repetition could ever be worth**. | It survives dropping every correction, so a reviewer who rejects our estimates still lands on impossible. The share stays and the gap is priced as a generation bill. |
| **3** | 60% of the long-context lane is **repo-packed code already counted under code**. | A 6% share would have double-counted 60B of corpus. It becomes a sequence-length schedule with no budget of its own. |

The spec also publishes the judgment it is *weakest* on, under a heading inviting a reviewer to
attack it: the inventory's largest Indic row is named "synthetic" and tagged as translated, and
which one wins decides which tier is fundable. Choosing the other reading moves the hole rather
than filling it, and the spec shows both.

**Thirteen invariants run in CI**, each paired with a twin proving it fails when broken — and
`tests/test_mixture_mutation.py` disables every guard in turn and requires the suite to go red, so
none of them is decorative.

**And the proxy it commits to has been run.** Four arms × five seeds over a corpus built entirely
from text this repo already tracks, scored on held-out bits per byte:

| | claim | effect | threshold | seed noise | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| H1 | a composed mixture beats crawling what is cheap | +3.00% | 2% | 1.45% | supported |
| H2 | removing the protected floor hurts Indic | +7.36% | 5% | 0.93% | supported |
| H3 | halving Indic costs Indic more than it gains others | +3.53% | 3% | 0.85% | **qualified** |

Every effect is quoted against the spread its own arm shows against itself, and **H3 is qualified
rather than supported** because its declared refutation had a second clause the first
implementation did not check. [`EXPERIMENTS.md`](src/exercises/05-datamixtures-and-curriculum/EXPERIMENTS.md)
says plainly what a 523k-token corpus does and does not license — it does not validate the mixture
at 40B, and is not offered as doing so.

> **Hosting:** live at <https://llm-pretraining-demos.vercel.app/05-datamixtures-and-curriculum/> —
> drag the lane shares and watch supply, floors and verdicts respond. Three rules live in both
> Python and JavaScript so the page can recompute per frame; a node harness diffs them and fails on
> disagreement.

```bash
uv run python -m mixture              # rebuild SPEC.md, EXPERIMENTS.md and the page's data
uv run python -m mixture.inventory    # lane supplies, itemised vs the session's headlines
uv run python -m mixture.checks       # the invariants
uv run python -m mixture.bench        # measure this machine's throughput
uv run python -m mixture.experiment   # run the four arms
```

## Development

- **Tests:** `uv run pytest` (fast unit) · `uv run pytest -m integration` (slower end-to-end). Each exercise owns its `tests/`.
- **Lint / format:** `uv run ruff check --fix .` and `uv run ruff format .`. The enforceable style spec (PEP 8/257, modern typing, line length 100) lives in `pyproject.toml`.
- **CI** (`.github/workflows/ci.yml`, on every push & PR): `uv sync --all-packages` → `ruff check` → `ruff format --check` → unit tests → integration tests → `node --check` on web JS, plus a parallel **gitleaks** secret scan.

## Adding a new exercise

Every exercise follows the same skeleton, so the repo stays predictable:

```bash
mkdir -p src/exercises/03-slug/{src,tests}
# add pyproject.toml (workspace member) and README.md
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
