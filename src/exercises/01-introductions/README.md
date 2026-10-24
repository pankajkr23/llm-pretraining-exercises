# 01 · Introductions — Four Live Proofs

**Notebook:** [`notebooks/01-introductions.ipynb`](../../../notebooks/01-introductions.ipynb) — run this exercise's pipeline yourself, every setting in one cell · [open in Colab](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/01-introductions.ipynb) · [all fourteen, in order](../../../notebooks/README.md)

Exercise 01 requirement. A small site that *proves* four foundational ML claims by training tiny
models **live in the browser** — no server, no pre-baked figures, and **no dependencies at all**.
Every page inlines its own CSS and JS; the neural networks (forward pass, backprop, Adam) are
hand-written in plain JavaScript, and nothing is fetched from a CDN.

**[Open the site →](https://llm-pretraining-demos.vercel.app/01-introductions/)**

Every number you see on these pages was computed in your browser, in the moment, from weights
initialised when the page loaded. Nothing is a screenshot. That is the whole point: a claim you
can re-run with different random weights is a claim you can check, and a figure in a slide deck
is one you have to take on trust.

## How to read this

| you are | start here | then |
| --- | --- | --- |
| **Meeting this for the first time** | [The four proofs](#the-four-proofs) — one row each, plain language | [What each proof actually runs](#what-each-proof-actually-runs) for the model and data behind it |
| **Changing the code** | [Layout](#layout) and [Preview locally](#preview-locally) | [Tests](#tests) — what is guarded, and what is not |
| **Deciding whether to believe it** | [What each proof actually runs](#what-each-proof-actually-runs) — every architecture and dataset stated | [What these demos cannot show](#what-these-demos-cannot-show) |
| **Running it yourself** | [`notebooks/01-introductions.ipynb`](../../../notebooks/01-introductions.ipynb) — this exercise's own pipeline, step by step, every setting in one cell | [Open in Colab](https://colab.research.google.com/github/pankajkr23/llm-pretraining-exercises/blob/main/notebooks/01-introductions.ipynb) · [the map of all fourteen](../../../notebooks/README.md) |

## Best result, and what could beat it

**Best result.** This exercise ranks nothing, so its best result is a claim that survived its own
refutation: all four proofs hold in the seeded `foundations` experiments, each asserted beside the
observation that would refute it in
[`tests/test_foundations_claims.py`](tests/test_foundations_claims.py). A linear model stays below
`0.6` test accuracy on the rings while one hidden layer with any activation clears `0.97`; a linear
stack differs from its single matrix by less than `1e-12` while a ReLU stack differs by more than
`1e-2`; trained embeddings reach nearest-neighbour purity `1.0`; and the largest dataset closes the
train-to-test gap to under `0.05`. Those are thresholds the runs must clear, not point estimates —
the pages retrain from fresh weights on every load, so their own figures move by a point or two.

**Candidates that did or could do better.**

- **Depth at a fixed budget did not win.** At about 300 parameters a deep-narrow ReLU network did
  not beat a shallow-wide one — the verdict sits inside the seed spread, against the expectation
  written before it ran ([`PROGRESS.md`](PROGRESS.md#what-is-here-and-what-is-deliberately-not)).
  Three seeds is the whole sample, so "no difference" is not yet "no effect".
- **The pages and the package are separate implementations of the same ideas.** The browser proofs
  are hand-written JavaScript and the package is torch; nothing checks that they agree on a number
  ([What these demos cannot show](#what-these-demos-cannot-show)).

**The next hypothesis.** On the spirals in `DepthConfig`, a deep-narrow ReLU network beats a
shallow-wide one by more than the seed-to-seed range once the budget is large enough to give each of
its layers real width; running `foundations.experiments.depth` across a gapless ladder of budgets
with more seeds would show where, if anywhere, `spread_verdict` turns from "within the seed spread"
to "first higher" — and a ladder on which it never turns would say depth buys nothing on this task.

## The four proofs

| Page | Claim | The interactive |
| --- | --- | --- |
| `index.html` | — | Landing page linking the four proofs. |
| `s1.html` — **The bend** | Activations exist for a reason | Rotate a 3-D neuron surface as you switch none/ReLU/tanh/GELU; then train a linear model vs a ReLU layer on two rings (~55% vs ~99%). |
| `s2.html` — **Five maps, one matrix** | Depth without nonlinearity is a lie | Watch N linear layers collapse into one matrix (gap ≈ 1e-16); flip on ReLU and it breaks. Then train 1-linear ≈ 5-linear vs 5+ReLU on the rings. |
| `s3.html` — **Meaning from company** | Embeddings learn similarity from next-token alone | A next-token model on a toy grammar; tokens migrate into animal/fruit/verb clusters. Click a token to see why (its next-token distribution). |
| `s4.html` — **Memorise, or generalise** | Data closes the generalization gap | Drag the dataset-size slider (20→2000) and watch the memorised boundary smooth out and the train→test gap close. |

## What each proof actually runs

The claims above are the headline. This is the apparatus underneath each one — the model, the data,
and what you should actually watch for. Read this before deciding whether a demo proves what it says.

### s1 · The bend — why a nonlinearity is not optional

- **The model:** a *single* neuron, `z = w₁·x₁ + w₂·x₂ + b`, with `w₁`, `w₂` and `b` on sliders.
  The surface `y = f(z)` is drawn over the `(x₁, x₂)` plane and can be dragged and scrolled.
- **The interaction:** switch `f` between **none, ReLU, tanh and GELU**. With `none`, the surface
  is a flat plane — the ghosted reference stays visible so you can see the others depart from it.
- **The consequence, trained live:** two concentric rings, which no straight line can separate.
  A linear model lands near **chance (~55%)**; the same setup with a **12-unit ReLU hidden layer**
  reaches **~99%**. Both are trained in the page, so the exact figures move a little run to run.
- **What to watch:** the linear model does not fail because it is small. It fails because its
  decision boundary is *always* a straight line, at any width.

### s2 · Five maps, one matrix — why depth needs a nonlinearity

- **The model:** a stack of **1 to 6 linear layers** (default **5**), weights randomisable.
- **The proof:** the page multiplies the weight matrices into a single matrix `M` and compares
  `M·x` against running `x` through the whole stack. The largest difference is about **1e-16** —
  float64 round-off, i.e. they are the same function. Flip the layers to **ReLU** and the collapse
  breaks immediately.
- **The consequence, trained live:** the same rings, three networks — 1 linear, 5 linear, and
  5 with ReLU. The two linear networks train to **the same** accuracy, because they *are* the same
  function; only the ReLU stack separates the rings.
- **What to watch:** "deeper" bought nothing measurable until a nonlinearity was inserted. This is
  the same claim as s1, arrived at from the opposite direction.

### s3 · Meaning from company — where embeddings come from

- **The data:** a toy grammar over three word classes — **animals, fruits, verbs**. Every sentence
  has the shape `animal · verb · (animal | fruit) · .`
- **The model:** a next-token predictor with a **2-D embedding**, so the embedding can be drawn
  directly with no projection step (no PCA or t-SNE deciding what you see).
- **The training signal:** *only* next-token prediction. Nothing tells the model that `cat` and
  `dog` are alike. They end up together because they appear in the same company, so they need the
  same predictions.
- **What to watch:** click any token on the map to see its next-token distribution beside the
  grammar's true distribution. The clustering is a *consequence* of those distributions matching,
  not a separate objective — which is the mechanism the rest of this repo's tokenizer and mixture
  work depends on.

### s4 · Memorise, or generalise — data as the regulariser

- **The model:** one over-parameterized network, **2·64·1 with ReLU**, held fixed throughout.
- **The variable:** dataset size only, at five discrete sizes — **N ∈ {20, 60, 200, 600, 2000}**.
  The same architecture is retrained at each.
- **What it shows:** at N = 20 the network drives *training* accuracy to 100% by memorising every
  point, noise included, while held-out test accuracy lags far behind. As N grows the jagged
  boundary relaxes and the train→test gap closes.
- **What to watch:** the network never changes. Only the data does. That is what makes this a
  statement about data rather than about capacity or regularisation tricks.

## Layout

```text
web/
  index.html          # landing page (hero + four cards)
  s1.html … s4.html   # one self-contained proof each (inline CSS + JS, no deps)
src/foundations/      # the four ideas as small, seeded torch experiments the notebook imports
  config.py           #   one frozen dataclass per experiment
  data.py             #   rings, blobs and the toy grammar
  models.py · train.py · experiments.py
tests/                # bundle-integrity, browser render, and the `foundations` claims
tools/build_notebook.py  # writes notebooks/01-introductions.ipynb (tracked; rebuild, never hand-edit)
package.json          # deactivated Netlify scripts, retained pending decommission
pyproject.toml        # workspace member; torch is the `train` extra
```

The pages have no build step and no Python in them. The `foundations` package is separate: it
holds the same four ideas as runnable experiments for the notebook, so the notebook imports tested
code instead of carrying its own. It does not feed the pages, and the pages do not cite it.

## Preview locally

```bash
cd web
python3 -m http.server 8000    # then open http://localhost:8000
```

Any static server works — the pages have no origin requirements beyond being served over HTTP.

## Tests

```bash
uv run pytest src/exercises/01-introductions                     # 149 tests with torch, 113 without
uv run pytest src/exercises/01-introductions -m "not integration" # 40 with torch, 4 without; no browser
```

The counts are what `uv run pytest src/exercises/01-introductions --collect-only -q` reported on
2026-10-24; that command is the authority, not this paragraph.

Three suites, and they answer different questions.

`tests/test_web_bundle.py` (**4 tests**) is bundle integrity: the landing page links all four
proofs, every referenced local asset resolves, and each proof carries its own inline `<script>`, a
`<canvas>` and a back-link to the index. It proves the site is wired together and self-contained,
and it needs no browser.

`tests/test_foundations_parts.py` and `tests/test_foundations_claims.py` (**18 tests each**) cover
the `foundations` package the notebook imports. The first checks the parts — data generators,
networks, the training loop; the second holds every claim the notebook makes at its default
settings and over several seeds, each paired with the observation that would refute it. Torch is
the `train` extra, so both **skip as whole files** without it (`uv sync --all-packages --extra
train`); in CI they run in the `train` job.

`tests/test_introductions_render.py` (**109 tests**, integration-marked) opens the real pages in
Chromium and asserts what a reader sees. It exists because **this section used to say the opposite**
— that these pages had no browser suite and were "verified by being opened and used, not by CI".
That stopped being true at `5232586` (2026-09-04), which added the suite after a page threw before
its first statement, and never updated this paragraph. A README that describes a weaker test suite
than the one that ships is the same defect as one that describes a stronger one: both mislead about
what is actually guarded.

Without Chromium the render suite **skips** rather than fails, so a fresh checkout still works.
One-time setup: `uv run playwright install chromium`.

## What these demos cannot show

- **Scale.** Every network here has tens to thousands of parameters and trains in under a second.
  The claims are about *mechanism* — a linear map cannot bend, depth without nonlinearity collapses
  — which is why they survive the scale gap. Nothing about the training dynamics of a large model
  can be read off these pages.
- **The rings are synthetic.** Two concentric rings are chosen because they are exactly what a
  straight line cannot separate. They demonstrate the limitation; they do not measure how often it
  matters on real data.
- **The grammar in s3 is a toy.** Three word classes and one sentence template. It shows the
  mechanism by which distributional similarity becomes geometric similarity, at a scale where you
  can check every token by hand. Real corpora are what exercises 02–05 are about.
- **Figures move between runs.** Weights are initialised fresh on each load, so accuracies shift
  by a percentage point or two. Any number quoted here is approximate by construction.

## Deploy (Vercel)

Served by the repo-wide Vercel project at **`/01-introductions/`** — `deploy/vercel/build.sh` copies
this `web/` into the assembled `public/`. Previews auto-deploy per PR; production is on-demand. No
per-exercise config here. See [`deploy/`](../../../deploy/) for the setup.

- **Previews (automatic):** Vercel's Git integration deploys a preview URL for every PR.
- **Production (on-demand):** run the `Deploy to production` GitHub Action (Actions tab → Run
  workflow), which is gated by the `production` environment.

The `package.json` retains deactivated **Netlify** scripts and devDependencies from the prior host,
pending decommission — see [`deploy/netlify/`](../../../deploy/netlify/). Nothing in `web/` depends
on them.
