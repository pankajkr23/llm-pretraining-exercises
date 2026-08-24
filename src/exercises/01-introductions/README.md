# 01 · Introductions — Four Live Proofs

Session 1 assignment. A small site that *proves* four foundational ML claims by training tiny
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
tests/                # bundle-integrity smoke test (run via repo-root pytest)
package.json          # deactivated Netlify scripts, retained pending decommission
pyproject.toml        # workspace member; no Python dependencies
```

There is no build step and no Python code. `pyproject.toml` exists so the folder is a workspace
member and its tests are collected from the repository root.

## Preview locally

```bash
cd web
python3 -m http.server 8000    # then open http://localhost:8000
```

Any static server works — the pages have no origin requirements beyond being served over HTTP.

## Tests

```bash
uv run pytest src/exercises/01-introductions      # the bundle-integrity suite
```

`tests/test_web_bundle.py` asserts that the landing page links all four proofs, that every
referenced local asset resolves, and that each proof carries its own inline `<script>`, a
`<canvas>`, and a back-link to the index.

**Be clear about what that does and does not buy you.** It is a *bundle-integrity* suite: it proves
the site is wired together and self-contained. It does **not** open a browser, so it cannot see a
JavaScript error, a canvas that renders blank, or a training loop that silently diverges. Exercises
02–05 add Playwright suites that do exactly that; this one does not have one, and the honest
statement is that **these four pages are verified by being opened and used, not by CI.**

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
