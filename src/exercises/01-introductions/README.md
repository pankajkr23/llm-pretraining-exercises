# 01 · Introductions — Four Live Proofs

Session 1 assignment (see [`BRIEF.md`](./BRIEF.md)). A single static webapp that *proves* four
foundational ML claims by training tiny models **live in the browser** with TensorFlow.js —
no server, no pre-baked figures. Hit **Train** on each and watch the proof form.

| # | Claim | What you see |
| --- | --- | --- |
| 1 | **Activations exist for a reason** | Linear+sigmoid is stuck at a straight cut (~55% on two rings); one ReLU layer wraps the ring (~99%). Only the activation changed. |
| 2 | **Depth without nonlinearity is a lie** | 1 linear layer ≈ 5 stacked linear layers (identical straight cut); ReLU between the same 5 solves it. Bonus: the 5 weight matrices are multiplied to show they collapse to one 2×1 map. |
| 3 | **Embeddings learn similarity from next-token alone** | A tiny grammar trains an embedding→softmax next-token model; the 2D embeddings cluster into animals / fruits / verbs though similarity was never supplied. |
| 4 | **Data closes the generalization gap** | An over-parameterized net at n = 20 / 200 / 2000; the train→test accuracy gap is huge at 20 and shrinks as data grows. |

## Layout

```
web/
  index.html          # the page (loads TF.js from a pinned CDN)
  css/styles.css      # light/dark, responsive
  js/utils.js         # seeded RNG, data generators, Plotter, training loop
  js/demo1..4.js      # one file per proof
  js/main.js          # wires up the Train buttons
tests/                # bundle-integrity smoke test (see repo-root pytest)
```

## Preview locally

```bash
cd web
python -m http.server 8000
# open http://localhost:8000
```

## Deploy (Netlify)

The publish directory is `web/` (`netlify.toml`).

- **One-off / manual:** from this folder, `npm install` then
  `npm run deploy` (draft URL) or `npm run deploy:prod`. Run `netlify login` and link the
  site first.
- **Continuous (recommended):** connect the repo in the Netlify UI and set the site's
  **Base directory** to `src/exercises/01-introductions` — you then get preview deploys per
  PR and prod on `main`, no config needed.

Then submit the public URL as the assignment deliverable (test it in an incognito window).
