# 01 · Introductions — Four Live Proofs

Session 1 assignment (see [`BRIEF.md`](./BRIEF.md)). A small site that *proves* four foundational
ML claims by training tiny models **live in the browser** — no server, no pre-baked figures, and
**no dependencies at all** (every page inlines its own CSS + JS; the neural nets are hand-written).

| Page | Claim | The interactive |
| --- | --- | --- |
| `index.html` | — | Landing page linking the four proofs. |
| `s1.html` — **The bend** | Activations exist for a reason | Rotate a 3-D neuron surface as you switch none/ReLU/tanh/GELU; then train a linear model vs a ReLU layer on two rings (~55% vs ~99%). |
| `s2.html` — **Five maps, one matrix** | Depth without nonlinearity is a lie | Watch N linear layers collapse into one matrix (gap ≈ 1e-16); flip on ReLU and it breaks. Then train 1-linear ≈ 5-linear vs 5+ReLU on the rings. |
| `s3.html` — **Meaning from company** | Embeddings learn similarity from next-token alone | A next-token model on a toy grammar; tokens migrate into animal/fruit/verb clusters. Click a token to see why (its next-token distribution). |
| `s4.html` — **Memorise, or generalise** | Data closes the generalization gap | Drag the dataset-size slider (20→2000) and watch the memorised boundary smooth out and the train→test gap close. |

## Layout

```text
web/
  index.html          # landing page (hero + four cards)
  s1.html … s4.html   # one self-contained proof each (inline CSS + JS, no deps)
tests/                # bundle-integrity smoke test (run via repo-root pytest)
```

## Preview locally

```bash
cd web
python3 -m http.server 8000    # then open http://localhost:8000
```

## Deploy (Vercel)

Served by the repo-wide Vercel project at **`/01-introductions/`** — `deploy/vercel/build.sh` copies
this `web/` into the assembled `public/`. Preview-per-PR, prod on `main`; no per-exercise config here.
See [`deploy/`](../../../deploy/) for the setup.

- **Continuous (default):** the repo's Vercel Git integration deploys automatically on PR/merge.
- **Manual (rare):** `npx vercel` (preview) / `npx vercel --prod` from the repo root.

Submit the public URL as the deliverable (test it in an incognito window first).

> Netlify was the prior host; its config is deactivated in `deploy/netlify/` (pending decommission).
