# CLAUDE.md — 01-introductions

Component-specific notes for this exercise. Repo-wide conventions: root `AGENTS.md`.

- **Static webapp**, no build step, in `web/`. Five **self-contained** HTML pages — each inlines
  its own CSS and JS, **zero dependencies** (no CDN, no TF.js):
  - `index.html` — landing page; links to the four proofs.
  - `s1.html` — Activations ("The bend"): interactive 3-D neuron surface + rings classifier.
  - `s2.html` — Depth ("Five maps, one matrix"): linear-collapse geometry + rings.
  - `s3.html` — Embeddings ("Meaning from company"): next-token model, emergent clusters.
  - `s4.html` — Data ("Memorise, or generalise"): dataset-size slider, generalization gap.
- The neural nets (forward pass, backprop, Adam) are **hand-written vanilla JS** inside each page —
  there is no shared module and no bundler. Edit the `<script>` block in the relevant page.
- Design system (kept consistent across pages): warm-paper palette, single teal `--accent`, serif
  display headings, mono for all numbers, generous whitespace. Light + dark via `prefers-color-scheme`.
- **Verify science headlessly** before shipping UI changes: extract the model math into a small
  Node script and check the numbers (see how S1-2/3/4 were validated), since a browser can't be driven here.
- **Publish dir is `web/`** (`netlify.toml`). Local preview: `python3 -m http.server` inside `web/`.
- Deploy: `npm run deploy` / `npm run deploy:prod`, or Netlify Git integration (base dir = this folder).
- `tests/` holds a bundle-integrity smoke test (index links every proof; each proof is self-contained).
