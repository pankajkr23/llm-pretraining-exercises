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
- **Design system:** follows the repo-wide `docs/DESIGN.md` (Apple-style — cool-gray/black
  surfaces, single blue `--accent`, system sans, soft-shadow panels; light + dark via
  `prefers-color-scheme`). Each page reuses those exact token names in its inlined `:root`. Data
  plots use the `docs/DESIGN.md` data-viz hues (blue/orange/green/purple), not UI chrome colours.
- **Navigation & voice:** `index.html` has a `← Back` pill to the site root; each proof links
  `← the four proofs` back to the index. Footers are short, blog-style captions written for a
  general reader (see `docs/DESIGN.md` › Copy & tone).
- **Motion:** canvas state changes animate. `s2.html`'s linear↔ReLU toggle morphs the grid with an
  eased transition (framing held constant so panels don't resize); `render()` draws instantly,
  `transition()` animates — mirror that split if you add another toggle.
- **Non-ASCII:** these pages use `—`, `→`, `·`, subscripts, `2×2`. Edit with Edit/Write only —
  never `perl -0pi`/`sed` with wide-char escapes (it mojibakes UTF-8).
- **Verify science headlessly** before shipping UI changes: extract the model math into a small
  Node script and check the numbers (see how S1-2/3/4 were validated), since a browser can't be driven here.
- **`web/` is served at `/01-introductions/`** by the repo-wide Vercel project (`deploy/vercel/build.sh` → `public/`). Local preview: `python3 -m http.server` inside `web/`.
- Deploy: PR previews auto (Vercel Git integration); production is on-demand via the `Deploy to production` workflow (`.github/workflows/deploy.yml`). Netlify config is deactivated in `deploy/netlify/` — deploy scripts renamed `deploy:netlify*`, pending decommission.
- `tests/` holds a bundle-integrity smoke test (index links every proof; each proof is self-contained).
