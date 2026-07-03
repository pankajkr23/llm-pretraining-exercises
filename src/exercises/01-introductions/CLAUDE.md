# CLAUDE.md — 01-introductions

Component-specific notes for this exercise. Repo-wide conventions: root `AGENTS.md`.

- **Static webapp**, no build step. All source is in `web/` (plain HTML/CSS/JS).
- **TensorFlow.js** is loaded from a pinned CDN in `web/index.html`
  (`@tensorflow/tfjs@4.22.0`). Bump the version there if needed; there is no bundler.
- No modules/imports — scripts load in order via `<script>` tags: `utils.js` defines the
  shared globals (RNG, data generators, `Plotter`, `trainBinary`), then `demo1..4.js`, then
  `main.js` wires the Train buttons on load.
- **Publish dir is `web/`** (`netlify.toml`). Local preview: `python -m http.server` inside `web/`.
- Deploy: `npm run deploy` (draft) / `npm run deploy:prod`. CD is Netlify Git integration —
  set the site's base directory to this folder in the Netlify UI.
- No Python here yet; `tests/` holds a bundle-integrity smoke test only.
