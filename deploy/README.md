# deploy/ — hosting

How the web demos in this repo are hosted. **Active provider: Vercel.**

## Active — Vercel (single project + path routing)

One Vercel project serves every exercise's static `web/` bundle under its slug, behind one domain:

| URL | Serves |
| --- | --- |
| `/` | landing page ([`vercel/index.html`](vercel/index.html)) |
| `/01-introductions/` | `src/exercises/01-introductions/web/` |
| `/02-tokenization/` | `src/exercises/02-tokenization/web/` |

Config is code:

- [`/vercel.json`](../vercel.json) (repo root) — `buildCommand: bash deploy/vercel/build.sh`, `outputDirectory: public`.
- [`vercel/build.sh`](vercel/build.sh) — no framework build; just assembles `public/` by copying the landing
  page plus every `src/exercises/*/web/` into `public/<slug>/`. Any exercise with a `web/` dir is picked
  up automatically; only the landing page's cards are hand-maintained.

**Deploy model — gated:**

- **Previews: automatic.** Vercel's Git integration deploys a preview URL for every PR branch.
- **Production: on-demand.** `main` does **not** auto-deploy (`vercel.json` → `git.deploymentEnabled.main: false`).
  Production ships only when you run the **`Deploy to production`** workflow
  ([`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)) — Actions tab → *Run workflow*, or
  `gh workflow run deploy.yml`. It runs `vercel build/deploy --prod` and is gated by the `production`
  GitHub environment (add required reviewers there for an approval step).

**Required GitHub secrets** (for the deploy workflow) — set under repo *Settings → Secrets and variables → Actions*:

| Secret | Where to get it |
| --- | --- |
| `VERCEL_TOKEN` | Vercel → Account Settings → Tokens |
| `VERCEL_ORG_ID` | `.vercel/project.json` after `vercel link` (or Vercel project settings) |
| `VERCEL_PROJECT_ID` | same as above |

Manual deploy from a laptop if ever needed: `npx vercel` (preview) / `npx vercel --prod`.
`public/` and `.vercel/` are gitignored.

**Adding an exercise:** nothing required for it to be served (the build script globs `src/exercises/*/web`).
Add a card to `vercel/index.html` so it shows on the landing page.

## Decommissioned — Netlify

Netlify was the previous host. Its config is **retained but deactivated** in [`netlify/`](netlify/) so the
switch is reversible during the transition; it is slated for removal once Vercel is confirmed stable
(**decommissioned, not a permanent archive**). See [`netlify/README.md`](netlify/README.md) to reactivate.

> Operational note: moving the `netlify.toml` files here deactivates the *config*, but the Netlify
> **site** is connected via the Netlify dashboard. Pause or disconnect that site so both providers
> don't deploy on every push.
