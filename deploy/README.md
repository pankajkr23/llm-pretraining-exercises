# deploy/ — hosting

How the web demos in this repo are hosted. **Active provider: Vercel.**

## Active — Vercel (single project + path routing)

One Vercel project serves every exercise's static `web/` bundle under its slug, behind one domain:

| URL | Serves |
| --- | --- |
| `/` | landing page ([`vercel/index.html`](vercel/index.html)) |
| `/01-introductions/` | `src/exercises/01-introductions/web/` |
| `/02-tokenization/` | `src/exercises/02-tokenization/web/` |
| `/03-data-collection-framework/` | `src/exercises/03-data-collection-framework/web/` |
| `/04-data-cleaning-dedup/` | `src/exercises/04-data-cleaning-dedup/web/` |
| `/05-datamixtures-and-curriculum/` | `src/exercises/05-datamixtures-and-curriculum/web/` |
| `/06-build-training-dataset/` | `src/exercises/06-build-training-dataset/web/` |

Config is code:

- [`/vercel.json`](../vercel.json) (repo root) — `buildCommand: bash deploy/vercel/build.sh`, `outputDirectory: public`.
- [`vercel/build.sh`](vercel/build.sh) — no framework build. It assembles `public/` by copying the landing
  page, `deploy/vercel/_shared/` into `public/_shared/` (the tokens and theme script every page links
  **absolutely**, so it must exist at the site root), each exercise's `NOTICE`, and every
  `src/exercises/*/web/` into `public/<slug>/`. It then fingerprints asset references for cache-busting.
  Any exercise with a `web/` dir is picked up automatically; only the landing page's cards are
  hand-maintained, and `tests/test_deploy_registration.py` fails when one is missing.

**Deploy model — gated:**

- **Previews: automatic.** Vercel's Git integration deploys a preview URL for every PR branch.
- **Production: on-demand.** `main` does **not** auto-deploy (`vercel.json` → `git.deploymentEnabled.main: false`).
  **Two** workflows reach production, both through the reusable
  [`deploy-production.yml`](../.github/workflows/deploy-production.yml), and both gated by the
  `production` GitHub environment (add required reviewers there for an approval step):

  - [`deploy.yml`](../.github/workflows/deploy.yml) — `workflow_dispatch`, deploys `main` on demand.
    Actions tab → *Run workflow*, or `gh workflow run deploy.yml`.
  - [`release.yml`](../.github/workflows/release.yml) — fires on a `v*` tag, creates the GitHub
    Release from that `CHANGELOG.md` section, then deploys the **tagged** commit.

  So tagging a release deploys production. That is deliberate, and it is why the tag is pushed only
  after `main` is verified green on the exact commit being tagged.

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
