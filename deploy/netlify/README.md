# netlify/ — deactivated (pending decommission)

The repo's previous host. These files are **retained but inactive** — kept for reference and a clean
rollback while Vercel is proven out. Netlify is being **decommissioned, not permanently archived**:
delete this directory once Vercel is confirmed stable.

They are deactivated simply by *not* living at the paths Netlify auto-detects. Nothing here is read
by any tool in its current location.

## Original locations

| File here | Original path |
| --- | --- |
| `netlify.root.toml` | `/netlify.toml` (repo root) |
| `01-introductions.netlify.toml` | `/src/exercises/01-introductions/netlify.toml` |

The deploy scripts and dev-dependencies also remain, deactivated, in
`src/exercises/01-introductions/package.json` (renamed `deploy:netlify` / `deploy:netlify:prod`).

## To reactivate Netlify

1. Move the configs back:
   ```bash
   git mv deploy/netlify/netlify.root.toml netlify.toml
   git mv deploy/netlify/01-introductions.netlify.toml src/exercises/01-introductions/netlify.toml
   ```
2. Re-enable the site in the Netlify dashboard (reconnect the repo / unpause auto-publishing), with
   **Base directory** = `src/exercises/01-introductions` (its `netlify.toml` publishes `web/`).
3. To avoid double-deploys, pause or disconnect the Vercel project (see [`../README.md`](../README.md)).

## Decommission checklist

The connected Netlify site is **`lovely-profiterole-3afece`** (app.netlify.com). Work top to bottom;
the first step is reversible, the last ones are not.

1. **Pause builds now (reversible).** Site → *Site configuration → Build & deploy → Continuous
   deployment → Build settings* → **Stop builds**. This halts PR deploy-previews and the Netlify bot
   PR comments immediately. (Reverse with *Start builds* if you ever need to.)
2. **Verify Vercel in production.** Ship via the `Deploy to production` workflow and confirm
   `llm-pretraining-demos.vercel.app` + the `/01-introductions/` and `/02-tokenization/` routes serve
   correctly. Let it run for a few days.
3. **Disconnect Git (harder).** Same *Continuous deployment* page → *Manage repository* → **Unlink**.
   Netlify stops watching the repo entirely; the last deploy stays served.
4. **Delete the site (final).** *Site configuration* → bottom → **Delete site**.
5. **Remove the retained repo config** (once the site is gone):
   - delete this `deploy/netlify/` directory,
   - drop the `deploy:netlify` / `deploy:netlify:prod` scripts and the `@netlify/sdk` /
     `netlify-cli` devDependencies from `src/exercises/01-introductions/package.json`,
   - regenerate `package-lock.json`.

Until step 1 is done, Netlify and Vercel both build on every PR (harmless, just noisy — two preview
comments). Step 1 alone is enough to stop that.
