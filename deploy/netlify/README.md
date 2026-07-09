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

## To finish decommissioning

Once Vercel is confirmed: delete this directory, drop the `deploy:netlify*` scripts and
`@netlify/sdk` / `netlify-cli` devDependencies from the exercise-01 `package.json`, and delete the
Netlify site in the dashboard.
