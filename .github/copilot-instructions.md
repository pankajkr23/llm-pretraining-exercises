See [`AGENTS.md`](../AGENTS.md) at the repo root — it is the single source of truth for this
repo's conventions (layout, naming, Python style, tests, CI/CD). Follow it.

Two rules are repeated here, because this file is loaded on every request and `AGENTS.md` is not,
and both are irreversible if you learn them late:

- **Never delete, move, rename or overwrite anything under `notebooks/` or any
  `src/exercises/*/tools/`, and never any `BRIEF.md`.** They are gitignored, so git cannot restore
  them and no second copy exists. This does not yield to a tidy-up or to a file that looks stale.
  If something there looks wrong, say so and stop.
- **Never commit a secret, and never name a content digest like a credential.** gitleaks gates CI
  and pre-commit; a hex digest goes in a `*_digest`/`*_hash` field, never `*_key`, `*_token`,
  `*_secret` or `*_api*`, or the scanner reads it as a leak.
