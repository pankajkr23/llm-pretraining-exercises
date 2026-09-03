# CLAUDE.md — 10-training-loop

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, and `REQUIREMENTS.md` is the requirements (local only, gitignored).

**Status: scaffolded.** Nothing measured yet.

## The rules this exercise adds

Replace these with the rules that are specific here — ideally the ones learned by getting something
wrong, since those are the ones worth writing down.

- **A rule that is specific to this exercise.**

## Running it

```bash
uv sync --all-packages
uv run pytest src/exercises/10-training-loop
```

Test modules are prefixed `test_trainloop_*`. pytest imports by **basename**, so a second
`test_config.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.
