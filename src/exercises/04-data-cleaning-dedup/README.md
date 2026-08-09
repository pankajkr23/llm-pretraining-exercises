# 04 · Data Cleaning & Deduplication

**Scaffold only — no work has landed yet.** The brief is pending; see [`BRIEF.md`](./BRIEF.md).

The folder exists so the workspace member, the package, and the test directory are wired up before
the content arrives. Nothing here computes anything.

## Layout

```text
BRIEF.md              # the assignment (placeholder)
README.md             # this file
pyproject.toml        # workspace member — installs src/datacleaning
src/datacleaning/     # the Python package
  __init__.py
  config.py           # the one @dataclass of knobs (currently just paths)
tests/                # discovered by `uv run pytest` from the repo root
artifacts/            # generated outputs (git-ignored)
```

## Run it

There is nothing to run yet. The env wiring is verifiable:

```bash
uv sync --all-packages                    # installs this member into the shared .venv
uv run pytest src/exercises/04-data-cleaning-dedup   # the wiring test
```
