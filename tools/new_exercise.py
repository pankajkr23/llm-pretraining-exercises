r"""Scaffold a new exercise, and register it everywhere registration is not automatic.

    uv run python tools/new_exercise.py 09 loss-functions-output-heads \\
        --title "Loss functions and output heads" \\
        --package lossheads \\
        --summary "What the model is actually scored on, and why the head is where it shows."

**Why this exists.** Setting an exercise up by hand takes about forty minutes and gets a different
subset of the conventions right each time. `AGENTS.md` is explicit that the folder is set up
*before* the code — *"The skeleton is not paperwork to backfill"* — and exercise 06 was still built
without a `CLAUDE.md`, `PROGRESS.md`, `NOTICE` or `BRIEF.md`, "because a convention that lives only
in prose gets skipped under momentum". This script is that convention as code.

**The sequencing fact it exists to get right.** `tests/_exercises.py::exercises_in` only counts a
directory that has a `pyproject.toml`. Until that file lands, a new exercise is invisible to every
guard; the moment it lands, six test families apply at once — including three that check for
**gitignored** files a fresh clone will never have. So everything is written in one pass, or the
suite goes red locally with no obvious cause.

**What it deliberately does NOT do.** The landing card in `deploy/vercel/index.html` and the
`SPINE_ENFORCED` entry in `tests/test_page_spine.py` are *forbidden* until `web/` exists: both
guards fail in **two** directions, so a premature entry is exactly as red as a missing one. The
script prints them as deferred steps instead of guessing.

This file is TRACKED, unlike `tools/build_notebook.py`. It is not course material in another form;
it is repo infrastructure, and `tests/test_new_exercise.py` runs it for real.
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXERCISES = REPO / "src" / "exercises"
CI = REPO / ".github" / "workflows" / "ci.yml"
ROOT_README = REPO / "README.md"

#: The shard a new exercise joins. `ci.yml`'s matrix enumerates paths explicitly rather than
#: globbing, so a new exercise belongs to no shard until it is added here — and
#: `tests/test_ci_shards_cover_everything.py` requires every test file to be in exactly one.
DEFAULT_SHARD = "rest"


@dataclass(frozen=True)
class Spec:
    """Everything the templates need.

    Attributes:
        number: Zero-padded exercise number, e.g. `"09"`.
        slug: Lowercase hyphenated name, e.g. `"loss-functions-output-heads"`.
        title: Human title for headings.
        package: Bare import name for `src/<package>/`, e.g. `"lossheads"`.
        summary: One sentence for the root README row.
        session: Session number for the notebook name; defaults to `number`.
    """

    number: str
    slug: str
    title: str
    package: str
    summary: str
    session: str

    @property
    def dirname(self) -> str:
        """`09-loss-functions-output-heads` — the directory name and the public URL slug."""
        return f"{self.number}-{self.slug}"

    @property
    def root(self) -> Path:
        """Where the exercise is written."""
        return EXERCISES / self.dirname

    @property
    def notebook(self) -> str:
        """`S09-loss-functions-output-heads` — the session notebook's stem."""
        return f"S{self.session}-{self.slug}"


# --------------------------------------------------------------------------------- templates
#
# Each template is written to satisfy a specific guard, and says which. Editing one without
# checking its guard is how a generator drifts from the conventions it exists to encode.


def readme(spec: Spec) -> str:
    """`README.md`.

    Satisfies `tests/test_readme_structure.py`, which requires a heading **exactly**
    `## How to read this` *followed by another `## ` heading*, containing the substrings
    `first time`, `changing the code` and `believe it`; a ```` ```bash ```` block anywhere; and a
    heading matching one of *cannot tell you / cannot show / cannot establish*.
    """
    return f"""# {spec.number} · {spec.title}

**One sentence saying what this exercise establishes.** Replace this — it is the first thing a
reader sees and the last thing anyone remembers to write.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is), which should explain the
  problem before any of the machinery.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it).
- **Deciding whether to believe it** — go to [The evidence](#the-evidence), then
  [What this cannot establish](#what-this-cannot-establish).

## What this is

Replace this section with the problem, in plain words, before any notation.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | every dimension this exercise measures against, in one dataclass |

## Run it

```bash
uv sync --all-packages
uv run pytest src/exercises/{spec.dirname}
```

## The evidence

Replace this with what was measured, how, and against what noise floor.

## What this cannot establish

Replace this. `AGENTS.md` requires it and it is not a formality: state the scale, what the
measurement was blind to, and which claims are read from sources rather than reproduced here.
"""


def claude_md(spec: Spec) -> str:
    """`CLAUDE.md` — required by `tests/test_exercise_skeleton.py`."""
    return f"""# CLAUDE.md — {spec.dirname}

Component notes. Repo-wide conventions: root `AGENTS.md`. The reasoning is `DECISIONS.md`, the
running log is `PROGRESS.md`, and `BRIEF.md` is the assignment (local only, gitignored).

**Status: scaffolded.** Nothing measured yet.

## The rules this exercise adds

Replace these with the rules that are specific here — ideally the ones learned by getting something
wrong, since those are the ones worth writing down.

- **A rule that is specific to this exercise.**

## Running it

```bash
uv sync --all-packages
uv run pytest src/exercises/{spec.dirname}
```

Test modules are prefixed `test_{spec.package}_*`. pytest imports by **basename**, so a second
`test_config.py` anywhere in the repo would abort collection rather than fail a test;
`tests/test_module_names.py` enforces this repo-wide.
"""


def notice(spec: Spec) -> str:
    """`NOTICE` — served beside the page by `deploy/vercel/build.sh` when a `web/` exists."""
    return f"""NOTICE — {spec.dirname}

WHAT THIS IS

A coursework exercise. It is a study, not a product, not a proposal to any
organisation, and not infrastructure anybody operates.

NO AFFILIATION

This work is not affiliated with, endorsed by, sponsored by, or produced on
behalf of any company, university, research institute or funding programme.
Every organisation and paper named here is named because their public work is
the evidence being examined. Naming them is citation, not association.

WHAT IS MEASURED AND WHAT IS NOT

Replace this section. State plainly which numbers were measured here, at what
scale, and which are read from a source and attributed to it. The distinction
is the whole point of this file.
"""


def progress(spec: Spec) -> str:
    """`PROGRESS.md` — the running log."""
    return f"""# PROGRESS — Session {spec.session}

A running log of what was built, what was measured, what changed and what is still open. Written so
the work can be picked up cold. Newest entries at the top of each section.

**Where the work lives:** on a branch, not yet merged. This file does not name branch or PR numbers
— `git log` and `gh pr list` answer that correctly and a markdown file goes stale.

**Deliverable shape — read this before calling the session done.** Check the submission platform's
own field list, not `BRIEF.md`, which can be truncated. Record the required *shape* here.

---

## Open items — for review

| # | item | status | note |
| --- | --- | --- | --- |
| O1 | **Scaffold** | **done** | Created by `tools/new_exercise.py`. |

---

## Change log

### (dated entry) — scaffolded

- Exercise created from the skeleton, registered in the `{DEFAULT_SHARD}` integration shard and the
  root README table.
"""


def decisions(spec: Spec) -> str:
    """`DECISIONS.md` — long reasoning, so the README can stay a guide."""
    return f"""# DECISIONS — {spec.dirname}

Why this exercise is shaped the way it is, and what would overturn each choice.

---

## D1 · (the first real decision)

**Decision.** Replace this.

**Why.** Replace this.

**What would overturn it.** Replace this. A decision with no stated falsifier is a preference.
"""


def pyproject(spec: Spec) -> str:
    """The workspace member. The root glob `src/exercises/[0-9][0-9]-*` picks it up automatically.

    torch, if ever needed, goes behind `[project.optional-dependencies] train` — and then the file
    must also be added to the `train` job's explicit list in `ci.yml` and to
    `OPTIONAL_DEPENDENCY_GATES`, or its tests run nowhere while CI stays green.
    """
    return f"""[project]
name = "exercise-{spec.number}-{spec.slug}"
version = "0.1.0"
description = "Session {spec.session} — {spec.title}"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{spec.package}"]
"""


def package_init(spec: Spec) -> str:
    """The package docstring — what the exercise establishes and how its modules divide the work."""
    return f'''"""{spec.title}.

Replace this docstring with what the exercise establishes and how its modules divide the work.
"""
'''


def config_module(spec: Spec) -> str:
    """`AGENTS.md`: one `config.py` dataclass per exercise."""
    return '''"""Every dimension this exercise measures against, in one place.

`AGENTS.md` asks for one `config.py` dataclass per exercise. Recording the configuration here
rather than inlining it is what makes "we reproduce the published number" checkable: change a field
and the test that reproduces it fails.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Replace with the dimensions this exercise is measured at.

    Attributes:
        source: Where these values come from, so a reader can check them.
    """

    source: str = "replace with the document these values are taken from"
'''


def smoke_test(spec: Spec) -> str:
    """A first test, so `tests/` is neither empty nor untracked (git stores no empty directory)."""
    return f'''"""The exercise imports and its configuration is readable.

Replace this with real guards. It exists so `tests/` is tracked from the first commit — git stores
no empty directory, and `tests/test_exercise_skeleton.py` requires the directory to exist.
"""

from {spec.package}.config import Config


def test_the_package_imports_and_its_config_names_its_source() -> None:
    """A config whose values came from nowhere checkable is the failure this repo pays for most."""
    config = Config()
    assert config.source, "Config.source must say where its numbers come from"
'''


def notebook_builder(spec: Spec) -> str:
    """`tools/build_notebook.py` — **gitignored**, and required by two local-only guards.

    `tests/test_notebook_builders.py` requires the source to contain the literal `'clone'` (the
    clone is spawned as an argument list, so `"git clone"` never appears as one string) and
    `IN_COLAB`, and requires `NOTEBOOK_OUT` to be treated as a full output **file** path.
    """
    return f'''"""Build `notebooks/{spec.notebook}.ipynb`.

    uv run python src/exercises/{spec.dirname}/tools/build_notebook.py

**Local only — this file is gitignored, and so is the notebook it writes.** A generator is the
notebook in another form, so tracking it would keep the same course material in the repo as Python.
Back both up outside the repo; nothing tracked can rebuild them.

**The notebook imports the package. It never re-implements it.**
"""

import json
import os
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
REPO = EXERCISE.parents[2]
SLUG = "{spec.notebook}"

#: `NOTEBOOK_OUT` is the full output path, not a directory — `tests/test_notebook_builders.py`
#: passes a file. It redirects the build so a test never overwrites the copy a developer has open.
OUT = Path(os.environ.get("NOTEBOOK_OUT") or (REPO / "notebooks" / f"{{SLUG}}.ipynb"))

REPO_URL = "https://github.com/pankajkr23/llm-pretraining-exercises"
_next_id = 0


def _cell_id() -> str:
    """Deterministic cell ids, so a rebuild does not diff every cell."""
    global _next_id
    _next_id += 1
    return f"{spec.number}-{{_next_id:02d}}"


def md(text: str) -> dict:
    """A markdown cell."""
    return {{
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {{}},
        "source": text.splitlines(keepends=True),
    }}


def code(text: str) -> dict:
    """A code cell with no stored output — outputs are stripped, always."""
    return {{
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {{}},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }}


def cells() -> list[dict]:
    """The notebook, in reading order."""
    return [
        md("# Session {spec.session} — {spec.title}\\n\\nReplace this."),
        code(
            f"""# Colab: clone the repo and install this exercise. Locally this is a no-op.
import sys, subprocess, pathlib

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not pathlib.Path("llm-pretraining-exercises").exists():
        subprocess.run(['git', 'clone', '--depth', '1', '{{REPO_URL}}'], check=True)
    root = pathlib.Path("llm-pretraining-exercises")
    subprocess.run(
        ['pip', '-q', 'install', '-e', str(root / 'src/exercises/{spec.dirname}')], check=True
    )
else:
    root = pathlib.Path("{{REPO}}")
    sys.path.insert(0, str(root / "src/exercises/{spec.dirname}/src"))

print("repo root:", root)"""
        ),
        code("from {spec.package}.config import Config\\n\\nprint(Config())"),
    ]


def main() -> int:
    """Write the notebook and report what was written."""
    notebook = {{
        "cells": cells(),
        "metadata": {{
            "kernelspec": {{"display_name": "Python 3", "language": "python", "name": "python3"}},
            "language_info": {{"name": "python", "version": "3.12"}},
        }},
        "nbformat": 4,
        "nbformat_minor": 5,
    }}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\\n", encoding="utf-8")
    print(f"wrote {{OUT}}  ({{len(notebook['cells'])}} cells, outputs stripped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def brief(spec: Spec) -> str:
    """`BRIEF.md` — **gitignored**, seeded from the session's assignment when one is present."""
    header = f"""# BRIEF — Session {spec.session}

**Local only. Never tracked.** `BRIEF.md` is gitignored everywhere in this repo: a brief is the
course's text and is *input* for whoever builds the exercise, not our deliverable. Never link to it
from a tracked file — the link resolves on a working checkout and 404s for everyone else.

"""
    assignment = REPO / "docs" / "sessions" / f"s{int(spec.session)}_assignment.md"
    if assignment.is_file():
        body = re.sub(r"!\[\]\([^)]*\)", "", assignment.read_text(encoding="utf-8"))
        return f"{header}Source: `docs/sessions/{assignment.name}`.\n\n---\n\n{body.strip()}\n"
    return (
        f"{header}No `docs/sessions/s{int(spec.session)}_assignment.md` was found when this was "
        f"scaffolded. Paste the assignment here.\n"
    )


# ------------------------------------------------------------------------------ registration


def register_ci_shard(spec: Spec, shard: str = DEFAULT_SHARD, apply: bool = True) -> str:
    """Add the exercise to a CI integration shard's explicit path list.

    Required **always**. `ci.yml`'s matrix enumerates paths rather than globbing, and
    `tests/test_ci_shards_cover_everything.py` asserts every test file is owned by exactly one
    shard — so without this, every test in the new exercise runs nowhere.
    """
    text = CI.read_text(encoding="utf-8")
    target = f"src/exercises/{spec.dirname}"
    if target in text:
        return "already in ci.yml"

    # Insert before the shard's trailing `tests` entry, preserving its indentation.
    pattern = re.compile(rf"(- name: {shard}\n\s+paths: >-\n)((?:\s+\S+\n)+)")
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"could not find the {shard!r} shard in {CI}; add the path by hand")
    block = match.group(2)
    lines = [line for line in block.splitlines() if line.strip()]
    indent = re.match(r"\s*", lines[0]).group(0)

    # Insert BEFORE the shard's trailing `tests` entry rather than after it. The repo's `ci.yml`
    # lists the exercises in order and keeps `tests` last; appending would put a new exercise below
    # it, which still runs but reads as an accident and makes the next insertion ambiguous.
    entry = f"{indent}{target}"
    tail = [line for line in lines if line.strip() == "tests"]
    if tail:
        at = lines.index(tail[0])
        lines.insert(at, entry)
    else:
        lines.append(entry)

    if apply:
        CI.write_text(text.replace(block, "\n".join(lines) + "\n", 1), encoding="utf-8")
    return f"added to the {shard!r} integration shard"


def register_readme_row(spec: Spec, apply: bool = True) -> str:
    """Add a row to the root README's exercise table.

    Hand-maintained by design: `AGENTS.md` keeps the root a *map*, one row per exercise, and forbids
    a per-exercise section there. The row must start with the literal `| NN ` — that prefix is how
    `tests/test_doc_counts_match.py` finds it.
    """
    text = ROOT_README.read_text(encoding="utf-8")
    if f"| {spec.number} |" in text:
        return "already in the root README"
    row = (
        f"| {spec.number} | [{spec.title}](src/exercises/{spec.dirname}/) | {spec.summary} "
        f"[The argument, the evidence and its limits](src/exercises/{spec.dirname}/README.md). "
        f"*Scaffolded.* |\n"
    )
    anchor = "\nMore exercises are added each week."
    if anchor not in text:
        raise SystemExit(
            "could not find the exercise table in the root README; add the row by hand"
        )
    if apply:
        ROOT_README.write_text(text.replace(anchor, f"\n{row}{anchor}", 1), encoding="utf-8")
    return "added a row to the root README table"


# ------------------------------------------------------------------------------------ driver


def files(spec: Spec) -> dict[Path, str]:
    """Every file to write, mapped to its content.

    The three gitignored ones are written **with** the rest, not afterwards: the moment
    `pyproject.toml` exists, `tests/test_local_only_files_present.py` and
    `tests/test_notebook_builders.py` start expecting a brief, a builder and a notebook for this
    exercise, and fail locally until they are there.
    """
    root = spec.root
    return {
        root / "README.md": readme(spec),
        root / "CLAUDE.md": claude_md(spec),
        root / "NOTICE": notice(spec),
        root / "PROGRESS.md": progress(spec),
        root / "DECISIONS.md": decisions(spec),
        root / "pyproject.toml": pyproject(spec),
        root / "src" / spec.package / "__init__.py": package_init(spec),
        root / "src" / spec.package / "config.py": config_module(spec),
        root / "tests" / f"test_{spec.package}_smoke.py": smoke_test(spec),
        root / "tools" / "build_notebook.py": notebook_builder(spec),  # gitignored
        root / "BRIEF.md": brief(spec),  # gitignored
    }


def main(argv: list[str] | None = None) -> int:
    """Create the exercise and report exactly what was and was not done."""
    parser = argparse.ArgumentParser(
        description="Scaffold a new exercise and register it where registration is manual.",
    )
    parser.add_argument("number", help="zero-padded exercise number, e.g. 09")
    parser.add_argument("slug", help="lowercase hyphenated name, e.g. loss-functions-output-heads")
    parser.add_argument("--title", required=True, help="human title for headings")
    parser.add_argument("--package", required=True, help="bare import name for src/<package>/")
    parser.add_argument("--summary", default="Replace this summary.", help="root README row text")
    parser.add_argument("--session", help="session number for the notebook; defaults to `number`")
    parser.add_argument("--shard", default=DEFAULT_SHARD, help="CI integration shard to join")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"\d\d", args.number):
        raise SystemExit(f"number must be two digits, got {args.number!r}")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", args.slug):
        raise SystemExit(f"slug must be lowercase alphanumeric with hyphens, got {args.slug!r}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.package):
        raise SystemExit(f"package must be a bare python identifier, got {args.package!r}")

    spec = Spec(
        number=args.number,
        slug=args.slug,
        title=args.title,
        package=args.package,
        summary=args.summary,
        session=args.session or args.number,
    )
    if spec.root.exists() and any(spec.root.iterdir()):
        raise SystemExit(f"{spec.root} already exists and is not empty")

    plan = files(spec)
    apply = not args.dry_run
    for path, content in plan.items():
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"  {'would write' if args.dry_run else 'wrote'}  {path.relative_to(REPO)}")

    print()
    print(f"  {register_ci_shard(spec, args.shard, apply)}")
    print(f"  {register_readme_row(spec, apply)}")

    if apply:
        builder = spec.root / "tools" / "build_notebook.py"
        result = subprocess.run([sys.executable, str(builder)], capture_output=True, text=True)
        print(f"  {result.stdout.strip() or result.stderr.strip()}")

    print(
        f"""
Still yours to do:

  1. Fill the templates. Every one has a "Replace this" in it; `git grep -n "Replace this"
     src/exercises/{spec.dirname}` finds them all.
  2. `uv sync --all-packages` — the workspace glob picks the member up on its own.
  3. Back up the three gitignored files: BRIEF.md, tools/build_notebook.py and the notebook.
     `uv run python tools/backup_local_only.py`

Deferred until `web/` exists, and FORBIDDEN before then — both guards fail in two directions,
so a premature entry is exactly as red as a missing one:

  * a landing card in deploy/vercel/index.html   (tests/test_deploy_registration.py)
  * an entry in SPINE_ENFORCED or SPINE_EXEMPT   (tests/test_page_spine.py)

Verify with:

  uv run pytest src/exercises/{spec.dirname} tests
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
