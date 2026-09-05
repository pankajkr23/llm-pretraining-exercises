"""A tracked notebook must run on a machine that is not the one that built it.

`notebooks/S10-training-loop.ipynb` is the one topic notebook that ships in the repository — a
named exemption in `.gitignore`, because exercise 10's submission requires the ipynb and offers no
alternative. It is therefore the one notebook a stranger opens, and it spent its whole life
containing:

    root = pathlib.Path("/Users/<the author>/git/tsai/era5/llm-pretraining-exercises")

because its builder interpolated `EXERCISE.parents[2]` — the **build** machine's own absolute path —
into the cell at build time. It imported on exactly one computer.

**Nothing could have caught it.** `tests/test_notebook_builders.py` builds through `NOTEBOOK_OUT`
into a temporary directory and asserts on the result, so the baked-in path was correct there too;
and it skips entirely on a fresh clone, so CI never reads a notebook at all. This guard reads the
**tracked** notebooks, which a clone does have, and asks only whether the thing is portable.

It deliberately does not look for one author's home directory. The property is that no absolute
path into someone's machine survives into a shipped artefact, whoever built it and on whatever
operating system.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A POSIX home, a macOS home, a Windows drive, and a Linux home. Each names a location that exists
# on the machine that wrote it and nowhere else.
MACHINE_PATHS = re.compile(r"(?:/Users/|/home/|/root/|[A-Za-z]:\\\\)")

# `/home` and `/root` also appear in ordinary prose, and a Colab path is legitimately absolute:
# the notebook runs there and the directory really is `/content`. Only code is scanned, and only
# for a path used as a path.
ALLOWED = re.compile(r"/content(?:/|\b)")


def _tracked_notebooks() -> list[Path]:
    """Every notebook git actually carries.

    Returns:
        Paths to tracked `.ipynb` files, sorted.
    """
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.ipynb"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(ROOT / name for name in listed.stdout.split("\0") if name)


def _machine_paths_in(notebook: Path) -> list[str]:
    """Lines of code in a notebook that name a path on somebody's machine.

    Args:
        notebook: The `.ipynb` to read.

    Returns:
        The offending source lines.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    found = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            if MACHINE_PATHS.search(line) and not ALLOWED.search(line):
                found.append(line.strip())
    return found


def test_there_is_at_least_one_tracked_notebook_to_check() -> None:
    """Otherwise every assertion below is vacuously true and this file reads as coverage."""
    assert _tracked_notebooks(), (
        "no tracked notebooks — this guard is inert. If the S10 exemption was removed, remove "
        "this file too rather than leaving a check with nothing to check."
    )


@pytest.mark.parametrize("notebook", _tracked_notebooks(), ids=lambda p: p.name)
def test_no_tracked_notebook_hardcodes_a_path_on_someones_machine(notebook: Path) -> None:
    """The defect this file exists for, asserted on the artefact rather than on the builder."""
    offenders = _machine_paths_in(notebook)
    assert not offenders, (
        f"{notebook.relative_to(ROOT)} hardcodes {len(offenders)} path(s) into a specific "
        f"machine, so it cannot run anywhere else: {offenders}. Find the repository root at run "
        "time by walking up for a marker directory; never interpolate the builder's own path."
    )


def test_the_scan_can_actually_fail() -> None:
    """The twin. A guard nobody has watched fail is not a guard.

    Runs against a fabricated notebook rather than by mutating a real one — a mutation restored on
    the happy path is a mutation an early return leaves behind, and a backup written inside the
    working tree is a file `git add -A` commits.
    """
    planted = {
        "cells": [
            {"cell_type": "code", "source": ['root = pathlib.Path("/Users/someone/repo")\n']},
            {"cell_type": "markdown", "source": ["/home/ in prose must not count\n"]},
            {"cell_type": "code", "source": ['data = pathlib.Path("/content/sample_data")\n']},
        ]
    }
    scratch = ROOT / ".pytest_cache" / "planted.ipynb"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(json.dumps(planted), encoding="utf-8")
    try:
        offenders = _machine_paths_in(scratch)
    finally:
        scratch.unlink(missing_ok=True)

    assert len(offenders) == 1, f"expected exactly the /Users/ line, got {offenders}"
    assert "/Users/someone/repo" in offenders[0]
