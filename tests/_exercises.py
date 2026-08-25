"""Shared helper: which directories under `src/exercises/` are actually exercises.

Not a test module. It lives here because two repo-wide guards need the same answer, and importing
one test module from another depends on pytest's `sys.path` insertion, which is fragile.
"""

from pathlib import Path


def exercises_in(root: Path) -> list[Path]:
    """The exercise directories under `root`, in name order.

    An exercise is a directory matching `NN-slug` that is a **workspace member** — it has a
    `pyproject.toml`. A bare `NN-slug/` directory is a scaffold someone just created, not an
    exercise yet.

    Globbing on the name alone made both notebook guards report a *loss* the moment an empty
    `06-build-training-dataset/` appeared, when nothing had been lost. A guard that cries wolf is a
    guard people start ignoring, so the false positive is as much a defect as a miss.

    Args:
        root: A `src/exercises` directory to scan.

    Returns:
        The matching directories, sorted by name.

    Note: `is_dir()` is belt-and-braces. The `pyproject.toml` check already excludes a regular
    file, because a file has no children -- so mutating `is_dir()` away does not turn any test red.
    It is kept because it states the intent, not because it is load-bearing. Do not read the
    file-named-like-an-exercise test as covering it.
    """
    return sorted(
        p for p in root.glob("[0-9][0-9]-*") if p.is_dir() and (p / "pyproject.toml").is_file()
    )
