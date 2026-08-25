"""Warn loudly when a local-only file has gone missing from this checkout.

`notebooks/S*.ipynb` and `src/exercises/*/tools/build_notebook.py` are gitignored, which makes them
the only files in the repo with **no second copy**. Losing one is permanent in a way no other
deletion here is.

It has already happened once, and not because anyone deleted anything: after the builders were
untracked, an ordinary `git checkout main && git pull` destroyed all five. `checkout` restored them
as tracked files from the pre-merge `main`, then the fast-forward applied the commit that removed
them from the index, so git deleted the working-tree copies. They were recovered from `db9b288^`
only because that commit was still reachable.

So this is not a test of the code. It is a **tripwire on the working tree**, and it exists because
the danger is routine git operations rather than carelessness.

**On a fresh clone every file here is legitimately absent, so it skips** — it cannot run in CI and
is not meant to. Run it on the checkout that holds the files, especially after a branch switch,
pull, merge, rebase or stash.
"""

from pathlib import Path

import pytest
from _exercises import exercises_in

REPO_ROOT = Path(__file__).resolve().parents[1]


EXERCISES = exercises_in(REPO_ROOT / "src" / "exercises")

#: One notebook and one builder per exercise, by convention.
EXPECTED_NOTEBOOKS = [
    REPO_ROOT / "notebooks" / f"S{p.name[:2]}-{p.name[3:]}.ipynb" for p in EXERCISES
]
EXPECTED_BUILDERS = [p / "tools" / "build_notebook.py" for p in EXERCISES]

#: The assignment text for each exercise. Gitignored by name everywhere, so a clone has none and a
#: healthy working checkout has one per exercise.
EXPECTED_BRIEFS = [p / "BRIEF.md" for p in EXERCISES]


def _partial(paths: list[Path]) -> bool:
    """True when some but not all exist — a clone has none, a healthy checkout has all."""
    present = [p for p in paths if p.is_file()]
    return 0 < len(present) < len(paths)


def test_no_session_notebook_has_gone_missing() -> None:
    """All of them or none of them. A gap means one was destroyed on this machine."""
    present = [p for p in EXPECTED_NOTEBOOKS if p.is_file()]
    if not present:
        pytest.skip("no session notebooks here — a fresh clone has none (they are gitignored)")
    missing = [p.name for p in EXPECTED_NOTEBOOKS if not p.is_file()]
    assert not missing, (
        f"{len(present)} session notebooks are present but {missing} are gone. These are "
        f"gitignored and have no second copy. Rebuild them from the exercise's "
        f"tools/build_notebook.py before doing anything else, and do not delete anything under "
        f"notebooks/ without PK's explicit permission (see AGENTS.md)."
    )


def test_no_notebook_builder_has_gone_missing() -> None:
    """The builders rebuild a lost notebook, and nothing tracked can restore a lost builder."""
    present = [p for p in EXPECTED_BUILDERS if p.is_file()]
    if not present:
        pytest.skip("no builders here — a fresh clone has none (they are gitignored)")
    missing = [
        f"{p.parents[1].name}/tools/build_notebook.py" for p in EXPECTED_BUILDERS if not p.is_file()
    ]
    assert not missing, (
        f"{len(present)} notebook builders are present but {missing} are gone. Nothing tracked "
        f"can restore them. If they were removed by a branch switch or pull, recover with:\n"
        f'  git checkout "$(git log --all --diff-filter=D --format=%H -1 -- '
        f"'src/exercises/*/tools/build_notebook.py')^\" -- "
        f"'src/exercises/*/tools/build_notebook.py'\n"
        f"and keep a backup outside the repo (see AGENTS.md)."
    )


def test_the_tripwire_distinguishes_a_clone_from_a_loss() -> None:
    """The guard must skip on "none present" and fail on "some present".

    Without this the test would be satisfied by an empty checkout, which is the state it most needs
    to tell apart from a deletion.
    """
    assert not _partial([]), "an empty set is a clone, not a loss"
    here, gone = Path(__file__), REPO_ROOT / "does-not-exist-xyz"
    assert _partial([here, gone]), "a mixed set is a loss and must be flagged"
    assert not _partial([here]), "a complete set is healthy"


def test_a_bare_scaffold_directory_is_not_yet_an_exercise(tmp_path: Path) -> None:
    """The regression this filter exists for, pinned.

    An empty `06-build-training-dataset/` appeared before exercise 06 had any content, and both
    guards below reported a *loss* -- 5 notebooks present, 1 "gone" -- when nothing had been lost.
    A guard that cries wolf gets ignored, so the false positive is as much a defect as a miss.
    """
    root = tmp_path / "exercises"
    real = root / "01-real"
    real.mkdir(parents=True)
    (real / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (root / "06-scaffold-only").mkdir()

    found = [p.name for p in exercises_in(root)]
    assert found == ["01-real"], f"the bare scaffold was counted as an exercise: {found}"


def test_the_filter_still_counts_a_real_exercise(tmp_path: Path) -> None:
    """The twin: narrowing the filter must not narrow it to nothing.

    Without this, deleting the glob entirely would make every guard in this file vacuously pass.
    """
    root = tmp_path / "exercises"
    for name in ("01-a", "02-b", "03-c"):
        d = root / name
        d.mkdir(parents=True)
        (d / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert [p.name for p in exercises_in(root)] == ["01-a", "02-b", "03-c"]


def test_the_filter_ignores_directories_that_do_not_match_the_naming_rule(tmp_path: Path) -> None:
    """`NN-slug` is the convention; `docs/`, `common/` and `9-x` are not exercises."""
    root = tmp_path / "exercises"
    root.mkdir(parents=True)
    for name in ("common", "docs", "9-single-digit", "abc-not-numeric"):
        d = root / name
        d.mkdir()
        (d / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert exercises_in(root) == []


def test_a_file_named_like_an_exercise_is_not_an_exercise(tmp_path: Path) -> None:
    """`is_dir()` matters: a stray `07-notes.md` must not be scanned for a notebook."""
    root = tmp_path / "exercises"
    root.mkdir(parents=True)
    (root / "07-notes.md").write_text("not a directory", encoding="utf-8")

    assert exercises_in(root) == []


def test_no_brief_has_gone_missing() -> None:
    """Briefs are local-only, so nothing tracked can restore one.

    This is not hypothetical. All four of exercises 01-04's briefs were destroyed by an ordinary
    branch switch after the commit that untracked them, and were only recoverable because
    `18015b1^` was still reachable. They had existed in git once; a brief written *after* the
    untracking convention would not have that safety net at all.
    """
    present = [p for p in EXPECTED_BRIEFS if p.is_file()]
    if not present:
        pytest.skip("no briefs here — a fresh clone has none (they are gitignored)")
    missing = [f"{p.parent.name}/BRIEF.md" for p in EXPECTED_BRIEFS if not p.is_file()]
    assert not missing, (
        f"{len(present)} briefs are present but {missing} are gone. Nothing tracked can restore "
        f"a brief written since the untracking convention. If these were lost to a branch switch, "
        f"recover them with:\n"
        f"  git show 18015b1^:src/exercises/<slug>/BRIEF.md > src/exercises/<slug>/BRIEF.md\n"
        f"and keep a backup outside the repo (see AGENTS.md)."
    )
