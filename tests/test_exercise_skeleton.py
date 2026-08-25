"""Every exercise carries the same skeleton, and briefs are never versioned.

`AGENTS.md` has specified this since the repo began — *"identical skeleton per exercise"* — and it
was still skipped when exercise 06 was created: code was written before `CLAUDE.md`, `PROGRESS.md`,
`NOTICE` and `BRIEF.md` existed. A convention that lives only in prose is a convention that gets
skipped under momentum, so this makes it checkable.

Only the genuinely universal files are required. `DECISIONS.md`, `PROGRESS.md` and `NOTICE` are
"when the work needs them" and several exercises legitimately have none, so requiring them here
would be inventing a rule the repo does not follow.
"""

import subprocess
from pathlib import Path

import pytest
from _exercises import exercises_in

REPO_ROOT = Path(__file__).resolve().parents[1]
EXERCISES = exercises_in(REPO_ROOT / "src" / "exercises")

#: Present in every exercise, without exception.
REQUIRED = ("README.md", "CLAUDE.md", "pyproject.toml")

#: Directories a FRESH CLONE has.
#:
#: `tools/` is deliberately NOT here. For most exercises its only content is the gitignored
#: `build_notebook.py`, and git does not track empty directories — so `tools/` exists on a working
#: checkout and not in a clone. Requiring it passed locally and failed CI, which is the same
#: mistake in the same shape as requiring `artifacts/` would be: **write the guard for what a clone
#: has, not for what your machine has.**
REQUIRED_DIRS = ("tests",)


def _ids(path: Path) -> str:
    return path.name


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_the_exercise_has_the_required_files(exercise: Path) -> None:
    """`CLAUDE.md` is the one most easily forgotten — it is the only agent-facing file here."""
    missing = [name for name in REQUIRED if not (exercise / name).is_file()]
    assert not missing, (
        f"{exercise.name} is missing {missing}. AGENTS.md specifies an identical skeleton per "
        f"exercise; set the folder up before writing code, not after."
    )


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_the_exercise_has_the_required_directories(exercise: Path) -> None:
    """`tools/` holds the notebook builder, `tests/` is discovered from the repo root."""
    missing = [name for name in REQUIRED_DIRS if not (exercise / name).is_dir()]
    assert not missing, f"{exercise.name} is missing directories {missing}"


@pytest.mark.parametrize("exercise", EXERCISES, ids=_ids)
def test_no_brief_is_ever_tracked(exercise: Path) -> None:
    """A brief is the course's text and is input, never the deliverable.

    `AGENTS.md`: *"`BRIEF.md` is gitignored by name everywhere."* Checked with `git ls-files`
    rather than by looking at `.gitignore`, because a file already added to the index stays tracked
    no matter what the ignore rules say afterwards — which is exactly how this would go wrong.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", f"{exercise.relative_to(REPO_ROOT)}/BRIEF.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode != 0, (
        f"{exercise.name}/BRIEF.md is TRACKED. A brief is the course's text, not our deliverable, "
        f"and it must never be versioned. Remove it from the index with "
        f"`git rm --cached` — the file itself stays on disk."
    )


def test_the_skeleton_check_can_actually_fail() -> None:
    """The twin. An empty exercise list would make every check above vacuous."""
    assert len(EXERCISES) >= 5, f"only {len(EXERCISES)} exercises found — the scan has drifted"
    assert all((e / "README.md").is_file() for e in EXERCISES)

    missing = [n for n in REQUIRED if not (Path("/nonexistent-exercise") / n).is_file()]
    assert missing == list(REQUIRED), "the missing-file predicate does not detect an empty folder"
