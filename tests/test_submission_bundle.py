"""The submission bundle must be trackable, and the heavy output must not be.

The requirements wants a repository containing a generated execution log and evidence bundle, so
those files have to be **in git**. Everything else a run produces — checkpoints at ~67 MiB each,
shard arrays, full token traces — must stay out of it.

That split rests on a `.gitignore` subtlety worth pinning, because getting it wrong fails
*silently*:

- `*.log` (line ~59) is a **file** pattern, so `!.../submission_artifacts/run.log` re-includes it.
- `**/artifacts/` is a **directory** pattern, and git cannot re-include a file whose parent
  directory is excluded. A negation there is inert — and `git add -A` reports success while staging
  nothing, which is the silent-failure class this repo has been bitten by before.

`git check-ignore` is the authority here, not the filesystem: none of these paths need to exist.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXERCISE = "src/exercises/06-build-training-dataset"


def _ignored(rel: str) -> bool:
    """Ask git whether it would ignore `rel`.

    Args:
        rel: A repo-relative path. It does not have to exist.

    Returns:
        True when git ignores the path.
    """
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


@pytest.mark.parametrize(
    "name",
    ["run.log", "evidence.json", "evidence.md", "performance.json", "manifests/shards.json"],
)
def test_the_submission_bundle_is_trackable(name: str) -> None:
    """Every deliverable the requirements names must be committable."""
    rel = f"{EXERCISE}/submission_artifacts/{name}"
    assert not _ignored(rel), (
        f"{rel} is ignored, so the submission bundle could not be committed. If this is `run.log`, "
        f"the `!` negation for it has been dropped from .gitignore."
    )


def test_run_log_is_only_trackable_because_of_the_negation() -> None:
    """The negation is load-bearing: without it, `*.log` would swallow the file.

    Asserted by checking a sibling `.log` that has no negation. If that one were *also* trackable
    the `*.log` rule would be gone, and this test would be passing for the wrong reason.
    """
    assert _ignored(f"{EXERCISE}/submission_artifacts/debug.log"), (
        "a sibling .log is trackable, so `*.log` no longer applies and run.log is trackable by "
        "accident rather than by the negation this test exists to protect"
    )
    assert not _ignored(f"{EXERCISE}/submission_artifacts/run.log")


@pytest.mark.parametrize(
    "name", ["run.log", "checkpoints/step-200.pt", "shards/shard-0001.bin", "traces/tokens.u8"]
)
def test_the_heavy_output_stays_out_of_git(name: str) -> None:
    """`artifacts/` is where regenerable output goes, and it must stay ignored.

    Note `run.log` is in this list on purpose: the same filename is trackable under
    `submission_artifacts/` and ignored under `artifacts/`. That asymmetry is the whole point.
    """
    assert _ignored(f"{EXERCISE}/artifacts/{name}"), (
        f"artifacts/{name} is trackable — heavy regenerable output would enter git history"
    )


def test_a_negation_under_artifacts_would_be_inert() -> None:
    """Why the bundle is not simply called `artifacts/`, pinned as a fact rather than a comment.

    git's own rule: "it is not possible to re-include a file if a parent directory of that file is
    excluded". `**/artifacts/` is a directory pattern, so no `!` under it can ever work. Anyone
    tempted to rename the bundle needs this to fail loudly rather than discover it after a run.
    """
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "**/artifacts/" in gitignore, "the directory pattern this test reasons about is gone"
    assert not any(
        line.strip().startswith("!") and "artifacts/" in line and "submission" not in line
        for line in gitignore.splitlines()
    ), (
        "a negation was added under the ignored `artifacts/` directory. It cannot work — git will "
        "not re-include a file whose parent directory is excluded — and `git add -A` will report "
        "success while staging nothing."
    )
