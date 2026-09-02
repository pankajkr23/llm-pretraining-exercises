"""The backup tool, tested against the losses it exists to prevent.

`tools/backup_local_only.py` is the only thing standing between a gitignored file and a permanent
loss, so the interesting question is never "does it copy files" — it is: what does it fail to copy,
what does it copy that it must not, and can it report success while the store is wrong?

Everything here runs against a synthetic repo in `tmp_path`. A test that pointed at the real one
would risk writing to the very paths `AGENTS.md` forbids touching.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "backup_local_only.py"

sys.path.insert(0, str(REPO_ROOT / "tools"))

import backup_local_only as backup  # noqa: E402


def _make(root: Path, relative: str, text: str = "x") -> Path:
    """Create a file and its parents.

    Args:
        root: Base directory.
        relative: Path under it.
        text: Contents.

    Returns:
        The file.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A repo holding one file of every protected class.

    Returns:
        Its root.
    """
    root = tmp_path / "repo"
    for relative in (
        "notebooks/S01-introductions.ipynb",
        "src/exercises/01-introductions/tools/build_notebook.py",
        "src/exercises/01-introductions/BRIEF.md",
        "docs/BRIEF.md",
        "docs/EXPLAINER_PROMPT.md",
        "docs/EXPLAINER_PATTERN.md",
        "docs/notes/s1.md",
        "docs/notes/s1_transcript.md",
        "docs/notes/media/s1/diagram.svg",
        "TODO.md",
    ):
        _make(root, relative, f"contents of {relative}")
    return root


# --- what it must collect ----------------------------------------------------------------------


def test_it_collects_every_protected_class(fake_repo: Path) -> None:
    """One miss here is one file with no copy anywhere."""
    found = {str(p) for p in backup.collect(fake_repo)[0]}
    for expected in (
        "notebooks/S01-introductions.ipynb",
        "src/exercises/01-introductions/tools/build_notebook.py",
        "src/exercises/01-introductions/BRIEF.md",
        "docs/BRIEF.md",
        "docs/EXPLAINER_PROMPT.md",
        "docs/notes/s1.md",
        "docs/notes/media/s1/diagram.svg",
    ):
        assert expected in found, f"{expected} would not be backed up"


def test_the_notes_corpus_glob_reaches_both_depths(fake_repo: Path) -> None:
    """**The glob most likely to be silently wrong.**

    `docs/notes/**/*.md` must match `docs/notes/s1.md` — a file directly in the directory —
    as well as one nested under `media/`. Python's `**` matches zero directories, but that is worth
    an assertion rather than a memory: if it did not, the entire course corpus would be skipped and
    the tool would still report success.
    """
    found = {str(p) for p in backup.collect(fake_repo)[0]}
    assert "docs/notes/s1.md" in found, "a file directly under docs/notes/ was not matched"
    assert "docs/notes/media/s1/diagram.svg" in found, "a nested file was not matched"


def test_a_regenerable_artifact_is_not_collected(fake_repo: Path) -> None:
    """Backing up derived output teaches the reader that the backup is optional."""
    _make(fake_repo, "src/exercises/01-introductions/artifacts/plot.png", "binary-ish")
    _make(fake_repo, "data/corpus/web.jsonl", "{}")
    found = {str(p) for p in backup.collect(fake_repo)[0]}
    assert not any("artifacts/" in f or f.startswith("data/") for f in found)


# --- what it must refuse -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "deploy.env",
        "server.pem",
        "api.key",
        "credentials.json",
        "id_rsa",
        "id_ed25519.pub",
        "client_secret.txt",
    ],
)
def test_a_credential_shaped_name_is_recognised(name: str) -> None:
    """**Refusing loudly, never skipping quietly.**

    A secret copied into a backup repo outlives every decision to delete it, and the backup is the
    place nobody thinks to look. Skipping it silently would leave the operator believing the
    snapshot is complete, which is the worst of both.

    Asserted against the predicate rather than by writing `.env` to disk. The sandbox this repo
    runs under refuses to even `stat` a file with these names — correctly — so the on-disk form of
    this test cannot run, and creating credential-shaped fixtures is itself how one ends up copied
    somewhere later.
    """
    assert backup.looks_like_a_credential(Path("docs/notes") / name)


@pytest.mark.parametrize(
    "name", ["s1.md", "BRIEF.md", "build_notebook.py", "S01-introductions.ipynb", "diagram.svg"]
)
def test_an_ordinary_name_is_not_flagged(name: str) -> None:
    """**The twin.** A rule broad enough to refuse everything would look like perfect safety."""
    assert not backup.looks_like_a_credential(Path("docs/notes") / name)


def test_collect_skips_a_credential_and_names_it_rather_than_aborting(fake_repo: Path) -> None:
    """**It must skip and report, not abort — and that is a correction, not a preference.**

    The first version raised, so one innocuously-named document took the run from a hundred files
    protected to *zero*: no backup at all, plus a frightening message. The file must be excluded,
    named, and the other hundred still snapshotted.

    `deploy.env` is used because the sandbox permits it while still matching `*.env`; the
    dot-prefixed forms are covered by the predicate test above.
    """
    _make(fake_repo, "docs/notes/deploy.env", "SECRET")
    files, skipped = backup.collect(fake_repo)

    assert skipped == ["docs/notes/deploy.env"]
    assert not any("deploy.env" in str(f) for f in files), "the credential was collected anyway"
    assert len(files) >= 9, "skipping one file cost the rest their backup"


def test_the_real_corpus_is_not_swallowed_by_the_forbidden_rules(fake_repo: Path) -> None:
    """End to end: the protected set survives the credential filter."""
    assert len(backup.collect(fake_repo)[0]) >= 9, "the forbidden patterns swallowed the corpus"


# --- the store ---------------------------------------------------------------------------------


def test_a_snapshot_round_trips_byte_for_byte(fake_repo: Path, tmp_path: Path) -> None:
    """A backup that differs from the original is not a backup."""
    dest = tmp_path / "store"
    files, _ = backup.collect(fake_repo)
    backup.snapshot(fake_repo, dest, files, message="test")

    absent, differing, lost = backup.verify(fake_repo, dest, files)
    assert not absent and not differing and not lost
    for relative in files:
        assert (dest / relative).read_bytes() == (fake_repo / relative).read_bytes()


def test_verify_reports_a_file_the_store_never_received(fake_repo: Path, tmp_path: Path) -> None:
    """**The twin for verify.** A checker that cannot fail is not a checker."""
    dest = tmp_path / "store"
    files, _ = backup.collect(fake_repo)
    backup.snapshot(fake_repo, dest, files, message="test")

    _make(fake_repo, "notebooks/S02-tokenization.ipynb", "new work")
    absent, _, _ = backup.verify(fake_repo, dest, backup.collect(fake_repo)[0])
    assert "notebooks/S02-tokenization.ipynb" in absent


def test_verify_reports_a_file_that_has_changed_since_the_snapshot(
    fake_repo: Path, tmp_path: Path
) -> None:
    """The likelier loss is an overwrite, not a deletion, so staleness has to be visible."""
    dest = tmp_path / "store"
    files, _ = backup.collect(fake_repo)
    backup.snapshot(fake_repo, dest, files, message="test")

    (fake_repo / "notebooks/S01-introductions.ipynb").write_text("rebuilt", encoding="utf-8")
    _, differing, _ = backup.verify(fake_repo, dest, files)
    assert "notebooks/S01-introductions.ipynb" in differing


def test_the_store_keeps_the_previous_version_of_an_overwritten_file(
    fake_repo: Path, tmp_path: Path
) -> None:
    """**Why this is a git store and not a copy.**

    These files are regenerated constantly, so the realistic loss is a bad rebuild overwriting a
    good notebook. A plain copy would faithfully replace the good version with the broken one and
    call it a backup. Reaching the earlier version is the whole point.
    """
    dest = tmp_path / "store"
    notebook = "notebooks/S01-introductions.ipynb"

    backup.snapshot(fake_repo, dest, backup.collect(fake_repo)[0], message="good")
    (fake_repo / notebook).write_text("BROKEN REBUILD", encoding="utf-8")
    backup.snapshot(fake_repo, dest, backup.collect(fake_repo)[0], message="broken")

    log = subprocess.run(
        ["git", "-C", str(dest), "log", "--format=%H", "--", notebook],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert len(log) >= 2, "the store kept only one version, so an overwrite is unrecoverable"

    earlier = subprocess.run(
        ["git", "-C", str(dest), "show", f"{log[-1]}:{notebook}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "BROKEN" not in earlier
    assert "contents of" in earlier


def test_a_second_snapshot_with_no_changes_reports_nothing_changed(
    fake_repo: Path, tmp_path: Path
) -> None:
    """Noise in the store makes a real change harder to see in `git log`."""
    dest = tmp_path / "store"
    files, _ = backup.collect(fake_repo)
    backup.snapshot(fake_repo, dest, files, message="first")
    assert backup.snapshot(fake_repo, dest, files, message="second") == 0


# --- the tool as a command ----------------------------------------------------------------------


def _run(*args: str, home: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the tool as a command.

    Args:
        *args: Command-line arguments.
        home: When given, run with **no git identity and no guessing**. That is what the CI runner
            looked like when it exposed the swallowed `git commit` failure — and clearing the
            config alone does not reproduce it on macOS, where git invents an identity from the
            hostname and commits regardless.

    Returns:
        The finished process.
    """
    env = None
    if home is not None:
        import os

        env = {
            **os.environ,
            "HOME": str(home),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            # Clearing the config is not enough on macOS: git happily GUESSES an identity from the
            # username and hostname and commits anyway, so the failure reproduces on the Linux
            # runner and not on the developer's machine — which is precisely how the defect
            # shipped. `user.useConfigOnly` makes git refuse to guess, so the test means the same
            # thing on both.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "user.useConfigOnly",
            "GIT_CONFIG_VALUE_0": "true",
        }
    return subprocess.run(
        [sys.executable, str(TOOL), *args], capture_output=True, text=True, check=False, env=env
    )


def test_the_dry_run_writes_nothing(fake_repo: Path, tmp_path: Path) -> None:
    """A dry run that created the store would be the opposite of a dry run."""
    dest = tmp_path / "store"
    finished = _run("--dry-run", "--root", str(fake_repo), "--dest", str(dest))
    assert finished.returncode == 0, finished.stderr
    assert not dest.exists()


def test_verify_exits_non_zero_when_the_store_is_missing_files(
    fake_repo: Path, tmp_path: Path
) -> None:
    """It has to be usable as a gate, which means the exit code has to mean something.

    Pointed at a repo that genuinely **has** local-only files. Pointing it at this checkout would
    prove nothing in CI, where a clone has none and the tool correctly returns 0 with nothing to
    protect — which is how the first version of this test passed locally and failed in CI for the
    opposite reason to the one it was written for.
    """
    finished = _run("--verify", "--root", str(fake_repo), "--dest", str(tmp_path / "absent"))
    assert finished.returncode == 1, finished.stdout + finished.stderr


def test_a_snapshot_commits_on_a_machine_with_no_git_identity(
    fake_repo: Path, tmp_path: Path
) -> None:
    """**The defect CI found, pinned.**

    `git commit` exits non-zero with no `user.email` configured. The failure was swallowed by
    `check=False`, so the tool copied every file, printed success, and left a directory with **no
    commits** — losing the version history that is the entire reason for a git store rather than
    `cp -r`. The only symptom was an empty `git log` nobody ran.
    """
    dest = tmp_path / "store"
    finished = _run("--root", str(fake_repo), "--dest", str(dest), home=tmp_path / "home")
    assert finished.returncode == 0, finished.stdout + finished.stderr

    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert head.returncode == 0 and head.stdout.strip(), (
        "the store has files but no commits; every earlier version is unreachable"
    )


def test_a_store_that_cannot_commit_is_refused_rather_than_reported_as_a_success(
    fake_repo: Path, tmp_path: Path
) -> None:
    """**The twin.** If a broken git could not fail the run, the test above would prove nothing.

    A store directory whose `.git` is a file rather than a directory makes every git call fail.
    The tool must exit non-zero and say the snapshot is not safe.
    """
    dest = tmp_path / "store"
    dest.mkdir()
    (dest / ".git").write_text("not a git directory", encoding="utf-8")

    finished = _run("--root", str(fake_repo), "--dest", str(dest))
    assert finished.returncode != 0, "a store where git cannot run reported success"
    assert "NOT safe" in (finished.stdout + finished.stderr)


def test_the_default_destination_is_outside_the_repository() -> None:
    """**A backup inside the working tree dies with the working tree.**

    One of the two failure modes here is losing the checkout entirely. A store under the repo root
    would look like a backup and protect against neither that nor a `git clean`.
    """
    assert REPO_ROOT not in backup.DEFAULT_DEST.parents
    assert backup.DEFAULT_DEST != REPO_ROOT


# --- detecting a loss, not just staleness --------------------------------------------------------


def test_verify_reports_a_file_that_has_been_lost_from_the_checkout(
    fake_repo: Path, tmp_path: Path
) -> None:
    """**The defect that made `--verify` useless at the exact moment it mattered.**

    The first version asked only "is everything the checkout has also in the store?". A deleted
    file is not in the checkout, so it was never asked about — deleting all six notebooks made the
    answer *better*. And this is the command `AGENTS.md` tells you to run first after a checkout or
    pull, which is the operation class that has already destroyed these files twice.
    """
    dest = tmp_path / "store"
    files, _ = backup.collect(fake_repo)
    backup.snapshot(fake_repo, dest, files, message="test")

    (fake_repo / "notebooks/S01-introductions.ipynb").unlink()
    remaining, _ = backup.collect(fake_repo)
    _, _, lost = backup.verify(fake_repo, dest, remaining)

    assert "notebooks/S01-introductions.ipynb" in lost


def test_a_total_loss_is_reported_rather_than_looking_like_a_clone(
    fake_repo: Path, tmp_path: Path
) -> None:
    """Losing *everything* must be the loudest result, not the quietest.

    Emptying the checkout is indistinguishable from a fresh clone if you only look at the checkout.
    The store is what tells the two apart.
    """
    dest = tmp_path / "store"
    files, _ = backup.collect(fake_repo)
    backup.snapshot(fake_repo, dest, files, message="test")

    for relative in files:
        (fake_repo / relative).unlink()

    remaining, _ = backup.collect(fake_repo)
    assert remaining == [], "the fixture did not actually empty"
    _, _, lost = backup.verify(fake_repo, dest, remaining)
    assert len(lost) == len(files), f"{len(lost)} of {len(files)} losses reported"


def test_the_verify_command_exits_non_zero_on_a_loss(fake_repo: Path, tmp_path: Path) -> None:
    """And the exit code has to carry it, or the hook that runs this cannot fail."""
    dest = tmp_path / "store"
    assert _run("--root", str(fake_repo), "--dest", str(dest)).returncode == 0

    (fake_repo / "notebooks/S01-introductions.ipynb").unlink()
    finished = _run("--verify", "--root", str(fake_repo), "--dest", str(dest))

    assert finished.returncode == 1
    assert "LOST FROM THE CHECKOUT" in finished.stdout


def test_finder_metadata_is_not_stored(fake_repo: Path, tmp_path: Path) -> None:
    """`.DS_Store` is not content, and storing it makes the store and the checkout disagree.

    The directory sweep over `docs/notes/` picks it up, and once stored, any Finder visit to
    either side reads as drift — a guard that cries wolf gets ignored, which is how a real loss
    gets missed.
    """
    _make(fake_repo, "docs/notes/.DS_Store", "finder noise")
    files, _ = backup.collect(fake_repo)
    assert not any(f.name == ".DS_Store" for f in files)


def test_a_tracked_file_is_never_backed_up(fake_repo: Path, tmp_path: Path) -> None:
    """The store holds what git cannot restore. Anything else makes its contents ambiguous.

    Without this, `src/exercises/*/docs/*.md` would sweep in seven tracked documents and a reader
    could no longer tell which paths in the store git can already recover.
    """
    import subprocess

    subprocess.run(["git", "-C", str(fake_repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(fake_repo), "add", "TODO.md"], check=True)

    files, _ = backup.collect(fake_repo)
    assert Path("TODO.md") not in files, "a tracked file was collected into the store"
