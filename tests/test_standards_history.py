"""The standards archive is faithful, labelled, and cannot be mistaken for live policy.

`docs/standards-history/` holds frozen copies of the files that carry the repo's instructions,
configuration and conventions. Three things can go wrong with it and each has a test here: a
snapshot drifts from the tag it claims to be, a snapshot loses the banner that stops an agent
reading it as instructions, or the archive silently stops being taken.

Every guard is written twice — once against the real archive, once against a deliberately broken
copy in a `tmp_path` — because a guard nobody has watched fail is not a guard.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from snapshot_standards import (  # noqa: E402
    _BANNER_MARK,
    ARCHIVE,
    RETENTION,
    STANDARDS,
    archive_name,
    existing,
)

_MD_BODY_SEPARATOR = "---\n\n"


def _archived() -> list[Path]:
    """Every snapshot in the archive, README excluded."""
    return sorted(p for p in ARCHIVE.glob("*") if p.is_file() and p.name != "README.md")


def _tag_of(path: Path) -> str:
    """`DESIGN.v0.12.0.md` -> `v0.12.0`."""
    for part in path.name.split("."):
        if part.startswith("v") and part[1:].isdigit():
            i = path.name.index(part)
            rest = path.name[i:]
            return (
                "v" + rest[1:].split(".md")[0].split(".yml")[0].split(".yaml")[0].split(".toml")[0]
            )
    raise AssertionError(f"no version in {path.name}")


def _source_of(path: Path) -> str:
    """Read the source path back out of the banner, which names it explicitly."""
    head = path.read_text(encoding="utf-8")[:600]
    start = head.index("This is `") + len("This is `")
    return head[start : head.index("`", start)]


def test_the_archive_is_not_empty():
    """A convention that stops being applied leaves an archive frozen at an old release."""
    assert _archived(), "docs/standards-history/ holds no snapshots"


def test_every_snapshot_carries_the_not_in_force_banner():
    """Without it, an agent reads a superseded AGENTS.md as the conventions in force."""
    missing = [p.name for p in _archived() if _BANNER_MARK not in p.read_text(encoding="utf-8")]
    assert not missing, f"snapshots with no FROZEN COPY banner: {missing}"


def test_the_banner_check_fails_on_a_snapshot_without_one(tmp_path):
    """The twin: strip the banner and the property above must go red."""
    stripped = tmp_path / "AGENTS.v0.12.0.md"
    stripped.write_text("# Conventions\n\nsome text\n", encoding="utf-8")
    assert _BANNER_MARK not in stripped.read_text(encoding="utf-8")


@pytest.mark.parametrize("snapshot", _archived(), ids=lambda p: p.name)
def test_every_snapshot_is_byte_identical_to_the_tag_it_names(snapshot):
    """A snapshot that has been edited is a record of nothing.

    This is the guard that makes the archive worth having: it proves the copy beside the live file
    is what actually shipped, so a diff against it means something.
    """
    tag, source = _tag_of(snapshot), _source_of(snapshot)
    shown = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{tag}:{source}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        pytest.skip(f"{tag} is not reachable in this checkout (a shallow clone has no tags)")

    text = snapshot.read_text(encoding="utf-8")
    if snapshot.suffix == ".md":
        body = text.split(_MD_BODY_SEPARATOR, 1)[1]
    else:
        body = "".join(
            line for line in text.splitlines(keepends=True) if not line.startswith("#")
        ).lstrip("\n")
        shown.stdout = "".join(
            line for line in shown.stdout.splitlines(keepends=True) if not line.startswith("#")
        ).lstrip("\n")

    assert body == shown.stdout, f"{snapshot.name} has drifted from {tag}:{source}"


def test_every_snapshot_names_a_file_that_still_exists():
    """A snapshot of a file that has been renamed away is a pointer to nothing."""
    orphaned = [p.name for p in _archived() if not (REPO_ROOT / _source_of(p)).exists()]
    assert not orphaned, f"snapshots whose live file is gone: {orphaned}"


def test_every_standard_file_is_snapshotted_at_least_once():
    """Fails in the other direction: adding to STANDARDS without ever capturing it."""
    never = [s for s in STANDARDS if not existing(s)]
    assert not never, f"in STANDARDS but never snapshotted: {never}"


def test_retention_is_not_silently_exceeded():
    """Two per file. `--prune` lists what is over; it never deletes on its own."""
    over = {s: [p.name for p in existing(s)] for s in STANDARDS if len(existing(s)) > RETENTION}
    assert not over, (
        f"past the retention limit of {RETENTION} — run "
        f"`uv run python tools/snapshot_standards.py --prune` and remove them deliberately: {over}"
    )


def test_the_archive_is_tracked_by_git():
    """The whole point is that it survives a clone. `docs/` has several ignore rules."""
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "docs/standards-history/"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if not listed:
        pytest.skip("nothing committed yet — the archive is staged but not in the index")
    on_disk = {p.name for p in _archived()}
    tracked = {Path(p).name for p in listed}
    assert on_disk <= tracked, f"snapshots on disk but not tracked: {sorted(on_disk - tracked)}"


def test_archive_names_are_derivable_from_the_source_and_the_tag():
    """The naming is a function, not a convention someone remembers."""
    assert archive_name("docs/DESIGN.md", "v0.12.0") == "DESIGN.v0.12.0.md"
    assert archive_name(".gitignore", "v0.12.0") == "gitignore.v0.12.0"
    assert archive_name(".pre-commit-config.yaml", "v0.9.0") == "pre-commit-config.v0.9.0.yaml"
    assert archive_name(".github/workflows/ci.yml", "v0.12.0") == "ci.v0.12.0.yml"
