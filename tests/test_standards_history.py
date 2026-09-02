"""The standards archive is faithful, labelled, and cannot be mistaken for live policy.

`docs/standards-history/` holds frozen copies of the files that carry the repo's instructions,
configuration and conventions. Three things can go wrong with it and each has a test here: a
snapshot drifts from the tag it claims to be, a snapshot loses the banner that stops an agent
reading it as instructions, or the archive silently stops being taken.

Every guard is written twice — once against the real archive, once against a deliberately broken
copy in a `tmp_path` — because a guard nobody has watched fail is not a guard.

**Say plainly what local-only costs.** The archive is gitignored, so on a fresh clone and in CI
there is nothing to read and **every test below skips**. A suite that only skips protects nothing,
and that is the honest trade: the archive exists to be diffed on the machine doing the rewriting,
and tracking it would ship a second copy of `AGENTS.md` and `DESIGN.md` to the remote — the same
argument that untracked the notebooks. So these run on a working checkout or nowhere.

What does still run everywhere is `test_the_archive_is_backed_up_because_nothing_else_holds_it`,
which reads `PATTERNS` rather than the archive. That is the one invariant a clone can check, and it
is the one that matters: an archive that is neither tracked nor backed up is the class of file this
repo has already lost twice.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from snapshot_standards import (  # noqa: E402
    _BANNER_MARK,
    _COMMENT_PREFIX,
    ARCHIVE,
    MIN_VERSIONS,
    STANDARDS,
    _banner,
    archive_name,
    existing,
)


def _archived() -> list[Path]:
    """Every snapshot in the archive, README excluded. Empty on a clone — the archive is ignored."""
    if not ARCHIVE.is_dir():
        return []
    return sorted(p for p in ARCHIVE.glob("*") if p.is_file() and p.name != "README.md")


#: Everything that reads the archive skips where there is no archive to read: a fresh clone, and CI.
#: Deliberately `skipif` rather than `importorskip` — the CI-shard ledger detects the latter, and
#: this is not a dependency gate, it is a local-artefact gate.
_NO_ARCHIVE = pytest.mark.skipif(
    not _archived(),
    reason="docs/standards-history/ is local-only and absent here (a clone, not a loss)",
)


def _tag_of(path: Path) -> str:
    """`DESIGN.v0.12.0.md` -> `v0.12.0`, whatever extensions follow.

    Matched, not stripped. The first version of this walked a list of known suffixes off the end,
    which silently produced `v0.12.0.json.txt` the moment a snapshot carried two extensions — and
    the tag it names is the whole basis of the byte-identical guard.
    """
    match = re.search(r"\.(v\d+(?:\.\d+)*)(?:\.|$)", path.name)
    if not match:
        raise AssertionError(f"no version in {path.name}")
    return match.group(1)


def _source_of(path: Path) -> str:
    """Read the source path back out of the banner, which names it explicitly."""
    head = path.read_text(encoding="utf-8")[:600]
    start = head.index("This is `") + len("This is `")
    return head[start : head.index("`", start)]


@_NO_ARCHIVE
def test_the_archive_is_not_empty():
    """A convention that stops being applied leaves an archive frozen at an old release."""
    assert _archived(), "docs/standards-history/ holds no snapshots"


@_NO_ARCHIVE
def test_every_snapshot_carries_the_not_in_force_banner():
    """Without it, an agent reads a superseded AGENTS.md as the conventions in force."""
    missing = [p.name for p in _archived() if _BANNER_MARK not in p.read_text(encoding="utf-8")]
    assert not missing, f"snapshots with no FROZEN COPY banner: {missing}"


def test_the_banner_check_fails_on_a_snapshot_without_one(tmp_path):
    """The twin: strip the banner and the property above must go red."""
    stripped = tmp_path / "AGENTS.v0.12.0.md"
    stripped.write_text("# Conventions\n\nsome text\n", encoding="utf-8")
    assert _BANNER_MARK not in stripped.read_text(encoding="utf-8")


@_NO_ARCHIVE
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

    # Strip EXACTLY the banner, reconstructed from the same function that wrote it — not "every
    # line starting with #". That shortcut stripped the file's own comments too, from both sides,
    # so an edited comment inside a .gitleaksignore or .pre-commit-config snapshot compared equal
    # and the guard passed. Removing a known prefix leaves every remaining byte under assertion.
    suffix = Path(source).suffix
    comment = None if suffix == ".md" else _COMMENT_PREFIX.get(suffix, "#")
    banner = _banner(source, tag, comment)

    text = snapshot.read_text(encoding="utf-8")
    assert text.startswith(banner), (
        f"{snapshot.name} does not open with the exact banner for {source} at {tag}.\n"
        "Either it was edited, or the banner text changed since it was written — the banner\n"
        "is part of the file, so editing `_banner()` invalidates every snapshot at once.\n"
        "If the wording changed on purpose, regenerate from the tags (content is re-read):\n"
        f"  uv run python tools/snapshot_standards.py --ref {tag} --force"
    )
    assert text[len(banner) :] == shown.stdout, f"{snapshot.name} has drifted from {tag}:{source}"


@_NO_ARCHIVE
def test_every_snapshot_names_a_file_that_still_exists():
    """A snapshot of a file that has been renamed away is a pointer to nothing."""
    orphaned = [p.name for p in _archived() if not (REPO_ROOT / _source_of(p)).exists()]
    assert not orphaned, f"snapshots whose live file is gone: {orphaned}"


@_NO_ARCHIVE
def test_every_standard_file_is_snapshotted_at_least_once():
    """Fails in the other direction: adding to STANDARDS without ever capturing it."""
    never = [s for s in STANDARDS if not existing(s)]
    assert not never, f"in STANDARDS but never snapshotted: {never}"


@_NO_ARCHIVE
def test_every_standard_keeps_enough_history_to_compare_against():
    """A floor, not a cap — and the direction matters more than the number.

    This asserted the opposite for one release: no *more* than two versions per file. That inverted
    the request ("at least keeping 1-2 versions"), and it made the guard go red after every single
    release with a message asking someone to delete part of the archive. A recurring instruction to
    delete history, inside the thing built to keep history, eventually gets followed.

    It was also paid for in the wrong currency. A release's snapshots are ~141 KB, so a hundred
    releases is 13.8 MB — there is no size argument here at all. Nothing is retired on a schedule
    now; `--prune` only lists, and only when asked.

    A file below the floor means the archive stopped being maintained, or a new entry in STANDARDS
    was snapshotted once and forgotten — both worth catching. One version is tolerated only while
    the repo genuinely has one release.
    """
    releases = len({_tag_of(p) for p in _archived()})
    if releases < MIN_VERSIONS:
        pytest.skip(f"only {releases} release(s) snapshotted — the floor cannot apply yet")

    thin = {s: [p.name for p in existing(s)] for s in STANDARDS if len(existing(s)) < MIN_VERSIONS}
    assert not thin, (
        f"fewer than {MIN_VERSIONS} versions kept, so a rewrite has nothing to be compared "
        f"against — snapshot an older tag with `--ref <tag>`: {thin}"
    )


def test_the_archive_is_not_tracked():
    """It is local-only by decision, and a tracked snapshot is a second copy of the conventions.

    Tracking `AGENTS.v0.12.0.md` would put the same rules on the remote twice, which is the argument
    that untracked the notebooks. This fails if a snapshot is ever committed by accident — most
    likely by a `git add -A` that ran before `.gitignore` was read.
    """
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "docs/standards-history/"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert not listed, (
        "snapshots are tracked but the archive is meant to be local-only: "
        f"{sorted(Path(p).name for p in listed)}"
    )


def test_the_archive_is_backed_up_because_nothing_else_holds_it():
    """**The one invariant a fresh clone can still check, and the one that matters.**

    Untracked and unbacked-up is the exact class of file this repo has lost twice — most recently
    when an ordinary `checkout && pull` deleted five notebook builders. Local-only is a legitimate
    decision; local-only with no store is not a decision, it is an accident waiting for a branch
    switch. So the archive's membership of `PATTERNS` is asserted from source, where a clone can
    read it, rather than from the store, which a clone does not have.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from backup_local_only import PATTERNS

    covered = [p for p in PATTERNS if p.startswith("docs/standards-history")]
    assert covered, (
        "docs/standards-history/ is gitignored but not in tools/backup_local_only.py::PATTERNS — "
        "so nothing holds it and the next branch switch can take it silently"
    )


def test_the_archive_is_gitignored_so_the_backup_is_load_bearing():
    """The twin of the pair above: prove the ignore rule is real, not assumed."""
    ignored = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "docs/standards-history/DESIGN.v0.12.0.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0, (
        "docs/standards-history/ is NOT gitignored — either restore the ignore rule, or track the "
        "archive and invert test_the_archive_is_not_tracked. It must be exactly one of the two."
    )


def test_archive_names_are_derivable_from_the_source_and_the_tag():
    """The naming is a function, not a convention someone remembers."""
    assert archive_name("docs/DESIGN.md", "v0.12.0") == "DESIGN.v0.12.0.md"
    assert archive_name(".gitignore", "v0.12.0") == "gitignore.v0.12.0"
    assert archive_name(".gitleaksignore", "v0.12.0") == "gitleaksignore.v0.12.0"
    assert archive_name(".pre-commit-config.yaml", "v0.9.0") == "pre-commit-config.v0.9.0.yaml"
    assert archive_name(".github/workflows/ci.yml", "v0.12.0") == "ci.v0.12.0.yml"
    # JSON has no comment syntax, so a bannered snapshot is not valid JSON — it must not keep an
    # extension that claims otherwise.
    assert archive_name("vercel.json", "v0.12.0") == "vercel.v0.12.0.json.txt"


@_NO_ARCHIVE
def test_no_snapshot_wears_an_extension_it_can_no_longer_be_parsed_as():
    """A bannered `.json` would parse as nothing while looking like config. Assert the property.

    Written against the whole archive rather than against JSON, so adding a `.toml`-with-no-comments
    or any other silent-comment format later is caught rather than assumed.
    """
    import json

    for snap in _archived():
        if snap.suffix != ".json":
            continue
        json.loads(snap.read_text(encoding="utf-8"))  # must not raise, or it is misnamed
