"""A file copied into the backup store but never committed has no history, and that is the point.

The store is a git repository rather than a directory of copies for exactly one reason, stated in
`AGENTS.md`: *"these files are regenerated constantly, so the likelier loss is a **bad overwrite**,
and a plain copy would faithfully replace the good version with the broken one."* Every earlier
version is the product. A file that git declines to track still gets copied — the bytes are on disk,
`--verify` is satisfied because they match — and it has **no history at all**. It looks backed up
and it is not.

**This happened, silently, for months.** The store inherits the user's global gitignore like any
other git repository. A global `~/.config/git/ignore` containing `**/.claude/settings.local.json` —
an entirely reasonable line to have — meant that file was copied on every run and committed on none.
`PATTERNS` listed it, the tool reported it among the files it protected, and the store held exactly
one version of it: whatever was there last. It was found only by trying to recover an earlier one
and discovering there was no history to recover from.

`snapshot()` now sets `core.excludesFile` to the null device on every run, and asserts per file that
what it copied is what git tracks. The old assertion asked only whether *any* commit existed in the
store, which is true of a store that is versioning eighty-three of eighty-four files.

**Two tests here are local-only** and skip on a clone and in CI, because the store lives outside
the repo. The twin pair does not: it builds a store in `tmp_path`, so the assertion is proved
capable of failing on every machine.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from backup_local_only import (  # noqa: E402
    DEFAULT_DEST,
    collect,
    snapshot,
)

_NO_STORE = pytest.mark.skipif(
    not (DEFAULT_DEST / ".git").is_dir(),
    reason="the backup store is outside the repo and absent here (a clone, not a loss)",
)


def _git(store: Path, *args: str) -> str:
    """Run git inside a store and return stdout."""
    return subprocess.run(
        ["git", "-C", str(store), *args], capture_output=True, text=True, check=False
    ).stdout


@_NO_STORE
def test_the_store_does_not_inherit_a_global_gitignore() -> None:
    """Without this, a user's global ignore silently un-versions files the store claims to hold.

    Asserted as a property of the store's own config rather than by checking one filename, so a
    different global rule tomorrow is covered by the same test.
    """
    configured = _git(DEFAULT_DEST, "config", "--get", "core.excludesFile").strip()
    assert configured, (
        "the store has no core.excludesFile override, so `git add` there honours the user's global "
        "gitignore and any file it matches is copied without ever being committed. Run "
        "`uv run python tools/backup_local_only.py` to repair it."
    )


@_NO_STORE
def test_every_file_the_tool_collects_is_actually_versioned() -> None:
    """The invariant the old assertion could not see: per file, not per store."""
    files, _ = collect(REPO_ROOT)
    if not files:
        pytest.skip("no local-only files here — a fresh clone, not a loss")

    # `-z`, because git octal-quotes non-ASCII paths by default and several stored filenames
    # contain them. Comparing against the quoted form reports tracked files as missing, which is a
    # false alarm that trains someone to ignore this test.
    listed = _git(DEFAULT_DEST, "ls-files", "-z")
    versioned = {name for name in listed.split("\0") if name}
    unversioned = sorted(str(f) for f in files if str(f) not in versioned)
    assert not unversioned, (
        f"{len(unversioned)} file(s) are in the store as bytes but not as history, so a bad "
        f"overwrite would be unrecoverable: {unversioned}"
    )


def test_snapshot_refuses_to_report_success_when_git_will_not_track_a_file(tmp_path) -> None:
    """The twin, and it runs everywhere — a guard nobody has watched fail is not a guard.

    `.git/info/exclude` is used rather than a global ignore because it is the residual risk after
    the fix: `core.excludesFile` disarms the *global* file, and a repository-local exclude would
    still silence `git add`. So this proves the assertion fires and documents the one route that
    can still reach it.
    """
    root, store = tmp_path / "repo", tmp_path / "store"
    (root / "keep").mkdir(parents=True)
    wanted = Path("keep/notes.txt")
    (root / wanted).write_text("the only copy\n", encoding="utf-8")

    subprocess.run(["git", "-C", str(root.parent), "init", "--quiet"], check=False)
    store.mkdir()
    subprocess.run(["git", "-C", str(store), "init", "--quiet"], check=True)
    (store / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (store / ".git" / "info" / "exclude").write_text("keep/notes.txt\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        snapshot(root, store, [wanted], message="planted")

    assert "no version history" in str(raised.value), str(raised.value)
    assert "keep/notes.txt" in str(raised.value), str(raised.value)


def test_the_twin_would_pass_without_the_planted_exclude(tmp_path) -> None:
    """The other half of the twin: the same call succeeds when git is willing to track the file.

    Without this, the test above would still pass if `snapshot()` raised for some unrelated reason,
    and the pair would prove nothing about the exclude.
    """
    root, store = tmp_path / "repo", tmp_path / "store"
    (root / "keep").mkdir(parents=True)
    wanted = Path("keep/notes.txt")
    (root / wanted).write_text("the only copy\n", encoding="utf-8")

    snapshot(root, store, [wanted], message="planted")

    listed = _git(store, "ls-files", "-z")
    assert "keep/notes.txt" in {n for n in listed.split("\0") if n}
