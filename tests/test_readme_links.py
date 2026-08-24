"""Every relative link in every README must resolve from the file that contains it.

Written after moving five per-exercise deep-dives out of the root README and into the exercises
they describe. Three links went with them unchanged and silently broke, because a path that is
correct from the repository root is wrong from two directories down:

    [`deploy/`](deploy/)   ->  src/exercises/02-tokenization/deploy/   (does not exist)

Nothing failed. Markdown has no link checker, GitHub renders a dead link exactly like a live one,
and the reader who clicks it is the first to find out. This is the guard for that.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: `[text](target)` — captures the target only.
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

#: Targets that are not paths on disk and cannot be checked by opening a file.
_EXTERNAL = re.compile(r"^(https?:|mailto:|#)")


def _readmes() -> list[Path]:
    """Every tracked README: the root one plus one per exercise."""
    return [REPO_ROOT / "README.md", *sorted(REPO_ROOT.glob("src/exercises/*/README.md"))]


def _relative_targets(doc: Path) -> list[str]:
    """The link targets in `doc` that name a path, with any `#anchor` suffix dropped."""
    targets = []
    for raw in _LINK.findall(doc.read_text(encoding="utf-8")):
        target = raw.split("#", 1)[0].strip()
        if target and not _EXTERNAL.match(raw):
            targets.append(target)
    return targets


@pytest.mark.parametrize("doc", _readmes(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_relative_readme_link_resolves(doc: Path) -> None:
    """A link is checked from its own directory, which is the whole point.

    Resolving from the repository root instead would pass the exact bug this exists to catch.
    """
    broken = [t for t in _relative_targets(doc) if not (doc.parent / t).exists()]
    assert not broken, (
        f"{doc.relative_to(REPO_ROOT)} links to {broken}, which do not exist relative to "
        f"{doc.parent.relative_to(REPO_ROOT)}/"
    )


def test_the_link_check_reads_paths_relative_to_the_document() -> None:
    """The guard above is only meaningful if it fails on a root-relative path in a subdirectory.

    `deploy/` exists at the repository root, so a check that resolved from there would call this
    link fine. Asserting it is seen as broken from inside an exercise is what proves the guard
    measures the right thing.
    """
    assert (REPO_ROOT / "deploy").exists(), "fixture assumes deploy/ exists at the root"
    exercise = REPO_ROOT / "src" / "exercises" / "02-tokenization"
    assert not (exercise / "deploy").exists(), "the whole premise of this test has changed"
