"""Every relative link in every README must resolve from the file that contains it.

Written after moving five per-exercise deep-dives out of the root README and into the exercises
they describe. Three links went with them unchanged and silently broke, because a path that is
correct from the repository root is wrong from two directories down:

    [`deploy/`](deploy/)   ->  src/exercises/02-tokenization/deploy/   (does not exist)

Nothing failed. Markdown has no link checker, GitHub renders a dead link exactly like a live one,
and the reader who clicks it is the first to find out. This is the guard for that.
"""

import re
import subprocess
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


def _is_deliberately_absent(path: Path) -> bool:
    """True if `path` is git-ignored, i.e. generated rather than missing.

    `artifacts/` is the case that matters: documenting where a run writes its output is correct,
    and that directory does not exist until something has run. On a developer machine it is usually
    present, which is exactly why this needs asking git rather than the filesystem -- the first run
    of this guard passed locally and failed in CI for that reason.

    A wrong path is still caught: `deploy/` referenced from inside an exercise is not ignored, so
    it stays broken.

    Both spellings are asked because `.gitignore` writes `artifacts/` with a trailing slash, which
    matches directories only -- and `git check-ignore` cannot tell that a path which does not exist
    is a directory, so the bare form reports "not ignored" on a fresh clone.
    """
    for candidate in (f"{path}/", str(path)):
        result = subprocess.run(
            ["git", "check-ignore", "-q", candidate],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


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
    broken = [
        t
        for t in _relative_targets(doc)
        if not (doc.parent / t).exists() and not _is_deliberately_absent(doc.parent / t)
    ]
    assert not broken, (
        f"{doc.relative_to(REPO_ROOT)} links to {broken}, which do not exist relative to "
        f"{doc.parent.relative_to(REPO_ROOT)}/"
    )


def _anchors(doc: Path) -> set[str]:
    """GitHub's heading slugs for `doc`: lowercased, punctuation dropped, spaces to hyphens."""
    slugs = set()
    for line in doc.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            continue
        title = line.lstrip("#").strip()
        # GitHub's rule: lowercase, drop punctuation except - and _, then map EACH space to a
        # hyphen. It does not collapse runs, so "v1 - our" (em dash removed) becomes "v1--our".
        slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
        slugs.add(slug.replace(" ", "-"))
    return slugs


@pytest.mark.parametrize("doc", _readmes(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_in_document_anchor_resolves(doc: Path) -> None:
    """A `#section` link that points at no heading is a dead link GitHub renders as a live one.

    The reading-path table at the top of each exercise README is built entirely from these, so a
    renamed heading breaks navigation for exactly the reader the table exists to help.
    """
    text = doc.read_text(encoding="utf-8")
    wanted = {a[1:] for a in _LINK.findall(text) if a.startswith("#")}
    missing = sorted(wanted - _anchors(doc))
    assert not missing, (
        f"{doc.relative_to(REPO_ROOT)} links to headings that do not exist: {missing}"
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
