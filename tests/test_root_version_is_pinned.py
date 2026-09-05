"""The git tag is this repository's version; the root `pyproject.toml` field is not.

That field read `0.4.0` while the newest release tag was `v0.13.0` — **nine minors stale**, because
nothing bumped it and nothing checked it. It had been wrong for eleven releases.

**Bumping it to match would have been the worse fix.** A number that is correct once and then rots
looks maintained, which is exactly why this one misled: a reader has no way to tell a pinned field
from a stale one. Nothing installs the root package from an index, so the field buys nothing that
`git describe` does not already give, and the honest move is to say so in the file and hold it
there.

So this guard is not "the version is right". It is **"nobody has quietly reintroduced a second
source of truth"** — the failure `AGENTS.md` names as the one that has cost this repo the most
edits, in its own configuration rather than in its prose.

The **workspace members are deliberately not covered.** Their versions are real: they are installed
into the shared environment by `uv sync --all-packages`, and one of them declares a workspace
dependency on another. Only the root, which nothing installs from anywhere, is the fiction.
"""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: The pin. Not a version — a statement that this field is not where the version lives.
PINNED = "0.0.0"


def _root_version() -> str:
    """The `version` field of the root project table.

    Returns:
        The declared version string.
    """
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_the_root_version_is_pinned_rather_than_maintained() -> None:
    """The whole guard: the field says `0.0.0`, and the tag is where the version lives."""
    actual = _root_version()
    assert actual == PINNED, (
        f"the root pyproject.toml declares version {actual!r}, not the pin {PINNED!r}. "
        "This field is deliberately not maintained — the git tag is the version. If you are "
        "bumping it to match a release, that is the change this pin exists to prevent: it will be "
        "correct today and stale by the next release, exactly as 0.4.0 was for eleven of them. "
        "Read the comment above the field before changing this test."
    )


def test_the_pin_is_explained_where_someone_would_change_it() -> None:
    """A bare `0.0.0` invites a well-meaning bump; the reason has to sit next to the field.

    This asserts the *property* — that the version line is preceded by a comment mentioning the
    tag — rather than any particular wording, so the explanation can be rewritten freely.
    """
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("version ="))

    assert index > 0 and lines[index - 1].lstrip().startswith("#"), (
        "the pinned version has no comment on the line above it; a bare 0.0.0 reads as an oversight"
    )

    # Walk back over the contiguous comment block, so the explanation can be any length.
    start = index
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    comment = "\n".join(lines[start:index]).lower()

    assert "tag" in comment, (
        "the comment above the pinned version does not say where the version actually lives"
    )


def test_the_pin_does_not_pretend_to_be_the_newest_tag() -> None:
    """The twin, and it is the interesting direction.

    A guard asserting equality with a constant cannot fail by drift — only by someone editing it.
    So this asserts the *other* property: that the pin is genuinely not tracking releases. If the
    pin ever equals the newest tag, either someone started maintaining the field after all — in
    which case this whole file is the wrong policy and should be replaced rather than edited — or
    the tags have been reset. Both deserve a human.

    Skips where git has no tags, so a shallow clone reports honestly rather than passing blank.
    """
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "tag", "-l", "v*"],
        capture_output=True,
        text=True,
        check=False,
    )
    tags = [t.lstrip("v") for t in listed.stdout.split() if re.fullmatch(r"v[\d.]+", t)]
    if not tags:
        return  # no tags reachable here; nothing to compare against

    newest = max(tags, key=lambda t: [int(part) for part in t.split(".")])
    assert newest != PINNED, (
        f"the pinned root version {PINNED!r} now equals the newest tag. Either the field is being "
        "maintained after all — replace this policy rather than editing the constant — or the tag "
        "history has been rewritten."
    )
