"""Every exercise README has to work as the low-level guide for three different readers.

The root README is a map: it routes, and for the exercise under submission it carries a short
block saying what was found and which file to open. That block is a *submission affordance* — the
grader lands on the root and nowhere else — and it is deliberately the only per-exercise detail
there. Which is exactly why the depth has to live one directory down: if the exercise README is
not the complete end-to-end guide, then nothing is.

So each one states, in its own words, where a first-time reader starts, where someone changing the
code starts, and where someone deciding whether to believe it starts. These tests check that the
reading path exists and names all three, because a guide nobody can enter is a guide nobody reads.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXERCISE_READMES = sorted(REPO_ROOT.glob("src/exercises/*/README.md"))

#: The three readers of `AGENTS.md`'s "Documentation is written for more than one reader".
_READERS = ("first time", "changing the code", "believe it")


def _ids(path: Path) -> str:
    return path.parent.name


def _reading_path(doc: Path) -> str:
    """The `## How to read this` section of `doc`, or "" when it has none."""
    text = doc.read_text(encoding="utf-8")
    match = re.search(r"^## How to read this$(.*?)^## ", text, re.M | re.S)
    return match.group(1) if match else ""


@pytest.mark.parametrize("doc", EXERCISE_READMES, ids=_ids)
def test_every_exercise_readme_has_a_reading_path(doc: Path) -> None:
    """Without one, a 597-line README is a wall rather than a guide."""
    assert _reading_path(doc), (
        f"{doc.parent.name}/README.md has no '## How to read this' section — the root README "
        f"routes here and this is where the end-to-end detail is supposed to live"
    )


@pytest.mark.parametrize("doc", EXERCISE_READMES, ids=_ids)
def test_the_reading_path_addresses_all_three_readers(doc: Path) -> None:
    """Naming only the newcomer is the common failure; the reviewer is the one usually dropped."""
    section = _reading_path(doc).lower()
    missing = [reader for reader in _READERS if reader not in section]
    assert not missing, f"{doc.parent.name}/README.md's reading path does not address: {missing}"


@pytest.mark.parametrize("doc", EXERCISE_READMES, ids=_ids)
def test_every_exercise_readme_states_how_to_run_it_and_what_it_cannot_do(doc: Path) -> None:
    """A guide that cannot be run, or whose limits are unstated, is not the deliverable it claims.

    Both halves are required by `AGENTS.md`: limits belong in the open text beside the numbers,
    and the reproduce path is what separates a guide from a write-up.
    """
    text = doc.read_text(encoding="utf-8")
    assert "```bash" in text.lower(), (
        f"{doc.parent.name}/README.md shows no command to run anything"
    )

    # A *heading*, not a mention. Checking the whole document passes on the reading path's own
    # link text -- "[What it cannot tell you](#what-it-cannot-tell-you)" -- so renaming the section
    # away left the guard green. It was watched surviving that exact mutation before this changed.
    headings = [
        line.lstrip("#").strip().lower() for line in text.splitlines() if line.startswith("#")
    ]
    assert any(
        phrase in heading
        for heading in headings
        for phrase in ("cannot tell you", "cannot show", "cannot establish", "criticism of")
    ), f"{doc.parent.name}/README.md has no section stating what the work cannot establish"
