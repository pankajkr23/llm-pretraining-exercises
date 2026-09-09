"""`RESEARCH.md` marks where every claim came from, and the marks have to stay honest.

The document mixes three very different kinds of statement: things measured here, things an agent
reported that were re-derived, and things an agent reported that were **not** checked. One research
pass in this exercise corrected itself twice and named two papers that do not exist, so the
difference is not a nicety — a reader has to be able to tell without asking.

These guards keep the marks alive rather than decorative. They are lexical and need nothing but the
repository, so they run in the fast job.
"""

import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
RESEARCH = EXERCISE / "RESEARCH.md"

MARKS = ("[measured]", "[verified]", "[reported]", "[reasoning]")


def _text() -> str:
    return RESEARCH.read_text(encoding="utf-8")


def test_the_research_record_exists_and_explains_its_own_marks() -> None:
    """A legend a reader cannot find makes the marks worse than nothing."""
    text = _text()
    for mark in MARKS:
        assert f"**{mark}**" in text, f"{mark} is used but never defined in the legend"


@pytest.mark.parametrize("mark", MARKS)
def test_every_mark_is_actually_used(mark: str) -> None:
    """A category nothing falls into is a category that is not doing any work.

    If everything ends up `[measured]`, the marks have stopped distinguishing anything and the
    document is claiming more certainty than it has.
    """
    uses = _text().count(mark)
    assert uses >= 2, f"{mark} appears {uses} time(s); a mark used once is not a category"


def test_the_unverified_material_is_marked_and_the_reason_is_stated() -> None:
    """The literature is the weakest part and the document has to say so where a reader meets it.

    Not in a footnote: the pass that produced those citations named two papers that do not exist,
    and a reader who repeats one has been let down by this document rather than by the agent.
    """
    text = _text()
    assert "do not exist" in text, (
        "the record no longer states that a research pass named papers that do not exist -- that "
        "is the single most important caveat in it"
    )
    literature = text[text.index("What the literature says") :][:1200]
    assert "[reported" in literature, "the literature section no longer marks itself as unverified"


def test_every_number_in_a_table_row_sits_beside_a_mark_or_a_heading_that_carries_one() -> None:
    """A table of figures with no provenance is exactly what this exercise published before.

    The check is deliberately coarse: each table must have a mark somewhere in the paragraph
    introducing it. Requiring a mark per cell would be noise nobody reads, which `AGENTS.md` warns
    about directly.
    """
    text = _text()
    blocks = re.split(r"\n(?=#{1,3} )", text)
    unmarked = []
    for block in blocks:
        if "| ---" not in block:
            continue
        if not any(mark in block for mark in MARKS):
            unmarked.append(block.splitlines()[0][:60])
    # The legend table and the five-problems table are definitions, not findings.
    allowed = {"# Where Kronecker v2 goes next", "## The five problems, in plain words"}
    offenders = [b for b in unmarked if b.strip() not in allowed]
    assert not offenders, f"tables with no provenance mark anywhere in their section: {offenders}"


def test_it_says_what_it_cannot_establish() -> None:
    """The repository requires this of every document that reports a result, and it is the section
    a reader deciding whether to believe the work goes to first."""
    text = _text()
    assert "## What none of this establishes" in text
    section = text[text.index("## What none of this establishes") :]
    # THE PROPERTY, NOT A PHRASING. The first version of this demanded the exact string "not a
    # trained result" and went red against "Nothing here is a trained result" -- a guard that names
    # one wording fails every other correct wording, and the pressure is then to reword good prose
    # to satisfy the test. The question being asked is: does this section say the results are not
    # from training?
    assert re.search(r"\btrain(ed|ing)\b", section), (
        "the limits section no longer says these results are not from a training run, which is the "
        "largest thing this work cannot establish"
    )
    assert section.count("\n- ") >= 4, "the limits section has shrunk to a token gesture"


def test_everything_it_points_a_reader_at_exists() -> None:
    """A document that sends a reader to a file that is not there has wasted their time.

    **It counts what it checked, and fails if that is nothing.** The first version scanned only for
    `src/exercises/.../*.py` paths, and once the one such command was removed it had nothing left to
    look at -- it then passed for every possible document, including one pointing at files that do
    not exist. A guard that cannot fail reads as coverage and is worse than none.
    """
    text = _text()
    checked = 0

    for match in re.finditer(r"(src/exercises/[\w/.-]+\.py)", text):
        checked += 1
        assert (EXERCISE.parents[2] / match.group(1)).is_file(), (
            f"RESEARCH.md offers a command running {match.group(1)}, which is absent"
        )

    for match in re.finditer(r"\[[^\]]+\]\((?!https?:)([^)#]+)\)", text):
        checked += 1
        assert (EXERCISE / match.group(1)).exists(), (
            f"RESEARCH.md links to {match.group(1)}, which does not exist beside it"
        )

    for module in re.findall(r"`embeddings\.(\w+)`", text):
        checked += 1
        assert (EXERCISE / "src" / "embeddings" / f"{module}.py").is_file(), (
            f"RESEARCH.md names embeddings.{module}, which is not a module"
        )

    assert checked >= 3, (
        f"this guard examined {checked} references, which is too few to be checking anything -- "
        "either the document stopped pointing anywhere, or the patterns stopped matching it"
    )
