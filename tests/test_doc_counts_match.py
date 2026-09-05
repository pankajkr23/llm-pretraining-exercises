"""Prose that states a count must agree with the thing it counts.

`AGENTS.md` names this as the failure that has cost this repo the most edits, and both cases here
are ones it had already suffered:

- `AGENTS.md`'s own section heading read **"Three data concerns"** above **five** bullets, in the
  document that forbids exactly that.
- The root `README.md`'s row for exercise 06 read **"Stage 1 of 8"** while the exercise's own stage
  table marked seven done — and the root README is the front door a grader lands on.

Neither was caught by anything. A rule the rulebook breaks is a rule nobody is enforcing.
"""

import re
from pathlib import Path

import pytest
from _exercises import exercises_in

ROOT = Path(__file__).resolve().parents[1]

_NUMBER = re.compile(r"^\|\s*(\d{2})\s*\|")
AGENTS = ROOT / "AGENTS.md"
README = ROOT / "README.md"

WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _section_bullets(text: str, heading: str) -> list[str]:
    """Top-level bullets under a heading, stopping at the next heading of any level.

    Args:
        text: The document.
        heading: The exact heading line.

    Returns:
        The bullet lines.
    """
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = rest.find("\n## ")
    body = rest if end == -1 else rest[:end]
    return [line for line in body.splitlines() if line.startswith("- **")]


def test_the_data_concerns_heading_counts_its_own_bullets() -> None:
    """The heading said "Three" above five bullets, in the file that bans that exact defect."""
    text = AGENTS.read_text()
    match = re.search(r"^## ([A-Za-z]+) data concerns[^\n]*$", text, re.M)
    assert match, "AGENTS.md lost its data-concerns heading; this guard is now inert"

    claimed = WORDS.get(match.group(1).lower())
    assert claimed is not None, f"unrecognised number word in the heading: {match.group(1)!r}"

    actual = len(_section_bullets(text, match.group(0)))
    assert claimed == actual, (
        f"the heading claims {claimed} data concerns and the section has {actual}"
    )


def _exercise_row(number: str) -> str:
    """The root README's table row for an exercise.

    Args:
        number: Zero-padded exercise number.

    Returns:
        The row.
    """
    for line in README.read_text().splitlines():
        if line.startswith(f"| {number} "):
            return line
    raise AssertionError(f"the root README has no table row for exercise {number}")


TABLE_HEADER = "| # | Exercise | Summary |"


def _rows_github_would_render(text: str) -> list[str]:
    """The exercise rows that are actually inside the table, as GitHub parses it.

    **A blank line terminates a GitHub-Flavoured Markdown table.** A `| 09 | ... |` line after one
    is not a row — it is a paragraph that happens to contain pipe characters, and it renders as
    literal text. So this walks from the header and stops where the table stops, rather than
    scanning the file for lines that look like rows.

    Args:
        text: The README's contents.

    Returns:
        Every line inside the table body, in order.
    """
    lines = text.splitlines()
    if TABLE_HEADER not in lines:
        raise AssertionError(f"the root README has no exercise table header: {TABLE_HEADER!r}")

    body = lines[lines.index(TABLE_HEADER) + 2 :]  # past the header and its `| --- |` delimiter
    kept: list[str] = []
    for line in body:
        if not line.startswith("|"):
            break
        kept.append(line)
    return kept


def test_every_exercise_has_a_row_inside_the_table_github_renders() -> None:
    """**The front door's table, and a guard that read rows instead of the table.**

    `README.md` carried a blank line between row 08 and rows 09 and 10 for a day. GitHub ended the
    table there, so the two newest exercises — one of them the one being submitted — rendered as a
    paragraph of literal `|` characters on the page a grader lands on and nowhere else.

    Nothing went red, because `_exercise_row` looks for a line *starting* `| 09 ` and finds it
    whether or not the table around it survives. Reading a row cannot see the table; this does.
    """
    numbered = [row for row in _rows_github_would_render(README.read_text()) if _NUMBER.match(row)]
    rendered = [match.group(1) for match in (_NUMBER.match(row) for row in numbered) if match]
    expected = [path.name[:2] for path in exercises_in(ROOT / "src" / "exercises")]

    assert rendered == expected, (
        f"the table GitHub renders holds rows {rendered} and the repository has exercises "
        f"{expected}. A row outside the table renders as literal text, so a missing number here "
        "usually means a blank line crept into the table rather than a row being deleted."
    )


def test_the_table_contiguity_check_can_actually_fail() -> None:
    """The twin: plant the exact defect and watch the reader lose what it should have kept.

    A guard nobody has watched fail is not a guard, and this one is a reader rather than an
    assertion, so its teeth are entirely in whether the reader stops where the table stops.
    """
    intact = _rows_github_would_render(README.read_text())
    last = intact[-1]
    broken = README.read_text().replace(f"\n{last}", f"\n\n{last}", 1)

    assert len(_rows_github_would_render(broken)) < len(intact), (
        "a blank line planted before the last row did not shorten the table, so this reader "
        "cannot see the defect it exists for"
    )


def test_the_root_readme_row_states_the_exercise_s_real_stage() -> None:
    """**The front door, and the one nothing was checking.**

    A grader lands on the root README and nowhere else. It read "Stage 1 of 8" while the exercise
    had shipped seven — a six-stage lie on the first page, with every downstream document correct.
    The number is derived here from the exercise's own stage table, which is the same source its
    README's status line uses.
    """
    row = _exercise_row("06")
    claimed = re.search(r"Stage (\d+) of (\d+)", row)
    assert claimed, f"the exercise 06 row no longer states a stage: {row!r}"

    table = (ROOT / "src/exercises/06-build-training-dataset/README.md").read_text()
    rows = re.findall(r"^\|\s*(\d+)\s*\|[^|]*\|\s*([^|]*?)\s*\|\s*$", table, re.M)
    done = [int(n) for n, status in rows if "done" in status.lower()]

    assert int(claimed.group(2)) == len(rows), (
        f"the root row says there are {claimed.group(2)} stages; the table has {len(rows)}"
    )
    assert int(claimed.group(1)) == max(done, default=0), (
        f"the root row claims stage {claimed.group(1)}; the exercise's table marks "
        f"{max(done, default=0)} as the furthest one done"
    )


@pytest.mark.parametrize("number", ["06"])
def test_the_exercise_row_links_its_readme_directly(number: str) -> None:
    """ "Without a detour" is satisfied by a link, not by a section.

    `AGENTS.md` is explicit that asserting the *filename* passes against a front door that names the
    file and never links it. So this asserts the link.
    """
    row = _exercise_row(number)
    slug = "06-build-training-dataset"
    assert f"](src/exercises/{slug}/README.md)" in row, (
        f"the exercise {number} row does not link its README directly: {row!r}"
    )
