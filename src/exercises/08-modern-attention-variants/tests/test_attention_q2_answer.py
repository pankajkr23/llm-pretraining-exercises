"""Question 2's answer says what the code computes, and changes when the evidence does.

`tools/build_q2_answer.py` writes `artifacts/q2_answer.txt` from the page's own payload. The file
itself is gitignored, so these tests check what the generator *renders*: that every mechanism
claimed for extra credit is named with its date and paper, that each sentence about the arc is the
one the verdict calls for, and that none of the text that went stale in the hand-written version
can come back. One test hands the generator a fabricated verdict, because a sentence that cannot
change with the evidence is a sentence typed, not derived.
"""

import copy
import importlib.util
import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "build_q2_answer", EXERCISE / "tools" / "build_q2_answer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()
DATA = TOOL._payload()
TEXT = TOOL.render(DATA)


def test_every_mechanism_outside_the_list_is_named_with_its_date_and_paper() -> None:
    """The extra credit asks for the name, the date and where the date came from — all three."""
    bonus = [m for m in DATA["mechanisms"] if m["bonus"]]
    assert bonus, "no mechanism is marked as outside the list; the reader has rotted"
    missing = [
        m["key"]
        for m in bonus
        if not (
            m["name"] in TEXT
            and m["date"] in TEXT
            and m["source"]["url"] in TEXT
            and m["source"]["quoted"] in " ".join(TEXT.split())
        )
    ]
    assert not missing, f"named without its date, paper or submission line: {missing}"
    assert f"{len(bonus)} of the {DATA['counts']['total']} are outside" in TEXT


def test_the_arc_heading_is_the_one_the_verdict_calls_for() -> None:
    holds = DATA["arc"]["robust"]["matchesAnywhere"]
    assert ("DOES NOT SURVIVE THE DATES" in TEXT) is (not holds)
    assert ("HOLDS IN THE DATES" in TEXT) is holds


def test_a_different_verdict_produces_a_different_answer() -> None:
    """The twin. Feed in an arc that holds and a cache bill that wins, and the text must follow."""
    flipped = copy.deepcopy(DATA)
    arc = flipped["arc"]
    arc["matches"] = True
    arc["observed"] = list(arc["claimed"])
    arc["neverDominates"] = []
    arc["robust"]["matchesAnywhere"] = True
    arc["robust"]["cacheNeverDominates"] = False
    text = TOOL.render(flipped, shifted=[])
    assert "HOLDS IN THE DATES" in text
    assert "DOES NOT SURVIVE" not in text
    assert "never wins a single window" not in text, "the cache sentence did not depend on data"
    assert "memory wins no window" not in text
    assert "That is the claimed sequence" in " ".join(text.split())


def test_the_window_table_is_the_catalogue_s() -> None:
    # The table's own indent: finding 3 also opens a line with a window, in prose.
    rows = [line.split() for line in TEXT.splitlines() if re.match(r"^ {5}\d{4}-\d{4}  ", line)]
    expected = [(f"{p['start']}-{p['end']}", p["dominant"] or "no") for p in DATA["periods"]]
    assert [(r[0], r[1]) for r in rows] == expected


def test_the_opening_count_is_the_number_of_findings_written() -> None:
    findings = re.findall(r"^(\d+)\. [A-Z]", TEXT, flags=re.MULTILINE)
    assert findings == [str(i) for i in range(1, len(findings) + 1)]
    assert f"{TOOL._spoken(len(findings))} things are visible" in " ".join(TEXT.split())


@pytest.mark.parametrize(
    "stale",
    [
        "404",  # the banner about a broken link, kept long after the link was fixed
        "!!",
        "compute, then cache, then both",  # an arc the verdict refutes
        "Every mechanism that cuts the cache sits in a window",  # false: one such window is a tie
    ],
)
def test_text_that_went_stale_cannot_come_back(stale: str) -> None:
    assert stale not in " ".join(TEXT.split())


def test_the_corrected_cache_figure_is_the_one_the_formula_gives() -> None:
    disc = DATA["transcriptDiscrepancy"]
    flat = " ".join(TEXT.split())  # a figure and its unit may wrap onto two lines
    assert f"{disc['computedBytes'] / 1e12:.2f} TB" in flat
    assert f"about {disc['claimedTB']:g} TB" in flat


def test_the_answer_is_the_same_every_time_it_is_rendered() -> None:
    assert TOOL.render() == TEXT
