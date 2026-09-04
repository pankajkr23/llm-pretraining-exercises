"""A rail that never says where you are is styled, inert, and indistinguishable from a working one.

`_shared/page.css` has styled `.rail-link.on` — an accent bar and a bold label — since before most
of these pages existed. Exercise 03 was the only one that ever set the class. 05, 06 and 07 each
shipped a contents rail that looked complete and never moved, and nothing failed: the markup is
there, the styles are there, and the only way to notice is to scroll and watch nothing happen.

**This guard is lexical, and that is the deliberate half.** The browser check that actually proves
the behaviour lives in each exercise's own render suite, where it needs chromium — and a rule that
only runs when chromium happens to be installed is one that can silently stop running, which has
already cost this repository 46 tests. So the structural half runs everywhere: a page that builds a
rail must also contain the code that marks it.

What it cannot see is whether the marking is *correct* — that the right entry lights up, and that it
follows rather than leading. That is the browser's job, and `tests/test_page_spine.py` already
enforces the same split for the twelve-part spine.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every page script that builds a contents rail, found rather than listed.
CHAPTERS = sorted(REPO_ROOT.glob("src/exercises/*/web/chapters.js"))


def _builds_a_rail(source: str) -> bool:
    """A page builds a rail when it creates `.rail-link` entries."""
    return "'rail-link'" in source or '"rail-link"' in source


def _marks_position(source: str) -> bool:
    """...and marks position when it toggles `on` against a scroll or intersection signal.

    Two implementations are in use and both are fine: a scroll listener that finds the last heading
    past a threshold, and an `IntersectionObserver`. What is asserted is that *something* sets the
    class in response to the reader moving — not which technique, because a guard that names one
    implementation fails every other one.
    """
    toggles = re.search(r"classList\.toggle\(\s*['\"]on['\"]", source)
    if not toggles:
        return False
    reacts = "addEventListener('scroll'" in source or "IntersectionObserver" in source
    return bool(reacts)


def test_every_page_that_builds_a_rail_also_marks_position() -> None:
    """The defect, stated as the property.

    A rail is a promise that the reader can see where they are. Building one and never marking it
    keeps the promise's shape and drops its content.
    """
    silent = [
        str(p.relative_to(REPO_ROOT))
        for p in CHAPTERS
        if _builds_a_rail(p.read_text(encoding="utf-8"))
        and not _marks_position(p.read_text(encoding="utf-8"))
    ]
    assert not silent, (
        "these pages build a contents rail and never mark the section in view:\n  "
        + "\n  ".join(silent)
        + "\n\n`.rail-link.on` is already styled in the shared stylesheet, so the rail looks "
        "finished and simply never moves. Set the class from a scroll listener or an "
        "IntersectionObserver."
    )


def test_the_marking_reacts_to_the_reader_rather_than_being_set_once() -> None:
    """A class set once at build time would satisfy the test above and mark the wrong section.

    This is the twin, and it is the difference between "the rail has an accent bar somewhere" and
    "the accent bar is on the section you are reading".
    """
    inert = []
    for p in CHAPTERS:
        source = p.read_text(encoding="utf-8")
        if not _builds_a_rail(source) or not re.search(
            r"classList\.toggle\(\s*['\"]on['\"]", source
        ):
            continue
        if "addEventListener('scroll'" not in source and "IntersectionObserver" not in source:
            inert.append(str(p.relative_to(REPO_ROOT)))

    assert not inert, (
        "these pages set `.rail-link.on` but nothing re-runs it when the reader scrolls:\n  "
        + "\n  ".join(inert)
        + "\n\nA rail marked once at build time points at the top of the page forever."
    )


def test_at_least_one_page_is_covered_so_this_guard_cannot_pass_vacuously() -> None:
    """A guard over an empty set is green and worthless.

    `CHAPTERS` is globbed, so a rename or a moved directory would empty it silently — and both
    assertions above would then pass by having nothing to check.
    """
    railed = [p for p in CHAPTERS if _builds_a_rail(p.read_text(encoding="utf-8"))]
    assert len(railed) >= 4, (
        f"only {len(railed)} page(s) were found to build a rail, which is fewer than the pages "
        "known to have one. The glob has probably stopped matching, and both assertions above are "
        "passing over an empty set."
    )
