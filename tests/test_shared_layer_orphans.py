"""The vendored stylesheet styles more than any page emits, and the gap must not grow silently.

`web/_shared/page.css` is copied byte-identically into every deployable exercise. It has accumulated
rules for components that no page builds — and the cost is not the bytes. It is that a reader
vendoring the file into a new exercise cannot tell which of its rules will do anything, so they
build to a stylesheet that half-applies and debug the half that does not.

**This file measures the gap and refuses to let it grow. It deliberately does not delete anything**,
and the reason is a near-miss this repository has already had: in #101 a class-usage extractor
looked for `el(tag, className)` while exercise 03 calls a local `$(tag, className)`, reported an
entire live stylesheet as used by nobody, and deleting on that evidence would have removed it.

The measurement here is a *lower bound on what is live*, never a proof of what is dead:

- A class can be added by JavaScript, so the count comes from a **rendered DOM**, not a grep.
- A class can appear only after the reader does something. Driving the pages' inputs and buttons
  moved **two** classes out of this list — `.filter-none` and `.rail-shut` — both of which a static
  scan would have called dead.
- A class can appear only in a state no automated pass reaches: a particular filter combination, a
  search with no results, an error path. Several names below are mentioned in exercise 03's source
  and are almost certainly in that category.

So the ledger is a **budget**, not a hit list. Adding a rule for a component no page builds is
allowed only by adding its name here, deliberately — and removing a name is only ever done after
confirming in a browser that the class is really gone.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The vendored copies are byte-identical, so any one of them measures all of them — and a separate
#: guard already asserts they do not diverge.
SHARED = sorted(REPO_ROOT.glob("src/exercises/*/web/_shared/page.css"))

#: Classes `page.css` styles that no rendered page was observed to emit.
#:
#: Measured on 2026-09-04 by serving the assembled site and collecting `classList` from every
#: element of all 13 pages, after scrolling to the bottom and after driving every text input,
#: button and checkbox. Two candidates left this list at that last step, which is the whole reason
#: the measurement is not a grep.
#:
#: **Being in this list does not mean a class is dead.** Where the source still mentions one, the
#: reason is noted: it is reachable, just not by any pass that runs unattended.
KNOWN_ORPHANS: dict[str, str] = {
    # No source file mentions these at all. The likeliest explanation is a component that was
    # renamed or removed while its rules stayed behind — but "likeliest" is not "verified".
    "appendix-h": "no source mentions it",
    "band-head": "no source mentions it; the .band-* family looks like one removed component",
    "band-label": "no source mentions it",
    "band-n": "no source mentions it",
    "band-none": "no source mentions it",
    "band-sub": "no source mentions it",
    "band-swatch": "no source mentions it",
    "catsearch-hits": "no source mentions it",
    "gapmark": "no source mentions it",
    # Exercise 03 builds these, in states an unattended pass does not reach — an empty gate card,
    # a filtered catalogue row. Live, and listed only because nothing automated has seen them.
    "bands": "exercise 03 and others mention it; not reached unattended",
    "cardrow": "exercise 03 builds it",
    "catalogue": "exercise 03 and 08 build it",
    "catrow": "exercise 03 builds it",
    "catrow-name": "exercise 03 builds it",
    "catrow-why": "exercise 03 builds it",
    "gatecard-gotcha": "exercise 03 builds it",
    "gatecard-kv": "exercise 03 builds it",
    "gatecard-links": "exercise 03 builds it",
    "gatecard-name": "exercise 03 builds it",
    "gatecard-none": "exercise 03's empty state",
    "gatecard-note": "exercise 03 builds it",
    "gatecard-owner": "exercise 03 builds it",
    "gateverdict": "exercise 03 builds it",
    # Ordinary words that also happen to be class names. Kept for completeness, since the
    # measurement cannot tell a class from a word without rendering.
    "css": "also an ordinary word in prose and filenames",
    "json": "also an ordinary word in prose and filenames",
    "sel": "short name; also appears inside identifiers",
    "unit": "also an ordinary word in prose",
}


def _declared() -> set[str]:
    """Every class `_shared/page.css` selects, across **all** the vendored copies.

    Reading one copy would be enough only while they are identical — which the test above asserts,
    but the two would then be checking each other in a circle. Taking the union means a rule added
    to any copy is counted, and verifying that took two attempts: the first break added a rule to
    exercise 07's copy while this function read exercise 03's, so the guard looked blind when the
    *break* was aimed at the wrong file.
    """
    found: set[str] = set()
    for path in SHARED:
        found |= set(re.findall(r"\.([a-z][a-z0-9-]*)", path.read_text(encoding="utf-8")))
    return found


def test_the_shared_stylesheet_is_vendored_identically() -> None:
    """One measurement stands for all the copies only while the copies are the same."""
    assert SHARED, "no exercise vendors _shared/page.css"
    first = SHARED[0].read_bytes()
    different = [str(p.relative_to(REPO_ROOT)) for p in SHARED[1:] if p.read_bytes() != first]
    assert not different, (
        "these vendored copies of _shared/page.css have diverged, so the orphan measurement below "
        "speaks for only one of them:\n  " + "\n  ".join(different)
    )


def test_every_known_orphan_is_still_a_class_the_stylesheet_selects() -> None:
    """The ledger fails in **both** directions.

    A name here that the stylesheet no longer styles means somebody removed the rule and left the
    entry — and an entry that cannot fire is an entry nobody will ever question. This is the half
    that keeps the list honest as things get fixed.
    """
    declared = _declared()
    stale = sorted(name for name in KNOWN_ORPHANS if name not in declared)
    assert not stale, (
        "these names are in KNOWN_ORPHANS but `_shared/page.css` no longer styles them. The rule "
        "was removed and the entry was not:\n  " + "\n  ".join(stale)
    )


def test_the_orphan_budget_has_not_grown() -> None:
    """The one that actually protects anything.

    It does not check *which* classes are unused — that needs a browser, and a browser guard is one
    that stops running the moment chromium is unavailable. It checks the **size** of the styled
    surface against the size that was measured, so a new rule for a component nothing builds cannot
    arrive unnoticed.
    """
    declared = _declared()
    # Measured alongside KNOWN_ORPHANS, in the same pass.
    measured_total = 101
    assert len(declared) <= measured_total, (
        f"`_shared/page.css` now styles {len(declared)} classes, up from {measured_total} when the "
        f"orphan set was last measured in a browser. {len(declared) - measured_total} rule(s) were "
        "added.\n\n"
        "If they are for a component a page actually builds, raise `measured_total` and say so. If "
        "they are not, they are new orphans — and this file exists because the last count of those "
        "was 29 and nobody noticed it climbing.\n\n"
        "Re-measure by serving `public/` and collecting classList from every element of every "
        "page, after scrolling AND after driving each input and button: two classes only appear "
        "at that last step."
    )
