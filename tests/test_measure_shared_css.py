"""`base_selector()` decides which unmatched rules the driven measurement may call dead.

`tools/measure_shared_css.py --drive` judges a rule written for a reader-reached state — a focus,
a hover — or for a pseudo-element by whether the element it styles exists, because
`querySelector` sees none of those at rest. So this one helper stands between a live rule and a
deletion: strip too little and `.step:focus-visible` reads as dead on a page full of steps; strip
too much and `:has()` or `:not()` stop narrowing the selector, and a rule for an element that
never exists reads as alive.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import measure_shared_css as tool  # noqa: E402

REACHED = [
    (".step:focus-visible", ".step"),
    (".x:focus-within", ".x"),
    (".a:hover .b", ".a .b"),
    ("details.arithmetic > summary::-webkit-details-marker", "details.arithmetic > summary"),
    ("details.arithmetic[open] > summary::before", "details.arithmetic[open] > summary"),
    ("input[type='range']::-moz-range-thumb", "input[type='range']"),
]

STRUCTURAL = [
    ".stagerow.missing",
    ".canvas:has(span.unit)",
    ".x:not(.y)",
    "li:nth-child(2n)",
]


@pytest.mark.parametrize(("selector", "element"), REACHED)
def test_a_reached_state_or_pseudo_element_is_judged_by_its_element(
    selector: str, element: str
) -> None:
    assert tool.base_selector(selector) == element


@pytest.mark.parametrize("selector", STRUCTURAL)
def test_a_selector_that_only_narrows_which_elements_is_left_alone(selector: str) -> None:
    """`:has()`, `:not()` and `:nth-child()` choose elements; none is a state a reader reaches."""
    assert tool.base_selector(selector) is None
