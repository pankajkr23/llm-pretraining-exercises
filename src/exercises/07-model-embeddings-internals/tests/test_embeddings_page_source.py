"""What may and may not be typed into the page's source.

Two rules this exercise states and nothing enforced, so both were broken:

- **No measurement is written into `chapters.js`.** The rule is in that exercise's `CLAUDE.md` and
  four literals had accumulated under it — `+1.225`, `0.141` and `−0.002` twice — which are exactly
  the numbers a clean re-run moves. The tables would update and the sentences beside them would
  quietly disagree.
- **No shell command is written into `chapters.js`.** A page is read far more often than it is
  executed and cannot be tested; commands belong in the README, beside the code they operate on.

Neither guard reads the rendered DOM, so both run in the plain `test` job with no browser. That is
deliberate: a structural rule that only runs where chromium happens to be installed is one that can
silently stop running, and this repository has already lost tests exactly that way.
"""

import re
from pathlib import Path

import pytest

CHAPTERS = Path(__file__).resolve().parents[1] / "web" / "chapters.js"

MEASUREMENT = re.compile(
    r"(?<![\w.])[+\-\u2212]?\d+\.\d+(?![\w])|(?<![\w.])[+\-\u2212]\d+(?![\w.])"
)
"""A figure shaped like a measurement: a decimal, or a signed integer.

**Scanned over the whole file rather than over parsed string literals, and that is a correction.**
The first version hand-rolled a JS string parser and matched quoted bodies. It desynchronised on the
first apostrophe inside a double-quoted string — `a word\u2019s bytes` — and silently stopped seeing
everything after it, including a literal planted directly to test it. A guard that quietly covers
half a file is worse than none, because it reads as coverage.

Scanning the raw text is safe here precisely because the pattern is narrow: a decimal point or a
sign. Ordinary code numbers — array indices, loop bounds, colour stops — are bare integers and do
not match. The handful of signed or decimal numbers that are genuinely code live in `ALLOWED`.
"""

ALLOWED = {
    "0.1": "a yardstick for the reader — 'on this page 0.1 nats is a large gap' — not a figure "
    "from any run",
    "0.31": "an illustration of a precision failure, not a figure from any run here",
    "0.30": "the other half of that illustration",
    "-90": "an SVG rotate() argument inside a template string",
    "0.141": "appears only in a source comment explaining the guard, and is not rendered",
    "\u22120.031": "a source comment explaining which baseline the MLP arm is measured against; "
    "not rendered, and the rendered figure beside it IS derived",
    "0.15": "axis padding in the chart builder - `Math.min(...all) - 0.15` - which is layout, not "
    "a measurement of anything",
}
"""Literals that are not measurements of this exercise's runs, each with the reason.

**This list may only shrink by someone deciding to remove a literal, never grow to clear a red
gate.** Every entry is a number whose value no run of this exercise can change; a figure that a run
CAN move belongs in `results/measurements.json` and is read from `M`.
"""


def _lines() -> list[tuple[int, str]]:
    """Every line of the page source, with a `${...}` substitution stripped.

    A `${...}` is a value read from `M` at render time, which is the correct thing to do — only the
    literal text around it is being judged.
    """
    source = CHAPTERS.read_text(encoding="utf-8")
    return [
        (number, re.sub(r"\$\{[^}]*\}", "", line))
        for number, line in enumerate(source.splitlines(), start=1)
    ]


def test_no_measurement_is_typed_into_the_page() -> None:
    """Every figure a run can move is read from the measurements, never written by hand."""
    offenders = []
    for number, line in _lines():
        for found in MEASUREMENT.finditer(line):
            if found.group() not in ALLOWED:
                offenders.append(f"chapters.js:{number}: {found.group()}  in {line.strip()[:70]!r}")
    assert not offenders, (
        f"{len(offenders)} measurement(s) typed into the page. Read them from `M` instead — the "
        "tables already do, so a typed one drifts the moment a run moves it:\n  "
        + "\n  ".join(offenders)
    )


# `test_no_shell_command_is_typed_into_the_page` lived here and is now
# `tests/test_no_commands_on_pages.py`, which sweeps every deployed page instead of this one.
# Promoting it found eight commands on exercise 05's page and six on 06's, none of them a clean
# duplicate of its README — the thing a single-exercise guard structurally cannot see. The
# `COMMAND` pattern moved with it; leaving a copy here would be a second definition of the same
# rule, and the second copy is the one that drifts.


def test_the_allowance_list_is_alive() -> None:
    """An exemption for a literal nothing uses is a hole the size of the next literal.

    Both directions, as this repository's ledgers do: every allowed value must actually appear in
    the page, and each must carry a reason.
    """
    body = CHAPTERS.read_text(encoding="utf-8")
    for value, reason in ALLOWED.items():
        assert value in body, (
            f"{value!r} is allowed and no longer appears in the page — remove the entry rather "
            "than leaving an exemption nothing needs"
        )
        assert len(reason) > 20, f"{value!r} is exempted without a real reason"


@pytest.mark.parametrize(
    ("planted", "guard"),
    [
        ("'the gap was \u22120.404 nats'", "test_no_measurement_is_typed_into_the_page"),
        # The command guard's twin moved with it to `tests/test_no_commands_on_pages.py`, which
        # plants into a `tmp_path` copy rather than monkeypatching this module. Its note is worth
        # carrying: **no backticks in planted text** — `STRING` deliberately excludes them from a
        # string body, so a command wrapped in backticks is invisible to the guard and the twin
        # passes for the wrong reason, which is exactly what this one did once.
    ],
)
def test_each_guard_fails_on_a_deliberately_broken_page(planted, guard, monkeypatch, tmp_path):
    """The twin. A guard nobody has watched fail is not a guard.

    The break is held in a temporary file rather than written into the page, so an exception cannot
    leave a planted literal behind for `git add` to commit.
    """
    import sys

    module = sys.modules[__name__]
    broken = tmp_path / "chapters.js"
    broken.write_text(CHAPTERS.read_text(encoding="utf-8") + f"\nconst planted = {planted};\n")
    monkeypatch.setattr(module, "CHAPTERS", broken)
    with pytest.raises(AssertionError):
        getattr(module, guard)()
