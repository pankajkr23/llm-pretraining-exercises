"""Every evidence figure the README states must be in an evidence file.

**Why this file exists.** `AGENTS.md`'s rule is that prose stating a number is generated too, or it
goes stale beside the table that is right — and this exercise's byte-recovery figures were exactly
the case it warns about. The README quotes `100.00%`, `15.05%` and `0.00%` for wrapped positions,
`results/wrap_recovery.json` holds them, and **nothing checked that the two agreed**. That is how
`19.1%` survived in four documents after the measurement said `15.05%`.

The guard is deliberately about the *property* rather than a phrasing: a delimited block of the
README may state any percentage it likes, provided every one of them appears in the evidence files
the block itself names. Rewriting the prose is free; inventing a number is not.

    <!-- evidence-numbers: results/wrap_recovery.json results/position_schemes.json -->
    ...prose and tables...
    <!-- /evidence-numbers -->

**Both directions, as this repository's ledgers do.** The block must exist and must contain
percentages, because a guard over an empty block passes for every input — and this one would then
report as coverage of the numbers a reader actually sees.

**The `skip` when an evidence file is absent is not an escape hatch, and it works because of a rule
one directory up.** `tests/_skips.py` makes an undeclared skip a CI failure, so a block naming a
file that `results/` does not carry turns the job red rather than quietly passing. Deleting the
evidence is therefore as loud as contradicting it, which is the property this file needs: prose
whose source has gone missing is the same failure as prose whose source disagrees.
"""

import json
import re
from pathlib import Path

import pytest

EXERCISE = Path(__file__).resolve().parents[1]
README = EXERCISE / "README.md"

BLOCK = re.compile(
    r"<!--\s*evidence-numbers:\s*(?P<files>[^>]*?)\s*-->(?P<body>.*?)<!--\s*/evidence-numbers\s*-->",
    re.S,
)
FIGURE = re.compile(
    r"\b\d+\.\d{1,4}%?(?P<escape><!--\s*unmeasured:\s*(?P<reason>(?:(?!-->).)*?)\s*-->)?",
    re.S,
)
"""A decimal figure, with or without a percent sign, and the escape that exempts one.

**Any decimal, not only a percentage.** It began as a percent-only pattern for the byte-recovery
tables and could not see the coherence table three sections above — four rows of bare decimals like
`0.051`, hand-typed, matching `measurements.json::coherence` by nothing but somebody's care. Same
failure, different notation.

**Decimals only, and that is a stated limit rather than an oversight.** Round integers in this
prose are approximations and yardsticks — "about 47%", "a third", "128 bytes" — while every figure
that came out of a measurement here is quoted to at least one place. Widening the pattern to bare
integers would mean exempting every one of them, and an exemption list that long is one nobody
reads.

**The escape exists because a corrected number is deliberately not measured.** This exercise's
recovery section quotes the figures it *replaced* — the `19.1%` that four documents carried, the
`14.6%` that is unreproduced — and demanding evidence for those would force the correction to be
deleted, which is the opposite of what it is for. Each escape carries its reason inline, so a
reader meets the exemption where the number is rather than in a table three files away.

The reason may span lines, `>` included: these sentences live inside a blockquote, so a first
version that forbade `>` inside the comment silently failed to recognise the escape at all and
reported both exempted numbers as unmeasured.
"""


def _licensed(files: list[str]) -> set[str]:
    """Every percentage the named evidence files support, formatted the way prose writes one.

    **Two conventions, because the files use two.** The band-recovery bundles store a share in
    `[0, 1]`; `measurements.json`'s inherited `d_p_128` block stores `99.9` meaning 99.9%. A guard
    that knew only one would have demanded evidence for a number sitting in the file it was reading.
    A value of `1.0` is therefore licensed both as `1.00%` and as `100.00%` — a small widening, and
    the alternative is teaching the guard which key means which, which is a second copy of the
    schema.

    One to four decimal places, with and without a percent sign, because `100.0%`, `100.00%` and
    `0.051` are all the same kind of claim written the way its own table writes it. A guard that
    accepted one spelling would be asking the prose to match the test's formatting rather than the
    evidence, which is how a guard ends up rewording correct work.
    """
    out: set[str] = set()

    def offer(value: float) -> None:
        for places in range(1, 5):
            out.add(f"{value:.{places}f}")
            out.add(f"{value:.{places}f}%")

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            if 0.0 <= node <= 1.0:
                offer(node * 100.0)
            if 0.0 <= node <= 100.0:
                offer(node)

    for name in files:
        path = EXERCISE / name
        if not path.is_file():
            pytest.skip(f"{name} is not published yet, so this block cannot be checked")
        walk(json.loads(path.read_text(encoding="utf-8")))
    return out


def _blocks() -> list[tuple[list[str], str]]:
    text = README.read_text(encoding="utf-8")
    return [(found.group("files").split(), found.group("body")) for found in BLOCK.finditer(text)]


def test_the_readme_delimits_its_recovery_numbers_at_all() -> None:
    """A guard with nothing to read passes for every input, which is worse than no guard."""
    blocks = _blocks()
    assert blocks, "no <!-- evidence-numbers: ... --> block in the README"
    for files, body in blocks:
        assert files, "a evidence-numbers block must name the evidence files that license it"
        assert FIGURE.search(body), (
            "a evidence-numbers block with no percentage in it is checking nothing — either the "
            "prose moved out of the block, or the markers are around the wrong lines"
        )


def test_every_recovery_percentage_in_the_readme_is_in_an_evidence_file() -> None:
    """The rule itself. A number a reader can see and nobody can check is folklore."""
    offenders = []
    for files, body in _blocks():
        licensed = _licensed(files)
        for found in FIGURE.finditer(body):
            if found.group("escape"):
                continue
            if found.group() not in licensed:
                line = body[: found.start()].count("\n")
                offenders.append(f"{found.group()} (block line {line}) — not in {', '.join(files)}")
    assert not offenders, (
        f"{len(offenders)} evidence figure(s) in the README are in no evidence file. Re-run\n"
        "the measurement and quote what it says, or move the sentence out of the block:\n  "
        + "\n  ".join(offenders)
    )


def test_every_exemption_carries_a_real_reason() -> None:
    """An escape with no reason is a hole the size of the next number someone wants to type."""
    for _, body in _blocks():
        for found in FIGURE.finditer(body):
            if found.group("escape"):
                assert len(found.group("reason")) > 20, (
                    f"{found.group()!r} is exempted without saying why it is not measured"
                )


def test_the_guard_fails_on_a_number_that_is_not_measured(monkeypatch, tmp_path) -> None:
    """The twin. The break is a temporary copy, never a write into the real README."""
    import sys

    module = sys.modules[__name__]
    files, body = _blocks()[0]
    planted = tmp_path / "README.md"
    planted.write_text(
        f"<!-- evidence-numbers: {' '.join(files)} -->\n{body}\nand also **73.41%** of them\n"
        "<!-- /evidence-numbers -->\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "README", planted)
    with pytest.raises(AssertionError, match="73.41%"):
        test_every_recovery_percentage_in_the_readme_is_in_an_evidence_file()
