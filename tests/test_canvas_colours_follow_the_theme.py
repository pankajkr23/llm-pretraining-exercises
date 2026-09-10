"""No canvas paints chrome in a colour the theme cannot move.

This site has **six** themes, and every text and surface colour in every one of them was generated
against a contrast checker rather than chosen by eye. A `ctx.strokeStyle = '#fff'` obeys none of
that: it is correct in the two themes with a white ground and invisible or wrong in the other four,
and **no test in this repository could see it**, because the guards that enforce the palette read
CSS and these colours live inside a 2D canvas context.

Exercise 01 is the only exercise that draws with canvas, and it carried **twelve** such literals
against sixteen theme-aware colour writes in the same three files — a white ring around scatter
dots, grey gridlines, and the accent blue written out as `rgba(0,113,227,…)` beside code two lines
away reading `--accent` from the live tokens.

**Measured before and after, through the site's own theme mechanism**: `s1.html` and `s2.html` each
produced **two** distinct canvas renderings across the six themes, and now produce six.

**This guard is lexical, and the alternative is worse.** A rendered guard — "the canvases differ
across themes" — is the more direct property and it would have passed `s4.html`, which already
rendered six distinct images because its other colour writes dominated the picture while five of its
literals were still wrong in the details. Asking whether a colour *can* move is the question with
teeth; asking whether the picture changed is the one a coarse instrument answers yes to.

**Category colours are deliberately out of scope, and that is a decision rather than an oversight.**
The scatter dots encode a data class — warm for one label, cool for the other — and one of them is
also interpolated per pixel into an `ImageData` buffer, where a CSS variable cannot go without being
parsed. Those are semantics carried by colour, not chrome that should follow the ground.
`RECORDED_INCONSISTENCY` names the one place this is untidy, so it is written down rather than
discovered again.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A colour written into a canvas context as a literal: hex, `rgb()`/`rgba()`, or a CSS keyword.
#: Deliberately does not match a template literal — `rgb(${r},${g},${b})` is a computed value, and
#: the two places that build one are the category colours named below.
LITERAL_COLOUR = re.compile(
    r"(fillStyle|strokeStyle|shadowColor)\s*=\s*"
    r"(['\"])(#[0-9a-fA-F]{3,8}|rgba?\([^)$]*\)|[a-z]+)\2"
)

#: Every canvas colour write, literal or not, so the guard can report the ratio rather than a
#: bare count — "0 of 28" says the file is covered; "0" alone says nothing about whether it draws.
ANY_COLOUR_WRITE = re.compile(r"(fillStyle|strokeStyle|shadowColor)\s*=")

#: Written down so it is not rediscovered. Exercise 01's two-class colours are hardcoded as RGB
#: triples (`[77, 118, 168]` cool, `[178, 88, 46]` / `[199, 138, 66]` warm) while `s2.html` and
#: `s4.html` also define `--warm` and `--cool` in their own `<style>` blocks and read those for the
#: line charts. So the same idea is expressed twice in one file, once as a token and once as an
#: array. It is not a defect of the kind this guard exists for — a class colour is semantics, and
#: the interpolated decision surface writes into an `ImageData` buffer where a CSS variable cannot
#: go — but the inconsistency is real and belongs on a list rather than in someone's memory.
RECORDED_INCONSISTENCY = (
    "01-introductions: the two class colours are RGB triples in JS while `--warm`/`--cool` exist "
    "as tokens in the same files and are used for the line charts. Tokenising them means parsing a "
    "CSS colour to interpolate the decision surface per pixel, which is why it was not done here."
)


def _drawing_files() -> list[Path]:
    """Every deployed file that could paint on a canvas.

    Exercise 01 inlines its scripts in `.html`, and every other exercise builds pages from `.js`, so
    both extensions are swept. Reading the filesystem rather than a list means a new canvas is
    covered the day it ships — which matters more here than usual, since the whole failure mode is a
    guard that was written for the pages that existed when it was written.
    """
    return sorted(
        path
        for pattern in ("*/web/**/*.html", "*/web/**/*.js")
        for path in (REPO_ROOT / "src" / "exercises").glob(pattern)
    )


def test_some_file_paints_on_a_canvas() -> None:
    """Otherwise the sweep below passes by having nothing to look at."""
    painting = [
        p for p in _drawing_files() if ANY_COLOUR_WRITE.search(p.read_text(encoding="utf-8"))
    ]
    assert painting, (
        "no deployed file assigns a canvas colour at all. Either nothing draws any more, or the "
        "pattern has stopped matching how drawing is written — and the second is far more likely."
    )


def test_no_canvas_colour_is_written_as_a_literal() -> None:
    """One assertion over every file, because the value of the report is the whole list.

    Split per file, the first failure hides the rest and the shape of the problem — three files,
    twelve literals, all in the one exercise that draws — only appears on the third run.
    """
    offenders: list[str] = []
    for path in _drawing_files():
        text = path.read_text(encoding="utf-8")
        for match in LITERAL_COLOUR.finditer(text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0)}")

    assert not offenders, (
        f"{len(offenders)} canvas colour(s) are written as literals, so no theme can move them:\n  "
        + "\n  ".join(offenders)
        + "\nThis site has six themes and every colour in them was generated against a contrast "
        "checker. A literal is right in the theme it was typed for and wrong in the others, and no "
        "CSS-reading guard can see it, because it lives inside a 2D context. Read the token "
        "instead — `getComputedStyle(document.body).getPropertyValue('--ink').trim()` — and carry "
        "any transparency with `ctx.globalAlpha`, restoring it to 1 afterwards."
    )


def test_the_recorded_inconsistency_still_describes_the_repository() -> None:
    """A note about untidiness that has since been tidied is worse than no note.

    It sends the next reader looking for something that is not there, and it makes every other note
    beside it a little less trustworthy. So the two halves of the claim are asserted: the class
    colours are still RGB triples, and the tokens they duplicate still exist.
    """
    files = [
        REPO_ROOT / "src/exercises/01-introductions/web" / name
        for name in ("s1.html", "s2.html", "s4.html")
    ]
    triples = sum(
        len(re.findall(r"\[\s*\d{2,3},\s*\d{2,3},\s*\d{2,3}\s*\]", f.read_text())) for f in files
    )
    tokens = sum("--warm:" in f.read_text() or "--cool:" in f.read_text() for f in files)

    assert triples and tokens, (
        f"RECORDED_INCONSISTENCY no longer describes the repository — {triples} colour triples and "
        f"{tokens} of the files defining `--warm`/`--cool`. If the class colours have been "
        "tokenised, delete the note; a stale note is worse than none.\n\n" + RECORDED_INCONSISTENCY
    )
