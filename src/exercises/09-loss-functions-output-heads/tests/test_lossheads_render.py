"""Exercise 09's page, tested in a browser, because `node --check` proves almost nothing about it.

A call to an undefined function, a table that renders every cell as `undefined`, a figure reading
`NaN`, and a layout that scrolls sideways on a phone all parse perfectly. `AGENTS.md` names this as
a rule the repo learned by shipping it.

**Served, not opened as a `file://`.** ES modules refuse to load over `file://`, and the shell links
`/_shared/tokens.css` from the site root — so a `file://` test renders a blank, unstyled page and
passes any assertion that only checks the title.
"""

import functools
import http.server
import os
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
PUBLIC = REPO / "public"
SLUG = "09-loss-functions-output-heads"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def page():
    """Serve the assembled site and open the exercise page once for the whole module."""
    if os.environ.get("PYTEST_XDIST_WORKER") and not (PUBLIC / SLUG / "index.html").exists():
        pytest.fail(
            "running under -n with no assembled site. `build.sh` would race across workers "
            "(it begins `rm -rf public/`). Run `bash deploy/vercel/build.sh` first."
        )
    if not (PUBLIC / SLUG / "index.html").exists():
        script = REPO / "deploy" / "vercel" / "build.sh"
        if not script.exists():
            pytest.skip("no build script; cannot assemble the site under test")
        subprocess.run(["bash", str(script)], check=True, capture_output=True)
    if not (PUBLIC / "_shared" / "tokens.css").exists():
        pytest.skip("the assembled site has no root stylesheet; the page under test would be bare")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # no browser installed, or a sandbox blocking it
                pytest.skip(f"chromium unavailable: {exc}")
            view = browser.new_page(viewport={"width": 1280, "height": 900})
            problems: list[str] = []
            view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            view.goto(f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html")
            view.wait_for_selector("section#reproduce", timeout=10_000)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


def test_the_page_loads_without_console_errors(page):
    """A page that throws halfway through renders its first half and looks fine."""
    assert page.console_problems == []


#: The spine the page must have, in order. Roles rather than ids, so wording can change freely.
REQUIRED_ROLES = (
    "thesis",
    "glossary",
    "problem",
    "mechanism",
    "method",
    "expected",
    "results",
    "negatives",
    "conclusion",
    "limits",
    "next",
    "reproduce",
)


def test_the_page_has_the_required_spine_in_order(page):
    """A reader arriving cold must be able to find every part of the story, in a sensible order.

    Checked by `data-role` rather than by heading text, so the prose stays free to change. The order
    is asserted because "limits" before "results" would read as hedging, and "conclusion" before the
    evidence would read as a press release.
    """
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"
    seen = [r for r in roles if r in REQUIRED_ROLES]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(REQUIRED_ROLES), f"the spine is out of order: {first}"


def test_every_figure_has_a_caption_that_says_something(page):
    """A figure with a bare label makes the reader do the interpreting. Captions here state what to
    conclude, so a short one is a caption that has not done its job."""
    caps = page.eval_on_selector_all("figure figcaption", "els => els.map(e => e.innerText)")
    figs = page.evaluate("() => document.querySelectorAll('figure').length")
    assert len(caps) == figs > 0, f"{figs} figures but {len(caps)} captions"
    short = [c[:40] for c in caps if len(c) < 120]
    assert not short, f"these captions are too short to be doing any work: {short}"


def test_no_element_is_truncated_at_any_width(page):
    """Visible is not legible, and this repo has shipped the difference.

    Exercise 08's invoice cut line — the sentence its whole figure existed to deliver — was
    `white-space: nowrap` inside `overflow: hidden`, so it read "…the cache alone needs a second ma"
    at every width narrower than itself. Its visibility test passed the entire time.

    The general property is cheap and catches the whole class: nothing may have a `scrollWidth`
    larger than its `clientWidth`. One pixel is allowed for sub-pixel rounding.
    """
    for width in (1440, 1180, 768):
        page.set_viewport_size({"width": width, "height": 900})
        clipped = page.evaluate(
            """() => [...document.querySelectorAll('main *')]
                 // HTML boxes only. scrollWidth/clientWidth are HTML box properties; on an SVG
                 // <text> they report something unrelated to whether the glyphs are clipped, and
                 // this guard's first run flagged four labels that render perfectly.
                 .filter(e => e instanceof HTMLElement)
                 .filter(e => e.scrollWidth > e.clientWidth + 1 &&
                              getComputedStyle(e).overflowX !== 'auto' &&
                              getComputedStyle(e).overflowX !== 'scroll')
                 .map(e => `${e.tagName.toLowerCase()}.${e.className}: ` +
                           `${e.scrollWidth} > ${e.clientWidth}`)"""
        )
        assert not clipped, f"at {width}px these are cut off rather than merely narrow: {clipped}"
    page.set_viewport_size({"width": 1280, "height": 900})


def test_no_svg_label_spills_outside_its_own_figure(page):
    """The SVG half of the same property, asserted the way SVG actually works.

    An `<svg>` scales to its `viewBox`, so a label is clipped when its rendered box escapes that
    box — not when `scrollWidth` exceeds `clientWidth`, which reports something unrelated here.
    """
    spills = page.evaluate(
        """() => {
             const out = [];
             for (const svg of document.querySelectorAll('main svg')) {
               const box = svg.viewBox.baseVal;
               if (!box || !box.width) continue;
               for (const t of svg.querySelectorAll('text')) {
                 const b = t.getBBox();
                 if (b.x < box.x - 1 || b.x + b.width > box.x + box.width + 1) {
                   out.push(`${t.textContent}: ${b.x.toFixed(0)}..${(b.x + b.width).toFixed(0)} ` +
                            `outside 0..${box.width}`);
                 }
               }
             }
             return out;
           }"""
    )
    assert not spills, f"these labels render outside their figure: {spills}"


def test_no_token_label_overflows_the_box_it_sits_in(page):
    """The guard that would have caught the defect the first version of Figure 1 shipped.

    Its cell width was a guessed constant and five of seven token strings rendered past the edge of
    their own rounded box — "apital → apita", "_India → _Indi". Nothing caught it:
    `scrollWidth`/`clientWidth` mean nothing on an SVG `<text>`, and a viewBox check only sees the
    figure's outer edge, which the text stayed inside.

    So the property has to be stated at the right scale: each label against **its own rect**.
    """
    overflows = page.evaluate(
        """() => {
             const out = [];
             for (const svg of document.querySelectorAll('main svg.shiftfig')) {
               const rects = [...svg.querySelectorAll('rect.tok')];
               for (const t of svg.querySelectorAll('text.tok-t')) {
                 const b = t.getBBox();
                 const mid = b.x + b.width / 2;
                 const box = rects.find(r => {
                   const x = parseFloat(r.getAttribute('x'));
                   const w = parseFloat(r.getAttribute('width'));
                   const y = parseFloat(r.getAttribute('y'));
                   return mid >= x && mid <= x + w && b.y >= y - 4 && b.y <= y + 40;
                 });
                 if (!box) continue;
                 const x = parseFloat(box.getAttribute('x'));
                 const w = parseFloat(box.getAttribute('width'));
                 if (b.x < x - 0.5 || b.x + b.width > x + w + 0.5) {
                   out.push(`"${t.textContent}" is ${b.width.toFixed(0)} wide in a ${w} box`);
                 }
               }
             }
             return out;
           }"""
    )
    assert not overflows, f"these labels render outside their own box: {overflows}"


def test_the_page_never_scrolls_sideways(page):
    """A body that scrolls horizontally is the phone-sized version of the same defect."""
    for width in (1440, 1180, 768, 390):
        page.set_viewport_size({"width": width, "height": 900})
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 1, f"the page scrolls {overflow}px sideways at {width}px"
    page.set_viewport_size({"width": 1280, "height": 900})


def test_the_rail_marks_where_the_reader_is(page):
    """`.rail-link.on` has been styled since before any of these pages existed.

    Only exercise 03 ever set it, so 05, 06 and 07 all ship a rail that never says where you are —
    styled, inert, and indistinguishable from a working one unless somebody looks. This asserts
    that the class actually lands on exactly one link once the reader has moved.
    """
    page.evaluate("() => document.getElementById('results').scrollIntoView()")
    page.wait_for_timeout(400)
    marked = page.eval_on_selector_all(
        ".rail-link.on", "els => els.map(e => e.getAttribute('href'))"
    )
    assert len(marked) == 1, f"expected exactly one marked rail entry, got {marked}"


def test_the_gutter_the_shared_stylesheet_reserves_is_actually_filled(page):
    """`_shared/page.css` reserves 260px for a rail on every wide screen, whether or not one exists.

    Exercises 06 and 07 vendored it without building the element, so both render an empty 260px
    column at every width above 1180px and nothing fails. The pairing is what has to be asserted:
    reserved **and** filled.
    """
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(200)
    measured = page.evaluate(
        """() => {
             const wrap = document.querySelector('.wrap');
             const rail = document.querySelector('#rail');
             const inner = document.querySelector('.rail-inner');
             return {
               reserved: parseFloat(getComputedStyle(wrap).paddingLeft),
               rail: rail ? rail.getBoundingClientRect().width : 0,
               inner: inner ? inner.getBoundingClientRect().height : 0,
             };
           }"""
    )
    page.set_viewport_size({"width": 1280, "height": 900})
    if measured["reserved"] < 100:
        pytest.skip("this width reserves no rail gutter, so there is nothing to fill")
    assert measured["rail"] > 100, (
        f"{measured['reserved']:.0f}px of gutter is reserved and the rail is "
        f"{measured['rail']:.0f}px wide — that is an empty column"
    )
    assert measured["inner"] > 0, (
        ".rail-inner is missing. The shared stylesheet centres the rail with "
        "`.rail-inner { margin-block: auto }`, so without it the contents hang at the top — which "
        "is exactly what exercise 08 shipped."
    )


def test_every_number_on_the_page_came_from_the_run(page):
    """The page must not invent a figure, and `undefined`/`NaN` is what it looks like when it does.

    A missing key in the generated data file renders as the string `undefined` in perfectly valid
    HTML. Nothing else on this page would notice.
    """
    text = page.inner_text("main")
    for poison in ("undefined", "NaN", "[object Object]"):
        assert poison not in text, f"the page rendered {poison!r} — a figure came from nowhere"


def test_no_prose_table_hides_a_cell_behind_its_own_scrollbar(page):
    """A table of sentences must fit its container. A table of figures may scroll.

    **The page-level overflow guard is blind to this and cannot be fixed to see it.** Every table
    sits in a `.tablewrap` with `overflow-x: auto`, which is exactly what `AGENTS.md` asks for —
    wide content scrolls in its own container rather than pushing the page sideways. So a cell
    hidden inside that scroll is, to any generic check, indistinguishable from a wide table behaving
    correctly.

    It was not correct here. `table.grid` sets `white-space: nowrap` on every cell but the first,
    which is right for a column of numbers and wrong for a column of prose: the corrections table
    ran roughly two hundred characters past its right edge, and the column it took with it was the
    one carrying the transferable lesson from each row. Marking a table `prose` says its cells wrap;
    this asserts that they do.
    """
    offenders = page.evaluate("""() => {
      const bad = [];
      for (const t of document.querySelectorAll('table.prose')) {
        const wrap = t.closest('.tablewrap');
        if (wrap && t.scrollWidth > wrap.clientWidth + 1) {
          bad.push(`${t.closest('section').id}: ${t.scrollWidth} > ${wrap.clientWidth}`);
        }
      }
      return bad;
    }""")
    assert not offenders, (
        f"a prose table is wider than its container, so part of every row is behind a scrollbar: "
        f"{offenders}. Prose cells must wrap — check the `prose` class is on the table."
    )


def test_no_two_figures_carry_the_same_number(page):
    """Figure numbers must be distinct and consecutive from one.

    They were positional arguments, so adding a figure at the top of the page produced **two Figure
    1s** — the new one and the mechanism plate, in the same document, each captioned "Figure 1".
    Nothing failed: a caption is prose to every other guard here, and a duplicate number reads as a
    typo rather than as a broken cross-reference until someone tries to cite one.
    """
    numbers = page.evaluate("""() => [...document.querySelectorAll('figcaption b')]
        .map((b) => b.textContent.trim())
        .filter((t) => t.startsWith('Figure '))
        .map((t) => parseInt(t.slice(7), 10))""")
    assert numbers, "no numbered figures found; the caption format has changed"
    assert numbers == sorted(numbers), f"figure numbers are out of order: {numbers}"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"figure numbers are {numbers}, which is not 1..{len(numbers)}. Derive them from the count "
        "of figures already built rather than passing each one in."
    )


def test_every_glossary_entry_carries_a_number_from_the_run(page):
    """The glossary says every entry carries a real figure. That is checkable, so it is checked.

    **This replaces a claim that was not.** The page used to promise that every term the opening
    tiles used was defined here, and it was false twice — first for `head`, `logits`, `output head`
    and `tokenizer`, then, after those were added, for `packed` and `projection`. The obvious guard
    for it does not work: the tiles emphasise words for stress as often as for terminology, so a
    check keyed on emphasis flags `broken` and `estimated` and cannot tell a term from a raised
    voice. I tried it, watched it fire on correct prose, and removed the promise instead.

    What is left is a promise the page can keep. A definition carrying a figure from this run is the
    difference between a glossary and a dictionary — it is what makes `perplexity` mean *12,078
    here* rather than a paraphrase — and it is the property that decays first when an entry is
    added in a hurry.
    """
    import re as _re

    entries = page.evaluate("""() => {
      const out = [];
      const dl = document.querySelector('.gloss');
      if (!dl) return out;
      const kids = [...dl.children];
      for (let i = 0; i < kids.length; i += 1) {
        if (kids[i].tagName !== 'DT') continue;
        const dd = kids[i + 1];
        if (dd && dd.tagName === 'DD') {
          out.push({term: kids[i].textContent.trim(), body: dd.textContent.trim()});
        }
      }
      return out;
    }""")
    assert entries, "no glossary entries found; the selector has gone stale"

    numberless = [e["term"] for e in entries if not _re.search(r"\d", e["body"])]
    assert not numberless, (
        f"these glossary entries carry no figure from the run: {numberless}. The section promises "
        "that every entry does, and a definition without one is a dictionary entry — it tells a "
        "reader what a word means in general rather than what it is on this page."
    )


def _decimals_for(spread: float) -> int:
    """Decimals a spread will support — the rule the page and the renderer both implement.

    Written a third time here on purpose. A guard that imports the implementation it is checking
    asserts only that the implementation equals itself; this one states the rule independently, so
    it goes red when either copy drifts from it.

    Args:
        spread: The measured spread of the quantity being quoted.

    Returns:
        Decimal places, 0 to 4.
    """
    import math

    if spread <= 0:
        return 4
    return min(4, max(0, math.floor(-math.log10(spread))))


def test_no_ratio_on_the_page_is_quoted_finer_than_its_own_spread_supports(page):
    """The page's stated precision rule, asserted against the page rather than against the code.

    **This is the guard the previous fix did not have, and the fix shipped broken without it.** The
    page says the memory ratio is quoted "and no finer" than its noise floor allows. That sentence
    was broken by five hand-chosen `toFixed` calls; the repair replaced them with a function — and
    the function's thresholds were hand-chosen too, so the page kept offering a tenth against a
    spread of 0.44, one paragraph below the promise. Every test was green, because every test
    checked the README or the results file and none of them read the rendered page.

    So this reads the page. For each ratio drawn from a repeated measurement it asserts two things:
    the figure at the precision its own spread earns is **present**, and no **finer** rendering of
    the same value appears anywhere on the page. The second half is the one that fails on the bug —
    a page that quotes 9x in one place and 9.09x in another satisfies presence and breaks the rule.

    It also pins the pairing. The softmax-only ratio was quoted against the *memory* ratio's spread:
    an absolute 0.44 measured on a value of 9, applied to a value of 1.8. Both spreads are measured
    now, they differ by roughly thirty times, and passing the wrong one reds this test.
    """
    import json
    import re as _re

    results = REPO / "src/exercises/09-loss-functions-output-heads/results"
    harness = json.loads((results / "harness.json").read_text())["item_7_memory"]
    memory = json.loads((results / "sensitivity.json").read_text())["memory"]

    text = page.evaluate("() => document.body.innerText")

    def _appears(figure: str) -> bool:
        """Is this exact figure on the page, rather than the tail of a longer one?

        **A substring match reported a defect that was not there.** The first version of this asked
        whether `"9.1×"` was in the page text, and the page carries `39.1×` — the logits-to-hidden
        ratio, a different quantity in a different section — so the guard failed against correct
        prose. The lookbehind is the whole fix: a figure preceded by a digit or a point is part of
        a larger number and is not this one.
        """
        return _re.search(rf"(?<![\d.]){_re.escape(figure)}", text) is not None

    pairs = (
        ("the memory ratio", harness["ratio"], memory["spread"]),
        ("the softmax-only ratio", harness["softmax_only_ratio"], memory["softmax_only_spread"]),
    )
    for name, value, spread in pairs:
        earned = _decimals_for(spread)
        assert _appears(f"{value:.{earned}f}×"), (
            f"{name} is measured to a spread of {spread:.4f}, which earns {earned} decimals — so "
            f"the page should quote it as {value:.{earned}f}×, and that figure is not on the page."
        )
        for finer in range(earned + 1, 5):
            assert not _appears(f"{value:.{finer}f}×"), (
                f"{name} appears on the page as {value:.{finer}f}×, which is {finer - earned} "
                f"digit(s) finer than its measured spread of {spread:.4f} supports. The page's own "
                f"results section promises it is quoted 'and no finer'. Every digit past "
                f"{earned} is noise being published as a measurement."
            )
