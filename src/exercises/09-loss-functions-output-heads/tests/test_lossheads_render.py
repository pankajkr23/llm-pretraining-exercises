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
