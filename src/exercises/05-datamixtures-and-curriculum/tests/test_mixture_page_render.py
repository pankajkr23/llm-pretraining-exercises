"""The page, tested in a browser rather than parsed.

`node --check` proves a file has no syntax error and nothing more. A call to an undefined function,
a slider wired to nothing, a colour token that does not exist, and a headline reading `0` all parse
perfectly. This suite loads the built site in Chromium and asserts what a reader sees.

**It serves `public/`, not `web/`.** The palette lives in the *site-root* `/_shared/tokens.css`,
which only exists once `deploy/vercel/build.sh` has assembled the site — serving `web/` directly
would test a page with no colours and no fonts, which is what exercise 04 did for a while without
noticing. The fixture builds the site if it is missing.

Integration-marked, and skips cleanly when Chromium is absent so a fresh checkout still works. That
means it protects the page silently or not at all: `uv run playwright install chromium` once.
"""

import http.server
import re
import socket
import subprocess
import threading

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

from mixture import export  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = export.EXERCISE_ROOT.parents[2]
PUBLIC = REPO / "public"
SLUG = "05-datamixtures-and-curriculum"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def site() -> str:
    """Serve the assembled site and yield its base URL.

    Assembles it first when absent: `public/` is gitignored, so a fresh checkout and CI both start
    without one, and a suite that skipped in that case would protect nothing on the machine that
    matters most.
    """
    if not (PUBLIC / SLUG / "index.html").exists():
        script = REPO / "deploy" / "vercel" / "build.sh"
        if not script.exists():
            pytest.skip("no build script; cannot assemble the site under test")
        subprocess.run(["bash", str(script)], check=True, capture_output=True)
    if not (PUBLIC / "_shared" / "tokens.css").exists():
        pytest.skip("the assembled site has no root stylesheet; the page under test would be bare")

    handler = type(
        "Handler",
        (http.server.SimpleHTTPRequestHandler,),
        {
            "__init__": lambda self, *a, **kw: http.server.SimpleHTTPRequestHandler.__init__(
                self, *a, directory=str(PUBLIC), **kw
            ),
            "log_message": lambda *a, **kw: None,
        },
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/{SLUG}/"
    finally:
        server.shutdown()


@pytest.fixture(scope="module")
def page(site):
    """A loaded page, with any console error recorded."""
    with sync_playwright() as play:
        try:
            browser = play.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no chromium available: {exc}")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        p = context.new_page()
        errors: list[str] = []
        p.on("pageerror", lambda e: errors.append(str(e)))
        p.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        p.goto(site, wait_until="networkidle")
        p.wait_for_selector("section", timeout=10_000)
        p.errors = errors  # type: ignore[attr-defined]
        yield p
        browser.close()


# ---- it renders at all -----------------------------------------------------------------------


def test_the_page_renders_every_chapter_without_throwing(page):
    sections = page.query_selector_all("section")
    assert len(sections) >= 5, f"only {len(sections)} chapters rendered"
    assert not page.query_selector_all(".err"), "a chapter reported an error on the page"
    assert not page.errors, f"console errors: {page.errors[:3]}"


def test_the_lede_placeholders_were_filled(page):
    """`data-fact` spans ship with placeholder text. Unfilled, the page reads as a draft."""
    for el in page.query_selector_all("[data-fact]"):
        text = el.inner_text().strip()
        assert text, f"lede fact {el.get_attribute('data-fact')} is empty"
        assert "…" not in text


def test_no_headline_figure_reads_as_nothing(page):
    """Exercise 04's lesson: a headline reading 0 is a wrong question, not a caption problem."""
    for el in page.query_selector_all(".bignum-v"):
        text = el.inner_text().strip()
        assert text not in {"", "0", "—", "NaN", "undefined", "null"}, (
            f"a headline figure reads {text!r}"
        )


def test_no_undefined_or_nan_leaks_into_the_page(page):
    body = page.inner_text("body")
    for bad in ("undefined", "NaN", "[object Object]"):
        assert bad not in body, f"{bad!r} is visible on the page"


def test_the_rail_lists_every_chapter_with_its_title(page):
    """The bug this pins: exercise 04's rail stripped a leading token from `h2.textContent`, which
    runs the number and title together, and ate the first word of every label.
    """
    links = page.query_selector_all(".rail-link .rail-t")
    assert len(links) >= 5
    titles = {el.inner_text().strip() for el in links}
    assert "Out of what?" in titles, f"rail titles are {titles}"
    assert all(t and not t[0].isdigit() for t in titles), f"a rail label kept its number: {titles}"


# ---- the design system actually applied -------------------------------------------------------


def test_the_root_palette_is_loaded(page):
    """Colour tokens live in the *site-root* stylesheet, not the per-exercise copy.

    If this fails the page is being served from `web/` rather than the assembled `public/`, and
    every semantic colour on it is inherited nothing.
    """
    accent = page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()"
    )
    assert accent, "--accent is undefined; the root stylesheet did not load"


def test_every_token_the_stylesheet_references_is_defined(page):
    """An undefined custom property fails silently, so every reference is checked, not a hand list.

    This page used invented names (`--good`, `--warn`, `--bad`) at first, and every verdict badge
    rendered with no colour while looking perfectly fine.

    The first version of this test named the tokens it expected and asserted *those* were defined.
    Mutation testing killed it: swapping a `var(--grade-x)` usage for `var(--nonexistent)` left
    `--grade-x` defined, so the test passed against a stylesheet referencing a token that does not
    exist. It now reads the references out of the stylesheet, which is the thing that can go wrong.
    """
    css = (export.EXERCISE_ROOT / "web" / "page-extra.css").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"var\((--[a-z0-9-]+)", css)))
    assert referenced, "no custom properties found; the pattern or the stylesheet has moved"

    undefined = [
        token
        for token in referenced
        if not page.evaluate(
            f"getComputedStyle(document.documentElement).getPropertyValue('{token}').trim()"
        )
    ]
    assert not undefined, f"the stylesheet styles with undefined tokens: {undefined}"


def test_verdict_badges_are_actually_coloured(page):
    """The consequence of the above, checked on a rendered element rather than on the stylesheet."""
    badge = page.query_selector(".verdict")
    assert badge, "no verdict badge rendered"
    colour = page.evaluate("el => getComputedStyle(el).color", badge)
    assert colour not in ("", "rgba(0, 0, 0, 0)"), f"badge colour is {colour!r}"


# ---- layout ----------------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", [(1500, 900), (900, 800), (390, 844)])
def test_the_page_never_scrolls_sideways(page, width, height):
    """A page that scrolls horizontally on a phone is broken, and it is easy not to notice.

    Exercise 04 shipped 312px of sideways scroll from invisible absolutely-positioned tooltips.
    """
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(150)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"{overflow}px of horizontal scroll at {width}x{height}"


def test_wide_tables_scroll_inside_their_own_container(page):
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(150)
    wrappers = page.query_selector_all(".tblwrap")
    assert wrappers, "no tables rendered"
    for wrapper in wrappers:
        style = page.evaluate("el => getComputedStyle(el).overflowX", wrapper)
        assert style in ("auto", "scroll"), f"a table wrapper has overflow-x: {style}"


# ---- the interactions actually do something ---------------------------------------------------


def test_dragging_a_lane_changes_the_other_lanes(page):
    """A slider wired to nothing looks identical to one that works."""
    page.set_viewport_size({"width": 1280, "height": 900})
    shares = page.query_selector_all("#composer .compose-share")
    before = [el.inner_text() for el in shares]

    slider = page.query_selector("#composer input[type=range]")
    slider.fill("600")
    slider.dispatch_event("input")
    page.wait_for_timeout(120)

    after = [el.inner_text() for el in page.query_selector_all("#composer .compose-share")]
    assert before != after, "moving a slider changed nothing"
    changed = sum(1 for a, b in zip(before, after, strict=True) if a != b)
    assert changed >= 2, "only the dragged lane moved; the others did not renormalise"


def test_starving_a_protected_lane_shows_the_breach(page):
    """The claim the composer exists to prove: a floor is visible when it is crossed."""
    page.click("#composer .btn.ghost")  # the "crawl what is cheap" preset
    page.wait_for_timeout(150)
    assert page.query_selector("#composer .compose-row.breached"), (
        "the naive preset starves Indic below its floor and the page did not say so"
    )
    summary = page.inner_text("#composer .compose-summary")
    assert "floor" in summary.lower()


def test_the_reset_button_restores_the_specification(page):
    page.click("#composer .btn:not(.ghost)")
    page.wait_for_timeout(150)
    assert not page.query_selector("#composer .compose-row.breached")


def test_the_repetition_slider_moves_both_bars(page):
    seen = page.query_selector("#repetition .rep-bar.seen")
    worth = page.query_selector("#repetition .rep-bar.worth")
    widths = lambda: (  # noqa: E731
        page.evaluate("el => el.style.width", seen),
        page.evaluate("el => el.style.width", worth),
    )
    before = widths()
    slider = page.query_selector("#repetition input[type=range]")
    slider.fill("40")
    slider.dispatch_event("input")
    page.wait_for_timeout(120)
    assert widths() != before, "the repetition chart did not respond"


def test_the_tier_toggle_moves_the_hole_rather_than_filling_it(page):
    """The point of that interaction: if both readings showed no gap it would prove the reverse."""
    first = page.inner_text("#tiers .tier-out")
    buttons = page.query_selector_all("#tiers .toggle .btn")
    assert len(buttons) == 2
    buttons[1].click()
    page.wait_for_timeout(150)
    second = page.inner_text("#tiers .tier-out")
    assert first != second, "the tier toggle changed nothing"
    for text in (first, second):
        assert "must be generated" in text.lower() or "generated" in text.lower()


def test_the_results_chapter_shows_the_seed_spread_by_default(page):
    """Hiding it is the interaction; showing it is the default, because that is the honest view."""
    scores = page.inner_text("#results")
    assert "±" in scores, "the seed spread is not shown by default"
    page.click("#results .btn.ghost")
    page.wait_for_timeout(120)
    assert "±" not in page.inner_text("#results table"), "hiding the spread did nothing"


def test_the_qualified_verdict_is_on_the_page_with_its_reason(page):
    """The result that cost the specification a clean sweep has to be visible, not buried."""
    text = page.inner_text("#results")
    assert "qualified" in text.lower()
    assert "second clause" in text.lower()
