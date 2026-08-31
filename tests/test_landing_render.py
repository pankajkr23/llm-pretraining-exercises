"""The site's front door, checked in a browser.

The landing page was a single 640px column at every viewport width, so on a 1920px screen it was a
tall narrow ribbon between two empty margins — a third of the screen used. Widening it naively is
the wrong fix, because a 1200px line of prose is unreadable. The page is now **two measures**: the
header keeps a readable line length, and the exercise cards become a responsive grid.

These tests pin both halves, because each is easy to break by fixing the other.
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

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "public"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def page():
    """Serve the assembled site and open the landing page."""
    if os.environ.get("PYTEST_XDIST_WORKER") and not (PUBLIC / "index.html").exists():
        pytest.fail(
            "running under -n with no assembled site. `build.sh` begins `rm -rf public/` and would "
            "race across workers. Run `bash deploy/vercel/build.sh` first."
        )
    if not (PUBLIC / "index.html").exists():
        script = REPO / "deploy" / "vercel" / "build.sh"
        if not script.exists():
            pytest.skip("no build script; cannot assemble the site under test")
        subprocess.run(["bash", str(script)], check=True, capture_output=True)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # no browser installed, or a sandbox blocking it
                pytest.skip(f"chromium unavailable: {exc}")
            view = browser.new_page(viewport={"width": 1440, "height": 1000})
            view.goto(f"http://127.0.0.1:{httpd.server_address[1]}/")
            view.wait_for_selector("a.item", timeout=10_000)
            yield view
            browser.close()
    finally:
        httpd.shutdown()


def _rows(page):
    """Card heights grouped by the row they sit on."""
    return page.evaluate(
        """() => {
            const by = {};
            for (const e of document.querySelectorAll('a.item')) {
                const r = e.getBoundingClientRect();
                const k = Math.round(r.top);
                (by[k] = by[k] || []).push(Math.round(r.height));
            }
            return Object.values(by);
        }"""
    )


@pytest.mark.parametrize("width", [1920, 1440, 1180, 900, 390, 320])
def test_the_landing_page_never_scrolls_sideways(page, width):
    """`min(340px, 100%)` on the grid track exists for the 320px case; this is what proves it."""
    page.set_viewport_size({"width": width, "height": 1000})
    page.wait_for_timeout(200)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"the landing page scrolls sideways by {overflow}px at {width}px"


def test_the_cards_form_more_than_one_column_on_a_wide_screen(page):
    """The original defect: one 640px column at every width, whatever the screen."""
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.wait_for_timeout(200)
    widest = max(len(r) for r in _rows(page))
    assert widest >= 2, f"the cards are still a single column at 1440px ({widest} per row)"


def test_the_cards_collapse_to_one_column_on_a_phone(page):
    """The opposite failure: a grid that keeps its columns and squeezes the text to nothing."""
    page.set_viewport_size({"width": 390, "height": 1000})
    page.wait_for_timeout(200)
    assert all(len(r) == 1 for r in _rows(page)), "the cards do not stack on a narrow screen"


def test_cards_in_a_row_share_a_height(page):
    """Ragged bottoms read as an accident. `align-items: stretch` plus a bottom-pinned meta line."""
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.wait_for_timeout(200)
    for row in _rows(page):
        assert len(set(row)) == 1, f"a row ends at {len(set(row))} different heights: {row}"


def test_the_index_number_does_not_stretch(page):
    """The exact bug this layout introduced once, caught by geometry.

    `.idx` is also a direct-child `span`, so a bare `a.item > span { flex: 1 }` gives it flex-grow
    too and it fills the card — measured 93px tall for one line — pushing the title into the middle.
    """
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.wait_for_timeout(200)
    heights = page.eval_on_selector_all(
        "a.item .idx", "els => els.map(e => Math.round(e.getBoundingClientRect().height))"
    )
    assert heights, "no index labels found"
    assert all(h < 40 for h in heights), f"an index label is stretching: {heights}"

    gaps = page.evaluate(
        """() => [...document.querySelectorAll('a.item')].map(e => {
            const i = e.querySelector('.idx').getBoundingClientRect();
            const h = e.querySelector('h2').getBoundingClientRect();
            return Math.round(h.top - i.bottom);
        })"""
    )
    assert all(g < 30 for g in gaps), f"a title floats away from its index: {gaps}"


def test_the_prose_keeps_a_readable_measure(page):
    """Widening the page must not widen the sentences. This is the half a naive fix breaks.

    Roughly 75 characters is the upper bound for comfortable reading; at the page's 15px base that
    is about 640px. The lede is capped at 52ch by CSS, and this checks the rendered result.
    """
    page.set_viewport_size({"width": 1920, "height": 1000})
    page.wait_for_timeout(200)
    lede = page.eval_on_selector(".lede", "el => el.getBoundingClientRect().width")
    assert lede < 700, f"the lede is {lede:.0f}px wide — too long a line to read comfortably"


def test_the_grid_actually_uses_the_screen(page):
    """The complaint that started this: a third of a wide screen used, the rest empty margin."""
    page.set_viewport_size({"width": 1920, "height": 1000})
    page.wait_for_timeout(200)
    share = page.evaluate(
        """() => {
            const r = document.querySelector('.list').getBoundingClientRect();
            return r.width / window.innerWidth;
        }"""
    )
    assert share > 0.5, f"the card grid uses only {share:.0%} of a 1920px viewport"
