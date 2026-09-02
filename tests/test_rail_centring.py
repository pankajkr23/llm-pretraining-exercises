"""The pinned rail's contents must be vertically centred, on every page that builds a rail.

**A stylesheet cannot centre an element the page never creates.** `_shared/page.css` makes the
pinned rail a full-height flex column and centres its contents with
`.rail-inner { margin-block: auto }` — a rule that needs a wrapper each page has to add itself.
Exercises 03, 05, 06 and 07 all add it. Exercise 08 did not, so its contents hung at the top of a
1,100px column while every sibling page sat centred, and it looked wrong beside them for a reason
no existing test could see and no console error reported.

This is the third time this exact shape has cost this repo something: the shared stylesheet also
reserves a 260px gutter that only some pages fill, and vendors marks whose colours only resolve
when the real token file is linked. Vendoring `web/_shared/` copies the styles and not the markup
they assume. So this guard is repo-wide and lexical about which pages it applies to: any page that
builds a rail is held to it, discovered from the filesystem rather than from a list someone
maintains.
"""

import functools
import http.server
import re
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC = REPO_ROOT / "public"

pytestmark = pytest.mark.integration


def _pages_that_build_a_rail() -> list[str]:
    """Read it off the filesystem, so a new railed page is covered the day it lands."""
    found = []
    for chapters in sorted((REPO_ROOT / "src" / "exercises").glob("*/web/chapters.js")):
        text = chapters.read_text(encoding="utf-8")
        if re.search(r"getElementById\(['\"]rail['\"]\)", text):
            found.append(chapters.parents[1].name)
    return found


@pytest.fixture(scope="module")
def browser():
    if not (PUBLIC / "index.html").is_file():
        pytest.skip("run deploy/vercel/build.sh first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            yield b, f"http://127.0.0.1:{httpd.server_address[1]}"
            b.close()
    finally:
        httpd.shutdown()


def test_some_page_builds_a_rail() -> None:
    """Otherwise the parametrised guard below is vacuous and would pass silently."""
    assert _pages_that_build_a_rail(), "no page builds a rail — this whole file tests nothing"


@pytest.mark.parametrize("slug", _pages_that_build_a_rail())
def test_the_rail_contents_are_vertically_centred(browser, slug: str) -> None:
    """Equal air above and below the contents, in the wrapper the shared stylesheet centres."""
    b, base = browser
    if not (PUBLIC / slug / "index.html").is_file():
        pytest.skip(f"{slug} is not published")
    page = b.new_page(viewport={"width": 1500, "height": 1100})
    try:
        page.goto(f"{base}/{slug}/", wait_until="networkidle")
        page.wait_for_timeout(1200)
        m = page.evaluate(
            """() => {
              const rail = document.querySelector('.rail');
              if (!rail) return null;
              if (getComputedStyle(rail).position !== 'fixed') return {skip: true};
              const inner = document.querySelector('.rail-inner');
              if (!inner) return {missing: true};
              const r = rail.getBoundingClientRect(), i = inner.getBoundingClientRect();
              return {above: i.top - r.top, below: r.bottom - i.bottom};
            }"""
        )
        assert m is not None, f"{slug} builds a rail but none is in the DOM"
        if m.get("skip"):
            pytest.skip(f"{slug}'s rail is not pinned at this width")
        assert not m.get("missing"), (
            f"{slug} has no .rail-inner, so the shared stylesheet's centring rule "
            f"(`margin-block: auto`) applies to nothing and the contents hang at the top"
        )
        assert abs(m["above"] - m["below"]) <= 30, (
            f"{slug}'s rail contents are not centred: {m['above']:.0f}px above, "
            f"{m['below']:.0f}px below"
        )
    finally:
        page.close()
