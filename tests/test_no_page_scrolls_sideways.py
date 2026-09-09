"""No deployed page scrolls sideways, at any width — including the one nobody tests.

Several exercises assert this about themselves. Every one of them drives 1280, 1500, 900 or 390, and
**the width that actually broke was 1180.**

That is not arbitrary. `_shared/page.css` starts reserving a 260px rail gutter at exactly 1180px, so
the content box is **narrower at 1180 than it is at 1179** — the only place in the range where a
bigger window means a smaller reading area. `_shared/explainer.css` already names it as "the
tightest squeeze" in a comment, and set a `48ch` floor to fix a different symptom there. Its wide
scrollytelling strip then demanded `48ch + 48px + 400px = 931.75px` inside an 896px box, so the
tracks ran 35.75px past `#main` and put 12px of the figure past the right edge of the window.

**Exercises 03 and 06 scrolled sideways at 1180 and at no other width, for as long as that rule has
existed, with every per-exercise sideways-scroll guard green.** A guard that samples widths cannot
find a defect that lives at one of them unless it happens to sample it, and the width worth sampling
is the one the layout itself calls out.

So this sweeps the deployable set across the breakpoint and the phone sizes below it. It is one
assertion per page over all widths rather than one per pair, because the useful report is *which
widths* a page fails at — a single width in a list of eight is a squeeze, and all eight is a layout.
"""

import functools
import http.server
import os
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC = REPO_ROOT / "public"

pytestmark = pytest.mark.integration

#: Chosen where the layout changes, not for round numbers.
#:
#: 1180 and 1179 straddle the rail gutter and are the pair this file exists for — keep them adjacent
#: so a failure at one and not the other reads as the breakpoint rather than as a width. 1081 is
#: where `explainer.css` switches its grid. 768 and 800 are a tablet in portrait, which is the gap a
#: `max-width: 760px` collapse left open. 390 and 320 are phones, and 320 is where an `auto-fit`
#: track that cannot shrink pushes the page sideways.
WIDTHS = (2560, 1920, 1440, 1280, 1200, 1180, 1179, 1100, 1081, 1080, 900, 800, 768, 390, 320)

#: Sub-pixel rounding. The defect this was written for was 12px.
TOLERANCE_PX = 1


def _deployable() -> list[str]:
    """From the filesystem, so a new exercise is covered the day it ships."""
    return sorted(p.parent.parent.name for p in REPO_ROOT.glob("src/exercises/*/web/index.html"))


@pytest.fixture(scope="module")
def site():
    if not (PUBLIC / "_shared" / "tokens.css").is_file():
        pytest.skip("run deploy/vercel/build.sh first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # pragma: no cover - environment, not logic
                if os.environ.get("CI"):
                    pytest.fail(f"chromium did not launch on CI: {exc}")
                pytest.skip(f"chromium unavailable: {exc}")
            yield browser, f"http://127.0.0.1:{httpd.server_address[1]}"
            browser.close()
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("slug", _deployable())
def test_a_page_never_scrolls_sideways(site, slug: str) -> None:
    """Sweep the widths inside one case, and name the culprits when it fails.

    The overflow number alone sends a reader hunting, so a failure also reports the first few
    elements whose right edge is past the viewport. That is what turned "06 scrolls 12px" into
    "`.sticky` ends at 1192 in a 1180 window", which is one step from the grid rule that put it
    there.
    """
    browser, base = site
    ctx = browser.new_context(viewport={"width": WIDTHS[0], "height": 900})
    page = ctx.new_page()
    failures = []
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        for width in WIDTHS:
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(150)
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            if overflow > TOLERANCE_PX:
                culprits = page.evaluate(
                    """() => [...document.querySelectorAll('body *')]
                         .filter(e => e.getBoundingClientRect().right
                                      > document.documentElement.clientWidth + 0.5)
                         .slice(0, 3)
                         .map(e => (e.className || e.tagName).toString().slice(0, 34) + ' ends at '
                                   + Math.round(e.getBoundingClientRect().right))"""
                )
                failures.append((width, overflow, culprits))
    finally:
        ctx.close()

    assert not failures, (
        f"{slug} scrolls sideways at "
        + "; ".join(
            f"{w}px by {over}px ({', '.join(who) or 'no element found'})"
            for w, over, who in failures
        )
        + ".\nA page that scrolls sideways is broken for every reader at that width, and the width "
        "worth suspecting first is 1180 — where `page.css` starts reserving the rail gutter, so "
        "the content box is narrower than it is at 1179."
    )
