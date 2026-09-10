"""Anything exposed as an image announces what it is, on every deployed page.

An `<svg role="img">` with no name is announced as "image" and nothing else. The reader is told a
figure is there and given no way to know what it showed — worse than omitting it, because the page
has spent their attention and returned nothing.

**Exercise 07 shipped six such figures and named two.** The other four — the three-bar loss chart,
the rectangle in byte space, the per-seed pairing, and the reusable bar chart — announced as bare
images for as long as the page has existed. Every other exercise was already clean: 01's canvases
carry `aria-labelledby`, 03 and 05 use `aria-label`, 09 and 10 name every one of theirs.

**The name is computed, not written.** `<title>`, `aria-label` and `aria-labelledby` all produce
one, in that order of precedence, and a source-level count of any single mechanism would report a
page using a different one as broken. So this asks the browser for the accessible name through
Playwright's accessibility snapshot, which is the same computation a screen reader performs.

Two things are deliberately not asserted. **Length**, because a good name for a two-mark diagram is
short and a rule about characters would push authors to pad. And **decorative graphics**: an SVG
with no `role="img"` and `aria-hidden="true"` is furniture, and demanding a name for it is how a
reader ends up hearing every rule and gradient on the page.
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

#: Ask for what a screen reader would be told, following the accessible-name algorithm —
#: `aria-labelledby`, then `aria-label`, then `<title>` — so a page is judged on the result rather
#: than on whichever mechanism it happened to use.
NAMES_JS = """() => {
  const out = [];
  const sel = 'svg[role="img"], canvas[role="img"], [role="img"]';
  document.querySelectorAll(sel).forEach((el, i) => {
    if (el.getAttribute('aria-hidden') === 'true') return;
    const labelledby = el.getAttribute('aria-labelledby');
    let named = (el.getAttribute('aria-label') || '').trim();
    if (!named && labelledby) {
      named = labelledby
        .split(/\\s+/)
        .map((id) => (document.getElementById(id) || {}).textContent || '')
        .join(' ')
        .trim();
    }
    if (!named) {
      const title = el.querySelector(':scope > title');
      named = title ? (title.textContent || '').trim() : '';
    }
    out.push({
      index: i,
      tag: el.tagName.toLowerCase(),
      cls: (el.getAttribute('class') || '').slice(0, 30),
      name: named,
    });
  });
  return out;
}"""


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


def _figures(site, slug: str) -> list[dict]:
    browser, base = site
    ctx = browser.new_context(viewport={"width": 1440, "height": 950})
    page = ctx.new_page()
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(400)
        return page.evaluate(NAMES_JS)
    finally:
        ctx.close()


@pytest.mark.parametrize("slug", _deployable())
def test_every_figure_exposed_as_an_image_has_a_name(site, slug: str) -> None:
    """One case per page. A page with no exposed figures passes, and that is honest.

    Not every exercise draws. Asserting that each page *has* figures would be a different rule
    about a different thing, and would fail the two pages predating the spine for a reason
    unrelated to whether their graphics are usable. The vacuity guard below covers what matters:
    that the repository as a whole has figures for this to be checking at all.
    """
    unnamed = [
        f"{fig['tag']}.{fig['cls'] or '(no class)'} — figure {fig['index'] + 1} on the page"
        for fig in _figures(site, slug)
        if not fig["name"]
    ]
    assert not unnamed, (
        f"{slug} exposes {len(unnamed)} figure(s) as an image with no accessible name:\n  "
        + "\n  ".join(unnamed)
        + "\nA screen reader announces these as 'image' and nothing else, so the reader is told a "
        "figure is there and given no way to know what it showed. Add an `svg('title')` child, an "
        "`aria-label`, or an `aria-labelledby` — and make it say what to conclude, the way a "
        "caption does, rather than naming the axes. If the graphic is decoration, mark it "
        "`aria-hidden` and drop the role instead."
    )


def test_the_repository_has_figures_for_this_to_check(site) -> None:
    """Otherwise every case above passes by having nothing to look at.

    This is the half that fails if the selector rots — a renamed role, a change in how figures are
    built, and the sweep goes quietly green across ten pages.
    """
    total = sum(len(_figures(site, slug)) for slug in _deployable())
    assert total >= 10, (
        f"only {total} figures are exposed as images across every deployed page. Either the "
        "selector has stopped matching how figures are built, or the pages have stopped drawing — "
        "and the first is far more likely than the second."
    )
