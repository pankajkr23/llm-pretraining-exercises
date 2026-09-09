"""A table of sentences fits its container. A table of figures may scroll.

`table.grid` sets `white-space: nowrap` on every cell but the first — right for a column of numbers,
catastrophic for a column of prose. Marking a table `prose` is how a page says its cells wrap, and
this asserts that they actually do.

**The page-level overflow guard is blind to this and cannot be fixed to see it.** Every table sits
in a `.tablewrap` with `overflow-x: auto`, which is exactly what `AGENTS.md` asks for: wide content
scrolls in its own container rather than pushing the page sideways. A cell hidden inside that scroll
is, to any generic check, indistinguishable from a wide table behaving correctly. The element-level
check has the same hole from the other side — it skips anything whose computed `overflow-x` is
`auto`, which is precisely `.tablewrap`, and the `td` itself grows to its content so its own
`scrollWidth` equals its `clientWidth`.

**Exercise 09 hit this, fixed it, wrote a guard — and exercise 10 then copied the markup without the
fix.** Its `chapters.js` builds the audit table with the same `'grid prose'` class list while no
stylesheet it links defined `.prose` at all, so the modifier was inert and looked deliberate. The
table laid out **3,635px wide inside a 1,156px wrapper**, cells of up to 433 characters each on one
unwrapped line, on the section whose own standfirst reads *"the gap between the two columns is the
whole argument"*.

That is the third single-exercise guard this week to miss the exercise next door, so it is repo-wide
now and discovers its pages from the filesystem.
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

#: Widths where the wrapper is tightest. 1180 is where `page.css` starts reserving the rail gutter,
#: so the content box is narrower there than at 1179 — the width that has already produced two
#: separate layout defects in this repository.
WIDTHS = (1440, 1280, 1180, 900)

OFFENDERS_JS = """() => {
  const bad = [];
  for (const t of document.querySelectorAll('table.prose')) {
    const wrap = t.closest('.tablewrap') || t.parentElement;
    if (!wrap) continue;
    if (t.scrollWidth > wrap.clientWidth + 1) {
      const section = t.closest('section');
      const widest = Math.max(...[...t.querySelectorAll('td')].map((td) => td.innerText.length));
      bad.push(
        (section ? section.id : '(no section)') +
          ': table ' + t.scrollWidth + 'px in a ' + wrap.clientWidth +
          'px wrapper, longest cell ' + widest + ' characters'
      );
    }
  }
  return bad;
}"""


def _deployable() -> list[str]:
    """From the filesystem, so a new page is covered the day it ships."""
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


def _tables(site, slug: str, width: int) -> tuple[int, list[str]]:
    """`(how many prose tables the page has, which of them overflow)` at one width."""
    browser, base = site
    page = browser.new_page(viewport={"width": width, "height": 950})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(400)
        count = page.evaluate("() => document.querySelectorAll('table.prose').length")
        return count, page.evaluate(OFFENDERS_JS)
    finally:
        page.close()


@pytest.mark.parametrize("slug", _deployable())
def test_a_prose_table_fits_its_container(site, slug: str) -> None:
    """One case per page, sweeping the widths inside it.

    A page with no prose table passes, which is honest: most pages have none, and asserting that
    each *has* one would be a different rule about a different thing. The vacuity guard below covers
    what matters — that the repository has some for this to be checking.
    """
    failures = []
    for width in WIDTHS:
        _, offenders = _tables(site, slug, width)
        failures.extend(f"{width}px — {line}" for line in offenders)

    assert not failures, (
        f"{slug} has a prose table wider than its container, so part of every row sits behind a "
        "scrollbar:\n  " + "\n  ".join(failures) + "\nA table marked `prose` is one whose cells "
        "are sentences, and `table.grid` sets `white-space: nowrap` on every cell but the first. "
        "The page needs `table.grid.prose td { white-space: normal }` — check it is defined in a "
        "stylesheet this page actually links, because the class is inert without it and an inert "
        "modifier looks deliberate."
    )


def test_some_page_has_a_prose_table(site) -> None:
    """Otherwise every case above passes by having nothing to measure.

    This is the half that fails if the class is renamed or the tables stop being marked — at which
    point the sweep goes quietly green across every page.
    """
    total = sum(_tables(site, slug, WIDTHS[0])[0] for slug in _deployable())
    assert total >= 2, (
        f"only {total} prose table(s) across every deployed page. Either the pages have stopped "
        "using them, or `table.prose` has stopped being how one is marked — and the second is far "
        "more likely, at which point this file measures nothing."
    )
