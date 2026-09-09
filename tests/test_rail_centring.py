"""Where the pinned rail sits, on every page that builds one — both axes.

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

**The horizontal half was added after the same mistake was made a second time.** The rail is chrome
at the page edge and `.wrap` reserves a 260px gutter for it, which leaves the reading column centred
in what remains — equal air either side, at every width. Exercise 08 has asserted that since a
previous agent moved its rail inward to sit against the text, misreading a complaint about the
column being off centre as one about the rail being too far left, and then wrote a guard demanding
the broken version.

It happened again, on three pages at once. A fix for exercise 09 reading "the text is squeezed and
the other half is just lying empty" was mostly the type scale — but it also added
`left: max(0px, calc((100vw - 1500px) / 2))` to 07, 09 and 10, which at 2560 put **24px of air on
the left of the column and 554px on its right**. `max(0px, …)` clamps below 1440, so every width a
laptop opens looked right, and 08's guard is parametrised over widths while hard-coded to one page.

What is asserted here is the **property** and never a distance: 08's wrap is 2200px and everyone
else's is 1500px, so the correct gap is 204px on one page and 554px on another at the same
viewport. A guard naming either number fails the other page while both are right, which is exactly
the guard that shipped the first time. The rail's own offset is not asserted either — a page may
have a reason to place it differently; what no page may do is leave the column off centre.
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


#: Widths where the layout actually changes. 1440 is the one that matters most: it is where the
#: override that caused this stopped being a no-op, so a sweep that skipped it would have called
#: three broken pages clean.
CENTRING_WIDTHS = (2560, 1920, 1600, 1440, 1280, 1180)

#: Sub-pixel rounding only. The failure this exists for was 530px, so a generous tolerance would
#: still have caught it — but a tolerance wide enough to hide a real asymmetry is one that gets
#: widened again the next time it is inconvenient.
CENTRING_TOLERANCE_PX = 2


@pytest.mark.parametrize("slug", _pages_that_build_a_rail())
def test_the_reading_column_is_centred_in_the_space_the_rail_leaves(browser, slug: str) -> None:
    """Equal air either side of the reading column, at every width, beside a pinned rail.

    Sweeping the widths inside one case rather than parametrising over them is what makes the
    report usable: this failure exists at 1600 and above and at no width anyone develops at, so a
    case named only by exercise would send a reader to open a page that is correct on their screen.

    The measurement is `.wrap`'s **padding box**, not its border box, because the 260px the shared
    stylesheet reserves for the rail is padding — measuring the border box would compare the rail
    against space the rail itself occupies and report every correct page as off centre by 260px.
    """
    b, base = browser
    if not (PUBLIC / slug / "index.html").is_file():
        pytest.skip(f"{slug} is not published")

    page = b.new_page(viewport={"width": CENTRING_WIDTHS[0], "height": 900})
    off_centre, pinned_at = [], []
    try:
        page.goto(f"{base}/{slug}/", wait_until="networkidle")
        for width in CENTRING_WIDTHS:
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(200)
            m = page.evaluate(
                """() => {
                  const rail = document.querySelector('.rail');
                  const wrap = document.querySelector('.wrap');
                  if (!rail || !wrap) return null;
                  if (getComputedStyle(rail).position !== 'fixed') return {skip: true};
                  const cs = getComputedStyle(wrap), wb = wrap.getBoundingClientRect();
                  return {
                    railRight: rail.getBoundingClientRect().right,
                    left: wb.left + parseFloat(cs.paddingLeft),
                    right: wb.right - parseFloat(cs.paddingRight),
                    viewport: document.documentElement.clientWidth,
                  };
                }"""
            )
            assert m is not None, f"{slug} builds a rail but no .rail/.wrap is in the DOM"
            if m.get("skip"):
                continue
            pinned_at.append(width)
            gap_left = m["left"] - m["railRight"]
            gap_right = m["viewport"] - m["right"]
            assert gap_left > 0, f"{slug} at {width}px: the text starts before the rail ends"
            if abs(gap_left - gap_right) > CENTRING_TOLERANCE_PX:
                off_centre.append((width, round(gap_left), round(gap_right)))
    finally:
        page.close()

    assert pinned_at, (
        f"{slug} pins no rail at any of {CENTRING_WIDTHS}, so this case asserted nothing. A page "
        "that has stopped pinning its rail has left the layout standard, and a skip would report "
        "that as a pass."
    )
    assert not off_centre, (
        f"{slug}: the reading column is off centre beside the rail at "
        + ", ".join(f"{w}px ({left}px left, {right}px right)" for w, left, right in off_centre)
        + ". The rail is chrome at the page edge and the column centres in what it leaves — equal "
        "air either side, at every width. A rail moved inward to sit against the text leaves dead "
        "space on both sides of itself and pushes the column off centre, which is what a reader "
        "reports as the page having walked away from its own contents."
    )
