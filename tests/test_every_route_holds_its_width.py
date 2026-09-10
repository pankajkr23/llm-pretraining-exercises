"""No deployed route pushes the page sideways, and no control overhangs the box that lays it out.

**Two defects, one shape: something is wider than what holds it, and the page's own padding hides
it from any check that measures the viewport.**

`test_the_page_never_scrolls_sideways` exists in five exercises' render tests and has never been
repo-wide. It also globs only `*/web/index.html` where it sweeps at all, so three routes have never
been loaded by anything: exercise 03 ships `reasoning/` and `report/`, exercise 08 ships
`field-guide/`, and `deploy/vercel/build.sh` does `cp -R`, so a sub-route deploys with no build
change and nothing notices. All three of 03's routes, plus 06, scrolled sideways by **12px at
exactly 1180px** — the width where `page.css` starts reserving a 260px rail gutter, so the content
box is *narrower* at 1180 than at 1179. `explainer.css` set two grid floors, 48ch of prose and 400px
of figure with a 48px gap, which is a **931.75px** minimum inside an **896px** box.

The second half is narrower and nastier. A `<input type="range">` at `width: 100%` and
`box-sizing: border-box` still carries the browser's own `margin: 2px`, so it occupies 100% of its
parent **plus 4px** and sits 2px past the container it is nominally filling. On exercises 04 and 05
that was live at every width. Nothing caught it because the *page* did not scroll: the overhang
landed inside the page's own padding, so the viewport-level check read zero. A control is measured
against **its parent**, not against the window.

Both halves sweep every route the filesystem knows about, at the widths where the layout changes.
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

#: 2560 and 1440 are displays; **1180 is the rail-gutter boundary and is not negotiable** — it is
#: where the content box gets *narrower* as the window gets wider, and it has now produced three
#: separate layout defects in this repository. 900 is the scrollytelling breakpoint, 640 the table
#: one, 390 a current handset, 320 the narrowest still supported.
WIDTHS = (2560, 1440, 1180, 900, 768, 640, 390, 320)

#: Sub-pixel rounding. The same allowance `tests/_page_invariants.py` uses.
SLACK = 1

#: Controls whose overhang is deliberate, with the reason. **Empty.** An entry here is a control
#: sticking out of its container on a live page, so it is not the way to clear a red gate.
OVERHANG_ALLOWED: dict[tuple[str, str], str] = {}

OVERHANG_JS = """() => {
  const out = [];
  for (const el of document.querySelectorAll('input, select, textarea, button')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    // Positioned things are placed against an ancestor on purpose; this is about flow layout.
    if (cs.position === 'fixed' || cs.position === 'absolute') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const parent = el.parentElement;
    if (!parent) continue;
    const pr = parent.getBoundingClientRect();
    const pcs = getComputedStyle(parent);
    // A parent that scrolls is holding wide content on purpose.
    if (['auto', 'scroll'].includes(pcs.overflowX)) continue;
    const padR = parseFloat(pcs.paddingRight) || 0;
    const padL = parseFloat(pcs.paddingLeft) || 0;
    const past = Math.max(
      Math.round(r.right - (pr.right - padR)),
      Math.round((pr.left + padL) - r.left)
    );
    if (past <= 1) continue;
    out.push({
      sel: `${el.tagName.toLowerCase()}${el.type ? '[type=' + el.type + ']' : ''}`,
      cls: (el.className || '').toString().split(' ')[0] || '(none)',
      past,
      width: Math.round(r.width),
      margin: cs.marginLeft + '/' + cs.marginRight,
      box: cs.boxSizing,
      parent:
        parent.tagName.toLowerCase() +
        '.' +
        (parent.className || '').toString().split(' ')[0],
      parentWidth: Math.round(parent.clientWidth),
    });
  }
  return out;
}"""


def _routes() -> list[str]:
    """Every deployed route, sub-routes included — see the module docstring for why that matters."""
    return sorted(
        str(page.parent.relative_to(REPO_ROOT / "src" / "exercises")).replace("/web", "", 1)
        for page in REPO_ROOT.glob("src/exercises/*/web/**/index.html")
    )


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


def _at(site, slug: str, width: int, js: str):
    browser, base = site
    page = browser.new_page(viewport={"width": width, "height": 950})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(450)
        return page.evaluate(js)
    finally:
        page.close()


@pytest.mark.parametrize("slug", _routes())
def test_the_route_never_scrolls_sideways(site, slug: str) -> None:
    """The whole document, at every width where this layout changes shape."""
    failures = []
    for width in WIDTHS:
        overflow = _at(
            site,
            slug,
            width,
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth",
        )
        if overflow > SLACK:
            failures.append(f"{width}px — {overflow}px of horizontal scroll")

    assert not failures, (
        f"{slug} scrolls sideways:\n  "
        + "\n  ".join(failures)
        + "\nWide content scrolls inside its own `overflow-x: auto` container; the page body never "
        "does. Check 1180px first — that is where the rail gutter starts and the content box gets "
        "narrower as the window gets wider."
    )


@pytest.mark.parametrize("slug", _routes())
def test_no_control_overhangs_the_box_that_lays_it_out(site, slug: str) -> None:
    """A control is measured against its parent's content box, not against the window.

    The parent's padding is subtracted, because a control that sits inside its container's padding
    is inside its container. A parent that scrolls is skipped: it is holding wide content on
    purpose, which is what the repository asks of tables and diagrams.
    """
    failures = []
    for width in WIDTHS:
        for bad in _at(site, slug, width, OVERHANG_JS):
            if (slug, bad["cls"]) in OVERHANG_ALLOWED:
                continue
            failures.append(
                f"{width}px — {bad['sel']}.{bad['cls']} is {bad['past']}px outside "
                f"{bad['parent']} ({bad['width']}px in a {bad['parentWidth']}px box; "
                f"margin {bad['margin']}, {bad['box']})"
            )

    assert not failures, (
        f"{slug} lays a control outside its own container:\n  "
        + "\n  ".join(failures[:8])
        + "\nA browser's own stylesheet gives `input[type=range]` `margin: 2px`, so `width: 100%` "
        "at `border-box` occupies the parent's full width PLUS 4px. `margin-inline: 0` is the fix. "
        "The page will not scroll from this — the overhang lands in the page's own padding — which "
        "is exactly why it has to be measured against the parent."
    )


def test_there_are_controls_and_routes_to_measure(site) -> None:
    """Otherwise both sweeps above pass by having nothing to look at."""
    routes = _routes()
    assert len(routes) >= 10, f"only {len(routes)} deployed routes found; the glob has rotted"

    count_js = "() => document.querySelectorAll('input, select, button').length"
    controls = sum(_at(site, slug, WIDTHS[0], count_js) for slug in routes)
    assert controls >= 30, (
        f"only {controls} form controls across every route. Either the pages have stopped using "
        "them or the selector has rotted, and the overhang sweep is measuring nothing."
    )


def test_the_overhang_ledger_has_not_gone_stale(site) -> None:
    """An entry excusing a control that now fits is a control no longer checked."""
    stale = []
    for (slug, cls), reason in OVERHANG_ALLOWED.items():
        still = any(
            bad["cls"] == cls for width in WIDTHS for bad in _at(site, slug, width, OVERHANG_JS)
        )
        if not still:
            stale.append(f"{slug}:{cls} fits now ({reason})")

    assert not stale, (
        "remove these entries; each excuses a control that no longer needs it:\n  "
        + "\n  ".join(stale)
    )
