"""A page does not link a stylesheet that styles nothing on it.

`_shared/explainer.css` is 560 lines, vendored byte-identically into eight `web/` directories and
linked by eight pages. **Measured against the built site, exercise 07 matched zero of its 143
selectors** — the file styles the scrollytelling strip, the step panels, the derivation badges and
the stage lists, and that exercise builds none of them. Screenshots with and without the link hashed
identically at 2000, 1180 and 390px, so it was doing nothing but arriving.

The rest of the spread is worth having written down, because it is the argument for measuring before
deleting anything:

| page | selectors matched |
| --- | ---: |
| 03 | 108 |
| 06 | 30 |
| 05 | 4 |
| 04 | 2 |
| 08, 09, 10 | 1 each |

**What this asserts is the linking, not the deleting**, and the distinction is the whole design.
`AGENTS.md` records that removing shared CSS here has a history of taking away something a page
quietly depended on, and a browser at rest cannot tell a dead rule from one that needs a click: a
`:focus-visible` rule matches nothing until something is focused, and a `.stagerow.missing` appears
only when a reader removes a stage. So this never says a rule is unused. It says a **page** is not
using the stylesheet *at all*, which is a much stronger claim and the only one a resting page
supports.

`tools/measure_shared_css.py` produces the numbers, and prints the same caveat.
"""

import functools
import http.server
import os
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

#: The shared stylesheets this holds. Only `explainer.css` for now: it is the one that is vendored
#: eight times, linked by pages that do not use it, and named in `AGENTS.md` as the one whose real
#: usage was unmeasured. `page.css` and `components.css` are linked by every page and used by every
#: page, so the question does not arise for them.
WATCHED = ("_shared/explainer.css",)

#: Pages that link a watched stylesheet and match none of it at rest, each with the reason it is
#: still linked. **Empty on purpose, and it should stay that way.** An entry here is a page paying
#: for a stylesheet whose rules only appear after an interaction — legitimate, but worth being
#: written down rather than assumed, because the alternative reading is that the link is dead.
LINKED_BUT_UNMATCHED_AT_REST: dict[tuple[str, str], str] = {}


def _selectors(stylesheet: str) -> list[str]:
    """Every selector in the vendored stylesheet, comments and at-rules dropped."""
    source = next(REPO_ROOT.glob(f"src/exercises/*/web/{stylesheet}"), None)
    assert source is not None, f"no exercise vendors {stylesheet}"
    body = re.sub(r"/\*.*?\*/", "", source.read_text(encoding="utf-8"), flags=re.S)
    found: set[str] = set()
    for block in re.finditer(r"([^{}]+)\{[^{}]*\}", body):
        head = block.group(1).strip()
        if not head or head.startswith("@"):
            continue
        found.update(s.strip() for s in head.split(",") if s.strip() and not s.startswith("@"))
    return sorted(found)


def _linking_pages(stylesheet: str) -> list[str]:
    """Pages whose own `index.html` has a `<link>` to it.

    Matched as a `<link>` element rather than as a substring: exercise 07's link was replaced by a
    comment saying why the stylesheet is not linked, and that comment names the file. A bare
    substring check reported the page as still linking it — which would have made this guard demand
    that a page use a stylesheet it had deliberately dropped.
    """
    name = re.escape(Path(stylesheet).name)
    pattern = re.compile(rf"<link[^>]+href=[\"'][^\"']*{name}[\"'][^>]*>")
    return sorted(
        path.parent.parent.name
        for path in REPO_ROOT.glob("src/exercises/*/web/index.html")
        if pattern.search(path.read_text(encoding="utf-8"))
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


def _matches(site, slug: str, selectors: list[str]) -> int:
    browser, base = site
    page = browser.new_page(viewport={"width": 1440, "height": 950})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(500)
        return page.evaluate(
            """(sels) => sels.filter((s) => {
                 try {
                   return document.querySelector(s) !== null;
                 } catch (e) {
                   return false;
                 }
               }).length""",
            selectors,
        )
    finally:
        page.close()


@pytest.mark.parametrize("stylesheet", WATCHED)
def test_every_page_that_links_it_actually_uses_it(site, stylesheet: str) -> None:
    """One case per stylesheet, reporting every page at once.

    Reported together because the useful output is the spread — one page at zero against another at
    108 is what makes the zero legible as a defect rather than as a small number.
    """
    selectors = _selectors(stylesheet)
    assert len(selectors) > 20, (
        f"{stylesheet} yielded only {len(selectors)} selectors; the parser has stopped matching "
        "how it is written, and every count below would be meaninglessly small"
    )

    linking = _linking_pages(stylesheet)
    assert linking, f"no page links {stylesheet}; if that is deliberate, stop watching it here"

    unused = []
    for slug in linking:
        if (slug, stylesheet) in LINKED_BUT_UNMATCHED_AT_REST:
            continue
        if _matches(site, slug, selectors) == 0:
            unused.append(slug)

    assert not unused, (
        f"these pages link {stylesheet} and match none of its {len(selectors)} selectors: "
        f"{unused}. Either the page does not need it — check by driving the page first, because a "
        "rule behind a click, a focus or a scroll matches nothing at rest and is not dead — or it "
        "does and the entry belongs in LINKED_BUT_UNMATCHED_AT_REST with the reason. "
        "`uv run python tools/measure_shared_css.py` prints the same numbers with the spread."
    )


def test_the_ledger_of_unmatched_pages_has_not_gone_stale(site) -> None:
    """The other direction: an entry describing a page that now uses the stylesheet is a hole.

    It exempts a page from the check for a reason that has stopped being true, and nothing else
    would ever notice.
    """
    stale = []
    for (slug, stylesheet), reason in LINKED_BUT_UNMATCHED_AT_REST.items():
        if slug not in _linking_pages(stylesheet):
            stale.append(f"{slug} no longer links {stylesheet} ({reason})")
        elif _matches(site, slug, _selectors(stylesheet)) > 0:
            stale.append(f"{slug} now matches {stylesheet} at rest ({reason})")

    assert not stale, (
        "remove these entries; each exempts a page for a reason that no longer holds:\n  "
        + "\n  ".join(stale)
    )
