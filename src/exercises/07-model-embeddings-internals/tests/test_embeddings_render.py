"""The page is tested in a browser, because `node --check` proves almost nothing about it.

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
SLUG = "07-model-embeddings-internals"

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
            view.wait_for_selector("section#cost", timeout=10_000)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


def test_the_page_loads_without_console_errors(page):
    """A page that throws halfway through renders its first half and looks fine."""
    assert page.console_problems == []


def test_every_chapter_rendered(page):
    """The six sections the argument needs, in order. A missing one is a silently dropped claim."""
    ids = page.eval_on_selector_all("main section", "els => els.map(e => e.id)")
    assert ids == ["doors", "reading", "lock", "breaking", "attribution", "cost"]


def test_no_cell_rendered_as_undefined_or_nan(page):
    """The failure mode of a data-driven page: the shape changed and every cell says `undefined`."""
    text = page.eval_on_selector("main", "el => el.innerText")
    for bad in ("undefined", "NaN", "[object Object]", "null"):
        assert bad not in text, f"the page renders the literal {bad!r}"


def test_the_tables_have_data_in_them(page):
    """Guards the opposite failure: the page renders, but every table is empty."""
    counts = page.eval_on_selector_all(
        "main table.grid tbody", "els => els.map(e => e.querySelectorAll('tr').length)"
    )
    assert len(counts) >= 6, f"expected at least six tables, found {len(counts)}"
    assert all(c > 0 for c in counts), f"an empty table body: {counts}"


def test_the_headline_numbers_come_from_the_measurements(page):
    """The page must render the measured figures, not placeholders.

    Pinned deliberately: if `results/measurements.json` changes, this goes red and someone has to
    look at the page rather than discovering the drift after it is published.
    """
    text = page.eval_on_selector("main", "el => el.innerText")
    for expected in ("44,888,832", "768,000,000", "6,291,457", "100.00%", "99.85%"):
        assert expected in text, f"{expected!r} is missing from the rendered page"


def test_the_limits_are_in_the_open_text(page):
    """`AGENTS.md`: a limitation a reader has to open a drawer to find is one the page is hiding."""
    assert page.locator("footer .limits").count() == 1
    assert page.locator("footer details").count() == 0
    items = page.eval_on_selector_all("footer .limits li", "els => els.map(e => e.innerText)")
    assert len(items) >= 4, f"only {len(items)} stated limits"
    assert all(len(i) > 60 for i in items), "a limit stated too briefly to be useful"


def test_the_borrowed_credit_is_on_the_page(page):
    """The n-gram term is prior work. The page has to say so where a reader will see it."""
    text = page.eval_on_selector("footer", "el => el.innerText").lower()
    assert "borrowed" in text


@pytest.mark.parametrize("width", [1280, 900, 390, 320])
def test_the_page_never_scrolls_sideways(page, width):
    """Wide tables scroll inside their own box; the body must not."""
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(250)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"the page scrolls sideways by {overflow}px at {width}px wide"


def test_the_lock_demonstration_holds_under_reroll(page):
    """The interaction's whole point: the four scores move, their alternating sum does not."""
    page.set_viewport_size({"width": 1280, "height": 900})
    seen = set()
    for _ in range(5):
        page.click("#lock .btn")
        rows = page.eval_on_selector_all(
            "#lock .lockrow:not(.sum) .v", "els => els.map(e => e.textContent)"
        )
        total = page.eval_on_selector("#lock .lockrow.sum .v", "el => el.textContent")
        seen.add(tuple(rows))
        assert abs(float(total)) < 1e-9, f"the alternating sum moved: {total}"
    assert len(seen) > 1, "re-rolling never changed the four scores; the control does nothing"
