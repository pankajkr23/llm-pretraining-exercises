"""The page, tested in a browser rather than parsed.

`node --check` proves a file has no syntax error and nothing more. A call to an undefined function,
a slider wired to nothing, a headline reading `0`, and a table that pushes the page sideways on a
phone all parse perfectly. This suite loads the built page in chromium and asserts what a reader
actually sees.

Every assertion here is tied to a class of bug that has genuinely shipped somewhere in this repo:
exercise 02's `0`-reading headline, exercise 03's clipped chart labels, and — for the slider test —
this exercise's own dedup guard that could not fail. A control that renders but does nothing is the
web equivalent.

Marked `integration` and skipping when chromium is absent, which means it protects the page on CI
and on a machine where `uv run playwright install chromium` has been run, and not otherwise.
"""

import http.server
import socketserver
import threading
from contextlib import contextmanager

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")
from datacleaning.config import Config  # noqa: E402
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CFG = Config()
WEB = CFG.web_dir

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (WEB / "data.json").exists(), reason="bundle not built"),
]

# The widths that actually break things: 1500 is the design target, 900 is where a figure column
# gets narrow enough to starve a flexible grid track, and 390 is the smallest phone worth serving.
WIDTHS = (1500, 900, 390)


@contextmanager
def _serve(root):
    """Serve the bundle over http. The page refuses to run from file:// — modules and fetch both."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):
            """Silence the request log; a failing test should be the only output."""

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}/"
        finally:
            httpd.shutdown()


@contextmanager
def _page(width=1500):
    """Open the built page and collect anything it throws."""
    with _serve(WEB) as base:
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                ctx = browser.new_context(viewport={"width": width, "height": 1000})
                page = ctx.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(base, wait_until="networkidle")
                page.wait_for_timeout(700)
                try:
                    yield page, errors
                finally:
                    browser.close()
        except PlaywrightError as exc:
            pytest.skip(f"no chromium available: {exc}")


def test_the_page_renders_every_chapter_without_throwing():
    with _page() as (page, errors):
        assert page.locator("main section").count() >= 12, "chapters are missing"
        assert not errors, f"the page threw: {errors[:3]}"
        assert page.locator("main .err").count() == 0, "a chapter reported a build failure"


def test_no_headline_number_reads_as_zero_or_dash():
    """Exercise 02 shipped a headline reading `0`, which was a wrong question rather than a caption
    problem. A big number that renders as `0`, `—` or `NaN` means its data never arrived."""
    with _page() as (page, _):
        values = page.locator(".bignum-v").all_inner_texts()
        assert values, "no headline numbers rendered at all"
        for v in values:
            text = v.strip()
            assert text not in {"0", "—", "-", "", "NaN", "undefined", "null"}, (
                f"a headline number reads as {text!r} — its data did not arrive"
            )


@pytest.mark.parametrize("width", WIDTHS)
def test_the_page_never_scrolls_sideways(width):
    """Wide tables must scroll inside their own box, never push the body."""
    with _page(width) as (page, _):
        overflow = page.evaluate(
            "() => { const d = document.documentElement; return d.scrollWidth - d.clientWidth; }"
        )
        assert overflow <= 1, f"the page scrolls sideways by {overflow}px at {width}px wide"


@pytest.mark.parametrize("width", (1500, 390))
def test_no_label_is_silently_clipped(width):
    """A truncated label is a number the reader cannot check."""
    with _page(width) as (page, _):
        clipped = page.evaluate(
            "() => [...document.querySelectorAll('.barlabel, .strat-name, .bignum-s, .kv')]"
            "  .filter((e) => e.scrollWidth > e.clientWidth + 1)"
            "  .map((e) => e.textContent.slice(0, 40))"
        )
        assert not clipped, f"labels clipped at {width}px: {clipped[:5]}"


def test_the_dedup_sliders_actually_change_the_verdict():
    """The chapter's whole claim is that the threshold is a decision, not a setting.

    A slider that renders but is wired to nothing would look identical in a screenshot and prove
    nothing. This is the browser equivalent of the dedup guard that could not fail.
    """
    with _page() as (page, _):
        readout = page.locator("#dedup .kv.strong")
        before = readout.inner_text()

        rows = page.locator("#dedup input[type=range]").nth(1)
        rows.fill("20")
        rows.dispatch_event("input")
        page.wait_for_timeout(200)

        after = readout.inner_text()
        assert before != after, f"moving the rows slider did not change the threshold ({before})"


def test_the_cleaning_toggles_actually_change_the_output():
    """Same rule for chapter 4: unchecking an operation must change what the reader sees."""
    with _page() as (page, _):
        out = page.locator("#clean-text .sample-out")
        before = out.inner_text()

        page.locator("#clean-text input[data-op=whitespace]").uncheck()
        page.wait_for_timeout(200)

        assert out.inner_text() != before, "unchecking whitespace collapse changed nothing"


def test_the_pii_dial_can_produce_a_false_positive():
    """Turning the dial up must mask a place name — the reader causes the failure themselves."""
    with _page() as (page, _):
        dial = page.locator("#pii input[type=range]")
        dial.fill("0.9")
        dial.dispatch_event("input")
        page.wait_for_timeout(200)

        found = page.locator("#pii .pii-found").inner_text()
        assert "Mysuru" in found, f"a high dial should over-reach onto place names, got: {found}"


def test_the_strategy_toggle_moves_a_row_between_lists():
    """Chapter 1's claim is that the count stays 8 while the membership changes."""
    with _page() as (page, _):
        count = page.locator("#strategies .strategy-count")
        first = count.inner_text()

        page.locator("#strategies .tog").nth(1).click()
        page.wait_for_timeout(600)

        assert "8 strategies" in first, f"the first list should hold 8, got: {first}"
        assert "8 strategies" in count.inner_text(), "the second list should also hold 8"

        active = page.locator("#strategies .strat.in .strat-name").all_inner_texts()
        assert "Format discipline" in active, (
            "the commitments list should include format discipline"
        )
        assert "Extract" not in active, "the commitments list should drop Extract"


def test_glossary_terms_carry_a_definition():
    """A tooltip with no text is a term the reader cannot look up."""
    with _page() as (page, _):
        empty = page.evaluate(
            "() => [...document.querySelectorAll('.term')]"
            "  .filter((e) => !e.querySelector('.tip')?.textContent?.trim())"
            "  .map((e) => e.dataset.term)"
        )
        assert not empty, f"terms with no definition: {sorted(set(empty))}"


def test_the_page_states_what_it_does_not_cover():
    """A page that hides its limits reads as coverage."""
    with _page() as (page, _):
        appendix = page.locator("#appendix").inner_text()
        assert "illustrative" in appendix.lower()
        assert "UNCHECKED" in appendix or "unchecked" in appendix.lower()
