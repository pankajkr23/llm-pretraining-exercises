"""What the widget does when a browser actually runs it.

`node --check` proves `encoder.js` parses. It cannot prove the page imports it, that the import
resolves at the URL Vercel serves it from, or that a click handler calls a function that exists —
all of which are valid syntax and all of which render a blank panel. The paste-box encoder is the
part a grader is told to use, so it is the part that has to be loaded and driven, not inspected.

Marked integration: needs a browser, and is slower than the rest of the suite by an order of
magnitude. Skipped rather than failed when Playwright or its browser is absent, so a checkout
without ``uv run playwright install chromium`` still runs everything else — which also means this
protects you silently or not at all.

Run: ``uv run pytest -m integration``
"""

import http.server
import json
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"

pytest.importorskip("playwright", reason="playwright is not installed")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (WEB / "data.json").exists(), reason="web/data.json not built"),
]

WIDTHS = (1500, 900, 390)


@contextmanager
def _serve(root: Path) -> Iterator[str]:
    """Serve a directory over http — ES module imports do not resolve from file://."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):  # noqa: D102 - silence per-request logging
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}/"
        finally:
            httpd.shutdown()


@contextmanager
def _page(width: int = 1500):
    """One loaded page, with every console error collected."""
    with _serve(WEB) as base, sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except PlaywrightError as exc:  # pragma: no cover - environment, not logic
            pytest.skip(f"no chromium available: {exc}")
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base, wait_until="networkidle")
        page.wait_for_timeout(500)
        try:
            yield page, errors
        finally:
            browser.close()


def _bundle() -> dict:
    return json.loads((WEB / "data.json").read_text(encoding="utf-8"))


def test_the_page_renders_the_score_and_every_language():
    data = _bundle()
    first = data["profiles"][0]["name"]
    expected = sum(1 for c in data["configs"] if c["profile"] == first)
    assert expected, "no configs in the opening section, so this test proves nothing"

    with _page() as (page, errors):
        assert not errors, f"the page threw: {errors[:3]}"
        assert page.locator("#err").inner_text().strip() == "", "the page showed its load error"
        assert page.locator("#selector button").count() == expected
        # Four languages, and a score that is a number rather than NaN/undefined.
        assert page.locator("tbody tr").count() == 4
        score = page.locator(".score").inner_text().replace(",", "")
        assert float(score) > 0, f"score panel reads {score!r}"


def test_the_two_measurements_are_never_shown_as_one_ranked_list():
    """v1 and v2 are denominated differently; one list across them would be meaningless."""
    data = _bundle()
    profiles = [p["name"] for p in data["profiles"]]
    assert len(profiles) > 1, "only one profile exported, so this test proves nothing"

    with _page() as (page, errors):
        assert not errors, f"the page threw: {errors[:3]}"
        assert page.locator("#profiles button").count() == len(profiles)
        for name in profiles:
            page.locator(f'#profiles button[data-p="{name}"]').click()
            page.wait_for_timeout(300)
            assert not errors, f"switching to {name} threw: {errors[:3]}"
            shown = page.locator("#selector button").count()
            expected = sum(1 for c in data["configs"] if c["profile"] == name)
            assert shown == expected, f"{name} shows {shown} tabs, expected {expected}"
            # Every section says, in words, that its numbers do not travel.
            assert "cannot be compared" in page.locator("#profilenote").inner_text()


def test_each_section_labels_the_denominator_it_is_scored_in():
    """A column header reading `units` over word counts is how the two get conflated."""
    data = _bundle()
    with _page() as (page, errors):
        for p in data["profiles"]:
            page.locator(f'#profiles button[data-p="{p["name"]}"]').click()
            page.wait_for_timeout(300)
            assert not errors, f"the page threw on {p['name']}: {errors[:3]}"
            headers = page.locator("thead th").all_inner_texts()
            assert p["denominator"] in [h.strip().lower() for h in headers], (
                f"{p['name']} table headers {headers} do not name its denominator"
            )


def test_the_paste_box_actually_tokenizes():
    """The deliverable the grader is told to use. A blank panel here is the whole failure."""
    with _page() as (page, errors):
        page.fill("#paste", "India is a country in South Asia.")
        page.wait_for_timeout(300)
        assert not errors, f"encoding threw: {errors[:3]}"
        chips = page.locator("#chips .chip").count()
        assert chips > 3, f"only {chips} tokens rendered for a full sentence"
        assert "tokens" in page.locator("#pastecount").inner_text()


def test_unknown_characters_are_shown_not_silently_dropped():
    """Dropping them is how a tokenizer flatters its own count; the chip is the whole point."""
    with _page() as (page, errors):
        page.fill("#paste", "hello 🚀 world")
        page.wait_for_timeout(300)
        assert not errors, f"encoding threw: {errors[:3]}"
        assert page.locator("#chips .chip.unk").count() > 0, "the rocket vanished instead of [UNK]"
        assert "unknown" in page.locator("#pastecount").inner_text()


def test_every_tokenizer_tab_renders():
    """A tab whose encoder is unsupported must say so, not throw or render an empty panel."""
    with _page() as (page, errors):
        for i in range(page.locator("#selector button").count()):
            page.locator("#selector button").nth(i).click()
            page.wait_for_timeout(250)
            assert not errors, f"tab {i} threw: {errors[:3]}"
            assert page.locator("tbody tr").count() == 4
            assert page.locator("#pastecount").inner_text().strip() != ""


@pytest.mark.parametrize("width", WIDTHS)
def test_the_page_never_scrolls_sideways(width: int):
    """Wide content scrolls inside its own container; the body never does."""
    with _page(width) as (page, errors):
        assert not errors, f"the page threw at {width}px: {errors[:3]}"
        overflow = page.evaluate(
            "() => { const d = document.documentElement;"
            " return d.scrollWidth > d.clientWidth ? [d.scrollWidth, d.clientWidth] : null; }"
        )
        assert not overflow, f"{width}px scrolls sideways: {overflow[0]}px into {overflow[1]}px"
