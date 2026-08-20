"""What the page does when a browser actually runs it.

Everything else in this suite checks the bundle. Nothing checked the page, and the two most
embarrassing bugs this project has shipped both lived in that gap:

* the containment subtraction (X28) was correct in `data.json` and silently never fired in the
  browser, so every stage overstated its reachable supply by 6.3T while the guard watching the
  producer passed;
* the post-training headline read "0 of 55", which was arithmetically true of a question nobody
  meant to ask and read to any human as "we have nothing".

`node --check` catches neither, because both files parsed perfectly. These tests load the built
site in a real browser and assert what a reader would see.

Marked integration: they need a browser, and they are slower than the rest of the suite by an
order of magnitude. Skipped rather than failed when Playwright or its browser is absent, so a
checkout without `playwright install chromium` still runs everything else.

Run: ``uv run pytest -m integration``
"""

import http.server
import json
import os
import re
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from dataframework.config import Config

CFG = Config()
WEB = CFG.web_dir

pytest.importorskip("playwright", reason="playwright is not installed")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (WEB / "data.json").exists(), reason="bundle not built"),
]

# The widths that have actually broken. 1500 is the design target, 900 is where the figure column
# gets narrow enough to starve a flexible grid track, and 390 is the smallest phone worth serving.
# 320 is the narrowest phone still in real use, and the width where exercise 05 actually broke:
# an `auto-fit` grid track with a fixed minimum cannot shrink below itself, so it sat 310px wide in
# a 272px container and pushed the whole document sideways. Every suite here stopped at 390, so the
# guard existed and could not fail. It runs at 320 now.
WIDTHS = (1500, 900, 390, 320)


@contextmanager
def _serve(root: Path) -> Iterator[str]:
    """Serve a directory over http, because the page refuses to run from file://."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):  # noqa: D102 - silence the per-request logging
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
            # Skipping keeps a fresh checkout working. On CI it would turn "the browser
            # never launched" into a green run with no rendering coverage at all, which is
            # what this suite exists to prevent. CI has no excuse for a missing browser.
            if os.environ.get("CI"):
                pytest.fail(f"chromium did not launch on CI: {exc}")
            pytest.skip(f"no chromium available: {exc}")
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base, wait_until="networkidle")
        page.wait_for_timeout(700)
        try:
            yield page, errors
        finally:
            browser.close()


def test_the_page_renders_every_chapter_without_throwing():
    """A JS error takes out every chapter after it, and the build cannot see that happen.

    This is not hypothetical: a stray `ref()` call once left the page rendering zero sections while
    `node --check` passed, because a call to an undefined function is valid syntax.
    """
    expected = len(
        json.loads((WEB / "data.json").read_text(encoding="utf-8"))["milestones"]["presets"]
    )
    assert expected, "no presets in the bundle, so this test proves nothing"

    with _page() as (page, errors):
        sections = page.locator("main section").count()
        assert not errors, f"the page threw: {errors[:3]}"
        # Twelve chapters plus an appendix, per the exercise's own naming rule.
        assert sections >= 13, f"only {sections} sections rendered"
        assert page.locator("main .err").count() == 0, "the page rendered its load-failure message"


def test_no_headline_fact_reads_as_a_zero():
    """A headline reading `0 of 55` says "we have nothing", whatever its caption says.

    That number was arithmetically true of a question nobody meant to ask — how many post-training
    corpora state a size in *tokens*, when they are counted in tasks and trajectories. A figure
    this prominent reading zero is nearly always the wrong question rather than a real finding, so
    it has to be defended deliberately rather than shipped by accident.
    """
    with _page() as (page, errors):
        assert not errors, f"the page threw: {errors[:3]}"
        facts = page.locator(".lede-facts .f").all_inner_texts()
        assert facts, "no headline facts found, so this test proves nothing"
        zeros = [f.replace("\n", " — ") for f in facts if re.match(r"^\s*0\s+of\s+\d", f)]
        assert not zeros, f"a headline fact reads as nothing: {zeros}"


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


@pytest.mark.parametrize("width", (1500, 390))
def test_no_chart_label_is_silently_cut_off(width: int):
    """An ellipsis is honest only when the full text is reachable another way.

    A catalogue row opens its dataset's card, so it may truncate. A chart label has no such escape:
    fourteen of them were losing up to 82px of text, and one grid track was collapsing to 4px,
    which stacks letters vertically and is worse than truncating.
    """
    with _page(width) as (page, errors):
        assert not errors, f"the page threw at {width}px: {errors[:3]}"
        clipped = page.evaluate(
            """() => [...document.querySelectorAll(
                 '.tiername, .stageplain, .stagetools, .stagereg-n, .modkey-i, .covstat')]
               .filter((e) => e.scrollWidth > e.clientWidth + 1)
               .map((e) => `${e.className}: ${e.textContent.trim().slice(0, 40)}`)"""
        )
        assert not clipped, f"{len(clipped)} label(s) losing text at {width}px: {clipped[:4]}"
