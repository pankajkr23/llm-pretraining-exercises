"""The A/B harness, and the half of the page that differs between its two variants.

TEMPORARY. This whole file is deleted with the harness once PK picks a variant — it exists so the
variant nobody is looking at cannot rot while the decision is open.

**Its own module on purpose.** `test_attention_render.py` already holds a module-scoped
`sync_playwright()` fixture, and two live sync contexts in one module raise "Playwright Sync API
inside the asyncio loop". Separating them is also what makes the deletion a single `git rm`.

What is NOT here: everything the two variants share — the invoice, the themes, the glyph boxes, the
sweep, the centrefold. Those are the same code under either flag and are covered once, next door.
"""

import functools
import http.server
import importlib.util
import json
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC = REPO_ROOT / "public"
SLUG = "08-modern-attention-variants"

pytestmark = pytest.mark.integration


def _required_spine() -> tuple[str, ...]:
    """The spine, read from the repo-wide guard so this list cannot drift from it."""
    path = REPO_ROOT / "tests" / "test_page_spine.py"
    spec = importlib.util.spec_from_file_location("_page_spine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPINE


def _bundle() -> dict:
    """The generated data the page renders, read from the served copy."""
    text = (PUBLIC / SLUG / "data.js").read_text(encoding="utf-8")
    return json.loads(text.split("Object.freeze(", 1)[1].rsplit(");", 1)[0])


@pytest.fixture(scope="module")
def page_b():
    """The same page with both A/B flags set to `b`.

    A second fixture rather than a parameterised `page`, deliberately. Parameterising would run all
    sixty-six browser assertions twice for a harness that is scaffolding with an end date, and most
    of them cannot differ between variants — the invoice, the themes, the glyph boxes and the sweep
    are shared code. What DOES differ is structural, and that is what the tests below cover.

    TEMPORARY — deleted with the harness once PK picks.
    """
    if not (PUBLIC / SLUG / "index.html").is_file():
        pytest.skip("run deploy/vercel/build.sh first")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            view = browser.new_page(viewport={"width": 1400, "height": 950})
            problems: list[str] = []
            view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            port = httpd.server_address[1]
            view.goto(f"http://127.0.0.1:{port}/{SLUG}/index.html?v=story:b,measure:b")
            view.wait_for_selector("section#reproduce", timeout=15_000)
            view.wait_for_timeout(1500)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


def test_the_other_variant_renders_without_a_console_error(page_b) -> None:
    """A variant nobody renders is a variant nobody can choose."""
    assert page_b.eval_on_selector("html", "e => e.dataset.story") == "b"
    assert page_b.eval_on_selector("html", "e => e.dataset.measure") == "b"
    assert not page_b.console_problems, page_b.console_problems


def test_both_variants_keep_the_spine_in_order(page_b) -> None:
    """The twelve-part spine is a repo-wide rule; a flag may move content, never a role."""
    spine = _required_spine()
    roles = page_b.eval_on_selector_all("#main > section", "els => els.map(e => e.dataset.role)")
    seen = [r for r in roles if r in spine]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(spine), f"the spine is out of order under story=b: {first}"


def test_every_chapter_names_its_own_entries(page_b) -> None:
    """The point of `story = b`: a chapter has a body.

    Three of the six were a heading and nothing else, and the thirty mechanisms they are chapters
    about were named in none of them. This asserts the property that fixes it — every chapter
    renders exactly the mechanisms `story.py` assigns it, in date order — rather than a count, so
    re-grouping the catalogue keeps it green and dropping an entry does not.
    """
    bundle = _bundle()
    by_key = {m["key"]: m for m in bundle["mechanisms"]}
    rendered = page_b.evaluate(
        """() => [...document.querySelectorAll('#results .well')].map(w =>
             [...w.querySelectorAll('.ce-row')].map(r => r.id.replace('m-', '')))"""
    )
    expected = [list(w["keys"]) for w in bundle["wells"]]
    assert len(rendered) == len(expected), "a chapter rendered no entry block at all"
    for got, want in zip(rendered, expected, strict=True):
        assert sorted(got) == sorted(want), f"chapter holds {got}, story.py says {want}"
        dates = [by_key[k]["date"] for k in got]
        assert dates == sorted(dates), f"a chapter's entries are not in date order: {got}"


def test_the_catalogue_is_still_tabulated_once_in_the_other_variant(page_b) -> None:
    """Moving the entries into the chapters must not leave a copy behind in the index.

    Both containers would otherwise claim `id="m-<key>"`, which is invalid, and a deep link would
    land on whichever came first.
    """
    dupes = page_b.evaluate(
        """() => {
          const seen = new Set(), dup = [];
          for (const e of document.querySelectorAll('[id]')) {
            if (seen.has(e.id)) dup.push(e.id);
            seen.add(e.id);
          }
          return dup;
        }"""
    )
    assert not dupes, f"duplicate ids under story=b: {dupes[:5]}"
    anchors = page_b.eval_on_selector_all('[id^="m-"]', "els => els.length")
    assert anchors == len(_bundle()["mechanisms"])


@pytest.mark.parametrize("width", [1440, 900, 390])
def test_the_other_variant_never_scrolls_sideways(page_b, width: int) -> None:
    """The variant that is not the default gets the same width guarantee as the one that is."""
    page_b.set_viewport_size({"width": width, "height": 900})
    try:
        over = page_b.evaluate(
            "() => Math.max(0, document.documentElement.scrollWidth - window.innerWidth)"
        )
        assert over <= 1, f"story=b scrolls sideways by {over}px at {width}px"
    finally:
        page_b.set_viewport_size({"width": 1400, "height": 950})
