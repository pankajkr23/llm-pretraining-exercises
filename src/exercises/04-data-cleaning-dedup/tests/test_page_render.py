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
import os
import socketserver
import subprocess
import threading
from contextlib import contextmanager

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")
from datacleaning.config import Config  # noqa: E402
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CFG = Config()
WEB = CFG.web_dir

# The site root, as assembled by `deploy/vercel/build.sh`. Tests serve **this** when it exists,
# because it is what actually ships — and the difference is not cosmetic. Font tokens live only in
# the site-root `_shared/tokens.css`; the per-exercise copy does not define `--sans` or `--display`.
# Serving `web/` alone therefore resolves `/_shared/tokens.css` to the partial copy and renders the
# whole page in a serif fallback that no reader will ever see. A suite that green-lights a page
# rendering differently from production is testing the wrong artifact.
REPO = CFG.web_dir.parents[3]
PUBLIC = REPO / "public"
SLUG = CFG.web_dir.parent.name
BUILT = PUBLIC / SLUG


def _root_and_path() -> tuple:
    """Serve the built site, assembling it first if it is not there.

    `public/` is git-ignored, so a fresh checkout and CI both start without it. Building it here —
    the same one-second bash script the deploy runs — is what keeps these tests pointed at the
    artifact that ships rather than at a bundle that renders differently.
    """
    # `build.sh` starts with `rm -rf public/`, so two xdist workers both finding `public/` absent
    # would delete each other's site mid-test. Fail loudly instead of racing: CI assembles it once
    # up front (see `.github/workflows/ci.yml`), and locally the fix is one command.
    if os.environ.get("PYTEST_XDIST_WORKER") and not (BUILT / "index.html").exists():
        pytest.fail(
            "running under -n with no assembled site. `build.sh` would race across workers "
            "(it begins `rm -rf public/`). Run `bash deploy/vercel/build.sh` first."
        )
    if not (BUILT / "index.html").exists():
        script = REPO / "deploy" / "vercel" / "build.sh"
        if script.exists():
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["bash", str(script)], cwd=REPO, capture_output=True, check=False, timeout=180
            )
    if (BUILT / "index.html").exists():
        return PUBLIC, f"/{SLUG}/"
    pytest.skip("the site could not be assembled; the page under test would not match production")
    return WEB, "/"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (WEB / "data.json").exists(), reason="bundle not built"),
]

# The widths that actually break things: 1500 is the design target, 900 is where a figure column
# gets narrow enough to starve a flexible grid track, and 390 is the smallest phone worth serving.
# 320 is the narrowest phone still in real use, and the width where exercise 05 actually broke:
# an `auto-fit` grid track with a fixed minimum cannot shrink below itself, so it sat 310px wide in
# a 272px container and pushed the whole document sideways. Every suite here stopped at 390, so the
# guard existed and could not fail. It runs at 320 now.
WIDTHS = (1500, 900, 390, 320)


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
    root, path = _root_and_path()
    with _serve(root) as origin:
        base = origin.rstrip("/") + path
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
            # Skipping keeps a fresh checkout working. On CI it would turn "the browser
            # never launched" into a green run with no rendering coverage at all, which is
            # what this suite exists to prevent. CI has no excuse for a missing browser.
            if os.environ.get("CI"):
                pytest.fail(f"chromium did not launch on CI: {exc}")
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


def test_the_page_uses_the_shared_sans_type_stack():
    """`AGENTS.md`: one design language, system sans, no serif.

    This failed silently for a while because the test harness served `web/` rather than the built
    site, and the font tokens live only in the site-root stylesheet. The page looked serif under
    test and sans in production — the suite was checking a different artifact from the one shipping.
    """
    with _page() as (page, _):
        for selector in ("body", "h1", "h2"):
            stack = page.evaluate(
                f"() => getComputedStyle(document.querySelector('{selector}')).fontFamily"
            )
            assert "serif" not in stack.replace("sans-serif", ""), (
                f"{selector} renders in a serif stack: {stack}"
            )
            assert "system-ui" in stack or "-apple-system" in stack, (
                f"{selector} is not using the shared type tokens: {stack}"
            )


def test_every_rail_label_matches_its_chapter_heading():
    """The sidebar shipped with the first word missing from every entry.

    `buildRail` stripped a leading `\\S+\\s` off `h2.textContent` to drop the chapter number — but
    textContent runs the number and title together as `1How many...`, so it ate the first word and
    the sidebar read "many strategies are there?". Nothing in the suite noticed, because every other
    test asked whether an element *existed* rather than whether it said the right thing.
    """
    with _page() as (page, _):
        pairs = page.evaluate(
            "() => [...document.querySelectorAll('#rail .rail-link')].map((a) => {"
            "  const sec = document.querySelector(a.getAttribute('href'));"
            "  return [a.querySelector('.rail-t')?.textContent?.trim(),"
            "          sec?.dataset?.title?.trim()];"
            "})"
        )
        assert pairs, "the rail rendered no links"
        for label, title in pairs:
            assert label and title, f"a rail entry has no label or no target: {label!r}/{title!r}"
            assert label == title, f"rail says {label!r}, the chapter is titled {title!r}"


def test_the_pinned_rail_is_vertically_centred():
    """The shared stylesheet centres the rail with `.rail-inner { margin-block: auto }`.

    Markup that omits that wrapper gets the rule and none of its effect: the list hangs off the top
    edge with a screen of empty column beneath it. Asserted as a measurement rather than by looking
    for the class, so the check survives a change of technique.
    """
    with _page(1500) as (page, _):
        box = page.evaluate(
            "() => { const r = document.getElementById('rail');"
            "  const inner = r.querySelector('.rail-inner') || r;"
            "  const a = r.getBoundingClientRect(), b = inner.getBoundingClientRect();"
            "  return {above: b.top - a.top, below: a.bottom - b.bottom, rail: a.height,"
            "          content: b.height}; }"
        )
        if box["content"] >= box["rail"] - 4:
            pytest.skip("the rail fills its column; there is nothing to centre")

        slack = box["above"] + box["below"]
        assert abs(box["above"] - box["below"]) <= max(8, slack * 0.2), (
            f"the rail is not centred: {box['above']:.0f}px above, {box['below']:.0f}px below"
        )


def test_the_rail_uses_the_shared_stylesheet_classes():
    """Bare <a> elements in the rail render as raw underlined links in the gutter.

    The shared page.css styles `.rail-list` and `.rail-link`; markup that does not use them is
    unstyled rather than differently styled, which is how the first version looked.
    """
    with _page() as (page, _):
        assert page.locator("#rail nav.rail-list").count() == 1
        assert page.locator("#rail a.rail-link").count() >= 12
        assert page.locator("#rail .rail-head").count() == 1


def test_no_markdown_syntax_leaks_into_the_rendered_text():
    """A literal `*proves*` shipped because `rich()` handled `**bold**` and not `*italic*`."""
    with _page() as (page, _):
        text = page.locator("main").inner_text()
        for marker in ("**", "[[", "]]"):
            assert marker not in text, f"unrendered markup {marker!r} is visible to the reader"
        import re as _re

        leaked = _re.findall(r"(?<![\w*])\*[^*\s][^*]{0,60}\*(?![\w*])", text)
        assert not leaked, f"unrendered italics visible: {leaked[:3]}"


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
