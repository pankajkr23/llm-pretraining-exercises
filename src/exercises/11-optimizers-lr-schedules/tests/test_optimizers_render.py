"""Exercise 11's page, tested in a browser, because `node --check` proves almost nothing about it.

Ported from exercise 10's `test_trainloop_render.py`. A call to an undefined function, a table that
renders every cell as `undefined`, a figure reading `NaN`, and a layout that scrolls sideways on a
phone all parse perfectly — and this page has four figures a reader drives, each of which can render
a control that changes nothing.

**Served, not opened as a `file://`.** ES modules refuse to load over `file://`, and the shell links
`/_shared/tokens.css` from the site root.
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
SLUG = "11-optimizers-lr-schedules"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def site():
    """Serve the assembled site and launch one browser for the whole module."""
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
            yield browser, f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html"
            browser.close()
    finally:
        httpd.shutdown()


def _open(browser, url, **context):
    view = browser.new_context(viewport={"width": 1280, "height": 900}, **context).new_page()
    problems: list[str] = []
    view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    view.goto(url)
    view.wait_for_selector("section#reproduce", timeout=10_000)
    view.console_problems = problems
    return view


@pytest.fixture(scope="module")
def page(site):
    """The page as most readers meet it: motion allowed, nothing touched."""
    browser, url = site
    view = _open(browser, url)
    yield view
    view.context.close()


@pytest.fixture(scope="module")
def reduced(site):
    """The page under `prefers-reduced-motion: reduce`, where every end state must be painted."""
    browser, url = site
    view = _open(browser, url, reduced_motion="reduce")
    yield view
    view.context.close()


@pytest.fixture
def fresh(site):
    """A page nobody else has clicked, for tests that drive a control."""
    browser, url = site
    view = _open(browser, url)
    yield view
    view.context.close()


def test_the_page_loads_without_console_errors(page, reduced):
    """A page that throws halfway through renders its first half and looks fine."""
    assert page.console_problems == []
    assert reduced.console_problems == []


#: The spine the page must have, in order. Roles rather than ids, so wording can change freely.
REQUIRED_ROLES = (
    "thesis",
    "glossary",
    "problem",
    "mechanism",
    "method",
    "expected",
    "results",
    "negatives",
    "conclusion",
    "limits",
    "next",
    "reproduce",
)


def test_the_page_has_the_required_spine_in_order(page):
    """Checked by `data-role` rather than by heading text, so the prose stays free to change."""
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"
    seen = [r for r in roles if r in REQUIRED_ROLES]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(REQUIRED_ROLES), f"the spine is out of order: {first}"


def test_every_figure_has_a_caption_that_says_something(page):
    """Captions here state what to conclude, so a short one has not done its job."""
    caps = page.eval_on_selector_all("figure figcaption", "els => els.map(e => e.innerText)")
    figs = page.evaluate("() => document.querySelectorAll('figure').length")
    assert len(caps) == figs > 0, f"{figs} figures but {len(caps)} captions"
    short = [c[:40] for c in caps if len(c) < 120]
    assert not short, f"these captions are too short to be doing any work: {short}"


def test_every_svg_is_an_image_with_a_name(page):
    """Every drawing is announced as an image and says what it shows."""
    unnamed = page.evaluate(
        """() => [...document.querySelectorAll('main svg')]
             .filter(s => s.getAttribute('role') !== 'img' || !s.getAttribute('aria-label'))
             .map(s => s.getAttribute('class'))"""
    )
    assert not unnamed, f"these drawings have no role or name: {unnamed}"


def test_no_element_is_truncated_at_any_width(page):
    """Visible is not legible: nothing may have a `scrollWidth` larger than its `clientWidth`."""
    for width in (1440, 1180, 768, 390, 320):
        page.set_viewport_size({"width": width, "height": 900})
        clipped = page.evaluate(
            """() => [...document.querySelectorAll('main *')]
                 .filter(e => e instanceof HTMLElement)
                 .filter(e => e.scrollWidth > e.clientWidth + 1 &&
                              getComputedStyle(e).overflowX !== 'auto' &&
                              getComputedStyle(e).overflowX !== 'scroll')
                 .map(e => `${e.tagName.toLowerCase()}.${e.className}: ` +
                           `${e.scrollWidth} > ${e.clientWidth}`)"""
        )
        assert not clipped, f"at {width}px these are cut off rather than merely narrow: {clipped}"
    page.set_viewport_size({"width": 1280, "height": 900})


def test_no_svg_label_spills_outside_its_own_figure(page):
    """A label is clipped when its rendered box escapes the `viewBox`."""
    spills = page.evaluate(
        """() => {
             const out = [];
             for (const svg of document.querySelectorAll('main svg')) {
               const box = svg.viewBox.baseVal;
               if (!box || !box.width) continue;
               for (const t of svg.querySelectorAll('text')) {
                 if (!t.getClientRects().length) continue;
                 const b = t.getBBox();
                 if (b.x < box.x - 1 || b.x + b.width > box.x + box.width + 1 ||
                     b.y < box.y - 1 || b.y + b.height > box.y + box.height + 1) {
                   const xs = `${b.x.toFixed(0)}..${(b.x + b.width).toFixed(0)}`;
                   const ys = `${b.y.toFixed(0)}..${(b.y + b.height).toFixed(0)}`;
                   out.push(`${t.textContent}: x ${xs}, y ${ys} outside ` +
                            `${box.width}×${box.height}`);
                 }
               }
             }
             return out;
           }"""
    )
    assert not spills, f"these labels render outside their figure: {spills}"


def test_the_page_never_scrolls_sideways(page):
    """A body that scrolls horizontally is the phone-sized version of the same defect."""
    for width in (1440, 1180, 768, 390, 320):
        page.set_viewport_size({"width": width, "height": 900})
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 1, f"the page scrolls {overflow}px sideways at {width}px"
    page.set_viewport_size({"width": 1280, "height": 900})


def test_the_rail_marks_where_the_reader_is(page):
    """`.rail-link.on` must land on exactly one link once the reader has moved."""
    page.evaluate("() => document.getElementById('results').scrollIntoView()")
    page.wait_for_timeout(400)
    marked = page.eval_on_selector_all(
        ".rail-link.on", "els => els.map(e => e.getAttribute('href'))"
    )
    assert len(marked) == 1, f"expected exactly one marked rail entry, got {marked}"


def test_the_gutter_the_shared_stylesheet_reserves_is_actually_filled(page):
    """`_shared/page.css` reserves 260px for a rail on every wide screen: reserved AND filled."""
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(200)
    measured = page.evaluate(
        """() => {
             const wrap = document.querySelector('.wrap');
             const rail = document.querySelector('#rail');
             const inner = document.querySelector('.rail-inner');
             return {
               reserved: parseFloat(getComputedStyle(wrap).paddingLeft),
               rail: rail ? rail.getBoundingClientRect().width : 0,
               inner: inner ? inner.getBoundingClientRect().height : 0,
             };
           }"""
    )
    page.set_viewport_size({"width": 1280, "height": 900})
    if measured["reserved"] < 100:
        pytest.skip("this width reserves no rail gutter, so there is nothing to fill")
    assert measured["rail"] > 100, f"{measured['reserved']:.0f}px reserved, rail is empty"
    assert measured["inner"] > 0, ".rail-inner is missing, so the rail hangs at the top"


def test_every_number_on_the_page_came_from_the_run(page, reduced):
    """`undefined`/`NaN` is what a figure from nowhere looks like — checked on both paints."""
    for view in (page, reduced):
        text = view.inner_text("main")
        for poison in ("undefined", "NaN", "[object Object]", "Infinity"):
            assert poison not in text, f"the page rendered {poison!r} — a figure came from nowhere"


#: Markup that means a formatting step was skipped: a source token printed instead of applied.
MARKUP_LEAKS = ("[[", "**", "`", "<b>", "</b>", "&amp;", "&lt;", "&gt;")

#: Every region the page writes text into, read as `textContent`, not `innerText`: the rail's
#: sub-labels are hidden at some widths and shown at others, and `innerText` cannot see hidden text.
#: Adopted from exercise 13's page, which found the gap by planting a leak in the rail.
PAGE_TEXT_JS = """() => ['.wrap > .eyebrow', 'h1', '.lede', 'main', '#rail', '#foot']
  .flatMap((s) => [...document.querySelectorAll(s)])
  .map((e) => e.textContent)
  .join(' ')"""


def test_no_markup_leaks_into_the_text(reduced):
    """Markup meant for a renderer must never reach a reader as literal characters.

    `[[`, `**` and backticks are the markdown that a template literal passes through untouched; a
    literal `<b>` is a tag set with `textContent` instead of `innerHTML`; `&amp;`, `&lt;` and
    `&gt;` are entities escaped twice. Checked on the reduced-motion paint, where every figure's end
    state — and so every state-dependent sentence — is on the page.
    """
    text = reduced.evaluate(PAGE_TEXT_JS)
    assert len(text) > 20_000, "the page text came back nearly empty; the selectors have gone stale"
    leaked = [m for m in MARKUP_LEAKS if m in text]
    assert not leaked, f"the rendered page shows raw markup: {leaked}"


def test_the_markup_guard_can_fail(fresh):
    """Its twin: a planted escaped `<b>` and a literal `**` are caught, even hidden in the rail."""
    planted = fresh.evaluate(
        """(js) => {
             const p = document.createElement('span');
             p.textContent = 'a <b>planted</b> **leak**';
             p.style.display = 'none';
             document.querySelector('#rail').append(p);
             const text = (0, eval)(js)();
             p.remove();
             return text;
           }""",
        PAGE_TEXT_JS,
    )
    assert [m for m in MARKUP_LEAKS if m in planted] == ["**", "<b>", "</b>"]


def test_every_glossary_entry_carries_a_number_from_the_run(page):
    """The glossary heading promises a figure from these runs in every entry; it is checked."""
    import re as _re

    entries = page.evaluate("""() => {
      const out = [];
      const dl = document.querySelector('.gloss');
      if (!dl) return out;
      const kids = [...dl.children];
      for (let i = 0; i < kids.length; i += 1) {
        if (kids[i].tagName !== 'DT') continue;
        const dd = kids[i + 1];
        if (dd && dd.tagName === 'DD') {
          out.push({term: kids[i].textContent.trim(), body: dd.textContent.trim()});
        }
      }
      return out;
    }""")
    assert entries, "no glossary entries found; the selector has gone stale"
    numberless = [e["term"] for e in entries if not _re.search(r"\d", e["body"])]
    assert not numberless, f"these glossary entries carry no figure from the run: {numberless}"


def test_the_opening_tiles_carry_a_failure(page):
    """A page that shows only its wins has not earned them: one tile is a failure."""
    marks = page.eval_on_selector_all("#thesis .tile", "els => els.map(e => e.className)")
    assert 3 <= len(marks) <= 4, f"the opening should have three or four tiles, has {marks}"
    assert any("bad" in m for m in marks), f"no failure among the opening tiles: {marks}"


# ------------------------------------------------------------------------- the interactive figures
#
# One assertion per figure a reader drives: its control changes what is drawn, and under reduced
# motion its end state is painted without anyone touching it.

#: The `visibility` of every element a selector matches, as a JS function of the selector.
VISIBILITY = "sel => [...document.querySelectorAll(sel)].map(g => g.getAttribute('visibility'))"


def test_the_adam_ledger_fills_one_step_at_a_time(fresh, reduced):
    """Accumulator: each step reveals one more column of Adam's arithmetic."""
    shown = "() => document.querySelectorAll('.adamfig text.adam-v:not(.later)').length"
    first = fresh.evaluate(shown)
    fresh.click(".adam .tabs button:last-child")
    last = fresh.evaluate(shown)
    assert 0 < first < last, (
        f"stepping to the end showed {last} values, against {first} at the start"
    )
    assert reduced.evaluate(shown) == last, "reduced motion does not paint the final step"
    pressed = reduced.eval_on_selector_all(
        ".adam .tabs button", "els => els.map(e => e.getAttribute('aria-pressed'))"
    )
    assert pressed[-1] == "true", f"reduced motion leaves the stepper at {pressed}"


def test_the_bias_simulator_reveals_only_after_a_guess(fresh, reduced):
    """Simulator with a prediction: thresholds are hidden until the reader locks a guess in."""
    hidden = f"() => ({VISIBILITY})('.simfig g.thr')"
    assert set(fresh.evaluate(hidden)) == {"hidden"}, "the answer is drawn before the guess"
    fresh.click(".sim button.btn")
    fresh.wait_for_timeout(700)
    assert set(fresh.evaluate(hidden)) == {"visible"}, "locking a guess did not reveal the answer"
    assert fresh.locator(".sim .cascade .cas-row").count() > 0, "the cascade never appeared"
    path_before = reduced.evaluate(
        "() => document.querySelector('.simfig path.series').getAttribute('d').length"
    )
    assert set(reduced.evaluate(hidden)) == {"visible"}, "reduced motion does not paint the reveal"
    assert path_before > 1000, "reduced motion paints only the first steps of the curve"


def test_the_warmup_diff_switches_what_is_drawn(fresh, reduced):
    """Diff: the tabs change which run every panel shows, and the end state shows both."""
    opacity = """() => {
      const one = (sel) => getComputedStyle(document.querySelector(sel)).opacity;
      return [one('.sm .ser-warm'), one('.sm .ser-none')];
    }"""
    assert fresh.evaluate(opacity) == ["1", "0"], "the opening view should show the warmup run only"
    fresh.click(".sm .tabs button:nth-child(2)")
    fresh.wait_for_timeout(800)
    assert fresh.evaluate(opacity) == ["0", "1"], "switching the tab did not change the panels"
    assert reduced.evaluate("() => document.querySelector('.sm').dataset.mode") == "both"
    assert reduced.evaluate(opacity) == ["1", "1"], "reduced motion does not overlay both runs"


def test_the_width_sweep_hides_its_prediction_until_a_guess(fresh, reduced):
    """Optimizer with a prediction: the power-law prediction appears only after the guess."""
    shown = f"() => ({VISIBILITY})('.optfig g.pred')"
    assert set(fresh.evaluate(shown)) == {"hidden"}, "the prediction is drawn before the guess"
    fresh.click(".opt button.btn")
    fresh.wait_for_timeout(700)
    assert set(fresh.evaluate(shown)) == {"visible"}, (
        "locking a guess did not reveal the prediction"
    )
    fit = (
        "() => [...document.querySelectorAll('.optfig .fit-line')]"
        ".map(p => (p.getAttribute('d') || '').length)"
    )
    assert all(n > 0 for n in fresh.evaluate(fit)), "the fitted line was never drawn"
    assert set(reduced.evaluate(shown)) == {"visible"}, (
        "reduced motion does not paint the prediction"
    )
