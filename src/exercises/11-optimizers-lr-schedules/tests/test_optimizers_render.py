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


#: For every figure: does it scroll as a whole, which of its inner boxes scroll, does a visible cue
#: say so, and which SVG labels are painted outside what the reader can see without scrolling.
NARROW_FIGURES_JS = """() => [...document.querySelectorAll('main figure')].map((f, i) => {
  const visible = (el) =>
    el && el.getClientRects().length && getComputedStyle(el).display !== 'none';
  const scrollers = [...f.querySelectorAll('.fig-scroll, .opt-panel')]
    .filter((b) => b.scrollWidth > b.clientWidth + 1);
  const cue = [...f.querySelectorAll('.scroll-cue')].some(visible);
  const clipped = [];
  for (const t of f.querySelectorAll('svg text')) {
    if (!t.getClientRects().length || !t.textContent.trim()) continue;
    if (t.closest('.fig-scroll, .opt-panel')) continue;  // inside a deliberate scroller
    const r = t.getBoundingClientRect();
    const box = t.closest('svg').getBoundingClientRect();
    if (r.left < box.left - 1 || r.right > box.right + 1) clipped.push(t.textContent.trim());
  }
  return {i, figure: f.scrollWidth - f.clientWidth, scrollers: scrollers.length, cue, clipped};
})"""


def test_figures_on_a_phone_are_whole_or_say_they_scroll(page):
    """At phone widths a figure is either whole, or scrolls in a box with a visible cue.

    A drawing never shrinks below its designed width, so on a phone it scrolls — and a scroller
    with no cue reads as a figure cut off at the edge, with its last column or label simply gone.
    The figure itself must never be the thing that scrolls, and no label outside a deliberate
    scroller may be painted past its drawing's edge.
    """
    for width in (390, 320):
        page.set_viewport_size({"width": width, "height": 900})
        page.wait_for_timeout(150)
        for f in page.evaluate(NARROW_FIGURES_JS):
            where = f"figure {f['i'] + 1} at {width}px"
            assert f["figure"] <= 1, f"{where} scrolls as a whole by {f['figure']}px"
            assert not f["scrollers"] or f["cue"], f"{where} scrolls with no visible cue"
            assert not f["clipped"], f"{where} paints labels past its edge: {f['clipped']}"
    page.set_viewport_size({"width": 1280, "height": 900})


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


def _placeholders(view) -> list[str]:
    """What a figure from nowhere looks like once rendered.

    `undefined` and `NaN` are the loud ones. The quiet ones are what an ABSENT value becomes after
    formatting: `int(null)` prints a confident "step 0", `Spell(0)` a "Zero layer", and an empty
    list joined into `<code>` an empty box followed by "is still moving".
    """
    import re as _re

    text = view.evaluate(PAGE_TEXT_JS)
    found = [p for p in ("undefined", "NaN", "[object Object]", "Infinity", "null") if p in text]
    # Not `\b` after the zero: `textContent` runs a tile's value straight into its label, so "step
    # 0" is followed by a letter, and the first version of this pattern let "step 0where" through.
    found += [m.group(0) for m in _re.finditer(r"\bstep 0(?![\d.,])|\bZero\b", text)]
    empty = view.evaluate(
        "() => [...document.querySelectorAll('main code')]"
        ".filter(c => !c.textContent.trim()).length"
    )
    if empty:
        found.append(f"{empty} empty <code>")
    return found


def test_every_number_on_the_page_came_from_the_run(page, reduced):
    """No placeholder of any kind — checked on both paints."""
    for view in (page, reduced):
        assert _placeholders(view) == [], "the page rendered a placeholder for a missing figure"


def _real_m(view) -> dict:
    return view.evaluate("async () => JSON.parse(JSON.stringify((await import('./data.js')).M))")


def _render_with(site, m: dict):
    """The page, built from a fabricated `M` served in place of `data.js`."""
    import json as _json

    browser, url = site
    view = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    problems: list[str] = []
    view.on("console", lambda msg: problems.append(msg.text) if msg.type == "error" else None)
    view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    body = "export const M = " + _json.dumps(m) + ";"
    view.route(
        "**/data.js*",
        lambda route: route.fulfill(content_type="application/javascript", body=body),
    )
    view.goto(url)
    view.wait_for_selector("section#reproduce", timeout=10_000)
    view.console_problems = problems
    return view


def _reversed(m: dict) -> dict:
    """Every verdict on the page turned the other way, and the numbers moved to match."""
    import copy as _copy

    m = _copy.deepcopy(m)
    w = m["sweep"]
    w["drift"]["mup"] = w["noise"] * 0.9
    w["margin"]["mup"] = 0.9
    w["above_noise"]["mup"] = False
    w["conclusion"] = "holds"
    w["falls"]["sp"] = False
    s = m["schedules"]
    s.update(
        stop_gap=s["stop_noise"] * 2,
        stop_lower="cosine",
        stop_resolved=True,
        finished_lower="planned",
        finished_resolved=False,
        decay_bought=-s["decay_noise"] * 2,
        decay_inside_noise=False,
        end_gap=s["end_noise"] / 2,
        end_lower="cosine",
        end_resolved=False,
        planned_minus_cut=-s["planned_cut_noise"] / 2,
        planned_cut_resolved=False,
    )
    b = m["bias"]
    b["settles_at_step"] = b["horizon"] // 2
    b["peak_step"] = len(b["ratio_by_step"]) + 5
    return m


def test_every_verdict_flips_when_the_data_does(page, site):
    """The words that rank things come from `M`, so reversed data must reverse them.

    A sentence that says "inside the noise" while the data says otherwise is the failure this
    repository has paid for most: a correct table under a stale verdict. Each pair below is a
    phrase the real page prints and the phrase the reversed data must print instead.
    """
    real = page.evaluate(PAGE_TEXT_JS)
    view = _render_with(site, _reversed(_real_m(page)))
    try:
        flipped = view.evaluate(PAGE_TEXT_JS)
        assert view.console_problems == []
        marks = view.eval_on_selector_all("#thesis .tile", "els => els.map(e => e.className)")
    finally:
        view.context.close()
    pairs = [
        ("narrowed, not removed", "held still"),
        ("best rate falls", "best rate rises"),
        ("WSD is lower by", "cosine is lower by"),
        ("the WSD branch is lower", "is lower than the WSD branch"),
        ("WSD ends lower", "cosine ends lower"),
        ("bought nothing outside the noise", "bought more than the noise"),
        ("cannot rank them", "ranks them"),
        ("never stays there", f"from step {_reversed(_real_m(page))['bias']['settles_at_step']}"),
        ("has already turned", "still climbing"),
    ]
    for before, after in pairs:
        assert before in real, f"the real page no longer says {before!r}; update this pair"
        assert before not in flipped, f"{before!r} survived data that says the opposite"
        assert after in flipped, f"reversed data did not produce {after!r}"
    assert marks[2:] == ["tile good", "tile good"], f"tile marks did not follow the data: {marks}"


def test_absent_values_render_as_absent_rather_than_as_zero(page, site):
    """A missing floor, a layer list with nobody in it, a median with nothing to take it of."""
    m = _real_m(page)
    m["sweep"]["run_floor"] = None
    m["sweep"]["run_floor_loss_gap"] = None
    m["ratios"]["unsettled"] = []
    m["ratios"]["median_settle"] = None
    m["ratios"]["median_settle_none"] = None
    for panel in m["ratios"]["panels"]:
        panel["settles"] = None
    view = _render_with(site, m)
    try:
        assert view.console_problems == []
        assert _placeholders(view) == []
    finally:
        view.context.close()


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


def test_the_opening_tiles_are_marked_by_the_data(page):
    """A tile's mark is a verdict, so it is read from the data rather than typed.

    The muP tile is good if the drift is held inside the floor, a warning if it is only narrowed,
    and bad if muP does not narrow it at all; the decay tile is the page's failure exactly when the
    decay bought nothing outside the noise. With today's data that gives one failure, which a page
    that shows only its wins would not have.
    """
    m = _real_m(page)
    want = [
        "tile watch",
        "tile watch",
        "tile " + {"holds": "good", "narrows": "watch"}.get(m["sweep"]["conclusion"], "bad"),
        "tile " + ("bad" if m["schedules"]["decay_inside_noise"] else "good"),
    ]
    marks = page.eval_on_selector_all("#thesis .tile", "els => els.map(e => e.className)")
    assert marks == want, f"the tiles say {marks}; the data says {want}"
    assert any("bad" in w for w in want), "today's data should put a failure in the opening"


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
    # It opens on both runs, because its legend and caption describe both.
    assert fresh.evaluate(opacity) == ["1", "1"], "the opening view should overlay both runs"
    fresh.click(".sm .tabs button:nth-child(2)")
    fresh.wait_for_timeout(800)
    assert fresh.evaluate(opacity) == ["0", "1"], "switching the tab did not change the panels"
    fresh.click(".sm .tabs button:nth-child(1)")
    fresh.wait_for_timeout(800)
    assert fresh.evaluate(opacity) == ["1", "0"], "the warmup tab did not isolate the warmup run"
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
