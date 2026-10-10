"""Exercise 12's page, tested in a browser, because `node --check` proves almost nothing about it.

Copied in shape from exercise 10's `test_trainloop_render.py`. A call to an undefined function, a
table that renders every cell as `undefined`, a figure reading `NaN`, and a layout that scrolls
sideways on a phone all parse perfectly.

**Served, not opened as a `file://`.** ES modules refuse to load over `file://`, and the shell links
`/_shared/tokens.css` from the site root — so a `file://` test renders a blank, unstyled page and
passes any assertion that only checks the title.

Each interactive figure gets two assertions: its control changes what is drawn, and its end state
is painted when the reader asks for reduced motion — because the page promises that a reader who
never touches a control still sees every figure's conclusion.
"""

import functools
import http.server
import importlib.util
import json
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
SLUG = "12-distributed-training"
EXERCISE = Path(__file__).resolve().parents[1]

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


def _open(site, data_js: str | None = None, **context):
    """Open the page; with `data_js`, serve that in place of the tracked `data.js`."""
    browser, url = site
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, **context)
    view = ctx.new_page()
    problems: list[str] = []
    view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    if data_js is not None:
        view.route(
            f"**/{SLUG}/data.js*",
            lambda route: route.fulfill(body=data_js, content_type="application/javascript"),
        )
    view.goto(url)
    view.wait_for_selector("section#reproduce", timeout=10_000)
    view.console_problems = problems
    return ctx, view


@pytest.fixture(scope="module")
def page(site):
    """The page as most readers get it: motion allowed, nothing touched yet."""
    ctx, view = _open(site)
    yield view
    ctx.close()


@pytest.fixture(scope="module")
def still(site):
    """The page as a reader who asked for reduced motion gets it."""
    ctx, view = _open(site, reduced_motion="reduce")
    view.wait_for_timeout(300)
    yield view
    ctx.close()


def test_the_page_loads_without_console_errors(page, still):
    """A page that throws halfway through renders its first half and looks fine."""
    assert page.console_problems == []
    assert still.console_problems == []


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
    """Checked by `data-role` rather than heading text, so the prose stays free to change."""
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"
    seen = [r for r in roles if r in REQUIRED_ROLES]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(REQUIRED_ROLES), f"the spine is out of order: {first}"


def test_every_figure_has_a_caption_that_says_something(page):
    """A caption argues; a short one has not done its job."""
    caps = page.eval_on_selector_all("figure figcaption", "els => els.map(e => e.innerText)")
    figs = page.evaluate("() => document.querySelectorAll('figure').length")
    assert len(caps) == figs > 0, f"{figs} figures but {len(caps)} captions"
    short = [c[:40] for c in caps if len(c) < 120]
    assert not short, f"these captions are too short to be doing any work: {short}"


def test_every_svg_is_named_for_a_screen_reader(page):
    """`role="img"` and an `aria-label` on every drawing, or it is silent to a screen reader."""
    unnamed = page.evaluate(
        """() => [...document.querySelectorAll('main svg')]
             .filter(s => s.getAttribute('role') !== 'img' || !s.getAttribute('aria-label'))
             .map(s => s.getAttribute('class'))"""
    )
    assert not unnamed, f"these drawings have no accessible name: {unnamed}"


def test_no_element_is_truncated_at_any_width(page):
    """Visible is not legible: nothing may have a `scrollWidth` beyond its `clientWidth`."""
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
    """An `<svg>` scales to its `viewBox`, so a label is clipped when it escapes that box."""
    spills = page.evaluate(
        """() => {
             const out = [];
             for (const svg of document.querySelectorAll('main svg')) {
               const box = svg.viewBox.baseVal;
               if (!box || !box.width) continue;
               for (const t of svg.querySelectorAll('text')) {
                 if (!t.textContent) continue;
                 const b = t.getBBox();
                 if (b.x < box.x - 1 || b.x + b.width > box.x + box.width + 1 ||
                     b.y < box.y - 1 || b.y + b.height > box.y + box.height + 1) {
                   const xs = `${b.x.toFixed(0)}..${(b.x + b.width).toFixed(0)}`;
                   out.push(`${t.textContent}: x ${xs} ` +
                            `y ${b.y.toFixed(0)}..${(b.y + b.height).toFixed(0)}`);
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
    """`_shared/page.css` reserves 260px for a rail on wide screens; reserved AND filled."""
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
    assert measured["inner"] > 0, ".rail-inner is missing; the rail's contents would hang"


#: Every region the page writes text into, read as `textContent` rather than `innerText`. The
#: rail's sub-labels are hidden at desktop widths and shown on a phone, and `innerText` skips what
#: is not rendered — a leak there was invisible to the first version of page 13's guard, found by
#: planting one. `body` itself is not used because its `textContent` includes the module script.
PAGE_TEXT_JS = """() => ['.wrap > .eyebrow', 'h1', '.lede', 'main', '#rail', '#foot']
  .flatMap((s) => [...document.querySelectorAll(s)])
  .map((e) => e.textContent)
  .join(' ')"""

#: A figure from nowhere, as valid HTML renders it.
POISON = ("undefined", "NaN", "[object Object]", "null")

#: Markup printed instead of applied.
MARKUP_LEAKS = ("[[", "**", "`", "<b>", "</b>", "&amp;", "&lt;", "&gt;", "&nbsp;")


def test_every_number_on_the_page_came_from_the_run(page, still):
    """A missing key in the generated data renders as `undefined` in perfectly valid HTML."""
    for view in (page, still):
        text = view.evaluate(PAGE_TEXT_JS)
        assert len(text) > 10_000, "the page text came back nearly empty; selectors are stale"
        found = [p for p in POISON if p in text]
        assert not found, f"the page rendered {found} — a figure came from nowhere"


def test_no_markup_leaks_into_the_rendered_text(page, still):
    """Markdown or escaped HTML shown as literal characters is a defect no lexical guard sees.

    Exercise 05's retro-fit shipped a literal `<b>` and stray `*` markers with the suite green; the
    page is built from template strings full of `<b>`, so one mis-placed `textContent` prints them.
    """
    for view in (page, still):
        text = view.evaluate(PAGE_TEXT_JS)
        leaked = [m for m in MARKUP_LEAKS if m in text]
        assert not leaked, f"the rendered page shows raw markup: {leaked}"


def test_the_text_guards_see_what_is_hidden(page):
    """The twin: a leak and a poison planted in the rail, hidden, are both still read."""
    planted = page.evaluate(
        """(js) => {
             const p = document.createElement('span');
             p.textContent = 'a <b>planted</b> **leak** undefined';
             p.style.display = 'none';
             document.querySelector('#rail').append(p);
             const text = (0, eval)(js)();
             p.remove();
             return text;
           }""",
        PAGE_TEXT_JS,
    )
    assert [m for m in MARKUP_LEAKS if m in planted] == ["**", "<b>", "</b>"]
    assert "undefined" in planted


def test_every_glossary_entry_carries_a_number_from_the_run(page):
    """The glossary's heading promises a figure in every entry; checkable, so checked."""
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


def _page_data() -> dict:
    """The tracked `data.js` as a dict — what the page under test was built from."""
    text = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    return json.loads(
        text[text.index("export const M = ") + len("export const M = ") :].rstrip()[:-1]
    )


def test_the_opening_tiles_lead_with_a_failure(page):
    """A page that shows only its wins has not earned them: one tile must be a failure — and the
    failure tile's colour must come from the data, not from a literal class in the source."""
    marks = page.eval_on_selector_all(".tiles .tile", "els => els.map(e => e.className)")
    assert 3 <= len(marks) <= 4, marks
    assert sum("bad" in m for m in marks) >= 1, f"no failure among the opening tiles: {marks}"
    never = _page_data()["page"]["slider"]["never_fits"]["1"]
    assert ("bad" in marks[-1]) == never, f"the ZeRO-1 tile is {marks[-1]!r}, never_fits={never}"


def _reversed_data_js() -> str:
    """The real bundle with the three verdicts the page states turned the other way round.

    A 13-billion-weight ladder puts ZeRO-1's floor under the card and makes no first size fill it
    exactly; halving the key-bias difference stops it drifting further than the rest. Built by the
    real renderer, so `page_numbers()` decides every flag exactly as it does for the tracked page.
    """
    import copy

    from zerosim import formulas

    spec = importlib.util.spec_from_file_location(
        "zerosim_render_for_flip", EXERCISE / "tools" / "render_results.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    run = copy.deepcopy(
        json.loads((EXERCISE / "results" / "zero.json").read_text(encoding="utf-8"))
    )
    run["ladder"]["params"] = 13_000_000_000
    run["ladder"]["rows"] = formulas.ladder(
        run["ladder"]["params"],
        tuple(run["config"]["ladder_world_sizes"]),
        run["main_mode"],
        run["ladder"]["card_bytes"],
    )
    eq = run["equivalence"]
    # Both to zero: the key bias no longer drifts further, and the single device no longer differs.
    eq["fp32_weights_max_abs_key_bias"] = 0.0
    eq["fp32_weights_max_abs_except_key_bias"] = 0.0
    return renderer.render_page_data(run)


def test_the_words_flip_when_the_data_does(site):
    """Every verdict the review found typed — "never fits", "fills the card exactly", the key bias
    "drifting further" — must change when the data changes. Rendered twice: the tracked data, then
    a reversed copy. A sentence that reads the same both times was typed, whatever it looks like."""

    def read(view) -> dict:
        view.click(".predict button")
        view.click(".adv-cell.ref")
        return view.evaluate(
            """() => ({
                 text: document.querySelector('main').textContent,
                 tile: document.querySelector('.tiles .tile:last-child').className,
                 tileText: document.querySelector('.tiles .tile:last-child').textContent,
                 verdict: document.querySelector('.predict-v').textContent,
                 expected: document.querySelector('section[data-role="expected"]').textContent,
                 ref: document.querySelector('.adv-cell.ref').className,
               })"""
        )

    ctx, real = _open(site)
    try:
        before = read(real)
    finally:
        ctx.close()
    ctx, flipped = _open(site, data_js=_reversed_data_js())
    try:
        after = read(flipped)
        assert flipped.console_problems == []
    finally:
        ctx.close()

    assert "never fits" in before["tileText"] and "bad" in before["tile"]
    assert "never fits" not in after["tileText"] and "bad" not in after["tile"]
    assert "fits from" in after["tileText"]

    assert "fill the card exactly" in before["text"] and "exactly" in before["verdict"]
    assert (
        "fill the card exactly" not in after["text"]
        and "fills the card exactly" not in after["text"]
    )
    assert "exactly" not in after["verdict"]

    assert "drift further" not in before["text"]
    assert "did not drift further" in after["text"]
    assert "held, except" in before["expected"] and "held, except" not in after["expected"]

    assert "diff" in before["ref"] and "diff" not in after["ref"] and "same" in after["ref"]


def test_the_limits_are_in_the_open(page):
    """Limits sit in `.notice` with a list, never in a `<details>` a reader must open."""
    info = page.evaluate(
        """() => {
             const s = document.querySelector('section[data-role="limits"]');
             return {
               notice: s.classList.contains('notice'),
               items: s.querySelectorAll('ul.limitlist li').length,
               details: s.querySelectorAll('details').length,
             };
           }"""
    )
    assert info["notice"] and info["items"] >= 3 and info["details"] == 0, info


# ------------------------------------------------------------------ the interactive figures


def _ring_state(view):
    return view.evaluate(
        """() => {
             const svg = document.querySelector('svg.ringfig');
             return {
               full: svg.querySelectorAll('rect.chunk.full').length,
               cells: svg.querySelectorAll('rect.chunk').length,
               arrows: svg.querySelectorAll('.ring-arrows path').length,
               out: document.querySelector('.ring-out').textContent,
             };
           }"""
    )


def test_the_ring_step_changes_what_is_drawn(page):
    """Moving the step must move chunks: arrows appear, and sums complete."""
    before = _ring_state(page)
    page.evaluate(
        """() => {
             const r = document.getElementById('ring-step');
             r.value = r.max;
             r.dispatchEvent(new Event('input'));
           }"""
    )
    after = _ring_state(page)
    assert before["arrows"] == 0 and before["full"] == 0, before
    assert after["arrows"] > 0, after
    assert after["full"] == after["cells"], f"the last step should complete every sum: {after}"


def test_the_ring_end_state_is_painted_under_reduced_motion(still):
    state = _ring_state(still)
    assert state["full"] == state["cells"] > 0, state


def test_the_budget_precision_changes_what_is_drawn(page):
    """fp32 has no master copy, so switching precision must change the bars' geometry."""

    def widths():
        return page.eval_on_selector_all(
            "svg.budgetfig rect.seg", "els => els.map(e => Number(e.getAttribute('width')))"
        )

    before = widths()
    page.click(".budget-ctl .tabs button[data-value='fp32']")
    page.wait_for_timeout(800)  # the morph is ~550ms
    after = widths()
    assert before != after, "switching precision redrew nothing"
    pressed = page.eval_on_selector_all(
        ".budget-ctl button[aria-pressed='true']", "els => els.map(e => e.dataset.value)"
    )
    assert "fp32" in pressed, pressed
    page.click(".budget-ctl .tabs button[data-value='bf16-mixed']")
    page.wait_for_timeout(800)


def test_the_budget_end_state_is_painted_under_reduced_motion(still):
    """No morph to wait for: the bars are at their final widths and a stage is read out."""
    widths = still.eval_on_selector_all(
        "svg.budgetfig rect.seg", "els => els.map(e => Number(e.getAttribute('width')))"
    )
    assert max(widths) > 0, widths
    assert still.locator(".budget-read table").count() == 1


def test_the_ladder_slider_changes_what_is_drawn(page):
    """Moving N must move the cursor and change at least one stage's verdict."""

    def state():
        return page.evaluate(
            """() => ({
                 x: document.querySelector('svg.ladderfig line.cursor').getAttribute('x1'),
                 read: document.querySelector('.ladder-read').innerText,
               })"""
        )

    page.evaluate(
        """() => {
             const r = document.getElementById('ladder-n');
             r.value = r.min;
             r.dispatchEvent(new Event('input'));
           }"""
    )
    low = state()
    page.evaluate(
        """() => {
             const r = document.getElementById('ladder-n');
             r.value = r.max;
             r.dispatchEvent(new Event('input'));
           }"""
    )
    high = state()
    assert low["x"] != high["x"]
    assert low["read"] != high["read"]
    assert "does not fit" in low["read"]


def test_the_ladder_reveal_pins_the_guess_beside_the_answer(page):
    """The prediction is kept on the chart after the answer appears — the gap is the lesson."""
    page.fill("#ladder-guess", "8")
    page.click(".predict button")
    marks = page.eval_on_selector_all(
        "svg.ladderfig .guessmarks line", "els => els.map(e => e.getAttribute('class'))"
    )
    assert sorted(marks) == ["answer", "guess"], marks
    assert "You guessed" in page.inner_text(".predict-v")


def test_the_ladder_end_state_is_painted_under_reduced_motion(still):
    marks = still.eval_on_selector_all("svg.ladderfig .guessmarks line.answer", "els => els.length")
    assert marks == 1
    assert "the formula says" in still.inner_text(".predict-v")


def test_the_adversary_reveals_what_it_is_asked(page):
    """Pressing a run reveals its difference from data parallelism; untouched runs stay hidden."""
    assert page.locator(".adv-cell.same, .adv-cell.diff").count() == 0
    page.locator(".adv-cell").first.click()
    assert page.locator(".adv-cell.same, .adv-cell.diff").count() == 1
    assert "Tried 1" in page.inner_text(".adv-read")


def test_the_adversary_end_state_is_painted_under_reduced_motion(still):
    """Every run revealed, and each one's red or green decided by the data: the ZeRO runs by their
    own difference from data parallelism, the single device by whether it differs at all."""
    page_data = _page_data()["page"]
    expected_diff = sum(
        not same
        for mode in page_data["identical_to_dp"].values()
        for k, same in mode.items()
        if k != "0"
    ) + int(page_data["reference_differs"])
    same = still.locator(".adv-cell.same").count()
    diff = still.locator(".adv-cell.diff").count()
    assert diff == expected_diff and same >= 1, (same, diff, expected_diff)
    assert still.locator(".adv-cell:not(.same):not(.diff)").count() == 0


def test_a_drawing_that_scrolls_says_so_on_a_phone(page):
    """At 390 and 320 the drawings are wider than the screen by design — they keep a legible label
    size by scrolling inside their own box. That is only honest if the reader is told: every box
    that overflows shows its cue, no box that fits shows one, and no figure itself overflows."""
    for width in (390, 320):
        page.set_viewport_size({"width": width, "height": 900})
        page.wait_for_timeout(250)
        state = page.evaluate(
            """() => ({
                 boxes: [...document.querySelectorAll('.figscroll')].map((b) => ({
                   over: b.scrollWidth > b.clientWidth + 1,
                   cue: !b.parentElement.querySelector('.scrollcue').hidden,
                 })),
                 figures: [...document.querySelectorAll('main figure')]
                   .filter((f) => f.scrollWidth > f.clientWidth + 1).length,
               })"""
        )
        assert state["boxes"], "no drawing scrollers found; the selector has gone stale"
        wrong = [b for b in state["boxes"] if b["over"] != b["cue"]]
        assert not wrong, f"at {width}px a scroll cue disagrees with its box: {wrong}"
        assert any(b["over"] for b in state["boxes"]), f"nothing overflows at {width}px?"
        assert state["figures"] == 0, f"at {width}px a figure itself overflows"
    page.set_viewport_size({"width": 1280, "height": 900})
    page.wait_for_timeout(250)
