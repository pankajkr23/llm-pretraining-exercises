"""Exercise 13's page, tested in a browser, because `node --check` proves almost nothing about it.

A call to an undefined function, a figure reading `NaN`, a control that redraws nothing, and a
layout that scrolls sideways on a phone all parse perfectly. Ported from exercise 10's
`test_trainloop_render.py`, plus one assertion per interactive figure: its control changes what is
drawn, and its end state is painted when the reader has asked for reduced motion.

Named `..._page_render` because `test_reversible_render.py` already exists and tests the renderer,
not the page.

**Served, not opened as a `file://`.** ES modules refuse to load over `file://`, and the shell links
`/_shared/tokens.css` from the site root — so a `file://` test renders a blank, unstyled page.
"""

import functools
import http.server
import json
import os
import re
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
PUBLIC = REPO / "public"
SLUG = "13-reversibility"

pytestmark = pytest.mark.integration

WIDTHS = (1440, 1180, 768, 390, 320)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence the per-request log
        pass


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

    handler = functools.partial(_Quiet, directory=str(PUBLIC))
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


def _open(site, **context):
    browser, url = site
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, **context)
    view = ctx.new_page()
    problems: list[str] = []
    view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    view.goto(url)
    view.wait_for_selector("section#reproduce", timeout=10_000)
    view.console_problems = problems
    return ctx, view


@pytest.fixture(scope="module")
def page(site):
    """The page as most readers get it: motion allowed, nothing touched."""
    ctx, view = _open(site)
    yield view
    ctx.close()


@pytest.fixture(scope="module")
def still(site):
    """The page for a reader who asked for reduced motion: every figure's end state, painted."""
    ctx, view = _open(site, reduced_motion="reduce")
    yield view
    ctx.close()


def test_the_page_loads_without_console_errors(page):
    """A page that throws halfway through renders its first half and looks fine."""
    assert page.console_problems == []


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
    """Every part of the story is present, in an order that reads as an argument.

    Checked by `dataset.role` rather than by heading text, so the prose stays free to change.
    """
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"
    seen = [r for r in roles if r in REQUIRED_ROLES]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(REQUIRED_ROLES), f"the spine is out of order: {first}"


def test_every_figure_has_a_caption_that_says_something(page):
    """Captions here argue — what to conclude and what would refute it — so a short one has not."""
    caps = page.eval_on_selector_all("figure figcaption", "els => els.map(e => e.innerText)")
    figs = page.evaluate("() => document.querySelectorAll('figure').length")
    assert len(caps) == figs > 0, f"{figs} figures but {len(caps)} captions"
    short = [c[:40] for c in caps if len(c) < 120]
    assert not short, f"these captions are too short to be doing any work: {short}"


def test_figures_are_numbered_in_order(page):
    """`figure()` counts; a hand-passed number would eventually give two figures the same one."""
    nums = page.eval_on_selector_all(
        "figure figcaption > b:first-child", "els => els.map(e => e.textContent.trim())"
    )
    assert nums == [f"Figure {i}." for i in range(1, len(nums) + 1)], nums


def test_no_element_is_truncated_at_any_width(page):
    """Visible is not legible: nothing may have a `scrollWidth` larger than its `clientWidth`."""
    try:
        for width in WIDTHS:
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
            assert not clipped, f"at {width}px these are cut off, not merely narrow: {clipped}"
    finally:
        page.set_viewport_size({"width": 1280, "height": 900})


def test_no_svg_label_spills_outside_its_own_figure(page):
    """An `<svg>` scales to its viewBox, so a label is clipped when it escapes that box."""
    spills = page.evaluate(
        """() => {
             const out = [];
             for (const svg of document.querySelectorAll('main svg')) {
               const box = svg.viewBox.baseVal;
               if (!box || !box.width) continue;
               for (const t of svg.querySelectorAll('text')) {
                 if (!t.textContent) continue;
                 const b = t.getBBox();
                 if (b.x < box.x - 1 || b.x + b.width > box.x + box.width + 1) {
                   out.push(`${t.textContent}: ${b.x.toFixed(0)}..${(b.x + b.width).toFixed(0)} ` +
                            `outside 0..${box.width}`);
                 }
               }
             }
             return out;
           }"""
    )
    assert not spills, f"these labels render outside their figure: {spills}"


def test_the_page_never_scrolls_sideways(page):
    """A body that scrolls horizontally is the phone-sized version of the same defect."""
    try:
        for width in WIDTHS:
            page.set_viewport_size({"width": width, "height": 900})
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            assert overflow <= 1, f"the page scrolls {overflow}px sideways at {width}px"
    finally:
        page.set_viewport_size({"width": 1280, "height": 900})


def test_the_rail_marks_where_the_reader_is(page):
    """`.rail-link.on` lands on exactly one link once the reader has moved."""
    page.evaluate("() => document.getElementById('results').scrollIntoView()")
    page.wait_for_timeout(400)
    marked = page.eval_on_selector_all(
        ".rail-link.on", "els => els.map(e => e.getAttribute('href'))"
    )
    assert len(marked) == 1, f"expected exactly one marked rail entry, got {marked}"


def test_the_gutter_the_shared_stylesheet_reserves_is_actually_filled(page):
    """`_shared/page.css` reserves 260px for a rail on wide screens: reserved **and** filled."""
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
    assert measured["rail"] > 100, f"{measured['reserved']:.0f}px reserved, rail {measured['rail']}"
    assert measured["inner"] > 0, ".rail-inner is missing, so the rail's contents hang at the top"


def test_every_number_on_the_page_came_from_the_run(page):
    """A missing key in the generated data renders as `undefined` in perfectly valid HTML."""
    text = page.inner_text("main")
    for poison in ("undefined", "NaN", "Infinity", "[object Object]"):
        assert poison not in text, f"the page rendered {poison!r} — a figure came from nowhere"
    labels = page.evaluate(
        "() => [...document.querySelectorAll('main svg text')].map(t => t.textContent).join(' ')"
    )
    for poison in ("undefined", "NaN", "Infinity"):
        assert poison not in labels, f"a figure label reads {poison!r}"


#: Markup that means a formatting step was skipped: a source token printed instead of applied.
MARKUP_LEAKS = ("[[", "**", "`", "<b>", "</b>", "&amp;", "&lt;", "&gt;")


#: Every region the page writes text into. `textContent`, not `innerText`: the rail's sub-labels are
#: hidden at desktop widths and shown on a phone, and a leak in them was invisible to `innerText` —
#: found by planting one there and watching this guard stay green.
PAGE_TEXT_JS = """() => ['.wrap > .eyebrow', 'h1', '.lede', 'main', '#rail', '#foot']
  .flatMap((s) => [...document.querySelectorAll(s)])
  .map((e) => e.textContent)
  .join(' ')"""


def test_no_source_markup_leaks_into_the_rendered_text(page):
    """`**bold**` that never became bold, or a `<b>` escaped into visible text, reads as a typo."""
    text = page.evaluate(PAGE_TEXT_JS)
    assert len(text) > 10_000, "the page text came back nearly empty; the selectors have gone stale"
    leaked = [m for m in MARKUP_LEAKS if m in text]
    assert not leaked, f"the rendered page shows raw markup: {leaked}"


def test_the_markup_guard_can_fail(page):
    """Its twin: a planted escaped `<b>` and a literal `**` are both caught, even when hidden."""
    planted = page.evaluate(
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
    """The glossary heading promises a figure in every entry; it is checkable, so it is checked."""
    entries = page.evaluate(
        """() => {
             const out = [];
             const dl = document.querySelector('.gloss');
             if (!dl) return out;
             const kids = [...dl.children];
             for (let i = 0; i < kids.length; i += 1) {
               if (kids[i].tagName !== 'DT') continue;
               const dd = kids[i + 1];
               if (dd && dd.tagName === 'DD') {
                 out.push({term: kids[i].textContent, body: dd.textContent});
               }
             }
             return out;
           }"""
    )
    assert entries, "no glossary entries found; the selector has gone stale"
    numberless = [e["term"] for e in entries if not re.search(r"\d", e["body"])]
    assert not numberless, f"these glossary entries carry no figure from the run: {numberless}"


# ------------------------------------------------------------------------ the interactive figures


def _walk(view):
    return view.evaluate(
        """() => {
             const wrap = document.querySelector('.walk');
             return {
               pos: wrap.dataset.pos,
               held: [...document.querySelectorAll('.walkfig .w-row')]
                 .map((g, i) => g.getAttribute('class').includes('held') ||
                                g.getAttribute('class').includes('rebuilt') ? i : -1)
                 .filter(i => i >= 0),
               out: document.querySelector('.walk-out').textContent,
               max: document.querySelector('#walk-pos').max,
             };
           }"""
    )


def test_the_walk_moves_the_held_pair_down_the_stack(page):
    """Figure 1: moving the position changes which states are drawn as held."""
    before = _walk(page)
    page.evaluate(
        """() => {
             const r = document.querySelector('#walk-pos');
             r.value = '5';
             r.dispatchEvent(new Event('input', { bubbles: true }));
           }"""
    )
    after = _walk(page)
    assert before["held"] != after["held"], "the slider moved and the drawing did not"
    assert len(after["held"]) == len(before["held"]) == 2, "more than two states were drawn held"
    assert after["out"] != before["out"]


def test_the_walk_paints_its_end_state_under_reduced_motion(still):
    """With reduced motion the walk opens at the bottom of the stack, rebuild complete."""
    state = _walk(still)
    assert state["pos"] == state["max"], state
    assert "of" in state["out"]


def _trials(view):
    return view.evaluate(
        """() => {
             const w = document.querySelector('.trials');
             return { state: w.dataset.state, selected: w.dataset.selected,
                      verdict: document.querySelector('.tverdict').innerText,
                      pressed: document.querySelectorAll('.tcell[aria-pressed="true"]').length };
           }"""
    )


def test_the_gate_answers_for_whichever_candidate_is_chosen(page):
    """Figure 3: choosing a candidate changes the verdict, and the chosen rule is accepted."""
    page.click(".tcell.chosen")
    chosen = _trials(page)
    assert chosen["state"] == "chosen" and chosen["pressed"] == 1
    page.click(".tcell.refused")
    refused = _trials(page)
    assert refused["state"] == "refused" and refused["verdict"] != chosen["verdict"]
    rules = _data()["trials"]["rules"]
    cand = next(r for r in rules if f"{r['rule']}{r['h']:g}" == refused["selected"])
    error = _pct_sig(cand["agreement"]["gradient_error"])
    assert error in refused["verdict"], (
        f"a refusal must show the measured error it was refused for ({error}): {refused['verdict']}"
    )


def _pct_sig(x: float) -> str:
    """The page's `pctSig`: a share as a percentage to two significant figures."""
    return f"{float(f'{x * 100:.2g}'):g}%"


def test_the_gate_shows_a_refusal_with_its_error_under_reduced_motion(still):
    """The end state is the best-scoring refused candidate, with its own measured error."""
    refused = [r for r in _data()["trials"]["rules"] if not r["eligible"]]
    state = _trials(still)
    if not refused:
        assert state["state"] == "chosen", state
        return
    best = min(refused, key=lambda r: r["final_val"])
    assert state["state"] == "refused", state
    assert best["rule"] in state["verdict"]
    assert _pct_sig(best["agreement"]["gradient_error"]) in state["verdict"]


def _budget(view):
    return view.evaluate(
        """() => ({
             batch: document.querySelector('.budget').dataset.batch,
             widths: [...document.querySelectorAll('.budgetfig .b-act')]
               .map(r => r.getAttribute('width')),
             measured: document.querySelectorAll('.budgetfig .m-measured').length,
             read: document.querySelector('.b-read').textContent,
           })"""
    )


def test_the_budget_bar_follows_the_batch(site):
    """Figure 4: the slider changes the bars and the reading; the measured markers are always drawn.

    There is no reveal any more: the opening tiles state the measured limits before this figure,
    so asking the reader to predict them would only pretend to ask.
    """
    ctx, view = _open(site)
    try:
        before = _budget(view)
        assert before["measured"] == 2, "both measured limits should be drawn"
        view.evaluate(
            """() => {
                 const r = document.querySelector('#budget-batch');
                 r.value = '800';
                 r.dispatchEvent(new Event('input', { bubbles: true }));
               }"""
        )
        moved = _budget(view)
        assert moved["widths"] != before["widths"], "the batch moved and the bars did not"
        assert moved["read"] != before["read"] and moved["batch"] != before["batch"]
    finally:
        ctx.close()


def test_the_budget_paints_the_measured_limit_under_reduced_motion(still):
    """With reduced motion the figure opens at the reversible model's own measured largest batch."""
    data = _data()
    assert _budget(still)["batch"] == str(data["max_batch"]["reversible"]["measured_max_batch"])


# ------------------------------------------------------------------------ verdicts follow the data


def _data() -> dict:
    """The page's generated data, read the way a reader's browser does — from `data.js`."""
    text = (REPO / "src" / "exercises" / SLUG / "web" / "data.js").read_text(encoding="utf-8")
    head = "export const M = "
    return json.loads(text[text.index(head) + len(head) :].rstrip()[:-1])


#: Rebuild the page from a copy of its data in which every comparison comes out the other way:
#: the reversible model keeps more, fits a smaller batch, runs faster in every trial, does better at
#: the large batch, and has nothing refused by the gate. A fresh module instance builds into a
#: cleared page, so the text read back is only what the reversed data produced.
REVERSED_JS = """async () => {
  const { M } = await import('./data.js?reversed');
  const R = structuredClone(M);
  R.fixed.kept_ratio = 1 / M.fixed.kept_ratio;
  R.fixed.baseline.saved_bytes = M.fixed.reversible.saved_bytes;
  R.fixed.reversible.saved_bytes = M.fixed.baseline.saved_bytes;
  R.max_batch.batch_ratio = 1 / M.max_batch.batch_ratio;
  R.max_batch.baseline.measured_max_batch = M.max_batch.reversible.measured_max_batch;
  R.max_batch.reversible.measured_max_batch = M.max_batch.baseline.measured_max_batch;
  R.fixed.loss_gap = -M.fixed.loss_gap;
  R.max_run.loss_gap = -M.max_run.loss_gap;
  R.trial_speeds.verdict = 'faster';
  for (const r of R.trials.rules) r.eligible = true;
  R.trials.ineligible = {};
  for (const id of ['main', 'rail', 'foot']) document.getElementById(id).replaceChildren();
  const { buildPage } = await import('./chapters.js?reversed');
  buildPage(R);
  return document.querySelector('main').textContent.replace(/\\s+/g, ' ');
}"""

#: Phrases each direction must produce, and the other direction must not.
FORWARD = (
    "fewer bytes kept for the backward pass by the reversible model",
    "slower than every baseline trial",
    "The large batch trained less",
    "Spending that memory on a larger batch cost loss.",
    "The gate did not change which rule was used",
)
REVERSED = (
    "more bytes kept for the backward pass by the reversible model",
    "faster than every baseline trial",
    "The large batch trained further",
    "Spending that memory on a larger batch did not cost loss.",
    "every candidate rebuilt its gradients within it",
)


def test_the_verdict_words_flip_when_the_data_does(site):
    """No verdict on this page is typed: reverse every comparison in the data and the words follow.

    The first review of this page found "fewer", "held", "slower" and "did not change which rule
    was used" typed into the source while the page claimed every number was computed — so a re-run
    that came out the other way would have printed the old verdicts over new numbers.
    """
    ctx, view = _open(site)
    try:
        forward = view.inner_text("main")
        missing = [w for w in FORWARD if w not in forward]
        assert not missing, f"the published page lacks its own verdicts: {missing}"
        assert not [w for w in REVERSED if w in forward]
        reversed_text = view.evaluate(REVERSED_JS)
        stale = [w for w in FORWARD if w in reversed_text]
        assert not stale, f"these verdicts did not follow the reversed data: {stale}"
        flipped = [w for w in REVERSED if w not in reversed_text]
        assert not flipped, f"the reversed data did not produce: {flipped}"
        for poison in ("undefined", "NaN", "Infinity", "[object Object]"):
            assert poison not in reversed_text, f"the reversed page rendered {poison!r}"
    finally:
        ctx.close()


def test_the_opening_tiles_are_marked_by_the_data(page):
    """Each tile's mark follows its number: the green, amber and red are not chosen by hand."""
    data = _data()
    rules = data["trials"]["rules"]
    expected = [
        "good" if data["fixed"]["kept_ratio"] > 1 else "bad",
        "good" if data["max_batch"]["batch_ratio"] > 1 else "bad",
        "bad" if data["max_run"]["loss_gap"] > 0 else "good",
        "bad" if any(not r["eligible"] for r in rules) else "good",
    ]
    marks = page.eval_on_selector_all(
        ".tiles .tile", "els => els.map(e => e.className.replace('tile ', ''))"
    )
    assert marks == expected, f"tile marks {marks} do not follow the data {expected}"


def test_no_figure_is_cut_off_on_a_phone(site):
    """At 390 and 320 no figure scrolls or clips, and every label lies inside its drawing's box.

    A figure that scrolls on a phone hides its right-hand side, which is where this page's results
    sat before every drawing was redrawn at a phone's width. Scrolling inside a figure is not
    deliberate anywhere here, so it is asserted absent rather than given a scroll cue.
    """
    ctx, view = _open(site)
    try:
        for width in (390, 320):
            view.set_viewport_size({"width": width, "height": 900})
            view.wait_for_timeout(150)
            found = view.evaluate(
                """() => {
                     const out = [];
                     for (const f of document.querySelectorAll('main figure')) {
                       for (const e of [f, ...f.querySelectorAll('*')]) {
                         if (!(e instanceof HTMLElement)) continue;
                         if (e.scrollWidth > e.clientWidth + 1) {
                           out.push(`${e.tagName.toLowerCase()}.${e.className}: ` +
                                    `${e.scrollWidth} > ${e.clientWidth}`);
                         }
                       }
                       for (const svg of f.querySelectorAll('svg')) {
                         const box = svg.getBoundingClientRect();
                         for (const t of svg.querySelectorAll('text')) {
                           if (!t.textContent) continue;
                           const r = t.getBoundingClientRect();
                           if (r.left < box.left - 1 || r.right > box.right + 1 ||
                               r.top < box.top - 1 || r.bottom > box.bottom + 1) {
                             out.push(`label "${t.textContent}" outside its drawing`);
                           }
                         }
                       }
                     }
                     return out;
                   }"""
            )
            assert not found, f"at {width}px: {found}"
    finally:
        ctx.close()
