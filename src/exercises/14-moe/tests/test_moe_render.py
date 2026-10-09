"""Exercise 14's page, tested in a browser, because `node --check` proves almost nothing about it.

A call to an undefined function, a table that renders every cell as `undefined`, a figure reading
`NaN`, and a layout that scrolls sideways on a phone all parse perfectly. Copied from exercise 10's
`test_trainloop_render.py`, plus one assertion per interactive figure: that its control changes what
is drawn, and that its end state is painted for a reader who prefers reduced motion.

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
SLUG = "14-moe"
EXERCISE = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration


def _data() -> dict:
    """The page's generated data, read the way the page reads it."""
    text = (EXERCISE / "web" / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("{") : text.rindex(";")])


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


@pytest.fixture(scope="module")
def page(site):
    """The page open once, at a laptop width, with every console error recorded."""
    browser, url = site
    view = browser.new_page(viewport={"width": 1280, "height": 900})
    problems: list[str] = []
    view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    view.goto(url)
    view.wait_for_selector("section#reproduce", timeout=10_000)
    view.console_problems = problems
    yield view
    view.close()


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
    """Checked by `data-role`, so the prose stays free to change; order, because "limits" before
    "results" reads as hedging and "conclusion" before the evidence reads as a press release."""
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


def test_every_figure_is_numbered_once_and_in_order(page):
    """`figure()` counts; a passed number is how a page ends up with two Figure 1s."""
    nums = page.eval_on_selector_all(
        "figure figcaption > b:first-child", "els => els.map(e => e.textContent.trim())"
    )
    assert nums == [f"Figure {i}." for i in range(1, len(nums) + 1)], nums


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
    """An `<svg>` scales to its `viewBox`, so a label is clipped when its box escapes that box."""
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
    """A missing key in the generated data renders as the string `undefined` in valid HTML."""
    text = page.inner_text("main")
    for poison in ("undefined", "NaN", "[object Object]", "null"):
        assert poison not in text, f"the page rendered {poison!r} — a figure came from nowhere"
    #: A missing step formats as 0 through `int(null)`, so "after step 0" is the same defect in a
    #: form that reads as a fact. Real steps here start at 1; the start of training is "start".
    #: "step 0." ending a sentence is the defect; "bias step 0.001" is a step size, not a step.
    zero = re.findall(r".{0,30}\bstep 0(?!\d|\.\d).{0,20}", text)
    assert not zero, f"a missing step was printed as step 0: {zero}"


#: Markup that means a formatting step was skipped: a source token printed instead of applied.
MARKUP_LEAKS = ("[[", "**", "`", "<b>", "</b>", "<i>", "&amp;", "&lt;", "&gt;")

#: Every region the page writes text into. `textContent`, not `innerText`: the rail's sub-labels
#: are hidden at desktop widths and shown on a phone, so `innerText` cannot see a leak in them.
#: Taken from exercise 13's guard, which found that by planting one there.
PAGE_TEXT_JS = """() => ['.wrap > .eyebrow', 'h1', '.lede', 'main', '#rail', '#foot']
  .flatMap((s) => [...document.querySelectorAll(s)])
  .map((e) => e.textContent)
  .join(' ')"""


def test_no_markup_reaches_the_reader_as_literal_text(page):
    """Markdown or escaped HTML that leaks through `innerHTML` shows as its own characters."""
    text = page.evaluate(PAGE_TEXT_JS)
    assert len(text) > 10_000, "the page text came back nearly empty; the selectors have gone stale"
    leaked = [m for m in MARKUP_LEAKS if m in text]
    assert not leaked, f"the rendered page shows raw markup: {leaked}"


def test_the_markup_guard_can_fail(page):
    """Its twin: a planted escaped `<b>` and a literal `**`, hidden in the rail, are both caught."""
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
    """The glossary's heading promises a figure in every entry; it is checkable, so checked."""
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
    numberless = [e["term"] for e in entries if not re.search(r"\d", e["body"])]
    assert not numberless, f"these glossary entries carry no figure from the run: {numberless}"


def test_the_opening_tiles_carry_a_failure(page):
    """A page that shows only its wins has not earned the ones it shows."""
    marks = page.eval_on_selector_all(".tiles .tile", "els => els.map(e => e.className)")
    assert any("bad" in m for m in marks), marks


# ---------------------------------------------------------------- the interactive figures


def _state(page, selector: str, attr: str = "state") -> str:
    return page.eval_on_selector(selector, f"e => e.dataset.{attr}")


def test_the_conversion_figure_changes_what_it_draws(page):
    """Rescaling must change the converted output; a wrong selection must be refused, not drawn."""
    wrap = "#fig-conversion .dx"
    out = "#fig-conversion rect.dx-out.conv"
    assert _state(page, wrap) == "same", "the figure must open on the state the model runs"
    page.click("#fig-conversion button:has-text('used as scored')")
    scored = float(page.get_attribute(out, "width"))
    assert _state(page, wrap) == "different"
    page.click("#fig-conversion button:has-text('rescaled to add up to one')")
    page.wait_for_timeout(50)
    rescaled = float(page.get_attribute(out, "width"))
    assert _state(page, wrap) == "same"
    assert rescaled > scored, f"rescaling drew the same output ({scored} → {rescaled})"
    # A third expert is a routing the model never does: refused, and nothing drawn.
    page.click("#fig-conversion .dx-pick button:has-text('E3')")
    assert _state(page, wrap) == "refused"
    assert float(page.get_attribute(out, "width")) == 0
    page.click("#fig-conversion .dx-pick button:has-text('E1')")
    assert _state(page, wrap) == "same", "any two experts must give the dense output"
    assert float(page.get_attribute(out, "width")) == rescaled


def test_the_balance_scrubber_changes_what_it_draws(page):
    """Moving through training must move the bars; guessing must unveil the idle strip."""
    data = _data()
    heights = (
        "() => [...document.querySelectorAll('#fig-balance rect.acc-bar')]"
        ".map(r => r.getAttribute('height')).join()"
    )
    slider = "#fig-balance input#acc-step"
    page.fill(slider, "0")
    page.dispatch_event(slider, "input")
    start = page.evaluate(heights)
    page.fill(slider, str(len(data["layer0"]) // 2))
    page.dispatch_event(slider, "input")
    assert page.evaluate(heights) != start, "the scrubber moved and the bars did not"
    hidden = page.eval_on_selector("#fig-balance line.acc-tick", "e => getComputedStyle(e).opacity")
    assert hidden == "0", "the idle strip is visible before the reader has guessed"
    page.click("#fig-balance button.acc-reveal")
    page.wait_for_timeout(700)
    shown = page.eval_on_selector("#fig-balance line.acc-tick", "e => getComputedStyle(e).opacity")
    assert shown == "1"
    most = page.inner_text("#fig-balance .acc-answer b.acc-most")
    assert most == str(data["balance"]["worst_dead"]), most


def test_the_budget_slider_changes_what_it_draws(page):
    """A k the run did not use must draw differently, and hatched, with no invented speed."""
    data = _data()
    slider = "#fig-cost input#bud-k"
    page.fill(slider, str(data["config"]["n_experts"]))
    page.dispatch_event(slider, "input")
    assert _state(page, "#fig-cost .budget", "k") == str(data["config"]["n_experts"])
    assert page.locator("#fig-cost rect.seg.expert.hatched").count() == data["config"]["n_experts"]
    assert "not run" in page.eval_on_selector("#fig-cost svg", "e => e.textContent")
    page.fill(slider, str(data["config"]["top_k"]))
    page.dispatch_event(slider, "input")
    assert page.locator("#fig-cost rect.seg.expert.hatched").count() == 0


def test_every_end_state_is_painted_under_reduced_motion(site):
    """A reader who prefers reduced motion gets every figure's end state without touching it."""
    browser, url = site
    ctx = browser.new_context(reduced_motion="reduce", viewport={"width": 1280, "height": 900})
    view = ctx.new_page()
    try:
        view.goto(url)
        view.wait_for_selector("section#reproduce", timeout=10_000)
        data = _data()
        assert _state(view, "#fig-conversion .dx") == "same"
        assert view.eval_on_selector("#fig-balance .acc", "e => e.classList.contains('revealed')")
        assert view.input_value("#fig-balance input#acc-step") == str(len(data["layer0"]) - 1)
        opacity = view.eval_on_selector(
            "#fig-balance line.acc-tick", "e => getComputedStyle(e).opacity"
        )
        assert opacity == "1", "the idle strip is still veiled under reduced motion"
        assert _state(view, "#fig-cost .budget", "k") == str(data["config"]["top_k"])
    finally:
        ctx.close()


#: What a phone shows of a figure: every label inside the drawing's own visible box, and no box in
#: the figure wider than itself. An internal scroll is not a fix here — at 390px the balance figure
#: showed four of its eight experts and cut every panel title, with nothing to say more was hidden.
PHONE_JS = """(ids) => {
  const out = [];
  for (const id of ids) {
    const fig = document.getElementById(id);
    for (const svg of fig.querySelectorAll('svg')) {
      // The VISIBLE box: the drawing, cut to whatever scrolling box holds it.
      const own = svg.getBoundingClientRect();
      const clip = (svg.closest('.chart-scroll') || fig).getBoundingClientRect();
      const left = Math.max(own.left, clip.left);
      const right = Math.min(own.right, clip.right);
      for (const t of svg.querySelectorAll('text')) {
        if (!t.textContent || getComputedStyle(t).display === 'none') continue;
        const r = t.getBoundingClientRect();
        if (r.left < left - 1 || r.right > right + 1) {
          out.push(`${id}: "${t.textContent}" ${r.left.toFixed(0)}..${r.right.toFixed(0)} ` +
                   `outside ${left.toFixed(0)}..${right.toFixed(0)}`);
        }
      }
    }
    for (const e of [fig, ...fig.querySelectorAll('*')]) {
      if (e instanceof HTMLElement && e.scrollWidth > e.clientWidth + 1) {
        out.push(`${id}: ${e.tagName.toLowerCase()}.${e.className} scrolls ` +
                 `${e.scrollWidth} > ${e.clientWidth}`);
      }
    }
  }
  return out;
}"""


@pytest.mark.parametrize("width", [390, 320])
def test_the_balance_and_cost_figures_fit_a_phone_whole(site, width):
    """Every label of figures 2 and 4 inside its drawing, and nothing in them scrolling sideways."""
    browser, url = site
    view = browser.new_page(viewport={"width": width, "height": 900})
    try:
        view.goto(url)
        view.wait_for_selector("section#reproduce", timeout=10_000)
        for k in ("1", "8"):  # the cost figure's longest labels appear at either end of k
            view.fill("#fig-cost input#bud-k", k)
            view.dispatch_event("#fig-cost input#bud-k", "input")
            problems = view.evaluate(PHONE_JS, ["fig-balance", "fig-cost"])
            assert not problems, f"at {width}px, k={k}: " + "; ".join(problems)
    finally:
        view.close()
