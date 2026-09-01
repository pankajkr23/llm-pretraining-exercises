"""The page as a reader sees it, in a real browser.

`node --check` proves a file has no syntax error and nothing more. A call to an undefined function,
a figure whose stages all draw the same thing, a grid of invisible chips, and a plate printing five
labels on top of each other all parse perfectly and log nothing.

**Every assertion below that names a defect is one this page actually shipped**, found by looking
at a screenshot while the whole suite was green. They are written down here so the next rebuild
cannot reintroduce them:

- ``test_the_verdict_grid_is_not_a_grid_of_invisible_chips`` — the grid was handed ``glyph()``,
  which returns an SVG ``<g>``; appended into an HTML ``<div>`` it renders nothing at all. Six
  windows of frames and stamps over twenty-three chips that were in the DOM and not on the screen.
- ``test_the_invoice_cut_line_is_visible`` — the cut line starts at ``opacity: 0`` and is revealed
  by an ``IntersectionObserver`` registered while the node was still detached, which never fires.
  The plate's entire argument was invisible.
- ``test_no_two_plate_labels_overlap`` — laddering used a fixed 48px separation for labels up to
  200px wide.
- ``test_no_glyph_draws_outside_its_own_box`` — a mark at a negative ``y`` rendered on top of the
  caption of the glyph in the row above, because SVG does not clip by default. Present, legible,
  and attributed to the wrong mechanism.
- ``test_the_page_shows_no_shell_commands`` — the page used to print ``uv sync`` and ``pytest``.
  Commands belong in the README.

Playwright is integration-marked and skips without a browser, so a fresh checkout still works. One
time setup: `uv run playwright install chromium`.
"""

import functools
import http.server
import importlib.util
import json
import re
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
def page():
    """The built page, served over HTTP and rendered.

    **Served, not opened as a file.** ES modules refuse to load over `file://` and the shell links
    `/_shared/tokens.css` from the site root, so a `file://` test renders a blank page with CORS
    errors and passes any assertion that only checks the title.
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
            except Exception as exc:  # no browser installed, or a sandbox blocking it
                pytest.skip(f"chromium unavailable: {exc}")
            view = browser.new_page(viewport={"width": 1400, "height": 950})
            problems: list[str] = []
            view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            view.goto(f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html")
            view.wait_for_selector("section#reproduce", timeout=15_000)
            # Every plate animates on first view; settle before measuring anything.
            view.wait_for_timeout(1500)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


# ---- the page renders at all --------------------------------------------------------------------


def test_the_page_renders_without_a_single_console_error(page) -> None:
    """A page that throws on load still shows its title, which is why this is asserted."""
    assert not page.console_problems, page.console_problems


def test_no_undefined_or_nan_reaches_the_reader(page) -> None:
    body = page.inner_text("body")
    for bad in ("undefined", "NaN", "[object Object]"):
        assert bad not in body, f"{bad!r} is visible on the page"


def test_no_markup_reaches_the_reader_as_literal_text(page) -> None:
    """`rich()` escapes first and marks up second, so a stray marker shows rather than a tag.

    The previous helper did not parse HTML at all, so `<b>H1</b>` written in a chapter reached the
    reader as five literal characters. The existing markup guard looked for `[[` and backticks and
    could see neither failure.
    """
    body = page.inner_text("main")
    for bad in ("<b>", "</b>", "<i>", "**", "[[", "&amp;", "&lt;"):
        assert bad not in body, f"{bad!r} is rendered as literal text"


def test_the_page_has_the_required_spine_in_order(page) -> None:
    """Presence is checked lexically by the repo-wide guard; only this can see DOM order.

    Order is asserted because `limits` before `results` reads as hedging, and `conclusion` before
    the evidence reads as a press release.
    """
    spine = _required_spine()
    roles = page.eval_on_selector_all("#main > section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in spine if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"

    seen = [r for r in roles if r in spine]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(spine), f"the spine is out of order: {first}"


def test_the_page_shows_no_shell_commands(page) -> None:
    """Commands live in the README. A page that opens with `uv sync` is written for its author."""
    body = page.inner_text("main")
    for bad in ("uv sync", "uv run", "pytest", "pip install", "npm "):
        assert bad not in body, f"the page prints a shell command: {bad!r}"


# ---- the twenty-three are all there, three ways -------------------------------------------------


def test_every_catalogued_mechanism_appears_on_the_page(page) -> None:
    """The assignment's score-zero clause, checked against what actually rendered."""
    bundle = _bundle()
    text = page.inner_text("main")
    missing = [m["name"] for m in bundle["mechanisms"] if m["name"] not in text]
    assert not missing, f"catalogued but not rendered: {missing}"


def test_the_index_plate_shows_all_of_them_without_any_clicking(page) -> None:
    """A grader must not have to click twenty-three times to see twenty-three mechanisms."""
    bundle = _bundle()
    rows = page.eval_on_selector_all(
        "#reproduce .ix-row", "els => els.map(e => e.id.replace('m-',''))"
    )
    assert rows == [m["key"] for m in bundle["mechanisms"]], (
        "the index plate is not the catalogue in date order"
    )


def test_the_index_plate_is_in_date_order_on_screen(page) -> None:
    """The assignment's central requirement, asserted on the rendered order rather than the data.

    The catalogue being sorted proves nothing about the page: a template that iterated a dictionary
    or reversed a list would still pass every catalogue test.
    """
    shown = page.eval_on_selector_all(
        "#reproduce .ix-date", "els => els.map(e => e.textContent.trim())"
    )
    assert shown == sorted(shown), "the index plate is not in date order on screen"
    assert len(shown) == len(_bundle()["mechanisms"])


def test_every_index_row_states_what_it_costs_and_not_only_what_it_buys(page) -> None:
    """The assignment: a technique written down with only pros has not been understood yet."""
    rows = page.eval_on_selector_all(
        "#reproduce .ix-row",
        "els => els.map(e => [e.id, (e.querySelector('.ix-ledger .c')||{}).textContent || '',"
        " (e.querySelector('.ix-ledger .d')||{}).textContent || ''])",
    )
    assert rows, "no index rows rendered"
    thin = [rid for rid, credit, debit in rows if len(credit) < 20 or len(debit) < 20]
    assert not thin, f"these rows do not state both a credit and a debit: {thin}"


def test_every_index_row_links_the_source_its_date_came_from(page) -> None:
    """A date a reader cannot check is the failure the whole exercise is built to avoid."""
    rows = page.eval_on_selector_all(
        "#reproduce .ix-row",
        "els => els.map(e => [e.id, !!e.querySelector('.ix-src a[href^=\"http\"]'),"
        " (e.querySelector('.ix-src .q')||{}).textContent || ''])",
    )
    unlinked = [rid for rid, has_link, _ in rows if not has_link]
    unquoted = [rid for rid, _, quoted in rows if len(quoted) < 10]
    assert not unlinked, f"these rows cite no source URL: {unlinked}"
    assert not unquoted, f"these rows quote no date string from their source: {unquoted}"


def test_the_dates_on_screen_are_the_dates_in_the_catalogue(page) -> None:
    """The page must not reformat a date into a different one."""
    bundle = _bundle()
    shown = page.eval_on_selector_all(
        "#reproduce .ix-date", "els => els.map(e => e.textContent.trim())"
    )
    assert shown == [m["date"] for m in bundle["mechanisms"]]


# ---- PLATE III: the chronology ------------------------------------------------------------------


def test_the_plate_places_every_mechanism(page) -> None:
    keys = page.eval_on_selector_all(
        "#results svg.plate-wide .plate-entry", "els => els.map(e => e.dataset.key)"
    )
    assert sorted(keys) == sorted(m["key"] for m in _bundle()["mechanisms"])


def test_the_plate_puts_time_on_the_x_axis_and_does_not_lie_about_it(page) -> None:
    """Entries must be ordered left to right by date, because the axis is the whole argument.

    A plate that grouped by bill and then laid entries out evenly would look almost identical and
    would destroy the finding — the gaps are the point.
    """
    bundle = _bundle()
    order = {m["key"]: i for i, m in enumerate(bundle["mechanisms"])}
    placed = page.eval_on_selector_all(
        "#results svg.plate-wide .plate-entry",
        "els => els.map(e => [e.dataset.key, e.querySelector('circle').getBoundingClientRect().x])",
    )
    by_date = sorted(placed, key=lambda kv: order[kv[0]])
    xs = [x for _, x in by_date]
    assert xs == sorted(xs), "plate entries are not left-to-right in date order"


def test_no_two_plate_labels_overlap(page) -> None:
    """The defect that made five staves unreadable.

    An earlier version laddered labels to a fixed 48px minimum separation while a label is up to
    200px wide, so names printed on top of each other — "SPARSE (FACTORISED) ATTENTREFORMER".
    Nothing failed, because every label was present and correctly positioned relative to its tick.
    """
    boxes = page.eval_on_selector_all(
        "#results svg.plate-wide .plate-entry text.pe-name",
        "els => els.map(e => { const r = e.getBoundingClientRect();"
        " return [e.textContent, r.x, r.y, r.width, r.height]; })",
    )
    assert len(boxes) >= 20, f"only {len(boxes)} plate labels rendered"
    clashes = []
    for i, (ta, xa, ya, wa, ha) in enumerate(boxes):
        for tb, xb, yb, wb, hb in boxes[i + 1 :]:
            if xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha:
                clashes.append(f"{ta!r} over {tb!r}")
    assert not clashes, f"plate labels overlap: {clashes}"


def test_the_quiet_stretch_is_drawn_as_area_and_labelled_with_its_own_number(page) -> None:
    """The 680-day silence is a finding, so it is a shape on the plate and not a sentence."""
    bundle = _bundle()
    days = bundle["quietStretch"]["days"]
    label = page.eval_on_selector("#results svg.plate-wide .quiet-lab", "e => e.textContent")
    assert str(days) in label, f"the quiet band is labelled {label!r}, not with {days}"
    width = page.eval_on_selector("#results svg.plate-wide .quiet-band", "e => e.getBBox().width")
    assert width > 40, "the quiet stretch is drawn too small to read as a gap"


def test_clicking_a_plate_entry_retypesets_the_reading_spread(page) -> None:
    """The spread is what replaced twenty-three collapsed cards. If it does not change, it is a
    static card wearing an interaction."""
    before = page.inner_text(".spread")
    page.eval_on_selector(
        '#results svg.plate-wide .plate-entry[data-key="mamba"]',
        "e => e.dispatchEvent(new MouseEvent('click', {bubbles: true}))",
    )
    page.wait_for_timeout(400)
    after = page.inner_text(".spread")
    assert after != before, "the reading spread did not change when a plate entry was clicked"
    assert "Mamba" in after, "the spread did not typeset the mechanism that was clicked"


# ---- PLATE II: the centrefold -------------------------------------------------------------------


def test_the_centrefold_runs_all_five_stages_including_the_weighted_sum(page) -> None:
    """The assignment names five steps and an earlier version of this figure had four.

    Stopping at softmax is the one place a reader concludes attention outputs weights. It outputs a
    vector, and the fifth stage is where that happens.
    """
    labels = page.eval_on_selector_all(
        "#mechanism .tabs button", "els => els.map(e => e.textContent.trim())"
    )
    assert len(labels) == 5, f"the centrefold has {len(labels)} stages, not five: {labels}"
    assert any("V" in t for t in labels), f"no weighted-sum stage: {labels}"


def test_changing_the_stage_changes_what_is_drawn(page) -> None:
    """**The test the figure rests on.** A stepper that changes a caption and not the picture is
    decoration, and would look completely normal.

    The first four stages are distinguished by the score grid's shading. The fifth is not, and that
    is correct rather than a bug: the weights are already final after softmax, and what x V adds is
    the output. So it is checked on the geometry it actually changes. An earlier version of this
    test compared opacity alone and reported four distinct states for five stages, which read as a
    broken figure when it was a badly aimed assertion.
    """
    shade = "els => els.map(e => e.getAttribute('opacity')).join(',')"
    width = "els => Math.round(els.reduce((a,e)=>a+e.getBoundingClientRect().width,0))"
    grids, widths = [], []
    for i in range(5):
        page.eval_on_selector_all("#mechanism .tabs button", f"els => els[{i}].click()")
        page.wait_for_timeout(800)
        grids.append(page.eval_on_selector_all("#mechanism svg rect.f-ink", shade))
        widths.append(page.eval_on_selector_all("#mechanism svg rect.f-ink", width))
    assert len(set(grids[:4])) == 4, "two of the first four stages shade the grid identically"
    assert widths[4] > widths[3], "the weighted-sum stage drew nothing the softmax stage did not"


def test_the_masked_stage_leaves_exactly_the_causal_triangle(page) -> None:
    """Six tokens make 36 scores and use 21. That arithmetic is the reason every later glyph is a
    triangle, so it is asserted rather than merely asserted in prose."""
    page.eval_on_selector_all("#mechanism .tabs button", "els => els[2].click()")
    page.wait_for_timeout(750)
    nums = page.eval_on_selector_all(
        "#mechanism svg text.num", "els => els.map(e => e.textContent)"
    )
    grid = nums[:36]
    filled = [t for t in grid if t and t.strip()]
    assert len(grid) == 36, f"the score grid has {len(grid)} cells, not 36"
    assert len(filled) == 21, f"the mask left {len(filled)} live cells, not 21"


def test_the_weighted_sum_stage_actually_draws_an_output(page) -> None:
    """Bay five must produce visible bars. A fifth tab that changes only the caption is the exact
    failure this figure was rebuilt to fix."""
    page.eval_on_selector_all("#mechanism .tabs button", "els => els[3].click()")
    page.wait_for_timeout(750)
    total = "els => els.reduce((a,e)=>a+e.getBoundingClientRect().width,0)"
    before = page.eval_on_selector_all("#mechanism svg rect.f-ink", total)
    page.eval_on_selector_all("#mechanism .tabs button", "els => els[4].click()")
    page.wait_for_timeout(950)
    after = page.eval_on_selector_all("#mechanism svg rect.f-ink", total)
    assert after > before + 50, "the weighted-sum stage drew no output bars"


# ---- the plates that were invisible -------------------------------------------------------------


def test_the_verdict_grid_is_not_a_grid_of_invisible_chips(page) -> None:
    """The defect: the grid was handed `glyph()`, which returns an SVG `<g>`.

    Appended into an HTML `<div>` a bare `<g>` renders nothing. The grid drew six windows of frames
    and TIE stamps over twenty-three chips that were all present in the DOM and none of them on the
    screen, and no test failed because every element the guard counted existed.
    """
    chips = page.eval_on_selector_all(
        "#conclusion .verdict .cell svg",
        "els => els.filter(e => e.getBoundingClientRect().width > 4).length",
    )
    total = len(_bundle()["mechanisms"])
    assert chips == total, f"{chips} of {total} verdict chips have a rendered size"


def test_the_verdict_stamps_a_tie_on_exactly_the_windows_that_tied(page) -> None:
    """`Period.dominant` returns None on a tie instead of picking a winner, and so does the page."""
    bundle = _bundle()
    tied = [p for p in bundle["periods"] if not p["dominant"]]
    assert tied, "no tied window in the data — this test would then be vacuous"
    stamps = page.eval_on_selector_all("#conclusion .verdict .tie", "els => els.length")
    assert stamps == len(tied), f"{stamps} TIE stamps for {len(tied)} tied windows"


def test_the_invoice_cut_line_is_visible(page) -> None:
    """The defect: it starts at `opacity: 0` and is revealed by an observer that never fired.

    Every figure asks for `onFirstView` before `chapters.js` appends it, so the node was detached;
    an IntersectionObserver on a detached node never fires. The plate's whole argument — the row
    where one accelerator is exhausted — was invisible, with a clean console.
    """
    # Measured WITHOUT scrolling to it first. The original version of this test scrolled the row
    # into view and then checked its opacity, which is why it passed against a cut line that was
    # invisible to every reader who had not scrolled — a screenshot, a print, an anchor landing.
    # A guard that triggers the behaviour it is testing for is not a guard.
    opacity = page.eval_on_selector("#problem .inv-cut", "e => getComputedStyle(e).opacity")
    assert float(opacity) > 0.9, f"the invoice cut line is at opacity {opacity} before any scroll"
    assert page.inner_text("#problem .inv-cut").strip(), "the cut line carries no label"
    rule = page.eval_on_selector(
        "#problem .inv-cut", "e => getComputedStyle(e, '::after').borderTopStyle"
    )
    assert rule == "dashed", f"the cut line draws no dashed rule (border-top-style: {rule})"


def test_the_invoice_prices_every_context_the_data_carries(page) -> None:
    bundle = _bundle()
    text = page.inner_text("#problem")
    for row in bundle["cache"]["contexts"]:
        shown = f"{row['oneUser'] / 1e9:.2f} GB"
        assert shown in text, f"the invoice does not print {shown}"


def test_the_race_ends_at_the_crossings_the_arithmetic_gives(page) -> None:
    """The figure and the invoice must not be able to disagree: both read `tokensBeforeWall`."""
    bundle = _bundle()
    # Scoped to the race's own figure. An unscoped "#results .runbtn" now finds the plate's sweep
    # control, which is a different button doing a different thing — and the test would have gone
    # green while clicking the wrong control for as long as the race numbers stayed on screen.
    page.eval_on_selector("#results .well .runbtn", "e => { e.scrollIntoView(); e.click(); }")
    page.wait_for_timeout(5200)
    text = page.inner_text("#results")
    for row in bundle["cache"]["sharing"]:
        shown = f"{row['tokensBeforeWall']:,}"
        assert shown in text, f"{row['name']} never reports its crossing at {shown}"


# ---- the glyph alphabet -------------------------------------------------------------------------


def test_no_glyph_draws_outside_its_own_box(page) -> None:
    """The defect: a mark at a negative `y` landed on the caption of the glyph in the row above.

    SVG does not clip by default, so an escaping mark renders — present, legible, and attributed to
    the wrong mechanism. Nothing failed; the element was in the DOM and correctly drawn.
    """
    escapes = page.evaluate(
        """() => {
      const bad = [];
      for (const s of document.querySelectorAll('svg.glyph-svg')) {
        const vb = s.viewBox.baseVal;
        const b = s.getBBox();
        if (b.x < vb.x - 0.5 || b.y < vb.y - 0.5 ||
            b.x + b.width > vb.x + vb.width + 0.5 ||
            b.y + b.height > vb.y + vb.height + 0.5) {
          bad.push([s.getAttribute('aria-label') || '?',
                    [b.x, b.y, b.width, b.height].map(n => +n.toFixed(1)).join(','),
                    [vb.x, vb.y, vb.width, vb.height].join(',')]);
        }
      }
      return bad;
    }"""
    )
    assert not escapes, f"these glyphs draw outside their viewBox: {escapes}"


def test_the_key_draws_one_exemplar_per_glyph_family_that_the_plate_uses(page) -> None:
    """The key is generated by the same functions as the plate, so it cannot describe a shape the
    plate does not draw."""
    bundle = _bundle()
    shown = page.eval_on_selector_all(
        "#glossary .key-alpha .it .lab", "els => els.map(e => e.textContent.split(' ')[0])"
    )
    assert sorted(s.lower() for s in shown) == sorted(bundle["counts"]["glyphKinds"])


# ---- the shell ----------------------------------------------------------------------------------


def test_the_left_rail_is_built_and_fills_the_gutter_it_reserves(page) -> None:
    """`_shared/page.css` reserves 260px at >=1180px whether or not the page builds a rail.

    Exercises 06 and 07 shipped an empty 260px gutter for months by copying the stylesheet and not
    the markup it assumes, and nothing failed.
    """
    page.set_viewport_size({"width": 1400, "height": 950})
    page.wait_for_timeout(300)
    padding = page.eval_on_selector(".wrap", "e => parseFloat(getComputedStyle(e).paddingLeft)")
    rail = page.eval_on_selector("#rail", "e => e.getBoundingClientRect().width")
    if padding > 100:
        assert rail > padding * 0.5, f"a {padding}px gutter is reserved and the rail fills {rail}px"


def test_the_rail_lists_every_section_in_order(page) -> None:
    links = page.eval_on_selector_all("#rail .rail-link", "els => els.map(e => e.hash.slice(1))")
    roles = page.eval_on_selector_all("#main > section", "els => els.map(e => e.id)")
    assert links == roles, "the contents list does not match the sections, in order"


@pytest.mark.parametrize("width", [1440, 1180, 900, 620, 390, 320])
def test_the_page_never_scrolls_sideways(page, width: int) -> None:
    """A full-bleed plate built on `100vw` scrolls sideways on every browser that reserves a
    scrollbar. This page uses named grid lines instead, and this is what proves it."""
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(400)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"the page scrolls sideways by {overflow}px at {width}px wide"


def test_every_spine_section_that_argues_from_a_figure_has_one(page) -> None:
    """A page with only results can be believed but not understood.

    `svg, .invoice` and not just `svg`: PLATE I is deliberately set as a printed invoice rather
    than drawn as a chart, because the data literally is a bill and typography states it better
    than two curves would.
    """
    for role in ("problem", "mechanism", "results", "conclusion", "negatives"):
        figures = page.eval_on_selector_all(
            f'section[data-role="{role}"] svg, section[data-role="{role}"] .invoice',
            "els => els.length",
        )
        assert figures >= 1, f"the {role} section argues from a figure it does not have"


def test_exactly_one_plate_is_visible_at_each_width(page) -> None:
    """Two plates are built; a reader must ever see one.

    They are separate SVGs rather than one responsive drawing, so the only thing stopping both from
    rendering is a CSS rule, and the first version of it lost: `.plate svg { display: block }`
    is (0,1,1) and out-specified a bare `.plate-tall` at (0,1,0), so the phone got the unreadable
    landscape smear stacked on top of the portrait plate built to spare it.
    """
    for width, expect_wide in ((1400, True), (900, True), (720, False), (390, False), (320, False)):
        page.set_viewport_size({"width": width, "height": 900})
        page.wait_for_timeout(250)
        seen = page.evaluate(
            """() => {
              const vis = (sel) => {
                const e = document.querySelector(sel);
                return !!e && getComputedStyle(e).display !== 'none';
              };
              return [vis('svg.plate-wide'), vis('svg.plate-tall')];
            }"""
        )
        assert seen == [expect_wide, not expect_wide], (
            f"at {width}px the visible plates are wide={seen[0]} tall={seen[1]}"
        )
    page.set_viewport_size({"width": 1400, "height": 950})
    page.wait_for_timeout(250)


def test_both_plates_carry_every_mechanism(page) -> None:
    """The portrait plate drops the NAMES on purpose. It must not drop an entry."""
    keys = _bundle()["mechanisms"]
    for sel in ("svg.plate-wide", "svg.plate-tall"):
        got = page.eval_on_selector_all(
            f"#results {sel} .plate-entry", "els => els.map(e => e.dataset.key)"
        )
        assert sorted(got) == sorted(m["key"] for m in keys), f"{sel} is missing entries"


def test_the_sweep_reads_the_plate_in_date_order(page) -> None:
    """The one motion on this page that teaches something static arrangement cannot.

    A sweep that lit entries in DOM order rather than date order would look completely normal and
    would destroy the only thing it exists to show — that the field raced through some years and
    stalled through others.
    """
    lit = page.evaluate(
        """() => {
          const svg = document.querySelector('#results svg.plate-wide');
          const seen = [];
          for (let i = 0; i <= 600; i++) {
            const key = svg.sweep(i / 600);
            if (key && key !== seen[seen.length - 1]) seen.push(key);
          }
          svg.sweepOff();
          return seen;
        }"""
    )
    mechanisms = _bundle()["mechanisms"]
    order = [m["key"] for m in mechanisms]
    assert lit == [k for k in order if k in lit], "the sweep lights entries out of date order"

    # The playhead reports the LAST entry it has passed, so two mechanisms sharing a date share a
    # position and only the later one is ever reported. The reachable count is therefore the number
    # of distinct dates, not the number of mechanisms — derived here rather than guessed, because a
    # guessed threshold is what made this assertion fail against a correct sweep.
    reachable = len({m["date"] for m in mechanisms})
    assert len(lit) == reachable, f"the sweep reached {len(lit)} of {reachable} distinct dates"


def test_the_plate_offers_a_sweep_control(page) -> None:
    """The control exists and belongs to the plate rather than to a well's figure.

    Its label is not asserted: it toggles to "Stop" while a sweep is running, and an earlier
    version of this test asserted the idle label and failed whenever another test in the module had
    left a sweep in flight. A guard that depends on test ordering is a flaky guard.
    """
    present = page.eval_on_selector_all("#results .sweep-ctl .runbtn", "els => els.length")
    assert present == 1, f"expected exactly one sweep control on the plate, found {present}"


def test_the_sweep_is_withheld_entirely_under_reduced_motion(page) -> None:
    """A sweep has no terminal state, so it is withheld rather than degraded.

    Every other figure here renders directly into a readable still. This one cannot: its whole
    content is an ordering over time. Offering a control that would do nothing is worse than not
    offering it, so the reduced-motion page must not build one — checked in a real reduced-motion
    context rather than by reading the source.
    """
    ctx = page.context.browser.new_context(reduced_motion="reduce")
    quiet = ctx.new_page()
    try:
        quiet.goto(page.url)
        quiet.wait_for_selector("section#reproduce", timeout=15_000)
        quiet.wait_for_timeout(800)
        assert (
            quiet.eval_on_selector_all("#results .sweep-ctl .runbtn", "els => els.length") == 0
        ), "a sweep control was built for a reader who asked for no motion"
        # And the rest of the plate must still be there: withholding the motion must not withhold
        # the evidence.
        entries = quiet.eval_on_selector_all(
            "#results svg.plate-wide .plate-entry", "els => els.length"
        )
        assert entries == len(_bundle()["mechanisms"])
    finally:
        quiet.close()
        ctx.close()


def test_the_page_states_its_own_size_correctly(page) -> None:
    """Whatever else it says, the page must spell its own catalogue size."""
    words = [
        "twenty",
        "twenty-one",
        "twenty-two",
        "twenty-three",
        "twenty-four",
        "twenty-five",
        "twenty-six",
        "twenty-seven",
        "twenty-eight",
        "twenty-nine",
        "thirty",
        "thirty-one",
        "thirty-two",
        "thirty-three",
        "thirty-four",
    ]
    total = len(_bundle()["mechanisms"])
    assert 20 <= total < 20 + len(words), f"extend the word list for a catalogue of {total}"
    correct = words[total - 20]
    body = page.inner_text("main").lower()
    assert re.search(rf"\b{re.escape(correct)}\b", body), (
        f"the page never states its own size ({correct})"
    )


def test_clicking_the_sweep_control_actually_sweeps(page) -> None:
    """The WIRING, not the mechanism. This is the test that was missing.

    `test_the_sweep_reads_the_plate_in_date_order` calls `sweep()` on the SVG directly, so it
    exercised the drawing and never the button. The wrapper that holds both plates forwarded
    `select` and not `sweep`, so a real click threw "p.sweep is not a function" inside the
    animation frame, the loop died on frame one, and the label sat on "Stop" forever — with the
    mechanism test green the whole time.
    """
    page.eval_on_selector("#results .sweep-ctl .runbtn", "e => e.scrollIntoView()")
    page.wait_for_timeout(200)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    label0 = page.eval_on_selector("#results .sweep-ctl .runbtn", "e => e.textContent.trim()")
    page.eval_on_selector("#results .sweep-ctl .runbtn", "e => e.click()")
    page.wait_for_timeout(900)

    assert not errors, f"the sweep threw: {errors}"
    running = page.eval_on_selector("#results .sweep-ctl .runbtn", "e => e.textContent.trim()")
    assert running != label0, "the control did not change state when clicked"

    # The playhead must be somewhere other than its parked position, and entries must be dimmed.
    moved = page.eval_on_selector(
        "#results svg.plate-wide line.s-accent", "e => e.getAttribute('opacity')"
    )
    assert moved == "1", "the playhead is not visible during a sweep"
    dimmed = page.eval_on_selector_all(
        "#results svg.plate-wide .plate-entry.dim", "els => els.length"
    )
    assert dimmed > 0, "nothing ahead of the playhead is dimmed, so the sweep is not running"

    # And clicking again stops it and puts the plate back.
    page.eval_on_selector("#results .sweep-ctl .runbtn", "e => e.click()")
    page.wait_for_timeout(300)
    assert (
        page.eval_on_selector("#results .sweep-ctl .runbtn", "e => e.textContent.trim()") == label0
    ), "the control did not return to its idle label after being stopped"
    assert (
        page.eval_on_selector_all("#results svg.plate-wide .plate-entry.dim", "els => els.length")
        == 0
    ), "stopping the sweep left the plate dimmed"


def test_the_sweep_control_does_not_collide_with_the_reading_spread(page) -> None:
    """It sat on the spread's 2px top rule and the line ran through the button."""
    box = page.eval_on_selector(
        "#results .sweep-ctl .runbtn",
        "e => { const r = e.getBoundingClientRect(); return [r.top, r.bottom]; }",
    )
    spread = page.eval_on_selector(".spread", "e => e.getBoundingClientRect().top")
    assert box[1] < spread - 4, (
        f"the sweep control (bottom {box[1]:.0f}) overlaps the reading spread (top {spread:.0f})"
    )


def test_the_diagram_is_present_the_instant_a_mechanism_is_chosen(page) -> None:
    """No waiting. A figure that needs a delay to appear is absent to anything that captures.

    The figure build is deferred by 220ms so a running sweep does not rebuild a 720-unit drawing
    thirty times. The first version applied that to every selection, including a reader's own
    click — invisible to a person, because the eye has not arrived yet, and total to a save, a
    print, a PDF or a screenshot tool. A page save taken just after a click came back with no
    diagram in it at all, which is how this was found.

    Same shape as the invoice cut line: content that exists only if you wait.
    """
    page.eval_on_selector(
        '#results .plate-entry[data-key="alibi"]',
        "e => e.dispatchEvent(new MouseEvent('click', {bubbles: true}))",
    )
    # deliberately no wait_for_timeout
    present = page.eval_on_selector_all(".spread .diagram-svg", "els => els.length")
    assert present == 1, (
        f"{present} diagrams in the spread immediately after a click — a capture taken now would "
        f"record the page without its figure"
    )
    shown = page.eval_on_selector(".spread .sp-name", "e => e.textContent")
    assert "ALiBi" in shown, f"the spread shows {shown!r} rather than the mechanism clicked"


def test_the_spread_never_stacks_more_than_one_diagram(page) -> None:
    """`render()` wipes its two columns by hand, so the figure needs its own wipe."""
    for key in ("mamba", "gqa", "yarn", "nsa"):
        page.eval_on_selector(
            f'#results .plate-entry[data-key="{key}"]',
            "e => e.dispatchEvent(new MouseEvent('click', {bubbles: true}))",
        )
        page.wait_for_timeout(120)
    count = page.eval_on_selector_all(".spread .diagram-svg", "els => els.length")
    assert count == 1, f"after four selections the spread holds {count} diagrams"


@pytest.mark.parametrize("width", [2560, 1920, 1600, 1500, 1400, 1180])
def test_the_rail_sits_against_the_text_it_indexes(page, width: int) -> None:
    """The gutter and the rail must be in the SAME place, not merely both present.

    `.wrap` is centred at `max-width: 1500px` and reserves 260px of left padding for the rail;
    the rail is `position: fixed`, so it was pinned to the window. Below 1500px those coincide and
    everything looked right. Past it they separate — the reserved gutter drifts right with the
    centred wrap while the rail stays welded to the far edge — so the page grows a widening void
    between the rail and the text it indexes, and an empty gutter indexing nothing. At 2560px the
    void was 554px. Nothing failed: exercise 07's guard asks whether the gutter is *filled*, which
    it was, by an element 554px away from it.

    A fixed offset is the wrong assertion here; the invariant is the relationship.
    """
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(160)
    m = page.evaluate(
        """() => {
          const rail = document.querySelector('.rail');
          const wrap = document.querySelector('.wrap');
          const rb = rail.getBoundingClientRect();
          const cs = getComputedStyle(wrap);
          return {
            fixed: getComputedStyle(rail).position === 'fixed',
            railRight: rb.right,
            textLeft: wrap.getBoundingClientRect().left + parseFloat(cs.paddingLeft),
          };
        }"""
    )
    assert m["fixed"], f"at {width}px the rail is not pinned — this guard is measuring nothing"
    gap = m["textLeft"] - m["railRight"]
    assert 0 <= gap <= 60, (
        f"at {width}px the rail ends at {m['railRight']:.0f} and the text starts at "
        f"{m['textLeft']:.0f} — a {gap:.0f}px gap between the rail and what it indexes"
    )


def test_the_link_to_the_field_guide_is_a_designed_control(page) -> None:
    """An unclassed `<a>` takes the generic link colour, which in dark mode is raw accent blue on
    near-black. It clears the contrast floor and still reads as broken, because it is the only
    untreated element on a page where every other control is a designed object — so it looks like
    something that failed to load rather than something to click.

    `.jump` puts `--on-accent` ON the accent, the pairing the token set is built around. The shared
    stylesheet has carried it all along and this exercise had never used it.
    """
    link = page.query_selector("a.jump[href='field-guide/']")
    assert link is not None, "the field-guide link is not a .jump — it will render as a raw anchor"
    paint = page.evaluate(
        """(el) => {
          const cs = getComputedStyle(el);
          return {bg: cs.backgroundColor, fg: cs.color, deco: cs.textDecorationLine};
        }""",
        link,
    )
    assert paint["bg"] not in ("rgba(0, 0, 0, 0)", "transparent"), (
        f"the pill has no ground of its own: {paint}"
    )
    assert paint["deco"] == "none", f"a pill should not also be underlined: {paint}"
