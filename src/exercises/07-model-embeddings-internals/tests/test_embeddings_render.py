"""The page is tested in a browser, because `node --check` proves almost nothing about it.

A call to an undefined function, a table that renders every cell as `undefined`, a figure reading
`NaN`, and a layout that scrolls sideways on a phone all parse perfectly. `AGENTS.md` names this as
a rule the repo learned by shipping it.

**Served, not opened as a `file://`.** ES modules refuse to load over `file://`, and the shell links
`/_shared/tokens.css` from the site root — so a `file://` test renders a blank, unstyled page and
passes any assertion that only checks the title.
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
SLUG = "07-model-embeddings-internals"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def page():
    """Serve the assembled site and open the exercise page once for the whole module."""
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
            view = browser.new_page(viewport={"width": 1280, "height": 900})
            problems: list[str] = []
            view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            view.goto(f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html")
            view.wait_for_selector("section#reproduce", timeout=10_000)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


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
    """A reader arriving cold must be able to find every part of the story, in a sensible order.

    Checked by `data-role` rather than by heading text, so the prose stays free to change. The order
    is asserted because "limits" before "results" would read as hedging, and "conclusion" before the
    evidence would read as a press release.
    """
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"
    seen = [r for r in roles if r in REQUIRED_ROLES]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(REQUIRED_ROLES), f"the spine is out of order: {first}"


def test_every_figure_has_a_caption_that_says_something(page):
    """A figure with a bare label makes the reader do the interpreting. Captions here state what to
    conclude, so a short one is a caption that has not done its job."""
    caps = page.eval_on_selector_all("figure figcaption", "els => els.map(e => e.innerText)")
    figs = page.evaluate("() => document.querySelectorAll('figure').length")
    assert len(caps) == figs > 0, f"{figs} figures but {len(caps)} captions"
    short = [c[:40] for c in caps if len(c) < 120]
    assert not short, f"these captions are too short to be doing any work: {short}"


def test_the_page_explains_the_solution_and_not_only_the_results(page):
    """The failure this rewrite exists to fix: a page that reports measurements without ever saying
    what the solution is. At least two MECHANISM sections, and the diagram of the tie itself."""
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    assert roles.count("mechanism") >= 2, "mechanism is what makes results comprehensible"
    text = page.eval_on_selector("#solution", "el => el.innerText").lower()
    assert "reuse" in text
    assert page.locator("#solution figure").count() >= 2, "the solution needs its own diagrams"


def test_the_glossary_defines_terms_before_they_are_used(page):
    """The newcomer's route in. Every entry carries a real number from our own run, so the
    definitions are grounded rather than generic."""
    terms = page.eval_on_selector_all("#glossary .gloss dt", "els => els.map(e => e.innerText)")
    defs = page.eval_on_selector_all("#glossary .gloss dd", "els => els.map(e => e.innerText)")
    assert len(terms) >= 6, f"only {len(terms)} terms defined"
    assert len(terms) == len(defs)
    assert all(len(d) > 60 for d in defs), "a definition too short to explain anything"


def test_no_cell_rendered_as_undefined_or_nan(page):
    """The failure mode of a data-driven page: the shape changed and every cell says `undefined`."""
    text = page.eval_on_selector("main", "el => el.innerText")
    for bad in ("undefined", "NaN", "[object Object]", "null"):
        assert bad not in text, f"the page renders the literal {bad!r}"


def test_the_tables_have_data_in_them(page):
    """Guards the opposite failure: the page renders, but every table is empty."""
    counts = page.eval_on_selector_all(
        "main table.grid tbody", "els => els.map(e => e.querySelectorAll('tr').length)"
    )
    assert len(counts) >= 6, f"expected at least six tables, found {len(counts)}"
    assert all(c > 0 for c in counts), f"an empty table body: {counts}"


def test_the_headline_numbers_come_from_the_measurements(page):
    """The page must render the measured figures, not placeholders.

    Pinned deliberately: if `results/measurements.json` changes, this goes red and someone has to
    look at the page rather than discovering the drift after it is published.
    """
    text = page.eval_on_selector("main", "el => el.innerText")
    for expected in ("44,888,832", "768,000,000", "6,291,457", "100.00%", "99.85%", "38,597,376"):
        assert expected in text, f"{expected!r} is missing from the rendered page"


def test_the_limits_are_in_the_open_text(page):
    """`AGENTS.md`: a limitation a reader has to open a drawer to find is one the page is hiding.

    The limits are a first-class section rather than a footer note, so they sit in the rail and a
    reader scanning the page can see that they exist.
    """
    assert page.locator("section[data-role='limits']").count() == 1
    assert page.locator("details").count() == 0, "a limit behind a disclosure is a limit hidden"
    items = page.eval_on_selector_all(
        "section[data-role='limits'] .limitlist li", "els => els.map(e => e.innerText)"
    )
    assert len(items) >= 4, f"only {len(items)} stated limits"
    assert all(len(i) > 60 for i in items), "a limit stated too briefly to be useful"
    rail = page.eval_on_selector_all(".rail-link", "els => els.map(e => e.getAttribute('href'))")
    assert "#limits" in rail, "the limits must be reachable from the rail, not buried"


def test_the_borrowed_credit_is_on_the_page(page):
    """The n-gram term is prior work. The page has to say so where a reader will see it."""
    text = page.eval_on_selector("section[data-role='limits']", "el => el.innerText").lower()
    assert "borrowed" in text


@pytest.mark.parametrize("width", [1280, 900, 390, 320])
def test_the_page_never_scrolls_sideways(page, width):
    """Wide tables scroll inside their own box; the body must not."""
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(250)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"the page scrolls sideways by {overflow}px at {width}px wide"


def test_the_lock_demonstration_shows_measured_numbers(page):
    """The interaction must step through MEASURED logits, not numbers the page invents.

    The first version generated five random values in JavaScript and combined them additively, so
    the alternating sum was zero because of how the demo was written — and the assertion that it
    was zero could never have failed. This reads the shipped samples and checks the page shows
    those, which means it goes red if the page ever starts making numbers up again.
    """
    import json

    measured = json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "measurements.json").read_text(
            encoding="utf-8"
        )
    )["lock"]["samples"]
    assert measured, "no measured lock samples shipped"

    page.set_viewport_size({"width": 1280, "height": 900})
    seen = []
    for _ in range(len(measured)):
        shown = page.eval_on_selector_all(
            "#lock .lockrow:not(.sum) .v", "els => els.map(e => e.textContent)"
        )
        seen.append(tuple(shown[:4]))
        page.click("#lock .btn")

    expected = {tuple(f"{v:.4f}" for v in s["logits"]) for s in measured}
    assert set(seen) <= expected, (
        "the page is showing logit values that are not in results/measurements.json — "
        "it is generating them rather than rendering measurements"
    )
    assert len(set(seen)) == len(measured), (
        f"stepped through {len(set(seen))} distinct samples, expected {len(measured)}"
    )


def test_the_lock_page_states_the_samples_are_measured(page):
    """A reader must be able to tell the demo is evidence and not an illustration, and must be told
    the samples come from an untrained model.

    Asserted by MEANING, not by vocabulary. An earlier version required the word "initialisation";
    the page now says "at its starting values, before any training", which is the same fact in words
    a newcomer can read — and the test, not the page, was what needed to change.
    """
    text = page.eval_on_selector("#lock", "el => el.innerText").lower()
    assert "measured" in text, "the reader cannot tell this is evidence rather than an illustration"
    untrained = ("before any training", "starting values", "initialis", "initializ")
    assert any(k in text for k in untrained), "the page does not say the samples are untrained"


def test_the_left_rail_is_built_and_lists_every_chapter(page):
    """The shared stylesheet styles `.rail` AND reserves 260px of left gutter on `.wrap` at 1180px
    and up — so a page that never builds the rail renders that gutter empty. 05 has had one since
    it shipped; 06 and 07 inherited the CSS and not the element, which is what this guards.
    """
    page.set_viewport_size({"width": 1400, "height": 900})
    page.wait_for_timeout(250)

    titles = page.eval_on_selector_all(".rail-link .rail-t", "els => els.map(e => e.innerText)")
    ids = page.eval_on_selector_all("main section", "els => els.map(e => e.id)")
    assert len(titles) == len(ids), f"{len(titles)} rail links for {len(ids)} sections"
    assert all(t.strip() for t in titles), "a rail entry with no text"

    hrefs = page.eval_on_selector_all(".rail-link", "els => els.map(e => e.getAttribute('href'))")
    assert hrefs == [f"#{i}" for i in ids], "the rail does not point at the sections in order"

    numbers = page.eval_on_selector_all(".rail-link .rail-n", "els => els.map(e => e.innerText)")
    assert numbers == [str(i + 1) for i in range(len(ids))], f"rail numbering is wrong: {numbers}"


def test_the_rail_number_and_title_are_grid_siblings(page):
    """`.rail-link` is a two-column grid. Nesting the number inside the body gives the grid one
    child, which lands in the 16px number column and squeezes every title to one word per line —
    the exact bug 05's builder carries a comment about. Checked by geometry, not by markup.
    """
    page.set_viewport_size({"width": 1400, "height": 900})
    page.wait_for_timeout(250)
    widths = page.eval_on_selector_all(
        ".rail-link .rail-t", "els => els.map(e => e.getBoundingClientRect().width)"
    )
    assert widths, "no rail titles to measure"
    assert all(w > 60 for w in widths), f"a rail title is squeezed into the number column: {widths}"


def test_the_reserved_left_gutter_is_actually_occupied(page):
    """The bug this fixes: `.wrap` pads 260px for the rail whether or not one exists."""
    page.set_viewport_size({"width": 1400, "height": 900})
    page.wait_for_timeout(250)
    pad = page.eval_on_selector(".wrap", "el => parseFloat(getComputedStyle(el).paddingLeft)")
    rail_w = page.eval_on_selector("#rail", "el => el.getBoundingClientRect().width")
    assert pad > 200, f"the shared stylesheet no longer reserves the gutter ({pad}px)"
    assert rail_w > 150, f"the gutter is reserved but the rail does not fill it ({rail_w}px)"


def test_the_first_chapter_is_not_flush_against_the_action_buttons(page):
    """The shared `section` rule has bottom spacing and no top spacing.

    05 and 06 both place a summary panel between the lede actions and their first chapter, so
    neither exposes it. This page goes straight into chapter one, and before `page-extra.css`
    compensated, the first heading sat at a measured **0px** below the buttons.
    """
    page.set_viewport_size({"width": 1440, "height": 940})
    page.wait_for_timeout(250)
    gap = page.evaluate(
        """() => {
            const a = document.querySelector('.lede-actions').getBoundingClientRect();
            const b = document.querySelector('main > section').getBoundingClientRect();
            return Math.round(b.top - a.bottom);
        }"""
    )
    assert gap >= 30, f"the first chapter sits {gap}px below the action buttons"


# ---- the page sweep's two findings -------------------------------------------------------------

#: The six themes. The default — no `data-theme` at all — is the one that failed here, which is
#: why a two-theme check would have missed it entirely.
SWEEP_THEMES = ("", "soft-light", "tinted-dark", "high-contrast", "neon")

#: Below this an SVG label is not readable. Effective size is the authored size times the scale the
#: viewBox renders at, so a label can read 11px in the source and 6.49px on a phone.
LEGIBLE_SVG_TEXT = 9.5

_CONTRAST_JS = """(sel) => {
  const rgb = (s) => {
    const m = (s || '').match(/-?[\\d.]+/g);
    return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : null;
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
    return (x + 0.05) / (y + 0.05);
  };
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    const box = el.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) continue;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || +s.opacity === 0) continue;
    let node = el, bg = s.backgroundColor;
    while (/rgba\\(0, 0, 0, 0\\)/.test(bg) && node.parentElement) {
      node = node.parentElement;
      bg = getComputedStyle(node).backgroundColor;
    }
    const ink = rgb(s.color), ground = rgb(bg);
    if (!ink || !ground) continue;
    const px = parseFloat(s.fontSize), wt = parseInt(s.fontWeight) || 400;
    out.push({
      r: ratio(ink, ground),
      need: (px >= 24 || (px >= 18.66 && wt >= 700)) ? 3.0 : 4.5,
      txt: el.textContent.trim().slice(0, 22),
    });
  }
  return out;
}"""


@pytest.mark.parametrize("theme", SWEEP_THEMES)
@pytest.mark.parametrize("sel", ("table.grid thead th", ".legend code"))
def test_text_on_a_token_surface_clears_aa(page, theme: str, sel: str) -> None:
    """Measured against the ELEMENT'S OWN painted ground, walking up through transparent ancestors.

    `--muted` on `--track` is 4.15:1 in the default light theme — the theme most readers are in,
    and the only one of the six that failed. Every other theme cleared it between 4.84 and 14.17:1,
    which is exactly why this survived: checking two themes would have found nothing.
    """
    try:
        page.evaluate(
            "(t) => t ? document.documentElement.setAttribute('data-theme', t)"
            "         : document.documentElement.removeAttribute('data-theme')",
            theme,
        )
        page.wait_for_timeout(180)
        rows = page.evaluate(_CONTRAST_JS, sel)
        assert rows, f"`{sel}` matched nothing; the selector has rotted"
        bad = sorted((round(r["r"], 2), r["need"], r["txt"]) for r in rows if r["r"] < r["need"])
        assert not bad, (
            f"under {theme or 'the default theme'}, {len(bad)} of {len(rows)} `{sel}` elements are "
            f"below WCAG AA:\n  "
            + "\n  ".join(f"{r}:1 (needs {n}:1)  {t!r}" for r, n, t in bad[:6])
        )
    finally:
        page.evaluate("() => document.documentElement.removeAttribute('data-theme')")


@pytest.mark.parametrize("width", (390, 768, 1440))
def test_no_svg_label_renders_too_small_to_read(page, width: int) -> None:
    """Legibility of a label inside a viewBox is a property of the render, not of the source.

    All 81 labels across the six figures sat between 6.49px and 9.4px at a 390px viewport. The
    authored `font-size` says 11px and tells you nothing, which is why this multiplies by the
    rendered scale.
    """
    try:
        page.set_viewport_size({"width": width, "height": 950})
        page.wait_for_timeout(400)
        rows = page.evaluate(
            """() => {
                 const out = [];
                 for (const svg of document.querySelectorAll('svg')) {
                   const vb = svg.viewBox && svg.viewBox.baseVal;
                   if (!vb || !vb.width) continue;
                   const scale = svg.getBoundingClientRect().width / vb.width;
                   for (const t of svg.querySelectorAll('text')) {
                     if (!t.textContent.trim()) continue;
                     out.push({
                       eff: parseFloat(getComputedStyle(t).fontSize) * scale,
                       txt: t.textContent.trim().slice(0, 24),
                     });
                   }
                 }
                 return out;
               }"""
        )
        assert len(rows) >= 20, f"only {len(rows)} svg labels at {width}px; the selector rotted?"
        tiny = sorted((round(r["eff"], 2), r["txt"]) for r in rows if r["eff"] < LEGIBLE_SVG_TEXT)
        assert not tiny, (
            f"at {width}px, {len(tiny)} of {len(rows)} svg labels render under "
            f"{LEGIBLE_SVG_TEXT}px:\n  " + "\n  ".join(f"{e}px  {t!r}" for e, t in tiny[:6])
        )
    finally:
        # A viewport left behind would silently change every test that runs after this one.
        page.set_viewport_size({"width": 1280, "height": 900})
        page.wait_for_timeout(150)
