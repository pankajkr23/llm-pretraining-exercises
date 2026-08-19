"""The page, tested in a browser rather than parsed.

`node --check` proves a file has no syntax error and nothing more. A call to an undefined function,
a slider wired to nothing, a colour token that does not exist, and a headline reading `0` all parse
perfectly. This suite loads the built site in Chromium and asserts what a reader sees.

**It serves `public/`, not `web/`.** The palette lives in the *site-root* `/_shared/tokens.css`,
which only exists once `deploy/vercel/build.sh` has assembled the site — serving `web/` directly
would test a page with no colours and no fonts, which is what exercise 04 did for a while without
noticing. The fixture builds the site if it is missing.

Integration-marked, and skips cleanly when Chromium is absent so a fresh checkout still works. That
means it protects the page silently or not at all: `uv run playwright install chromium` once.
"""

import http.server
import json
import os
import re
import socket
import subprocess
import threading

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

from mixture import export  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = export.EXERCISE_ROOT.parents[2]
PUBLIC = REPO / "public"
SLUG = "05-datamixtures-and-curriculum"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def site() -> str:
    """Serve the assembled site and yield its base URL.

    Assembles it first when absent: `public/` is gitignored, so a fresh checkout and CI both start
    without one, and a suite that skipped in that case would protect nothing on the machine that
    matters most.
    """
    if not (PUBLIC / SLUG / "index.html").exists():
        script = REPO / "deploy" / "vercel" / "build.sh"
        if not script.exists():
            pytest.skip("no build script; cannot assemble the site under test")
        subprocess.run(["bash", str(script)], check=True, capture_output=True)
    if not (PUBLIC / "_shared" / "tokens.css").exists():
        pytest.skip("the assembled site has no root stylesheet; the page under test would be bare")

    handler = type(
        "Handler",
        (http.server.SimpleHTTPRequestHandler,),
        {
            "__init__": lambda self, *a, **kw: http.server.SimpleHTTPRequestHandler.__init__(
                self, *a, directory=str(PUBLIC), **kw
            ),
            "log_message": lambda *a, **kw: None,
        },
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/{SLUG}/"
    finally:
        server.shutdown()


@pytest.fixture(scope="module")
def page(site):
    """A loaded page, with any console error recorded."""
    with sync_playwright() as play:
        try:
            browser = play.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            # Skipping keeps a fresh checkout working. On CI it would turn "the browser
            # never launched" into a green run with no rendering coverage at all, which is
            # what this suite exists to prevent. CI has no excuse for a missing browser.
            if os.environ.get("CI"):
                pytest.fail(f"chromium did not launch on CI: {exc}")
            pytest.skip(f"no chromium available: {exc}")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        p = context.new_page()
        errors: list[str] = []
        p.on("pageerror", lambda e: errors.append(str(e)))
        p.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        p.goto(site, wait_until="networkidle")
        p.wait_for_selector("section", timeout=10_000)
        p.errors = errors  # type: ignore[attr-defined]
        yield p
        browser.close()


# ---- it renders at all -----------------------------------------------------------------------


def test_the_page_renders_every_chapter_without_throwing(page):
    sections = page.query_selector_all("section")
    assert len(sections) >= 5, f"only {len(sections)} chapters rendered"
    assert not page.query_selector_all(".err"), "a chapter reported an error on the page"
    assert not page.errors, f"console errors: {page.errors[:3]}"


def test_the_lede_placeholders_were_filled(page):
    """`data-fact` spans ship with placeholder text. Unfilled, the page reads as a draft."""
    for el in page.query_selector_all("[data-fact]"):
        text = el.inner_text().strip()
        assert text, f"lede fact {el.get_attribute('data-fact')} is empty"
        assert "…" not in text


def test_no_headline_figure_reads_as_nothing(page):
    """Exercise 04's lesson: a headline reading 0 is a wrong question, not a caption problem."""
    for el in page.query_selector_all(".bignum-v"):
        text = el.inner_text().strip()
        assert text not in {"", "0", "—", "NaN", "undefined", "null"}, (
            f"a headline figure reads {text!r}"
        )


def test_no_undefined_or_nan_leaks_into_the_page(page):
    body = page.inner_text("body")
    for bad in ("undefined", "NaN", "[object Object]"):
        assert bad not in body, f"{bad!r} is visible on the page"


def test_the_rail_lists_every_chapter_with_its_title(page):
    """The bug this pins: exercise 04's rail stripped a leading token from `h2.textContent`, which
    runs the number and title together, and ate the first word of every label.
    """
    links = page.query_selector_all(".rail-link .rail-t")
    assert len(links) >= 5
    titles = {el.inner_text().strip() for el in links}
    assert "Out of what?" in titles, f"rail titles are {titles}"
    assert all(t and not t[0].isdigit() for t in titles), f"a rail label kept its number: {titles}"


# ---- the design system actually applied -------------------------------------------------------


def test_the_root_palette_is_loaded(page):
    """Colour tokens live in the *site-root* stylesheet, not the per-exercise copy.

    If this fails the page is being served from `web/` rather than the assembled `public/`, and
    every semantic colour on it is inherited nothing.
    """
    accent = page.evaluate(
        "getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()"
    )
    assert accent, "--accent is undefined; the root stylesheet did not load"


def test_every_token_the_stylesheet_references_is_defined(page):
    """An undefined custom property fails silently, so every reference is checked, not a hand list.

    This page used invented names (`--good`, `--warn`, `--bad`) at first, and every verdict badge
    rendered with no colour while looking perfectly fine.

    The first version of this test named the tokens it expected and asserted *those* were defined.
    Mutation testing killed it: swapping a `var(--grade-x)` usage for `var(--nonexistent)` left
    `--grade-x` defined, so the test passed against a stylesheet referencing a token that does not
    exist. It now reads the references out of the stylesheet, which is the thing that can go wrong.
    """
    css = (export.EXERCISE_ROOT / "web" / "page-extra.css").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"var\((--[a-z0-9-]+)", css)))
    assert referenced, "no custom properties found; the pattern or the stylesheet has moved"

    undefined = [
        token
        for token in referenced
        if not page.evaluate(
            f"getComputedStyle(document.documentElement).getPropertyValue('{token}').trim()"
        )
    ]
    assert not undefined, f"the stylesheet styles with undefined tokens: {undefined}"


def test_verdict_badges_are_actually_coloured(page):
    """The consequence of the above, checked on a rendered element rather than on the stylesheet."""
    badge = page.query_selector(".verdict")
    assert badge, "no verdict badge rendered"
    colour = page.evaluate("el => getComputedStyle(el).color", badge)
    assert colour not in ("", "rgba(0, 0, 0, 0)"), f"badge colour is {colour!r}"


# ---- layout ----------------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", [(1500, 900), (900, 800), (390, 844)])
def test_the_page_never_scrolls_sideways(page, width, height):
    """A page that scrolls horizontally on a phone is broken, and it is easy not to notice.

    Exercise 04 shipped 312px of sideways scroll from invisible absolutely-positioned tooltips.
    """
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(150)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"{overflow}px of horizontal scroll at {width}x{height}"


def test_wide_tables_scroll_inside_their_own_container(page):
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(150)
    wrappers = page.query_selector_all(".tblwrap")
    assert wrappers, "no tables rendered"
    for wrapper in wrappers:
        style = page.evaluate("el => getComputedStyle(el).overflowX", wrapper)
        assert style in ("auto", "scroll"), f"a table wrapper has overflow-x: {style}"


# ---- the interactions actually do something ---------------------------------------------------


def test_dragging_a_lane_changes_the_other_lanes(page):
    """A slider wired to nothing looks identical to one that works."""
    page.set_viewport_size({"width": 1280, "height": 900})
    shares = page.query_selector_all("#composer .compose-share")
    before = [el.inner_text() for el in shares]

    slider = page.query_selector("#composer input[type=range]")
    slider.fill("600")
    slider.dispatch_event("input")
    page.wait_for_timeout(120)

    after = [el.inner_text() for el in page.query_selector_all("#composer .compose-share")]
    assert before != after, "moving a slider changed nothing"
    changed = sum(1 for a, b in zip(before, after, strict=True) if a != b)
    assert changed >= 2, "only the dragged lane moved; the others did not renormalise"


def test_starving_a_protected_lane_shows_the_breach(page):
    """The claim the composer exists to prove: a floor is visible when it is crossed."""
    page.click("#composer .btn.ghost")  # the "crawl what is cheap" preset
    page.wait_for_timeout(150)
    assert page.query_selector("#composer .compose-row.breached"), (
        "the naive preset starves Indic below its floor and the page did not say so"
    )
    summary = page.inner_text("#composer .compose-summary")
    assert "floor" in summary.lower()


def test_the_reset_button_restores_the_specification(page):
    page.click("#composer .btn:not(.ghost)")
    page.wait_for_timeout(150)
    assert not page.query_selector("#composer .compose-row.breached")


def test_the_repetition_slider_moves_both_bars(page):
    seen = page.query_selector("#repetition .rep-bar.seen")
    worth = page.query_selector("#repetition .rep-bar.worth")
    widths = lambda: (  # noqa: E731
        page.evaluate("el => el.style.width", seen),
        page.evaluate("el => el.style.width", worth),
    )
    before = widths()
    slider = page.query_selector("#repetition input[type=range]")
    slider.fill("40")
    slider.dispatch_event("input")
    page.wait_for_timeout(120)
    assert widths() != before, "the repetition chart did not respond"


def test_the_tier_toggle_moves_the_hole_rather_than_filling_it(page):
    """The point of that interaction: if both readings showed no gap it would prove the reverse."""
    first = page.inner_text("#tiers .tier-out")
    buttons = page.query_selector_all("#tiers .toggle .btn")
    assert len(buttons) == 2
    buttons[1].click()
    page.wait_for_timeout(150)
    second = page.inner_text("#tiers .tier-out")
    assert first != second, "the tier toggle changed nothing"
    for text in (first, second):
        assert "must be generated" in text.lower() or "generated" in text.lower()


def test_the_results_chapter_shows_the_seed_spread_by_default(page):
    """Hiding it is the interaction; showing it is the default, because that is the honest view."""
    scores = page.inner_text("#results")
    assert "±" in scores, "the seed spread is not shown by default"
    page.click("#results .btn.ghost")
    page.wait_for_timeout(120)
    assert "±" not in page.inner_text("#results table"), "hiding the spread did nothing"


def test_the_verdict_that_went_against_us_is_on_the_page_with_its_reason(page):
    """The result that cost the specification a clean sweep has to be visible, not buried.

    Written against whatever the bundle says rather than against a hard-coded word: this test
    asserted `qualified` until the corpus grew and the same hypothesis came back `refuted`, at
    which point it failed for describing the old run rather than for anything being wrong. The
    guard that matters is *a losing verdict is shown, with the clause that produced it* — not which
    particular verdict it happened to be.
    """
    bundle = json.loads(
        (PUBLIC / SLUG / "data.js")
        .read_text(encoding="utf-8")
        .split("Object.freeze(", 1)[1]
        .rsplit(");", 1)[0]
    )
    losing = [
        c["verdict"]
        for c in bundle["experiment"]["comparisons"]
        if c["verdict"] in {"qualified", "refuted", "inconclusive"}
    ]
    assert losing, "no hypothesis lost; this test is watching for something that did not happen"

    # The badge, not just the prose. Blanking the badge left the word elsewhere in the section, so
    # an earlier version of this test passed against a table with an empty verdict column.
    badges = {
        element.inner_text().strip().lower()
        for element in page.query_selector_all("#results .verdict")
    }
    for verdict in losing:
        assert verdict in badges, f"the {verdict} verdict has no badge in the results table"
    assert "second clause" in page.inner_text("#results").lower(), (
        "the reason the verdict turned must be shown beside it"
    )


def test_the_results_takeaway_counts_the_same_verdicts_the_table_does(page):
    """The chapter's one-line takeaway must agree with the verdicts beside it.

    It used to be a hand-written sentence — "one verdict did not survive its own noise" — that was
    true of an earlier run and survived the one that replaced it. It is computed now, and this is
    what stops it drifting again: the number in the pill is checked against the badges rendered
    below it, so the summary cannot disagree with what it summarises.

    **What it does not catch**, said plainly rather than implied: this guards the *count*, not the
    wording. Appending "(all fine)" after a correct count survives it. Guarding arbitrary prose is
    not something a test can do; guarding the number it states is.
    """
    badges = [
        element.inner_text().strip().lower()
        for element in page.query_selector_all("#results .verdict")
    ]
    lost = [verdict for verdict in badges if verdict != "supported"]
    pill = page.inner_text("#results .takeaway").lower()

    expected = "one prediction" if len(lost) == 1 else f"{len(lost)} predictions"
    assert expected in pill, (
        f"the takeaway says {pill!r} but {len(lost)} verdicts are not supported"
    )
    assert f"of {len(badges)}" in pill, "the takeaway must say how many predictions there were"


def test_no_markup_reaches_the_reader_unrendered(page):
    """Nothing on the page may show `[[term|label]]`, `**bold**` or a stray backtick.

    Three of these shipped. `rich()` is a single flat pass and regex alternation picks the
    *earliest* match, not the first alternative — so `**[[supply|supply]] is ...**` matched the
    bold rule first and its contents went in as literal text. And table headers never called
    `rich()` at all, while body cells always did, so `[[BPB|bpb]]` rendered six times in one table.

    Asserted against the rendered text rather than the source, because the source is correct in
    both bugs; only the rendering is wrong.
    """
    leftovers = page.evaluate(
        r"""() => {
            const found = [];
            const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walk.nextNode())) {
                const text = node.textContent;
                if (/\[\[|\*\*|`/.test(text)) {
                    found.push(node.parentElement.tagName + ': ' + text.trim().slice(0, 70));
                }
            }
            return found;
        }"""
    )
    assert leftovers == [], f"unrendered markup reached the reader: {leftovers}"


def test_a_glossary_term_inside_bold_still_becomes_a_term(page):
    """The specific nesting that failed, pinned so the flat-parse regression cannot return."""
    terms = page.evaluate(
        "() => [...document.querySelectorAll('.term')].map(t => t.textContent.trim())"
    )
    assert "supply" in terms, "the glossary term inside a bold span did not render as a term"
    bolded = page.evaluate("() => [...document.querySelectorAll('b .term')].length")
    assert bolded > 0, "no glossary term is nested inside bold; the parser is not being exercised"


def test_the_lede_count_agrees_with_the_findings_it_introduces(page):
    """The lede states a count and then lists the findings. They have to be the same number.

    It said "1 of them stop being affordable" — wrong in two ways at once. The number counted only
    lanes with an `impossible` verdict, while `SPEC.md` is built on three findings, and the verb
    did not agree with the number it had just computed.
    """
    lede = page.inner_text(".lede")
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    stated = next((n for word, n in words.items() if f"{word} of them" in lede), None)
    assert stated is not None, f"the lede states no count: {lede[:160]!r}"

    assert ("stops being affordable" in lede) == (stated == 1), (
        "the verb does not agree with the count the lede just stated"
    )

    bundle = json.loads(
        (PUBLIC / SLUG / "data.js")
        .read_text(encoding="utf-8")
        .split("Object.freeze(", 1)[1]
        .rsplit(");", 1)[0]
    )
    short = len([d for d in bundle["headline_disagreements"] if d["gap"] < 0])
    impossible = len([lane for lane in bundle["lanes"] if lane["verdict"] == "impossible"])
    retired = len([lane for lane in bundle["lanes"] if lane["share"] == 0 and lane["raw_supply"]])
    assert stated == short + impossible + retired, (
        f"the lede says {stated} but the data carries {short + impossible + retired} findings"
    )


def test_the_rail_titles_are_readable_rather_than_one_word_per_line(page):
    """The rail's text must actually get the rail's text column.

    It did not. `.rail-link` is a two-column grid in the shared stylesheet — a number column and a
    text column — and this page nested the number *inside* the body, so the grid saw a single child
    and put it in the 16px number column. Every title then wrapped one word per line down the whole
    rail. Twenty-seven browser tests passed while it looked like that, because all of them asked
    whether things were present and none asked whether they were legible.

    Measured, not inspected: a title box narrower than its own number column cannot be right.
    """
    page.set_viewport_size({"width": 1400, "height": 900})
    page.wait_for_selector("#rail .rail-link")

    children = page.eval_on_selector(
        "#rail .rail-link", "el => [...el.children].map(c => c.className)"
    )
    assert children == ["rail-n", "rail-body"], (
        f"the rail link's children are {children}; the shared grid expects the number and the body "
        "as siblings"
    )

    boxes = page.evaluate(
        """() => [...document.querySelectorAll('#rail .rail-t')].map(t => ({
            width: t.getBoundingClientRect().width,
            height: t.getBoundingClientRect().height,
            text: t.textContent.trim(),
        }))"""
    )
    assert boxes, "the rail rendered no titles"
    for box in boxes:
        assert box["width"] > 100, (
            f"rail title {box['text']!r} is only {box['width']:.0f}px wide — it is being squeezed "
            "into the number column"
        )
        # Roughly three lines at this width; more means the text is wrapping far too early.
        assert box["height"] < 70, f"rail title {box['text']!r} wraps to {box['height']:.0f}px tall"


# ---- docs/EXPLAINER_PROMPT.md conformance ----------------------------------------------------


def test_every_number_that_claims_a_provenance_renders_its_mark(page):
    """§6 requires `data-provenance` on displayed numbers; §13 calls the absence of it the limit
    that "matters most", because a confirmed figure and a guess otherwise look identical.

    Checked on the rendered element rather than the stylesheet: the mark has to be *visible*, so
    an estimated number must differ from a measured one in its computed style.
    """
    marks = page.query_selector_all("[data-provenance]")
    assert marks, "no number on the page carries a provenance"

    kinds = {el.get_attribute("data-provenance") for el in marks}
    assert kinds <= {"measured", "estimated", "unknown"}, f"unexpected provenance: {kinds}"

    styles = {}
    for kind in kinds:
        el = page.query_selector(f"[data-provenance='{kind}']")
        styles[kind] = page.evaluate(
            "el => { const s = getComputedStyle(el);"
            " return [s.borderBottomStyle, s.fontStyle, s.opacity].join('|'); }",
            el,
        )
    assert len(set(styles.values())) == len(styles), (
        f"two provenance kinds render identically, so the mark carries no information: {styles}"
    )


def test_the_provenance_legend_explains_the_marks(page):
    """A mark nobody can decode is decoration."""
    legend = page.query_selector(".prov-legend")
    assert legend, "the provenance marks have no legend"
    text = legend.inner_text().lower()
    for kind in ("measured", "estimated", "unknown"):
        assert kind in text


def test_a_lane_whose_supply_has_an_uncounted_row_is_marked_unknown(page):
    """Indic carries two datasets with no token count, so its total cannot be `measured`.

    This is the case the mark exists for: the number looks as solid as any other until it says so.
    """
    row = page.query_selector("#composer .compose-row.indic, #composer .compose-row")
    assert row is not None
    marks = {
        el.get_attribute("data-provenance")
        for el in page.query_selector_all("#composer [data-provenance]")
    }
    assert "unknown" in marks, f"no lane is marked unknown; found {marks}"


def test_no_control_response_animates_for_longer_than_the_reader_can_compare(page):
    """§4 caps control responses at 200ms: "the reader is comparing, and animation between states
    destroys comparison". The repetition bars exist to be compared, and had 550ms.
    """
    bar = page.query_selector(".rep-bar")
    assert bar, "no repetition bar rendered"
    duration = page.evaluate("el => getComputedStyle(el).transitionDuration", bar)
    seconds = max(float(d.rstrip("s")) for d in duration.replace("ms", "e-3s").split(", "))
    assert seconds <= 0.2, f"a control response animates for {seconds}s"


def test_every_chapter_leaves_the_reader_with_one_number(page):
    """§7's checklist: a takeaway pill stating one number."""
    pills = page.query_selector_all(".takeaway")
    sections = page.query_selector_all("section")
    assert len(pills) >= len(sections) - 1, f"{len(pills)} pills for {len(sections)} chapters"
    for pill in pills:
        assert any(ch.isdigit() for ch in pill.inner_text()), (
            f"a takeaway pill states no number: {pill.inner_text()!r}"
        )


def test_the_page_fetches_nothing(page) -> None:
    """No script-initiated request leaves the page — the data is in the module graph.

    EXPLAINER_PROMPT §6 asks for the data precomputed and inlined, and names fetching as the thing
    not to do. The reason is a failure mode, not a preference: a fetch can 404, be blocked, or fail
    on a `file://` open *after* the page has already painted, so the page needs a loading state and
    an error path for a gap that need not exist. This asserts the gap is gone at runtime, by
    reading the browser's own resource timeline rather than grepping the source for `fetch`.

    Resource Timing rather than a second browser on purpose: the module-scoped `page` fixture holds
    a `sync_playwright()` context open for this file, and starting a second one in the same thread
    hangs until the timeout — which reads as "the page renders nothing" and sends you after the
    wrong bug.
    """
    initiators = page.evaluate(
        "performance.getEntriesByType('resource')"
        ".filter(e => e.initiatorType === 'fetch' || e.initiatorType === 'xmlhttprequest')"
        ".map(e => e.name)"
    )
    assert initiators == [], f"the page issued data requests after loading: {initiators}"


def test_the_bundle_is_a_module_the_page_imports() -> None:
    """The served bundle is importable JS, and the page imports it statically.

    The served copy, not `web/`: `build.sh` appends a `?v=<hash>` cache-buster to every local
    script, so the assertion has to tolerate one or it only ever passes on the unbuilt source.
    """
    html = (PUBLIC / SLUG / "index.html").read_text(encoding="utf-8")
    assert "fetch(" not in html, "the page still fetches its data"
    assert re.search(r"import \{ BUNDLE \} from '\./data\.js(\?v=[0-9a-f]+)?'", html), (
        "no static bundle import"
    )
    assert not (PUBLIC / SLUG / "data.json").exists(), "the superseded JSON bundle is still shipped"
    bundle = (PUBLIC / SLUG / "data.js").read_text(encoding="utf-8")
    assert bundle.startswith("/*") and "export const BUNDLE = Object.freeze({" in bundle


def test_the_page_declares_a_reduced_motion_end_state(page):
    """§6: `prefers-reduced-motion: reduce` renders the complete end state.

    For some readers the page never moves, so it has to be as informative standing still.
    """
    css = (export.EXERCISE_ROOT / "web" / "page-extra.css").read_text(encoding="utf-8")
    # Match the @media *rule*, not the comment above it explaining why the rule is there — the
    # first version of this test split on the bare phrase and landed inside the prose.
    match = re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(.*?)\n\}", css, re.S)
    assert match, "no prefers-reduced-motion rule in the stylesheet"
    block = match.group(1)
    assert ".rep-bar" in block, "the animated element is not covered by the reduced-motion block"
    assert "transition: none" in block
