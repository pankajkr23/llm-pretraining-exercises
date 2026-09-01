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
import importlib.util
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
    # `build.sh` starts with `rm -rf public/`, so two xdist workers both finding `public/` absent
    # would delete each other's site mid-test. Fail loudly instead of racing: CI assembles it once
    # up front (see `.github/workflows/ci.yml`), and locally the fix is one command.
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


# 320 is the narrowest phone still in use, and it is where this page actually broke: the rail
# list held a 310px minimum track inside a 272px container and pushed the page 14px sideways.
# The suite stopped at 390, which is why that shipped.
@pytest.mark.parametrize("width,height", [(1500, 900), (900, 800), (390, 844), (320, 568)])
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


def test_no_html_tag_reaches_the_reader_as_text(page):
    """The fourth way markup leaks, and the one the guard above was blind to.

    `rich()` understands `**bold**` and `[[term|key]]`; it does **not** understand HTML, so a cell
    written as `<b>H1</b>` is inserted with `createTextNode` and the reader sees the angle brackets.
    That shipped in the predictions table, and every existing guard passed: the text contains no
    `[[`, no `**` and no backtick, so the check above found nothing to complain about.

    A guard that only knows the three failures it was written for is not a guard against the fourth.
    """
    leaked = page.evaluate(
        r"""() => {
            const found = [];
            const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walk.nextNode())) {
                if (node.parentElement.closest('pre, code')) continue;  // code is meant to show it
                const t = node.textContent.trim();
                if (/<\/?[a-zA-Z][a-zA-Z0-9]*\s*\/?>/.test(t)) {
                    found.push(node.parentElement.tagName + ': ' + t.slice(0, 70));
                }
            }
            return found;
        }"""
    )
    assert leaked == [], f"an HTML tag reached the reader as literal text: {leaked}"


def test_no_stray_emphasis_marker_survives_rendering(page):
    """The fifth leak, and it is a real limitation of `rich()` rather than a typo.

    The bold pattern is `\\*\\*([^*]+)\\*\\*`, whose character class cannot cross an asterisk. So
    `**a *b* c**` never matches as bold; the single-asterisk rule fires on the leading `**` instead
    and the reader gets `*a b c*` with the markers showing. The guard above misses it because the
    rendered text no longer contains a doubled asterisk — only single ones.

    Checked on the edges of a text node rather than anywhere inside it, so an asterisk used as an
    ordinary character mid-sentence is not reported.

    **The lone-marker case is the one that matters and it was nearly missed.** When the parser gives
    up on `**a *b* c**` it emits the opening `*` as its own text node, so the stray marker is a
    single character. The first version of this check required a length above one — it passed
    against the real bug, and only breaking the page on purpose showed that.
    """
    strays = page.evaluate(
        r"""() => {
            const found = [];
            const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = walk.nextNode())) {
                if (node.parentElement.closest('pre, code')) continue;
                const t = node.textContent.trim();
                if (!t) continue;
                const lone = /^\*+$/.test(t);
                const edge = t.length > 1 && (t.startsWith('*') || t.endsWith('*'));
                if (lone || edge) {
                    found.push(node.parentElement.tagName + ': ' + t.slice(0, 70));
                }
            }
            return found;
        }"""
    )
    assert strays == [], f"an emphasis marker is visible to the reader: {strays}"


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


def test_the_results_chapter_describes_the_run_that_actually_happened(page):
    """Chapter 5's prose must match the corpus the bundle records.

    Three claims here were stale on production: the corpus was "built entirely from text this
    repository already tracks" months after three lanes became fetched stand-ins, and "four of the
    seven lanes have no committed text at all, so they were dropped" after all six were funded.
    Both sat beneath a correct, generated table.
    """
    bundle = json.loads(
        (PUBLIC / SLUG / "data.js")
        .read_text(encoding="utf-8")
        .split("Object.freeze(", 1)[1]
        .rsplit(");", 1)[0]
    )
    corpus = bundle["experiment"]["corpus"]
    stand_ins = [
        lane
        for lane, shard in corpus.items()
        if any(str(s).startswith("data/proxy/") for s in shard.get("sources", []))
    ]
    dropped = {
        lane for arm in bundle["experiment"]["arms"].values() for lane in arm["dropped_lanes"]
    }
    # `textContent`, not `inner_text`: the arithmetic block lives inside a collapsed <details>
    # ("under the hood"), and inner_text omits what is not rendered. The claim is still on the page
    # and still wrong when it is wrong, so the guard has to be able to read it.
    text = page.eval_on_selector("#results", "el => el.textContent")

    assert "built entirely from text this repository already tracks" not in text, (
        "the page claims a committed-only corpus while the bundle records fetched stand-ins"
    )
    for lane in stand_ins:
        assert lane in text, f"the {lane} lane is a stand-in and the page never says so"
    assert "stand-ins" in text, "the page must declare that some lanes are stand-ins"

    if not dropped:
        assert "were dropped" not in text, (
            "the page says lanes were dropped; the bundle records none"
        )


def test_the_second_clause_sentence_agrees_with_its_own_verdict(page):
    """A gain that clears its spread must not be described as sitting inside it.

    This sentence asserted "sits inside its own X% seed spread … settles it in neither direction"
    unconditionally. That was true while the verdict was `qualified`; once the gain cleared its
    noise the paragraph directly contradicted the `refuted` badge above it.
    """
    bundle = json.loads(
        (PUBLIC / SLUG / "data.js")
        .read_text(encoding="utf-8")
        .split("Object.freeze(", 1)[1]
        .rsplit(");", 1)[0]
    )
    withsecond = [c for c in bundle["experiment"]["comparisons"] if c.get("secondary")]
    if not withsecond:
        pytest.skip("no hypothesis has a second clause in this bundle")

    text = page.inner_text("#results").lower()
    for comparison in withsecond:
        clears = comparison["secondary"]["clears_noise"]
        says_clears = "clears its own" in text
        assert says_clears == clears, (
            f"{comparison['key']}'s second clause clears_noise={clears}, and the prose says "
            "otherwise"
        )
        if clears:
            assert "settle it in neither direction" not in text, (
                "the gain clears its spread, so these runs do settle it"
            )


def test_the_page_states_its_blind_spots(page):
    """§13: the blind spots are distinguishing content, and the reference format hides them.

    Every one of these lived only in the documents. The page — the artefact a reviewer actually
    opens — carried the findings and none of the limits, which is the format hiding the best
    material exactly as §13 warns.
    """
    text = page.eval_on_selector("#limits", "el => el.textContent")
    bundle = json.loads(
        (PUBLIC / SLUG / "data.js")
        .read_text(encoding="utf-8")
        .split("Object.freeze(", 1)[1]
        .rsplit(");", 1)[0]
    )
    stand_ins = [
        lane
        for lane, shard in bundle["experiment"]["corpus"].items()
        if any(str(s).startswith("data/proxy/") for s in shard.get("sources", []))
    ]
    assert "could not see" in text, "the page never says what the runs were blind to"
    for lane in stand_ins:
        assert lane in text, f"{lane} is a stand-in and the blind spots never name it"
    assert "not scheduled" in text, "the page must say the deciding run is not scheduled"
    assert "not independent" in text, (
        "two agreeing proxies must be declared non-independent, or agreement reads as corroboration"
    )


def test_the_blind_spots_are_not_hidden_behind_a_disclosure(page):
    """A limitation a reader must open a drawer to find is a limitation the page is hiding."""
    visible = page.inner_text("#limits")
    assert "could not see" in visible, (
        "the blind-spot block is inside a collapsed <details>; it must be in the open text"
    )


def test_the_corrections_log_is_on_the_page(page):
    """§13's third piece: what we got wrong, and how we found out.

    The strongest material in the exercise — a verdict that flipped because a lane that had been
    missing was funded, with the effect size essentially unchanged — appeared nowhere on the page.
    """
    text = page.eval_on_selector("#negatives", "el => el.textContent")
    assert "What we got wrong" in text
    assert "unfalsifiable" in text, "the transferable lesson must be stated, not just the anecdote"
    assert "Stack Exchange" in text, "the second stand-in check is what makes the finding hold up"


def test_the_prediction_is_asked_before_the_answer_is_shown(page):
    """§14.1: the reader commits, then the answer appears with their guess pinned.

    Revealing on load would make it a caption. The output must be empty until asked for, and the
    guess must survive the reveal — the gap is the whole lesson.
    """
    predicts = page.query_selector_all("#negatives .predict")
    assert len(predicts) == 1, (
        f"{len(predicts)} predict blocks; §14.1 caps this at three per page and one is what this "
        "page spends"
    )

    out = page.query_selector("#negatives .predict-out")
    assert out.inner_text().strip() == "", "the answer is on screen before the reader has guessed"

    page.eval_on_selector(
        "#negatives .predict input[type=range]",
        "el => { el.value = '1.80'; el.dispatchEvent(new Event('input', {bubbles: true})); }",
    )
    page.click("#negatives .predict .btn")
    page.wait_for_timeout(150)

    revealed = page.inner_text("#negatives .predict-out")
    assert "your guess" in revealed and "actual" in revealed, "the guess is not pinned beside it"
    assert "1.80" in revealed, "the reader's own guess was discarded on reveal"
    assert "out by" in revealed, "the gap is the lesson and it is not labelled"


def test_the_answer_is_reachable_without_playing(page):
    """Reveal must not require a guess first: a reader who will not play is not locked out."""
    page.reload(wait_until="load")
    page.wait_for_selector("#negatives .predict")
    page.click("#negatives .predict .btn")
    page.wait_for_timeout(150)
    assert "actual" in page.inner_text("#negatives .predict-out")


def test_the_page_defines_the_words_it_uses(page):
    """A reader meeting `arm`, `held-out` or `seed spread` must be able to find out what they mean.

    The results chapter used all of these as if they were common knowledge. They are not: `arm` in
    particular means something specific here — one training run with one mixture — and a table
    headed `arm` teaches a reader nothing without it.
    """
    defined = page.evaluate(
        """() => Object.fromEntries([...document.querySelectorAll('.term')]
            .map(t => [t.textContent.trim().toLowerCase(), (t.dataset.def || '').length]))"""
    )
    for term in ("arms", "held-out", "seed spread", "proxy model", "bits per byte"):
        assert term in defined, f"the page uses {term!r} without defining it"
        assert defined[term] > 40, f"{term!r} has a definition too short to be one"


def test_the_page_says_what_the_metric_measures_not_just_its_name(page):
    """ "Held-out BPB, lower is better" names a metric without explaining it.

    A reader cannot judge the table without knowing that it measures surprise, and that it is per
    byte specifically so it survives the tokenizer change this specification plans.
    """
    # Scoped to the caption itself. Searching all of #results passed against a mutant that gutted
    # the explanation, because "tokenizer" also appears in the collapsed block below — a guard is
    # only as tight as the element it reads.
    caption = page.evaluate(
        """() => {
            const p = [...document.querySelectorAll('#results p')]
                .find(el => el.textContent.includes('bits per byte'));
            return p ? p.textContent : '';
        }"""
    )
    assert caption, "the results table has no metric caption at all"
    assert "surprised" in caption, "the metric is named but never explained"
    # NOT `"per byte" in caption` — the term itself is "bits per byte", so that assertion is
    # satisfied by the name and would pass against a caption that explains nothing.
    assert "rather than" in caption and "per token" in caption, (
        "the caption never says the metric is per byte *rather than* per token"
    )
    assert "tokenizer" in caption, (
        "the reason it is measured per byte is the part that justifies the choice"
    )


def test_the_page_states_the_scale_of_the_model_in_the_open(page):
    """The proxy is ~7,000x smaller than the specification's subject, and that governs every claim.

    It was previously stated only inside the collapsed `under the hood` block, which is precisely
    where a reader who is deciding how much to trust the table will not look.
    """
    visible = page.inner_text("#results")
    assert "7,000" in visible, "the page does not say how much smaller the proxy is"
    assert "smaller" in visible


def test_the_page_points_somewhere_for_the_full_explanation(page):
    """Tooltips answer a word. Somebody who wants the whole apparatus needs a destination."""
    pointer = page.inner_text(".summary-more")
    assert "METHOD.md" in pointer
    href = page.eval_on_selector(".summary-more a", "el => el.href")
    assert href.endswith("METHOD.md"), f"the pointer does not resolve to the document: {href}"


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
    """§7's checklist: a takeaway pill stating one number.

    Scoped to the numbered chapters — the ones that argue from a figure. The spine's prose sections
    around them (the glossary, the problem, how to reproduce it) have no single number to leave a
    reader with, and inventing a pill for them would be decoration rather than a takeaway.
    """
    pills = page.query_selector_all(".takeaway")
    chapters = page.query_selector_all(
        'section[data-role="mechanism"], section[data-role="results"]'
    )
    assert chapters, "no numbered chapters found"
    assert len(pills) >= len(chapters), f"{len(pills)} pills for {len(chapters)} chapters"
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


def _required_spine() -> tuple[str, ...]:
    """The spine, read from the repo-wide guard so this list cannot drift from it.

    Loaded by path rather than imported: `tests/` is not a package, and adding an `__init__.py`
    to make it one would change how pytest collects every file in it. One source of truth is worth
    five lines of importlib.
    """
    path = REPO / "tests" / "test_page_spine.py"
    spec = importlib.util.spec_from_file_location("_page_spine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPINE


def test_the_page_has_the_required_spine_in_order(page):
    """A reader arriving cold must find every part of the story, in an order that makes sense.

    The repo-wide `tests/test_page_spine.py` checks that this page *declares* each role, reading the
    source. It cannot see DOM order, because `buildPage` decides that at runtime. This is the other
    half: order is asserted because `limits` before `results` reads as hedging, and `conclusion`
    before the evidence reads as a press release.
    """
    spine = _required_spine()

    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in spine if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"

    seen = [r for r in roles if r in spine]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(spine), f"the spine is out of order: {first}"


# ---- the mechanism figure ---------------------------------------------------------------------
#
# This page had no drawn figure at all until v0.11.1 — fifteen sections of sliders, tables and mark
# strips. `AGENTS.md`: "A mechanism figure is not a results chart, and a page needs both. Results
# say what happened; mechanism says why it must." The slider samples one point of the repetition
# curve; the curve's shape and its asymptote were reachable only by dragging.


def _bundle(page) -> dict:
    """The page's own data, read from the served bundle."""
    return json.loads(
        (PUBLIC / SLUG / "data.js")
        .read_text(encoding="utf-8")
        .split("Object.freeze(", 1)[1]
        .rsplit(");", 1)[0]
    )


def test_the_page_draws_a_mechanism_figure(page):
    """A page of results and no mechanism can be believed but not understood."""
    figures = page.eval_on_selector_all(
        'section[data-role="mechanism"] figure', "els => els.length"
    )
    assert figures >= 1, "no figure in any mechanism section; the page shows what, never why"


def test_the_figure_caption_argues_rather_than_labels(page):
    """A caption whose text is its title has made the reader do the interpreting."""
    caps = page.eval_on_selector_all("figure figcaption", "els => els.map(e => e.innerText)")
    assert caps, "a figure with no caption at all"
    for cap in caps:
        assert len(cap.split()) >= 40, f"caption too thin to be arguing anything: {cap[:80]!r}"


def test_the_figure_does_not_silently_drop_a_funded_lane(page):
    """**The failure this guard exists for, and it shipped in the first draft.**

    `docs/DESIGN.md`: *"Draw the whole object, not the part that fits."* The first version of this
    figure filtered out every lane past the axis maximum and then labelled the remainder *"all 5
    funded lanes"* — when there are six. The one it dropped is the lane that cannot be funded at any
    price, which is the entire point of the chapter after it.

    So: every funded lane must be **either** plotted on the axis **or** named in the off-scale
    annotation. Never absent.
    """
    bundle = _bundle(page)
    funded = [lane for lane in bundle["lanes"] if lane["share"] > 0 and lane["epochs"] > 0]
    assert funded, "no funded lanes in the bundle; this guard would pass vacuously"

    on_axis = page.eval_on_selector_all("figure .fig-dot:not(.bad)", "els => els.length")
    off_axis = page.eval_on_selector_all("figure .fig-dot.bad", "els => els.length")
    assert on_axis + off_axis == len(funded), (
        f"{len(funded)} funded lanes, but the figure draws {on_axis} on-axis and {off_axis} "
        f"off-axis — one is missing entirely"
    )

    # And the off-scale ones must be named, not merely dotted: an unlabelled dot at the edge says
    # nothing about which lane it is or how far past the axis it sits.
    text = page.eval_on_selector("figure", "el => el.textContent")
    axis_max = 40
    for lane in funded:
        if lane["epochs"] > axis_max:
            assert lane["key"] in text, (
                f"lane {lane['key']!r} is off this figure's axis at {lane['epochs']:.0f} passes "
                f"and the figure never names it"
            )


def test_the_figure_states_the_count_it_actually_draws(page):
    """A count in a figure's own annotation, beside the dots it counts."""
    bundle = _bundle(page)
    funded = [lane for lane in bundle["lanes"] if lane["share"] > 0 and lane["epochs"] > 0]
    on_axis = page.eval_on_selector_all("figure .fig-dot:not(.bad)", "els => els.length")
    text = page.eval_on_selector("figure", "el => el.textContent")
    assert f"{on_axis} of the {len(funded)} funded lanes" in text, (
        f"the figure's annotation does not say '{on_axis} of the {len(funded)} funded lanes'"
    )
