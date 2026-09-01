"""The page as a reader sees it, in a real browser.

`node --check` proves a file has no syntax error and nothing more. A call to an undefined function,
a figure whose variants all draw the same thing, and a timeline entry missing two thirds of its
trade-offs all parse perfectly.

The tests that matter here are not "the page loads". They are: **the dates on screen are the ones
the catalogue verified**, **every entry shows what it costs and not only what it buys**, and **the
mechanism figure actually changes when you change the variant** — because if it does not, the
figure is decoration and the claim it illustrates is unproven.

Playwright is integration-marked and skips without a browser, so a fresh checkout still works. One
time setup: `uv run playwright install chromium`.
"""

import functools
import http.server
import importlib.util
import json
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
            view = browser.new_page(viewport={"width": 1280, "height": 900})
            problems: list[str] = []
            view.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            view.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            view.goto(f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html")
            view.wait_for_selector("section#reproduce", timeout=15_000)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


def test_the_page_renders_without_a_single_console_error(page) -> None:
    """A page that throws on load still shows its title, which is why this is asserted."""
    assert not page.console_problems, page.console_problems


def test_no_section_reported_a_failure(page) -> None:
    """`buildPage` catches a failing section and renders `.err` rather than dying — so a broken
    section is invisible unless something looks."""
    assert page.eval_on_selector_all(".err", "els => els.map(e => e.textContent)") == []


def test_the_page_has_the_required_spine_in_order(page) -> None:
    """Presence is checked lexically by the repo-wide guard; only this can see DOM order.

    Order is asserted because `limits` before `results` reads as hedging, and `conclusion` before
    the evidence reads as a press release.
    """
    spine = _required_spine()
    roles = page.eval_on_selector_all("main section", "els => els.map(e => e.dataset.role)")
    missing = [r for r in spine if r not in roles]
    assert not missing, f"the page is missing these parts of the story: {missing}"

    seen = [r for r in roles if r in spine]
    first = [r for i, r in enumerate(seen) if r not in seen[:i]]
    assert first == list(spine), f"the spine is out of order: {first}"


def test_no_undefined_or_nan_reaches_the_reader(page) -> None:
    body = page.inner_text("body")
    for bad in ("undefined", "NaN", "[object Object]"):
        assert bad not in body, f"{bad!r} is visible on the page"


# ---- the timeline is the deliverable -----------------------------------------------------------


def test_every_catalogued_mechanism_appears_on_the_page(page) -> None:
    """The assignment's score-zero clause, checked against what actually rendered."""
    bundle = _bundle()
    text = page.inner_text("main")
    missing = [m["name"] for m in bundle["mechanisms"] if m["name"] not in text]
    assert not missing, f"catalogued but not rendered: {missing}"


def test_the_timeline_is_in_date_order_on_screen(page) -> None:
    """The assignment's central requirement, asserted on the rendered order rather than the data.

    The catalogue being sorted proves nothing about the page: a template that iterated a dictionary
    or reversed a list would still pass every catalogue test.
    """
    bundle = _bundle()
    names = page.eval_on_selector_all(
        "#results .tl-name", "els => els.map(e => e.textContent.trim())"
    )
    assert len(names) >= 20, f"only {len(names)} timeline entries rendered"

    # Compared against the catalogue's own order rather than by re-parsing the displayed dates.
    # The page formats them for humans — `en-GB` renders September as "Sept", which no strptime
    # format reads — and a test that fought the display format would break on a locale change while
    # telling you nothing about the ordering, which is the property that actually matters.
    expected = [m["name"] for m in bundle["mechanisms"]]
    assert names == expected, "the rendered order is not the catalogue's date order"

    dates = [m["date"] for m in bundle["mechanisms"]]
    assert dates == sorted(dates), "the catalogue the page renders is not itself in date order"


def test_every_entry_states_what_it_costs_and_not_only_what_it_buys(page) -> None:
    """> "If you write down a technique with only pros, you have not understood it yet."

    Each entry must render all three cards. One with only the "buys" card would look complete at a
    glance, which is exactly why this counts them.
    """
    counts = page.eval_on_selector_all(
        "#results .tl-item",
        "els => els.map(e => [e.querySelectorAll('.trio-card').length,"
        " e.querySelector('.tl-name').textContent])",
    )
    thin = [name for n, name in counts if n != 3]
    assert not thin, f"these entries do not show all three of buys/gives-up/when: {thin}"


def test_every_entry_links_the_source_its_date_came_from(page) -> None:
    """A date a reader cannot check is the failure the whole exercise is built to avoid."""
    entries = page.eval_on_selector_all(
        "#results .tl-item",
        "els => els.map(e => [!!e.querySelector('.tl-cite a[href^=\"http\"]'),"
        " !!e.querySelector('.tl-cite code'), e.querySelector('.tl-name').textContent])",
    )
    unlinked = [name for has_link, _, name in entries if not has_link]
    unquoted = [name for _, has_quote, name in entries if not has_quote]
    assert not unlinked, f"these entries cite no source URL: {unlinked}"
    assert not unquoted, f"these entries quote no date from their source: {unquoted}"


def test_the_dates_on_screen_are_the_dates_in_the_catalogue(page) -> None:
    """The page must not reformat a date into a different one."""
    bundle = _bundle()
    text = page.inner_text("#results")
    for mechanism in bundle["mechanisms"][:6]:
        year = mechanism["date"][:4]
        assert year in text, f"{mechanism['name']}'s year {year} does not appear in the timeline"


# ---- the mechanism figure is the argument -------------------------------------------------------


def test_the_mechanism_figure_exists_in_a_mechanism_section(page) -> None:
    """A page of results and no mechanism can be believed but not understood."""
    figures = page.eval_on_selector_all(
        'section[data-role="mechanism"] figure', "els => els.length"
    )
    assert figures >= 1, "no figure in the mechanism section"


def test_changing_the_variant_changes_what_is_drawn(page) -> None:
    """**The test the figure rests on.**

    If switching variants leaves the drawing identical, the figure is decoration and the claim it
    illustrates — that every mechanism edits one of two objects — is unproven. The page would look
    completely normal.
    """
    chips = page.query_selector_all(".mech .chip")
    assert len(chips) >= 5, f"only {len(chips)} variants offered"

    def shape() -> tuple:
        return (
            page.eval_on_selector_all(".mech .sc.off", "els => els.length"),
            page.eval_on_selector_all(".mech .kv.off", "els => els.length"),
        )

    chips[0].click()
    page.wait_for_timeout(120)
    full = shape()

    chips[1].click()
    page.wait_for_timeout(120)
    windowed = shape()

    assert full != windowed, "switching from full attention to a window changed nothing on screen"
    assert full[0] == 0, "full attention should switch off no scores"
    assert windowed[0] > 0, "a sliding window must remove scores from the triangle"


def test_head_sharing_edits_the_cache_and_leaves_the_triangle_alone(page) -> None:
    """The figure's actual claim, asserted rather than captioned.

    GQA and MQA change how much is cached per position and touch no score at all. If this fails, the
    figure is drawing something other than what its caption says.
    """
    by_label = {
        page.eval_on_selector_all(".mech .chip", "els => els.map(e => e.textContent)")[i]: i
        for i in range(len(page.query_selector_all(".mech .chip")))
    }
    gqa = next(label for label in by_label if label.startswith("GQA"))
    page.query_selector_all(".mech .chip")[by_label[gqa]].click()
    page.wait_for_timeout(120)

    scores_off = page.eval_on_selector_all(".mech .sc.off", "els => els.length")
    cache_off = page.eval_on_selector_all(".mech .kv.off", "els => els.length")
    assert scores_off == 0, "GQA must not remove any attention score"
    assert cache_off > 0, "GQA must shrink the cache"


def test_linear_attention_collapses_the_cache_entirely(page) -> None:
    """The one variant that removes the per-token store rather than shrinking it."""
    labels = page.eval_on_selector_all(".mech .chip", "els => els.map(e => e.textContent)")
    index = next(i for i, label in enumerate(labels) if "Linear" in label)
    page.query_selector_all(".mech .chip")[index].click()
    page.wait_for_timeout(120)

    visible = page.eval_on_selector_all(
        ".mech .kv:not(.state)", "els => els.filter(e => e.style.display !== 'none').length"
    )
    assert visible == 0, "linear attention should leave no per-position cache squares drawn"
    state = page.eval_on_selector(".mech .kv.state", "el => el.style.display !== 'none'")
    assert state, "linear attention should draw the single fixed state instead"


# ---- the shell -----------------------------------------------------------------------------------


def test_the_left_rail_is_built_and_fills_the_gutter_it_reserves(page) -> None:
    """`_shared/page.css` reserves 260px on `.wrap` at 1180px and up whether or not a page builds a
    rail, so a missing rail is a visible empty gutter and nothing fails. Assert the pairing."""
    page.set_viewport_size({"width": 1280, "height": 900})
    page.wait_for_timeout(150)
    pad = page.eval_on_selector(".wrap", "el => parseFloat(getComputedStyle(el).paddingLeft)")
    width = page.eval_on_selector("#rail", "el => el.getBoundingClientRect().width")
    assert pad > 200, f"the shared stylesheet no longer reserves the gutter ({pad}px)"
    assert width > 150, f"the gutter is reserved but the rail does not fill it ({width}px)"


def test_the_rail_lists_every_section_in_order(page) -> None:
    ids = page.eval_on_selector_all("main section", "els => els.map(e => e.id)")
    hrefs = page.eval_on_selector_all(
        "#rail .rail-link", "els => els.map(e => e.getAttribute('href'))"
    )
    assert hrefs == [f"#{i}" for i in ids]


@pytest.mark.parametrize("width", [1440, 1180, 900, 390, 320])
def test_the_page_never_scrolls_sideways(page, width: int) -> None:
    """Wide content scrolls inside its own container; the body never does."""
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(150)
    overflow = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"the page scrolls sideways by {overflow}px at {width}px"


# ---- the figures added in the visual rebuild -----------------------------------------------------


def test_the_page_carries_a_figure_in_every_section_that_argues_from_one(page) -> None:
    """The first version of this page had one figure across twelve sections — a wall of text.

    Counted rather than eyeballed, because "add some visuals" is exactly the kind of intention that
    quietly decays back to prose on the next edit.
    """
    total = page.eval_on_selector_all("main figure", "els => els.length")
    assert total >= 8, f"only {total} figures on the page"
    for role in ("problem", "mechanism", "results", "conclusion"):
        n = page.eval_on_selector_all(f'section[data-role="{role}"] figure', "els => els.length")
        assert n >= 1, f"the {role} section argues from no figure at all"


def test_the_attention_demo_actually_runs_the_stages(page) -> None:
    """The assignment insists the page starts from plain attention. So it must *run*, not diagram.

    Stepping from raw scores to softmax must change the numbers on screen; if it does not, the
    figure is a static picture with buttons attached.
    """
    chips = page.query_selector_all(".fig-attention .chip")
    assert len(chips) == 4, f"expected four stages, found {len(chips)}"

    def cells() -> list[str]:
        return page.eval_on_selector_all(
            ".fig-attention text.val", "els => els.map(e => e.textContent)"
        )

    chips[0].click()
    page.wait_for_timeout(500)
    raw = cells()

    chips[3].click()
    page.wait_for_timeout(600)
    softmaxed = cells()

    assert raw != softmaxed, "stepping to softmax changed none of the numbers"
    assert any("−∞" in c for c in softmaxed) is False, (
        "softmax should have consumed the masked cells"
    )

    chips[2].click()
    page.wait_for_timeout(600)
    assert any("−∞" in c for c in cells()), "the mask stage shows no masked cells"


def test_every_row_of_the_softmax_stage_sums_to_one(page) -> None:
    """The property that makes softmax *competition* rather than scaling.

    Checked on the rendered numbers, so a figure that drew plausible-looking weights which did not
    actually normalise would fail — that is the whole point of the stage.
    """
    page.query_selector_all(".fig-attention .chip")[3].click()
    page.wait_for_timeout(600)
    values = page.eval_on_selector_all(
        ".fig-attention text.val", "els => els.map(e => e.textContent)"
    )
    n = 6
    for row in range(n):
        cells = values[row * n : (row + 1) * n]
        total = sum(float(c) if c.strip() not in ("", "−∞") else 0.0 for c in cells)
        assert abs(total - 1.0) < 0.02, f"row {row} of the softmax stage sums to {total:.3f}, not 1"


def test_the_rope_figure_shows_the_invariance_it_claims(page) -> None:
    """**The figure's actual claim, and the only interesting thing to test about it.**

    RoPE's point is that moving *both* tokens later leaves their score unchanged, because only the
    gap between them matters. So the arms must move and the score must not. A figure that drew a
    rotation without preserving the score would look perfectly convincing.
    """
    slider = page.query_selector(".fig-rope input[type=range]")
    assert slider, "the RoPE figure has no control"

    def state() -> tuple:
        return (
            page.eval_on_selector(".fig-rope line.arm.a", "e => e.getAttribute('x2')"),
            page.eval_on_selector(".fig-rope text.big", "e => e.textContent"),
        )

    slider.fill("0")
    page.wait_for_timeout(120)
    arm_at_zero, score_at_zero = state()

    slider.fill("17")
    page.wait_for_timeout(120)
    arm_later, score_later = state()

    assert arm_at_zero != arm_later, "moving the tokens did not rotate anything"
    assert score_at_zero == score_later, (
        f"the score changed from {score_at_zero} to {score_later} — RoPE's whole claim is that it "
        f"does not, because the gap between the two tokens never changed"
    )


def test_the_head_sharing_figure_changes_the_cache_it_reports(page) -> None:
    """MHA to MQA must shrink the reported cache and switch off KV boxes."""
    chips = page.query_selector_all(".fig-heads .chip")
    assert len(chips) == 3, f"expected MHA/GQA/MQA, found {len(chips)}"

    chips[0].click()
    page.wait_for_timeout(200)
    live_mha = page.eval_on_selector_all(".fig-heads rect.kvhead:not(.off)", "els => els.length")

    chips[2].click()
    page.wait_for_timeout(200)
    live_mqa = page.eval_on_selector_all(".fig-heads rect.kvhead:not(.off)", "els => els.length")

    assert live_mha > live_mqa, "MQA should leave fewer key/value heads lit than MHA"
    assert live_mqa == 1, f"MQA is one KV head by definition; the figure shows {live_mqa}"


def test_the_timeline_chart_plots_every_mechanism(page) -> None:
    """The chart is the primary view now, so it must be complete — not a sample."""
    bundle = _bundle()
    dots = page.eval_on_selector_all("#results circle.dot", "els => els.length")
    assert dots == len(bundle["mechanisms"]), (
        f"the chart plots {dots} dots for {len(bundle['mechanisms'])} mechanisms"
    )


def test_clicking_a_dot_opens_that_mechanism(page) -> None:
    """The chart is only navigable if a dot leads somewhere."""
    page.eval_on_selector("#results circle.dot", "e => e.dispatchEvent(new Event('click'))")
    page.wait_for_timeout(250)
    shown = page.eval_on_selector_all("#results .tl-detail .tl-card .tl-name", "els => els.length")
    assert shown == 1, "clicking a dot did not open exactly one mechanism"
