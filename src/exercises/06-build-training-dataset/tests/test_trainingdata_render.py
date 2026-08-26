"""The page as a reader sees it, in a real browser.

**`node --check` proves a file has no syntax error and nothing more.** A call to an undefined
function, a figure that renders every mark the same colour, and a headline reading `0` all parse
perfectly. Exercise 03 learned this the expensive way; this file is that pattern applied here.

The tests that matter are not "the page loads". They are: the interaction **changes what the reader
sees** — because if it does not, the page is decoration and every claim on it is unproven — and the
numbers on screen are the ones the run actually produced.

Playwright is integration-marked and skips without a browser, so a fresh checkout still works. One
time setup: `uv run playwright install chromium`.
"""

import functools
import http.server
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC = REPO_ROOT / "public"
SLUG = "06-build-training-dataset"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def page():
    """The built page, served over HTTP and rendered.

    **Served, not opened as a file.** ES modules refuse to load over `file://` and the shell links
    `/_shared/tokens.css` from the site root, so a `file://` test renders a blank page with two
    CORS errors and passes any assertion that only checks the title.

    Yields:
        The Playwright page.
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
            view.wait_for_selector("section#replay", timeout=10_000)
            view.console_problems = problems
            yield view
            browser.close()
    finally:
        httpd.shutdown()


def test_the_page_renders_without_a_single_console_error(page) -> None:
    """A page that throws on load still shows its title.

    Which is what makes this worth asserting rather than assuming.
    """
    assert not page.console_problems, page.console_problems


def test_every_chapter_built(page) -> None:
    """`buildPage` catches a failing chapter and renders `.err` in its place, rather than dying.

    That is the right behaviour and it means a broken chapter is invisible unless something looks.
    """
    assert page.eval_on_selector_all(".err", "els => els.map(e => e.textContent)") == []
    ids = page.eval_on_selector_all("section", "els => els.map(e => e.id)")
    assert ids == ["summary", "replay", "floors", "chain"]


def test_each_chapter_title_is_a_claim_not_a_topic(page) -> None:
    """EXPLAINER_PROMPT §2①. A topic invites skimming; a claim invites checking."""
    titles = page.eval_on_selector_all(
        "section h2", "els => els.map(e => e.textContent.replace('#','').trim())"
    )
    assert len(titles) == 3
    for title in titles:
        # A claim has a verb. "Packing", "The ledger", "Selection" do not.
        assert any(
            word in title.lower()
            for word in (" is ", " means ", " held ", " can ", " cannot ", " read ")
        ), f"not a claim: {title}"


# --- the interaction is the argument -------------------------------------------------------------


def test_advancing_a_chapter_changes_what_the_reader_sees(page) -> None:
    """**The test the whole design rests on.**

    If moving between states leaves the figure identical, the interaction is decoration and every
    claim the page makes is unproven — and the page would look completely normal. So this reads the
    headline number and the marks at the first state, moves to the last, and requires both to move.
    """
    figure = "section#replay"
    steps = page.query_selector_all(f"{figure} .step")
    assert len(steps) >= 3

    steps[0].focus()
    page.wait_for_timeout(120)
    first_big = page.inner_text(f"{figure} .fig-big")
    first_red = page.eval_on_selector(
        f"{figure} .strip", "el => el.querySelectorAll('.hit').length"
    )

    steps[2].focus()
    page.wait_for_timeout(120)
    third_big = page.inner_text(f"{figure} .fig-big")
    third_red = page.eval_on_selector(
        f"{figure} .strip", "el => el.querySelectorAll('.hit').length"
    )

    assert first_big != third_big, "the headline number did not move between states"
    assert first_red != third_red, "the marks did not move between states"


def test_the_diverged_state_turns_every_mark_red(page) -> None:
    """Chapter 1's argument, asserted on the marks rather than on the prose.

    The claim is that recomputing after a planner change reproduces *nothing*. If the state showed
    a few red marks, the page would be arguing something much weaker than it says.
    """
    steps = page.query_selector_all("section#replay .step")
    steps[2].focus()
    page.wait_for_timeout(120)

    total, red = page.eval_on_selector(
        "section#replay .strip",
        "el => [el.children.length, el.querySelectorAll('.hit').length]",
    )
    assert red == total, f"the diverged state shows {red} of {total} red"
    assert page.inner_text("section#replay .fig-big").strip() == "0"


def test_the_bounded_claim_turns_exactly_one_mark_red(page) -> None:
    """And the last state's argument is *precision*, which is the opposite failure.

    One flipped bit in one shard must turn one microbatch red — not all of them, which would make
    the check useless, and not none, which would make it absent.
    """
    steps = page.query_selector_all("section#replay .step")
    steps[-1].focus()
    page.wait_for_timeout(120)

    red = page.eval_on_selector("section#replay .strip", "el => el.querySelectorAll('.hit').length")
    assert red == 1, f"the bounded state shows {red} red marks, not 1"


def test_every_step_is_reachable_by_keyboard(page) -> None:
    """Deleting the control row also deletes what a keyboard would land on (EXPLAINER_PROMPT §18).

    Focus is the interaction here, so a step that cannot be focused is a state the reader cannot
    reach at all without a pointer.
    """
    for section in ("replay", "floors", "chain"):
        indexes = page.eval_on_selector_all(
            f"section#{section} .step", "els => els.map(e => e.tabIndex)"
        )
        assert indexes and all(i >= 0 for i in indexes), f"{section} has unfocusable steps"


# --- what the reader is told ---------------------------------------------------------------------


def test_the_numbers_on_screen_come_from_the_run(page) -> None:
    """The summary strip must show the run's own figures, not placeholders.

    A page that rendered zeros would look like a working page reporting a broken system, which is
    strictly worse than failing to render.
    """
    values = page.eval_on_selector_all(".summary-v", "els => els.map(e => e.textContent.trim())")
    assert len(values) == 4
    assert all(v and v not in {"0", "—", "null", "undefined", "NaN"} for v in values), values


def test_no_headline_number_is_missing_or_broken(page) -> None:
    """`renderNumber` prints an em dash for a null. One in a headline is a hole in the evidence."""
    for section in ("replay", "floors", "chain"):
        big = page.inner_text(f"section#{section} .fig-big").strip()
        assert big and big not in {"—", "NaN", "undefined", "null"}, f"{section}: {big!r}"


def test_every_chapter_states_what_it_cannot_prove(page) -> None:
    """The standing caveat, always visible and never behind a disclosure.

    AGENTS.md: a limitation a reader has to open a drawer to find is a limitation the page is
    hiding. `.fig-rail` is the one element that never changes with state.
    """
    for section in ("replay", "floors", "chain"):
        rail = page.inner_text(f"section#{section} .fig-rail").strip()
        assert len(rail) > 80, f"{section} has no standing caveat: {rail!r}"


def test_the_vocabulary_is_defined_on_the_page_itself(page) -> None:
    """AGENTS.md: if a page's jargon is only defined in a Markdown file, it is not defined.

    A deployed page is read far more often than any README, and a reader who does not know what a
    microbatch is cannot check a single claim on it.
    """
    defined = page.eval_on_selector_all(".term", "els => els.map(e => e.dataset.def)")
    assert len(defined) >= 6, f"only {len(defined)} terms are explained on the page"
    assert all(d and len(d) > 30 for d in defined), "a term carries no definition"


def test_red_marks_only_the_excluded_thing(page) -> None:
    """EXPLAINER_PROMPT §5: red is reserved, and reserving it is what makes it mean something.

    Asserted by counting: if red were decorative it would appear on most marks in most states.
    """
    steps = page.query_selector_all("section#floors .step")
    steps[0].focus()
    page.wait_for_timeout(120)
    total, red = page.eval_on_selector(
        "section#floors .strip", "el => [el.children.length, el.querySelectorAll('.hit').length]"
    )
    assert 0 < red < total / 2, f"{red} of {total} marks are red; red has stopped meaning refused"


def test_the_page_does_not_scroll_sideways(page) -> None:
    """An invisible absolutely-positioned tooltip pushed exercise 04's page 312px sideways.

    Nothing in the source looks wrong when that happens, and it is only visible in a browser.
    """
    overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 0, f"the page scrolls {overflow}px sideways"


def test_the_page_survives_a_narrow_viewport(page) -> None:
    """The figure takes `order: -1` on narrow screens so it sits above the prose, not below all of
    it. If that regressed, a phone reader would scroll past the whole argument to reach the mark
    it is about."""
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(200)
    try:
        overflow = page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 1, f"the page scrolls {overflow}px sideways at 390px wide"
        assert page.is_visible("section#replay .fig-big")
    finally:
        page.set_viewport_size({"width": 1280, "height": 900})


# --- geometry a reader sees and a DOM assertion does not -----------------------------------------


@pytest.mark.parametrize("width", [600, 760, 1024, 1280, 1418, 1600])
def test_nothing_collides_with_the_summary_rules(page, width: int) -> None:
    """**The bug fourteen passing tests missed, because none of them looked at geometry.**

    Each summary cell carries a `border-top` hairline, and the lede's action buttons sat directly
    above with **zero** vertical gap at every width tested. So the rules rendered flush against two
    rounded pills: the two the buttons covered read as broken while the two beside them read as
    fine, and the page looked misaligned rather than tight.

    Nothing in the DOM was wrong. Every element existed, every number was right, no console error
    fired — the defect was entirely in where the boxes landed, which is only visible by measuring
    them or by looking.
    """
    page.set_viewport_size({"width": width, "height": 900})
    page.wait_for_timeout(150)
    try:
        gap = page.evaluate("""() => {
            const btns = [...document.querySelectorAll('.lede-actions .jump')];
            const cells = [...document.querySelectorAll('.summary-cell')];
            let smallest = Infinity;
            for (const b of btns) {
              const A = b.getBoundingClientRect();
              for (const c of cells) {
                const B = c.getBoundingClientRect();
                const overlapX = Math.min(A.right, B.right) - Math.max(A.left, B.left);
                if (overlapX > 0) smallest = Math.min(smallest, B.top - A.bottom);
              }
            }
            return smallest === Infinity ? null : Math.round(smallest);
        }""")
        assert gap is not None, "the buttons and the summary no longer share any column"
        assert gap >= 12, (
            f"at {width}px the summary rules sit {gap}px below the action buttons. A hairline "
            f"register needs air above it, or it reads as attached to whatever it touches."
        )
    finally:
        page.set_viewport_size({"width": 1280, "height": 900})


def test_the_summary_rules_all_sit_on_one_line(page) -> None:
    """Four cells in a row must share a baseline, or the register reads as broken rather than tight.

    A grid with `auto-fit` can silently wrap one cell onto a second row at an awkward width, and
    three-on-top-one-below is exactly what "dislocated" looks like to a reader.
    """
    tops = page.eval_on_selector_all(
        ".summary-cell", "els => els.map(e => Math.round(e.getBoundingClientRect().top))"
    )
    assert len(tops) == 4
    assert len(set(tops)) == 1, f"the four summary cells are on {len(set(tops))} rows: {tops}"
