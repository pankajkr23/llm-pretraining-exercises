"""What the widget does when a browser actually runs it.

`node --check` proves `encoder.js` parses. It cannot prove the page imports it, that the import
resolves at the URL Vercel serves it from, or that a click handler calls a function that exists —
all of which are valid syntax and all of which render a blank panel. The paste-box encoder is the
part a grader is told to use, so it is the part that has to be loaded and driven, not inspected.

Marked integration: needs a browser, and is slower than the rest of the suite by an order of
magnitude. Skipped rather than failed when Playwright or its browser is absent, so a checkout
without ``uv run playwright install chromium`` still runs everything else — which also means this
protects you silently or not at all.

Run: ``uv run pytest -m integration``
"""

import http.server
import json
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"

pytest.importorskip("playwright", reason="playwright is not installed")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (WEB / "data.json").exists(), reason="web/data.json not built"),
]

WIDTHS = (1500, 900, 390)


@contextmanager
def _serve(root: Path) -> Iterator[str]:
    """Serve a directory over http — ES module imports do not resolve from file://."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):  # noqa: D102 - silence per-request logging
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}/"
        finally:
            httpd.shutdown()


@contextmanager
def _page(width: int = 1500, path: str = ""):
    """One loaded page, with every console error collected."""
    with _serve(WEB) as base, sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except PlaywrightError as exc:  # pragma: no cover - environment, not logic
            pytest.skip(f"no chromium available: {exc}")
        page = browser.new_page(viewport={"width": width, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base + path, wait_until="networkidle")
        page.wait_for_timeout(500)
        try:
            yield page, errors
        finally:
            browser.close()


def _bundle() -> dict:
    return json.loads((WEB / "data.json").read_text(encoding="utf-8"))


def test_the_page_renders_the_score_and_every_language():
    data = _bundle()
    first = data["profiles"][0]["name"]
    expected = sum(1 for c in data["configs"] if c["profile"] == first)
    assert expected, "no configs in the opening section, so this test proves nothing"

    with _page() as (page, errors):
        assert not errors, f"the page threw: {errors[:3]}"
        assert page.locator("#err").inner_text().strip() == "", "the page showed its load error"
        assert page.locator("#selector button").count() == expected
        # Four languages, and a score that is a number rather than NaN/undefined.
        assert page.locator("#app tbody tr").count() == 4
        score = page.locator(".score").inner_text().replace(",", "")
        assert float(score) > 0, f"score panel reads {score!r}"


def test_the_two_measurements_are_never_shown_as_one_ranked_list():
    """v1 and v2 are denominated differently; one list across them would be meaningless."""
    data = _bundle()
    profiles = [p["name"] for p in data["profiles"]]
    assert len(profiles) > 1, "only one profile exported, so this test proves nothing"

    with _page() as (page, errors):
        assert not errors, f"the page threw: {errors[:3]}"
        assert page.locator("#profiles button").count() == len(profiles)
        for name in profiles:
            page.locator(f'#profiles button[data-p="{name}"]').click()
            page.wait_for_timeout(300)
            assert not errors, f"switching to {name} threw: {errors[:3]}"
            shown = page.locator("#selector button").count()
            expected = sum(1 for c in data["configs"] if c["profile"] == name)
            assert shown == expected, f"{name} shows {shown} tabs, expected {expected}"
            # Every section says, in words, that its numbers do not travel.
            assert "cannot be compared" in page.locator("#profilenote").inner_text()


def test_each_section_labels_the_denominator_it_is_scored_in():
    """A column header reading `units` over word counts is how the two get conflated."""
    data = _bundle()
    with _page() as (page, errors):
        for p in data["profiles"]:
            page.locator(f'#profiles button[data-p="{p["name"]}"]').click()
            page.wait_for_timeout(300)
            assert not errors, f"the page threw on {p['name']}: {errors[:3]}"
            headers = page.locator("#app thead th").all_inner_texts()
            assert p["denominator"] in [h.strip().lower() for h in headers], (
                f"{p['name']} table headers {headers} do not name its denominator"
            )


def test_every_tokenizer_in_a_section_is_measured_on_the_same_languages():
    """Apples to apples: same languages, in the same order, on every tab of a section.

    They always *were* the same four languages — but the table used to be sorted by fertility, so
    the rows reshuffled between tabs and two tokenizers measured identically looked like they had
    been measured on different things. Order is part of the comparison, not decoration.
    """
    data = _bundle()
    for name in {c["profile"] for c in data["configs"]}:
        rows = [c for c in data["configs"] if c["profile"] == name]
        orders = {tuple(lang["code"] for lang in c["languages"]) for c in rows}
        assert len(orders) == 1, f"{name} shows languages in {len(orders)} different orders"
        # And each row is flagged, so best/worst survive the fixed ordering.
        for c in rows:
            assert sum(lang["is_best"] for lang in c["languages"]) == 1
            assert sum(lang["is_worst"] for lang in c["languages"]) == 1


def test_non_benchmark_rows_show_what_they_moved_against_the_benchmark():
    """A score alone cannot tell you which language paid for the improvement."""
    data = _bundle()
    v2 = [c for c in data["configs"] if c["profile"] == "v2"]
    benchmark = next(c for c in v2 if c["is_reference"])
    assert all(lang["delta"] is None for lang in benchmark["languages"])

    submission = next(c for c in v2 if c["is_submission"])
    deltas = {lang["code"]: lang["delta"] for lang in submission["languages"]}
    assert all(d is not None for d in deltas.values()), "submission has no comparison to benchmark"
    # Maithili is the language the submission set out to rescue, so its delta must be negative
    # (fertility is tokens per unit — lower is better).
    assert deltas["mai"] < 0, f"Maithili should have improved, got {deltas['mai']}"


def test_every_tokenizer_explains_itself():
    """A row of numbers with no story is not a finding.

    Each tokenizer on the page carries a short what/why/outcome note and a badge saying whether it
    is the reference, the submission, a rejected experiment or an ablation. This fails if a new
    config ever reaches the page unexplained — including the rejected one, whose whole reason for
    being there is that a reader would otherwise never know a higher score was found and refused.
    """
    data = _bundle()
    assert all(c.get("blurb") for c in data["configs"]), "a config shipped without an explanation"
    rejected = [c for c in data["configs"] if c.get("is_rejected")]
    assert len(rejected) == 1, "the rejected experiment is missing from the bundle"
    assert rejected[0]["adjusted"] > max(
        c["adjusted"] for c in data["configs"] if c["profile"] == "v2" and not c.get("is_rejected")
    ), "the rejected row should be the highest v2 scorer — that is the point of showing it"

    with _page() as (page, errors):
        for i in range(page.locator("#selector button").count()):
            page.locator("#selector button").nth(i).click()
            page.wait_for_timeout(250)
            assert not errors, f"tab {i} threw: {errors[:3]}"
            blurb = page.locator(".blurb").first.inner_text()
            assert len(blurb) > 80, f"tab {i} explanation is too thin: {blurb!r}"
            assert page.locator(".blurb .badge").count() >= 1, f"tab {i} has no role badge"


def test_the_rejected_experiment_is_labelled_as_rejected():
    """It posts the biggest number on the page, so it has to say why it was not submitted."""
    with _page() as (page, errors):
        labels = page.locator("#selector button").all_inner_texts()
        idx = next(i for i, t in enumerate(labels) if "rejected" in t.lower())
        page.locator("#selector button").nth(idx).click()
        page.wait_for_timeout(300)
        assert not errors, f"the page threw: {errors[:3]}"
        assert page.locator(".badge.rej").count() == 1, "no rejected badge on the rejected config"
        blurb = page.locator(".blurb").first.inner_text().lower()
        # The reason has to be the durable one. An earlier version of this page rejected the row
        # on held-out score, which five splits later turned out to be noise — so the copy must
        # rest on total tokens, which is measured on the whole corpus and does not move.
        assert "192,713" in blurb and "189,785" in blurb, f"no token comparison given: {blurb!r}"
        assert "worse" in blurb, f"does not say what it degraded: {blurb!r}"


def test_the_paste_box_actually_tokenizes():
    """The deliverable the grader is told to use. A blank panel here is the whole failure."""
    with _page() as (page, errors):
        page.fill("#paste", "India is a country in South Asia.")
        page.wait_for_timeout(300)
        assert not errors, f"encoding threw: {errors[:3]}"
        chips = page.locator("#chips .chip").count()
        assert chips > 3, f"only {chips} tokens rendered for a full sentence"
        assert "tokens" in page.locator("#pastecount").inner_text()


def test_unknown_characters_are_shown_not_silently_dropped():
    """Dropping them is how a tokenizer flatters its own count; the chip is the whole point."""
    with _page() as (page, errors):
        page.fill("#paste", "hello 🚀 world")
        page.wait_for_timeout(300)
        assert not errors, f"encoding threw: {errors[:3]}"
        assert page.locator("#chips .chip.unk").count() > 0, "the rocket vanished instead of [UNK]"
        assert "unknown" in page.locator("#pastecount").inner_text()


def test_every_tokenizer_tab_renders():
    """A tab whose encoder is unsupported must say so, not throw or render an empty panel."""
    with _page() as (page, errors):
        for i in range(page.locator("#selector button").count()):
            page.locator("#selector button").nth(i).click()
            page.wait_for_timeout(250)
            assert not errors, f"tab {i} threw: {errors[:3]}"
            assert page.locator("#app tbody tr").count() == 4
            assert page.locator("#pastecount").inner_text().strip() != ""


@pytest.mark.parametrize("width", WIDTHS)
def test_the_page_never_scrolls_sideways(width: int):
    """Wide content scrolls inside its own container; the body never does."""
    with _page(width) as (page, errors):
        assert not errors, f"the page threw at {width}px: {errors[:3]}"
        overflow = page.evaluate(
            "() => { const d = document.documentElement;"
            " return d.scrollWidth > d.clientWidth ? [d.scrollWidth, d.clientWidth] : null; }"
        )
        assert not overflow, f"{width}px scrolls sideways: {overflow[0]}px into {overflow[1]}px"


# -- the explainer subpage ---------------------------------------------------------------------
#
# The long-form explanation lives at its own URL so the landing page stays a tool. That split is
# only safe if both halves are loaded: a broken link or a figure that never renders would leave
# the tool intact and the argument silently missing.

EXPLAINER = "how-it-works.html"


def test_the_landing_page_links_to_the_explainer():
    with _page() as (page, errors):
        assert not errors, f"the page threw: {errors[:3]}"
        link = page.locator(f'a[href="./{EXPLAINER}"]')
        assert link.count() >= 1, "no link from the tool to the explanation"
        assert link.first.is_visible(), "the link to the explanation is not visible"


def test_the_explainer_page_renders_all_three_figures():
    """Three figures, each driven by explainer.json. A blank one is the whole failure."""
    with _page(path=EXPLAINER) as (page, errors):
        assert not errors, f"the explainer threw: {errors[:3]}"
        assert page.locator("#err").inner_text().strip() == "", "the explainer showed a load error"
        # Fig. 0 — the corpus, one row per language.
        assert page.locator("#mixbody tr").count() == 4
        # Fig. 1 — the dial, with all three numbers populated.
        for slot in ("#maiscore", "#maiunseen", "#maitokens"):
            assert page.locator(slot).inner_text().strip(), f"{slot} is empty"
        assert page.locator("#maibars .bfill").count() == 4
        # Fig. 2 — one bar per recipe.
        assert page.locator("#exambars .bfill").count() == 3
        for note in ("#mixnote", "#mainote", "#examnote"):
            assert len(page.locator(note).inner_text()) > 80, f"{note} is too thin"


def test_dragging_the_dial_changes_the_figure():
    """Removing the interaction would destroy the argument, so the interaction must work."""
    with _page(path=EXPLAINER) as (page, errors):
        dial = page.locator("#maidial")
        before = page.locator("#maiscore").inner_text()
        dial.fill(str(int(dial.get_attribute("max"))))
        page.wait_for_timeout(300)
        assert not errors, f"the dial threw: {errors[:3]}"
        after = page.locator("#maiscore").inner_text()
        assert before != after, "the score did not change when the dial moved"
        assert "overshoot" in page.locator("#maiverdict").inner_text().lower()


def test_changing_the_held_out_slice_changes_the_numbers():
    """Fig. 2's claim is that the ranking is unstable, so the slices must actually differ."""
    with _page(path=EXPLAINER) as (page, errors):
        buttons = page.locator("#examseg button")
        assert buttons.count() == 5, "expected five held-out slices"
        first = page.locator("#exambars .bval").first.inner_text()
        buttons.nth(3).click()
        page.wait_for_timeout(400)
        assert not errors, f"switching slice threw: {errors[:3]}"
        assert page.locator("#exambars .bval").first.inner_text() != first


@pytest.mark.parametrize("width", WIDTHS)
def test_the_explainer_never_scrolls_sideways(width: int):
    with _page(width, EXPLAINER) as (page, errors):
        assert not errors, f"the explainer threw at {width}px: {errors[:3]}"
        overflow = page.evaluate(
            "() => { const d = document.documentElement;"
            " return d.scrollWidth > d.clientWidth ? [d.scrollWidth, d.clientWidth] : null; }"
        )
        assert not overflow, f"{width}px scrolls sideways: {overflow[0]}px into {overflow[1]}px"


def test_the_huggingface_download_is_only_offered_on_the_submission():
    """A static link to tokenizer.json on every tab hands you the wrong tokenizer.

    Only the submission is exported in HuggingFace format. The link used to sit on all five tabs,
    so opening it from the rejected or from-scratch tab silently downloaded the submission's
    vocabulary instead of the one on screen — five different tokenizers, one file, no warning.
    """
    with _page() as (page, errors):
        buttons = page.locator("#selector button")
        for i in range(buttons.count()):
            buttons.nth(i).click()
            page.wait_for_timeout(250)
            assert not errors, f"tab {i} threw: {errors[:3]}"
            label = buttons.nth(i).inner_text()
            note = page.locator(".panel", has_text="Take it with you").inner_text().lower()
            if "\u2713" in label:  # the submitted tab
                assert "only the submitted tokenizer" in note
            else:
                # Every other tab must say the file is not this tokenizer.
                assert "not the submission" in note, f"tab {label!r} does not disown tokenizer.json"


def test_the_default_sample_makes_the_tokenizers_disagree():
    """The paste box's default text must separate the tabs, not flatter them.

    It used to be a plain sentence of common words, which every tokenizer here splits into exactly
    16 tokens — because the frequent end of a 10,000-token vocabulary is identical across these
    recipes and they only differ in the rare 2-12%. A reader switching tabs saw the same number
    every time and reasonably concluded there was one tokenizer behind all five. There is not.
    """
    with _page() as (page, errors):
        counts = []
        buttons = page.locator("#selector button")
        for i in range(buttons.count()):
            buttons.nth(i).click()
            page.wait_for_timeout(300)
            assert not errors, f"tab {i} threw: {errors[:3]}"
            text = page.locator("#pastecount").inner_text()
            if "tokens" in text:
                counts.append(int(text.split()[0]))
        assert len(counts) >= 2, "fewer than two tabs could encode, so this proves nothing"
        assert len(set(counts)) > 1, (
            f"every tab tokenizes the default text into {counts[0]} tokens — the sample cannot "
            "show that these are different tokenizers"
        )
