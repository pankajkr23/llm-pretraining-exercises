"""Running prose holds a reading measure, on every page that already does — and on new ones.

**Exercise 08 has measured its own prose since it was built; nothing else ever has.** The standard
is `docs/DESIGN.md`'s: the design targets 46 characters a line and above, the guard fails below 42,
and the gap between those two numbers is deliberately there to leave room. The upper bound is 80.

This promotes that measurement out of exercise 08 and points it at the deployable set. The method is
copied from `08-.../tests/test_attention_measures.py` on purpose, down to the probe span: a
character width read from a `ch` unit, a `canvas.measureText`, or an assumed average glyph is a
*different* measurement, and two pages measured differently cannot be compared.

**The failure it was written for is a shared component, and it was invisible at every width anyone
checks.** `explainer.css`'s step strip put a fixed 296px figure column beside the prose and
collapsed to one column at `max-width: 760px` — while an iPad portrait is 768 and a common Android
tablet is 800. In that gap a step's paragraph ran **29 characters a line**. It also ran 41 at
exactly 1180px, and only there, because that is the width at which `page.css` begins reserving a
260px rail gutter, so the content box is *narrower* at 1180 than at 1179.
"""

import functools
import http.server
import os
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC = REPO_ROOT / "public"

pytestmark = pytest.mark.integration

#: `docs/DESIGN.md`'s band. The design targets 46 and up; the guard fails below 42.
NARROWEST, WIDEST = 42, 80

#: A short block is only a defect if it is ALSO leaving room unused, and dropping this qualifier is
#: how the first version of this file failed exercise 08 — the page it was copied from. At 390px a
#: standfirst is 29 characters because the phone is 390px wide, not because anything is wrong; the
#: same 29 characters beside an empty half-column is the defect. Copy a measurement and you must
#: copy its condition, or you have written a different test with the same numbers in it.
ROOM_TO_SPARE = 0.70

#: Widths chosen for where the layout actually changes, not for round numbers: 1180 is where the
#: rail gutter appears, 900 and 768 straddle the step strip's collapse, 390 is a phone.
WIDTHS = (2560, 1920, 1440, 1180, 1024, 900, 768, 390)

#: The exercises held to the band **today**. Two deployable pages are deliberately absent and both
#: have a fix in flight — 04 (#117) and 05 (#118) — measured here at 125 and 149 characters.
#:
#: They are listed as *not covered* rather than as ledgered exceptions, and that is a deliberate
#: trade. An exception ledger that fails in both directions is this repo's usual idiom and it is
#: the wrong one here: it would go red the moment either fix merges, in a pull request that has
#: nothing to do with this file, and a guard that reds on someone else's correct work is one that
#: gets edited rather than read. The cost is that widening this list is a manual step; the guard
#: below makes the list impossible to shrink by accident, which is the failure that actually costs
#: coverage.
COVERED = [
    "01-introductions",
    "02-tokenization",
    "03-data-collection-framework",
    "04-data-cleaning-dedup",
    "06-build-training-dataset",
    "07-model-embeddings-internals",
    "08-modern-attention-variants",
    "09-loss-functions-output-heads",
    "10-training-loop",
]

#: Deployable, measured, and not yet in the band. Each names what is fixing it.
NOT_YET_COVERED = {
    "05-datamixtures-and-curriculum": "#118 — measured 149 characters at 2560px",
}

MEASURE_JS = """() => {
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre';
  document.body.appendChild(probe);
  const chw = (el) => {
    const cs = getComputedStyle(el);
    probe.style.font = cs.font || (cs.fontSize + ' ' + cs.fontFamily);
    probe.textContent = '0'.repeat(100);
    return probe.getBoundingClientRect().width / 100;
  };
  const out = [], seen = new Set();
  const sel = '#main p, #main li, #main figcaption, #main .say, #main .preamble-row p';
  for (const el of document.querySelectorAll(sel)) {
    const txt = el.innerText.trim();
    if (txt.length < 90) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 4) continue;
    const cols = parseInt(getComputedStyle(el).columnCount) || 1;
    const key = el.className + '|' + Math.round(r.width) + '|' + txt.slice(0, 24);
    if (seen.has(key)) continue;
    seen.add(key);
    let host = el.parentElement;
    let hw = host ? host.getBoundingClientRect().width : 0;
    while (host && hw <= r.width + 2 && host !== document.body) {
      host = host.parentElement;
      hw = host ? host.getBoundingClientRect().width : 0;
    }
    out.push({
      cls: (el.className || el.tagName).toString().slice(0, 30),
      chars: Math.round((r.width / cols) / chw(el)),
      fill: hw ? r.width / hw : 1,
    });
  }
  probe.remove();
  return out;
}"""


def _deployable() -> set[str]:
    return {p.parent.parent.name for p in REPO_ROOT.glob("src/exercises/*/web/index.html")}


@pytest.fixture(scope="module")
def site():
    if not (PUBLIC / "_shared" / "tokens.css").is_file():
        pytest.skip("run deploy/vercel/build.sh first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # pragma: no cover - environment, not logic
                if os.environ.get("CI"):
                    pytest.fail(f"chromium did not launch on CI: {exc}")
                pytest.skip(f"chromium unavailable: {exc}")
            yield browser, f"http://127.0.0.1:{httpd.server_address[1]}"
            browser.close()
    finally:
        httpd.shutdown()


def _measure(site, slug: str, width: int) -> list[dict]:
    browser, base = site
    ctx = browser.new_context(viewport={"width": width, "height": 950})
    page = ctx.new_page()
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(500)
        return page.evaluate(MEASURE_JS)
    finally:
        ctx.close()


@pytest.mark.parametrize("slug", COVERED)
def test_prose_holds_a_reading_measure_at_every_width(site, slug: str) -> None:
    """One case per exercise, sweeping the widths inside it.

    Sweeping inside rather than parametrising over widths as well is what caught the step strip:
    the failure existed at 768 and 1180 and nowhere else, so a report naming only the exercise
    would have sent someone to look at a page that is correct at the width they opened it.
    """
    blocks, wide, narrow = 0, [], []
    for width in WIDTHS:
        for row in _measure(site, slug, width):
            blocks += 1
            if row["chars"] > WIDEST:
                wide.append((width, row["cls"], row["chars"]))
            elif row["chars"] < NARROWEST and row["fill"] < ROOM_TO_SPARE:
                narrow.append((width, row["cls"], row["chars"]))

    if slug in {"01-introductions", "02-tokenization"}:
        # These two predate `#main` and the spine entirely; the selector finds nothing, which is a
        # fact about them rather than a pass. Asserted below so it cannot become true of the others.
        assert blocks == 0, f"{slug} now has measurable prose ({blocks} blocks) — add it properly"
        return

    assert blocks >= 100, (
        f"only {blocks} block(s) of prose were measured on {slug} across {len(WIDTHS)} widths. "
        "The selector has probably stopped matching, and everything below is passing over almost "
        "nothing."
    )
    report = "\n".join(
        f"    at {w:>4}px  {cls:<30} {n:>3} chars" for w, cls, n in (wide + narrow)[:12]
    )
    assert not wide and not narrow, (
        f"{slug}: {len(wide)} block(s) wider than {WIDEST} characters and {len(narrow)} narrower "
        f"than {NARROWEST}, out of {blocks} measured.\n{report}\n\n"
        "`docs/DESIGN.md`: the design targets 46 characters and above. A block that is too narrow "
        "is usually a fixed-width neighbour taking the space rather than the prose asking for it."
    )


def test_the_uncovered_pages_are_still_the_two_expected() -> None:
    """`COVERED` must not be able to shrink quietly, and `NOT_YET_COVERED` must not rot.

    The list above is hand-maintained, which is the weakness of the trade it documents. This is the
    half that makes it safe: every deployable page is in exactly one of the two lists, so a new
    exercise lands in neither and turns this red — which is the whole point, since a page nobody
    listed is a page nobody measures.
    """
    deployable = _deployable()
    accounted = set(COVERED) | set(NOT_YET_COVERED)
    assert accounted == deployable, (
        f"  deployable but in neither list: {sorted(deployable - accounted)}\n"
        f"  listed but not deployable:      {sorted(accounted - deployable)}\n\n"
        "Add a new exercise to COVERED once it holds the measure, or to NOT_YET_COVERED naming "
        "what will fix it. Never leave it out of both."
    )
    assert len(COVERED) >= 6, (
        f"COVERED has shrunk to {len(COVERED)}: {COVERED}. Removing an exercise from this list "
        "silently drops it from the sweep — if a page genuinely regressed, fix the page."
    )


def test_the_measure_check_can_actually_fail(site) -> None:
    """Break it on purpose: widen a paragraph past the band and watch it report.

    The mutation is in the browser on a throwaway context, so nothing on disk is touched and there
    is no backup to restore — `AGENTS.md`'s rule about backups living outside the tree is satisfied
    by there being no backup at all.
    """
    browser, base = site
    ctx = browser.new_context(viewport={"width": 1440, "height": 950})
    page = ctx.new_page()
    try:
        page.goto(f"{base}/{COVERED[-1]}/index.html", wait_until="networkidle", timeout=25_000)
        page.evaluate(
            """() => document.querySelectorAll('#main p').forEach((el) => {
                 el.style.maxWidth = 'none';
                 el.style.width = '3000px';
               })"""
        )
        page.wait_for_timeout(200)
        rows = page.evaluate(MEASURE_JS)
        assert rows, "no prose was measured at all, so this twin proves nothing"
        assert any(r["chars"] > WIDEST for r in rows), (
            f"paragraphs forced to 3000px still measured at most "
            f"{max(r['chars'] for r in rows)} characters, which the checker did not report — so it "
            "cannot detect the failure it exists for"
        )
    finally:
        ctx.close()


# --------------------------------------------------------------------------- the step strip

#: The shared step strip, which the guard above provably cannot see.
#:
#: **This was found by watching that guard fail to fail.** Restored to the shipped stylesheet — the
#: one whose paragraphs ran 29 characters at 768px — every assertion above stayed green, because
#: `ROOM_TO_SPARE` asks whether a block is leaving room *inside its own box* and a `.step`
#: paragraph fills its box completely. The box is the thing that is too narrow. Widening the host
#: walk to the page would fix that and break every legitimate two-column layout, including the one
#: on the page the measurement was copied from, so the honest answer is a second, narrower guard
#: that asserts the component's own property rather than a looser version of the general one.
STEP_JS = """() => {
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre';
  document.body.appendChild(probe);
  const out = [];
  for (const p of document.querySelectorAll('.scrolly .step p')) {
    const txt = p.innerText.trim();
    if (txt.length < 90) continue;
    const cs = getComputedStyle(p);
    probe.style.font = cs.font || (cs.fontSize + ' ' + cs.fontFamily);
    probe.textContent = '0'.repeat(100);
    const ch = probe.getBoundingClientRect().width / 100;
    const w = p.getBoundingClientRect().width;
    if (w < 4) continue;
    // Fill against the PAGE, not against the paragraph's own box — the component's whole design
    // question is how much of the page the prose column is given, and its own box is exactly the
    // thing that was too small. This is the widened host walk the general guard cannot use.
    const main = p.closest('#main') || document.body;
    const mw = main.getBoundingClientRect().width;
    out.push({ chars: Math.round(w / ch), fill: mw ? w / mw : 1, text: txt.slice(0, 34) });
  }
  probe.remove();
  return out;
}"""

#: Every exercise whose page builds a step strip: the ones that IMPORT the shared explainer, not
#: the ones that vendor its stylesheet. All six deployable bundles carry `_shared/explainer.css`
#: and only two ever call `makeExplainer`, which is the same gap `AGENTS.md` records for the rest
#: of that directory. Detected rather than listed, so a third exercise adopting the component joins
#: this sweep by adopting it.
STEPPED = sorted(
    d.parent.name
    for d in REPO_ROOT.glob("src/exercises/*/web")
    if any("_shared/explainer.js" in f.read_text() for f in d.glob("*.js"))
)


@pytest.mark.parametrize("slug", STEPPED)
def test_a_step_never_reads_at_a_squeezed_measure(site, slug: str) -> None:
    """The step strip's prose column holds a reading measure at every width, or it is one column.

    Two columns are worth having only where the left one can still be read. Below the collapse
    breakpoint there is one column and the paragraph has the page; above it, the grid must give the
    prose enough room. The failure mode this exists for is the gap between those two claims.
    """
    bad = []
    seen = 0
    for width in WIDTHS:
        browser, base = site
        ctx = browser.new_context(viewport={"width": width, "height": 950})
        page = ctx.new_page()
        try:
            page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
            page.wait_for_timeout(500)
            for row in page.evaluate(STEP_JS):
                seen += 1
                # A phone cannot fit 42 characters at this font size whatever the layout does, so
                # the claim is only made where the page had room to give and did not.
                if row["chars"] < NARROWEST and row["fill"] < ROOM_TO_SPARE:
                    bad.append((width, row["chars"], row["text"]))
        finally:
            ctx.close()

    assert seen >= 20, (
        f"only {seen} step paragraph(s) were measured on {slug}. The selector "
        "`.scrolly .step p` has stopped matching, and this assertion is passing over nothing."
    )
    assert not bad, (
        f"{slug}: {len(bad)} step paragraph(s) under {NARROWEST} characters a line, out of "
        f"{seen}:\n"
        + "\n".join(f"    at {w:>4}px  {n:>3} chars  {txt}…" for w, n, txt in bad[:8])
        + "\n\nThe two-column step strip keeps a FIXED-width figure column, so every pixel the "
        "window loses comes out of the prose. Either collapse to one column sooner, or give the "
        "prose column a floor."
    )


def test_at_least_two_exercises_build_a_step_strip() -> None:
    """`STEPPED` is globbed, so an empty one would make the guard above vacuous."""
    assert len(STEPPED) >= 2, (
        f"only {len(STEPPED)} exercise(s) were found building a step strip: {STEPPED}. "
        "The detection has probably stopped matching `chapters.js`."
    )
