"""A label inside a `viewBox` is as small as the drawing was scaled, not as small as it was written.

**This rule already existed, twice, and neither copy could see the exercise next door.**
`test_no_svg_label_renders_too_small_to_read` lives in exercise 05's render test and again in
exercise 07's — the same measurement, written twice, with different plumbing and slightly different
prose. Both pages are clean: the smallest label on 05 paints at 9.6px and on 07 at 9.7px. Exercise
08 has no such guard at all, and 09 and 10 have only
`test_no_svg_label_spills_outside_its_own_figure`, which is about **position**. Measured at 390px on
the day this was written:

| page | labels | under the floor | smallest |
| --- | ---: | ---: | ---: |
| 08 | 175 | 145 | **2.4px** |
| 09 | 48 | 48 | 4.6px |
| 10 | 29 | 29 | 5.0px |

**And 08 fails on a desktop too**, which no phone breakpoint can fix: its `results` plate declares a
1440-unit viewBox and is painted into 1072px, so 86 labels land at 6.7px at a 1440px viewport, and
its `mechanism` plate puts 79 more at 8.2px.

**Nothing lexical can catch this and that is the whole point.** Every one of those labels declares a
compliant size — `docs/DESIGN.md` puts mono micro-labels at `9.5–11px` fixed and says they are
*"deliberately not scaled: they are furniture, not reading, and they stop being legible if they
shrink"*. The `viewBox` shrinks them anyway, silently, by a factor nothing in the source states. So
the measurement has to be `declared × (rendered width ÷ viewBox width)`, taken in a browser.

The fourth single-exercise guard this month to miss its neighbours, so it is repo-wide now and finds
its pages on the filesystem.
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

#: `docs/DESIGN.md`'s own floor for a mono micro-label. Both per-exercise copies of this rule chose
#: the same number independently, which is the best evidence it is the right one.
LEGIBLE = 9.5

#: 1440 is a desktop, 390 a current handset, 320 the narrowest still worth supporting. 768 is in
#: because it is where a figure typically stops being full-bleed and starts being scaled, and a
#: defect that appears only between two tested widths is one nobody finds. 2560 comes from exercise
#: 05's copy of this rule, which swept it where 07's did not — the union of the two, so promoting
#: them loses no width either had.
WIDTHS = (2560, 1440, 768, 390, 320)

#: Figures that may paint below the floor, each with the reason. **Empty, and an entry here is a
#: figure a reader cannot read** — so it is not the way to clear a red gate. Keyed by
#: `(page, section)`; the twin below fails when an entry stops being needed.
TOO_SMALL_ON_PURPOSE: dict[tuple[str, str], str] = {}

LABELS_JS = """() => {
  const out = [];
  for (const svg of document.querySelectorAll('svg')) {
    const vb = svg.viewBox && svg.viewBox.baseVal;
    if (!vb || !vb.width) continue;              // no viewBox: nothing is being scaled
    const box = svg.getBoundingClientRect();
    // A figure inside a closed `details`, a hidden tab or a `display: none` branch reports a zero
    // box, which divides to an effective size of 0 and would read as the worst defect on the page.
    // Neither per-exercise copy of this rule checked it; neither page has a hidden figure yet.
    if (box.width < 2) continue;
    const scale = box.width / vb.width;
    for (const t of svg.querySelectorAll('text')) {
      const txt = (t.textContent || '').trim();
      if (!txt) continue;
      if (!t.getClientRects().length) continue;  // painted nowhere
      out.push({
        painted: parseFloat(getComputedStyle(t).fontSize) * scale,
        section: (t.closest('section') || {}).id || '(none)',
        vb: Math.round(vb.width),
        box: Math.round(box.width),
        txt: txt.slice(0, 30),
      });
    }
  }
  return out;
}"""


def _deployable() -> list[str]:
    """Every deployed route, from the filesystem, so a new page is covered the day it ships.

    **Sub-routes included, because leaving them out is what this file got wrong first.** The
    original glob was `src/exercises/*/web/index.html`, which finds ten pages and misses three:
    exercise 03's `reasoning/` and `report/`, and exercise 08's `field-guide/`. The field guide
    renders the same diagrams as 08's essay page from the same module, so giving those diagrams a
    floor changed it too — and it scrolled **354px sideways at 390px** while this file reported
    every page green, because it had never loaded the page. `build.sh` does `cp -R`, so a sub-route
    ships without a build change and without anything here noticing.
    """
    return sorted(
        str(page.parent.relative_to(REPO_ROOT / "src" / "exercises")).replace("/web", "", 1)
        for page in REPO_ROOT.glob("src/exercises/*/web/**/index.html")
    )


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


def _labels(site, slug: str, width: int) -> list[dict]:
    browser, base = site
    page = browser.new_page(viewport={"width": width, "height": 950})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(500)
        return page.evaluate(LABELS_JS)
    finally:
        page.close()


@pytest.mark.parametrize("slug", _deployable())
def test_no_svg_label_renders_too_small_to_read(site, slug: str) -> None:
    """One case per page, sweeping the widths inside it.

    A page with no scaling figure passes, which is honest — most of exercises 01 to 04 have none.
    The vacuity guard below is what stops that from adding up to a sweep over nothing.
    """
    failures = []
    for width in WIDTHS:
        for label in _labels(site, slug, width):
            if (slug, label["section"]) in TOO_SMALL_ON_PURPOSE:
                continue
            if label["painted"] < LEGIBLE:
                failures.append(
                    f"{width}px — {label['section']}: {label['painted']:.1f}px "
                    f"(a {label['vb']}-unit viewBox painted into {label['box']}px) — "
                    f"{label['txt']!r}"
                )

    assert not failures, (
        f"{slug} paints {len(failures)} figure label(s) under {LEGIBLE}px:\n  "
        + "\n  ".join(failures[:10])
        + f"\n  ... and {max(0, len(failures) - 10)} more"
        + "\nThe authored `font-size` is compliant on every one of these — a `viewBox` scales the "
        "text down with the drawing, so the size a reader sees is `declared × (rendered ÷ "
        "viewBox)`. A diagram cannot reflow the way a paragraph can, so the fix is to stop it "
        "shrinking: give the svg a `min-width` and let its own container scroll, which is what "
        "`AGENTS.md` asks of wide content."
    )


def test_some_page_has_figure_labels_to_measure(site) -> None:
    """Otherwise every case above passes by having nothing to look at.

    This is the half that fails if the figures stop being svg, or the selector rots — at which point
    the sweep goes quietly green across every page.
    """
    total = sum(len(_labels(site, slug, WIDTHS[0])) for slug in _deployable())
    assert total >= 100, (
        f"only {total} scaled svg labels across every deployed page. Either the figures have "
        "changed shape or `svg text` has stopped being how a label is drawn — and the second is "
        "more likely, at which point this file measures nothing."
    )


def test_the_exemption_ledger_has_not_gone_stale(site) -> None:
    """An entry excusing a figure that is legible now is a figure no longer checked."""
    stale = []
    for (slug, section), reason in TOO_SMALL_ON_PURPOSE.items():
        seen = False
        worst = None
        for width in WIDTHS:
            for label in _labels(site, slug, width):
                if label["section"] != section:
                    continue
                seen = True
                if worst is None or label["painted"] < worst:
                    worst = label["painted"]
        if not seen:
            stale.append(f"{slug}:{section} has no scaled svg label any more ({reason})")
        elif worst is not None and worst >= LEGIBLE:
            stale.append(
                f"{slug}:{section} is legible now — smallest {worst:.1f}px, floor {LEGIBLE}px "
                f"({reason})"
            )

    assert not stale, "remove these entries; each excuses a figure that no longer needs it:\n  " + (
        "\n  ".join(stale)
    )
