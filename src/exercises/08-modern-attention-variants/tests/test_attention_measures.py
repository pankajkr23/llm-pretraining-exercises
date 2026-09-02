"""Every block of prose on the page reads at a sane line length, at every width.

A reader reported that "text goes too narrow in many places" and sent a saved copy of the page.
Measuring all 88 text blocks in that exact file found two causes: **eleven standfirsts at 34
characters** and **six plate captions at 40 characters per column**. Measuring the live page across
ten viewport widths found three more that only appear at some sizes -- the key's columns below
1600px, the reading spread's between 1024 and 1360, and the colophon running to **99** characters
under 800px, the mirror image of the 47 it read at when its columns were too narrow.

Two things this guard gets right that a simpler one would not.

**Characters per line, never pixels.** The same 500px is comfortable at 13px and a column of
fragments at 24px, which is exactly how the standfirsts reached 34 without anyone noticing: the
rule said `34ch` and looked deliberate.

**Narrow is only a defect when there is room to be wider.** On a 390px phone the standfirst reads
at 29 characters while filling 92% of everything available -- that is the device, not a decision,
and a flat floor would fail every page ever built on a phone. Equally, thirty-five blocks read at a
perfect 70 characters while using 41% of a 1,676px full-bleed track: that is the design, because
the wide track is for figures and widening prose to fill it would push every paragraph past 170
characters. The ratio alone invites a fix worse than the bug, so the rule is *short AND with space
beside it*. `docs/MEASURES.md` records the full audit.
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
SLUG = "08-modern-attention-variants"

#: Running prose. Below this a paragraph reads as fragments; above it the eye loses the line.
NARROWEST, WIDEST = 45, 80

#: A short block is only a defect if it is also leaving room unused.
ROOM_TO_SPARE = 0.70

#: Every defect found here appeared at some widths and not others, so one viewport proves little.
WIDTHS = (2560, 1920, 1600, 1440, 1280, 1180, 1024, 900, 768, 390)

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
  const sel = '#main p, #main li, #main figcaption, #main .say, #main .brief-row p';
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
      sec: (el.closest('section[data-role]') || {}).dataset?.role || '-',
      cols: cols,
      chars: Math.round((r.width / cols) / chw(el)),
      fill: hw ? r.width / hw : 1,
    });
  }
  probe.remove();
  return out;
}"""

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def measured():
    """Every text block, at every width, with the characters per line it actually renders."""
    if not (PUBLIC / SLUG / "index.html").is_file():
        pytest.skip("run deploy/vercel/build.sh first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page(viewport={"width": WIDTHS[0], "height": 1200})
            page.goto(
                f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/", wait_until="networkidle"
            )
            page.wait_for_timeout(2500)
            rows = []
            for width in WIDTHS:
                page.set_viewport_size({"width": width, "height": 1200})
                page.wait_for_timeout(400)
                for row in page.evaluate(MEASURE_JS):
                    row["at"] = width
                    rows.append(row)
            yield rows
            page.close()
            browser.close()
    finally:
        httpd.shutdown()


def test_enough_prose_was_measured_for_this_to_mean_anything(measured) -> None:
    """A selector that stops matching would make every assertion below pass on an empty list."""
    assert len(measured) >= 50 * len(WIDTHS), (
        f"only {len(measured)} measurements across {len(WIDTHS)} widths; the selector has rotted"
    )


def test_no_block_of_prose_is_too_narrow_when_it_could_be_wider(measured) -> None:
    """Eleven standfirsts sat at 34 characters and six captions at 40, and nothing failed."""
    narrow = [
        (r["at"], r["sec"], r["cls"], r["chars"])
        for r in measured
        if r["chars"] < NARROWEST and r["fill"] < ROOM_TO_SPARE
    ]
    assert not narrow, (
        f"{len(narrow)} blocks read under {NARROWEST} characters with room to spare:\n"
        + "\n".join(f"    at {w}px  {s:11} {c:28} {n:>3} chars" for w, s, c, n in narrow[:10])
    )


def test_no_block_of_prose_is_too_wide_to_track(measured) -> None:
    """The opposite failure, and just as real: the colophon ran to 99 characters under 800px."""
    wide = [(r["at"], r["sec"], r["cls"], r["chars"]) for r in measured if r["chars"] > WIDEST]
    assert not wide, f"{len(wide)} blocks read over {WIDEST} characters a line:\n" + "\n".join(
        f"    at {w}px  {s:11} {c:28} {n:>3} chars" for w, s, c, n in wide[:10]
    )


def test_nothing_is_split_into_columns_too_narrow_to_be_worth_splitting(measured) -> None:
    """Splitting a narrow block in two makes two narrower blocks.

    A `columns: 2` added to plate captions took them from 79 characters to 40, trying to stop them
    looking stranded. The fix was worse than what it fixed, and it is the specific mistake this
    guard exists to keep from recurring.
    """
    bad = [
        (r["at"], r["sec"], r["cls"], r["cols"], r["chars"])
        for r in measured
        if r["cols"] > 1 and r["chars"] < 55
    ]
    assert not bad, "these are split into columns too narrow to read:\n" + "\n".join(
        f"    at {w}px  {s:11} {c:26} {n} cols, {ch} chars each" for w, s, c, n, ch in bad
    )
