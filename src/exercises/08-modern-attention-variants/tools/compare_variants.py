"""Load all four A/B combinations and print one table, so the choice is made against numbers.

TEMPORARY. This file is deleted with the harness once PK picks a variant.

Two decisions about this page could reasonably go either way, and both were being made on the
author's taste. Rather than argue, the page ships both and this measures them:

    story    a = the six chapters are openers; the index at the back carries all thirty
             b = each chapter carries its own entries; the index becomes the receipt
    measure  a = 16px prose on a 68ch track
             b = fluid 17->22px on a 70ch track, at the same characters a line

Everything reported here is read off the rendered page, not computed from the source. Run it after
`bash deploy/vercel/build.sh`.

    uv run python src/exercises/08-modern-attention-variants/tools/compare_variants.py
"""

import functools
import http.server
import socketserver
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC = REPO_ROOT / "public"
SLUG = "08-modern-attention-variants"

#: The widths that decide it. 1920 is where the reader's complaint was measured; 1440 is the
#: commonest laptop; 2560 is where the old layout gave the prose 27% of the screen.
WIDTHS = (2560, 1920, 1440)

#: A page is not "long" in pixels, it is long in screens. 900 is the viewport the rest of this
#: exercise's measurements use.
SCREEN = 900

MEASURE_JS = """() => {
  const p = document.querySelector('#glossary p.say');
  const cs = getComputedStyle(p);
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre';
  probe.style.font = cs.font;
  probe.textContent = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789';
  document.body.append(probe);
  const per = probe.getBoundingClientRect().width / probe.textContent.length;
  probe.remove();
  const prose = Math.round(p.getBoundingClientRect().width);

  // Where the entries live. Under `story = a` one container holds all thirty; under `b` six
  // chapters hold them between them. Counting "containers with all thirty" would report 0 for b
  // and read as a defect, so count containers holding ANY entry, and the entries themselves.
  const entries = document.querySelectorAll('[id^="m-"]').length;
  let holders = 0;
  for (const e of document.querySelectorAll('#main div, #main section')) {
    if ([...e.children].some((c) => c.id && /^m-/.test(c.id))) holders += 1;
  }

  const wells = [...document.querySelectorAll('#results .well')];
  return {
    height: Math.round(document.body.scrollHeight),
    prose,
    fontSize: cs.fontSize,
    chars: Math.round(prose / per),
    words: (document.querySelector('#main').innerText.match(/\\S+/g) || []).length,
    entries,
    holders,
    chaptersWithBodies: wells.filter((w) => w.querySelector('.ce-row')).length,
    chapters: wells.length,
    sections: Object.fromEntries(
      [...document.querySelectorAll('#main > section')].map((e) => [
        e.id,
        Math.round(e.getBoundingClientRect().height),
      ])
    ),
  };
}"""


def _serve() -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main() -> int:
    """Print the decision table. Returns a shell exit code."""
    if not (PUBLIC / SLUG / "index.html").is_file():
        print("run `bash deploy/vercel/build.sh` first")
        return 1
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("needs playwright: uv run playwright install chromium")
        return 1

    combos = [(s, m) for s in ("a", "b") for m in ("a", "b")]
    httpd, port = _serve()
    results: dict[tuple[str, str], dict] = {}
    bytes_moved: dict[tuple[str, str], int] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for story, measure in combos:
                for width in WIDTHS:
                    page = browser.new_page(viewport={"width": width, "height": SCREEN})
                    moved = [0]
                    page.on(
                        "response",
                        lambda r, moved=moved: moved.__setitem__(
                            0, moved[0] + int(r.headers.get("content-length") or 0)
                        ),
                    )
                    page.goto(
                        f"http://127.0.0.1:{port}/{SLUG}/index.html"
                        f"?v=story:{story},measure:{measure}",
                        wait_until="networkidle",
                    )
                    results[(story, measure, width)] = page.evaluate(MEASURE_JS)
                    bytes_moved[(story, measure)] = moved[0]
                    page.close()
            browser.close()
    finally:
        httpd.shutdown()

    def row(label: str, fn) -> None:
        cells = "".join(f"{fn(s, m):>18}" for s, m in combos)
        print(f"  {label:<30}{cells}")

    head = "".join(f"{'story ' + s + ' / type ' + m:>18}" for s, m in combos)
    print(f"\n  {'':<30}{head}")
    print("  " + "-" * (30 + 18 * len(combos)))

    for width in WIDTHS:
        print(f"\n  at {width}px")
        row("page height", lambda s, m, w=width: f"{results[(s, m, w)]['height']:,}px")
        row("screens", lambda s, m, w=width: f"{results[(s, m, w)]['height'] / SCREEN:.1f}")
        row(
            "prose width",
            lambda s, m, w=width: (
                f"{results[(s, m, w)]['prose']}px ({100 * results[(s, m, w)]['prose'] / w:.0f}%)"
            ),
        )
        row("characters a line", lambda s, m, w=width: str(results[(s, m, w)]["chars"]))
        row("body type", lambda s, m, w=width: results[(s, m, w)]["fontSize"])

    print("\n  structure (width-independent)")
    w0 = WIDTHS[1]
    row("rendered words", lambda s, m: f"{results[(s, m, w0)]['words']:,}")
    row("entries rendered", lambda s, m: str(results[(s, m, w0)]["entries"]))
    row(
        "containers holding them",
        lambda s, m: str(results[(s, m, w0)]["holders"]),
    )
    row(
        "chapters with a body",
        lambda s, m: (
            f"{results[(s, m, w0)]['chaptersWithBodies']} of {results[(s, m, w0)]['chapters']}"
        ),
    )
    row("transferred", lambda s, m: f"{bytes_moved[(s, m)] / 1024:.0f} KB")

    print("\n  where the height is, at 1920px")
    for sid in results[("a", "a", w0)]["sections"]:
        row(f"  {sid}", lambda s, m, k=sid: f"{results[(s, m, w0)]['sections'][k]:,}px")

    print(
        "\n  Baseline before this pass: 29,999px / 33.3 screens, prose 685px (36% at 1920),"
        "\n  74 characters a line, thirty entries in TWO containers — the duplication a reader"
        "\n  found — and 3 of 6 chapters with a body.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
