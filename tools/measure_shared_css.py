"""Measure how much of a shared stylesheet each page that links it actually uses.

`AGENTS.md` names removing shared CSS as something with a history here: a rule that looks unused is
often a state a page reaches only after a click, and deleting it takes away something a page quietly
depended on. So this measures rather than guesses, and it prints the one number that makes a
deletion decision possible — how many of a stylesheet's selectors match anything, per page.

**It is a floor, not a verdict, and the difference is the whole reason this is a tool rather than a
one-off.** It loads each page and does nothing: no click, no focus, no scroll. A `:focus-visible`
rule matches nothing under it and is not dead; a `.stagerow.missing` that appears when a reader
removes a stage matches nothing and is not dead either. What it can say with confidence is the
opposite — that a page matching **zero** selectors of a stylesheet it links is not using it at rest,
which is worth checking by hand and was the finding that started this.

Run it against the built site:

    bash deploy/vercel/build.sh
    uv run python tools/measure_shared_css.py _shared/explainer.css

The first measurement, on 2026-09-10, over `explainer.css`'s 143 selectors:

| page | matches |
| --- | ---: |
| 03-data-collection-framework | 108 |
| 06-build-training-dataset | 30 |
| 05-datamixtures-and-curriculum | 4 |
| 04-data-cleaning-dedup | 2 |
| 08, 09, 10 | 1 each |
| **07-model-embeddings-internals** | **0** |

Exercise 07 linked a 560-line stylesheet and matched none of it. Screenshots with and without the
link hashed identically at 2000, 1180 and 390px, so it now does not link it.
"""

import argparse
import functools
import http.server
import re
import socketserver
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC = REPO_ROOT / "public"


def selectors_in(css: str) -> list[str]:
    """Every selector in a stylesheet, with comments and at-rule preludes dropped.

    Deliberately simple. A real CSS parser would handle nesting and `@supports` bodies, and this
    file's job is to produce a count that a person then checks — a selector this misses shows up as
    a smaller number, which is the safe direction for a tool whose output argues for deletion.
    """
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    found: set[str] = set()
    for block in re.finditer(r"([^{}]+)\{[^{}]*\}", body):
        head = block.group(1).strip()
        if not head or head.startswith("@"):
            continue
        for selector in head.split(","):
            selector = selector.strip()
            if selector and not selector.startswith("@"):
                found.add(selector)
    return sorted(found)


def _pages() -> list[str]:
    return sorted(p.parent.parent.name for p in REPO_ROOT.glob("src/exercises/*/web/index.html"))


def links(slug: str, stylesheet: str) -> bool:
    """Does this page's own `index.html` link the stylesheet?

    Read from the source rather than the built page, because the build fingerprints asset URLs and
    a substring match against `?v=…` is a fact about the build step rather than about the page.

    **Matched as a `<link>`, not as a substring**, and the first version of this got it wrong in a
    way worth keeping. Exercise 07's link was replaced by a comment explaining why the stylesheet is
    not linked — a comment that names the file — so a bare `"explainer.css" in text` reported the
    page as still linking it. A tool whose output argues for deleting things has to be read
    sceptically first, and this is the shape that mistake takes.
    """
    index = REPO_ROOT / "src" / "exercises" / slug / "web" / "index.html"
    pattern = re.compile(
        r"<link[^>]+href=[\"'][^\"']*" + re.escape(Path(stylesheet).name) + r"[\"'][^>]*>"
    )
    return pattern.search(index.read_text(encoding="utf-8")) is not None


def measure(stylesheet: str) -> dict[str, tuple[int, bool]]:
    """Per page: how many of the stylesheet's selectors match, and whether the page links it."""
    from playwright.sync_api import sync_playwright

    source = next(REPO_ROOT.glob(f"src/exercises/*/web/{stylesheet}"), None)
    if source is None:
        raise SystemExit(f"no exercise vendors {stylesheet!r}")
    wanted = selectors_in(source.read_text(encoding="utf-8"))
    print(f"{len(wanted)} selectors in {stylesheet}", file=sys.stderr)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    out: dict[str, tuple[int, bool]] = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for slug in _pages():
                page = browser.new_page(viewport={"width": 1440, "height": 950})
                page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
                page.wait_for_timeout(500)
                hits = page.evaluate(
                    """(sels) => sels.filter((s) => {
                         try {
                           return document.querySelector(s) !== null;
                         } catch (e) {
                           return false;
                         }
                       }).length""",
                    wanted,
                )
                out[slug] = (hits, links(slug, stylesheet))
                page.close()
            browser.close()
    finally:
        httpd.shutdown()
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the per-page table, and flag the two rows worth a person's attention.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success, 2 if the site has not been built yet.
    """
    parser = argparse.ArgumentParser(description="Measure a shared stylesheet's real use.")
    parser.add_argument(
        "stylesheet",
        nargs="?",
        default="_shared/explainer.css",
        help="path under a `web/` directory, e.g. `_shared/explainer.css`",
    )
    args = parser.parse_args(argv)

    if not (PUBLIC / "_shared" / "tokens.css").is_file():
        print("run `bash deploy/vercel/build.sh` first", file=sys.stderr)
        return 2

    results = measure(args.stylesheet)
    print(f"\n{'page':<34}{'matches':>9}  links it")
    for slug, (hits, linked) in results.items():
        flag = ""
        if linked and hits == 0:
            flag = "  <- links it and matches nothing at rest; check by hand"
        elif not linked and hits:
            flag = "  <- matches rules it does not link; it is being styled by something else"
        print(f"{slug:<34}{hits:>9}  {'yes' if linked else 'no':<4}{flag}")
    print(
        "\nA zero here is a floor, not a verdict: this loads each page and does nothing, so a rule "
        "that needs a click, a focus or a scroll matches nothing and is not dead."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
