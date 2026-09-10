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

**Driving the pages is `--drive`, and it answers the question the table above cannot.** It scrolls
every step, opens every disclosure, clicks every button and sweeps every slider, and it judges a
focus or pseudo-element rule by whether the element it styles exists — `querySelector` sees neither
at rest, so a raw count calls `.step:focus-visible` unused on a page full of steps. Run on
2026-09-10, after the change below, over `explainer.css`'s 132 selectors:

| | selectors |
| --- | ---: |
| match at rest | 113 |
| match only once the pages are driven | 11 |
| never match | 8 |

The 8 are kept. Each is a descendant today's content lacks — `code` inside an arithmetic note, a
link inside a disclaimer — or a value the data schema allows and the data does not use yet, such as
the `additional_to` derivation and the `research_papers` modality. That is the limit of this tool:
it can say a rule never applied, and it cannot say whether it ever could.

**What it found and what went.** Before the change the stylesheet carried 143 selectors, 11 of them
the "single canvas" block — `.canvas` and the `.unit` family, a grid of one square per dataset that
exercise 03's one-page rebuild replaced. No code in any exercise sets either class. They were
removed from all eight vendored copies, and screenshots of the seven pages that link the stylesheet
hashed identically before and after at 2000, 1180 and 390px.
"""

import argparse
import functools
import http.server
import re
import socketserver
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


#: States a reader reaches by pointing, tabbing or checking. `querySelector` sees none of them at
#: rest, so a rule written for one is judged by whether the element it styles exists at all.
#: Longest names first, so `:focus` never eats the start of `:focus-visible`.
_DYNAMIC = re.compile(
    r":(?:hover|focus-visible|focus-within|focus|active|checked|disabled|target)(?![\w-])"
)
#: Pseudo-elements. `querySelector` returns nothing for these whether or not the element exists.
_PSEUDO_ELEMENT = re.compile(
    r"::?(?:before|after|placeholder|marker|selection|backdrop|first-line|first-letter)(?![\w-])"
    r"|::-(?:webkit|moz)-[\w-]+"
)


def base_selector(selector: str) -> str | None:
    """The element a selector styles, without its reader-reached states and pseudo-elements.

    This is what lets a measurement judge `.step:focus-visible` or `summary::before` at all: both
    match nothing under `querySelector` even while the element they style is on the screen. What is
    left is the element, and whether *that* exists is the question a deletion needs answered.
    Structural pseudo-classes — `:has()`, `:not()`, `:nth-child()` — are left alone, because they
    select which elements exist rather than a state one of them is in.

    Args:
        selector: One selector from a stylesheet.

    Returns:
        The element the rule styles, or None when the selector already names nothing but elements.
    """
    stripped = _PSEUDO_ELEMENT.sub("", _DYNAMIC.sub("", selector))
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if stripped == selector.strip() or not stripped:
        return None
    return stripped


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


_MATCHING_JS = """(probes) => probes.map((s) => {
  try { return document.querySelector(s) !== null; } catch (e) { return false; }
})"""

_SWEEP_JS = """(fraction) => document.querySelectorAll('input[type=range]').forEach((r) => {
  const lo = Number(r.min || 0), hi = Number(r.max || 100);
  r.value = String(lo + (hi - lo) * fraction);
  r.dispatchEvent(new Event('input', {bubbles: true}));
  r.dispatchEvent(new Event('change', {bubbles: true}));
})"""


def _drive(page: Any, collect: Callable[[], None]) -> None:
    """Put a page through every state a reader can reach without typing, measuring after each.

    Scroll every step of a scrollytelling strip into the middle of the screen, open every
    disclosure, click every visible button once, and sweep every slider to its minimum, middle and
    maximum. It is deliberately blunt: the question is whether a rule can EVER apply, so reaching a
    state by an order no reader would choose is still a state the page can be in.
    """
    collect()
    for i in range(page.evaluate("document.querySelectorAll('.step').length")):
        page.evaluate(
            "(i) => document.querySelectorAll('.step')[i].scrollIntoView({block: 'center'})", i
        )
        page.wait_for_timeout(180)
        collect()
    page.evaluate("document.querySelectorAll('details').forEach((d) => { d.open = true; })")
    collect()
    buttons = page.locator("button:visible, [role=button]:visible")
    for i in range(min(buttons.count(), 200)):
        try:
            buttons.nth(i).click(timeout=800)
        except Exception:  # hidden or detached by an earlier click: that state was measured already
            continue
        page.wait_for_timeout(120)
        collect()
    for fraction in (0.0, 0.5, 1.0):
        page.evaluate(_SWEEP_JS, fraction)
        page.wait_for_timeout(200)
        collect()


def measure_driven(stylesheet: str) -> dict[str, tuple[set[str], set[str]]]:
    """Per selector: the pages it matches on at rest, and the pages it matches on once driven.

    A selector with a reader-reached state or a pseudo-element is judged by `base_selector()` —
    by whether the element it styles exists — because that is the only thing a page can show.
    """
    from playwright.sync_api import sync_playwright

    source = next(REPO_ROOT.glob(f"src/exercises/*/web/{stylesheet}"), None)
    if source is None:
        raise SystemExit(f"no exercise vendors {stylesheet!r}")
    wanted = selectors_in(source.read_text(encoding="utf-8"))
    probes = [base_selector(s) or s for s in wanted]
    at_rest: dict[str, set[str]] = {s: set() for s in wanted}
    driven: dict[str, set[str]] = {s: set() for s in wanted}

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for slug in _pages():
                if not links(slug, stylesheet):
                    continue
                page = browser.new_page(viewport={"width": 1440, "height": 950})
                page.on("dialog", lambda dialog: dialog.dismiss())
                page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
                page.wait_for_timeout(500)

                def matched(into: dict[str, set[str]], page: Any = page, slug: str = slug) -> None:
                    for selector, hit in zip(
                        wanted, page.evaluate(_MATCHING_JS, probes), strict=True
                    ):
                        if hit:
                            into[selector].add(slug)

                matched(at_rest)
                _drive(page, lambda: matched(driven))
                page.close()
            browser.close()
    finally:
        httpd.shutdown()
    return {s: (at_rest[s], driven[s]) for s in wanted}


def _report_driven(stylesheet: str) -> int:
    seen = measure_driven(stylesheet)
    at_rest = [s for s, (rest, _) in seen.items() if rest]
    only_driven = [s for s, (rest, drove) in seen.items() if not rest and drove]
    never = [s for s, (rest, drove) in seen.items() if not rest and not drove]
    print(
        f"\n{len(seen)} selectors in {stylesheet}: {len(at_rest)} match at rest, "
        f"{len(only_driven)} only once the pages are driven, {len(never)} never"
    )
    for selector in only_driven:
        print(f"  driven  {selector}  ({', '.join(sorted(seen[selector][1]))})")
    for selector in never:
        print(f"  never   {selector}")
    print(
        "\n'never' is still not a verdict. A descendant today's content lacks, or a value the data "
        "schema allows and the data does not use yet, matches nothing too. Read each one before "
        "deleting it."
    )
    return 0


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
    parser.add_argument(
        "--drive",
        action="store_true",
        help="also drive every page that links it: scroll every step, open every disclosure, "
        "click every button, sweep every slider; then list what never matches at all",
    )
    args = parser.parse_args(argv)

    if not (PUBLIC / "_shared" / "tokens.css").is_file():
        print("run `bash deploy/vercel/build.sh` first", file=sys.stderr)
        return 2

    if args.drive:
        return _report_driven(args.stylesheet)

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
