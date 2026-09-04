"""No link falls back to the browser's default blue, and every link clears WCAG AA.

**An anchor that no rule styles is not merely unstyled — it is `#0000EE`.** That is the User-Agent
stylesheet's link colour, chosen for a white page in 1994, and against these pages' dark grounds it
measures **1.74:1 to 2.23:1**: a link a reader cannot see. Seven anchors across three pages were in
that state, on 02, 07 and 08. They had no class that set a colour, so nothing in the cascade ever
reached them, and the defect was invisible in the three light themes.

**Only leaf anchors are measured, and that distinction is the whole reliability of this file.** A
link whose text all sits in child elements — `.rail-link`, which wraps `.rail-n`, `.rail-t` and
`.rail-sub`, each with its own colour — has a computed `color` that paints nothing. Measuring it
reports 4.05:1 for text that is actually rendered at 4.59:1, 4.66:1 and 15.46:1. Two probes during
this work produced exactly that false positive, and a "fix" for it would have darkened correct text.

**Contrast is computed against the composited stack, not the first opaque ancestor.** A translucent
ground has to be blended over what is behind it, or a chip whose background is 10% alpha of its own
text colour reads as 1.00:1 against itself. That false positive also happened, on exercise 02's
`.badge`, which is 4.66:1 in fact.
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

#: Chromium's User-Agent stylesheet link colour. Any anchor computing to this has no rule at all.
BROWSER_DEFAULT_LINK = "rgb(0, 0, 238)"

#: WCAG AA for text under 24px (or under 18.66px bold). Larger text is held to 3:1.
AA_SMALL, AA_LARGE = 4.5, 3.0

#: All six. The failure here was invisible in the three light themes, where `#0000EE` on a
#: near-white ground clears AA — so a light-and-dark check would have reported nothing.
#: ground clears AA comfortably — so a light-and-dark check would have reported nothing.
THEMES = [
    ("system-light", None, "light"),
    ("system-dark", None, "dark"),
    ("soft-light", "soft-light", "light"),
    ("tinted-dark", "tinted-dark", "dark"),
    ("high-contrast", "high-contrast", "light"),
    ("neon", "neon", "dark"),
]

MEASURE_JS = """() => {
  const rgb = (s) => {
    const m = (s || '').match(/-?[\\d.]+/g);
    return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : null;
  };
  const alpha = (s) => (/rgba/.test(s || '') ? parseFloat(s.split(',')[3]) : 1);
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
    return (x + 0.05) / (y + 0.05);
  };
  const out = [];
  // Measure the elements that actually PAINT text inside each link, not the link element. A
  // wrapper's computed colour renders nothing when its text lives in coloured children — and the
  // opposite trap is worse: filtering wrappers out entirely hides the very anchors whose children
  // INHERIT the unstyled colour, which is exactly the defect. So: every text-bearing element from
  // the anchor down, each measured with its own computed colour against its own ground.
  const painters = [];
  for (const a of document.querySelectorAll('a')) {
    const box = a.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) continue;
    const s0 = getComputedStyle(a);
    if (s0.visibility === 'hidden' || +s0.opacity === 0) continue;
    for (const el of [a, ...a.querySelectorAll('*')]) {
      const own = [...el.childNodes]
        .filter((n) => n.nodeType === 3)
        .map((n) => n.textContent.trim())
        .join('');
      if (own) painters.push([el, own]);
    }
  }
  for (const [a, own] of painters) {
    const box = a.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) continue;
    const s = getComputedStyle(a);
    if (s.visibility === 'hidden' || +s.opacity === 0) continue;
    const stack = [];
    let node = a;
    while (node) {
      const c = getComputedStyle(node).backgroundColor;
      const v = rgb(c), al = alpha(c);
      if (v && al > 0) stack.push([v, al]);
      if (v && al >= 1) break;
      node = node.parentElement;
    }
    stack.push([[255, 255, 255], 1]);
    let ground = stack[stack.length - 1][0];
    for (let i = stack.length - 2; i >= 0; i--) {
      const [c, al] = stack[i];
      ground = [0, 1, 2].map((k) => c[k] * al + ground[k] * (1 - al));
    }
    const px = parseFloat(s.fontSize), wt = parseInt(s.fontWeight) || 400;
    out.push({
      colour: s.color,
      ratio: ratio(rgb(s.color), ground),
      large: px >= 24 || (px >= 18.66 && wt >= 700),
      text: own.slice(0, 30),
    });
  }
  return out;
}"""


def _pages() -> list[str]:
    """Every page the build publishes, read from the filesystem rather than listed."""
    pages = ["index.html"]
    for idx in sorted(REPO_ROOT.glob("src/exercises/*/web/index.html")):
        slug = idx.parent.parent.name
        for html in sorted(idx.parent.glob("*.html")):
            pages.append(f"{slug}/{html.name}")
    return pages


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


def _measure(site, page_path: str, attr, scheme: str) -> list[dict]:
    browser, base = site
    ctx = browser.new_context(color_scheme=scheme, viewport={"width": 1440, "height": 950})
    page = ctx.new_page()
    try:
        page.goto(f"{base}/{page_path}", wait_until="networkidle", timeout=25_000)
        if attr:
            page.evaluate("(t) => document.documentElement.setAttribute('data-theme', t)", attr)
        page.wait_for_timeout(500)
        return page.evaluate(MEASURE_JS)
    finally:
        ctx.close()


@pytest.mark.parametrize("page_path", _pages())
@pytest.mark.parametrize("theme,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_no_link_falls_back_to_the_browser_default(
    site, page_path: str, theme: str, attr, scheme: str
) -> None:
    """The cause, asserted directly — cheaper to read than a contrast number, and unambiguous."""
    rows = _measure(site, page_path, attr, scheme)
    bare = [r["text"] for r in rows if r["colour"] == BROWSER_DEFAULT_LINK]
    assert not bare, (
        f"{page_path} under {theme}: {len(bare)} link(s) compute to {BROWSER_DEFAULT_LINK}, the "
        f"browser's own default — nothing in the cascade gives them a colour:\n  "
        + "\n  ".join(repr(t) for t in bare[:6])
    )


@pytest.mark.parametrize("page_path", _pages())
@pytest.mark.parametrize("theme,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_every_link_clears_wcag_aa(site, page_path: str, theme: str, attr, scheme: str) -> None:
    """The property underneath. A styled link can still be unreadable."""
    rows = _measure(site, page_path, attr, scheme)
    bad = sorted(
        (round(r["ratio"], 2), AA_LARGE if r["large"] else AA_SMALL, r["text"])
        for r in rows
        if r["ratio"] < (AA_LARGE if r["large"] else AA_SMALL)
    )
    assert not bad, (
        f"{page_path} under {theme}: {len(bad)} of {len(rows)} links are below WCAG AA:\n  "
        + "\n  ".join(f"{r}:1 (needs {n}:1)  {t!r}" for r, n, t in bad[:6])
    )


def test_enough_links_are_being_measured(site) -> None:
    """A parametrised guard that measures nothing is green and worthless."""
    seen = sum(len(_measure(site, p, None, "light")) for p in _pages())
    assert seen >= 100, (
        f"only {seen} links measured across {len(_pages())} pages. Either the leaf-text filter has "
        "become too strict or the page set has stopped resolving."
    )
