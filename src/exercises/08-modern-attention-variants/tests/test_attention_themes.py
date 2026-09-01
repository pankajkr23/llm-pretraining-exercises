"""Every theme, actually rendered — the gap this repo has carried since the theme picker shipped.

Six themes are offered (`:root`, `prefers-color-scheme: dark`, and four `data-theme` values) and
**nothing has ever rendered the page under five of them.** Theme correctness has been convention:
"every colour must be a token". A convention catches a colour that is wrong; it cannot catch a
colour that is *invisible*, and this exercise has already shipped that exact defect twice — a
scratch harness whose every mark was `stroke: var(--bg)` against an undefined token, painting
nothing with a clean console, and a set of glyph marks that were white-on-white until the real
token file was linked.

The failure mode being guarded is specific: **a mark that resolves to the same colour as the thing
behind it.** It throws nothing, logs nothing, and passes every structural assertion, because the
element is present and correctly positioned. It is simply not there to look at. The one thing that
catches it is measuring the contrast between what was painted and what it was painted on.

`high-contrast` is why this cannot be eyeballed on one theme and assumed for the rest: it sets
`--muted` and `--ink` to the *same* `#000000`. Any encoding that leans on ink-against-muted reads
perfectly in five themes and vanishes in the sixth.
"""

import functools
import http.server
import re
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC = REPO_ROOT / "public"
TOKENS = REPO_ROOT / "deploy" / "vercel" / "_shared" / "tokens.css"
SLUG = "08-modern-attention-variants"

pytestmark = pytest.mark.integration

#: The six the page actually offers: the default, the OS-dark default, and four explicit choices.
#: `attr` is what goes on `<html data-theme>`; `scheme` is the emulated OS preference.
THEMES: tuple[tuple[str, str | None, str], ...] = (
    ("system-light", None, "light"),
    ("system-dark", None, "dark"),
    ("soft-light", "soft-light", "light"),
    ("tinted-dark", "tinted-dark", "dark"),
    ("high-contrast", "high-contrast", "light"),
    ("neon", "neon", "dark"),
)

#: Contrast floors. Body text is held to WCAG AA; a graphical mark only has to be *seen*, and WCAG
#: puts non-text contrast at 3:1. The mark floor is deliberately lower than the text floor and
#: deliberately well above 1.0 — at 1.0 a mark is exactly its own background.
TEXT_CONTRAST = 4.5
MARK_CONTRAST = 2.0

#: Marks that must be readable against the PAGE. If one of these matches the background it is
#: invisible, and that is the defect this file exists to catch.
MARKS_ON_PAGE = (
    "f-ink",
    "f-accent",
    "f-muted",
    "f-faint",
    "s-ink",
    "s-accent",
    "s-muted",
    "gl-on",
    "gl-state",
    "gl-band",
    "gl-kv",
    "gl-q",
    "gl-gate",
    "gl-perm",
    "gl-permblock",
)

#: Marks painted ON TOP OF a filled field, so `--ink` is the surface behind them, not `--bg`.
#: `gl-tile` is literally `stroke: var(--bg)` — background-coloured on purpose, because it divides
#: a solid dark field. Measuring it against the page would report a real design decision as a bug.
MARKS_ON_INK = ("gl-edit", "gl-edit-s", "gl-cut", "gl-flush", "gl-tile")

#: Surfaces and hairlines. These are grounds and rules, not marks; they are meant to sit close to
#: the page and holding them to a mark's floor would be measuring the wrong thing.
SURFACES = (
    "f-panel",
    "f-track",
    "s-line",
    "s-strong",
    "gl-wall",
    "gl-cont",
    "gl-tilebox",
    "gl-unknown",
    "gl-schema",
    "gl-wire",
    "gl-band",
)


def _luminance_js() -> str:
    """Relative luminance and contrast ratio, per WCAG, evaluated in the browser.

    Done in the browser rather than by parsing the stylesheet because the question is what the
    reader's machine actually painted after the cascade, `prefers-color-scheme` and the
    `data-theme` attribute have all resolved. Parsing CSS would test our reading of the file; this
    tests the pixel.
    """
    return """
      const rgb = (s) => {
        const m = s.match(/-?[\\d.]+/g);
        if (!m) return null;
        return [Number(m[0]), Number(m[1]), Number(m[2]), m[3] === undefined ? 1 : Number(m[3])];
      };
      const lum = (c) => {
        const f = (v) => {
          v /= 255;
          return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
        };
        return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
      };
      const ratio = (a, b) => {
        const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
        return (x + 0.05) / (y + 0.05);
      };
    """


@pytest.fixture(scope="module")
def browser():
    if not (PUBLIC / SLUG / "index.html").is_file():
        pytest.skip("run deploy/vercel/build.sh first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PUBLIC))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            yield b, f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html"
            b.close()
    finally:
        httpd.shutdown()


def _open(browser, attr: str | None, scheme: str):
    b, url = browser
    ctx = b.new_context(color_scheme=scheme, viewport={"width": 1400, "height": 950})
    page = ctx.new_page()
    problems: list[str] = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.goto(url)
    page.wait_for_selector("section#reproduce", timeout=15_000)
    if attr:
        page.evaluate(f"document.documentElement.setAttribute('data-theme', {attr!r})")
    page.wait_for_timeout(900)
    page.console_problems = problems
    return ctx, page


@pytest.mark.parametrize("name,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_the_page_renders_under_every_theme(browser, name: str, attr: str | None, scheme: str):
    """No errors, tokens resolve, and body text clears WCAG AA on the ground it is painted on."""
    ctx, page = _open(browser, attr, scheme)
    try:
        assert not page.console_problems, f"{name}: {page.console_problems}"

        read = page.evaluate(
            _luminance_js()
            + """
            () => {
              const cs = getComputedStyle(document.body);
              const root = getComputedStyle(document.documentElement);
              const names = ['--bg','--ink','--accent','--panel','--line','--muted',
                             '--faint','--track'];
              const tokens = {};
              for (const n of names) tokens[n] = root.getPropertyValue(n).trim();
              return {
                tokens,
                bg: cs.backgroundColor,
                fg: cs.color,
                contrast: ratio(rgb(cs.color), rgb(cs.backgroundColor)),
              };
            }"""
        )

        empty = [n for n, v in read["tokens"].items() if not v]
        assert not empty, (
            f"{name}: these tokens resolve to nothing: {empty}. The commonest cause is a page "
            f"linking only the vendored _shared/tokens.css, which is a component stylesheet and "
            f"not the token file."
        )
        assert read["contrast"] >= TEXT_CONTRAST, (
            f"{name}: body text contrast is {read['contrast']:.2f}:1 "
            f"({read['fg']} on {read['bg']}), below the {TEXT_CONTRAST}:1 floor"
        )
    finally:
        page.close()
        ctx.close()


@pytest.mark.parametrize("name,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_no_painted_mark_is_the_same_colour_as_its_background(
    browser, name: str, attr: str | None, scheme: str
):
    """**The guard this file exists for.**

    Samples every fill and stroke class the figures use and measures it against the surface behind
    it. A mark at 1.0:1 is exactly its own background: present in the DOM, correctly positioned,
    and invisible. That is the defect this page has shipped twice, both times with a clean console
    and a green suite.
    """
    ctx, page = _open(browser, attr, scheme)
    try:
        bad = page.evaluate(
            _luminance_js()
            + """
            ([onPage, onInk, floor]) => {
              const svgNS = 'http://www.w3.org/2000/svg';
              const holder = document.createElementNS(svgNS, 'svg');
              holder.style.cssText = 'position:absolute;left:-9999px';
              document.body.appendChild(holder);

              /* A control with no class, so we can tell "this class sets fill" from "the browser
               * defaults every shape's fill to black". Probing the wrong property reports a stroke
               * class as an invisible black fill, which is a guard crying wolf. */
              const control = document.createElementNS(svgNS, 'rect');
              holder.appendChild(control);
              const cs0 = getComputedStyle(control);
              const baseFill = cs0.fill, baseStroke = cs0.stroke;

              const root = getComputedStyle(document.documentElement);
              const page = rgb(getComputedStyle(document.body).backgroundColor);
              const ink = rgb(root.getPropertyValue('color-mix') ? '' : '') || null;
              const inkColour = (() => {
                const probe = document.createElementNS(svgNS, 'rect');
                probe.setAttribute('style', 'fill: var(--ink)');
                holder.appendChild(probe);
                return rgb(getComputedStyle(probe).fill);
              })();

              const out = [];
              const check = (cls, ground) => {
                const r = document.createElementNS(svgNS, 'rect');
                r.setAttribute('class', cls);
                holder.appendChild(r);
                const cs = getComputedStyle(r);
                for (const [prop, base] of [['fill', baseFill], ['stroke', baseStroke]]) {
                  const paint = cs[prop];
                  if (paint === base) continue;              // this class does not set it
                  if (paint === 'none') continue;
                  const c = rgb(paint);
                  if (!c || c[3] === 0) continue;            // deliberately transparent
                  const op = Number(cs.opacity);
                  if (op < 0.2) continue;                    // deliberately a whisper
                  const v = ratio(c, ground);
                  if (v < floor) out.push([cls, prop, paint, +v.toFixed(2)]);
                }
              };
              for (const c of onPage) check(c, page);
              for (const c of onInk) check(c, inkColour);
              holder.remove();
              return out;
            }""",
            [
                [c for c in MARKS_ON_PAGE if c not in SURFACES],
                [c for c in MARKS_ON_INK if c not in SURFACES],
                MARK_CONTRAST,
            ],
        )
        assert not bad, (
            f"{name}: these marks are within {MARK_CONTRAST}:1 of the page background and will be "
            f"effectively invisible — {bad}"
        )
    finally:
        page.close()
        ctx.close()


def test_the_contrast_checker_can_actually_fail(browser):
    """Break it on purpose. A contrast guard that cannot report a failure is decorative.

    Paints a rule whose colour IS the background, then asserts the same measurement the guard above
    uses reports it. Without this the whole file could be silently measuring nothing.
    """
    ctx, page = _open(browser, None, "light")
    try:
        measured = page.evaluate(
            _luminance_js()
            + """
            () => {
              const ground = rgb(getComputedStyle(document.body).backgroundColor);
              const svgNS = 'http://www.w3.org/2000/svg';
              const holder = document.createElementNS(svgNS, 'svg');
              holder.style.cssText = 'position:absolute;left:-9999px';
              document.body.appendChild(holder);
              const r = document.createElementNS(svgNS, 'rect');
              // the exact defect: a mark painted on its own ground
              r.setAttribute('style', 'fill: var(--bg)');
              holder.appendChild(r);
              const v = ratio(rgb(getComputedStyle(r).fill), ground);
              holder.remove();
              return v;
            }"""
        )
        assert measured < MARK_CONTRAST, (
            f"a mark painted in var(--bg) on the page background measured {measured:.2f}:1 — the "
            f"checker is not measuring what it claims to"
        )
        assert abs(measured - 1.0) < 0.05, (
            f"expected ~1.0:1 for a mark on its own ground, got {measured}"
        )
    finally:
        page.close()
        ctx.close()


def test_the_theme_picker_offers_every_theme_the_token_file_defines() -> None:
    """Lexical, no browser. A theme defined and never offered is a theme nobody can reach.

    Reads the `data-theme` selectors out of the token file and the `<option value>` list out of the
    page shell. Both are facts about the source, which is the property a coverage check needs.
    """
    defined = set(
        re.findall(r':root\[data-theme="([a-z-]+)"\]', TOKENS.read_text(encoding="utf-8"))
    )
    shell = (REPO_ROOT / "src" / "exercises" / SLUG / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    offered = set(re.findall(r'<option value="([a-z-]+)"', shell)) - {"system"}
    assert defined == offered, (
        f"the token file defines {sorted(defined)} but the picker offers {sorted(offered)}"
    )


def test_every_theme_defines_the_whole_token_set() -> None:
    """The token file's own stated rule, enforced by nothing until now.

    A theme that omits a token inherits the previous block's value, which is how a dark theme ends
    up painting one light-theme colour and nobody notices.
    """
    css = TOKENS.read_text(encoding="utf-8")
    blocks = re.findall(
        r'(:root(?:\[data-theme="[a-z-]+"\])?|@media \(prefers-color-scheme: dark\))\s*\{(.*?)\n\}',
        css,
        re.S,
    )
    named = [
        (sel.strip(), dict(re.findall(r"(--[a-z-]+):\s*([^;]+);", body))) for sel, body in blocks
    ]
    named = [(s, v) for s, v in named if "--bg" in v or "--ink" in v]
    assert len(named) >= 6, f"expected at least six theme blocks, found {len(named)}"

    #: Font stacks are declared once and correctly inherited; only colours must be redeclared,
    #: because a colour inherited from a light block is how a dark theme paints one wrong surface.
    fonts = {"--sans", "--serif", "--mono", "--display"}
    base = set(named[0][1]) - fonts
    for sel, vals in named[1:]:
        missing = sorted(base - set(vals) - fonts)
        assert not missing, (
            f"{sel} inherits {missing} from an earlier block instead of setting them"
        )
