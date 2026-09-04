"""Every deployable page, under every one of the six themes, actually readable.

**Six exercises link the six-theme token file and one of them was tested.** Exercise 08 has a
thorough theme suite — but it carries a long allowlist of that page's own diagram classes, so it
cannot simply be pointed at the others. What generalises is the property underneath it, and that is
what this file asserts:

- the page renders under the theme with no console or page errors,
- the tokens actually resolve — a page linking only the *vendored* `_shared/tokens.css` and not the
  root one paints `var(--bg)` as nothing at all, and every mark drawn with it disappears silently,
- and the body text clears WCAG AA against the ground it is painted on.

**Six themes, not two.** The system light/dark pair plus `soft-light`, `tinted-dark`,
`high-contrast` and `neon`, each redefining the whole token set. A page styled for two of them is
unreadable in the other four, and nothing about working in one theme predicts the others: the
contrast between an accent and a surface can invert entirely between `neon` and `high-contrast`.

**Contrast is computed in the browser, not by parsing CSS.** The question is what the reader's
machine actually painted after the cascade, `prefers-color-scheme` and any `data-theme` attribute
have all resolved. Parsing the stylesheet would test our reading of the file; this tests the pixel.
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

#: The six themes. `None` means no `data-theme` attribute at all — the default, where only
#: `prefers-color-scheme` separates light from dark, and the state most readers are actually in.
THEMES = [
    ("system-light", None, "light"),
    ("system-dark", None, "dark"),
    ("soft-light", "soft-light", "light"),
    ("tinted-dark", "tinted-dark", "dark"),
    ("high-contrast", "high-contrast", "light"),
    ("neon", "neon", "dark"),
]

#: WCAG AA for body text.
AA = 4.5

LUMINANCE_JS = """
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


def _deployable() -> list[str]:
    """Every exercise slug the build publishes, read from the filesystem.

    Listed nowhere: a hand-maintained roster goes stale the first time an exercise is added, which
    is exactly the failure `tests/test_deploy_registration.py` exists to catch one level up.
    """
    return sorted(p.parent.parent.name for p in REPO_ROOT.glob("src/exercises/*/web/index.html"))


@pytest.fixture(scope="module")
def site():
    """Serve the assembled site once for the whole module."""
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


def _measure(site, slug: str, attr: str | None, scheme: str) -> dict:
    browser, base = site
    ctx = browser.new_context(color_scheme=scheme, viewport={"width": 1400, "height": 950})
    page = ctx.new_page()
    problems: list[str] = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        if attr:
            page.evaluate(f"document.documentElement.setAttribute('data-theme', {attr!r})")
        page.wait_for_timeout(700)
        result = page.evaluate(
            LUMINANCE_JS
            + """
            () => {
              const root = getComputedStyle(document.documentElement);
              const body = getComputedStyle(document.body);
              const unresolved = ['--bg', '--ink', '--accent', '--panel', '--line', '--muted']
                .filter((n) => !root.getPropertyValue(n).trim());
              const ink = rgb(body.color);
              const bg = rgb(body.backgroundColor);
              return {
                unresolved,
                contrast: ink && bg ? ratio(ink, bg) : null,
                ink: body.color,
                bg: body.backgroundColor,
              };
            }"""
        )
        result["problems"] = problems
        return result
    finally:
        ctx.close()


@pytest.mark.parametrize("slug", _deployable())
@pytest.mark.parametrize("name,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_the_page_is_readable_under_this_theme(site, slug: str, name: str, attr, scheme) -> None:
    """The whole property, one page and one theme at a time.

    Parametrised over both so a failure names exactly which page in which theme, rather than
    reporting "something, somewhere, is unreadable".
    """
    m = _measure(site, slug, attr, scheme)

    assert not m["problems"], f"{slug} under {name} logged errors: {m['problems'][:3]}"

    assert not m["unresolved"], (
        f"{slug} under {name} paints these tokens as nothing: {m['unresolved']}.\n\n"
        "The usual cause is linking only the vendored `_shared/tokens.css` — exercise 03's "
        "COMPONENT stylesheet, not the token file — and not the root `/_shared/tokens.css`. A mark "
        "drawn with an undefined custom property does not error; it simply does not paint."
    )

    assert m["contrast"] is not None, f"{slug} under {name}: could not read the body colours"
    assert m["contrast"] >= AA, (
        f"{slug} under {name}: body text is {m['contrast']:.2f}:1 against its own background "
        f"({m['ink']} on {m['bg']}), below WCAG AA of {AA}:1."
    )


def test_the_contrast_check_can_actually_fail(site) -> None:
    """Break it on purpose. A contrast guard that cannot report a failure is decorative.

    `AGENTS.md`: every invariant is written twice — once against the real thing, once against a
    deliberately broken one — and a guard nobody has watched go red is not a guard.
    """
    browser, base = site
    ctx = browser.new_context(viewport={"width": 1200, "height": 800})
    page = ctx.new_page()
    try:
        page.goto(f"{base}/{_deployable()[0]}/index.html", wait_until="networkidle", timeout=25_000)
        # Paint the body's text almost exactly its own background: the failure this guard is for.
        page.evaluate(
            """() => {
                 const bg = getComputedStyle(document.body).backgroundColor;
                 document.body.style.color = bg;
               }"""
        )
        page.wait_for_timeout(200)
        measured = page.evaluate(
            LUMINANCE_JS
            + """
            () => {
              const b = getComputedStyle(document.body);
              const ink = rgb(b.color), bg = rgb(b.backgroundColor);
              return ink && bg ? ratio(ink, bg) : null;
            }"""
        )
        assert measured is not None and measured < AA, (
            f"text painted its own background measured {measured}:1, which the checker did not "
            "report as a failure — so it cannot detect the thing it exists for"
        )
    finally:
        ctx.close()


def test_at_least_six_pages_are_covered() -> None:
    """A parametrised guard over an empty set is green and worthless.

    `_deployable()` is globbed, so a rename or a moved directory would empty it silently and every
    assertion above would pass by having nothing to check.
    """
    found = _deployable()
    assert len(found) >= 6, (
        f"only {len(found)} deployable page(s) were found: {found}. The glob has probably stopped "
        "matching, and the theme matrix above is passing over an empty set."
    )


def test_the_unresolved_token_check_can_actually_fail(site) -> None:
    """The other half, broken on purpose: disable the root token sheet and watch it go red.

    This is the failure the check exists for. Every page links two stylesheets named
    `tokens.css` — the vendored one, which is exercise 03's *component* stylesheet, and the root
    `/_shared/tokens.css`, which is the only file that declares the six themes. Dropping the second
    is a one-character edit to an `href`, it raises no console error, and every mark drawn with
    `var(--bg)` silently stops painting.
    """
    browser, base = site
    ctx = browser.new_context(viewport={"width": 1200, "height": 800})
    page = ctx.new_page()
    try:
        page.goto(f"{base}/{_deployable()[0]}/index.html", wait_until="networkidle", timeout=25_000)
        disabled = page.evaluate(
            """() => {
                 let n = 0;
                 for (const s of document.styleSheets) {
                   if ((s.href || '').includes('/_shared/tokens.css')) { s.disabled = true; n++; }
                 }
                 return n;
               }"""
        )
        assert disabled == 1, (
            f"expected exactly one root token stylesheet to disable, found {disabled} — the page's "
            "link to `/_shared/tokens.css` has moved, and this twin is no longer breaking anything"
        )
        page.wait_for_timeout(200)
        unresolved = page.evaluate(
            """() => {
                 const root = getComputedStyle(document.documentElement);
                 return ['--bg', '--ink', '--accent', '--panel', '--line', '--muted']
                   .filter((n) => !root.getPropertyValue(n).trim());
               }"""
        )
        assert unresolved, (
            "with the six-theme token file disabled, every one of those custom properties still "
            "resolved — so the check above cannot detect a page that never linked it"
        )
    finally:
        ctx.close()
