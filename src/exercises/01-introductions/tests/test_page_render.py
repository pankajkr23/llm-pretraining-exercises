"""The four proof pages, opened in a browser — which nothing here had ever done.

**This exercise had no browser test at all, and it shipped a page that was dead on arrival.**
`s3.html` threw `Identifier 't' has already been declared` before its first statement: the theme
bootstrap declared `var t` at the top level of a classic script, which is a *global*, and the page
script opens `let E, W, …, t, …`, which cannot redeclare it. So the whole 150-line script never ran
— no chips, no sample sentence, an empty canvas — and every file-level check passed, because every
file was perfectly well-formed. The other four pages leaked the same global and escaped only by not
happening to reuse the name.

**And three of the four had no light palette.** `--animal`, `--fruit`, `--verb`, `--end`,
`--warm` and `--cool` were declared *only* inside `prefers-color-scheme: dark` blocks and the two
dark `data-theme` selectors. On the three light themes — the default — they resolved to nothing:
a chip's background was never painted at all, so its white label sat on the white page at
**1.00:1**, and canvas strokes kept whatever fill was last set, so two series stopped being
distinguishable while the diagram still looked drawn.

Neither failure is visible in the source. Both are obvious the moment a browser opens the page,
which is the whole argument for this file existing.
"""

import functools
import http.server
import os
import re
import socketserver
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

# tests/ ← src/ ← exercises/ ← 01-introductions/ ← tests/  — five levels, not four.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests"))
from _page_invariants import attach, findings  # noqa: E402

EXERCISE = Path(__file__).resolve().parents[1]
REPO_ROOT = EXERCISE.parents[2]
PUBLIC = REPO_ROOT / "public"
SLUG = EXERCISE.name

PAGES = ["index.html", "s1.html", "s2.html", "s3.html", "s4.html"]

#: The six themes. `None` means no `data-theme` attribute — the default, and the state in which
#: every one of the failures above was live.
THEMES = [
    ("system-light", None, "light"),
    ("system-dark", None, "dark"),
    ("soft-light", "soft-light", "light"),
    ("tinted-dark", "tinted-dark", "dark"),
    ("high-contrast", "high-contrast", "light"),
    ("neon", "neon", "dark"),
]

AA = 4.5

pytestmark = pytest.mark.integration

_RATIO_JS = """
  const rgb = (s) => {
    const m = (s || '').match(/-?[\\d.]+/g);
    return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : null;
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
  };
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
    return (x + 0.05) / (y + 0.05);
  };
"""


def _page_local_properties(name: str) -> list[str]:
    """The custom properties this page declares for itself, read from the page.

    Read rather than listed, because a list in a test is a second copy of the palette and the
    second copy is the one that drifts. The shared six-theme tokens are excluded by construction:
    they are declared in `/_shared/tokens.css`, not here.
    """
    text = (EXERCISE / "web" / name).read_text(encoding="utf-8")
    head = text.split("* { box-sizing", 1)[0]
    return sorted(set(re.findall(r"(--[a-z][a-z0-9-]*)\s*:", head)))


@pytest.fixture(scope="module")
def site():
    if not (PUBLIC / SLUG / "index.html").is_file():
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


def _open(site, name: str, attr: str | None = None, scheme: str = "light"):
    browser, base = site
    ctx = browser.new_context(color_scheme=scheme, viewport={"width": 1400, "height": 950})
    page = ctx.new_page()
    problems: list[str] = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.goto(f"{base}/{SLUG}/{name}", wait_until="networkidle", timeout=25_000)
    if attr:
        page.evaluate(f"document.documentElement.setAttribute('data-theme', {attr!r})")
    page.wait_for_timeout(700)
    return ctx, page, problems


@pytest.mark.parametrize("name", PAGES)
def test_the_page_loads_without_throwing(site, name: str) -> None:
    """The failure that killed `s3.html` outright, and the one a file-level check cannot see."""
    ctx, _, problems = _open(site, name)
    try:
        assert not problems, (
            f"{name} logged errors on load: {problems[:3]}\n\n"
            "A page script that throws stops at its first statement, so the symptom is an empty "
            "page rather than a broken feature. `Identifier '<x>' has already been declared` means "
            "a `var` at the top level of one classic script collided with a `let` in another: "
            "both target the global scope. Wrap the bootstrap so it leaks nothing."
        )
    finally:
        ctx.close()


#: The pages that declare a palette of their own. Computed rather than listed, and used to
#: parametrise instead of skipping inside the test: in CI a skip reports as a pass, so a page that
#: quietly stopped declaring anything would read as covered. A case that never existed cannot.
PALETTED = [n for n in PAGES if _page_local_properties(n)]


def test_the_paletted_pages_are_the_ones_expected() -> None:
    """`PALETTED` drives a parametrisation, so an empty one would be green and worthless."""
    assert PALETTED == ["s1.html", "s2.html", "s3.html", "s4.html"], (
        f"the four proof pages each declare their own palette; found {PALETTED}. If a page "
        "legitimately dropped its palette, change this list deliberately — do not let the "
        "parametrisation below shrink without anyone deciding it should."
    )


@pytest.mark.parametrize("name", PALETTED)
@pytest.mark.parametrize("theme,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_every_page_local_property_resolves_in_this_theme(
    site, name: str, theme: str, attr, scheme: str
) -> None:
    """Declared for the dark themes only is declared for half the readers only."""
    wanted = _page_local_properties(name)
    ctx, page, _ = _open(site, name, attr, scheme)
    try:
        unresolved = page.evaluate(
            """(names) => {
                 const s = getComputedStyle(document.body);
                 return names.filter((n) => !s.getPropertyValue(n).trim());
               }""",
            wanted,
        )
        assert not unresolved, (
            f"{name} under {theme} resolves these to nothing: {unresolved}.\n\n"
            "A custom property declared only inside a `prefers-color-scheme: dark` block does not "
            "exist on the light themes. Nothing errors: a CSS background is simply never painted, "
            "and a canvas fill silently keeps whatever was set last."
        )
    finally:
        ctx.close()


@pytest.mark.parametrize("theme,attr,scheme", THEMES, ids=[t[0] for t in THEMES])
def test_the_category_chips_are_legible_in_this_theme(site, theme: str, attr, scheme: str) -> None:
    """Each chip's label against **its own** swatch, which is set from JavaScript.

    That is why this is measured and not read: the background never appears in any rule, so no
    amount of reading the stylesheet can tell you what the label is sitting on.
    """
    ctx, page, _ = _open(site, "s3.html", attr, scheme)
    try:
        rows = page.evaluate(
            _RATIO_JS
            + """
            () => [...document.querySelectorAll('.chip')].map((el) => {
              const s = getComputedStyle(el);
              let node = el, bg = s.backgroundColor;
              while (/rgba\\(0, 0, 0, 0\\)/.test(bg) && node.parentElement) {
                node = node.parentElement;
                bg = getComputedStyle(node).backgroundColor;
              }
              const ink = rgb(s.color), ground = rgb(bg);
              return { word: el.textContent.trim(), r: ink && ground ? ratio(ink, ground) : null };
            })"""
        )
        assert len(rows) >= 8, (
            f"only {len(rows)} chip(s) rendered, so this assertion is passing over almost nothing. "
            "The page script has probably thrown — check the load test above first."
        )
        bad = [f"{r['word']} {r['r']:.2f}:1" for r in rows if r["r"] is None or r["r"] < AA]
        assert not bad, (
            f"under {theme} these chip labels are below WCAG AA of {AA}:1: {bad}.\n\n"
            "A ratio at or near 1.00:1 means the swatch was never painted — the category palette "
            "is missing for this theme, so the label is sitting on the page itself."
        )
    finally:
        ctx.close()


def test_the_chip_check_can_actually_fail(site) -> None:
    """Break it on purpose: strip the swatches and watch the labels collapse onto the page.

    This is not a hypothetical shape. It is exactly what every light-theme reader saw, and the
    ratio it produces is the one this measured before the palette was fixed.
    """
    ctx, page, _ = _open(site, "s3.html")
    try:
        page.evaluate("""() => document.querySelectorAll('.chip')
                           .forEach((el) => { el.style.background = 'transparent';
                                              el.style.color = '#fff'; })""")
        page.wait_for_timeout(150)
        worst = page.evaluate(
            _RATIO_JS
            + """
            () => Math.min(...[...document.querySelectorAll('.chip')].map((el) => {
              const s = getComputedStyle(el);
              let node = el, bg = s.backgroundColor;
              while (/rgba\\(0, 0, 0, 0\\)/.test(bg) && node.parentElement) {
                node = node.parentElement;
                bg = getComputedStyle(node).backgroundColor;
              }
              return ratio(rgb(s.color), rgb(bg));
            }))"""
        )
        assert worst < AA, (
            f"white labels on unpainted chips measured {worst:.2f}:1, which the checker did not "
            "report as a failure — so it cannot detect the defect it was written for"
        )
    finally:
        ctx.close()


def test_the_load_check_can_actually_fail(site) -> None:
    """Break it on purpose, with the exact collision that killed `s3.html`.

    The bug is a *parse* error, so it cannot be injected into a page that has already parsed —
    the two scripts have to arrive together. `setContent` serves them, which tests the detector
    rather than the page: does a listener wired this way actually see a top-level SyntaxError?
    Before this, nothing in the exercise did.
    """
    browser, _ = site
    ctx = browser.new_context()
    page = ctx.new_page()
    problems: list[str] = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    try:
        page.set_content(
            "<script>var t = 1;</script><script>let t, other; document.title = 'ran';</script>"
        )
        page.wait_for_timeout(200)
        assert problems, (
            "a `var t` global colliding with a later `let t` produced no reported error, so the "
            "listener above cannot see the failure this file exists for"
        )
        assert "already been declared" in problems[0], (
            f"something else was reported instead: {problems[0]}"
        )
        assert page.title() != "ran", (
            "the second script ran despite the collision, so this fixture is not reproducing the "
            "failure — the whole point is that the script never starts"
        )
    finally:
        ctx.close()


# ---------------------------------------------------------------- the shared Tier-1 invariants

#: Three widths, because every layout defect this repo has shipped appeared at some and not others.
INVARIANT_WIDTHS = (1440, 900, 390)


@pytest.mark.parametrize("name", PAGES)
@pytest.mark.parametrize("width", INVARIANT_WIDTHS)
def test_the_page_passes_every_tier_one_invariant(site, name: str, width: int) -> None:
    """`tests/_page_invariants.py`, pointed at this exercise for the first time.

    **That module was built to be shared and had exactly one consumer** — the landing page. Every
    other exercise hand-rolled its own `scrollWidth` comparison, and this one had nothing at all,
    so the checks it offers for free (no console or page error, no failed request, nothing
    overflowing its own box, no text painted its own background, no image without real dimensions)
    were never asked of these five pages.

    They pass now. They would not have: the whole point of the first check is a page that throws
    while building itself, which is precisely what `s3.html` did.
    """
    browser, base = site
    ctx = browser.new_context(viewport={"width": width, "height": 950})
    page = ctx.new_page()
    try:
        seen = attach(page)
        page.goto(f"{base}/{SLUG}/{name}", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(700)
        problems = findings(page, seen)
        assert not problems, f"{name} at {width}px fails Tier-1 checks:\n  " + "\n  ".join(
            problems[:6]
        )
    finally:
        ctx.close()


def test_the_tier_one_checks_are_actually_reaching_this_exercise(site) -> None:
    """The twin: plant a defect, confirm it is reported, remove it in a `finally`.

    Without this the test above is indistinguishable from one that imported the module and called
    nothing — which is the failure mode `AGENTS.md` calls a tested feature with no caller, one level
    up. The mutation is removed on the way out rather than at the end of the happy path, so an
    early return or an exception cannot leave the page rewritten for whatever runs next.
    """
    browser, base = site
    ctx = browser.new_context(viewport={"width": 1200, "height": 800})
    page = ctx.new_page()
    try:
        seen = attach(page)
        page.goto(f"{base}/{SLUG}/s1.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(400)
        assert not findings(page, seen), "the page is not clean to begin with; fix that first"
        page.evaluate(
            """() => {
                 const box = document.createElement('div');
                 box.id = 'planted-overflow';
                 box.style.cssText = 'width:120px;overflow:hidden;white-space:nowrap';
                 box.textContent = 'x'.repeat(400);
                 document.body.appendChild(box);
               }"""
        )
        page.wait_for_timeout(100)
        problems = findings(page, seen)
        assert any("overflows its own box" in p for p in problems), (
            f"a 400-character line inside a 120px box was not reported: {problems}"
        )
    finally:
        page.evaluate("() => document.getElementById('planted-overflow')?.remove()")
        ctx.close()
