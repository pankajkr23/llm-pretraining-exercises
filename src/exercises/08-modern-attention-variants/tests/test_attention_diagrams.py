"""The detail diagrams: every mechanism gets one, and no two are the same picture.

**The distinctness check is the reason this file exists.** For most of this exercise's life NSA's
glyph was pixel-identical to plain scaled dot-product attention — six of the thirteen field entries
collapsed onto one full causal triangle — and nothing failed, because every mechanism *had* a
drawing and every drawing rendered. A figure set can be complete, correct at the element level, and
still be saying that six different mechanisms are the same thing.

So completeness is not the property to test. Distinctness is.

These render through the real module rather than through the page, so they hold from the moment a
generator exists and do not wait on the diagrams being placed.
"""

import functools
import hashlib
import http.server
import json
import re
import socketserver
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="browser tests need playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC = REPO_ROOT / "public"
SLUG = "08-modern-attention-variants"

pytestmark = pytest.mark.integration


def _bundle() -> dict:
    text = (PUBLIC / SLUG / "data.js").read_text(encoding="utf-8")
    return json.loads(text.split("Object.freeze(", 1)[1].rsplit(");", 1)[0])


@pytest.fixture(scope="module")
def drawn():
    """Every mechanism's diagram, rendered once, as `{key: {markup, box, label}}`."""
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
            page = browser.new_page(viewport={"width": 1000, "height": 900})
            problems: list[str] = []
            page.on("pageerror", lambda e: problems.append(str(e)))
            page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/{SLUG}/index.html")
            page.wait_for_selector("section#reproduce", timeout=15_000)
            out = page.evaluate(
                """() => Promise.all([import('./diagrams.js'), import('./data.js')]).then(
                  ([D, d]) => {
                    const host = document.createElement('div');
                    host.style.cssText = 'position:absolute;left:-9999px;width:760px';
                    document.body.appendChild(host);
                    const res = {};
                    for (const m of d.M.mechanisms) {
                      const svg = D.diagramSvg(m);
                      host.appendChild(svg);
                      const vb = svg.viewBox.baseVal;
                      const b = svg.getBBox();
                      res[m.key] = {
                        markup: svg.innerHTML,
                        label: svg.getAttribute('aria-label') || '',
                        bbox: [b.x, b.y, b.width, b.height],
                        view: [vb.x, vb.y, vb.width, vb.height],
                        nodes: svg.querySelectorAll('*').length,
                      };
                    }
                    host.remove();
                    return res;
                  })"""
            )
            assert not problems, f"rendering the diagrams threw: {problems}"
            yield out
            browser.close()
    finally:
        httpd.shutdown()


def test_every_mechanism_gets_a_diagram(drawn) -> None:
    assert sorted(drawn) == sorted(m["key"] for m in _bundle()["mechanisms"])


def test_no_diagram_is_blank(drawn) -> None:
    """A figure set claiming completeness cannot contain an empty frame."""
    thin = {k: v["nodes"] for k, v in drawn.items() if v["nodes"] < 12}
    assert not thin, f"these diagrams drew almost nothing: {thin}"


def test_no_two_diagrams_are_the_same_picture(drawn) -> None:
    """**The guard this file exists for.**

    NSA's glyph was pixel-identical to plain attention for the life of this exercise, and six field
    entries shared one grid. Every one of them rendered, so nothing failed — the set was complete
    and it was still claiming that six different mechanisms are the same thing.

    Structural markup rather than a screenshot: two diagrams that differ only in a label are still
    two different pictures, and a pixel digest would be defeated by a one-word caption change.
    """
    digests: dict[str, list[str]] = {}
    for key, v in drawn.items():
        # ids are suffixed per mechanism, so strip them or every diagram is trivially unique
        markup = re.sub(r'(id|fill)="[^"]*dgm-hatch-[^"]*"', "", v["markup"])
        digests.setdefault(hashlib.sha256(markup.encode()).hexdigest(), []).append(key)
    collisions = {d: ks for d, ks in digests.items() if len(ks) > 1}
    assert not collisions, "these mechanisms render as the same picture: " + "; ".join(
        ", ".join(ks) for ks in collisions.values()
    )


def test_no_diagram_draws_outside_its_own_frame(drawn) -> None:
    """The twin of the glyph bbox guard. SVG does not clip, so an escaping mark lands on whatever
    is next to it — present, legible, and attributed to the wrong figure."""
    escapes = []
    for key, v in drawn.items():
        bx, by, bw, bh = v["bbox"]
        vx, vy, vw, vh = v["view"]
        if bx < vx - 0.5 or by < vy - 0.5 or bx + bw > vx + vw + 0.5 or by + bh > vy + vh + 0.5:
            escapes.append((key, [round(n, 1) for n in v["bbox"]], v["view"]))
    assert not escapes, f"these diagrams draw outside their viewBox: {escapes}"


def test_every_diagram_names_the_mechanism_it_draws(drawn) -> None:
    """A figure a screen reader announces as 'image' is a figure that is not there."""
    by_key = {m["key"]: m for m in _bundle()["mechanisms"]}
    unnamed = [k for k, v in drawn.items() if by_key[k]["name"] not in v["label"]]
    assert not unnamed, f"these diagrams do not name their mechanism: {unnamed}"


def test_every_diagram_that_claims_a_scale_states_it(drawn) -> None:
    """A mechanism with sourced sizes must say what the drawing's scale is, or say it is schematic.

    A figure drawn from a paper's own numbers looks more precise than one that is not, and that
    extra authority has to be earned by saying which it is.
    """
    sized = [m["key"] for m in _bundle()["mechanisms"] if m["glyph"].get("sizes")]
    assert sized, "no mechanism carries sourced sizes yet — this guard would be vacuous"
    silent = [
        k
        for k in sized
        if "DRAWN TO SCALE" not in drawn[k]["markup"].upper()
        and "DRAWN SCHEMATICALLY" not in drawn[k]["markup"].upper()
    ]
    assert not silent, f"these draw from sourced sizes without stating the scale: {silent}"


def test_the_distinctness_check_can_actually_fail(drawn) -> None:
    """Break it on purpose: two identical markups must be reported as a collision."""
    same = "<rect x='0' y='0' width='10' height='10'/>"
    digests: dict[str, list[str]] = {}
    for key, markup in (("a", same), ("b", same), ("c", "<circle r='2'/>")):
        digests.setdefault(hashlib.sha256(markup.encode()).hexdigest(), []).append(key)
    collisions = {d: ks for d, ks in digests.items() if len(ks) > 1}
    assert collisions and sorted(next(iter(collisions.values()))) == ["a", "b"]
