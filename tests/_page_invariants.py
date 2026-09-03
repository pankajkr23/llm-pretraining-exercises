"""Deterministic checks any rendered page must pass, whatever it is about.

**These are the cheapest useful visual tests and the ones most often skipped.** The instinct is to
reach for screenshot diffing, which is expensive to maintain and flakes on font rendering. The
checks here never flake, need no baseline, and catch the class of failure an agent actually
produces: a page that throws half way through building itself, a stylesheet that 404s, a sentence
truncated at every width, a mark painted in a colour identical to its own background.

The alternative is worse than it looks. **DiffSpot** benchmarked 13 vision models on spotting
fine-grained differences in *web interfaces*: the best managed **47.2% accuracy and 40.7% recall**,
and on 500 pairs that were identical the aggressive models reported a difference on up to **24.2%**.
A model cannot gate a page. These can.

Every check here came from a defect this repo actually shipped:

- **No console or page errors.** Deleting a conditional took the `const body = …` line above it
  with it; the page threw half way through building its index and thirty rows became none.
- **No failed requests.** A page that loads while its data file 404s renders empty, and passes
  every markup test written about it.
- **Nothing overflows its own box.** An invoice cut line read *"…the cache alone needs a second
  ma"* at every width narrower than the sentence — for as long as the figure existed, with
  `test_the_invoice_cut_line_is_visible` passing throughout. Visible is not legible.
- **Text is not its own background.** Four contrast failures shipped here by being chosen by eye,
  and under `high-contrast` both `--muted` and `--ink` are `#000000`.
- **Images have real dimensions.** A broken `src` renders as nothing and asserts as present.

**Use `attach(page)` before navigating**, because console and network listeners only see what
happens after they are attached — the errors thrown while a page builds itself are the ones worth
having, and they are all emitted during load.
"""

from dataclasses import dataclass, field

#: How much a box may overflow before it counts. One pixel is sub-pixel rounding, not a defect.
OVERFLOW_SLACK = 1


@dataclass
class Recorder:
    """Console errors, page errors and failed requests seen since `attach`."""

    console: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)


def attach(page) -> Recorder:
    """Start recording. Call **before** `goto`, or the load-time failures are missed."""
    seen = Recorder()
    page.on("console", lambda m: seen.console.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: seen.errors.append(str(e)))
    page.on(
        "requestfailed",
        lambda r: seen.requests.append(f"{r.url} ({r.failure})"),
    )
    page.on(
        "response",
        lambda r: seen.requests.append(f"{r.url} -> {r.status}") if r.status >= 400 else None,
    )
    return seen


#: One pass over the DOM, in the page, because a round trip per element is unusably slow on a page
#: with a few thousand nodes. Returns plain data so the assertions live in Python where they read.
_MEASURE = """() => {
  const overflowing = [];
  const invisible = [];
  const broken = [];

  const label = (e) => {
    const id = e.id ? `#${e.id}` : '';
    const cls = typeof e.className === 'string' && e.className
      ? '.' + e.className.trim().split(/\\s+/).slice(0, 2).join('.')
      : '';
    return `${e.tagName.toLowerCase()}${id}${cls}`;
  };

  for (const e of document.querySelectorAll('*')) {
    const style = getComputedStyle(e);
    if (style.display === 'none' || style.visibility === 'hidden') continue;

    // Only elements that actually clip. A scrolling container is a deliberate choice and the
    // repo's own conventions ask for it on wide tables and diagrams.
    const clips = style.overflowX === 'hidden' || style.overflowX === 'clip';
    if (clips && e.scrollWidth > e.clientWidth + SLACK) {
      overflowing.push(`${label(e)} scrollWidth ${e.scrollWidth} > clientWidth ${e.clientWidth}`);
    }

    // Text the same colour as what it sits on. Walk up for the first non-transparent background,
    // because an element with a transparent background shows its ancestor's.
    const text = Array.from(e.childNodes)
      .filter((n) => n.nodeType === 3 && n.textContent.trim())
      .map((n) => n.textContent.trim())
      .join(' ');
    if (text) {
      let bg = style.backgroundColor;
      let up = e.parentElement;
      while (up && (bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent')) {
        bg = getComputedStyle(up).backgroundColor;
        up = up.parentElement;
      }
      if (bg === style.color) {
        invisible.push(`${label(e)} text and background are both ${bg}: ${text.slice(0, 40)}`);
      }
    }
  }

  for (const img of document.images) {
    if (!img.complete || img.naturalWidth === 0) {
      broken.push(`${img.getAttribute('src') || '(no src)'}`);
    }
  }
  return { overflowing, invisible, broken };
}""".replace("SLACK", str(OVERFLOW_SLACK))


def findings(page, seen: Recorder) -> list[str]:
    """Everything wrong with this page right now. Empty means it passes every Tier-1 check.

    Args:
        page: A Playwright page, already navigated.
        seen: The recorder returned by `attach`, from before navigation.

    Returns:
        Human-readable findings, each naming the element or URL involved.
    """
    found: list[str] = []
    found += [f"page error: {e}" for e in seen.errors]
    found += [f"console error: {c}" for c in seen.console]
    found += [f"failed request: {r}" for r in seen.requests]

    measured = page.evaluate(_MEASURE)
    found += [f"overflows its own box: {o}" for o in measured["overflowing"]]
    found += [f"invisible text: {i}" for i in measured["invisible"]]
    found += [f"image did not load: {b}" for b in measured["broken"]]
    return found
