"""A table of sentences fits its container. A table of figures may scroll.

`table.grid` sets `white-space: nowrap` on every cell but the first — right for a column of numbers,
catastrophic for a column of prose. Marking a table `prose` is how a page says its cells wrap, and
this asserts that they actually do.

**The page-level overflow guard is blind to this and cannot be fixed to see it.** Every table sits
in a `.tablewrap` with `overflow-x: auto`, which is exactly what `AGENTS.md` asks for: wide content
scrolls in its own container rather than pushing the page sideways. A cell hidden inside that scroll
is, to any generic check, indistinguishable from a wide table behaving correctly. The element-level
check has the same hole from the other side — it skips anything whose computed `overflow-x` is
`auto`, which is precisely `.tablewrap`, and the `td` itself grows to its content so its own
`scrollWidth` equals its `clientWidth`.

**Exercise 09 hit this, fixed it, wrote a guard — and exercise 10 then copied the markup without the
fix.** Its `chapters.js` builds the audit table with the same `'grid prose'` class list while no
stylesheet it links defined `.prose` at all, so the modifier was inert and looked deliberate. The
table laid out **3,635px wide inside a 1,156px wrapper**, cells of up to 433 characters each on one
unwrapped line, on the section whose own standfirst reads *"the gap between the two columns is the
whole argument"*.

That is the third single-exercise guard this week to miss the exercise next door, so it is repo-wide
now and discovers its pages from the filesystem.

**And a table can fail the same reader in the other direction.** Wrapping every cell fixed the
desktop and left the phone worse in a new shape: exercise 10's six tables measured **8,455px** of
vertical space at 390px — ten screens — with prose cells at **12 to 19 characters a line**. A
3,635px-wide table of 330-character lines and a 4,959px-tall one of 13-character lines are not
better and worse; they are the same content failing in two directions. So the second half of this
file measures a phone, using the qualifier the prose-measure guard already proved necessary: a short
line is only a defect when it is *also* leaving room unused. A 36-character cell filling a 390px
phone is the phone; a 16-character cell in a 340px box is a column that should have stopped being
one.
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

#: Widths where the wrapper is tightest. 1180 is where `page.css` starts reserving the rail gutter,
#: so the content box is narrower there than at 1179 — the width that has already produced two
#: separate layout defects in this repository.
WIDTHS = (1440, 1280, 1180, 900)

OFFENDERS_JS = """() => {
  const bad = [];
  for (const t of document.querySelectorAll('table.prose')) {
    const wrap = t.closest('.tablewrap') || t.parentElement;
    if (!wrap) continue;
    if (t.scrollWidth > wrap.clientWidth + 1) {
      const section = t.closest('section');
      const widest = Math.max(...[...t.querySelectorAll('td')].map((td) => td.innerText.length));
      bad.push(
        (section ? section.id : '(no section)') +
          ': table ' + t.scrollWidth + 'px in a ' + wrap.clientWidth +
          'px wrapper, longest cell ' + widest + ' characters'
      );
    }
  }
  return bad;
}"""


def _deployable() -> list[str]:
    """From the filesystem, so a new page is covered the day it ships."""
    return sorted(p.parent.parent.name for p in REPO_ROOT.glob("src/exercises/*/web/index.html"))


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


def _tables(site, slug: str, width: int) -> tuple[int, list[str]]:
    """`(how many prose tables the page has, which of them overflow)` at one width."""
    browser, base = site
    page = browser.new_page(viewport={"width": width, "height": 950})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(400)
        count = page.evaluate("() => document.querySelectorAll('table.prose').length")
        return count, page.evaluate(OFFENDERS_JS)
    finally:
        page.close()


@pytest.mark.parametrize("slug", _deployable())
def test_a_prose_table_fits_its_container(site, slug: str) -> None:
    """One case per page, sweeping the widths inside it.

    A page with no prose table passes, which is honest: most pages have none, and asserting that
    each *has* one would be a different rule about a different thing. The vacuity guard below covers
    what matters — that the repository has some for this to be checking.
    """
    failures = []
    for width in WIDTHS:
        _, offenders = _tables(site, slug, width)
        failures.extend(f"{width}px — {line}" for line in offenders)

    assert not failures, (
        f"{slug} has a prose table wider than its container, so part of every row sits behind a "
        "scrollbar:\n  " + "\n  ".join(failures) + "\nA table marked `prose` is one whose cells "
        "are sentences, and `table.grid` sets `white-space: nowrap` on every cell but the first. "
        "The page needs `table.grid.prose td { white-space: normal }` — check it is defined in a "
        "stylesheet this page actually links, because the class is inert without it and an inert "
        "modifier looks deliberate."
    )


def test_some_page_has_a_prose_table(site) -> None:
    """Otherwise every case above passes by having nothing to measure.

    This is the half that fails if the class is renamed or the tables stop being marked — at which
    point the sweep goes quietly green across every page.
    """
    total = sum(_tables(site, slug, WIDTHS[0])[0] for slug in _deployable())
    assert total >= 2, (
        f"only {total} prose table(s) across every deployed page. Either the pages have stopped "
        "using them, or `table.prose` has stopped being how one is marked — and the second is far "
        "more likely, at which point this file measures nothing."
    )


#: `docs/DESIGN.md`'s floor. A line below this is hard to read; the guard above the file uses the
#: same number for running prose.
NARROWEST = 42

#: A narrow cell is only a defect if it is ALSO leaving room unused. Copied from
#: `test_prose_measure_repo_wide.py`, deliberately, including the reason: at 390px a stacked cell is
#: 36 characters because the phone is 390px wide, and a rule without this qualifier would demand 42
#: characters in a container that cannot hold them. The same 16 characters beside an empty
#: half-container is the defect.
ROOM_TO_SPARE = 0.70

#: Phone widths. 390 is a current handset, 320 the narrowest still worth supporting and the width at
#: which an auto-fit track that cannot shrink pushes a page sideways.
PHONE_WIDTHS = (390, 320)

#: **Registers: tables nobody should stack, named one at a time with what makes each one.**
#:
#: The first version of the sweep below flagged exercise 03's dataset catalogue and it was wrong to.
#: That table is a **reference register** — 109 rows, 7 columns, and exactly **one** body cell in
#: 763 long enough to count as prose, because a dataset happens to be called *"Internet Archive -
#: Public Library of India"*. It is read by scanning a column, never by reading a row, and stacking
#: it would turn 109 rows into 109 cards. `_shared/page.css` already gives it `overflow-x: auto`,
#: which is what `AGENTS.md` asks of wide content.
#:
#: **The fill qualifier cannot see this, and no threshold fixes it.** `fill` compares a cell against
#: its container, so a 7-column table gives every cell a low share by construction — the room is not
#: unused, six other columns are using it. Widening `ROOM_TO_SPARE` to cover the case would excuse
#: genuinely narrow cells everywhere else. So the exemption is a **named decision** rather than a
#: number, and `test_the_register_ledger_has_not_gone_stale` fails in both directions: if this table
#: stops being a register, or another one becomes one, the ledger is wrong and says so.
#:
#: Keyed by `(page, section)`. **Adding an entry to clear a red gate is forbidden** — the reason has
#: to be a property of the table, not of the test result.
REGISTERS: dict[tuple[str, str], str] = {
    ("03-data-collection-framework", "datasets"): (
        "the dataset catalogue: 109 rows and 7 columns of values, with one long proper noun among "
        "763 cells. A register is scanned down a column; stacked it is 109 cards"
    ),
}

#: What makes a register, measured rather than asserted. Both hold for the entry above (109 rows,
#: a prose share of 0.001) and neither holds for any other table in the repository — the next
#: longest is 9 rows and the next lowest share 0.056.
REGISTER_ROWS = 25
REGISTER_PROSE_SHARE = 0.02

CELLS_JS = """() => {
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre';
  document.body.appendChild(probe);
  const chw = (el) => {
    const cs = getComputedStyle(el);
    probe.style.font = cs.font || (cs.fontSize + ' ' + cs.fontFamily);
    probe.textContent = '0'.repeat(100);
    return probe.getBoundingClientRect().width / 100;
  };
  const out = [];
  for (const table of document.querySelectorAll('table')) {
    // Measured per table rather than per cell, because whether a cell is too narrow to read is a
    // question about the cell and whether the table should have stacked is a question about the
    // table. The register ledger needs the second, and a `td` cannot answer it.
    const body = [...table.querySelectorAll('tbody td')].filter(
      (td) => td.getBoundingClientRect().width >= 4
    );
    if (!body.length) continue;
    const prose = body.filter((td) => td.innerText.trim().length >= 60);
    const shape = {
      rows: table.querySelectorAll('tbody tr').length,
      cells: body.length,
      prose: prose.length,
      share: prose.length / body.length,
      section: (table.closest('section') || {}).id || '(none)',
    };
    for (const td of prose) {
      const box = td.getBoundingClientRect();
      const host = table.parentElement;
      const room = host ? host.getBoundingClientRect().width : 0;
      out.push({
        ...shape,
        chars: Math.round(box.width / chw(td)),
        fill: room ? box.width / room : 1,
        // A cell that has stopped being a column: the table stacked, so this is a card. `hidden` is
        // how much of the sentence is past the right edge of its own box.
        stacked: getComputedStyle(td).display === 'block',
        hidden: td.scrollWidth - td.clientWidth,
        head: td.dataset.head || '',
        sample: td.innerText.trim().slice(0, 40),
      });
    }
  }
  probe.remove();
  return out;
}"""


@pytest.mark.parametrize("slug", _deployable())
def test_a_table_cell_of_prose_is_readable_on_a_phone(site, slug: str) -> None:
    """A table that cannot be narrow must stop being a table, not become a ribbon.

    **Two failures, and the second was found by looking at a page this file had just passed.**

    The first is a cell laid out at a width no sentence can be read in while the container beside it
    sits empty — no overflow, nothing clipped, the page not scrolling sideways. That is why the fill
    qualifier matters as much as the floor.

    The second is the opposite and far quieter. Stacking a table makes each cell a full-width block,
    so it *measures* 35 characters at 99% of the container and passes — while `white-space: nowrap`,
    correct for the columns that no longer exist, runs the text straight past the right edge. On
    exercise 07 one cell laid out **3,222px of sentence in a 337px box**: nine tenths of it
    invisible, with no ellipsis and no scrollbar, and the `overflow-x: auto` on the wrapper above
    making the whole thing read as a wide table behaving correctly. Seven such cells on 07 and ten
    on 09 were live while every assertion here was green, which is `AGENTS.md`'s rule that a guard
    asserting an element is *visible* has not asserted it is *legible*.

    `tests/_page_invariants.py` cannot see it either, deliberately: it flags only elements whose own
    `overflow-x` is `hidden` or `clip`, and a stacked `td` is `visible` — it spills into an ancestor
    that scrolls on purpose. A card has no columns, so text wider than the card is never right, and
    that is a property this file is the right place to assert.
    """
    browser, base = site
    failures = []
    for width in PHONE_WIDTHS:
        page = browser.new_page(viewport={"width": width, "height": 844})
        try:
            page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
            page.wait_for_timeout(400)
            for cell in page.evaluate(CELLS_JS):
                if (slug, cell["section"]) in REGISTERS:
                    continue
                where = f"{width}px — {cell['section']}" + (
                    f" (under '{cell['head']}')" if cell["head"] else ""
                )
                if cell["chars"] < NARROWEST and cell["fill"] < ROOM_TO_SPARE:
                    failures.append(
                        f"{where}: {cell['chars']} characters a line in "
                        f"{cell['fill']:.0%} of the container — {cell['sample']!r}"
                    )
                # 1px of slack for sub-pixel rounding, the same allowance `_page_invariants` uses.
                if cell["stacked"] and cell["hidden"] > 1:
                    failures.append(
                        f"{where}: {cell['hidden']}px of the sentence is past the right edge of a "
                        f"stacked cell {cell['chars']} characters wide — {cell['sample']!r}"
                    )
        finally:
            page.close()

    assert not failures, (
        f"{slug} lays out a sentence in a table cell too narrow to read, with room beside it:\n  "
        + "\n  ".join(failures[:8])
        + "\nA table whose columns are sentences cannot be narrow. Below a phone breakpoint it "
        "should stop being a table: each row a card, each cell a labelled block, with the column "
        "head carried on the cell so a reader still knows which column a line belongs to — and "
        "`white-space: normal` on the stacked cell, because the `nowrap` that was right for a "
        "column of figures runs a sentence off the edge of a card with nothing to show for it."
    )


def _shapes(site, slug: str) -> dict[str, dict]:
    """Every table on one page, by section, with the shape the ledger reasons about."""
    browser, base = site
    page = browser.new_page(viewport={"width": PHONE_WIDTHS[0], "height": 844})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(400)
        found: dict[str, dict] = {}
        for cell in page.evaluate(CELLS_JS):
            found.setdefault(cell["section"], cell)
        return found
    finally:
        page.close()


def test_the_register_ledger_has_not_gone_stale(site) -> None:
    """Both directions, because an exemption is only honest while its reason is still true.

    A stale entry here is worse than a missing guard: it silently excuses a table that has since
    become three rows of sentences, and nothing anywhere would say so. So this asserts the *reason*
    — many rows, prose a rarity — rather than the fact that the entry exists.
    """
    wrong = []
    for (slug, section), reason in REGISTERS.items():
        shapes = _shapes(site, slug)
        table = shapes.get(section)
        if table is None:
            wrong.append(f"{slug}:{section} has no table with a prose cell any more ({reason})")
            continue
        if table["rows"] < REGISTER_ROWS or table["share"] > REGISTER_PROSE_SHARE:
            wrong.append(
                f"{slug}:{section} is no longer a register — {table['rows']} rows and "
                f"{table['prose']} prose cells in {table['cells']} "
                f"({table['share']:.1%}). Exempted because: {reason}"
            )

    # And the other direction: a table that IS a register and is not named here is one the sweep
    # will flag the first time a long value lands in it, which is a red gate with no decision behind
    # it. Naming it is cheap; discovering it during an unrelated pull request is not.
    for slug in _deployable():
        for section, table in _shapes(site, slug).items():
            if (slug, section) in REGISTERS:
                continue
            if table["rows"] >= REGISTER_ROWS and table["share"] <= REGISTER_PROSE_SHARE:
                wrong.append(
                    f"{slug}:{section} looks like a register ({table['rows']} rows, "
                    f"{table['share']:.1%} prose) and is not in REGISTERS"
                )

    assert not wrong, "the register ledger disagrees with the pages:\n  " + "\n  ".join(wrong)


EMPTY_LABEL_JS = """() => {
  const bad = [];
  for (const td of document.querySelectorAll('table td')) {
    const label = getComputedStyle(td, '::before').content;
    if (label !== '""' && label !== "''") continue;
    bad.push(
      ((td.closest('section') || {}).id || '(none)') +
        ': ' + td.innerText.trim().slice(0, 40)
    );
  }
  return bad;
}"""


@pytest.mark.parametrize("slug", _deployable())
def test_no_stacked_cell_prints_an_empty_label(site, slug: str) -> None:
    """`content: attr(data-head)` on an empty attribute paints an empty line, not nothing.

    Exercise 07 builds key/value tables from `['', '']`, so writing the head onto every cell gave
    four of them `data-head=""` — which still matches `td[data-head]`, so each printed a blank
    labelled line above its value: 3px of margin and a gap where a column name should be. The
    builder now sets the attribute only when there is a label, and this is the assertion that says
    so, because the defect is invisible to every other check here — the cell is the right width,
    wraps correctly, hides nothing, and reads as slightly loose spacing.
    """
    browser, base = site
    page = browser.new_page(viewport={"width": PHONE_WIDTHS[0], "height": 844})
    try:
        page.goto(f"{base}/{slug}/index.html", wait_until="networkidle", timeout=25_000)
        page.wait_for_timeout(400)
        empty = page.evaluate(EMPTY_LABEL_JS)
    finally:
        page.close()

    assert not empty, (
        f"{slug} prints an empty column label above {len(empty)} cell(s):\n  "
        + "\n  ".join(empty[:8])
        + "\nSet `data-head` only when the header has text — an empty `attr()` still paints."
    )
