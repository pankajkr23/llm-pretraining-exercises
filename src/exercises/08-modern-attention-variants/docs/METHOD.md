# How the page is made

The apparatus, moved off the page so the page can be about attention.

The published colophon carries the three claims the *numbers* rest on — how a date was read, how a
byte figure was computed, and what an entry has to state before the catalogue will accept it. This
file carries everything a reader would only want if they were rebuilding the thing: how the page is
generated, how the figures are drawn, how the six themes work, and what to run.

It exists because the colophon reached 358 words and seven paragraphs, and five of six review
readers stalled in it. One called its closing paragraph "internal repo politics being litigated in
front of a stranger", which was fair: the paragraph argued about which README should hold commands.
That argument is worth having and it is not worth a reader's seventh screen.

## Nothing on the page is typed by hand

Every date, count, byte figure and trade-off rendered by `web/chapters.js` is generated from
`results/mechanisms.json` by `tools/build_web_data.py`, through the same Python functions the test
suite exercises. `web/data.js` is that output and is regenerated, never edited.

The rule is enforced rather than intended.
`tests/test_attention_docs.py::test_no_count_is_typed_into_the_page_as_a_word` reads the page source
and fails on a spelled number inside a string literal — because a number inside a `<script>` block
is read far more often than any file in the repo and tested by none of them. Where a literal is
genuinely fixed by construction (the 6×6 demo grid, a duration in months) the line carries a
`// count-literal-ok:` marker saying why.

```bash
uv run python tools/build_web_data.py      # results/mechanisms.json -> web/data.js
```

## Every figure is inline SVG

There is no chart library on this page and no third-party request of any kind — no fonts, no
analytics, no CDN. Each figure is a function in `web/figures.js` or `web/diagrams.js` that returns
an SVG element built from `M`, the same bundle the prose reads. A figure and the sentence beside it
therefore cannot disagree about a number.

The page is set in whatever sans-serif your system uses, for the same reason: a webfont is a
third-party request.

## No colour is fixed

Every colour is a named token looked up from whichever of the six themes you are reading in —
`system-light`, `system-dark`, `soft-light`, `tinted-dark`, `high-contrast`, `neon`. A figure that is
legible in one has to be legible in all six, which is a real constraint rather than a claim:
`tests/test_attention_themes.py` renders the page under each theme and asserts that text printed on
a painted mark clears 4.5:1 against it, and that no mark is the same colour as its own background.

Two traps that cost real time and are worth knowing before editing any of it:

- **`web/_shared/tokens.css` is not the token file.** It is a vendored copy of exercise 03's
  *component* stylesheet, under a name that says otherwise. The six-theme token file is
  `deploy/vercel/_shared/tokens.css`, served at `/_shared/tokens.css`, and `index.html` links both.
  A scratch page linking only the vendored one renders every glyph mark invisible, because
  `stroke: var(--bg)` against an undefined `--bg` simply does not paint.
- **`--muted` and `--ink` are both `#000000` under `high-contrast`.** Any encoding that leans on
  ink-versus-muted to carry meaning collapses there, silently.

## The dates

Every date is the arXiv **v1** submission date, read from the paper's own abstract page rather than
from memory or from a secondary source. Later versions move by months and sometimes years —
Bahdanau's v1 is Sep 2014 and its v7 is May 2016 — so "the date of the paper" is ambiguous unless
the version is named. Each entry stores the source's own date string beside our parsed date, so a
reader compares two fields instead of trusting one.

## The commands

Everything below runs from the repository root.

```bash
uv sync --all-packages                                   # once per clone
uv run pytest src/exercises/08-modern-attention-variants  # unit + integration
bash deploy/vercel/build.sh                              # assemble public/
uv run playwright install chromium                       # once, for the browser tests
uv run ruff check --fix . && uv run ruff format .
```

Without chromium the browser tests **skip** rather than fail, which keeps a fresh clone working and
means they protect the page silently or not at all. Run them before opening a pull request.

## What this file is not

It is not the argument, the findings, or the reasoning behind a decision. The argument is the
published page. The decisions — why the field guide is exempt from the twelve-part spine, why the
glyphs carry no sizes, why a chapter's membership is data rather than prose — are in
[`../DECISIONS.md`](../DECISIONS.md). The reader-facing width and readability audit is in
[`MEASURES.md`](MEASURES.md).
