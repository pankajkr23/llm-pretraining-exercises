# Web design system

Every deployable exercise ships a static `web/` bundle — plain HTML, CSS and JS, zero runtime
dependencies, no third-party request of any kind. They share **one** design language so the site
reads as a single product.

This is the canonical reference. `AGENTS.md` carries the short version and owns the rules that are
*enforced*; this document owns the rules that are *drawn*. Where they disagree, `AGENTS.md` wins and
this file is the thing to fix.

**Exercise 08 is the reference implementation.** Every component below is live there, and where a
number appears it was measured on that page rather than intended. Exercises 01–07 predate most of
it; [Retro-fitting an older exercise](#retro-fitting-an-older-exercise) is the checklist for
bringing one up to standard.

---

## Principles

- **Apple-flavoured minimal.** Cool neutral surfaces, one bright accent, generous whitespace, system
  sans, soft shadows. No decorative colour, no flashy motion.
- **One shell on every page.** Same header, back pill, theme picker, type scale, accent, panel
  treatment and footer voice — landing page and every exercise.
- **Written for a general audience.** Blog-style and self-contained; a first-time visitor should
  enjoy it with no course context. See [Copy and tone](#copy-and-tone).
- **One narrative spine, twelve parts, declared as `data-role`:**
  `thesis · glossary · problem · mechanism · method · expected · results · negatives · conclusion ·
  limits · next · reproduce`. **`AGENTS.md` owns that rule and `tests/test_page_spine.py` enforces
  it repo-wide**; this document owns only how the sections look. Write each role as a literal string
  where the section is built — the guard reads the source, so a role assembled from a variable is
  invisible to it and passes on a page with no spine.
- **Give each kind of content the width it deserves.** Prose keeps a reading measure; tabular data
  and figures take the full width. This is the single most valuable rule here and the one most
  often broken — see [Layout](#layout) and [Density](#density).
- **An interaction is never the only route to a lesson.** A control may make a point *vividly*; the
  point itself lives in prose that is always visible. Print readers, reduced-motion readers and
  readers who never touch a control get the whole argument.

---

## Palette tokens

**`deploy/vercel/_shared/tokens.css` is the source of truth.** It is served at `/_shared/tokens.css`
and every `index.html` links it. It defines **six complete themes**: `:root` (light), a
`prefers-color-scheme: dark` block, and four explicit `[data-theme]` choices — `soft-light`,
`tinted-dark`, `high-contrast`, `neon`.

> **Two traps, both of which have cost real time.**
>
> **`web/_shared/tokens.css` is NOT the token file.** Every exercise vendors a byte-identical copy
> of exercise 03's *component* stylesheet under that name — its own first line says so. A page must
> link **both** `/_shared/tokens.css` (absolute, the real tokens) and `./_shared/tokens.css`. A
> scratch harness that links only the vendored one renders every glyph invisible, because
> `stroke: var(--bg)` against an undefined `--bg` simply does not paint.
>
> **Under `high-contrast`, `--muted` and `--ink` are both `#000000`.** Any encoding that leans on
> ink-versus-muted to carry meaning reads in five themes and vanishes in the sixth.

| Token | Role |
| --- | --- |
| `--bg` | page ground |
| `--panel` | raised surface — cards, pills, the `.defs` box |
| `--track` | recessed fill — segmented controls, bar tracks |
| `--line` / `--line-strong` | hairline rule / heavier rule |
| `--ink` / `--muted` / `--faint` | primary / secondary / label text |
| `--accent` / `--accent-soft` / `--on-accent` | the one bright accent, its wash, text on it |
| `--shadow` | panel elevation (light only; borders carry it in dark) |
| `--hi` / `--lo`, `--grade-a…x` | semantic good/bad and grading |
| `--part-q/k/v/store` | domain roles reserved so they never collide with `--accent` |
| `--sans` / `--mono` | the two faces; no serif anywhere |

**No page introduces a colour that is not already a token there.** If a value is missing, add it to
the token file across all six themes — never a per-exercise literal.

**Colour can only carry semantics while there are more colours than meanings.** When the count of
things to distinguish can exceed the palette, encode it in **form** — an ordinal, a shape, a texture
— and let colour keep its one job. A semantic palette of four asked to distinguish six steps
silently rendered two of them identically.

**Data-viz hues** — for plot series and categorical marks only, **never for UI chrome**. Apple
system colours, distinct in both themes:

| Role | Light | Dark |
| --- | --- | --- |
| blue | `#0071e3` | `#2997ff` |
| orange | `#d1730a` | `#ff9f0a` |
| green | `#248a3d` | `#30d158` |
| purple | `#a03fce` | `#bf5af2` |
| indigo | `#5856d6` | `#5e5ce6` |

**The blue here is deliberately not `--accent`** (`#0068d1`). `--accent` is text and UI chrome, so
it is held to 4.5:1 — it shipped at `#0071e3` and measured **4.31:1**, one of four contrast failures
that got onto this site by being chosen by eye. A categorical mark sits *on* a panel rather than
behind prose, so it keeps the brighter value. Two roles, two values; conflating them is what
darkened the accent in the first place.

---

## Layout

A named-line grid on `#main`, repeated on every container that must let a child escape it. Named
lines rather than negative margins, so `.bleed` and `.wide` are declarations rather than tricks.

```css
#main {
  display: grid;
  grid-template-columns:
    [full-start] minmax(14px, 1fr)
    [wide-start] minmax(0, 150px)
    [text-start] min(70ch, 100% - 28px)
    [text-end]   minmax(0, 150px)
    [wide-end]   minmax(14px, 1fr)
    [full-end];
}
#main > *,
#main > section > * { grid-column: text; min-width: 0; }
#main > section > .bleed { grid-column: full; padding-inline: clamp(14px, 3vw, 42px); }
#main > section > .wide  { grid-column: wide; }
```

- **`text`** — prose. Everything defaults here.
- **`wide`** — a table or a figure that needs more than a reading measure but is not a plate.
- **`full`** — plates, the index, the masthead. **A full-bleed element still needs its own inset**,
  or it prints flush to the window on both sides; the padding belongs on the element, not the track.

**No `100vw` anywhere.** It includes the scrollbar, so a full-bleed built on it scrolls sideways on
every desktop browser that reserves one.

**The same track list must be repeated on every nesting level a child needs to escape** — `#main`,
`#main > section`, `.plate`, `.plate > .plate-body`, and any chapter wrapper. Exercise 08 repeats it
five times; change one and you must change all five.

**A vendored stylesheet can reserve space for an element the page has to build itself.**
`web/_shared/page.css` sets `.wrap { padding-left: 260px }` at ≥1180px whether or not the page
builds a rail, and centres the rail's contents with `.rail-inner { margin-block: auto }` — a wrapper
the page must create. When you vendor `web/_shared/`, diff what its rules select against what your
page emits.

---

## Typography

```css
--sans: -apple-system, system-ui, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
--mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
```

Sans for prose and display; **mono only for data and technical labels** — token lists, numbers,
matrix cells, tab labels, eyebrows. No serif anywhere.

### The scale is fluid, and the lever is size — not measure

```css
#main { font-size: clamp(19px, 1.2vw + 1.7px, 22px); }
#main .say { font-size: inherit; max-width: 70ch; }
```

Measured on exercise 08, holding the repo's own 42–80 character guard:

| viewport | body | prose width | share of screen | characters a line |
| --- | --- | --- | --- | --- |
| 1180 | 19px | 835px | 71% | 77 |
| 1440 | 19px | 835px | 58% | 77 |
| 1920 | 22px | 951px | 50% | 77 |
| 2560 | 22px | 951px | 37% | 77 |

**Not one longer line — 39% more screen.** A reader complaining that a page "narrows too much" is
almost never asking for longer lines; they are asking for the page to use the display.

Secondary type scales **with** the body in `em`, or a 22px paragraph sits beside a 13.5px caption and
the hierarchy reads as a rendering fault:

| element | size | measure |
| --- | --- | --- |
| body prose `.say` | `1em` | `70ch` |
| standfirst | `clamp(18px, 2.2vw, 24px)` | `52ch` |
| section `h2` | `clamp(30px, 5vw, 48px)` | — |
| notice `h2` | `21px` | — |
| figure caption | `0.63em` | `78ch` |
| figure orientation body | `0.70em` | `68ch` |
| small print / colophon | `0.56em` | on the `p`, not the block |
| mono labels, eyebrows | `9.5–11px` fixed | — |

Mono micro-labels are deliberately **not** scaled: they are furniture, not reading, and they stop
being legible if they shrink.

### The `ch` trap — the most expensive recurring bug in this repo

**`ch` resolves against the element's OWN computed font-size**, not its parent's.

- `.say { font-size: 16px; max-width: 68ch }` pins the paragraph at 685px whatever `#main` does.
  Growing the grid track alone changes nothing — the paragraph is the binding constraint.
- `.limitlist { max-width: 80ch }` on a `<ul>` that inherits a 22px root, with 13.5px `<li>` text
  inside it, produced **111 characters a line**.
- A pull quote wrapper at `24ch` and `16px`, holding 38px type, came out one word per line.

**Put the measure on the element that carries the type.** Every time.

---

## Density

**The narrowness *is* the length.**

A page is long because its content is narrow, far more often than because it has too many words.

Exercise 08's index was **30 rows at 306px each**. Widening the container from 720px to 1,676px —
more than double — moved a row to 292px, a 5% gain. The row was not too wordy; it was **six stacked
bands on a four-column grid**, so extra width only shortened lines that were already short.

**Two bands, with the prose in columns, is 238px.** The rule:

- an **identity line** — glyph, date, name, family, one short field
- a **body row** of prose columns, sized by the character floor rather than by taste:
  `repeat(auto-fit, minmax(min(315px, 100%), 1fr))` — at 13px, 315px is a 42-character line, so the
  column count follows the width the page actually has.

Fixed `fr` weights were tried first and put one column at 35 characters. **Let the floor decide the
count.**

### The measures discipline

`tests/test_attention_measures.py` is the pattern: measure **characters per line**, never pixels,
at ten widths, and fail outside **42–80**.

- **Narrow is only a defect when there is room to be wider.** On a 390px phone a standfirst reads at
  29 characters while filling 92% of everything available — that is the device, not a decision.
- **The same pixels measure ~4% fewer characters on CI's Linux fonts than on macOS.** Do not tune a
  design until it scrapes past on both; leave room. A two-column grid at 42 characters locally was
  39 on CI.
- The guard only sees what its selector matches. A `div` carrying prose, or a string under its
  sampling threshold, is invisible — which is how a ledger cell at 36 characters and a definition
  block at 23 both shipped.

---

## Components

Every component below is live in exercise 08. Copy the markup shape and the class names.

### Shell

- **Back pill** — top-left of every page: `border-radius: 980px` on `--panel` with `--shadow`,
  `--accent` text, a leading `←`. Hover fills `--accent` with `--on-accent`.
- **Theme picker** — `.themepick`, a `<select>` of the six themes. The stored choice is applied by
  an **inline `<script>` in `<head>`**, before first paint; resolving it in a module renders one
  frame of the wrong theme and repaints. Any other pre-paint setting (a type scale, a layout flag)
  belongs in that same script, and it must be the **only** place the value is decided — a second
  derivation in a module is a second chance to disagree, invisibly.
- **The shellbar wraps** (`flex-wrap: wrap`). It is a `space-between` flex row in the vendored sheet
  with no wrapping, and a third control pushes a 320px screen sideways.

### The contents rail

`#rail` is fixed at 236px from ≥1180px; `.wrap` reserves 260px for it. The page must build
`.rail-inner`, or the contents hang at the top of a full-height column while every sibling page sits
centred.

- **Mark the section in view.** The vendored sheet already styles `.rail-link.on` — accent bar, soft
  accent wash, bold label. The rule is **"the last heading whose top has passed the first third of
  the viewport"**, not "the nearest heading": sections run several screens, so from the middle of one
  the *next* heading is often closer than the one behind you, and the rail runs a section ahead of
  the reader. A proportion of the viewport rather than a pixel count, so it means the same on a
  laptop and a tall monitor. rAF-throttled on `scroll` and `resize`.
- **A read-time** under the contents, derived from the rendered word count — never typed. Fill it
  **one frame after the build**: `innerText` is a laid-out measurement, so counting during the build
  comes out short.

### Sections and headings

- `section[data-role]` with an eyebrow (`.role`, mono, uppercase), an `h2`, and `.say` paragraphs.
- A hairline above every section and generous vertical air. **The rule is the structure — there are
  no boxes on these pages.**
- **A heading names its subject and never states a count.** A count in a heading is always a count of
  that section's own contents, so it goes stale the moment the contents change — a section headed
  "Three things this opens" above four items shipped green because the lexical count guard starts at
  *eleven*. Derive it or drop it.

### Notices — apparatus that stops wearing display type

A section that a reader *consults* rather than *follows* — limits, colophon, provenance — gets
`.notice`: a **21px** heading instead of the masthead's `clamp(30px, 5vw, 48px)`, and a tight
`.limitlist` (`<ul>` with `<b>Lead.</b> body` items) instead of full paragraphs.

**It stays fully visible.** Never a `<details>`: a limitation a reader has to open a drawer to find
is a limitation the page is hiding. Smaller type, fewer words, always on screen.

### Plates — numbered figures

`.plate` is a full-bleed `<figure>` with a head (`Figure N` + a short name) and a caption.

- **The plate carries the page grid inward**, so its prose sits in the same text column as every
  other paragraph. Without it one figure had four different left edges and a reader called it
  "random".
- **A caption argues; it does not label.** State what to conclude, and where useful what would
  falsify it. A caption that repeats the figure's title has made the reader do the interpreting.
- **A caption may not claim what the drawing cannot show.** A figure with one axis cannot invite a
  judgement about a second.
- **`.preamble`** is the orientation *before* the figure — a mono label (`.preamble-lab`) over a
  short paragraph, one `.preamble-row` each, saying what you are looking at. **Once is orientation;
  three times is a template readers skip**, so keep it to one or two rows and put the argument in
  the caption.
- **Figures are inline SVG built from the page's own data, never a chart library.** Exercise 07
  draws six from `data.js` with `createElementNS`: a scatter on a grid, a flow diagram, two bar
  charts and a paired-lines plot. No dependency, no CDN, and they inherit the theme for free because
  **every colour is a token**. Never a literal — a literal is right in one of the six themes and
  wrong in the other five, and the theme picker shows it instantly.
- **Every figure sits in `<figure>` with a `<figcaption>`.** Number them (`Figure 3.`). `figure`
  gets `overflow-x: auto` and the svg a `min-width`, so a wide diagram scrolls inside its own box
  and the page body never does.
- **Draw the whole object, not the part that fits.** 07's grid figure originally stopped at column
  32, so the discarded bytes landed outside the viewBox and stacked into one dot — a figure whose
  caption said nineteen bytes were thrown away while showing one. Extend the domain and shade the
  region being lost.
- **Check a mechanism figure's mapping against the data, not against how it looks.** The same page
  shipped a draft with two of four labelled points on the wrong rows: a correct-looking rectangle,
  wrongly labelled, which is the one thing a mechanism figure must never be.

### Pipeline figures

- **A pipeline figure is boxes and arrows that wrap** (`.flow`): `display: flex` with
  `flex-wrap: wrap` and `overflow-x: auto`, arrows as separate `.flow-arrow` siblings between the
  boxes.
- **Mark the stages that matter with an explicit class, never `:nth-child`** — the arrows are
  siblings of the boxes, so any positional rule counts them too and selects the wrong stages the
  moment one is added.
- **Let it wrap rather than scroll.** 06's first version put the two stages its caption called out
  off-screen behind a horizontal scroll, so the figure's punchline was invisible.

### Glossary

- **A glossary is a table of terms, not a paragraph of them** (`dl.defs`): a two-column grid,
  `minmax(110px, 165px)` for the term and `1fr` for the definition, collapsing to **one column below
  640px** — a 110px term column beside a definition is unreadable on a phone. The term is `--mono`,
  the definition `--muted`. A grid rather than a float, so a long definition wraps under itself
  instead of under the term.
- **A glossary must not be hover-only.** Exercise 06 defined ten terms as tooltips and nothing else,
  which is a definition that does not exist on a touch screen, does not survive printing, and cannot
  be reached with a keyboard. Render the *same object* as a visible section and keep the tooltips —
  two presentations, one source, so they cannot disagree. If the heading states a count, derive it
  from the list it heads and test it.

### Interactive figures

- **`.tabs`** — the standard stepper: `<button aria-pressed>` per step, arrow-key support on the
  holder, and a short line that changes with the step.
- **Every step must draw something different.** A figure whose tabs all render the same picture is a
  control that teaches nothing — assert it.
- **Teach a taxonomy by drawing one object N ways.** Exercise 08's four families are four answers to
  one question, so the same score grid is drawn once and each tab changes only what happens to it.
  Four abstract marks with terse labels taught nothing; the same object changing is the lesson.
- **All the explanations stay visible.** The tabs change the drawing; they never gate the prose.
- Canvas/SVG state changes **morph** with a short eased transition (~550ms), keeping the framing
  stable so panels do not resize mid-toggle. **Prefer a painted terminal state to an animated one
  wherever the motion buys nothing** — reveal-on-scroll is a decision to hide something by default.
- **An `IntersectionObserver` on a detached node never fires.** Every figure builder returns its
  element before the page appends it, so register the observer from the code that appends, or defer
  a frame and check `isConnected`.

### Command blocks

`pre.code`: `--panel` on `--line`, `border-radius: 10px`, `--mono` at 12.5px/1.65, `overflow-x:
auto`. A `reproduce` section is mostly these, so they have to read as **runnable** rather than as
decoration.

### The landing page

- **The landing page is two measures, not one.** Prose keeps a readable line length (`.head` at
  54ch, the lede at 52ch) while the exercise cards go wide as a grid —
  `repeat(auto-fill, minmax(min(340px, 100%), 1fr))`, three columns at 1440px, one on a phone. It
  was a single 640px column at every width for a long time, using a third of a 1920px screen.
  Widening the *column* instead would have made the sentences unreadable, so **both halves are
  tested**: that the grid uses the screen, and that the prose does not widen with it.
- **`min(340px, 100%)`, never a bare `340px`** — an auto-fill track cannot shrink below its own
  minimum and pushes a 320px phone sideways.
- **Cards in a row share a height.** `align-items: stretch` on the grid, and the meta line pinned
  with `margin-top: auto`, so a row does not end at three different depths. Watch the selectors: the
  index label is *also* a direct-child `span`, so `a.item > span { flex: 1 }` gives it flex-grow and
  it stretches to fill the card — 93px tall for one line, pushing the title into the middle. Scope
  it `:not(.idx)`.

### Reference tables

- **Two bands, not six** — see [Density](#density).
- Every row anchored `id="m-<key>"` so it can be linked, and `:target` gets `--accent-soft`.
- **The catalogue is tabulated exactly once.** A second table of the same rows is duplication however
  differently it is styled — assert it.

### Navigation into a long page

- **Reader doors** (`.guide-paths`) — three or four labelled routes in the opening, each a link with
  a `who` and a `what`. Four items want an explicit column count, not `auto-fit`: four floors fit
  almost everywhere and the doors read at 27 characters.
- **Chapter strips** (`.ch-strip`) — a chapter names its own entries in one line, **each with its
  year**, in date order, linked to the full entry. A bare list of names is no evidence for a claim
  about order, and a chapter's claim is usually about order.
- **Finding tiles** (`.find-grid`) — the page's conclusions in the opening, so a reader who stops
  early stops *correctly* rather than partially. **Put a failure among them**: a page that shows only
  its wins has not earned the ones it shows.
- **An exit line** after the argument, telling a reader they may stop and what remains.

---

## The shared bundle — `web/_shared/`

Every deployable exercise vendors a **byte-identical** copy of six files, with exercise 03 as the de
facto origin. They are copies, not a package: editing one changes one page, and a fix has to be
re-vendored to all of them.

| file | owns | actually used by |
| --- | --- | --- |
| `page.css` | the shell — wrap, shellbar, back pill, theme picker, rail, `.jump`, `.disclaim` | all |
| `tokens.css` | **misnamed** — exercise 03's *component* sheet, not the token file | all |
| `explainer.css` | the scrollytelling vocabulary — `.scrolly`, `.step`, `.sticky`, `.fig-*`, `.qbox` | 03, 06 |
| `explainer.js` | `makeExplainer` — the scrollytelling builder | 03, 06 |
| `num.js` | number formatting | 03, 06 — **removed from 04, 05, 07, 08**, which never imported it |

**Vendoring copies the styles and not the markup they assume.** Three defects have come from that:
a 260px rail gutter reserved on pages that build no rail; `.rail-inner` centring a wrapper the page
never created; and marks whose colours resolve only when the real token file is also linked. When
you vendor this directory, **diff what its rules select against what your page emits**, and write
down what you chose not to build.

**Dead weight was real and 2,578 lines of it are gone**, and the counts that described it were
wrong in the direction that matters. `anim.js` — 167 lines vendored six times, exporting seven
helpers, imported by nothing and assembled into `public/` on every deploy — is removed, along with
`explainer.js` and `num.js` from the four exercises that link neither.

**`explainer.css` was NOT "used by two".** It is used by 01, 02, 03 and 05; exercise 03 alone emits
36 of its 56 classes, and **12** are orphaned. `page.css` has **10** orphans, not fifteen. The first
measurement said `explainer.css` was entirely unused, because it looked for the `el(tag, className)`
helper this document describes while exercise 03 almost exclusively calls a local `$(tag,
className)`. Deleting on that evidence would have taken out a live stylesheet.

So these counts are no longer published here. `tests/test_shared_layer.py` derives them, refuses any
vendored file the vendoring exercise does not reference, and carries the full list of class-setting
idioms — because a count in prose beside a fact a test can compute is the failure this repository has
paid for most often. The remaining orphan *classes* are a separate, riskier change: a class emitted
by a path the extractor cannot see is indistinguishable from a dead one, which is exactly the
mistake above.

---

## Motion, reduced motion, and print

- **Scroll-triggered reveals and state morphs**: ~550ms, `easeInOutCubic`. Cancel any in-flight
  animation before starting the next, and keep the framing stable so panels do not resize mid-toggle.
- **Control responses**: ≤200ms. A control that takes half a second feels broken.
- **`prefers-reduced-motion: reduce` is not optional.** Jump straight to the end state — no sweep, no
  auto-play, no reveal. The state must be *painted*, not merely reachable.
- **Prefer a painted terminal state to an animated one wherever the motion buys nothing.**
  Reveal-on-scroll is a decision to hide something by default; never make it for the one element
  that carries the point. A cut line that faded in was invisible in every screenshot, print and
  in-page anchor for as long as it existed.
- **Print**: the page must be readable on paper. Pinned columns unpin, steps stop reserving a
  viewport of height, and anything that only exists to be interacted with is hidden.
- **An `IntersectionObserver` on a detached node never fires and reports nothing.** Builders return
  their element before the page appends it — register from the appending code, or defer a frame and
  check `isConnected`.

---

## Numbers and provenance

The most repeated rule across every page here, and the one that decides whether a reader can check
anything.

- **Every number a page displays is generated from tracked data**, never typed into prose. A
  generated table under a hand-written sentence looks maintained, and only the sentence is wrong —
  this is the most expensive recurring defect in the repo.
- **A displayed number carries where it came from.** Measured by this exercise, quoted from a named
  source, or constructed by us — and a constructed figure says so rather than borrowing the
  authority of a measured one.
- **A quote is gated mechanically**, as a contiguous run of the source document's own characters,
  before it is allowed on the page. Verbatim is not the same as correct: check separately that the
  quote is about the quantity being claimed.
- **Where nothing states a value, the field stays empty.** An empty column is often the most
  informative one on the page — it separates what the field adopted from what it admired.
- **A quantity pinned to a constant by construction is not a measurement.** Before publishing a
  derived number, ask what input would change it; if none can, give it a denominator that varies or
  delete the field.

---

## Interaction and accessibility

- Keyboard: every control reachable and operable; `:focus-visible` always has a visible ring.
- `aria-pressed` on toggles and tabs; `role="group"` with an `aria-label` on figure holders;
  `role="img"` plus `<title>` on every standalone SVG.
- Respect `prefers-reduced-motion`: no sweep, no auto-play, terminal states painted immediately.
- Wide content scrolls inside **its own** `overflow-x: auto` container. The page body never scrolls
  sideways — assert it at 1440, 900, 620, 390 and 320.
- **Visible is not legible.** `white-space: nowrap` inside `overflow: hidden` truncates with no
  ellipsis and no warning. Assert that no element's `scrollWidth` exceeds its `clientWidth`.

---

## Copy and tone

- Plain, explanatory, and self-contained. The numbered eyebrow (`NN · Topic`) is the only course
  reference on a page.
- **Define every term where the reader first meets it**, and give each definition a real number from
  the exercise's own run rather than a textbook gloss. A definition is useful where the reader meets
  the thing, not where the page finds it convenient — an alphabet of marks four thousand words before
  the first mark is used is a wall, not a key.
- **Say what you expected before what you found.** It is the only way a reader can tell a finding
  from a story told backwards.
- **Every number in prose is generated**, never typed. A generated table under a hand-written
  sentence looks maintained and only the sentence is wrong.
- **State scale and limits in the open text**, never inside a collapsed disclosure.
- **Naming a real product, model or vendor is a claim** and gets sourced like any other — quote the
  sentence, gate the quote against the downloaded document, and leave the field empty where nothing
  says so. An empty column is often the most informative one on the page.

### Editing caution (non-ASCII)

`—`, `→`, `·`, `~`, math glyphs: use the Edit/Write tools or a UTF-8 heredoc. **Never** `perl -0pi`
or `sed` with wide-char escapes — byte-mode rewrites double-encode UTF-8 into mojibake.

And beware the escaping order in a markup helper: a `rich()` that escapes `&` before applying
markup will render `&nbsp;` as its six literal characters. Use a real non-breaking space.

---

## What enforces what

A rule with no guard decays. The repo-wide guards live in `tests/`; per-exercise ones in
`src/exercises/*/tests/`.

| rule | guard |
| --- | --- |
| twelve-part spine present, in order | `tests/test_page_spine.py` + a per-exercise DOM-order test |
| every deployable exercise is enforced or exempt, with a reason | `tests/test_page_spine.py` |
| README reading path, command, limits section | `tests/test_readme_structure.py` |
| every relative link resolves from its own directory | `tests/test_readme_links.py` |
| exercise skeleton present; no `REQUIREMENTS.md` ever tracked | `tests/test_exercise_skeleton.py` |
| the rail is built and fills the gutter it reserves | `tests/test_rail_centring.py` |
| every test file is in a CI shard **and collects there** | `tests/test_ci_shards_cover_everything.py` |
| 42–80 characters a line, ten widths | `test_attention_measures.py` (the pattern to copy) |
| no count typed into page prose, or into a heading or rail label | `test_attention_docs.py` |
| no text clipped by its own box; no sideways scroll | `test_attention_render.py` |
| six themes render, contrast holds, no mark equals its ground | `test_attention_themes.py` |
| **this document's own last two released versions are kept and unedited** | `tests/test_standards_history.py` |

**Nothing here enforces that a rule stays written down, and that is the gap this document fell into.**
Rewriting it as the standard dropped nine rules with no replacement — never a chart library, a
glossary must not be hover-only, never `:nth-child` for pipeline stages, and six more, each a lesson
from a defect that had already cost a page. Every guard above stayed green, because a guard can only
check the page against a rule that exists. So `docs/standards-history/` keeps the last two released
versions of this file beside it: **diff before you rewrite, and list what the rewrite drops.**

```bash
diff docs/standards-history/DESIGN.v0.12.0.md docs/DESIGN.md
uv run python tools/snapshot_standards.py --ref v0.12.0    # if you do not have it yet
```

The archive is **local-only** — tracking it would put a second copy of this document on the remote —
so it is absent on a fresh clone and the guard above skips there. Build it once and it stays.

**Every invariant is written twice** — once against the real page, once against a deliberately broken
fixture. When you add a guard, break the thing on purpose and watch it go red before you commit.

**A guard must test the property, not one phrasing of it.** Two guards in one topic demanded a
specific string and failed correct work. Ask the underlying question instead.

---

## Retro-fitting an older exercise

Exercises 01–07 predate most of this. In order, cheapest and highest-value first:

1. **Link both stylesheets** — `/_shared/tokens.css` and `./_shared/tokens.css`. Check for any
   hardcoded colour and replace it with a token.
2. **Build `.rail-inner`** if the page has a rail, and add the scroll-spy — the CSS for `.rail-link.on`
   is already vendored and unused in 05, 06 and 07.
3. **Adopt the fluid type scale** and move every `ch` measure onto the element that carries the type.
4. **Adopt the named-line grid**, then give tables and figures the `wide` or `full` track. Check for
   a `.bleed` class on an element nested too deep for any rule to match it.
5. **Demote apparatus to `.notice`** — limits, colophon, provenance.
6. **Compact any stacked reference table** to two bands.
7. **Copy the measures guard**, run it, and fix what it finds before believing the page.
8. **Screenshot every section at 2560/1920/1440/1180/768 in four themes, and read them.** Every
   defect that mattered on exercise 08 was found by looking, with the suite green.

---

## Hosting

One Vercel project serves every exercise's static `web/` under its slug (`/NN-slug/`), assembled by
`deploy/vercel/build.sh` into `public/`. Previews auto-deploy per PR; **production never
auto-deploys** (`vercel.json` → `git.deploymentEnabled.main: false`) and is promoted through the
gated `deploy-production.yml`.

A preview URL is **login-walled** — it can never satisfy a requirement for a link a logged-out
stranger can open. Only production can.
