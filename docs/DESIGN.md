# Web design system

Every deployable exercise ships a static `web/` bundle (plain HTML/CSS/JS, zero runtime
dependencies). They all share **one** design language so the site reads as a single product.
This is the canonical reference; `AGENTS.md` carries the short version.

## Principles

- **Apple-flavoured minimal.** Cool neutral surfaces, one bright accent, generous whitespace,
  system sans, soft shadows. No decorative colour, no flashy motion.
- **One consistent shell across pages.** Same header structure, back navigation, type scale,
  accent, panel/card treatment, and footer voice on every page (landing, each exercise).
- **Written for a general audience** — blog-style and self-contained (see [Copy & tone](#copy--tone)).
- **One content width; prose is limited by line length, not by the container.** The wrapper is a
  single width and never moves, so every left edge on a page lines up. What varies is how far an
  element *fills* it: prose stops at its own `ch` measure and leaves the right ragged, while
  tables, figures and registers use the whole width. An earlier version capped the wrapper narrow
  and let wide content break out past it — both widths were defensible and the alternation was
  not, because every breakout moved the left edge and the eye lost its anchor twice per chapter.
- **Tokens live in one file: `deploy/vercel/_shared/tokens.css`**, copied to `/_shared/` by the
  build and linked absolutely by every page. A page defines no colour of its own beyond what is
  genuinely local to it (a diagram's own hues); everything shared is declared once.
- **Six themes.** The system light/dark pair is the default, joined by four the reader chooses —
  soft light, tinted dark, high contrast, neon — via `:root[data-theme="…"]`, with the choice kept
  in `localStorage` under `era5-theme` and applied by a few inline lines in each `<head>` before
  first paint. Three rules make this safe rather than decorative:
  1. The `prefers-color-scheme` block is scoped to `:root:not([data-theme])`, so a chosen theme
     always wins regardless of source order.
  2. **Every theme defines the whole token set.** A theme inheriting half its colours is how a
     token ends up unreadable in exactly one combination nobody tested.
  3. A page with its own local tokens repeats them for the dark themes, or a reader on Neon with a
     light OS gets light diagram colours on a dark page.
  Every text token in every theme is verified at 4.5:1 against both surfaces **before** it ships,
  generated from a checked palette rather than chosen by eye — four contrast failures reached
  production here when they were eyeballed, `--faint` at 3.33:1 and `--accent` at 4.31:1 among them.

## Palette tokens

Declare as CSS custom properties in `:root`, overridden under
`@media (prefers-color-scheme: dark)`.

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| `--bg` | `#f5f5f7` | `#000000` | Page background |
| `--panel` | `#ffffff` | `#1c1c1e` | Cards / panels |
| `--track` | `#e8e8ed` | `#2a2a2c` | Segmented-control track |
| `--ink` | `#1d1d1f` | `#f5f5f7` | Primary text |
| `--muted` | `#6e6e73` | `#a1a1a6` | Secondary text |
| `--faint` | `#6f6f74` | `#8a8a90` | Labels, captions |
| `--line` | `#d2d2d7` | `#2f2f31` | Borders, dividers |
| `--accent` | `#0068d1` | `#2997ff` | The single bright accent |
| `--accent-soft` | `rgba(0,113,227,0.1)` | `rgba(41,151,255,0.14)` | Focus glow |

> **These values are copied from `deploy/vercel/_shared/tokens.css`, which is the source of
> truth, and the copy has gone stale before.** `--faint` and `--accent` sat here at `#86868b`
> and `#0071e3` long after that file had corrected them — those two were the contrast failures
> (3.33:1 and 4.31:1) whose fix is described in its own header comment. If they disagree,
> `tokens.css` wins, and the table above is the thing that is wrong.
>
> The file also ships **32 light tokens**; only the core set is tabulated here. Read it directly
> before inventing a colour — the rule is that no page introduces a value that is not already a
> token there.

Shadow: `--shadow: 0 1px 3px rgba(0,0,0,0.04), 0 6px 20px rgba(0,0,0,0.03)` in light; `none`
in dark (borders carry the elevation there).

**Data-viz hues** — for plot series / categorical marks only, never for UI chrome. Apple system
colours, distinct in both themes (light / dark):

| Role | Light | Dark |
| --- | --- | --- |
| blue | `#0071e3` | `#2997ff` |
| orange | `#d1730a` | `#ff9f0a` |
| green | `#248a3d` | `#30d158` |
| purple | `#a03fce` | `#bf5af2` |
| indigo | `#5856d6` | `#5e5ce6` |

## Typography

```css
--sans: -apple-system, system-ui, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
--mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
```

- **Sans for everything display and prose** (SF Pro Display for large headings). No serif.
- **Mono only for data/technical labels** — token lists, numbers, matrix cells, tab labels.
- Headings: weight `600`, tight tracking (`letter-spacing: -0.025em` on the H1).
- Eyebrow: 12px, weight 600, uppercase, `letter-spacing: 0.1em`, `--accent`, in the pattern
  `NN · Topic` (e.g. `02 · Tokenization`).

## Components

- **Back pill** (top-left of every exercise page): pill (`border-radius: 980px`) on `--panel`
  with `--shadow`, `--accent` text, a leading `←`; hover fills `--accent` with white text.
  Links to the site root `/` (the landing page). Label it **Back**.
- **Panels / cards:** `--panel` background, `1px solid var(--line)`, `border-radius: 18px`,
  ~22–24px padding, `box-shadow: var(--shadow)`. Cards that are links lift on hover
  (`translateY(-3px)` + a deeper shadow).
- **The landing page is two measures, not one.** Prose keeps a readable line length (`.head` at
  54ch, the lede at 52ch) while the exercise cards go wide as a grid —
  `repeat(auto-fill, minmax(min(340px, 100%), 1fr))`, three columns at 1440px, one on a phone. It
  was a single 640px column at every width for a long time, which used a third of a 1920px screen.
  Widening the column instead would have made the sentences unreadable, so **both halves are
  tested**: that the grid uses the screen, and that the prose does not widen with it.
  `min(340px, 100%)` and never a bare `340px` — an auto-fill track cannot shrink below its own
  minimum and pushes a 320px phone sideways.
- **Cards in a row share a height.** `align-items: stretch` on the grid, and the meta line pinned
  down with `margin-top: auto`, so a row does not end at three different depths. Watch the
  selectors: the index label is *also* a direct-child `span`, so `a.item > span { flex: 1 }` gives
  it flex-grow and it stretches to fill the card — 93px tall for one line, pushing the title into
  the middle. Scope it `:not(.idx)`.
- **Segmented control:** `--track` background; the selected button is a `--panel` "knob" with
  `--shadow`.
- **Inputs:** on focus, `--accent` border plus a 3px `--accent-soft` glow ring.
- **Header:** left-aligned eyebrow → H1 → lede. Not centered.

## Interaction

- **Animate state changes** in canvases/visualisations — morph between states with a short eased
  transition (≈550ms, easeInOutCubic) instead of an instant redraw. Keep the framing (scale +
  centering) stable across the states being toggled so panels don't resize mid-transition; only
  the data should move. Cancel any in-flight animation when a new interaction starts.

## Copy & tone

Public pages are **standalone, blog-style demos of an idea** — written so a first-time visitor can
enjoy them without any course context.

- Favor plain, explanatory copy aimed at a general reader over internal labels like "Session N"
  or "assignment".
- Keep the numbered topic eyebrow (`NN · Topic`) — it reads as a friendly section label.
- Footers are short, descriptive captions (e.g. "No dependencies. Each layer is an
  area-preserving 2×2 map.").

> These are style notes for the **web pages**. The source course, instructor, and platform are
> credited warmly in a single **Credits** section at the bottom of the root `README.md` — that's the
> home for attribution, which keeps the demo pages themselves focused on the ideas.

## Editing caution (non-ASCII)

These pages intentionally contain `—`, `→`, `←`, `·`, `×`, subscripts, and math glyphs. Edit
them with the **Edit/Write tools**. Do **not** rewrite them with byte-mode stream editors
(`perl -0pi`, `sed`) using wide-character escapes — that reads UTF-8 as Latin-1 and re-encodes
it, double-encoding every multibyte character into mojibake (`—` → `â€"`, `·` → `Â·`).

## Hosting

`deploy/vercel/build.sh` assembles the landing page at `/` plus every `src/exercises/*/web`
under its slug (`/NN-slug/`). Any new exercise with a `web/` dir is picked up automatically; only
the landing page's cards are hand-maintained (`deploy/vercel/index.html`). The Back pill's `/`
target resolves to that landing page in the deployed site.
