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
- **Theme-aware.** Style light and dark via `prefers-color-scheme`; every token has both values.

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
| `--faint` | `#86868b` | `#6e6e73` | Labels, captions |
| `--line` | `#d2d2d7` | `#2f2f31` | Borders, dividers |
| `--accent` | `#0071e3` | `#2997ff` | The single bright accent |
| `--accent-soft` | `rgba(0,113,227,0.1)` | `rgba(41,151,255,0.14)` | Focus glow |

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
