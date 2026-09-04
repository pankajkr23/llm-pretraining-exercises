# EXPLAINER PATTERN — the implementation

`EXPLAINER_PROMPT.md` decides *what* an explainer must be. This file records *how* one is built, so
the next one does not re-invent the skeleton.

It began as an extraction from **§1 "Thirteen words in a row is a fingerprint"** and now describes
the shipped implementation across both surfaces. The skeleton lives in
`web/_shared/explainer.js` (`makeExplainer`) with its rules in `web/_shared/explainer.css`; every
explainer on the site is built through it, which is what makes the identical-slot discipline
mechanical rather than aspirational.

Everything below is descriptive of code that ships and passes CI. Where the reference deviates from
what this file prescribes, §9 says so rather than pretending otherwise.

---

## 1. The shape in one paragraph

An explainer is **a list of states, a pure function from state to numbers, and a renderer**. The
reader moves between states (by scrolling, focusing, or typing), the pure function recomputes, and
the renderer redraws one pinned figure. There is no per-state markup and no branching render path:
every state fills the same slots, which is what makes the states comparable — obligation ③.

## 2. DOM skeleton

Built in JS by `makeExplainer` — you supply states and a `render(i, api)`, not markup. The shape it
produces, for reference when debugging:

```html
<section id="sN">
  <h2><span class="n">N</span>A claim, not a topic</h2>
  <p class="claim">Standfirst. Names the <b>variables</b> and says what the reader does.</p>

  <figure>
    <div class="scrolly">                    <!-- grid: steps | pinned figure -->

      <div class="qbox">                     <!-- optional: the reader's own input (§11) -->
        <label for="…">What we are protecting — replace it with one of your own</label>
        <textarea class="qinput" id="…"></textarea>
      </div>

      <div class="steps">                    <!-- the prose column: one .step per state -->
        <div class="step" tabindex="0" data-i="0">
          <div class="marg">The index</div>  <!-- gutter label, uppercase mono -->
          <p>Prose with one <b>bolded variable</b>.</p>
          <div class="shard">…</div>         <!-- the state's raw input, monospace -->
          <div class="inline">→ …</div>      <!-- verdict; hidden on screen, shown in print -->
        </div>
        …
      </div>

      <div class="sticky">                  <!-- pinned: does not scroll away -->
        <div class="fig">
          <div class="fig-num">Fig. N — …</div>
          <div class="fig-big">…</div>          <!-- ONE oversized mono number -->
          <div class="fig-sub">…</div>          <!-- what that number is of -->
          <div class="fig-verdict">…</div>      <!-- 1–2 words, shouted -->
          <div class="strip">…</div>         <!-- per-unit marks; red = excluded -->
          <div class="fig-note">…</div>         <!-- interprets THIS state -->
        </div>
        <div class="fig-rail">…</div>           <!-- the standing limitation -->
        <div class="pill">…</div>            <!-- one number to remember -->
      </div>

    </div>
    <figcaption>Fig. N — method, source, and what the page is not doing.</figcaption>
  </figure>
</section>
```

Order matters once: `.qbox` spans both columns (`grid-column: 1 / -1`), and on narrow screens
`.sticky` takes `order: -1` so the figure sits above the prose rather than below all of it.

Do not hand-build this. Call `makeExplainer({ $, onPlay })` once per page and `buildExplainer(cfg)`
per section; `cfg.anchor` gives a slug id instead of `s<n>`.

## 3. CSS class names

Canonical names by role. Reuse these; do not invent a per-section vocabulary.

| Class | Role |
|---|---|
| `.scrolly` | The two-column grid. `minmax(0, 1fr)` + a fixed figure column |
| `.qbox` / `.qinput` | The reader's own input and its label |
| `.step` | One state. Tall enough to own the viewport centre; `.on` when active |
| `.marg` | Gutter label — uppercase mono, the §15 marginalia |
| `.shard` | The state's raw text, monospace, `white-space: pre-wrap` |
| `.inline` | Per-step verdict. `display: none` on screen; shown in print and reduced-motion |
| `.sticky` | The pinned column (`position: sticky`) |
| `.fig` | Hairline rule above and below. **No card, no border, no shadow** |
| `.fig-num` | `Fig. N —` label |
| `.fig-big` | The one oversized number. Mono, `tabular-nums`, ~3.3rem |
| `.fig-sub` | What the big number is a count *of* |
| `.fig-verdict` | Bordered chip, 1–2 words |
| `.strip` / `.fig-tick` | One mark per unit. `.fig-tick.hit` is the excluded thing |
| `.fig-note` | Interprets the current state; changes with it |
| `.fig-rail` | The standing caveat. Always visible, never state-dependent |
| `.pill` | One number, accent-tinted |

Only three colours are in play: `var(--accent)` for the live/selected thing, `var(--grade-x)` for
the excluded thing, and greyscale (`--ink` / `--muted` / `--faint` / `--track`) for everything else.
`var(--grade-b)` marks a known-unknown inside `.fig-rail`. **Never introduce a colour that is not
already a token in `_shared/tokens.css`.**

Three media blocks are not optional:

```css
@media (max-width: 760px)               { /* one column; .sticky gets order: -1 */ }
@media (prefers-reduced-motion: reduce) { /* .step { min-height: 0 } .inline { display: block } */ }
@media print                            { /* same as above + .qbox { display: none } */ }
```

Both reduced-motion and print render **the complete end state**: every step visible, every `.inline`
verdict shown, the figure un-stuck. A reader who cannot scroll still gets the whole argument.

## 4. The JS state-and-render shape

Four functions, in this order. Keep the boundaries — most bugs come from `show()` computing
something it should have been handed.

```js
/* 1. CONSTANTS — ported from the pipeline, named after their source. */
const N = 13;                       // dataframework/shingles.py SHINGLE_N
const DEFAULT_INPUT = '…';          // synthetic; never real eval/benchmark text

/* 2. PURE HELPERS — a port of pipeline logic. Must be verified against it (§6). */
const words   = (t) => t.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
const windows = (t) => …;

/* 3. THE UNITS — one entry per state. Prose lives here, not in the DOM builder. */
const STATES = [{ marg, lead, bold, tail, transform }, …];

/* 4. STATE — the minimum that cannot be derived. */
let derived = new Set();   // rebuilt by recompute()
let current = 0;           // which state is showing
```

```js
/* guard() — the precondition. Returns null when the widget can answer, or a refusal.
 * Without this, a widget renders a confident verdict for an input it cannot handle,
 * and "0 collisions" reads as a successful attack rather than "unanswerable". */
const guard = () => {
  const n = words(input.value).length;
  if (!n)     return { verdict: 'NOTHING TO PROTECT', note: '…', inline: '→ …' };
  if (n < N)  return { verdict: 'NOT INDEXABLE',      note: '…', inline: '→ …' };
  return null;
};

/* stateFor(i) — PURE. index → every number the renderer needs. No DOM. */
const stateFor = (i) => ({ text, marks, hits });

/* show(i) — RENDER ONLY. Checks guard() first, computes nothing itself. */
const show = (i) => {
  current = i;
  stepEls.forEach((el, k) => el.classList.toggle('on', k === i));
  const refused = guard();
  if (refused) { /* render the refusal, return early */ return; }
  const st = stateFor(i);
  big.replaceChildren(renderNumber({ value: st.hits, unit: '…',
    provenance: 'measured', source: 'computed in your browser' }, { unit: false }));
  …
};

/* recompute() — the input changed. Rebuild derived state, refresh EVERY step's
 * inline verdict (print depends on them), then re-render the current state. */
const recompute = () => {
  derived = new Set(windows(input.value));
  const refused = guard();
  stepEls.forEach((el, i) => { /* .shard and .inline for every step */ });
  show(current);
};
```

Wiring — three ways in, one renderer:

```js
input.addEventListener('input', recompute);              // the reader types
stepEls.forEach((el) => el.addEventListener('focus', …)); // Tab — the keyboard path
new IntersectionObserver(…, { rootMargin: '-45% 0px -45% 0px' });  // scroll — the pointer path
playAll.push(() => show(STATES.length - 1));             // print forces the end state
recompute();                                             // first paint
```

`rootMargin: '-45% 0px -45% 0px'` makes the step crossing the viewport's middle band the active one.
Focus and scroll both call `show()`, so the keyboard reader gets the identical sequence — this is
how a scrollytelling explainer stays operable without a pointer, since deleting the control row
(§18.2) also deletes the thing a keyboard would otherwise land on.

**Non-negotiables, all enforced in the reference:**

- Every number reaches the DOM through `renderNumber()` from `_shared/num.js`. It throws on a bare
  number, which is the point.
- Ported logic is **verified against the Python**, not assumed. §1's port was checked on all six
  states before it shipped.
- Invent nothing. If a value is unavailable, render `{value: null, provenance: 'unknown'}` and let
  the UI say so. Do not fabricate plausible-looking intermediates — §1 shows window *counts* rather
  than fake blake2b digests, and the caption explains why.

## 5. Copy voice

| Slot | Rule | Reference |
|---|---|---|
| `h2` | A claim. Never a topic | "Thirteen words in a row is a fingerprint" |
| `.claim` | Say what changes and what the reader does. Bold the **variables** | "the **sentence is yours to change**, and scrolling attacks it" |
| `.marg` | 2–4 words. A label plus a verb | "Attack 3 · reflow it" |
| `.step p` | One sentence of mechanism, one bolded variable, one consequence | "Whitespace is not a token either, so **the window count is identical**." |
| `.fig-verdict` | 1–2 words, shouted | `SHARD DROPPED` |
| `.fig-note` | Interprets *this* state — must differ between states or it is decoration | "No red anywhere is the alarm." |
| `.fig-rail` | The standing limitation, in plain words | "…not a certificate that the corpus is clean." |
| `.pill` | **One** number, under ~30 characters | `13 words = a fingerprint` |

Collective, present tense: *"what we are protecting is…"*. Bold the thing that changes, never the
adjective. Never write "click here", "explore", "interactive", or "dashboard" — describe what
happens and let the reader infer the affordance.

The last state is allowed to defeat the widget. §1's fifth attack walks straight through the gate,
and saying so is worth more than showing only where the method works — §11 self-refutation.

## 6. Before you ship

- The title is a claim; the standfirst says what changes.
- Removing the interaction would destroy the argument, not just the polish.
- Every state fills the same slots.
- Red appears only on the excluded thing.
- `guard()` covers every input that has no honest answer.
- Ported logic verified against its source; every number provenance-typed.
- Tab reaches every control and walks every state.
- Print and reduced-motion render the complete end state.
- The pill states one number.

## 7. Topologies in use

Not every section is an explainer, and forcing one where the claim needs no reader-driven
comparison is what `EXPLAINER_PROMPT.md` §9 warns against. What shipped:

| Topology | Class | Where |
|---|---|---|
| Scrollytelling | `.scrolly` (`.wide` for charts) | report §§1, 2, 4, 5, 6, 7, 10 |
| Single canvas | `.canvas` + `.unit` | atlas `#data`, `#confidence` |
| Small multiples | `.canvas` (static), `.tierbars` | atlas `#benchmarks`, report §5's tiers |
| Inline | `.inlinectl` inside a `<p class="claim">` | report §8 |
| Margin-driven | `.marginal` + `.gutter` | report §9 |
| Chart / prose | `.chartblock`, `.compare` | report §§3, 11 |

Two rules that survived contact with all of them: **the canvas never filters** — controls change
the encoding, never which units are visible — and **red is only the excluded thing under the
current encoding**, which is why `#data` can recolour five ways without red ever meaning two
things at once.

## 8. Known deviations

- **`.tick` is `.fig-tick`.** Moving the rules to `_shared/` collided with `.axis .tick` in the
  competitive-frame chart, which would have put a 9×19px box behind every axis label. Check for
  collisions before adding a name to the shared sheet — the report is a big file.
- **Rename class strings, never JS identifiers.** `const gsub = …` renamed wholesale becomes
  `const fig-sub = …`, which is not JavaScript.
- **`EXPLAINER_PROMPT.md` §15 asks for a serif display face.** `AGENTS.md` forbids serif, and the
  repo design language wins. The rest of the editorial register — hairlines instead of cards, mono
  numerals, marginalia, `Fig. n`, one accent — carries it without the typeface.
- **"Red only on the excluded thing" now holds page-wide** on both surfaces. It did not when this
  file was written; the sections that used `--grade-x` for other purposes have been rebuilt.
