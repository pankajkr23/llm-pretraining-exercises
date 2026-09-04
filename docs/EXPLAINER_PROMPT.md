# EXPLAINER PROMPT — reusable template

Copy the block in §7, fill the six slots, hand it to the agent. Sections 1–6 are the spec the prompt refers to; keep this file at `docs/EXPLAINER_PROMPT.md` so the agent can read it.

---

## 1. The one rule everything else serves

**The interaction *is* the argument.**

An explainer makes exactly one claim, and the thing the reader does with their mouse is the proof of that claim. Not an illustration of it — the proof.

- *"One pipeline, many datasets"* claims a data point means something different at every stage. You **walk the stages** and watch the volume bar collapse a millionfold. The walking is the argument.
- *"The V4 corpus, dissected"* claims a third of the corpus was deliberately hidden from the selector. You **flip the OPUS lens** and see it. The flip is the argument.

If the reader could learn the same thing from a static image, you have built decoration. Delete it and write a paragraph instead.

---

## 2. The five obligations

**There is no house layout.** An earlier draft of this file printed a nine-box diagram traced from two existing widgets; it has been removed, because a reader given a diagram builds the diagram. Topology is chosen per widget in §17.

What every explainer owes its reader, regardless of shape:

**① A claim as the title.** "One pipeline, many datasets," not "Data Pipeline Overview." A topic invites skimming; a claim invites checking.

**② An entry sentence that says what changes.** Name the variables and bold them. If the reader must discover the interaction by hovering, you have hidden it.

**③ Identical slots for comparable things.** Whatever facets you show for one unit, show for all, in the same order. This is what makes comparison possible — it is the single most transferable discipline here, and it constrains content, not layout.

**④ The exception, marked.** The held-out set, the dropped pool, the blocked tier. Give it visual separation and the page's only red. Where a design excludes something is where its reasoning becomes visible.

**⑤ Provenance on the mark.** Measured, estimated, unknown — expressed in the geometry (§14.2), never as a footnote.

Five obligations. **No prescribed arrangement.** Where controls sit, whether controls exist, whether it scrolls or clicks or does neither — §17 decides, from what the surface is for.

## 3. Voice

Collective, present tense, walking alongside: *"What we are looking at is…"*, *"as we select a pool we can read its real numbers"*, *"we can now see why…"*

Bold the **variables**, not the adjectives. In "the **cache grows eightfold** while the **tokens per second** barely move," the bold marks the two things the control actually changes — that is a promise the widget keeps, and a reader learns within one interaction that bold means *watch this*.

Never write "click here," "explore," "interactive," or "dashboard." Describe what happens; the reader infers the affordance.

---

## 4. Interaction — vocabulary for the Inspector family

*Applies when §17 selects Inspector. Other families have their own grammar.*

| Level | Changes | Example |
|---|---|---|
| **Global control** | The frame. Everything re-renders | modality · growth stage · tokenizer |
| **Local selection** | The detail panel only | which stage · which pool |
| **Lens (optional)** | Reveals a hidden dimension of the same data | OPUS eligibility · licence tier · contamination |

The lens is the highest-value control in the vocabulary, because it shows the reader something that was already there and invisible. Reach for it whenever your claim is *"look at what's being excluded."*

State changes are instant. No transitions longer than 200ms on control response — the reader is comparing, and animation between states destroys comparison.

---

## 5. Visual rules — general craft

**Numbers.** Large, bare, tabular figures. Unit and context small and grey beneath. `164.1B` / `4,894 shards`. Never `164.1 Billion tokens` on one line.

**Colour is semantic and consistent.** A pool's colour is identical in the legend, the bar, the card border, and the dot beside its detail heading. Colour is never decorative. **Red is reserved for the excluded thing** — dropped, blocked, never-trained-on — and appears nowhere else.

**Chips carry short factual labels**: `FineWeb 15T`, `leak 0.87%`, `garbage 1.36%`. Chips never carry sentences.

**Bars.** Two-tone, no gradient, no axis, no gridlines. A bar here is a proportion you feel, not a value you read — the number is already printed next to it.

**Density is the point.** These are dense on purpose. Whitespace between *groups*, tight within them.

---

## 6. Technical constraints

- Self-contained: one HTML block, inline `<style>` and `<script>`. **No React, no D3, no CDN imports.**
- All data precomputed and inlined as a JS object at the top of the script. **No fetches.** No values computed at render time that could have been computed at build time.
- Hand-written SVG for charts.
- Every displayed number carries `data-provenance="measured|estimated|unknown"`; estimated renders with a dotted underline, unknown with reduced opacity and italics.
- Full keyboard operation: arrow keys move within a segmented control, Tab reaches every unit, Enter selects, visible focus ring.
- `prefers-reduced-motion: reduce` → jump to end state, no transitions.
- **Invent nothing.** Every figure comes from the supplied data. Anything unavailable is `null` and the UI says so.

---

## 7. THE PROMPT — copy this

**Fill the TOPOLOGY and FAMILY slots first (§§17, 10). They decide everything below them.**

````markdown
Build one interactive explainer widget, following `docs/EXPLAINER_PROMPT.md`.

## Topology  (§17 — decide this before anything else)
<scrollytelling | small multiples | single canvas | margin-driven | inline |
 dashboard column>
Justify in one line from what the SURFACE is for, not what the data is.

## Family  (§10)
<Inspector | Simulator | Optimizer | Destroyer | Diff | Adversary |
 Accumulator | Budget>
Must differ from the family of the adjacent widget.

## The claim
<ONE SENTENCE. The single thing this widget proves. If you cannot say it in
one sentence, the widget is doing two jobs — stop and ask.>

## The interaction that proves it
<WHAT THE READER DOES, and what they see change as a result. This must BE the
argument, not illustrate it.>

## Controls
- Global: <segmented control — 2 to 5 options>
- Lens (optional): <a toggle revealing a hidden dimension of the same data>

## Overview strip
<Stage cards | stacked bar | ranked bars. What responds to the global control.>

## Special rail
<The exception: held-out, dropped, blocked. Its red chip label.>

## Selectable units
<What the reader clicks. Big number + small qualifier for each.>

## Detail panel slots — identical for every unit
1. <SLOT>   2. <SLOT>   3. <SLOT>   4. <SLOT>
(4–7 slots. One MUST be "WHY IT EXISTS". Vary the register: code block for a
formula, chips for exemplars, prose for reasoning, mini-chart for a distribution.)

## The data
```json
<PASTE THE FULL DATA OBJECT. Every number provenance-typed:
 {"value": 164.1, "unit": "B tokens", "provenance": "measured",
  "source": "arXiv:2606.07404"}>
```

## Footer note
<Provenance caveat, or the one load-bearing fact the design turns on.>

## Takeaway pill
<The number to remember. Under ~30 characters.>

## Constraints
Self-contained HTML: inline style and script, no framework, no CDN, no fetch.
Hand-written SVG. Data inlined at the top of the script. Keyboard-operable.
Honours prefers-reduced-motion. Every number carries data-provenance.
Invent nothing — unavailable values are null and the UI says so.

## Before you return, verify
- [ ] Topology was chosen from §17 and is NOT a dashboard column unless justified
- [ ] Could everything be shown at once? If yes, the selector was deleted (§18)
- [ ] The reader is asked to predict before being told, or there is a reason not to
- [ ] Estimated values are visually distinct from measured ones IN THE MARK
- [ ] Screenshot beside the reference: could a reader confuse them? (§16)
- [ ] The title is a claim, not a topic
- [ ] The standfirst tells the reader what to do
- [ ] Removing the interaction would destroy the argument — not just the polish
- [ ] The detail panel has identical slots for every unit
- [ ] Red appears only on the excluded thing
- [ ] Every number is provenance-typed; nothing was invented
- [ ] Tab reaches every control; arrows move within the segmented control
- [ ] reduced-motion renders the complete end state
- [ ] The takeaway pill states one number
````

---

## 8. Worked example — filled slots

For §6.3 of the report, "The Tax":

| Slot | Filled |
|---|---|
| **Claim** | Indic fertility is a tariff, not a property of the scripts — and it is priced in crores |
| **Interaction** | Switch tokenizer and watch every language's bar move; switch the unit and watch tokens become rupees |
| **Global control** | Tokenizer: `Gemma 4 262K · Sarvam 200K · o200k · cl100k · ours 208,896` |
| **Lens** | ☐ Show merge-failure rate — reveals the *mechanism* behind the tax (r = 0.89) |
| **Overview strip** | 22 language bars + English baseline; parity ratio as a large number with a pass/fail ring at 1.5 |
| **Special rail** | Tier-3 scripts (Ol Chiki, Meitei Mayek) — chip: `coverage guaranteed, fertility best-effort` |
| **Selectable units** | Each language. Big: `2.41` · small: `tok/word · IN22-Gen` |
| **Detail slots** | ① FERTILITY (mean/median/P95) ② VS ENGLISH (expansion ratio) ③ COST (₹ and H100-days for this language's slice) ④ MERGE HEALTH (chips) ⑤ **WHY IT EXISTS** (prose: script properties vs tokenizer design) ⑥ ACROSS TOKENIZERS (mini bar chart) |
| **Footer** | "Fertility measured on IN22-Gen (source-original, all 22 scheduled languages) under NFC, special tokens excluded. FLORES-200 figures shown for comparability with published work are translated-from-English and marked as such." |
| **Pill** | `parity 1.48 · target ≤1.50` |

Note what this does: the **lens shows the mechanism**, so the reader stops thinking "Malayalam is expensive" and starts thinking "this tokenizer failed to learn Malayalam merges." Same data, different conclusion. That is what a lens is for.

---

## 9. When *not* to build one

- The claim needs no comparison → write a sentence.
- You have more than one claim → build two explainers, or cut one.
- The data is a single series → a static chart is more honest and loads faster.
- You cannot name what the reader does → you have a chart, not an explainer. That is fine; call it a chart.

---

## 10. Interaction families — vary these, or the page reads as one note

Sections 2–7 describe **one** family: the *Inspector*. Both reference screenshots are Inspectors. It is a good pattern and a bad default — a page of twelve Inspectors is monotonous however well each is built.

Pick the family from the **shape of the claim**, not from the shape of the data.

| Family | The claim it proves | What the reader does | Effort |
|---|---|---|---|
| **Inspector** | "X differs across categories" | Select a unit, read identical slots | low |
| **Simulator** | "X causes Y causes Z" | Move a dial, watch consequences cascade | med |
| **Optimizer** | "There is an optimum, and here it is" | Drag along an axis until two curves cross | med |
| **Destroyer** | "Something is being lost" | Raise a threshold, watch things die — then a rescue toggle | med |
| **Diff** | "These look equivalent and are not" | Feed identical input to two systems, watch them diverge | low |
| **Adversary** | "This boundary is real" | **Try to defeat it. Fail.** | low |
| **Accumulator** | "This compounds" | Step forward; each step multiplies the last | med |
| **Budget** | "You cannot have everything" | Allocate a fixed pool; taking from one takes from another | high |

**Mapping for this project** — note that variety is available for free once you stop defaulting to Inspector:

| Widget | Family | Why that one |
|---|---|---|
| The Gate | **Adversary** | Paste a MILU question, watch it get rejected by name. The reader tries to break it |
| The Vocab Crossing | **Optimizer** | Cost curve vs saving curve; the crossing *is* V = 208,896 |
| The Filter | **Destroyer** | Languages collapse as the threshold rises; Always-ON restores the floor |
| The Budget | **Accumulator** | ×1 → ×4 epochs; watch 300B become 1,200B |
| The Mix | **Budget** | Zero-sum. Move share to Indic, watch code fall |
| The Tax | **Diff** | Same sentence, five tokenizers, five token counts |
| The Pipeline | **Accumulator** | One document traverses nine stages and survives or is stamped |
| The Chain | **Simulator** | Four dials, cascading verdict |
| Instruments · Competitive Frame · Critical Path · Languages | **Inspector** | Genuinely the right pattern here |

**The rhythm rule: no two adjacent explainers share a family.** If the audit hands you two Inspectors in a row, re-frame one of the claims until it wants a different interaction.

---

## 11. Four devices that cut across every family

**The lens** *(already in §4)* — reveal a dimension that was present and invisible.

**The ghost** — render the counterfactual behind the current state, dimmed. The Filter shows live language heights **and** a ghost outline of "without the Always-ON lane." Cheap to build, and it makes an absence visible, which is otherwise nearly impossible.

**The reader's own input** — a text field instead of a canned example. Type your own sentence and watch five tokenizers split it differently. Personal input beats any prepared example, because the reader cannot suspect you chose it favourably.

**Self-refutation** — one control that lets the reader break your recommendation and watch the design fail. Drag Malayalam's fertility target to 1.4 and the required vocab blows past 262K while the embedding table eats 10% of a 40B model. **Showing exactly where your own design collapses is the least fakeable evidence that you understood it.** Put this on the Vocab Crossing.

---

## 12. Two structural moves worth one use each per page

**Scale distortion.** When a number spans orders of magnitude, a linear bar lies. The 10⁶ volume collapse from pretraining to RL deserves a log axis or a zoom-out mechanic — something that forces the reader to *travel* the distance rather than read it.

**The wrong answer first.** Open in the state most people assume is correct, let them see it fail, then reveal the fix. The Filter does this naturally: the intuitive move is "raise quality," and raising quality kills your languages. Do not spoil it in the standfirst.

Use each **once per page**. They are strong effects and they cheapen fast.

---

## 13. The reference pattern is a floor, not a ceiling

Sections 2–7 were reverse-engineered from two existing widgets. Everything there is sound, but it describes a **selector-and-inspector**, and that pattern has five structural limits. Naming them is how you get past it.

| Limit | What it costs |
|---|---|
| **The detail panel is a dead end** | You click, you read, and nothing follows. It is a reference lookup wearing an explainer's clothes. You learn facts, never relationships |
| **The reader is never wrong** | It states a claim then demonstrates it. People remember what they got wrong far better than what they were told. This format never lets them be wrong |
| **No comparison across selections** | The identical-slot discipline makes comparison *possible* and the UI never *delivers* it. You see D1, then D2, never D1 beside D2 |
| **Selection has no consequence** | Picking a pool changes a panel. It does not change the world — no downstream number moves |
| **Certainty is the only available mode** | There is no way to render "we measured this" versus "we are guessing." Every number looks equally solid |

That last limit matters most here. The distinguishing content of this research is its **confidence ledger, blind spots and corrections log** — and the reference format has no way to express any of it. Adopting the format unchanged would hide your best material.

---

## 14. Four upgrades that lift it above the reference

### 14.1 ★ Predict before you reveal

The strongest single change, and roughly twenty lines of JS.

Before showing an answer, ask for one. *"Drag the marker to where you think Malayalam's fertility sits."* The reader commits. Then reveal, and **keep their guess pinned on the chart** as a ghost marker with the gap labelled.

```
your guess  ▲ 3.2
actual      ● 6.8      you were 2.1× low
```

The gap is the lesson. Nobody forgets a number they were wrong about. Use it on **The Tax** (guess Malayalam's tax), **The Budget** (guess what fraction of Indic is natural), and **The Filter** (guess how many languages survive a 0.7 threshold). Everyone guesses too optimistically on all three, which is exactly the point.

Never do this more than three times on a page — it demands effort, and effort spent is a budget.

### 14.2 Uncertainty in the mark, not the caption

Provenance belongs in the geometry, not a footnote:

- **Solid fill** = measured
- **Hatched fill** = estimated
- **Open outline + range band** = a known range, not a point
- **Grey slash** = unknown, and the widget says so where the bar would be

A chart that visibly shows what it doesn't know is rarer, and more credible, than one that doesn't. When you have 22 language bars and four are hatched, the reader learns your epistemic state at a glance, without reading a word.

### 14.3 A comparison tray

Selections accumulate into a strip along the bottom instead of replacing each other. Three pools pinned side by side, identical slots aligned, differences highlighted. `Clear` resets.

This is the fix for the dead-end panel: it converts lookup into analysis. Cheap, because the identical-slots discipline already guarantees the rows line up.

### 14.4 Consequence propagation

When a selection changes something downstream, show it moving. Select a dataset in The Mix and the parity ratio, the ₹ figure and the effective-Indic share all re-compute in a persistent header strip.

The reference pattern's selections are inert. Making them consequential is what turns a catalogue into an instrument.

---

## 15. Visual register — do not ship a clone

The reference widgets use a product-dashboard register: white cards, rounded corners, pastel accents, chip clusters, small-caps labels. It is competent, it is also everywhere, and a submission that reproduces it reads as derivative regardless of how good the underlying analysis is.

**Use an editorial-scientific register instead.** It is not merely different — it is *more appropriate*, because this is a research atlas, not an ops console. The form should say *paper*, not *product*.

One deliberate exception: **the typeface stays system sans.** An earlier draft of this table asked for a serif display face, which contradicts `AGENTS.md` ("system sans, no serif") — the repo-wide design language wins, and it is the one row of this table that does not change. Everything else below — hairlines instead of cards, mono numerals, marginalia, `Fig. n`, one accent — carries the register on its own.

| | Dashboard register (reference) | Editorial register (use this) |
|---|---|---|
| Container | Card with border + shadow | **Hairline rule above and below.** No card |
| Claim type | Sans, bold, medium | **System sans**, larger, generous leading |
| Labels | Small-caps grey chips | **Marginalia** in the left gutter, small and quiet |
| Numbers | Sans tabular | **Mono tabular**, larger than feels comfortable |
| Palette | 4–5 pastels | **One accent + one alert. Everything else greyscale** |
| Caption | Italic below | **`Fig. 3 —` numbered**, italic below, referenced from prose |
| Provenance | Footnote | **Gutter marginalia beside the mark it qualifies** |
| Density | Uniform | **Hierarchical** — one number per widget is 3× the size of the rest |

Two consequences worth the effort:

**Figure numbers make the report citable.** "As Fig. 7 shows" in the prose, linking to the widget. That is how papers work, and it is why the two-surface design works — the report cites, the atlas holds.

**A restrained palette forces semantic colour.** With one accent and one alert, colour *must* mean something. The reference's five pastels let colour drift into decoration.

---

## 16. The test for whether you have escaped the reference

Screenshot your widget next to the original. If a reader could mistake one for the other, you have copied a solution instead of solving a problem. Three questions:

1. **Can the reader be wrong here?** If not, add §14.1.
2. **Does the widget show what it doesn't know?** If not, add §14.2.
3. **Would this look at home in a journal rather than a SaaS console?** If not, apply §15.

Keep from the reference only what is *general good practice*, not signature: claim-titles, standfirsts that say what to do, identical detail slots, an exception rail. Those belong to nobody. The card-and-chip aesthetic belongs to someone.

---

## 17. Topology — the layer §15 did not change

§15 swapped typefaces and removed card borders. That is paint. The **layout topology** is what a reader recognises, and the reference's topology is a dashboard column: controls on top → overview strip → selectable units → detail panel below. Keep that arrangement and the work reads as derivative no matter what typeface it wears.

Six topologies. Pick by what the *surface* is for, not by what the widget contains.

| Topology | Shape | Reader does | Fits |
|---|---|---|---|
| **Dashboard column** *(the reference)* | Stacked panels, controls above | Clicks | Ops consoles |
| **Scrollytelling** | Visual pinned, prose scrolls past it, each paragraph advances the state | **Reads. Nothing else** | A linear argument |
| **Margin-driven (Tufte)** | Visual in the text column, controls and provenance in a wide gutter | Reads, occasionally nudges | Dense reference prose |
| **Small multiples** | **All** units rendered at once, small; interaction only highlights | Compares at a glance | Anything with 10–200 comparable units |
| **Single canvas** | One coordinate space; controls change the *encoding*, not the content | Re-maps axes | Replacing six widgets with one |
| **Inline / in-sentence** | The control lives inside a sentence, which recomputes | Reads and adjusts | Micro-claims inside prose |

### Which topology each of these two surfaces gets

**`/report` → scrollytelling.** It is a linear argument, so make the reader's only job reading. Visual sticky on one side, prose scrolling on the other; each paragraph sets the visual state. Pure `IntersectionObserver` — about forty lines, no library. It prints beautifully because you design the terminal state and `@media print` just freezes it.

**`/reasoning` → small multiples on a single canvas.** It is a reference, so show everything: all 22 languages, all 145 datasets, all 31 benchmarks at once. Controls change *encoding* — colour by tier, size by tokens, sort by fertility — never *which items are visible*. One canvas replaces six explorers.

Neither is a dashboard column. That is the actual escape.

---

## 18. Elegance is subtraction

> **The most elegant interaction is the one you removed.**

The reference needs a selector because it can only show one pool at a time. That is a **failure of layout dressed as a feature**. If all twenty-two languages fit on screen at once, the selector was never necessary — and the reader learns more, faster, because comparison is free rather than a memory exercise across clicks.

This inverts §2's advice, and the inversion is correct. Apply in this order:

1. **Can everything be shown at once?** → small multiples. Delete the selector.
2. **If not, can the reader's position do the work?** → scrollytelling. Delete the controls.
3. **If not, can the control live in a sentence?** → inline. Delete the control row.
4. **Only then** → add an explicit control.

Three subtractions worth making everywhere:

- **Delete the card.** A hairline and whitespace separate content as well as a border does, at a fraction of the ink.
- **Delete the legend.** Label the marks directly. A legend forces the eye to travel and remember.
- **Delete the axis.** If the number is printed on the bar, the axis is redundant. The reference already does this — it is the one place it is most disciplined.

Ink you removed cannot mislead. Elegance in data graphics is almost entirely subtraction, and it happens to also be less code.

---

## 19. The page *is* the widget

The reference's deepest assumption is that an explainer is a **component embedded in a document**. Twelve sections, twelve boxes.

The stronger move: **two continuous interactive documents.** `/report` is one scroll-driven argument where the visual never fully leaves; `/reasoning` is one canvas you interrogate from many angles. Not twelve widgets — two instruments.

Practically this means:

- A **persistent state header** on `/report` — parity ratio, effective Indic share, ₹ cost — that updates as you scroll and as you touch anything. The reader always knows where they stand.
- **One coordinate space** on `/reasoning` that everything lives in, so moving between licence, fertility and contamination views is a re-encoding rather than a page change. Continuity of object teaches relationships that separate boxes cannot.
- Section boundaries become **rules and figure numbers**, not container edges.

This is less work than twelve widgets, not more. And it is structurally impossible to mistake for the reference.

---

## 20. Will it look like the screenshots?

Follow §§2–7 alone: **yes**, and that is the failure mode.

Apply §§17–19 and the answer is no, for three reasons a reader can see in one second:

| | Reference | Yours |
|---|---|---|
| Topology | Dashboard column, click-driven | Scroll-driven argument · single interrogable canvas |
| What is shown | One unit at a time | **All units at once**; interaction highlights |
| Register | Cards, chips, five pastels | Rules, marginalia, one accent, mono numerals, `Fig. n` |

And two things a reader *feels* rather than sees: they were asked to guess before being told (§14.1), and the graphics visibly distinguish what you measured from what you estimated (§14.2). Neither exists in the reference.

**What you keep** — claim-titles, standfirsts that say what to do, identical comparison slots, an exception rail in red. That is general craft and belongs to nobody.
**What you drop** — the dashboard column, the card-and-chip surface, one-unit-at-a-time. That is the signature.
