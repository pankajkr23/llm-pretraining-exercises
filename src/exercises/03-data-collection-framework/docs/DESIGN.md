# Design Document — `03-data-collection-framework`
### Two surfaces: **The Decision** (linear argument) and **The Reasoning** (explorable reference)

Plain HTML/CSS/JS, zero runtime dependencies, precomputed `data.json`, deployed at `/03-data-collection-framework/`. Matches the repo's existing exercise pattern.

---

## 0. Why two surfaces, and what each is for

The grading constraint — *"evaluated on how much you thought through; longer submissions score lower"* — is only contradictory if the report carries the depth. It shouldn't.

| | **The Decision** | **The Reasoning** |
|---|---|---|
| Read | Once, top to bottom | Repeatedly, by jumping |
| Length | ~4 printed pages | Unbounded |
| Contains | Answers, numbers, verdict | Why each number is that number |
| Structure | Linear argument | Anchored sections + explorers |
| Audience | The grader | You, in three weeks, and anyone who challenges a number |

**The binding rule: every number in The Decision is a link into The Reasoning.** The report asserts `V = 208,896`; the link lands on the cost-crossing chart that produced it. Depth without length. The Reasoning is an appendix that doesn't count against you.

---

## 1. Information architecture

```
/03-data-collection-framework/
├── index.html                 THE THESIS — one paragraph, The Chain, two doors
├── report/index.html          THE DECISION — 12 sections, print-ready
└── reasoning/index.html       THE REASONING — 14 explorers + 8 intuitions, anchored
```

Three pages, not more. The index is a claim, not a menu — matching the 01-introductions pattern where each page proves one thing.

**Index page contents, in order:**
1. Title + one-sentence thesis
2. **The Chain** widget (the hero — the Decision's §6.11 Chain, composed) — immediately interactive
3. Four stat cards: `15T seen / 9.65T unique` · `V = 208,896` · `parity 1.48` · `22 languages, 12 scripts`
4. Two doors: *Read the decision →* and *Explore the reasoning →*
5. Honesty line: which numbers are measured, which are estimated

---

## 2. Design system

Inherit `docs/DESIGN.md` (Apple-style) unchanged. Additions specific to this exercise:

**Semantic colour — earned, never decorative.**

| Token | Use | Constraint |
|---|---|---|
| `--grade-a` | Green tier | Must survive a colour-blind check; always paired with a letter (A/B/C/X), never colour alone |
| `--grade-b` / `--grade-c` | Amber tiers | |
| `--grade-x` | Blocklist | The only red on the site. Reserve it |
| `--lang-tier-1/2/3` | Language triage | Sequential, not categorical |
| `--measured` / `--estimated` | **Provenance of a number** | ★ Every figure carries one in the data. On the page it is marked **by exception only** — see below. Hover reveals the source |

That last token is the most important design decision on the site, and the first version of it was
wrong. It underlined every figure by provenance: solid for measured, dotted for estimated, faded for
unknown. On paper that reads as a running admission of what you do and don't know, costing zero
words. In practice the bundle is **80% unknown, 17% estimated and 2% measured**, so a mark fired on
*every number on every page* — and a signal that is always on is not a signal. Worse, a screen of
dotted underlines reads as broken links or spell-check squiggles, so it actively miscommunicated.

**Mark by exception instead:**

- **estimated** — the working default. **No mark.** Labelling the ordinary case tells the reader nothing.
- **measured** — a rule in `--grade-a`. It can afford to be loud because there are four of them.
- **unknown** — faded italics. These mostly render as an em dash anyway, so they read as the absence they are.

Where provenance genuinely changes how a figure should be read — a headline number, a stat tile —
say it **in words** underneath (`renderProvenance` in `web/_shared/num.js`), which cannot be mistaken
for a hyperlink. The invariant is unaffected: every figure still carries
`{value, unit, provenance, source}` through the pipeline and through `renderNumber`. This governs
how provenance is *shown*, not whether it is *recorded*.

**Typography:** one display face for claims, one mono for numbers and code. Numbers in tabular figures (`font-variant-numeric: tabular-nums`) so they don't jitter during count-up animations.

---

## 3. Animation grammar — what makes it smart rather than decorative

An animation earns its place only if it shows something the reader cannot compute in their head. Four legitimate uses:

| Kind | Shows | Example here |
|---|---|---|
| **Transformation** | A → B where B is non-obvious | 300B unique × 4 epochs → 1,200B seen |
| **Crossing** | Two curves meeting | Softmax cost vs fertility saving → the vocab answer |
| **Loss** | Something disappearing | Languages dying as the quality threshold rises |
| **Causation** | Move a dial, watch consequences | The Chain |

Everything else — fade-ins, parallax, decorative motion — is banned.

**Implementation rules:**
- Trigger on `IntersectionObserver`, play **once**, then hold the end state
- Respect `prefers-reduced-motion: reduce` → jump to end state, no motion
- 200–400ms, `ease-out`. Nothing longer than 800ms except deliberate step-throughs
- SVG line drawing via `stroke-dasharray` / `stroke-dashoffset`
- Number count-up via `requestAnimationFrame`, tabular figures, ~600ms
- No `setInterval` loops. Nothing that moves while the reader is reading text
- Every animated widget has a **Replay** control

---

## 4. THE DECISION and THE REASONING — interactive explainers

**They are explainers, not widgets, and the distinction is the standard.** A widget renders data
and leaves the reader to work out what it means. An interactive explainer teaches exactly one idea:
it states a claim, hands you controls to test it, and then says *in words* what the state you have
just created implies. A reader who touches nothing should still come away with the idea; a reader
who plays should be able to break the claim and see it admit that.

Anything on either surface that only draws a shape is unfinished. The shape is the evidence, not
the point.

**The seven parts.** Not every explainer needs all of them, but this is the shape they take:

1. **A framing line** above the card — why this question arises here, in one sentence.
2. **A title that is a question or a claim**, never a noun phrase. "When the text runs out, how many
   times can you re-read it?" beats "Repetition analysis".
3. **Teaching prose** with the numbers that matter in bold. This carries the idea for the reader who
   never touches a control.
4. **Controls with presets.** Sliders for continuous quantities, segmented buttons for real
   alternatives, and presets that correspond to *actual options in the data* — so moving between
   them is an argument, not a demo.
5. **Live stat tiles** showing the derived quantities, each with the arithmetic beneath it in small
   type, so the reader can check the sum rather than trust it.
6. **The chart**, with named zones and a marker for where the current setting sits. Zones are what
   turn a curve into a judgement.
7. **A callout that interprets the current state**, changing as the state does — and a footnote
   naming where the numbers came from and what is still unmeasured.

Worked examples in the code: `/report` §11 (repetition) and §6 (the mix, dissected). §3 shows the
variant where the honest answer is that we have not measured it — it puts a published measurement
and our empty column side by side rather than hiding the gap.

### 4.1 · The Instrument Panel — benchmarks, done properly
`/reasoning#benchmarks` + `/report` §8

Three linked views over all 31 benchmarks.

**View A — Coverage Matrix.** Capability rows × benchmark columns. Filled cell = measured; **empty cell = you have a goal with no instrument.** Capabilities: Indic knowledge · Indic generation · Indic MT · code · agentic · math · long-context · safety · instruction-following · cultural alignment · speech · multimodal.

> **The surprise:** *India-context worldview* has almost no column. You can allocate 5% of budget to it and no benchmark would notice its removal. That's the orphan-tier check made visual.

**View B — Trust Stratification.** Benchmarks sorted into four bands, because a leaderboard makes them look equivalent and they are not:

| Band | Members | Why |
|---|---|---|
| **Native-sourced** | MILU (1,500+ Indian competitive exams), IN-22 (source-original), INCLUDE, BhashaBench | Measure Indic competence |
| **Translation-derived** | IndicMMLU-Pro (IndicTrans2 + back-translation, 13 reviewers) | Measures *translated-English* competence. Report separately, weight lower |
| **Harness-dependent** | SWE-bench Verified, Terminal-Bench 2.0, BFCL, τ-bench | Measure model **+ harness**. Never publish without naming harness, scaffold, turn limit, context |
| **Contamination-prone** | Anything web-scraped or long-public | Requires the gate |

**View C — Split Policy.** Per benchmark: TEST-only · TEST + dedicated VAL · **fully held out**. MILU's 8,933-sample validation set gets called out as the one legitimate tuning surface. BhashaBench is flagged as the nominated honest broker — never looked at during development.

Each benchmark card carries: owner · coverage · size · access · **what it does not measure** · contamination status · the atlas note.

### 4.2 · Milestone presets on the Budget
`/report` §2

Four buttons above the Budget Sankey: **5T · 10T · 15T · 20T**. Clicking reconfigures the tiers and swaps the verdict strip.

| | Verdict shown |
|---|---|
| 5T | "Trivially achievable today — all green-licence. Under-ambitious." |
| 10T | "Safe target. ~8–12 weeks of pipeline work." |
| 15T | "**Recommended.** Requires manufacturing ~2.4T synthetic Indic." |
| 20T | "Phase 2. Principled once epoch scheduling is accounted for — ~11–13T unique + scheduled epochs." |

Cheap to build (it's a preset swap) and it answers the milestone question without a separate section.

### 4.3 · The Competitive Frame
`/report` §10, replacing the flat verdict card

A parameter axis with five real models placed on it — **Sarvam-30B** (2.4B active, Apache 2.0, 16T tokens) · **Sarvam-105B** (10.3B active, Apache 2.0, 12T) · **LightningLM 120B** (5.93B active, grown from a 1.78B seed on one 8-GPU node) · **Param-2-17B-A2.4B** (BharatGen non-commercial, ~22T tokens) · **BharatGen 1T** (₹988.6 crore, planned) — and **yours at 40B**.

Each node opens the full architecture row: experts, routing, attention, FFN, context, router, PT tokens, licence, access.

> **The surprise, stated plainly:** you sit between a **free Apache-2.0 105B that already exists** and a **publicly-funded 1T that is coming.** That framing is the honest competitive case, and showing it is stronger than hiding it.

Toggle: **total params** ⇄ **active params** — the ordering changes, which is the point.

### 4.4 · The Risk Matrix and The Four Unknowns
`/reasoning#risks`

**Matrix:** 4 classes (Technical / Legal / Safety / Unknown) × 3 severities. Cells sized by count, click to list. 21 risks, each with its mitigation.

**Separate, and prominent — The Four Unknowns.** These are not risks; they are admissions, and they deserve different visual weight:

| Unknown | Status |
|---|---|
| **Real deduplicated Indic volume** | ⚠️ **CRITICAL — never measured by anyone.** The 250–500B figure is an inference. It drives the entire corpus architecture |
| R\*_D for Indic / translated / synthetic text | Measured only on English web at ≤9B params. Synthetic plausibly has a lower ceiling |
| Bharat Data Sagar's 20T | Unaudited CEO statement; "tokens" is a soft unit for a 3PB multimodal store |
| Whether weights are a derivative of CC BY-SA data | Unresolved anywhere in the world |

Render as four large cards, not table rows. **This is the most credible section on the site** — publishing what you don't know, with the consequence attached.

### 4.5 · The Market
`/reasoning#market`

16 reported licensing deals as horizontal bars, ₹/$ scaled, buyer→seller. Every value labelled **reported, not confirmed**.

Three trend annotations overlaid:

1. **Training → live access.** Deal counts 2 (2023) → 11 (2024) → 18 (2025) → ~34 (2026 projected). Washington Post/OpenAI and Le Monde/Perplexity **explicitly exclude training**. → *For Indian news, negotiate grounding, not training.*
2. **No seller has leverage.** Largest counts are single-digit (Shutterstock 7, Wikimedia 6, Reddit 5) while "other" publishers account for 36. → *You don't need Times of India. You need forty regional-language publishers.*
3. **The market is small.** ~$4B globally in 2026. → *₹5–20 crore is a rounding error against a training budget, and buys the only thing money can buy: exclusive natural Indic text.*

### 4.6 · The Ledger — acquisition, ranked by rupee
`/reasoning#acquisition`

Eight items, ranked. **The top three render at ₹0 in a visually distinct band:**

1. NCERT / MoE commercial-training licence — **₹0, a letter** — unlocks K-12 in 36 languages, currently CC BY-NC-ND
2. BharatGen / Bharat Data Sagar MoU — **₹0–small** — ~20T claimed, ₹1,293 crore publicly funded
3. AIKosh contributor status — **₹0** — access, goodwill, compute credits

Then paid: regional publishers ₹3–15 cr · Tier-3 collection ₹2–5 cr · private eval set ₹15–30 lakh · 22-language red-team ₹2–8 cr · speech vendors (*likely unnecessary — 100,000+ free hours exist*).

> **The surprise:** the three highest-value acquisitions are free letters, not engineering.

### 4.7 · The Critical Path
`/report` §11 + `/reasoning#plan`

12 actions on a week axis, Week 1 → Week 12+. Dependency arrows. Colour by workstream (BizDev / Data / Tokenizer / Scaling / Architecture / Agentic / Evaluation).

**Two GATES render as hard stops with a lock:**
- **Tokenizer gate** (W3–6) — nothing starts until fertility is validated across all 22 scripts
- **Scratch-vs-grow gate** (W4–10) — head-to-head at ~2B on identical data

Plus the **Indic repetition-value ablation** (W3–6) — the second-most-important unmeasured number, a one-week experiment.

Click any bar → why · workstream · what it unblocks. Week-1 items carry a **₹0** badge.

> **The surprise:** the critical path does not begin with engineering. It begins with two letters and one measurement.

### 4.8 · The Evidence Timeline — the papers
`/reasoning#papers`

19 sources on a time axis, newest first. Marker shape by effect:

| Marker | Meaning | Examples |
|---|---|---|
| ▲ | **Reversed a recommendation** | Devanagari OCR study (olmOCR 40.5 → Qwen3-VL-8B 75.2); Muennighoff (unique → effective tokens) |
| ● | Validated / sharpened | Tokenizer Tax (8.0× avg, 13.0× Malayalam, r=0.89) |
| ◆ | New corpus | IndicTalk (1.33M code-mixed), IKS-Instruct (24,795, 41 techniques), PatiGonit22K |
| ■ | New benchmark | Indic DiarBench (all 22), XIH-Bench (Language Boundary Effect) |
| ★ | Changed the strategy | IndicKLAR (code-mixed closes a 0.50 gap to 0.05); MoLGE (language-group experts) |

Each card: date · arXiv ID (linked) · category · finding · **action taken**. Filter by effect type.

Separately pinned: the **eight in-project papers** — LightningLM 0.1V, OPUS, Muennighoff, Sardana & Frankle, BrahmicTokenizer-131K, Kronecker, MUTANT-Indic, Tokenizer Tax — each with its extracted priors linked to `#priors`.

> **The surprise:** two of nineteen reversed a specific recommendation, and both were found in a single 30-day window.

### 4.9 · Tool overlay on the Pipeline
`/report` §7 + `/reasoning#tools`

Each of the nine cleaning stages shows the tool that implements it, with an **adopt / build / avoid** verdict.

- Setu (MIT, Spark, Indic-specific) — **adopt or fork, don't rebuild**
- datatrove (Apache-2.0) — the FineWeb pipeline
- NeMo Curator (Apache-2.0) — GPU-scale curation
- GlotLID — script-aware, 2000+ labels
- **Qwen3-VL-8B — adopt for Indic OCR**
- ~~olmOCR~~ — **struck through, with the 40.5 chrF++ figure inline.** The visual moment of the whole section
- IndicTrans2 / IndicTrans3-beta / IndicSeamless / IndicXlit / IndicConformer / IndicParler-TTS
- NeMo Gym, lm-evaluation-harness, Shoonya, DataOrchestra (evaluate)

---

## 5. The Dataset Card — where the per-dataset challenges live

This was the biggest omission. Every one of the 145 rows carries a *Risk & Notes* field that is the actual research value. The card must render it as structured, actionable content — not a paragraph.

```
┌─────────────────────────────────────────────────────────┐
│ IND-01  Sangraha                          [GRADE B]     │
│ AI4Bharat, IIT Madras                                   │
│ funded by EkStep · Rohini Nilekani Philanthropies ·     │
│ Google India                                            │
├─────────────────────────────────────────────────────────┤
│ LICENCE   CC-BY-4.0 (data) · MIT (tooling)              │
│           ✔ commercial  ✔ attribution  ✘ share-alike    │
│ SIZE      251B tokens  ▸ verified 64B · unverified 24B  │
│                        · synthetic 162B                 │
│ STAGE     PT       LANGUAGES  22 Indian languages       │
├─────────────────────────────────────────────────────────┤
│ ⚠ GOTCHAS                                               │
│  [COMPOSITION]  162B "synthetic" is machine-translated  │
│                 Wikimedia + transliteration — do NOT    │
│                 count as natural Indic                  │
│  [DEDUP]        known Varta news overlap — dedup vs     │
│                 Varta and IndicCorp before mixing       │
├─────────────────────────────────────────────────────────┤
│ ✦ OPPORTUNITY                                           │
│  Companion pipeline Setu (Spark, MIT) is arguably as    │
│  valuable as the data itself                            │
├─────────────────────────────────────────────────────────┤
│ FIVE GATES  provenance PASS · composition CONDITIONAL   │
│             contamination PASS · yield UNKNOWN          │
│             evidence PASS                               │
│ USED BY     foundation for most Indic LLM work          │
│             post-2024; MUTANT sampled from it           │
│ ACCESS      huggingface.co/… · github.com/… · 2403.06350│
│ CONFIDENCE  high — primary source                       │
└─────────────────────────────────────────────────────────┘
```

**Gotchas are typed**, so they can be filtered across all 145 at `/reasoning#gotchas`:

| Type | Meaning | Examples from the atlas |
|---|---|---|
| `LICENCE` | Terms restrict or complicate use | BhashaKritika (bespoke Krutrim licence + Gemma-3/Llama-3.3 generator chain) · EKA (licence string unstated) · Stack v2 (per-file + **live opt-out list**) |
| `DEDUP` | Duplication the publisher didn't remove | HPLT v2 (**17.3% byte-exact dupes remained** after official dedup) · DCLM (**~80% fuzzy**) · Varta ↔ Sangraha |
| `COMPOSITION` | It isn't what the headline says | Sangraha (65% translated, labelled synthetic) · CC-100 (paragraph-split, breaks long context) |
| `PROVENANCE` | Contaminated upstream | The Pile (**strip Books3 first**) · Bactrian-X (Alpaca → OpenAI-output lineage) |
| `SAFETY` | Harmful content | LAION-5B (**CSAM — absolute blocklist**) · DataComp CommonPool (documented privacy failures) |
| `ATTRIBUTION` | Ongoing obligation | Bhashini/ULCA (**record-level** attribution — maintain a manifest) · Wikipedia (share-alike, weights-as-derivative unresolved) |
| `AVAILABILITY` | May not actually exist yet | IndicTalk (*"we will release"* — **verify upload**) · Bharat Data Sagar (not redistributable) |
| `SOURCING` | A better route exists | Indian Kanoon → **use eCourts instead** (same content, far better legal posture) |
| `HETEROGENEITY` | Per-item review required | AIKosh (10,262 datasets, mixed terms — **do not bulk download**) · NDLI · Internet Archive DLI |

The `#gotchas` surface groups all 145 by type. **A "dedup traps" view that names HPLT's 17.3% and DCLM's 80% side by side is more useful than any chart on the site** — it is the thing that saves someone a month.

---


---

## 6. Data contract

Python pipeline → `web/data.json`, exactly the exercise-02 pattern.

```
web/
├── data.json              index: 145 datasets (core fields), 31 benchmarks,
│                          fertility table, priors, gaps, gates   (~98KB)
├── records.json           the reference arrays, fetched by /reasoning on demand
└── shingles.json          pre-hashed benchmark 13-gram shingles — HASHES ONLY

catalog.json               all 145 full dataset records — the reviewable register
benchmarks.json            all 31 full benchmark records
```

**Two registers, not 176 files.** This contract originally specified one JSON file per record — a
file per dataset and per benchmark — so that a licence downgrade would arrive as its own diff. In
practice that produced 176 files whose only reader was the build, and a pull request nobody could
review. A two-space-indented array gives the same line-level diff: change Sangraha's licence and the
diff shows those lines, with the record's `id` right above them.

**Nothing is duplicated into `web/`.** An intermediate draft had the pipeline write
`web/detail/<id>.json` for every dataset; those were byte-for-byte copies of the catalogue records.
`deploy/vercel/build.sh` now serves `catalog.json` and `benchmarks.json` alongside `web/`, and the
Reasoning surface reads them directly — the index already carries each `id`, `slug` and `grade`.

Pretty-printing is deliberate. It costs bytes on the wire and buys a readable diff; HTTP compression
recovers most of the difference, and an unreviewable register would defeat the point of tracking it.

Three hard rules:

1. **No benchmark source text in the bundle.** Only hashes. `tests/test_invariants.py` greps the built output for eval strings and fails if any appear
2. **Every numeric field carries `{value, unit, provenance: "measured"|"estimated", source}`.** The UI cannot render a bare number — the type system forbids it
3. **`generated_at` + `pipeline_version` in the header**, displayed in the site footer, so a stale build is visible

---

## 7. Print — one source, two outputs

`report/index.html` carries a `@media print` stylesheet:

- Widgets collapse to their **end-state SVG** (already rendered — no JS needed at print time)
- Controls, nav, and links-as-underlines hidden; link targets appended as endnotes
- Page breaks forced between sections
- Target: **4 pages of A4**

`Ctrl+P → Save as PDF` **is** the submission. One source of truth, no divergence between what you built and what you submitted, and the length constraint is enforced by the stylesheet rather than by editing discipline.

---

## 8. Build order

| # | Ship | Why |
|---|---|---|
| 1 | Data pipeline → `data.json` + provenance-typed numbers | Everything is a view over this |
| 2 | `#data` + `#benchmarks` explorers (5b) | Static tables first — the reference layer works before any animation |
| 3 | **The Gate (report §6.1)** | Smallest, most convincing. Build it before anything pretty |
| 4 | The Vocab Crossing (report §6.2) + Fertility lab | The technical spine of Q4 |
| 5 | The Budget (report §6.4) + The Filter (report §6.5) | The two transformation animations |
| 6 | The Chain (report §6.11) | Depends on §6.2–§6.4 existing; it composes them |
| 7 | The Instruments (report §6.8) + remaining intuitions | |
| 8 | Print stylesheet | Last, because it renders end states |

Ship each with `prefers-reduced-motion` and keyboard access from the start. Retrofitting accessibility into SVG interactions is miserable.

---

## 9. Anti-patterns

| Don't | Do |
|---|---|
| A dashboard grid of every chart | One claim per explainer, prose above, caption below |
| A chart that leaves the reader to interpret it | A callout that says what the current state means |
| Controls that only demo the interaction | Presets that are real alternatives from the data |
| Animate because it looks alive | Animate only transformation, crossing, loss, causation |
| Bare numbers | Every figure typed `measured` or `estimated`, visible in the underline |
| Hide uncertainty behind confident UI | The confidence ledger is a linked section, not a footnote |
| Colour alone to encode grade | Always a letter beside the colour |
| Autoplay loops | Scroll-triggered, once, with Replay |
| Show only what exists | `#gaps` is where the differentiation argument lives |
| Report restates the site | Report **decides**; site **justifies**; every number links |

---

## 10. Two open decisions for you

**Is the Gemma-4 fertility measurement runnable this week?** It's the spine of §4.3 and §4.4. If yes, those numbers ship as `measured` and this becomes the only submission with observed fertility against the named comparator. If no, they ship as `estimated` and the site is honest about it — but the difference in credibility is large.

**Confirm `d_model` for your 40B.** I assumed 6,144. The vocab optimum in §4.3 is sensitive to it, and the widget takes it as an input specifically so a reviewer can test the conclusion's robustness. Ship it with your real width as the default.

---

## 11. Traceability — every Atlas section

| # | Atlas section | Rows | Destination | Treatment | Prior status |
|---|---|---|---|---|---|
| — | Front matter: 3 Findings, milestone verdict, 3 free actions, honesty note | — | `/` index | Hero + stat cards + honesty line | ✅ covered |
| 1 | Master Dataset Catalog | 145 | `/reasoning#data` | Faceted explorer → **Dataset Card** (§4) | 🟡 thin — card undesigned |
| 2–4 | Green / Amber / Red tiers | 81/49/14 | `/reasoning#data` | Tier filter + **Tier Ribbon** on index | ✅ covered |
| 5 | **Benchmark Register** | 31 | `/reasoning#benchmarks` + `/report` §8 | **Instrument Panel** + Coverage Matrix + Split-Policy table (§4.1) | 🔴 **was one line** |
| 6 | Reference Token Budget | 12 tiers | `/report` §2 | The Budget Sankey | ✅ covered |
| 7 | **Milestone Ladder** | 4 | `/report` §2 | **Presets on the Budget** (§4.2) | 🔴 missing |
| 8 | 22 Languages triage | 22 | `/reasoning#languages` + `/report` §6 | **Language Cards** + The Filter | 🟡 thin — no standalone surface |
| 9 | Scaling & Selection Priors | 17 | `/reasoning#priors` | Prior cards, filter by paper | ✅ covered |
| 10 | **Reference MoE Architectures** | 5 | `/report` §10 | **The Competitive Frame** (§4.3) | 🔴 missing |
| 11 | **Risk Register** | 21 | `/reasoning#risks` | **Risk Matrix** + **The Four Unknowns** (§4.4) | 🔴 missing |
| 12 | Legal & Regulatory | 7 | `/reasoning#legal` | Timeline + obligation cards | 🟡 thin — buried in licensing |
| 13 | **Frontier Data Deals** | 16 | `/reasoning#market` | **The Market** (§4.5) | 🔴 missing |
| 14 | **Three Structural Trends** | 3 | `/reasoning#market` | Trend annotations on The Market | 🔴 missing |
| 15 | **Acquisition Plan** | 8 | `/reasoning#acquisition` | **The Ledger** — ₹-ranked (§4.6) | 🔴 missing |
| 16 | **Recommended Actions** | 12 | `/report` §11 + `/reasoning#plan` | **The Critical Path** (§4.7) | 🔴 missing |
| 17 | Confidence Ledger | 21 | `/reasoning#confidence` | Four-band list + provenance chips | ✅ covered |
| 18 | **Recency Log / Papers** | 19 | `/reasoning#papers` | **The Evidence Timeline** (§4.8) | 🔴 missing |
| 19 | Corrections Log | 14 | `/reasoning#changelog` | Before/after diff cards | ✅ covered |
| 20 | **Pipeline Tools** | 17 | `/reasoning#tools` + `/report` §7 | **Tool overlay** on the pipeline (§4.9) | 🔴 missing |

**Also missing and cross-cutting:** the *Risk & Notes* column — 145 rows of per-dataset gotchas. Designed in §4 and given its own surface at `/reasoning#gotchas`.

---


Enforced mechanically by TODO task 3.8 (INV-5).
