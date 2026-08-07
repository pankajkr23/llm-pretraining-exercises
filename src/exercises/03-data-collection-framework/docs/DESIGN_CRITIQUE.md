# DESIGN CRITIQUE

A self-critique of the first build of this exercise, written before rebuilding it. The verdict that
prompted it: *"not satisfied with the story telling and over complicated setup"*, and *"if I open the
decisioning page it should clearly tell me what datasets I could potentially use to create my corpus
of 15 trillion tokens and also the right mix."*

Both are correct. This document says why it happened, with evidence, so the same failure is harder to
repeat than to describe.

---

## 1. The scoping error, and where it propagated

The topic brief scopes this exercise as:

> Sourcing across the **full lifecycle: pre-training corpora, SFT, preference, safety, evaluation**

The exercise brief I then wrote (`BRIEF.md:5`) narrowed it to:

> a reusable, mechanically-checkable framework for deciding the **pre-training** data mix

That single word did more damage than every layout decision combined. Four of the five named
lifecycle stages fell out of scope in the first paragraph I wrote, and nothing downstream ever
questioned it. `FRAMEWORK.md` has five gates and three rules; **all eight are pre-training
constructs**. There is no gate for preference data, no rule for RL environments, no intuition about
alignment.

**The evidence was in the data the whole time.** Every one of the 145 catalogue records carries a
`stage` field:

| Stage | Records |
|---|---|
| PT | 101 |
| SFT | 47 |
| RL | 17 |
| EVAL | 15 |
| Safety | 1 |

44 datasets are post-training-only. That field is rendered in **exactly one place** — as a joined
string on a dataset card (`web/reasoning/index.html:297`) — and aggregated nowhere. **No module reads
it.** `sourcing.py` maps `category` to eight pre-training tiers and silently drops **36 of 145**,
including every `Preference / RL`, `SFT (General)` and `SFT (Indic Safety)` record, hiding ~1T tokens.

The framework answered a third of the question while discarding two-thirds of the evidence.

---

## 2. Structure derived from the data model, not from questions

`docs/DESIGN.md:430-449` lays out a traceability table mapping **record types to surfaces** — in
effect "21 records → 21 surfaces". The site's shape therefore came from the shape of the CSV, not
from any sequence a reader would ask questions in.

The result: 25 sections across two pages, ordered by record type. A reader looking for "which
datasets do I use" found it in section eleven of twelve, numbered **§5b** — a fractional number that
tells you it was inserted late, which it was. Its own source comment admits the framing:

> "which datasets fill this budget" had no answer on the page — which is the one question a data team
> would actually ask.

Meanwhile the strongest artefact on the site — the contamination gate you can attack with your own
sentence — sits at §1, answering the least urgent question a reader has.

---

## 3. The answers were already written and never published

This is the finding that most changes the rebuild, because it means the work is mostly publication.

`docs/DECISIONS.md` is titled **"The four answers, with arithmetic"**. Its headings are literally
`Q1 — What the data looks like`, `Q2 — How the data is cleaned`, `Q3 — How the model is tested`,
`Q4 — Fertility targets and vocabulary size`. **Seventeen of its concrete answers appear on neither
page:**

| Answer | Source |
|---|---|
| SFT ≈ 3M · DPO ≈ 200K · RLVR ≈ 60K × 8–16 rollouts, with per-category splits | `DECISIONS.md:48-52` |
| The SWE-smith Indic-issue-translation differentiator (verifiable RL reward in Hindi at zero annotation cost) | `DECISIONS.md:54-56` |
| Six objective-specific cleaning rules (Indic thresholds, code, agentic trajectories, LaTeX, temporal validity, OCR) | `DECISIONS.md:70-77` |
| NFC + homoglyph normalisation; India-specific PII (Aadhaar, PAN, voter, bank); provenance record | `DECISIONS.md:64` |
| R3 curriculum phases — Foundations / Knowledge / Reasoning / **Anneal** | `DECISIONS.md:89-94` |
| The trust-most / discount / never evaluation matrix | `DECISIONS.md:100-106` |
| The three-way train/validate/test split discipline; "macro average only" | `DECISIONS.md:110` |
| Per-tier fertility targets: 1.85 / 2.20 / 2.60, English 1.25, code ≥3.6 chars-per-token, JSON ≤25 tokens per tool call | `DECISIONS.md:127-135` |
| The 12 script blocks and their slot allocation, 96,000 Latin → 256 byte-fallback | `DECISIONS.md:143-153` |
| Sum 204,256 → **V = 208,896 = 1,632 × 128**, tensor-parallel aligned | `DECISIONS.md:155` |
| The 131,072 / 208,896 / 262,144 FLOP + embedding comparison | `DECISIONS.md:161-165` |
| 869B tokens saved → 144,800 H100-hours → ₹2.5 crore ≈ 5× return | `DECISIONS.md:169-176` |
| "262K buys ~53K more slots with no script left to serve" | `DECISIONS.md:178` |
| The five gates' names (Rights / Reality / Isolation / Economy / Evidence) and rules R1/R2/R3 | `FRAMEWORK.md:12-19` |
| The Indic alignment stack, 5 steps | `ATLAS.md:529-534` |
| The 9-box pipeline including the **safety & compliance gate** (PII, CSAM hashing, toxicity, licence-tag propagation) | `ATLAS.md:855-920` |
| The milestone verdicts ("Under-ambitious for 300B params") | shipped in `data.json`, read by no page |

"anneal", "curriculum", "phase", "SFT", "DPO", "RLVR", "preference", "alignment", "red-team",
"131,072", "262,144" — **zero occurrences in either page's markup.**

The site spent its length on the *epistemics* of its numbers and never printed the numbers.

---

## 4. Statements that are false as shipped

Several were introduced by me when I anchored the vocabulary sweep on the fertility measurement and
did not propagate the consequence:

| Where | Says | Actually |
|---|---|---|
| report §3 | "section 2's curve stays unanchored" | §2's own pill reads `anchored: 208,000` |
| report §3 | "three ungated tokenizers" | the chart directly beneath shows five |
| report §3 | corpus "translated from English" | IN22-Gen is `native-sourced` |
| report §3 | "Three languages are absent from FLORES entirely" | the corpus is IN22-Gen; zero missing |
| report §3 caption | renders "0 of 23 languages have no measurement" | a caveat asserting nothing is missing |
| report §11 | "the fertility figures are still unrun" | 230 measured values exist |
| `web/index.html:208` | "40 measured, 31 estimates, 121 nobody has measured — including the size of the Indian-language web" | all 121 unknowns are `size_tokens`; **none** is the Indic web; the page shows four figures |

Also: §3 hardcodes `'8.0×'`, `'13.0×'`, `'12%'`, `'73%'`, `'78%'` as plain strings — bypassing
`renderNumber`, on a site whose stated premise is that no bare number reaches the DOM and whose
`num.js` *throws* on one.

**A framework that enforces five invariants in CI shipped seven false sentences.** The invariants
guard the data; nothing guarded the prose.

---

## 5. Interactions that defeat themselves

- **The landing page prints the answer to the report's only quiz.** A stat card reads "22 scheduled
  languages, **10 of them web-viable**" (`web/index.html:178`); §4's predict-before-reveal asks how
  many of the 22 survive a hard filter. The answer is 10.
- **§4's guess marker is semantically wrong.** The reader supplies a *count*; the mark is placed at
  `k === guess - 1`, so typing 14 highlights *Kashmiri*.
- **Using §4's input suppresses the best fact on the page.** The note branch order puts the guess
  message before the Bodo branch, so a reader who uses the interaction never sees *"Bodo came out the
  other side as 77 words in a single document. Not 77 million — seventy-seven."*
- **The atlas TOC lists four categories eight times** — `Datasets → Plan → Evidence → Datasets →
  Evidence → Plan → Evidence → Honesty` — because sections are emitted out of group order. The one
  navigational aid on a 13-section page advertises that the taxonomy is unreliable.

---

## 6. Two of the five gates carry no information

The grade is computed from five gate verdicts. Across 145 datasets:

| Gate | Distribution |
|---|---|
| provenance | PASS 81 · CONDITIONAL 48 · FAIL 15 · UNKNOWN 1 |
| composition | **UNKNOWN 139** · CONDITIONAL 6 |
| contamination | **UNKNOWN 144** · FAIL 1 |
| yield | UNKNOWN 130 · CONDITIONAL 15 |
| evidence | **PASS 145** |

`evidence` passes everything and `contamination` is unknown for everything, so between them they
contribute a constant. In practice the grade is `provenance`, lightly adjusted. That is defensible —
UNKNOWN scoring zero is deliberate — but the site presents "five checks" as if five things were
measured, and 116 of 145 datasets grade C mostly because **nobody checked**, not because anything was
found wanting. The page never says so.

---

## 7. Over-build

I optimised against a spec instead of a reader. The explainer spec (local, gitignored) supplies six topologies,
eight interaction families, a no-two-adjacent-families rule, a subtraction hierarchy and a visual
register. I implemented all of it. The result:

- **45 scroll states, ~2,070vh ≈ 21 screens on the report alone**; the atlas adds 40+.
- The two longest sections carry the least: §6 (7 states, 3.2 screens) counts a tally to 17, and §10
  (7 states, 3.2 screens) performs one multiplication.
- Reader-facing project taxonomy: `EXPLAINER` / `MODELLED` badges with no key anywhere, an unlabelled
  `pill`, and captions written to a design reviewer — *"This is a chart, not an explainer"*, *"there
  are seven, not the nine an earlier draft of this section claimed"*, a run ID, and a `.py` filename.
- Three design generations ship in the CSS. Never constructed: `.stage/.stages`, `.dials`,
  `.presets`, `.callout`, `.tiles`, `.subpanel`, `.footnote`, `.legend`, `.frame`, `button.seg`,
  `.chainout`, `.bars/.barrow`, `.widget`. Dead JS: `tileOf`, `barRow`, `widget`, `animateOnce`,
  `countUp`, `prefersReducedMotion`. An empty `chain-slot` div renders nothing where the design doc
  promised the hero.
- **§1 is hand-rolled** — `report/index.html:372-562` duplicates `explainer.js` element for element.
  Two implementations of the same skeleton ship on the same page, and the one that predates the
  extraction is the reference implementation.

---

## 8. Computed and invisible

Work that runs and reaches nobody:

- `orphans.py` — answers "which mix tier can no benchmark detect, and what does that cost". Fully
  written, fully tested, **imported by no exporter**.
- `records.fertility` — the full 5-tokenizer × 22-language × 2-register matrix, **167 KB shipped to
  every visitor, zero references**.
- `web/shingles.json`, `data.grades`, `data.registry_root`, the milestone `verdict` strings — all
  exported, none read.

---

## 9. What was genuinely right, and is being kept

Not everything needs rebuilding, and pretending otherwise would be its own dishonesty.

- **The contamination gate.** Reader-supplied input, logic ported faithfully from `shingles.py` and
  verified against it, computed in-browser, with an honest boundary condition (attack 5 succeeds) and
  a guard that refuses a verdict rather than inventing one. The most credible artefact on the site.
- **The single-canvas catalogue.** 145 marks that never disappear, five re-encodings, every mark
  opening a five-gate card with reasoning, a roving-tabindex keyboard grid, and a final state showing
  the one dataset that *doesn't exist*.
- **The provenance discipline.** `renderNumber` throwing on a bare number is real, and mostly held.
- **The Bodo "seventy-seven words" line**, when reachable.
- **The confidence ledger and corrections log.** Few technical write-ups say "the design leans
  hardest on the claims it can least support" and then show them.
- **The accessibility work** — measured contrast, 24px hit areas around 14px marks, focus-driven
  state so scrollytelling works without a pointer, reduced-motion end states. Lighthouse 100 on all
  three pages.

The rebuild reuses every one of these. Nothing is deleted.

---

## 10. The lesson, stated once

Every failure above has the same shape: **I optimised against an artefact I controlled instead of the
question I was asked.** First the brief I wrote, then the design spec I wrote, then the explainer
spec I wrote. Each was internally consistent and each moved further from a reader who wants to know
what to put in a corpus.

The corrective is in the rebuild's structure: the page's chapters *are* the questions, one each, and a
chapter that answers no question does not exist.
