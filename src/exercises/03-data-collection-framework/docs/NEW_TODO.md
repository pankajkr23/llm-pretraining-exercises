# NEW TODO — the one-page rebuild

Execution order for the rebuild described in [`DESIGN_CRITIQUE.md`](DESIGN_CRITIQUE.md). Each phase
is independently shippable and leaves the site working.

**Ground rules for every phase**

- **Delete nothing.** No record, no module, no dataset. Repurpose, or move to the appendix.
- **Every chapter answers exactly one question.** A chapter that answers none does not exist.
- **Three layers, always the same shape** — see below.
- **Define before use.** No house term (tier, lane, gate, fingerprint, fertility, epoch) appears
  before one plain sentence explains it.
- **≤4 states per chapter**, `.step` 30vh, whole narrative ≤12 screens.

---

## The three layers

Every chapter, without exception:

| Layer | For | Content | Default |
|---|---|---|---|
| 1 · Headline | a 13-year-old | one plain sentence + one big number | visible |
| 2 · Explainer | a curious reader | the interaction that proves the claim | visible |
| 3 · `▸ The arithmetic` | an engineer, a PhD, a director | derivation, sources, caveats, provenance | **closed** |

Layer 1 must be true and complete on its own — a reader who stops there has not been misled. Layer 3
carries everything the old pages put in the main flow.

---

## The eleven chapters

| # | Chapter | Question | Interaction | Source |
|---|---|---|---|---|
| 1 | The target | — | none | `DECISIONS.md:4-8` |
| 2 | How much text, and can we get it | Q1 | epoch ladder, last state 300B | existing §10 |
| 3 | What goes in it | Q1 | zero-sum shares, **15T default** | existing §5 |
| 4 | Where we get it — pre-training | Q1 | the shortlist | existing §5b + `#shortlist` |
| 5 | What we may legally use | Q5 | licence funnel | repurpose `#data` licence encoding |
| 6 | Teaching it to behave | Q1 | shortlist shape, post-training data | **new records** |
| 7 | How we clean it | Q2 | stage walk + contamination gate | existing §6 + §1 |
| 8 | How we tokenise it | Q4 | the widest gap | existing §2 + §3, merged |
| 9 | How we'd know it worked | Q3 | coverage collapse | existing §7 + `orphans.py` |
| 10 | What it costs, and whether to build at all | Q6 + Q8 | the fork, priced | **new records** + §8 |
| 11 | What we'd do first | Q7 | 12 actions, 2 gates, 4 letters | existing §9, relabelled |
| A | Appendix | — | tables + the two canvases | everything else |

---

## Phase 0 — stop shipping falsehoods

Small, lands first, independent of the rebuild.

- [ ] **0.1** §3: remove "section 2's curve stays unanchored", "three ungated tokenizers", "translated
  from English", "Three languages are absent from FLORES"; fix the caption that renders
  "0 of 23 languages have no measurement".
- [ ] **0.2** §11: remove "the fertility figures are still unrun".
- [ ] **0.3** §3: route `8.0×`, `13.0×`, `12%`, `73%`, `78%` through `renderNumber` with provenance,
  or move them to layer 3 as cited figures.
- [ ] **0.4** Landing honesty line: state what is actually true (121 unknowns are all `size_tokens`).
- [ ] **0.5** Landing stat card: stop printing "10 of them web-viable" — it is §4's answer.
- [ ] **0.6** §4: fix the guess marker (a count must not highlight a language); reorder the note
  branches so the Bodo fact survives the interaction.
- [ ] **0.7** Atlas TOC: emit each group once. Remove the stray `void counts;`.
- [ ] **Checkpoint:** re-run the reader audit; no statement contradicts another.

## Phase 1 — extract what is already decided

New `models.Value`-typed record arrays so chapters render data, not prose.

- [ ] **1.1** `records/posttraining.json` ← `DECISIONS.md:46-56` + `ATLAS.md:483-535`. SFT 3M · DPO
  200K · RLVR 60K with per-category splits; the Indic alignment stack; the SWE-smith
  issue-translation differentiator. → ch. 6
- [ ] **1.2** `records/cleaning_rules.json` ← `DECISIONS.md:62-77` + `ATLAS.md:855-920`. The universal
  pipeline, the six objective-specific rules, and the **safety & compliance gate** that `tools.json`
  has no rows for. → ch. 7
- [ ] **1.3** `records/eval_policy.json` ← `DECISIONS.md:100-113` + `ATLAS.md:576-604`. The
  trust-most/discount/never matrix, three-way split discipline, "macro average only". → ch. 9
- [ ] **1.4** `records/vocab_blocks.json` ← `DECISIONS.md:143-165`. 12 blocks, the sum, the FLOP
  table. → ch. 8
- [ ] **1.5** `records/fertility_targets.json` ← `DECISIONS.md:127-135`. Per-tier targets incl. code
  and JSON. → ch. 8
- [ ] **1.6** `records/cost.json` ← `DECISIONS.md:169-176` + `market.json` + `acquisition.json`. → ch. 10
- [ ] **1.7** Extend `sourcing.py` to three lifecycle stages using the existing `stage` field.
  **Nothing unmapped may be dropped silently** — report a count and a reason.
- [ ] **1.8** Wire `orphans.py` into the export. It answers ch. 9's argument and has never shipped.
- [ ] **1.9** Reconcile or state two divergences: site mix 8 tiers / 23.3% Indic vs `DECISIONS.md`
  12 tiers / 25.3%; `recommended_vocab` 208,000 (sweep grid) vs 208,896 (bottom-up, 128-aligned).
- [ ] **1.10** `catalog.EXPECTED_COUNTS` updated for the new arrays; tests for each extraction.
- [ ] **Checkpoint:** `uv run pytest` green; index still <100KB.

## Phase 2 — build the page

- [ ] **2.1** `_shared/explainer.js`: add the layer-3 `<details>` slot; `.step` 46vh → 30vh.
- [ ] **2.2** Rebuild §1 (the gate) through `buildExplainer` — it currently duplicates the skeleton.
- [ ] **2.3** Chapters 1–4 in `web/index.html`, mix defaulting to **15T**.
- [ ] **2.4** Chapters 5–8.
- [ ] **2.5** Chapters 9–11.
- [ ] **2.6** Appendix: the two canvases + every reference table, tables only.
- [ ] **2.7** `web/report/` and `web/reasoning/` become redirect stubs; every retired anchor resolves.
- [ ] **Checkpoint:** each of the eight questions answered in exactly one chapter; ch. 4 and 6 name
  datasets with token counts; ≤12 screens.

## Phase 3 — remove the seams

- [ ] **3.1** Delete dead CSS (three generations) and dead JS imports; remove the empty `chain-slot`.
- [ ] **3.2** Remove reader-facing bookkeeping: `EXPLAINER`/`MODELLED` badges, the unlabelled `pill`,
  reviewer-addressed captions, the run ID in a caption, `§5b`, the `.py` filename.
- [ ] **3.3** Archive or wire up: `records.fertility` (167 KB, unread), `web/shingles.json`,
  `data.grades`, `data.registry_root`, milestone `verdict` strings.
- [ ] **3.4** `BRIEF.md`: correct the scope to the full lifecycle. `DESIGN.md`: replace the
  record-type traceability table with the question→chapter map.
- [ ] **Checkpoint:** `ruff` clean, Lighthouse ≥95 perf / 100 a11y on the one page.

## Phase 4 — ship

- [ ] **4.1** `CHANGELOG.md` entry.
- [ ] **4.2** Root `README.md` row updated for the one-page shape.
- [ ] **4.3** Re-run the reader audit end to end.
- [ ] **4.4** PR.

---

## Verification, every phase

```bash
uv run python -m dataframework                 # index <100KB
uv run pytest                                  # 161 today, more after phase 1
uv run ruff check . && uv run ruff format --check .
bash deploy/vercel/build.sh                    # one page + redirect stubs
```

## What must remain true at the end

- The eight questions each have one chapter.
- The mix defaults to **15T** and names real datasets.
- Every chapter has three layers; layer 3 is closed.
- No term is used before it is defined.
- No statement contradicts another.
- The five invariants still pass, and nothing was deleted to make that easier.
