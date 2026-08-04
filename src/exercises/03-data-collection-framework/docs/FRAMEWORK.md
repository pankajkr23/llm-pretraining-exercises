# FRAMEWORK
### Five questions, three mix rules, eight intuitions

The method. `DECISIONS.md` is what the method produced. This document supplies the copy for the eight intuition widgets on `/reasoning` and the framing for `/report`.

---

## The shape

Every dataset must answer five questions. Four are **gates** (before training); the fifth is a **loop** (after). Then three rules govern how admitted datasets become a *mix*.

```
   ADMISSION                              CONSTRUCTION
   Q1 May I?          → Rights            R1 Effective = unique × epochs
   Q2 Is it real?     → Reality           R2 Always-ON lane, 8% of every batch
   Q3 Is it clean?    → Isolation         R3 Curriculum phases
   Q4 What's the cost?→ Economy
   Q5 Did it help?    → Evidence  (loop)
```

Q1 and Q3 are **blocking**. Q2, Q4, Q5 are advisory — they downgrade a grade, they don't forbid ingestion.

---

## Q1 · MAY I? — the Rights ledger

> **Intuition: every token needs a passport, and the border check happens at boarding, not after landing.**

You cannot un-copy data. The copy *is* the act. Checking robots.txt after ingestion is theatre — fetch time is the only enforceable moment, and the only moment you can record what the signal said.

**The counterintuitive finding this gate produces:** India's own school curriculum — the best curriculum-graded, pedagogically-ordered, 36-language corpus in existence — is **CC BY-NC-ND**. Unusable commercially. Everyone assumes the government data is the easy part; it is the hardest part.

**The distinction most people get wrong:** the Delhi High Court's July 2026 fair-dealing ruling in ANI v. OpenAI improved the *copyright* position. It does **not** touch NonCommercial licences, because **an NC licence is a contract, not a copyright default.** A favourable ruling about statutory exceptions does not dissolve terms you agreed to.

*Fails if:* licence forbids commercial training · provenance unknown · opt-out signal present at fetch · personal data without lawful basis.

---

## Q2 · IS IT REAL? — the Reality ledger

> **Intuition: photocopies. Photocopy a page five times and you have five sheets and one page of information.**

This is deduplication, and it is the gap between a 23T-token plan and a 15T corpus. DCLM ships 3.8T raw for ~1.0T unique. Nemotron's authors found roughly 80% fuzzy duplication in corpora that had already been "deduplicated" — sharded approximate dedup misses cross-shard twins.

> **Second intuition, for synthetic data: a student rewriting a textbook in their own words.**

Genuinely useful — often *more* learnable than the original, because it is cleaner. But it cannot contain knowledge the textbook lacked. Which yields the sharpest rule in the framework:

**Translation and synthesis buy reasoning transfer and fluency. They do not buy culture.** Native data wins on idiom and cultural grounding throughout training; translated data closes the gap on commonsense reasoning and eventually overtakes. Spend scarce natural Indic tokens on the cultural surface; let synthetic carry the reasoning load.

*Fails if:* duplicate of an admitted document · synthetic counted as natural · a generator model's licence contaminates the chain.

---

## Q3 · IS IT CLEAN? — the Isolation ledger

> **Intuition: the answer key lives in a different building, with a different key.**

Intent is not a mechanism. If the exam and the study material share a drawer, contamination is a matter of time, not of ethics.

> **Second intuition: a ruler you quietly shaved down to fit the thing you were measuring.**

A contaminated benchmark does not produce a wrong number. It produces a *confident* wrong number, which is worse.

**Precedent:** the LightningLM 120B run dropped an entire 31.3B-token pool for contamination rather than down-weighting it, and kept its golden proxy strictly held out, never trained on. Deletion, not discounting.

*Fails if:* 13-gram or MinHash overlap with any registered evaluation item. **This gate blocks export. It does not warn.**

---

## Q4 · WHAT DOES IT COST? — the Economy ledger

> **Intuition: tokens are the currency. Fertility is the exchange rate.**

If English costs 1.3 tokens per word and Malayalam costs 13, you are paying a **10× tariff to express the same idea.** Measured: an 8.0× average tokenization tax for Indian languages under cl100k_base, 13.0× for Malayalam.

> **Second intuition: the context window is a suitcase. Bulky clothes mean fewer outfits.**

Indic effective context collapses to as little as **12%** of what an English user gets from the same nominal window. For a 256K-context agentic model doing 100-turn tool loops, that is not an inconvenience — you run out of suitcase before the task finishes.

> **Third intuition, and the important one: this is a tariff, not a law of nature.**

The mechanism is **failed byte-pair merges** leaving text shattered into single-byte fragments; merge failure correlates with the tax at **r = 0.89**. It is a design failure, and it is removable — o200k and XLM-R eliminate 73% of it. Indic scripts are not inherently expensive. Somebody never paid attention.

*Fails if:* fertility exceeds the per-tier threshold, or parity ratio > 1.5.

---

## Q5 · DID IT HELP? — the Evidence loop

> **Intuition: don't grade the exam you wrote yourself, in the language you translated it into.**

Translated benchmarks measure *translated-English competence*, not Indic competence. Report them; weight them below natively-sourced instruments — MILU (1,500+ Indian competitive exams), IN-22 (source-original), BhashaBench, INCLUDE.

> **Second intuition, for agentic evaluation: a lap time measures driver *and car*.**

SWE-bench leaderboard entries are literally written `harness + model`. A number without the harness, scaffold version, turn limit and context window is not a result.

*Fails if:* a tier is in the mix and **no benchmark can detect its contribution.** If you cannot measure a tier, you cannot defend its token budget.

---

## R1 · Effective tokens = unique pool × epoch schedule

> **Intuition: a library card, not a bookstore receipt.**

You do not own 15T tokens. You have access to a pool you may re-read. Re-reading a good book four times beats skimming four mediocre ones — measured: 4 epochs on 44B unique finished only **0.5% worse** on validation loss than 1 epoch on 178B unique.

This is the correction that changes everything for a low-resource language. Natural Indic at 0.3T unique × 4 epochs ≈ **1.2T effective**, lifting its share from ~2% to ~8% **without collecting a single new token.**

> **Guardrail intuition: re-reading one paragraph a hundred times is not re-reading the book.**

Up-sampling 0.1% of a corpus 100× significantly degrades performance. Repetition operates on **whole tiers**, never slivers. Decay half-life R\*_D ≈ 15; hard ceiling at ~16× the unique pool.

---

## R2 · The Always-ON lane — 8% of every batch

> **Intuition: reserved seating. If the general queue decides, the loudest group fills the room.**

Your quality classifiers were taught what "good" looks like using English. They will score Indic as low-quality — not from malice, but because they cannot see value they were never shown. Measured: LightningLM's selection proxy sat at **cosine 0.876 with the English web band** and systematically rejected Indic content. Sangraha's own filtering table shows **Bodo collapsing to 77 words / 1 document.**

Carve out ~8% of every batch for Tier-2/Tier-3 Indic scripts, agentic traces and benchmark training splits — **invisible to every selector and classifier.** This is the mechanism, not the aspiration, that keeps Bodo and Santali alive through training.

---

## R3 · Curriculum phases

> **Intuition: a child does not start with Kant, and does not still read picture books at twenty-five.**

| Phase | % of run | Emphasis |
|---|---|---|
| Foundations | 0–40% | Broad web, heavy multilingual, **Indic scripts front-loaded** — embeddings for rare scripts form early or not at all |
| Knowledge | 40–75% | Educational, Wikipedia, academic, code |
| Reasoning | 75–92% | Math, long chains, synthetic QA, curriculum-graded Indic |
| Anneal | 92–100% | Highest quality only, LR → 0. **This is where benchmark performance is bought** |

---

## The framework's own claim

> *We did not pick a data mix. We built the procedure that picks, made it executable, measured the one number the field asserts without measuring, and ran it once — on a 40B India-first model.*

A dataset list is a claim about one answer. A framework is a claim about a method, and it is testable on cases a reviewer invents.

---

## Naming note

Public artifacts use neutral naming. **Do not name the repository, package or deployment after a national term** — "Bharat Data Sagar" is BharatGen's existing corpus product, and a similarly-named public repository on the same subject matter creates a confusion and endorsement problem. Ship a `NOTICE` file disclaiming affiliation with BharatGen, IndiaAI, MeitY, AI4Bharat, Sarvam AI and the Government of India, and use no government emblems — the Emblems and Names (Prevention of Improper Use) Act, 1950 is stricter than most people expect.
