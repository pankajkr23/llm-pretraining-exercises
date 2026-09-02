# DECISIONS
### The four answers, with arithmetic

This is the substance of the submission. `FRAMEWORK.md` is the method; this is what the method produced. Every figure here is either **measured** (task 2.2b) or **estimated** — marked accordingly, and the site renders that distinction in the underline of every number.

**Target:** 40B, Gemma-4-class, primary capabilities coding · agentic · Indic languages · conversation · India-first worldview.
**Comparator:** Gemma 4 31B Dense — 30.7B params, 256K context, **262,144-token BPE vocab**, 140+ languages, Apache 2.0, released 2 April 2026.
**Confirmed:** `d_model = 6,144`.

---

## Q1 — What the data looks like

### 1.1 Pre-training: 15T seen tokens from a 9.65T unique pool

Framing in **seen** rather than **unique** tokens is the central move. It lets a scarce tier punch above its collection weight.

| Tier | Unique (B) | Epochs | Seen (B) | Share | Why |
|---|---|---|---|---|---|
| English web HQ | 3,000 | 1 | 3,000 | 20.0% | Reasoning substrate |
| English educational | 1,000 | 2 | 2,000 | 13.3% | FineWeb-Edu is the dominant MMLU/ARC lever |
| **Code** | 1,100 | 2 | 2,200 | **14.7%** | Primary objective; also buys 2× repetition headroom |
| Math / STEM | 500 | 2 | 1,000 | 6.7% | Reasoning chains |
| **Indic natural** | 300 | **4** | 1,200 | **8.0%** | The scarce gold |
| Indic synthetic / translated | 900 | 1.5 | 1,350 | 9.0% | Manufactured; capped at 50% of the Indic tier |
| Indic parallel / translit | 250 | 2 | 500 | 3.3% | BPCC + Aksharantar |
| **India-context English** | 250 | 3 | 750 | **5.0%** | Worldview ≠ language — see 1.2 |
| Multilingual non-Indic | 1,200 | 1 | 1,200 | 8.0% | Cross-lingual transfer |
| **Agentic + world-model traces** | 400 | 2 | 800 | **5.3%** | CoT state-transitions in *pre-training* |
| Clean-provenance | 500 | 1 | 500 | 3.3% | Legal insurance tier |
| Anneal (highest quality only) | 250 | 2 | 500 | 3.3% | Where benchmarks are bought |
| **TOTAL** | **9,650** | avg 1.55 | **15,000** | 100% | |

**India total 25.3%** (3,800B seen) · **code + agentic 20.0%** (3,000B seen).

Natural Indic reaches **8.0% of seen tokens from a 2.0% unique pool** purely through epoch scheduling. Muennighoff et al. measured 4 epochs on 44B unique finishing only **0.5% worse** on validation loss than 1 epoch on 178B unique. Guardrails: decay half-life R\*_D ≈ 15, so 4 epochs sits well inside the useful range; and repetition operates on **whole tiers only** — up-sampling 0.1% of a corpus 100× significantly degrades performance.

⚠️ **Estimated, not measured:** R\*_D ≈ 15 comes from English web data at ≤9B params. Nobody has measured it for Indic, translated or synthetic text. Synthetic plausibly has a lower ceiling. A one-week ablation at 1.5B would settle it.

### 1.2 "India-first" is not achieved by Indic-language data

A model with flawless Hindi that assumes American default law is not India-first. Worldview comes from India-**context** content, much of it in English: PIB multilingual releases · Lok Sabha and Rajya Sabha debates · eCourts judgments (25 High Courts, 1950–2025) · Indian science, business and history writing · IKS-Instruct (24,795 pairs, 41 pedagogical techniques, CBSE classes 6–12) · the English side of FineTranslations, which deliberately carries non-Western cultural knowledge.

Named as its own 5% tier with its own coverage metric.

### 1.3 Post-training

**SFT ≈ 3M examples** — agentic 600K · code 600K · Indic conversation 500K (IndicTalk, UPDESH, IndicAlign-Instruct) · reasoning 450K · general instruct 400K · India-context factuality 250K · safety 200K.

**DPO ≈ 200K pairs** — India factuality & plural perspectives 50K · safety incl. transliterated slurs 40K · code quality 35K · agentic tool-selection and termination 35K · stop discipline 25K · Indic register, native-speaker judged 15K.

**RLVR ≈ 60K problems × 8–16 rollouts** — math 20K (Python-verified) · code 20K (unit tests) · **Indic issue→patch 8K** · agentic multi-turn 7K · verifiable instruction-following 5K.

### 1.4 ★ The differentiator

Translate **only the issue text** of SWE-smith instances into Indic languages. Tests and patches stay byte-identical, so the verifier still works — **verifiable RL reward in Hindi, Tamil and Bengali at zero annotation cost.** No such dataset exists (catalogue entry `COD-03`). A Bengali-language GitHub issue resolving into a passing patch is the demo that proves this is not a fine-tune with a flag on it.

---

## Q2 — How the data is cleaned

### 2.1 Universal pipeline

NFC + homoglyph normalization → GlotLID script-aware language ID → India-specific PII (Aadhaar, PAN, bank, voter, phone) → global MinHash + SemDeDup → **13-gram benchmark decontamination** → provenance record (URL, licence, hash, retrieval date).

Adopt rather than rebuild: **Setu** (MIT, Spark, Indic-specific), **datatrove**, **NeMo Curator**, **GlotLID**.

### 2.2 Objective-specific rules — the counterintuitive ones

| Objective | The rule that isn't obvious |
|---|---|
| **Indic** | Quality classifiers are the adversary. Per-language calibrated thresholds, never a global one. Tier-3 languages bypass filtering entirely into the Always-ON lane |
| **Code** | Do not length-filter like prose. Licence allowlist + **live opt-out list**. Function-level near-dedup |
| **Agentic** | **Filter suboptimal trajectories hard.** Errors amplify over horizons — a mediocre 80-turn trace is worse than none. This inverts "more data is better" |
| **Math** | Preserve LaTeX; do **not** normalize digits |
| **India-context** | Temporal validity — law changes. Date-stamp everything or you train stale statute as current fact |
| **OCR** | Build on **Qwen3-VL-8B**, not olmOCR. Validate on real scans only. Report median and catastrophic-rate, never mean |

### 2.3 ★ The Always-ON lane — 8% of every batch

Reserved for Tier-2/Tier-3 Indic scripts, agentic traces and benchmark training splits, **invisible to every selector and quality classifier.**

The justification is measured, not theoretical: LightningLM's data-selection proxy sat at **cosine 0.876 with the English web band** and systematically under-valued Indic content. Sangraha's own Stage-3 filtering table shows **Bodo collapsing to 77 words / 1 document.**

Your FineWeb-Edu-style classifiers carry the same bias. The fix is not a better classifier — it is refusing to let the classifier vote.

### 2.4 Curriculum phases

| Phase | % of run | Emphasis |
|---|---|---|
| Foundations | 0–40% | Broad web, heavy multilingual, **Indic scripts front-loaded** — rare-script embeddings form early or not at all |
| Knowledge | 40–75% | Educational, Wikipedia, academic, code |
| Reasoning | 75–92% | Math, long chains, synthetic QA, curriculum-graded Indic |
| Anneal | 92–100% | Highest quality only, LR → 0 |

---

## Q3 — How the model is tested

| Objective | Trust most | Report but discount | Never |
|---|---|---|---|
| **Indic** | MILU (1,500+ Indian competitive exams), IN-22 (source-original), INCLUDE, IndicIFEval | IndicMMLU-Pro (translated via IndicTrans2) | Pooled scores — macro average only |
| **Coding** | HumanEval+, MBPP+, LiveCodeBench | HumanEval (saturated) | — |
| **Agentic** | SWE-bench Verified **+ named harness**, Terminal-Bench 2.0, BFCL, τ-bench | GAIA | Any number without harness, scaffold version, turn limit and context window |
| **India-first** | **BhashaBench — fully held out**, Sanskriti, Pariksha | — | Preferred wording as evidence of correctness |
| **Safety** | 22-language red team, XIH-Bench cross-language injection | HarmBench, XSTest | English-only safety testing |

### Four disciplines

1. **Contamination is a build gate, not a check.** 13-gram + MinHash against a registry that lives in CI and never enters the deployed bundle. Precedent: LightningLM dropped a full 31.3B-token pool rather than down-weight it.
2. **One benchmark is never looked at during development.** BhashaBench — domain-specific (Agriculture, Legal, Finance, Ayurveda) and unlikely to appear in web text.
3. **A private 3,000-item × 22-language set**, commissioned early. Every public Indic benchmark will be contaminated within 18 months.
4. **★ Every tier must have an instrument.** If no benchmark would detect a tier's removal, cut the tier or add an instrument. You cannot defend a token budget you cannot measure.

---

## Q4 — Fertility targets and vocabulary size

### 4.1 Scope: tier the languages, don't cut them

**12 script blocks** — 9 Brahmic (Devanagari, Bengali-Assamese, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam) **+ Perso-Arabic + Ol Chiki + Meitei Mayek.**

Dropping Urdu to stay Brahmic-only is the most conspicuous omission an India-first model can make. Perso-Arabic costs ~8K slots and covers Urdu, Sindhi and Kashmiri.

### 4.2 Targets

| Tier | Languages | Target |
|---|---|---|
| A | hi, bn, ta, te, mr, ml, kn, gu, pa, or, **ur** | ≤ 1.85 tok/word |
| B | as, sa, ne, kok, mai, sd | ≤ 2.20 |
| C | ks, brx, doi, mni, sat | ≤ 2.60 — coverage guaranteed, fertility best-effort |
| English | | ≤ 1.25 |
| Code | | ≥ 3.6 chars/token |
| Math | | digits **atomic**; frequent LaTeX commands single-token |
| Agentic / JSON | | ≤ 25 structural tokens per tool call |

**★ PARITY RATIO = max(Tier A) ÷ English ≤ 1.5.** One number that states the India-first claim quantitatively. Under cl100k_base it sits near 8.0.

The agentic line matters more than it looks: 100 turns × 30 wasted structural tokens = 3,000 tokens burned on JSON punctuation.

### 4.3 Vocabulary, derived bottom-up

| Block | Slots | Block | Slots |
|---|---|---|---|
| Latin / code base | 96,000 | Odia | 6,000 |
| Devanagari (8 languages) | 18,000 | **Perso-Arabic** | 8,000 |
| Bengali-Assamese | 12,000 | **Ol Chiki** | 1,500 |
| Tamil | 12,000 | **Meitei Mayek** | 1,500 |
| Malayalam | 12,000 | Math / LaTeX | 3,000 |
| Telugu | 10,000 | JSON / agentic | 1,000 |
| Kannada | 9,000 | Special tokens | 1,000 |
| Gujarati | 7,000 | Byte fallback | 256 |
| Gurmukhi | 6,000 | | |

**Sum ≈ 204,256 → V = 208,896** (= 1,632 × 128, aligned for tensor parallelism).

### 4.4 Why not 131K, and why not Gemma's 262K

Output projection costs `2 · d_model · V` FLOPs per token. At `d_model = 6,144`, total forward ≈ `2N` = 80 GFLOP/token.

| V | Output projection | % of forward | Embedding (tied) | % of 40B |
|---|---|---|---|---|
| 131,072 | 1.61 GFLOP | 2.01% | 805M | 2.0% |
| **208,896** | **2.57 GFLOP** | **3.21%** | **1.28B** | **3.2%** |
| 262,144 (Gemma 4) | 3.22 GFLOP | 4.03% | 1.61B | 4.0% |

**The trade:** 131K → 208K costs **1.2% of forward compute.** It is expected to improve Indic fertility from ~2.4 to ~1.85 (−22.9%) across a 25.3% slice of the mix:

```
token saving   = 0.253 × 0.229            = 5.79% of 15T   = 869B tokens
FLOPs saved    = 6 × 40e9 × 8.69e11       = 2.09e23
GPU-hours      = 2.09e23 / 4.0e14         ≈ 144,800 H100-hours
cost saved     ≈ $290,000                 ≈ ₹2.5 crore
```

One epoch, before any inference-side saving. Roughly a **5× return** on the 1.2% compute cost.

**262K buys ~53K more slots with no script left to serve.** Gemma needs it for 140+ languages; a 12-script tokenizer does not.

⚠️ **The 2.4 → 1.85 figures are estimated.** Task 2.2b measures Gemma 4's actual Indic fertility across all 22 scheduled languages on IN22-Gen (source-original), which replaces both numbers with observations and re-derives the optimum. **Until that lands, §4.4 renders as `estimated` on the site.**

### 4.5 Embedding cost is not the binding constraint

Kronecker factorization replaced a ~537M-parameter table with a 33.55M projection at 131K vocab. Applied here, the **input** embedding largely stops mattering — which is why the vocab decision rests on **softmax and output-projection compute**, not embedding parameters. State this explicitly; it is the part most analyses get backwards.

---

## The fork we are not hiding

Gemma 4 31B Dense is **30.7B parameters under Apache 2.0.** "Train a 40B" therefore has three legitimate readings:

| Path | Cost | Inherits |
|---|---|---|
| From scratch | 1.0× baseline | Nothing |
| **Vocabulary-expand + continue-pretrain** from Gemma-4-31B | A fraction | Gemma-4-class coding and agentic capability on day one |
| Upcycle to MoE (~40B total / ~5B active) | Middle | Competes with Gemma 4 26B-A4B, not the 31B Dense |

### What the fork now costs, measured

The "Inherits" column understates the second path. **Continue-pretraining inherits the tokenizer**, and
a tokenizer cannot be swapped without discarding the embedding table — which is most of what you were
reusing. So that path locks in Gemma 4's vocabulary for the life of the model.

Task 2.2b measured what that costs, on IN22-Gen, all 22 scheduled languages, run
`in22gen-20260805T061749Z`:

| Tokenizer | Mean Indic tax | Worst language |
|---|---|---|
| Gemma 4 31B | **×2.49** | ×8.84 |
| Sarvam-105B | ×1.81 | ×2.90 |
| XLM-R (2019, general-purpose) | ×1.66 | ×2.42 |

Gemma 4 is the requirements' named target and the worst of the three on Indic. Against Sarvam that is
roughly **37% more tokens for the same Indic text**, paid on every step of the entire run and on every
token served afterwards. Vocabulary expansion mitigates it — new tokens can be added — but the merges
already learned stay as they are, so expansion narrows the gap rather than closing it.

This does not settle the fork. It prices one side of it, which the table above did not.

**Resolution:** a head-to-head at ~2B scale on identical data, judged on Indic and code held-out loss.
State the fork, state the criterion, name the experiment. Do not pretend the choice is obvious — and
now, do not pretend the inherited tokenizer is free.

---

## What we don't know

| Claim | Status |
|---|---|
| Deduplicated natural Indic pool = 250–500B tokens | **Never measured by anyone.** An inference from Sangraha, FineWeb-2, IndicCorp and HPLT. It drives the entire corpus architecture |
| R\*_D for Indic / translated / synthetic text | Measured only on English web at ≤9B params |
| Indic fertility of Gemma 4's tokenizer | **Being measured — task 2.2b** |
| Whether weights are a derivative of CC BY-SA training data | Unresolved anywhere in the world |

Closing line for the report:

> *The estimate that deduplicated natural Indic text totals 250–500B tokens has never been measured by anyone. It drives our entire mix architecture. We are measuring it, and we will publish the number.*
