# The India LLM Data Atlas
### A complete corpus, licensing, and benchmark blueprint for a 300B-parameter MoE model expert in 22 Indian languages, English, coding, and agentic coding

**Version:** 1.0 · **Compiled:** 28 July 2026
**Scope:** Pre-training · Mid-training · SFT · Preference/RL alignment · Multimodal · Benchmarks · Licensing · Legal risk
**Target model profile:** ~300B total parameters, Mixture-of-Experts, ~128K context, primary capabilities = conversation, translation, coding, agentic coding, Indic-language expertise

---

## 0. How to read this document

Every dataset entry uses a consistent schema:

| Field | Meaning |
|---|---|
| **Owner** | Legal entity or consortium that controls the artefact |
| **License** | Exact license string where known. `⚠️` marks a license that blocks or complicates commercial pre-training |
| **Size** | Tokens / hours / images / pairs, as published by the owner |
| **Stage** | `PT` = pre-training, `MT` = mid-training/annealing, `SFT` = supervised fine-tuning, `RL` = preference/RLVR, `EVAL` = benchmark only |
| **Access** | How to actually get the bytes |
| **Used by** | Models/companies known to have trained on it |
| **Risk** | Known controversies, contamination, provenance, or legal exposure |

**Colour legend used throughout:**

- ✅ **Green** — permissive, commercially usable, low provenance risk. Use freely.
- 🟡 **Amber** — usable with conditions (attribution, share-alike, gated access, sign-up, or unresolved provenance). Requires a legal review sign-off before it enters the mix.
- 🔴 **Red** — non-commercial licence, litigation-tainted, pirated, or otherwise disqualified. **Do not put in the pre-training mix.** Listed here explicitly so you can build a *blocklist*, not a shopping list.

> **Honesty note.** Where a number is a vendor claim that has not been independently reproduced (notably the 20T-token "Bharat Data Sagar" figure), it is flagged as **[VENDOR CLAIM]**. Where I could not verify a direct download URL, the entry says how to *find* the source rather than inventing a link. Nothing in this document is a substitute for your own counsel's review — particularly Sections 12–14.

---

## 1. Executive summary — the answer up front

### 1.1 The single most important finding

**There is not enough naturally-occurring Indian-language text on Earth to pre-train a 300B model on Indic data alone. Not close.**

The best public estimates of *deduplicated, quality-filtered, naturally-written* Indic web text converge on **roughly 250–500 billion tokens across all 22 scheduled languages combined** — and a large fraction of even that is near-duplicate news wire copy. For calibration:

| Corpus | Indic token count | Note |
|---|---|---|
| Sangraha **verified** (AI4Bharat) | **64B** | Highest-quality tier, human-curated URL seeds |
| Sangraha **unverified** | 24B | Cleaned but unvetted sources |
| Sangraha **synthetic** | 162B | Machine-translated/transliterated — *not* natural text |
| IndicCorp v2 | 20.9B | 24 languages; largely subsumed by Sangraha |
| FineWeb-2 Indic slice | **~40B words** | Across all Indian languages, per the BhashaKritika authors |
| Varta (news) | 9B | News domain only |

Compare that to the **15T tokens** in FineWeb (English) or the **6.6T** in a single NVIDIA Nemotron pre-training release. The Indic natural-text pool is **roughly 0.3–2% the size of the English pool**.

**Consequence:** any credible 10T–20T token corpus for this model will be, by necessity:

```
~55-70%  English + code + math (the reasoning substrate)
~15-25%  Synthetic / translated / transliterated Indic  ← manufactured, not found
~3-6%    Natural Indic web + books + government + speech transcripts  ← the scarce gold
~5-10%   Multilingual non-Indic (helps cross-lingual transfer)
~2-5%    Curriculum / textbook / exam / reasoning-dense
```

This is exactly what every serious Indian lab has concluded independently. **Sarvam-105B was trained on 12T tokens; Sarvam-30B on 16T; BharatGen's Param-2-17B-A2.4B on ~22T tokens** — and BharatGen's own Param-1 disclosure states only "over one-third of the training data representing Indian content," where "Indian content" includes India-related English.

### 1.2 Milestone verdict (answering the exit criterion directly)

You asked for a decision among 5T / 10T / 15T / 20T. Here is the honest engineering answer:

| Milestone | Feasibility from **open + permissively licensed** sources | Verdict |
|---|---|---|
| **5T tokens** | **Trivially achievable today.** FineWeb-Edu (1.3T) + Nemotron-CC-v2 subset + Stack v2 (900B) + Sangraha (251B) + FinePDFs (3T) already overshoots. | Under-ambitious for 300B params |
| **10T tokens** | **Achievable in ~8–12 weeks of pipeline work.** Requires global cross-corpus dedup, which typically removes 40–70%. | ✅ **Safe target** |
| **15T tokens** | **Achievable but requires manufacturing.** You must generate 1.5–3T synthetic Indic tokens yourself (à la BhashaKritika's 540B, scaled up ~4x) plus heavy FinePDFs/FineTranslations use. | ✅ **Recommended target** |
| **20T tokens** | **Achievable only with aggressive multi-epoch reuse + very large synthetic generation + PDF/OCR/ASR pipelines that don't exist off-the-shelf for Indic.** BharatGen claims to be here **[VENDOR CLAIM]** but the corpus is not redistributable. | 🟡 Stretch goal, phase 2 |

**Recommendation: architect for 15T, with a validated 10T "must-have" core and a 5T synthetic/expansion tier that can grow.** For a 300B-total / ~30–40B-active MoE, Chinchilla-optimal is far below this; you are training well past compute-optimal for inference efficiency, which is correct for a production model. 15T unique tokens at ~35B active params is a sane, defensible ratio.

### 1.3 The three things that will actually determine whether this model is good

1. **Tokenizer fertility.** Indic scripts are punished by English-centric BPE at 4–8 tokens per word. Sarvam reports achieving **fertility of 1.4–2.1 across Indian languages**, which they translate into a 200–400% efficiency gain. BharatGen built phonetically-aware tokenisation layers. **Your tokenizer decision is worth more than 2T extra tokens.** Budget a dedicated workstream.
2. **Synthetic data quality control, not quantity.** BhashaKritika (540B synthetic Indic tokens) found that models trained on synthetic Indic converged *faster* than on web Indic because web Indic is so noisy — but only with a modular QC pipeline (script/language ID, n-gram repetition, KenLM perplexity).
3. **Evaluation you can trust.** Multiple 2026 analyses point out that India can now train sovereign models but struggles to *prove* they work, because Indic benchmarks are thin, translated, and contamination-prone. See Section 11.

---

## 2. The 22 scheduled languages: a data-availability triage

The Eighth Schedule of the Indian Constitution lists 22 languages. Their data situations are wildly unequal. Treat these as three distinct engineering problems, not one.

| Tier | Languages | Natural text available | Strategy |
|---|---|---|---|
| **Tier 1 — Web-viable** | Hindi, Bengali, Tamil, Telugu, Marathi, Malayalam, Kannada, Gujarati, Urdu, Punjabi | 1B–10B+ tokens each | Standard web pipeline + dedup + quality classifier |
| **Tier 2 — Thin** | Odia, Assamese, Nepali, Sanskrit, Sindhi, Kashmiri, Konkani, Maithili | 10M–1B tokens | Web + aggressive synthetic + OCR of print + ASR transcripts |
| **Tier 3 — Critical** | Bodo, Dogri, Manipuri (Meitei), Santali | <10M tokens; some near-zero | **Almost entirely synthetic + primary collection.** Sangraha's own cleaning stats show Bodo dropping to **77 words / 1 document** after Stage-3 filtering. This is not a data problem, it is a *fieldwork* problem. |

**Practical implication for Tier 3:** you will not find this data. You must either (a) commission collection, (b) rely on Bharat Data Sagar / AIKosh government contributions, or (c) accept that these four languages will be served by cross-lingual transfer + heavy synthetic augmentation and set honest expectations. Santali uses the Ol Chiki script; Manipuri uses both Bengali and Meitei Mayek scripts — script coverage in your tokenizer is a hard requirement for both.

---

## 3. Indic text — pre-training corpora

### 3.1 ✅ Sangraha (AI4Bharat / IIT Madras) — **the anchor corpus**

| Field | Value |
|---|---|
| **Owner** | AI4Bharat, IIT Madras. Funded by EkStep Foundation, Rohini Nilekani Philanthropies, Google India |
| **License** | **CC-BY-4.0** (data); MIT (tooling). Explicitly permits commercial use |
| **Size** | **251B tokens**, 22 languages. Verified 64B / Unverified 24B / Synthetic 162B |
| **Stage** | PT |
| **Access** | https://huggingface.co/datasets/ai4bharat/sangraha · Code: https://github.com/AI4Bharat/IndicLLMSuite |
| **Paper** | *IndicLLMSuite* — arXiv:2403.06350 |
| **Used by** | Foundation for most Indic LLM work post-2024; cited by BharatGen, Krutrim, tokenizer research |
| **Risk** | The 162B synthetic portion is machine-translated Wikimedia + transliteration. **Do not count it as natural Indic.** Some 2024 WMT analyses note Varta news content overlaps. Dedup against Varta and IndicCorp before mixing |

Companion pipeline: **Setu**, a Spark-based distributed cleaning/dedup/filtering pipeline purpose-built for Indian scripts (PDF, web, speech extraction). This is arguably as valuable as the data itself — it is MIT-licensed and you should adopt or fork it rather than rebuild.

### 3.2 ✅ IndicCorp v2 (AI4Bharat)

| Field | Value |
|---|---|
| **License** | **CC-0** (public domain dedication) — the most permissive Indic corpus available |
| **Size** | **20.9B tokens**, 24 languages, 4 language families |
| **Access** | https://huggingface.co/datasets/ai4bharat/IndicCorpV2 |
| **Used by** | IndicBERT v2, Paramanu family, numerous CPT efforts |
| **Risk** | Largely subsumed by Sangraha. Value is (a) CC-0 licence purity and (b) it is a clean dedup reference set |

### 3.3 🟡 IndicNLP Corpus v1 (AI4Bharat, 2020)

- 2.7B words, 10 languages. **License: CC BY-NC-SA 4.0 ⚠️ NonCommercial.**
- **Verdict: exclude from a commercial pre-training mix.** It is superseded by IndicCorp v2/Sangraha anyway. Listed here specifically so your pipeline blocklists it — it is frequently and incorrectly bundled into "Indic corpus" collections.
- https://github.com/AI4Bharat/indicnlp_corpus

### 3.4 🟡 BhashaKritika (Krutrim AI Labs / Ola) — largest synthetic Indic corpus

| Field | Value |
|---|---|
| **Owner** | Krutrim AI Labs (Ola) |
| **License** | **Krutrim Community License ⚠️** — a bespoke licence, not OSI-approved. **Requires legal review before commercial pre-training.** |
| **Size** | **540B synthetic tokens**, 10 Indic languages |
| **Method** | 5 generation strategies: document-grounded, persona-based, topic-guided, math/reasoning-grounded, translation-based. Backbones: Krutrim-2, Gemma-3, Llama-3.3, Sarvam-Translate |
| **Stage** | PT / MT |
| **Access** | https://huggingface.co/datasets/krutrim-ai-labs/BhashaKritika · Paper arXiv:2511.10338 (AAAI) |
| **Risk** | (1) Bespoke licence. (2) **Model-output provenance chain**: generated using Gemma-3 and Llama-3.3, whose own licences carry downstream terms on derived models — this is the classic "synthetic data licence contamination" problem. (3) Ships with per-record QC metadata, which is genuinely excellent — use the metadata to filter hard |

**This is the single most important dataset in this document for licence review.** It is exactly the shape of data you need, at exactly the scale you need, under exactly the licence you cannot casually accept. Either negotiate terms with Krutrim, or **replicate the methodology** — the paper is detailed enough to reproduce, and replication under your own licence is the cleaner path.

### 3.5 🟡 EKA Pre-training Indic Corpus v1 (Soket AI Labs + IIT Gandhinagar)

| Field | Value |
|---|---|
| **Owner** | Soket AI Labs + IIT Gandhinagar, under Project EKA, supported by IndiaAI Mission / MeitY |
| **License** | Listed as "Open Source" on AIKosh — **the specific licence string is not stated on the dataset page. Verify before use.** |
| **Size** | "Multi-billion-token", aggregated to October 2025. Stated long-term goal: multi-trillion-token |
| **Format** | CSV (`uri`, `text`) — note the URI field enables provenance auditing, which is valuable |
| **Access** | https://aikosh.indiaai.gov.in/home/datasets/details/eka_pretraining_indic_corpus_v1_1.html |
| **Risk** | Ambiguous licence string; unknown dedup status against Sangraha/CulturaX. Sample archive available for inspection first — **do that before committing pipeline time** |

### 3.6 🔴 Bharat Data Sagar (BharatGen consortium) — the biggest, and unavailable

| Field | Value |
|---|---|
| **Owner** | BharatGen — Section 8 not-for-profit consortium anchored by IIT Bombay, with IIT Madras, Kanpur, Hyderabad, Mandi, Kharagpur, IIIT Hyderabad, IIIT Delhi, IIM Indore |
| **Funding** | ₹235 crore (NM-ICPS) + ₹1,058 crore (MeitY / IndiaAI Mission) ≈ **₹1,293 crore total public investment** |
| **Size** | **"Crossed 20 trillion tokens", ~3 petabytes [VENDOR CLAIM]** — stated by BharatGen's CEO in Feb 2026 and echoed in DST press material. Includes text, speech, images, manuscripts, street-level collected regional material |
| **Stage** | PT (multimodal) |
| **Access** | **Not publicly redistributable.** Contribution is invited; bulk consumption is not offered. Models trained on it (Param-2-17B-A2.4B) are released under a **BharatGen non-commercial licence ⚠️** |
| **Risk** | Genuinely the largest India-centric corpus claimed to exist, and you cannot have it. Also: the 20T figure has not been independently audited, and "tokens" for a 3PB multimodal store is a soft unit |

**Strategic read:** Bharat Data Sagar is a *partnership target*, not a download. If your project is India-based and has public-interest framing, the realistic path is a formal data-sharing MoU with BharatGen and/or contributing to AIKosh in exchange for access. This is the single highest-leverage business-development action available to you, worth more than months of scraping.

### 3.7 ✅ AIKosh (IndiaAI Mission / MeitY) — the national aggregator

| Field | Value |
|---|---|
| **Owner** | IndiaAI (Independent Business Division, Digital India Corporation, MeitY), built with NeGD and Daffodil Software |
| **Scale** | **10,262+ datasets, 292 models, 20 sectors, 394 contributing organisations** (as of this writing) |
| **License** | **Per-dataset.** Contributors retain control; artefacts published as open, registered, or restricted. Many are CC-BY-4.0 |
| **Access** | https://aikosh.indiaai.gov.in/ · EOI to contribute: https://aikosh.indiaai.gov.in/static/datasets_EOI.pdf |
| **Notes** | Hosts IndicVoices, SPRING-INX, EKA corpus and much else. Also provides GPU/CPU notebooks via AIRAWAT |
| **Risk** | **Heterogeneous licensing is the whole risk.** You must build a per-dataset licence ingestion step — do not bulk-download. Some entries are "Redirect" links to Hugging Face rather than hosted data |

### 3.8 ✅ Bhashini / ULCA (National Language Translation Mission, MeitY)

| Field | Value |
|---|---|
| **Owner** | Digital India Bhashini Division, MeitY |
| **What** | ULCA (Universal Language Contribution API) — open, standardised data platform for Indic MT / ASR / TTS / OCR / NER / transliteration datasets, with **record-level attribution to every contributor** |
| **Access** | https://github.com/bhashini-dibd/ulca · Test sets: https://github.com/bhashini-dibd/ulca/tree/master/ulca-test-datasets |
| **Scale** | Millions of parallel sentences and thousands of audio hours across 21–22 languages, aggregated from many contributors |
| **Risk** | 🟡 Record-level attribution requirement means **you inherit a per-record attribution obligation**. Practically, maintain a manifest. Licences vary per contributed dataset |
| **Related** | **BhashaDaan** — crowdsourced citizen contribution (Suno India / Bolo India / Likho India / Dekho India) |

### 3.9 ✅ Global multilingual corpora with meaningful Indic slices

| Corpus | Owner | License | Total size | Indic relevance | Access |
|---|---|---|---|---|---|
| **FineWeb-2** | Hugging Face | **ODC-By 1.0** ✅ | ~3T words / 8TB compressed, 1000+ languages, 96 CC dumps (2013–Apr 2024) | **~40B words Indic** — small but high quality, per-language tuned filters using GlotLID | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **FineWeb-2-HQ** | EPFL ML | ODC-By ✅ | Top 10% of FineWeb-2 by XLM-R-based quality classifier, 20 languages | Matches FineWeb-2 with **6× fewer tokens** | https://huggingface.co/datasets/epfml/FineWeb2-HQ |
| **HPLT v3.0** | HPLT consortium (EU) | Open ✅ | **~30T tokens, 198 languages, ~50TB compressed** (July 2025) | Largest open multilingual corpus in existence. HPLT v2 was 7.6T/193 langs | https://hplt-project.org/datasets |
| **CulturaX** | UOregon / Nguyen et al. | ODC-By / mC4+OSCAR terms 🟡 | 6.3T tokens, 167 languages | Used as a Sangraha input; AI4Bharat re-cleaned it with Setu | https://huggingface.co/datasets/uonlp/CulturaX |
| **MADLAD-400** | Google | **CC-BY-4.0** ✅ | 2.6T tokens (clean), 419 languages | Manually audited — the audit is the value. Broad Indic coverage incl. long-tail | https://huggingface.co/datasets/allenai/MADLAD-400 |
| **MaLA / Glot500-c** | Helsinki NLP et al. | Mixed 🟡 | 939 languages (MaLA) | Best long-tail language coverage; tiny per-language volume. Useful for Tier-3 languages only | Search HF for `MaLA-LM` |
| **mC4 / OSCAR / CC-100** | Google / Inria / Meta | Varies 🟡 | — | **Largely superseded.** Retain only as dedup references | — |

**Critical engineering note:** Sangraha, CulturaX, MADLAD-400, FineWeb-2 and HPLT all derive from Common Crawl. **Expect 60–80% cross-corpus duplication.** Nemotron-CC's authors found FineWebEdu-2 and DCLM contained roughly 80% fuzzy duplicates because they only did sharded approximate dedup. Budget for **global MinHash+LSH dedup across the entire mix**, not per-corpus. This single step is the difference between "20T tokens" and "20T tokens of which 6T are unique."


---

## 4. English + global web — the reasoning substrate

You said English will be present "sufficiently." It needs to be more than sufficient: **English and code are where the model learns to reason**, and that reasoning transfers cross-lingually. Nemotron researchers found that adding *translated* synthetic QA data lifted Global-MMLU accuracy from 37.0 → 47.0. The English tier is not filler; it is the engine.

### 4.1 ✅ The Hugging Face FineData family — the open backbone

| Dataset | Size | License | Stage | Notes |
|---|---|---|---|---|
| **FineWeb** | **15T tokens** (18.5T in later revisions), English, 96 CC dumps | **ODC-By 1.0** ✅ | PT | The default open English base. GPT-2 tokenizer counts |
| **FineWeb-Edu** | **1.3T tokens** | ODC-By ✅ | PT/MT | Educational-quality classifier applied to FineWeb. **Punches far above its weight** — this is your "curriculum" spine for English |
| **FinePDFs** | **3T tokens** extracted from web PDFs | ODC-By ✅ | PT | Newer, far less duplicated against web-HTML corpora. High information density |
| **FinePDFs-Edu** | **350B+ tokens** | ODC-By ✅ | MT | Educational filter on FinePDFs |
| **FineWiki** | Wikipedia, 300+ languages, better extraction | ODC-By ✅ | PT | **Directly relevant** — better Indic Wikipedia extraction than raw dumps |
| **FineTranslations** | **1T + 1T tokens** parallel, 500+ languages ↔ English | ODC-By ✅ | PT/MT | Built by translating FineWeb-2 into English with Gemma-3-27B |

All at: https://huggingface.co/HuggingFaceFW

> **FineTranslations deserves emphasis.** It contains parallel English↔Indic derived from FineWeb-2's Indic content. For a model whose primary task list includes translation, this is a first-class asset that most teams overlook because it is filed as a "pre-training" dataset. Note the Gemma-3 generation provenance — same licence-chain question as BhashaKritika (§3.4). Verify Gemma terms as applied by HF's release.

### 4.2 ✅ NVIDIA Nemotron pre-training family — the highest-quality open mix

NVIDIA has been unusually generous, and these are explicitly released "with commercial use in mind."

| Dataset | Size | Contents |
|---|---|---|
| **Nemotron-CC** (v1) | **6.3T tokens** (4.4T globally-deduped real + 1.9T synthetic) | 99 CC snapshots, 2013–2024 |
| **Nemotron-CC-v2** | **6.6T tokens** | + 8 more CC snapshots (2024–25), synthetic rephrasing via Qwen3-30B-A3B, **synthetic Diverse-QA translated into 15 languages** |
| **Nemotron-CC-v2.1** | **+2.5T new tokens** | 3 more snapshots; 5 rephrase prompts applied to the Medium-High-Quality tier across 110 CC snapshots → **2.1T new tokens** |
| **Nemotron-Pretraining-Code-v1/v2/v3** | +377M GitHub files in v2 alone; **427.9B-token Common-Crawl-code corpus** in v2.1 | License-based removal (stricter than BigCode), exact + fuzzy dedup, OpenCoder heuristics, full metadata annotations |
| **Nemotron-CC-Math** | **133B tokens** | Purpose-built math pre-training corpus (arXiv:2508.15096) |
| **Nemotron-Pretraining-SFT-v1** | — | Pre-training-stage SFT blend |

Access: https://huggingface.co/nvidia (search "Nemotron Pretraining Dataset"). Papers: arXiv:2412.02595 (Nemotron-CC), arXiv:2508.14444, arXiv:2508.15096, arXiv:2512.20848 (Nemotron 3 Nano).

**Why this matters for you:** the "synthetic Diverse-QA translated into 15 languages" component is a validated recipe for exactly your problem, and they published the rephrasing prompts (Nemotron-CC paper, Appendix H). Port it to Indic.

### 4.3 ✅ Other major open English corpora

| Corpus | Owner | License | Size | Verdict |
|---|---|---|---|---|
| **DCLM-Baseline** | DataComp consortium | Open ✅ | 3.8T (~1.0T unique) | Strong quality-filter research; **heavily duplicated internally** |
| **Dolma** | Allen Institute for AI | ODC-By ✅ | ~3T | Extremely well documented; OLMo's corpus. Copy their governance model |
| **RedPajama-v2** | Together AI | Apache-2.0 (code) / mixed data 🟡 | 30T raw with quality signals | Use the **quality signals**, not the raw dump |
| **TxT360** | LLM360 | Open ✅ | ~15T deduplicated | Excellent global-dedup methodology reference |
| **Zyda-2** | Zyphra | ODC-By ✅ | 5T | Cross-dataset dedup + filtering already done |
| **The Pile** | EleutherAI | MIT (claimed) 🔴 | 825GB | **Contains Books3.** Original distribution taken down. Blocklist — see §13 |
| **C4 / mC4** | Google | ODC-By ✅ | 156B (en) | Superseded; keep as dedup reference |
| **Common Pile v0.1** | EleutherAI + collaborators | **Public domain / open licences only** ✅ | **8TB**, 30 sources | **The clean-provenance option.** >50% is Stack v2 code. Validated by training Comma v0.1-1T/2T (7B) to Llama-1/2-7B-comparable quality. arXiv:2506.05209 |
| **Common Corpus** | Pleias (AI Alliance Open Trusted Data Initiative) | Permissive only ✅ | **2,003,039,184,047 tokens** with provenance metadata | Ships a **multilingual historical-content toxicity classifier** — reusable for your Indic safety pipeline |

> **If you want a legally bulletproof floor:** Common Pile v0.1 + Common Corpus + IndicCorp v2 (CC-0) + MADLAD-400 (CC-BY) + Stack v2 permissive-only ≈ a **3–4T-token corpus with essentially zero provenance risk**. Build this tier first. It is your insurance policy and it de-risks every later decision.

---

## 5. Code and agentic coding — a primary capability

Coding and *agentic* coding are different data problems. Conflating them is the most common failure mode.

### 5.1 ✅ The Stack v2 (BigCode: Hugging Face + ServiceNow, with Software Heritage)

| Field | Value |
|---|---|
| **Owner** | BigCode Project; source archive from **Software Heritage** (Inria + UNESCO) |
| **License** | Per-file original licences; **permissively licensed or unlicensed files only**. StarCoder2 model licence is **BigCode OpenRAIL-M v1** 🟡 |
| **Size** | **67.5TB full / 32.1TB deduplicated / ~900B unique training tokens**, 3B+ files, **619 programming languages** |
| **Extras** | GitHub pull requests, Kaggle notebooks, code documentation |
| **Access** | https://huggingface.co/datasets/bigcode/the-stack-v2 · ID sets: `the-stack-v2-train-full-ids` (900B+ tokens), `the-stack-v2-train-smol-ids` (600B+) |
| **Used by** | StarCoder2 3B/7B/15B (3.3–4.3T tokens); Common Pile (>50% of it) |
| **Risk** | 🟡 **Opt-out obligation** — BigCode honours developer opt-out requests and the list changes over time; you must ingest the current list. **Copyleft contamination** — StarCoderData v1 explicitly excluded MPL/EPL/LGPL; Stack v2 includes "no license" files whose status is legally murky. **Recommendation: use the permissive-only subset for the base model.** Distributed as Software Heritage persistent IDs (SWHIDs), so retrieval is from SWH — budget the time |

### 5.2 Agentic coding — training environments, not text

Agentic capability comes from **executable environments with verifiable rewards**, not more GitHub text. This is where most Indic-LLM efforts will fall over.

| Resource | Owner | What it is | Size | Stage |
|---|---|---|---|---|
| **SWE-Gym** | Pan et al. | First public executable SWE environment: real repos + deps + unit tests + issues | 2,438 executable tasks | RL/SFT |
| **SWE-smith** | Yang et al. | Auto-generates bug-fix/issue tasks from *any* Python repo | **~50,000 instances from 128 repos** | RL/SFT |
| **R2E-Gym** | Jain et al. | Procedurally curated executable envs + hybrid verifiers | **8K+ tasks**; published 4,578-env SWE-Bench-disjoint subset | RL/SFT |
| **SWE-RL** | Wei et al. (Meta) | RL from commit histories/diffs as implicit demonstrations | 273K seed tasks | RL |
| **NVIDIA NeMo Gym** | NVIDIA | **Open RL environment framework.** Nemotron 3 Super used **21 environments and 37 RL datasets**: math (with and without Python tool), formal proof verification, competition code, single-step patch generation, STEM, instruction-following with rubric rewards, **safety (over-refusal reduction + jailbreak robustness)**, long context, conversational tool use, terminal use, Reasoning Gym | — | RL |
| **Nemotron-RL-Agentic-SWE-Pivot-v1** | NVIDIA | SWE-Gym + R2E-Gym refactored into NeMo Gym format, OpenHands environment | — | RL |
| **OpenThoughts-Agent (OT-Agent)** | OpenThoughts collective | **Fully open, reproducible SFT+RL curation pipeline for agentic models.** Ablates 6 pipeline stages over 100+ dataset variants against SWE-Bench Verified, Terminal-Bench 2.0, Aider Polyglot, BFCL-Parity, GAIA-127 | — | SFT/RL |
| **AutoTool / Open-AgentRL** | Gen-Verse | **200K tool-use trajectories** with explicit tool-selection rationales, **1,346 tools**, 120 task types; generalises from 460 seen tools to 886 unseen | 200K | SFT/RL |
| **CWM (Code World Model)** | Meta | 32B open-weights model + published methodology: up to **128 turns** over **131K context** in agentic SWE RL | — | Reference |

Access: NeMo Gym → https://docs.nvidia.com/nemo/gym/ and https://huggingface.co/collections/nvidia/nemo-gym · Open-AgentRL → https://github.com/Gen-Verse/Open-AgentRL · R2E-Gym → arXiv:2504.07164

> **The critical insight from NVIDIA's SWE-RL case study (Feb 2026):** agentic coding benchmarks measure a **model + harness pair**, not a model. SWE-bench Verified leaderboard entries are literally "harness + model" (e.g. "mini-SWE-agent + \<model\>"). If agentic coding is a primary capability, **you must co-develop a harness and train against it**, with rollouts of up to ~100 alternating model-call/command-execution steps, and a **separate container image per repository**. This is an infrastructure programme, not a dataset download.

🔴 **Contamination discipline:** SWE-bench Verified, Terminal-Bench, Aider Polyglot and BFCL are **test sets**. R2E-Gym's authors deliberately restricted SFT trajectory collection to repos with **no SWE-Bench overlap**. Replicate that. See §11.4.

### 5.3 Indic + code: the gap nobody has filled

There is **no meaningful "code with Indic-language comments/docstrings/issues" dataset in existence.** This is both a real gap and your clearest differentiation:

1. Take the Stack v2 permissive subset → translate comments/docstrings/READMEs into 10 Tier-1 Indic languages, preserving code tokens byte-exactly.
2. Generate Indic-language coding instructions grounded in real repositories (BhashaKritika's "document-grounded" strategy applied to code).
3. Synthesise Indic-language issue→patch pairs from SWE-smith instances by translating only the issue text, leaving tests and patches untouched — the verifier still works, so you get **verifiable RL signal in Hindi/Tamil/Bengali for free.**

Item 3 is the killer feature. A Bengali-language GitHub issue that resolves into a passing patch is the demo that proves this model is not just another Llama fine-tune.

---

## 6. Math, STEM, and reasoning-dense data

| Dataset | Owner | License | Size | Notes |
|---|---|---|---|---|
| **Nemotron-CC-Math** | NVIDIA | Open ✅ | **133B tokens** | arXiv:2508.15096 |
| **MegaMath** | LLM360 et al. | Open ✅ | ~370B tokens | Math-filtered web + code + synthetic |
| **OpenWebMath** | Open collective | ODC-By ✅ | 14.7B tokens | High-precision math web extraction |
| **Proof-Pile-2 / AlgebraicStack** | EleutherAI | Open ✅ | ~55B | Formal math + math-adjacent code |
| **peS2o / S2ORC** | Allen Institute for AI | ODC-By ✅ | ~40B tokens | Open-access academic papers, cleaned |
| **arXiv bulk** | Cornell/arXiv | Per-paper (many CC-BY) 🟡 | 100B+ | S3 requester-pays. **Per-paper licence check required** |
| **PubMed Central OA subset** | NIH | Open ✅ | — | Medical/biology reasoning |
| **MIND** | NVIDIA | Open ✅ | — | Math-Informed syNthetic Dialogues — converts math into dialogue for pre-training (ICLR 2025) |
| **OpenThoughts / OpenCodeReasoning / AceReason** | Open collectives / NVIDIA | Open ✅ | Millions of traces | Long-chain reasoning SFT |
| **Reasoning Gym** | Stojanovski et al. | Open ✅ | Procedural | Generates **unlimited verifiable** reasoning tasks — effectively infinite RLVR signal |

🔴 **Do not use:** anything scraped from paywalled journals (Elsevier, Springer, Wiley) without licence. Note **Wiley has publicly licensed content to AI firms for a reported $40M+ across two deals** — the market rate is established, which makes unlicensed use clearly wilful.

**Indic math/STEM is essentially absent.** Practical path: translate MegaMath / Nemotron-CC-Math solutions into Indic while preserving LaTeX and numerals exactly, plus generate from Indian competitive-exam *syllabi* (syllabus structure is not copyrightable; the question papers usually are).

---

## 7. Parallel and translation corpora

| Dataset | Owner | License | Size | Notes |
|---|---|---|---|---|
| **BPCC** (Bharat Parallel Corpus Collection) | AI4Bharat | CC-BY-4.0 ✅ | **Largest public Indic parallel corpus; all 22 languages** | Underlies IndicTrans2. Includes ~2.2M human-created pairs from AI4Bharat's 100+ in-house translators |
| **Samanantar** | AI4Bharat | CC-BY-4.0 ✅ | **~50M sentence pairs**, 11 Indic languages | Prior SOTA; subsumed into BPCC. TACL 2022 |
| **IN-22** | AI4Bharat | CC-BY-4.0 ✅ | n-way parallel, 22 languages | **EVAL.** Critically includes **source-original** test sets with Indian-origin content — not just translated-from-English |
| **FLORES-200** | Meta | CC-BY-SA-4.0 🟡 | 200 languages | Standard MT eval; share-alike applies |
| **Aksharantar** | AI4Bharat | CC-BY-4.0 ✅ | **26M transliteration pairs** | Essential for romanised Hindi/Hinglish — how Indians actually type |
| **FineTranslations** | Hugging Face | ODC-By ✅ | 1T + 1T tokens, 500+ langs | See §4.1 |
| **OPUS** | Helsinki NLP (Tiedemann) | **Per-subcorpus** 🟡 | Very large | Aggregator. **Contains OpenSubtitles (copyright-tainted) — filter explicitly** |
| **MultiSynt/MT** | Consortium (2026) | Open ✅ | Trillion-token multi-parallel, 36 languages | Useful native-vs-translated evidence base |
| **NLLB mined bitext** | Meta | **CC-BY-NC** 🔴 | 200+ languages | **NonCommercial. Blocklist.** |

Access: BPCC / Samanantar / Aksharantar → https://huggingface.co/ai4bharat and https://ai4bharat.iitm.ac.in/

> **A finding worth internalising** (MultiSynt/MT, 2026): translated data is **not** a substitute for native corpora on all phenomena. On culturally-grounded and idiomatic tasks, models trained on native data outperform translated-data models throughout training. On commonsense reasoning, translated data closes the gap and eventually overtakes. **Translation buys you reasoning transfer; it does not buy you culture.** Spend your scarce natural-Indic tokens on the cultural and idiomatic surface, and let synthetic carry the reasoning load.

---

## 8. Speech and audio

Two reasons this matters: your model is multimodal, **and ASR transcription is one of the only ways to manufacture genuinely natural Indic text at scale.** India is an oral-first information economy — enormous amounts of Indic language were simply never written down.

### 8.1 Indic speech corpora

| Dataset | Owner | License | Size | Notes |
|---|---|---|---|---|
| **IndicVoices** | AI4Bharat / IIT Madras + Sarvam AI; funded by Bhashini/MeitY | **CC-BY-4.0** ✅ | **23.7K hours total, 11.2K transcribed**, 51K speakers, **400+ districts, all 22 languages**. 8% read / 76% extempore / 15% conversational | The flagship. Grew 7,348h → 12K → 23.7K. **Spontaneous, not read** — vastly more valuable. Also on AIKosh |
| **IndicVoices-R** | AI4Bharat | CC-BY-4.0 ✅ | **1,704 hours, 10,496 speakers, 22 languages** | TTS-grade. 93.25% extempore. NeurIPS 2024 |
| **Project Vaani** | **IISc Bengaluru + Google** | CC-BY-4.0 ✅ (verify per release) | **~31,270 h audio, 2,067 h transcribed, 289K images, 112 languages, 165 districts, 31 States/UTs** | **Also an image dataset.** Image-prompted speech elicitation. Rare district-level dialect coverage. https://vaani.iisc.ac.in |
| **SPRING-INX** | SPRING Lab, IIT Madras (Prof. S. Umesh) | Open, on AIKosh ✅ | **~3,400 h labelled** (Ph1 ~2000h + Ph2 ~1400h), 10 languages, 16 kHz | **Code-mixed transcriptions** — native script with Romanised/English words inline. Official ESPnet recipe |
| **SYSPIN** | IISc Bengaluru | **CC-BY-4.0** ✅ | **900+ hours**, 9 Indian languages, studio TTS | https://vaani.iisc.ac.in/dataset/syspindataset |
| **Shrutilipi** | AI4Bharat | CC-BY-4.0 ✅ | ~6,400 h mined from All India Radio | Aligned audio-text from public broadcast |
| **Kathbath** | AI4Bharat | CC-BY-4.0 ✅ | ~1,684 h, 12 languages | Part of IndicSUPERB |
| **IndicTTS** | IIT Madras | Research-use 🟡 | 13 languages | Older; verify terms |
| **MUCS 2021** | Interspeech challenge | Research 🟡 | 6 languages | |
| **Gramvaani** | Gram Vaani | Open 🟡 | ~1,000 h Hindi | Rural telephony — valuable acoustic diversity |
| **Common Voice** | Mozilla | **CC-0** ✅ | Growing Indic subsets | Cleanest licence in speech |
| **FLEURS** | Google | CC-BY-4.0 ✅ | 102 languages incl. Indic | Standard multilingual eval |

**Speech benchmarks:** **Svarah** (Indian-accented English ASR), **LAHAJA** (Indic accent/dialect robustness), **Vistaar**, **IndicSUPERB**.

### 8.2 Global speech/audio

| Dataset | License | Size | Notes |
|---|---|---|---|
| **VoxPopuli** | CC-0 ✅ | 400K h unlabelled | EU Parliament |
| **Multilingual LibriSpeech** | CC-BY-4.0 ✅ | 50K h, 8 languages | |
| **People's Speech** | CC-BY-SA 🟡 | 30K h | Share-alike |
| **GigaSpeech / Emilia / YODAS** | Mixed 🟡 | 10K–100K+ h | **YODAS is YouTube-derived — ToS exposure, see §13** |
| **AudioSet** | CC-BY 🟡 | 2M clips | YouTube-ID-based; heavy link rot |

> **The high-leverage move.** You have **~35,000+ hours of open, permissively-licensed Indic speech** across IndicVoices + Vaani + SPRING-INX + Shrutilipi + Kathbath + SYSPIN. At roughly 9,000 words/hour, full ASR transcription yields on the order of **300M+ words of genuinely natural, spontaneous, dialect-rich Indic text that exists in no web corpus anywhere.** Small in raw token terms; extraordinary in *distributional* value, because it is the conversational register that web text completely lacks — and conversation is your stated primary task. **Transcribe all of it.**

---

## 9. Vision, documents, and multimodal

### 9.1 Indic multimodal

| Dataset | Owner | License | Size | Notes |
|---|---|---|---|---|
| **Chitrakshara-IL** | Krutrim AI Labs | Krutrim licence 🟡 | **193M images, 30B text tokens, 50M interleaved multilingual documents**, 11 Indian languages | Largest Indic interleaved image-text corpus. From Common Crawl WARC. arXiv:2603.23521 |
| **Chitrakshara-Cap** | Krutrim AI Labs | Krutrim licence 🟡 | **44M image-text pairs, 733M tokens** | Alt-text caption pairs |
| **Project Vaani images** | IISc + Google | CC-BY-4.0 ✅ | **289K images** from 165 districts | **Genuinely India-representative imagery**, not Western stock. Rare and valuable |
| **Nayana** | Kolavi et al. (CognitiveLab) | Open 🟡 | Document-level Indic VLM pre-training | Search HF `Nayana` |
| **Bharat Scene Text** | De et al. (2025) | Open 🟡 | Indic scene text | OCR-in-the-wild |
| **IndicSTR12** | Lunia et al. | Open 🟡 | 12-script scene text recognition | |
| **iiit-indic-hw-words** | IIIT Hyderabad | Research 🟡 | Indic handwriting | |
| **Patram / Patram-Bench** | BharatGen | **BharatGen non-commercial** 🔴 | **52,000 documents, 2.4B+** per BharatGen's own presentation | 7B document-vision model + eval suite |
| **Hindi / Bengali Visual Genome** | Parida, Sen et al. | CC-BY 🟡 | Multimodal MT | |

**Indic multimodal benchmarks:** **IndicVisionBench** (cultural + multilingual VLM eval, arXiv:2511.04727), **BharatBench** (Krutrim — text/vision/speech, 8–10 Indian languages), **Drishtikon** (multimodal cultural understanding, 15 Indic languages).

### 9.2 Global vision-language

| Dataset | License | Size | Verdict |
|---|---|---|---|
| **Re-LAION-5B** | CC-BY-4.0 (metadata) 🟡 | 5B pairs minus removed links | **The CSAM-remediated release (Aug 2024). Original LAION-5B is 🔴 — see §13** |
| **DataComp / CommonPool** | CC-BY-4.0 🟡 | 12.8B | **Documented privacy problems** — see §13 |
| **COYO-700M** | CC-BY-4.0 🟡 | 700M | Kakao Brain |
| **OBELICS** | CC-BY-4.0 ✅ | 141M interleaved docs | Hugging Face; well-governed |
| **MINT-1T** | CC-BY-4.0 ✅ | 1T tokens interleaved | Salesforce |
| **PixMo** | ODC-By ✅ | Human-annotated | **Cleanest provenance in VLM data.** Allen Institute for AI; Molmo's corpus |
| **The Cauldron** | Mixed 🟡 | 50 VQA datasets | Per-subset licences |
| **DOCCI / DocVQA / ChartQA / InfographicVQA** | Mixed ✅🟡 | — | Document understanding |
| **WebVid-10M** | 🔴 | — | **Withdrawn. Do not use** |
| **Panda-70M / InternVid / HD-VILA** | 🟡 | Video | YouTube-derived — ToS exposure |

**OCR note:** **olmOCR / olmOCR-2** (Allen Institute for AI, arXiv:2510.19817) uses unit-test rewards for document OCR and is open. **Indic OCR remains materially worse than Latin-script OCR**, and it is the single bottleneck standing between you and India's enormous scanned print archive. Fine-tuning olmOCR for Devanagari, Bengali, Tamil, Telugu and Perso-Arabic scripts is arguably the highest-ROI engineering project in this whole plan — it unlocks a corpus nobody else can reach.

---

## 10. The curriculum ladder: "small child to PhD"

You asked for curriculum-graded, quality-ordered data. Here is the ladder — and **this section contains the highest density of licence landmines in the entire document.**

### 10.1 🔴 The Indian school-curriculum trap

| Source | Content | License | Verdict |
|---|---|---|---|
| **NCERT textbooks** (via DIKSHA) | Complete K-12 curriculum, all subjects | **CC BY-NC-ND** 🔴 | **NonCommercial AND NoDerivatives.** Training a commercial model is a commercial derivative use. **Excluded.** |
| **DIKSHA resources** (non-textbook) | **36 Indian languages**, enormous volume of teacher/learner content | **CC BY-NC-SA** 🔴 | **NonCommercial. Excluded** from a commercial mix |
| **NPTEL / SWAYAM** | 2,500+ courses, IIT/IISc lectures + transcripts; 61M+ enrolments, 6.2M certificates | Typically **CC BY-NC-SA** 🔴 | NonCommercial in most cases. **Verify per course**; a minority may be more permissive |
| **eGyanKosh (IGNOU)** | Open University course material | Varies 🟡 | Per-item check |
| **National Digital Library of India (NDLI)** | Aggregator, millions of items | **Per-item** 🟡 | Aggregator — inherits source licences |
| **Shodhganga (INFLIBNET)** | Indian PhD theses | Per-thesis 🟡 | Author copyright retained; some CC |

**This is the most counter-intuitive finding in this report.** India's magnificent public educational corpus — precisely the curriculum-graded, pedagogically-ordered, multilingual material you want, in 36 languages — is **almost entirely NonCommercial-licensed and therefore unusable** in a commercial foundation model.

**Three legitimate options:**

1. **Seek an explicit licence grant.** NCERT / Ministry of Education can grant a commercial-training licence for a sovereign-AI public-interest project. This costs a letter, not money. **Do this first — it is the highest expected-value action in this document.** Note that the Delhi High Court's July 2026 fair-dealing ruling (§12) does *not* dissolve an NC licence; a contract does.
2. **Structure as non-commercial research**, publish under a research licence, and license a commercial variant trained on a clean-data-only base separately. Expensive but clean.
3. **Replace, don't copy.** Curriculum *structure* — syllabi, topic sequences, difficulty progressions — is fact/system, not protected expression. "Generate a Class 7 Marathi explanation of photosynthesis at CBSE level, 400 words, with two worked examples" produces original text you own outright. **This is the practical path and it works.** It is also how you get true difficulty-graded curriculum learning: generate the same concept at 5 difficulty tiers across 22 languages.

### 10.2 ✅ Clean curriculum sources

| Source | License | Value |
|---|---|---|
| **FineWeb-Edu / FinePDFs-Edu** | ODC-By ✅ | 1.3T + 350B tokens of educational English — your graduate-level English spine |
| **Wikipedia / FineWiki** (all Indic editions) | CC BY-SA 🟡 | Share-alike. Widely used commercially under fair-use/fair-dealing theories, but the SA obligation is a live legal question. Wikimedia has signed 6 AI licensing deals — a commercial channel exists |
| **Wikibooks / Wikiversity / Wikisource** | CC BY-SA 🟡 | **Wikisource holds substantial public-domain Indic literature** |
| **Project Gutenberg** | Public domain ✅ | Includes India-related and some Indic-language classics |
| **Internet Archive — Public Library of India / Digital Library of India** | **Per-item** 🟡 | ~500K+ scanned Indian books. **Mixed copyright status; many in-copyright.** Requires item-level rights clearance. Enormous prize, real risk |
| **Common Pile educational subset** | Open ✅ | Pre-cleared OER |
| **OpenStax** | **CC-BY** ✅ | Fully usable college textbooks |
| **MIT OpenCourseWare / OER Commons** | Mixed (much CC BY-NC-SA) 🟡 | Check per item |
| **PIB (Press Information Bureau)** | GoI Open Data / GODL-India 🟡 | **Multilingual government press releases in 10+ Indian languages — naturally parallel, clean formal register** |
| **Lok Sabha / Rajya Sabha debates** | Public record 🟡 | Decades of formal Hindi/English |
| **data.gov.in / NDAP (NITI Aayog)** | GODL-India 🟡 | Structured India data — good for grounded QA generation |

### 10.3 Indian legal corpus (public record, high formality)

| Dataset | Size | License | Notes |
|---|---|---|---|
| **eCourts High Court judgments** (via Open Justice India) | **25 High Courts, 1950–2025, ~1TB**, JSON+Parquet, quarterly updates | Public record 🟡 | https://openjustice-in.github.io/ |
| **eCourts district-court records** | **81M case records** | Public record 🟡 | Lower judiciary |
| **NyayaAnumana** (IIT Kharagpur Law-AI Lab) | **2,282,137 case proceedings** to April 2024 | Research 🟡 | **Scraped from Indian Kanoon — see risk note** |
| **ILDC** | 35K Supreme Court cases | **Research/non-commercial, by request** 🔴 | Malik et al. 2021 |
| **IndicLegalQA** | QA pairs | **CC BY 4.0** ✅ | Mendeley Data |
| **MILPaC** | Parallel legal corpus, English + 9 Indic | Open 🟡 | Rare Indic legal parallel data |

🟡 **Indian Kanoon risk note.** Indian Kanoon aggregates public-domain judgments, but its *compilation, formatting and metadata* may attract database/compilation rights, and its Terms of Service restrict bulk scraping. The judgments themselves are government works. **Recommendation: source judgments from eCourts and High Court sites directly, not from Indian Kanoon.** Same content, far better legal posture.

---

## 11. Post-training: SFT, preference alignment, and RL

### 11.1 ✅ Indic instruction-tuning data

| Dataset | Owner | License | Size | Notes |
|---|---|---|---|---|
| **IndicAlign-Instruct** | AI4Bharat | **CC-BY-4.0** ✅ | **74.7M prompt-response pairs, 20 languages** | The largest Indic SFT collection. Includes Dolly-T, OpenAssistant-T (translated), plus natively-generated conversations grounded in Wikipedia infoboxes |
| **IndicAlign-Toxic** | AI4Bharat | CC-BY-4.0 ✅ | **123K pairs** | Safety alignment: toxic prompts → non-toxic responses. Two parts: **HH-RLHF-T** (Anthropic HH-RLHF toxic prompts + Llama-2-70B-Chat refusals) and **Toxic Matrix** (synthetic, axes = Target Group × Prompt Style) |
| **UPDESH** | Research consortium (2026) | Open ✅ | **9.5M data points, 13 Indian languages** | **Bottom-up** synthesis: grounded in language-specific Wikipedia using ≥235B-parameter open models, rather than top-down English translation. arXiv:2509.21294. **Methodologically the most interesting Indic SFT work to date** |
| **Indic Instruct Data v0.1** | AI4Bharat (Gala et al.) | CC-BY ✅ | Hindi | Airavata's SFT set |
| **Aya Collection / Aya Dataset** | Cohere For AI | **Apache-2.0** ✅ | 65 languages, human-curated + templated | Strong Indic representation; human-written, not just translated |
| **Bactrian-X** | Li et al. | Open 🟡 | 3.4M pairs, 52 languages | Translated Alpaca+Dolly. Alpaca lineage = **OpenAI-output provenance** ⚠️ see §13 |
| **M2Lingual** | — | Open 🟡 | 70 languages | Evol-guided taxonomy generation |

### 11.2 ✅ General SFT / reasoning / agentic post-training data

| Dataset | Owner | License | Notes |
|---|---|---|---|
| **Nemotron post-training collections** (Nano/Super/Ultra v3) | NVIDIA | Open, commercial-use ✅ | **~50 items** in the collection; the most complete open post-training corpus available |
| **Nemotron agentic/tool-use collection** | NVIDIA | Open ✅ | **11 items** covering function calling, multi-step agentic, terminal use, SWE workflows |
| **Nemotron code collection** | NVIDIA | Open ✅ | **14 items**: competitive programming, SWE, code pre-training |
| **Tülu 3 / Tülu 3 SFT Mixture** | Allen Institute for AI | ODC-By ✅ | Fully documented, reproducible post-training recipe. **Copy their methodology** |
| **OpenThoughts / OpenThoughts-Agent** | OpenThoughts collective | Open ✅ | Reasoning + agentic traces with full ablations |
| **SmolTalk / Magpie** | HF / academic | Open 🟡 | Self-synthesised instruction data |
| **xLAM function-calling datasets** | Salesforce | CC-BY-NC 🔴 (check version) | **Verify — some Salesforce releases are NC** |
| **Glaive function calling** | Glaive | Apache-2.0 ✅ | Tool-use SFT |
| **BFCL (Berkeley Function Calling Leaderboard) data** | UC Berkeley | Apache-2.0 ✅ | **EVAL — do not train on the test split** |

### 11.3 ✅ Preference / RL data

| Dataset | Owner | License | Notes |
|---|---|---|---|
| **HH-RLHF** | Anthropic | MIT ✅ | Foundational helpfulness/harmlessness preference data. Translated versions exist for Indic via IndicAlign |
| **UltraFeedback** | OpenBMB | MIT ✅ | Widely used DPO data |
| **Nemotron RLVR datasets** (37 across 21 environments) | NVIDIA | Open ✅ | **Verifiable-reward RL** across math, code, STEM, instruction-following, safety, long-context, agentic tool use, terminal use |
| **Reasoning Gym** | Stojanovski et al. | Open ✅ | Procedurally generated verifiable tasks |
| **Multilingual preference data** (Cohere) | Cohere For AI | Varies 🟡 | *RLHF Can Speak Many Languages* — arXiv:2407.02552 |

### 11.4 The Indic alignment gap — and what to do about it

Findings you should design around:

- **Safety alignment does not transfer cleanly across languages.** The 2024 *Language Barrier* work showed multilingual safety degrades sharply in low-resource languages even after English RLHF. **A model that refuses correctly in English will comply with the same jailbreak in Bodo.** You must red-team in all 22 languages, not extrapolate.
- **Cross-lingual instruction-following *does* transfer.** NVIDIA's Nemotron-Mini-Hindi work found English-only SFT improved Hindi instruction-following, and that filtered translated data added little — but that **synthetic Hindi samples during DPO did help.** So: SFT can lean English; preference alignment must be genuinely multilingual.
- **Cultural alignment is separate from safety alignment.** Benchmarks like **Sanskriti** and **Pariksha** exist specifically because a model can be safe and still be culturally wrong. Budget for both.

**Recommended Indic alignment stack:**
1. English-heavy SFT (Tülu 3 + Nemotron + OpenThoughts) for capability.
2. IndicAlign-Instruct + UPDESH + Aya for Indic instruction-following.
3. Natively-generated (not translated) Indic preference pairs for DPO/GRPO — generate with a strong Indic model, judge with human annotators in-language.
4. IndicAlign-Toxic + your own 22-language red-team set for safety RL.
5. NeMo Gym-style verifiable-reward environments for coding/agentic/math, with issue text translated into Indic (§5.3).

---

## 12. Benchmarks — with an explicit train / validate / test policy

### 12.1 Indic benchmarks

| Benchmark | Owner | Coverage | Size | Type | Notes |
|---|---|---|---|---|---|
| **MILU** | AI4Bharat / IBM Research | **11 Indic languages**, 8 domains, 41 subjects | **~85K MCQs** (+ **8,933-sample validation set**) | EVAL | Sourced from **1,500+ Indian competitive exams**. GPT-4o topped it at **~74%** across 42–45 evaluated LLMs. **Gated — request access on HF.** github.com/AI4Bharat/MILU |
| **IndicXTREME** | AI4Bharat | **20–22 languages**, 9 NLU tasks | **105 evaluation sets** (52 new) | EVAL | The broadest Indic NLU benchmark |
| **IndicGLUE** | AI4Bharat | 11 languages | — | EVAL | The original; now a baseline |
| **IndicNLG Benchmark** | AI4Bharat | 11 languages, 5 generation tasks | — | EVAL | Biography/headline/summary/paraphrase/question generation |
| **IndicGenBench** | **Google Research** | **29 Indic languages** | — | EVAL | Generation: summarisation, translation, QA. ACL 2024 |
| **IndicMMLU-Pro** | Sankalp KJ et al. | **9 languages** (hi, bn, gu, mr, kn, pa, ta, te, ur) | MMLU-Pro derived | EVAL | Translated via IndicTrans2 + back-translation QA + **13 human reviewers**. arXiv:2501.15747. 🟡 **Translation-derived — carries English-benchmark cultural bias** |
| **BhashaBench V1** | **BharatGen** | English + Hindi | **74,166 QA pairs** (52,494 en / 21,672 hi) | EVAL | **4 domains: Agriculture, Legal, Finance, Ayurveda**; 90+ subdomains, 500+ topics. GPT-4o: 76.49% Legal vs 59.74% Ayurveda. arXiv:2510.25409 |
| **ParamBench** | BharatGen | Hindi | Graduate-level | EVAL | UGC-NET + UPSC, expert-verified |
| **IndicParam** | BharatGen | Low-resource Indic | Graduate-level | EVAL | HF: `bharatgenai/IndicParam` |
| **Indic QA Benchmark** | IBM Research + IIT Bombay | 11 Indic languages | Large-scale | EVAL | NAACL 2025 Findings |
| **IndicIFEval** | Research (2026) | **14 Indic languages** | Verifiable instruction-following | EVAL | arXiv:2602.22125. **Directly relevant to your agentic goals** |
| **BharatBench** | **Krutrim (Ola)** | **8–10 Indian languages**, text + vision + speech | — | EVAL | The only major *multimodal* Indic benchmark suite |
| **IndicVisionBench** | Research (2025) | Indic VLM, cultural + OCR + MMT | — | EVAL | arXiv:2511.04727 |
| **Drishtikon** | Maji et al. | **15 Indic languages** | Multimodal cultural | EVAL | |
| **Sanskriti / Pariksha** | Research | English, India-focused | Socio-cultural alignment | EVAL | Cultural correctness ≠ safety |
| **IN-22** | AI4Bharat | **22 languages** | n-way parallel MT | EVAL | **Source-original** test sets — the gold standard for Indic MT eval |
| **IndicSUPERB / Svarah / LAHAJA / Vistaar** | AI4Bharat | Speech | — | EVAL | ASR, Indian-accented English, dialect robustness |
| **Global-MMLU** | Cohere For AI + collaborators | 42 languages | — | EVAL | Culturally-annotated MMLU; separates culture-sensitive from culture-agnostic items |
| **INCLUDE** | Research | 44 languages | Regional exams | EVAL | Native, not translated |
| **Indic LLM Leaderboard** | CognitiveLab (Aditya S Kolavi) | Indic | — | Leaderboard | Community-run |

### 12.2 Global benchmarks you must also cover

**Knowledge/reasoning:** MMLU, MMLU-Pro, GPQA-Diamond, BIG-Bench Hard, ARC-Challenge, HellaSwag, WinoGrande
**Math:** GSM8K, MATH, AIME 2024/2025, HMMT, MathArena
**Code:** HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench, BigCodeBench, CRUXEval, Aider Polyglot
**Agentic coding:** **SWE-bench Verified** (500 tasks), SWE-bench Multilingual, Terminal-Bench 2.0, SWE-Lancer
**Tool use / agents:** **BFCL**, τ-bench (tau-bench), GAIA, WebArena, OSWorld, AgentBench, MedAgentBench, FinanceAgent-Terminal
**Long context:** RULER, LongBench v2, ∞Bench
**Safety:** HarmBench, JailbreakBench, AgentHarm, XSTest (over-refusal), **plus your own 22-language red-team set**
**Instruction following:** IFEval, MultiChallenge, **IndicIFEval**

### 12.3 🔴 Mandatory train / validate / test policy

This is a governance document, not a suggestion. Bake it into the pipeline.

**Rule 1 — Absolute test-set quarantine.**
Maintain a `BENCHMARK_BLOCKLIST` containing the **exact text** of every evaluation item across every benchmark above. Run **13-gram overlap detection** (the Llama/GPT convention) plus **MinHash near-duplicate detection** against the *entire* pre-training corpus at every stage. Any document with a hit is dropped, not down-weighted. Log every drop with the benchmark it hit — you will be asked to prove this.

**Rule 2 — Three-way split discipline.**

| Split | Source | Use | Who may see it |
|---|---|---|---|
| **Train** | The mix in §14 | Gradient updates | Pipeline |
| **Validation** | MILU's dedicated 8,933-sample val set; held-out 5% of each Indic corpus stratified by language; a 2,000-item internal Indic set never published | Checkpoint selection, mixture ablations, early stopping | Training team, freely |
| **Test** | MILU test, IndicXTREME, IN-22, BhashaBench, SWE-bench Verified, MMLU-Pro, GPQA, etc. | **Final reporting only** | **Run once per release candidate. Locked. Separate team.** |

**Rule 3 — Hold out one benchmark entirely.**
Pick one Indic benchmark (recommend **BhashaBench V1**, because it is domain-specific and unlikely to appear in web text) and never look at it during development. It is your honest-broker signal. If your held-out benchmark tracks your tuned benchmarks, you were not overfitting. If it does not, you were.

**Rule 4 — Build a private Indic eval set now.**
Every public Indic benchmark will be contaminated within 18 months. Commission **~3,000 native-speaker-written items across 22 languages, never published**, covering conversation, translation, cultural knowledge, and code-mixed input. This is a ~₹15–30 lakh project and it will be the most valuable evaluation asset you own.

**Rule 5 — Report translation-derived benchmarks separately.**
IndicMMLU-Pro and similar are translated from English. They measure *translated-English competence*, not Indic competence. Report them, but weight MILU / IN-22 / BhashaBench / INCLUDE (natively sourced) higher in decision-making.

**Rule 6 — Agentic benchmarks report model + harness.**
Never publish a SWE-bench number without naming the harness, the scaffold version, the turn limit, and the context window. Anything else is not reproducible.

---

## 13. 🔴 Explicit list: licensed, restricted, and affiliation-complex datasets

You asked for this explicitly. **Everything in this section requires either a licence, a legal sign-off, or exclusion.**

### 13.1 NonCommercial-licensed — exclude from a commercial model

| Dataset | Licence | Owner | Why it's here |
|---|---|---|---|
| **NCERT textbooks / DIKSHA textbook content** | CC BY-**NC**-ND | NCERT / MoE | K-12 curriculum, 36 languages. NC **and** ND |
| **DIKSHA non-textbook resources** | CC BY-**NC**-SA | NCERT / MoE | Vast; still NC |
| **NPTEL / SWAYAM course content** | Typically CC BY-**NC**-SA | MoE / IITs / IISc | 2,500+ courses |
| **AI4Bharat IndicNLP Corpus v1** | CC BY-**NC**-SA 4.0 | AI4Bharat | 2.7B words, 10 languages. *Note: IndicCorp v2 and Sangraha are NOT NC — only v1 is* |
| **NLLB mined bitext** | CC BY-**NC** | Meta | 200+ language pairs |
| **ILDC (Indian Legal Documents Corpus)** | Research/non-commercial, by request | Malik et al. | 35K SC cases |
| **BharatGen Param-2 / Patram post-trained checkpoints** | **BharatGen non-commercial licence** | BharatGen | Models, not data — but blocks distillation-based data generation |
| **xLAM (some releases)** | CC BY-NC | Salesforce | **Verify the specific release** |

### 13.2 Bespoke / custom licences requiring negotiation

| Dataset | Licence | Owner | Action |
|---|---|---|---|
| **BhashaKritika** (540B synthetic Indic tokens) | **Krutrim Community License** | Krutrim AI Labs (Ola) | **Negotiate or replicate.** See §3.4 |
| **Chitrakshara-IL / Chitrakshara-Cap** | Krutrim licence | Krutrim AI Labs | Largest Indic interleaved multimodal corpus |
| **Bharat Data Sagar** | Not distributed | BharatGen consortium | **Pursue an MoU.** ₹1,293 crore of public investment; ~20T tokens claimed |
| **The Stack v2 / StarCoder2** | BigCode OpenRAIL-M v1 + per-file licences + **live opt-out list** | BigCode / Software Heritage | Ingest the opt-out list; use permissive-only subset |
| **AIKosh datasets** | **Per-dataset: open / registered / restricted** | Various contributors via IndiaAI | Per-dataset licence ingestion required |
| **ULCA / Bhashini datasets** | Per-contributor, **record-level attribution** | Bhashini DIBD, MeitY | Maintain an attribution manifest |
| **EKA Pre-training Indic Corpus v1** | "Open Source" — **string unspecified** | Soket AI Labs + IIT Gandhinagar | **Get the exact licence in writing before ingesting** |

### 13.3 Share-alike (copyleft) — usable but with obligations

| Dataset | Licence | Obligation |
|---|---|---|
| Wikipedia / Wikisource / Wikibooks / FineWiki | CC BY-SA | Attribution + share-alike. **Whether model weights are a "derivative work" is legally unresolved.** Most labs proceed; document your position |
| FLORES-200 | CC BY-SA 4.0 | Same |
| People's Speech | CC BY-SA | Same |
| Stack Exchange / Stack Overflow dumps | CC BY-SA | **Plus: Stack Overflow now licenses data commercially to AI firms.** Free bulk use is contested |
| Common Crawl WET/WARC | CC BY (CC's own terms) but **contains third-party copyrighted content** | The fair-dealing question in §14 lives here |

### 13.4 🔴 Do NOT use — litigation-tainted, pirated, or harmful

| Dataset | Why | Status |
|---|---|---|
| **Books3** | Pirated books via Bibliotek torrent. Core to *Kadrey v. Meta*. Original Pile download URL dead since ~Sept 2023 | **Absolute blocklist.** If you ingest The Pile, strip Books3 first |
| **LibGen / Library Genesis** | Pirated. Central to *Tremblay v. OpenAI* allegations | **Absolute blocklist** |
| **Anna's Archive / Z-Library** | Pirated shadow libraries | **Absolute blocklist** |
| **LAION-5B (original)** | **CSAM found in the dataset** (Thiel, Dec 2023); taken down from hosting services. Also documented PII/medical-record leakage | **Absolute blocklist.** Use **Re-LAION-5B** (Aug 2024) only, and even then with a fresh safety scan |
| **DataComp CommonPool** | Documented large-scale privacy problems (arXiv:2506.17185); substantial overlap with LAION-5B | 🔴 Avoid, or apply your own PII/CSAM filtering pipeline first |
| **WebVid-10M** | Withdrawn by its creators | **Blocklist** |
| **OpenSubtitles** (within OPUS) | Copyrighted film/TV subtitles | **Filter out of any OPUS ingestion** |
| **Any Alpaca-lineage data** (Alpaca, Bactrian-X, mAlpaca, many "self-instruct" sets) | Generated with OpenAI models; **OpenAI ToS prohibits using outputs to train competing models** | 🟡🔴 **Provenance-tainted.** Materially risky for a model that will compete with OpenAI. Prefer Aya, Tülu 3, Nemotron, OpenThoughts, UPDESH |
| **Reddit / Pushshift bulk dumps** | Reddit now licenses data commercially (~$203M disclosed contract value at IPO; ~$60M/yr Google, ~$70M/yr OpenAI reported). Free bulk scraping is against ToS and actively enforced | 🔴 **Do not scrape. License or skip** |
| **Twitter/X archives** | ToS + active enforcement | 🔴 Skip |
| **YouTube-derived corpora** (YouTube-Commons, YODAS, Panda-70M, InternVid, HD-VILA) | YouTube ToS prohibits bulk download regardless of the underlying CC licence on individual videos | 🟡🔴 Document a position or skip |
| **Scraped Indian news sites** | **The ANI v. OpenAI litigation exists precisely because of this.** Digital News Publishers Association and Federation of Indian Publishers are intervenors | 🟡 See §14 — the legal position improved dramatically in July 2026, but this is where the risk concentrates |
| **Any government/personal/financial identity documents** | DPDP Act; also basic ethics | **Absolute blocklist.** Build automated Aadhaar/PAN/passport-pattern detectors into ingestion |

### 13.5 🟡 Datasets with quality or reliability caveats

- **Sangraha Synthetic (162B of the 251B)** — machine-translated Wikimedia + transliteration. Fine to use, but **do not report it as natural Indic data**, and expect translationese artifacts.
- **HPLT v2 "cleaned" partitions** — an independent 2026 audit of the Somali partition found **17.3% byte-exact duplicates remaining** after the official dedup. Assume similar for Indic partitions; re-dedup yourself.
- **CC-100** — paragraph-split, not document-level. Damages long-context and document coherence. Use only as a dedup reference.
- **Tier-3 language partitions everywhere** — Sangraha's own published Stage-3 filtering table shows Bodo collapsing to **77 words / 1 document**. Treat any published token count for Bodo, Dogri, Santali, or Manipuri with deep scepticism, in every corpus.
- **IndicMMLU-Pro and other translated benchmarks** — measure translated-English competence. See §12.5.


---

## 14. The Indian legal and regulatory position (as of 28 July 2026)

> **This section changed materially four days before this document was written. Read it carefully.**

### 14.1 ANI v. OpenAI — the first Indian judicial ruling on AI training

**Judgment: Delhi High Court, Justice Amit Bansal, 24 July 2026. 135 pages.**

**What happened.** ANI (Asian News International) sued OpenAI in November 2024, alleging that OpenAI used ANI's news reports without permission to train ChatGPT, and separately that ChatGPT fabricated stories attributed to ANI. The Digital News Publishers Association and the Federation of Indian Publishers intervened.

**What the court held (at the interim-injunction stage):**

- OpenAI's storage and use of ANI's content to train its models is **prima facie protected as "fair dealing" for "private or personal use, including research"** under **Section 52(1)(a)(i) of the Copyright Act, 1957**.
- ANI **failed to establish** that ChatGPT memorised or reproduced its copyrighted news reports.
- Copyright protects **expression**, not the underlying **facts or information** — "news of the day" is not protected.
- The court **rejected** the argument that commercial entities are automatically excluded from claiming the fair-dealing defence.
- Interim injunction **refused**. The suit continues to trial on the merits.

**What it does NOT mean:**

1. **These are prima facie findings.** The judgment expressly states they do not control the final outcome. Discovery, expert evidence and trial are still ahead.
2. **It is a single-judge interim order.** An appellate bench could land differently. Serious commentary has already identified vulnerabilities: that the court treated an internal technical process as "private use" even where the objective purpose is a public commercial product, that it applied Canadian "research" jurisprudence to differently-worded Indian text, and that it arguably used updating construction to create a commercial AI-training exception that Parliament did not enact.
3. **The hallucination/false-attribution claim is unresolved.** ANI's suit always had two limbs; only the training limb was addressed.
4. **It does not override contracts.** An NC licence (NCERT, DIKSHA, NPTEL) is a contract, not a copyright default. Fair dealing does not let you ignore it.
5. **It does not touch the DPDP Act.** Copyright and data protection are separate regimes.

**Practical effect:** the burden has shifted onto publishers to prove **memorisation or reproduction**, not merely ingestion — a considerably harder evidentiary bar. For an India-based project, this is a meaningfully favourable environment relative to 12 months ago. **It is not a licence to scrape indiscriminately.**

**India has no explicit TDM (text and data mining) exception.** Unlike the EU (DSM Directive Arts. 3–4) or Japan (Art. 30-4), Indian copyright law is right-holder-centric with a closed list of fair-dealing purposes. The ANI ruling stretched Section 52(1)(a)(i) to cover training. That stretch is the entire legal foundation, and it is one appeal away from re-examination.

### 14.2 The DPDP Act 2023 + DPDP Rules 2025 — the harder constraint

For an India-based training run, **data protection is a bigger operational constraint than copyright.**

- **You are a Data Fiduciary.** You determine purpose and means of processing training data. Obligations attach even when data comes from third parties or is processed on external infrastructure.
- **Consent-centric.** Free, specific, informed consent per specified purpose; notices must itemise the personal data collected and the exact purpose.
- **Erasure obligation.** Data must be erased when consent is withdrawn or the purpose is served. **This implies the ability to selectively remove data from training pipelines and logs** — and, arguably, to handle machine unlearning. Requests must be addressed within **90 days** (Rule 14).
- **Section 3(c)(ii)** excludes personal data the data principal made publicly available — but **the scope is contested.** MeitY stated in the Rajya Sabha (Aug 2024) that scraping publicly available user data remains subject to the IT Act, IT Rules and DPDP Act including consent and transparency obligations. That sits uneasily with the plain text. **IAMAI formally petitioned MeitY in August 2025 seeking an explicit AI-training exemption under Section 17(5).** As of this writing, no such blanket exemption has been notified.
- **Section 17(2)(b)** conditionally exempts research processing — but exemptions are **purpose-based**, and commercial model training is not obviously research.
- **Children's data (Section 9 / Rule 10):** verifiable parental consent required for under-18s; **no tracking, profiling, or behavioural targeting of children.** For a model trained on educational content this is directly relevant.
- **Cross-border transfer (Section 16 / Rule 15):** transfers permitted to notified countries. Relevant if you train on non-Indian cloud.

**Operational requirements this imposes on your pipeline:**

```
INGESTION GATE
 ├─ PII detection & redaction (names, phones, emails, addresses)
 ├─ India-specific ID pattern detection (Aadhaar, PAN, passport, voter ID, bank)
 ├─ Children's-data detection & exclusion
 ├─ Provenance record per document (source URL, licence, retrieval date, hash)
 └─ Erasure-request index (document hash → shard → checkpoint lineage)
```

The **erasure-request index** is the part teams skip and later regret. Build it on day one; retrofitting provenance into a 15T-token corpus is effectively impossible.

### 14.3 A defensible legal posture (not legal advice)

1. **Tier your corpus by risk** exactly as this document does: build the ✅ clean tier first and prove the model works on it.
2. **Honour robots.txt, `noai`/`noimageai` meta tags, and TDM reservation signals.** Even without a statutory obligation, ignoring them converts an arguable case into a wilful one.
3. **Publish a data statement** — sources, licences, filtering, opt-out mechanism. Dolma and Common Corpus are the models to copy. Transparency is a legal asset, not a liability.
4. **Run and document memorisation testing.** The ANI ruling turned on ANI's failure to prove memorisation. Being able to affirmatively demonstrate *low* memorisation is now a defensive asset with judicial backing.
5. **Maintain a takedown/opt-out channel** and act on it.
6. **Get NCERT/MoE and BharatGen licences in writing.** Two letters could be worth more than a trillion scraped tokens.

---

## 15. How frontier labs actually source data — including what they pay for

You asked for this explicitly. Frontier training data comes from **five** channels, and the industry has shifted markedly since 2024.

### 15.1 The five channels

| Channel | What | Cost | Who uses it |
|---|---|---|---|
| **1. Web crawl** | Common Crawl + proprietary crawlers | Infrastructure only | Everyone. Still the bulk by volume |
| **2. Open datasets** | FineWeb, Nemotron, Stack v2, Dolma, HPLT | Free | Everyone; open labs disclose, closed labs don't |
| **3. Licensed content** | Publisher, image, video, forum deals | **$5M–$250M per deal** | OpenAI (most), Meta, Google, Amazon, Microsoft, Apple, Mistral, Perplexity |
| **4. Human-generated / vendor data** | Annotation, expert demonstrations, preference labels, red-teaming | Very large, undisclosed | All frontier labs. Scale AI, Surge, Turing, Invisible, Mercor, Handshake |
| **5. Synthetic** | Model-generated rephrasing, translation, reasoning traces, RL rollouts | Compute | **Increasingly dominant.** Nemotron-CC is 30% synthetic; BhashaKritika is 100% synthetic |

### 15.2 Publicly reported licensing deals (all figures **reported**, not confirmed by parties)

| Buyer | Seller | Reported value | Type |
|---|---|---|---|
| **OpenAI** | News Corp (WSJ, The Times, NY Post, The Australian) | **~$250M over 5 years** — largest known deal | Training + display |
| **Meta** | News Corp | **Up to ~$50M/year over 3 years** | — |
| **Reddit** | Google + OpenAI | **$203M aggregate contract value disclosed at IPO**; ~$60M/yr Google, ~$70M/yr OpenAI | API / retrieval |
| **Amazon** | The New York Times | **~$20–25M/year** | First Amazon AI content deal |
| **OpenAI** | Dotdash Meredith | ~$16M/year | — |
| **OpenAI** | Axel Springer (Business Insider, Politico) | ~$13M/year over 3 years | — |
| **OpenAI** | Financial Times | ~$5–10M/year | — |
| **Microsoft** | Informa | ~$10M+ initially | Specialist content |
| **Various** | **Shutterstock** | **$138M total AI licensing revenue in 2024** (up from $104M in 2023); individual Big Tech deals **$25–50M each** | Images |
| **Various** | Wiley | **$40M+ across two deals** | Academic |
| **Mistral** | AFP | Undisclosed | **Explicitly grounding only, NOT training** |
| **Meta** | Reuters, CNN, Fox News, USA Today, People Inc. | Undisclosed | Real-time answers |
| **Apple** | Condé Nast, NBC News, IAC (reported approach), Shutterstock | $50M+ sought | Training |
| **Runway** | Getty Images | Undisclosed | Video/images |
| **Anthropic** | — | **No comparable publisher licensing programme.** In 2025 agreed to a proposed ~$1.5B settlement in a books-related class action | Notable outlier |

**Also relevant to OpenAI's other partners:** AP, Le Monde, Prisa, The Atlantic, Vox Media, TIME, Condé Nast, Hearst, Axios, Guardian Media Group, Grupo Folha, Grupo UOL, and a Disney/Sora arrangement.

### 15.3 Three structural trends you should copy

1. **The market shifted from "training" to "live access."** Attribution/live-access deals: **2 (2023) → 11 (2024) → 18 (2025) → ~34 projected (2026)**. The Washington Post's April 2025 OpenAI deal and Le Monde's Perplexity deal **explicitly exclude training** — content can be surfaced and attributed but not learned. **Implication for you:** for Indian news, negotiate a *retrieval/grounding* deal rather than a training deal. It is far cheaper, publishers accept it more readily, and for a conversational model it delivers most of the value.

2. **No single publisher has dominant leverage.** Even the largest sellers have single-digit deal counts (Shutterstock 7, Wikimedia 6, Reddit 5) while "other" publishers collectively account for 36 deals. **Implication:** you do not need Times of India. You need forty regional-language publishers, and each of them individually has very little leverage and considerable incentive. The Indian regional press — Dainik Jagran, Malayala Manorama, Eenadu, Daily Thanthi, Sakal, Anandabazar Patrika and the hundreds beneath them — is exactly the fragmented, monetisation-hungry, linguistically-irreplaceable long tail this pattern predicts you can afford.

3. **The total market is small relative to compute.** AI training-data licensing is estimated at **~$4B globally in 2026**. Against a 300B-model training budget, a ₹5–20 crore ($600K–$2.4M) Indic content licensing programme is a rounding error that buys you the one thing money can buy in this project: **exclusive, high-quality, natural Indic text that no competitor has.**

### 15.4 What this means for your "no paid data" constraint

You said you would consider paid data only if it is "super small money." Here is the ranked list by value-per-rupee:

| Rank | Acquisition | Estimated cost | Why |
|---|---|---|---|
| **1** | **NCERT / MoE commercial-training licence** | **₹0 (a letter)** | Unlocks the entire K-12 curriculum in 36 languages. Highest ROI action available |
| **2** | **BharatGen / Bharat Data Sagar MoU** | **₹0–small** | ~20T tokens claimed, publicly funded. Public-interest framing helps |
| **3** | **AIKosh contributor status** | ₹0 | Contribute your derived datasets, gain access + goodwill + compute credits |
| **4** | **Regional-language publisher archives** (20–40 publishers) | **₹3–15 crore total** | Natural, edited, high-register Indic text at a scale nothing else provides |
| **5** | **Commission Tier-3 language collection** (Bodo, Dogri, Santali, Manipuri) | **₹2–5 crore** | Nothing else will fix these four languages. Also excellent public positioning |
| **6** | **Private Indic evaluation set** (~3,000 native items × 22 languages) | **₹15–30 lakh** | Your only uncontaminated measuring instrument |
| **7** | **Human preference/red-team annotation in 22 languages** | ₹2–8 crore | Safety alignment does not transfer across languages (§11.4) |
| **8** | Commercial Indic speech vendors (e.g. Shaip and similar) | Varies | Only if open speech corpora prove insufficient — they probably will not |

Items 1–3 are **free** and worth more than everything else combined. Items 4–7 total roughly **₹8–30 crore** — genuinely small against a 300B training run, and they are the only line items that buy something a competitor cannot simply download.

---

## 16. The reference token budget — building 5T / 10T / 15T / 20T

All figures are **post-global-dedup unique tokens**, estimated with a fertility-optimised Indic tokenizer. Assume **40–70% loss** at global dedup for web-derived sources.

### 16.1 The 15T recommended mix (primary recommendation)

| Tier | Source | Raw tokens | After dedup | Share |
|---|---|---|---|---|
| **English web (HQ)** | Nemotron-CC-v2 + v2.1 | 9.1T | 4.5T | 30.0% |
| **English educational** | FineWeb-Edu + FinePDFs-Edu | 1.65T | 1.4T | 9.3% |
| **English PDFs** | FinePDFs | 3.0T | 1.6T | 10.7% |
| **Code** | Stack v2 permissive + Nemotron-Code-v2/v3 + CC-Code | 1.5T | 1.3T | 8.7% |
| **Math / STEM** | Nemotron-CC-Math + MegaMath + OpenWebMath + Proof-Pile-2 + peS2o | 0.6T | 0.5T | 3.3% |
| **Indic natural** | Sangraha verified+unverified (88B) + IndicCorp v2 + FineWeb-2 Indic + HPLT-3 Indic + MADLAD Indic + Indic Wikipedia/FineWiki + government/legal/PIB + **ASR transcripts** | ~0.45T | **0.30T** | **2.0%** |
| **Indic synthetic** | Your BhashaKritika-style generation + translated FineWeb-Edu + translated Nemotron Diverse-QA + curriculum generation | 2.4T | 2.2T | 14.7% |
| **Indic parallel / transliteration** | BPCC + Samanantar + Aksharantar + FineTranslations Indic slice | 0.35T | 0.30T | 2.0% |
| **Multilingual non-Indic** | FineWeb-2 + HPLT 3.0 (top 30 languages) | 2.5T | 1.6T | 10.7% |
| **FineTranslations (en side)** | Cultural knowledge in English | 1.0T | 0.7T | 4.7% |
| **Curated / clean-provenance** | Common Pile + Common Corpus + Gutenberg + OpenStax | 0.5T | 0.35T | 2.3% |
| **Reasoning / agentic / instruction (mid-train)** | OpenThoughts, Nemotron SFT-in-PT, tool-use traces, IndicAlign | 0.3T | 0.25T | 1.7% |
| **TOTAL** | | **~23.4T** | **~15.0T** | 100% |

**Indic total: ~2.8T tokens = 18.7%** — of which only 0.30T (2.0%) is naturally-occurring. That ratio is uncomfortable and it is also *unavoidable*. It is broadly consistent with what Sarvam and BharatGen have disclosed.

### 16.2 The milestone ladder

**5T — "Prove the recipe" (weeks 1–8)**
FineWeb-Edu (1.3T) + Nemotron-CC-v2 HQ subset (1.5T) + Stack v2 permissive (0.7T) + Nemotron-CC-Math (0.13T) + Sangraha full (0.25T) + BPCC/Samanantar/Aksharantar (0.05T) + FineWeb-2 Indic (0.04T) + multilingual (0.8T) + Common Pile (0.25T). **Everything here is ✅ green-licence.** Train a 7–15B dense proxy on this before touching the 300B.

**10T — "Safe production floor" (weeks 8–20)**
Add: full Nemotron-CC-v2 + v2.1, FinePDFs, FineWeb-2-HQ, full Stack v2 + Nemotron-Code, MegaMath + full math tier, HPLT 3.0 multilingual, 0.8T of your own Indic synthetic, ASR transcription of all 35K hours of Indic speech. **Still fully licence-clean if BhashaKritika is replicated rather than ingested.**

**15T — "Recommended target" (weeks 20–40)**
Add: 2.4T Indic synthetic (scaled generation), FineTranslations both sides, curriculum-generation tier across 22 languages × 5 difficulty levels, licensed regional-publisher archives, translated Nemotron Diverse-QA into 22 Indic languages.

**20T — "Stretch, phase 2"**
Requires: OCR of India's print archive at scale (fine-tuned olmOCR for Indic scripts), a BharatGen/Bharat Data Sagar data agreement, 2 epochs on the highest-quality 30% of the mix, and materially expanded synthetic generation. **Do not plan the first training run around 20T.** Plan to *reach* it in a continued-pretraining phase where the marginal token is cheap and the model can already generate its own high-quality Indic data.

### 16.3 Curriculum ordering (the "child to PhD" schedule)

Do not shuffle uniformly. Use a phased mixture that shifts across training:

| Phase | % of tokens | Mixture emphasis | Purpose |
|---|---|---|---|
| **Phase 1 — Foundations** | 0–40% | High web diversity, broad multilingual, Indic script exposure early and heavy | Language, script, and world coverage. **Front-load Indic here** — early exposure sets tokenizer-embedding quality for low-resource scripts |
| **Phase 2 — Knowledge** | 40–75% | FineWeb-Edu, FinePDFs-Edu, Wikipedia, academic, code | Compression of knowledge, code fluency |
| **Phase 3 — Reasoning** | 75–92% | Math, code, long chains, MIND-style dialogues, synthetic QA, curriculum-graded Indic | Reasoning depth |
| **Phase 4 — Anneal / mid-train** | 92–100% | Highest-quality only: FineWeb-Edu top decile, Nemotron HQ, curated Indic, instruction-formatted data, long-context, tool-use traces | LR decay to zero on the best data. **This is where you buy benchmark performance.** |

Then post-training: SFT → preference optimisation (DPO/GRPO) → multi-environment RLVR (NeMo Gym pattern, 20+ environments including 22-language safety).

---

## 17. The pipeline — what you actually have to build

```
┌─ ACQUISITION ────────────────────────────────────────────────┐
│ CC crawls · HF datasets · AIKosh/ULCA · eCourts · PIB        │
│ Licensed publisher feeds · Speech corpora · PDF/print scans  │
│ robots.txt + noai + TDM-reservation honoured at fetch time   │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ EXTRACTION ─────────────────────────────────────────────────┐
│ HTML→text (trafilatura/resiliparse) · PDF (olmOCR, Indic FT) │
│ ASR (IndicASR/Whisper-Indic FT) · Image-text interleaving     │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ LANGUAGE & SCRIPT ID ───────────────────────────────────────┐
│ GlotLID (2000+ labels, script-aware) ensemble                 │
│ Devanagari/Bengali/Tamil/Telugu/Kannada/Malayalam/Gurmukhi/   │
│ Gujarati/Odia/Perso-Arabic/Meitei Mayek/Ol Chiki + Latin      │
│ Code-mix & romanisation detection (Hinglish is a first-class  │
│ language here, not noise)                                     │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ QUALITY FILTERING ──────────────────────────────────────────┐
│ Per-language heuristics (fork FineWeb-2 configs + Setu)      │
│ KenLM perplexity vs Sangraha-verified reference              │
│ Model-based classifier (FineWeb-Edu-style, trained per-lang) │
│ n-gram repetition · doc length · stopword ratio              │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ SAFETY & COMPLIANCE GATE ───────────────────────────────────┐
│ PII redaction · Aadhaar/PAN/passport/bank pattern detection  │
│ CSAM hash matching (PhotoDNA-equivalent) on all image data   │
│ Toxicity (Common Corpus multilingual classifier + Indic FT)  │
│ Children's-data exclusion · Licence tag propagation          │
│ Provenance record: URL, licence, hash, retrieval date        │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ DEDUPLICATION ──────────────────────────────────────────────┐
│ Exact (SHA) → Global MinHash+LSH (fuzzy, cross-corpus)       │
│ → SemDeDup (embedding-based) on the top-quality tier         │
│ THIS IS THE STEP THAT DETERMINES YOUR REAL TOKEN COUNT       │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ BENCHMARK DECONTAMINATION ──────────────────────────────────┐
│ 13-gram overlap + MinHash vs BENCHMARK_BLOCKLIST             │
│ Every drop logged with the benchmark it matched              │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ SYNTHESIS ──────────────────────────────────────────────────┐
│ Translation (IndicTrans2 / Sarvam-Translate)                 │
│ Transliteration (IndicXlit) · Rephrasing (Nemotron 5 prompts)│
│ Curriculum generation (22 langs × 5 difficulty tiers)        │
│ Doc-grounded / persona / topic / math-grounded (BhashaKritika)│
│ → RE-ENTERS quality + dedup + decontamination gates          │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ MIXING & TOKENIZATION ──────────────────────────────────────┐
│ Custom tokenizer, fertility target 1.4–2.1 across all scripts│
│ Temperature-sampled multilingual mixture (α ≈ 0.3)           │
│ Phase-scheduled curriculum mixture (§16.3)                   │
└──────────────────────────────────────────────────────────────┘
```

**Tools you should adopt rather than rebuild:**

| Need | Tool | Licence |
|---|---|---|
| Indic-specific cleaning/dedup at scale | **Setu** (AI4Bharat, Spark-based) | MIT ✅ |
| General-purpose data processing | **datatrove** (Hugging Face) | Apache-2.0 ✅ |
| Curation at GPU scale | **NeMo Curator** (NVIDIA) | Apache-2.0 ✅ |
| Language ID | **GlotLID** | Open ✅ |
| OCR | **olmOCR / olmOCR-2** (Allen Institute for AI) | Apache-2.0 ✅ |
| Translation | **IndicTrans2** (AI4Bharat, all 22 languages) | MIT ✅ |
| Transliteration | **IndicXlit** (AI4Bharat) | MIT ✅ |
| RL environments | **NeMo Gym** (NVIDIA) | Open ✅ |
| Eval harness | **lm-evaluation-harness** (EleutherAI) — MILU ships a config | Apache-2.0 ✅ |

---

## 18. Risks, gaps, and honest unknowns

### 18.1 Technical risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Global dedup collapses 23T → 9T, not 15T** | High | Run dedup on a 10% sample *first* and extrapolate before committing to a budget. This is the single most likely schedule-breaker |
| **Synthetic data collapse** — training on translated/generated Indic degrades diversity | High | Cap synthetic at ~50% of the Indic tier; enforce KenLM-perplexity and n-gram diversity floors; keep natural data in every phase; monitor per-language output entropy |
| **Tokenizer fertility undermines everything** | High | Dedicated workstream, target 1.4–2.1 fertility. Test on all 22 scripts including Ol Chiki and Meitei Mayek before freezing |
| **Tier-3 languages (Bodo, Dogri, Santali, Manipuri) remain unusable** | High | Accept and communicate honestly, or commission collection. Do not publish inflated per-language claims |
| **Agentic capability requires infrastructure you have not budgeted** | High | Per-repo containers, 100-turn rollouts, harness co-development. Start the NeMo Gym work in parallel with data, not after |
| **Benchmark contamination** | Medium | §12.3, plus a fully held-out benchmark and a private eval set |
| **Indic OCR quality gates the print archive** | Medium | Fine-tune olmOCR per script; this is a 2–3 month project with very high payoff |
| **MoE routing collapses across 22 languages** | Medium | BharatGen reported MoE layers naturally grouping related languages (Hindi/Marathi). Use shared + routed experts (Param-2 uses 64 experts with 2 shared for code-mixing/domain-mixing) and monitor per-language expert utilisation from step one |

### 18.2 Legal and reputational risks

| Risk | Mitigation |
|---|---|
| ANI ruling reversed on appeal | Keep the ✅ clean tier separable so you can retrain a compliant base if needed. Do not architect a corpus you cannot decompose |
| DPDP erasure request against training data | Provenance index from day one; document-hash→shard→checkpoint lineage |
| NC-licensed content leaks into the mix | Automated licence-tag propagation; **hard-fail** the build on any NC tag, not a warning |
| Synthetic-data licence chain (Gemma/Llama/Qwen terms) | Prefer Apache-2.0 generators (Qwen3, some Nemotron) for anything that will be redistributed |
| CSAM in image data | PhotoDNA-equivalent hash matching on every image; never ingest original LAION-5B |
| Model memorises and regurgitates Indic news | Run memorisation testing and publish results — this is now an affirmative legal asset in India |

### 18.3 Honest unknowns

- **The 20T Bharat Data Sagar figure is unaudited.** "Tokens" across a 3PB multimodal store is a soft unit. Treat as directional.
- **EKA corpus licence string is unspecified.** Get it in writing.
- **Sarvam's and BharatGen's actual data mixtures are undisclosed.** We know token counts (12T, 16T, 22T) and rough Indic shares (15–25%, "over one-third Indian content") but not compositions.
- **Whether model weights are a derivative work of CC BY-SA training data is unresolved anywhere in the world.**
- **Real deduplicated Indic natural-text volume** has never been measured across all sources jointly. My 250–500B estimate is an inference from Sangraha, FineWeb-2, IndicCorp and HPLT figures, not a measurement. **Measuring it properly would itself be a publishable contribution and should be your week-one task.**

---

## 19. Recommended next actions, in order

1. **Week 1:** Send the NCERT/MoE licence request and open BharatGen MoU discussions. Zero cost, longest lead time, highest value.
2. **Week 1–2:** Measure the real deduplicated Indic pool. Ingest Sangraha + IndicCorp v2 + FineWeb-2 Indic + HPLT 3.0 Indic + MADLAD Indic, run global MinHash, and publish the number internally. Everything downstream depends on it.
3. **Week 2–4:** Build the ✅ clean-tier corpus (Common Pile + Common Corpus + IndicCorp v2 + MADLAD + Stack v2 permissive + FineWeb-Edu). ~4T tokens, zero legal risk. Train a 1.5B ablation model.
4. **Week 3–6:** Tokenizer workstream. Target fertility 1.4–2.1 across all 22 scripts.
5. **Week 4–8:** ASR-transcribe all 35K hours of open Indic speech. Nobody else has done this at scale and it produces text that exists nowhere else.
6. **Week 4–10:** Fine-tune olmOCR for Indic scripts. Unlocks the print archive.
7. **Week 6–12:** Stand up NeMo Gym with SWE-Gym/R2E-Gym/SWE-smith, plus the Indic-translated-issue variant (§5.3). Agentic infrastructure has the longest lead time.
8. **Week 8–16:** Replicate BhashaKritika methodology under your own licence; scale toward 2.4T Indic synthetic tokens.
9. **Week 10–14:** Commission the private 22-language evaluation set (~3,000 items).
10. **Week 12+:** Begin regional-publisher licensing conversations, prioritising retrieval/grounding deals over training deals.

---

## 20. Source index

**Indic corpora & tooling**
- AI4Bharat: https://ai4bharat.iitm.ac.in/ · https://huggingface.co/ai4bharat
- IndicLLMSuite / Sangraha / Setu: https://github.com/AI4Bharat/IndicLLMSuite · arXiv:2403.06350
- IndicCorp v2: https://huggingface.co/datasets/ai4bharat/IndicCorpV2
- Indic NLP Catalog (the best community index): https://ai4bharat.github.io/indicnlp_catalog/ · https://github.com/AI4Bharat/indicnlp_catalog
- BhashaSutra (task-centric survey of Indian NLP datasets, 2026): arXiv:2604.18423
- BhashaKritika: https://huggingface.co/datasets/krutrim-ai-labs/BhashaKritika · arXiv:2511.10338
- Chitrakshara: arXiv:2603.23521 · Chitrarth: arXiv:2502.15392 · https://github.com/ola-krutrim/Chitrarth
- UPDESH: arXiv:2509.21294

**India government / national infrastructure**
- AIKosh: https://aikosh.indiaai.gov.in/ · EOI: https://aikosh.indiaai.gov.in/static/datasets_EOI.pdf
- IndiaAI: https://indiaai.gov.in/
- Bhashini ULCA: https://github.com/bhashini-dibd/ulca
- BharatGen: https://bharatgen.com/ · Bharat Data Sagar: https://bharatgen.com/products/bharat-data-sagar/
- Open Justice India (eCourts data): https://openjustice-in.github.io/
- DIKSHA licensing: https://ciet.ncert.gov.in/initiative/diksha
- SWAYAM/NPTEL: https://swayam.gov.in/nc_details/NPTEL

**Speech**
- IndicVoices: arXiv:2403.01926 · https://huggingface.co/datasets/ai4bharat/IndicVoices
- IndicVoices-R: https://huggingface.co/datasets/ai4bharat/indicvoices_r
- Project Vaani (IISc + Google): https://vaani.iisc.ac.in · arXiv:2603.28714
- SYSPIN: https://vaani.iisc.ac.in/dataset/syspindataset
- SPRING-INX: https://asr.iitm.ac.in/ · ESPnet recipe: https://github.com/espnet/espnet/tree/master/egs2/spring_speech

**Global corpora**
- Hugging Face FineData (FineWeb, FineWeb-2, FinePDFs, FineWiki, FineTranslations): https://huggingface.co/HuggingFaceFW
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://github.com/huggingface/fineweb-2
- FineWeb2-HQ: https://huggingface.co/datasets/epfml/FineWeb2-HQ
- NVIDIA Nemotron datasets: https://huggingface.co/nvidia · arXiv:2412.02595, 2508.14444, 2508.15096, 2512.20848
- The Stack v2 / StarCoder2: https://huggingface.co/datasets/bigcode/the-stack-v2 · arXiv:2402.19173 · https://huggingface.co/blog/starcoder2
- Common Pile v0.1: arXiv:2506.05209 · https://blog.eleuther.ai/common-pile/
- Common Corpus (Pleias): https://huggingface.co/datasets/PleIAs/common_corpus
- HPLT: https://hplt-project.org/datasets · arXiv:2503.10267 (v2), arXiv:2511.01066 (v3.0)
- MADLAD-400: https://huggingface.co/datasets/allenai/MADLAD-400
- CulturaX: https://huggingface.co/datasets/uonlp/CulturaX

**Agentic / RL**
- NeMo Gym: https://docs.nvidia.com/nemo/gym/ · https://huggingface.co/collections/nvidia/nemo-gym
- R2E-Gym: arXiv:2504.07164 · Open-AgentRL: https://github.com/Gen-Verse/Open-AgentRL
- CWM: arXiv:2510.02387

**Benchmarks**
- MILU: https://github.com/AI4Bharat/MILU · arXiv:2411.02538
- IndicMMLU-Pro: arXiv:2501.15747 · BhashaBench V1: arXiv:2510.25409
- ParamBench: arXiv:2508.16185 · IndicParam: arXiv:2512.00333
- IndicIFEval: arXiv:2602.22125 · IndicVisionBench: arXiv:2511.04727
- BharatBench: https://ai-labs.olakrutrim.com/

**Legal / policy**
- ANI v. OpenAI judgment coverage (24 July 2026): Business Standard, LawBeat, The New Publishing Standard
- Kluwer Copyright Blog on Indian TDM: https://legalblogs.wolterskluwer.com/copyright-blog/
- DPDP Act 2023 + DPDP Rules 2025: MeitY; Future of Privacy Forum analysis: https://fpf.org/blog/five-ways-in-which-the-dpdpa-could-shape-the-development-of-ai-in-india/
- AI licensing deal trackers: Media & the Machine (Rob Kelly), LLM Pulse, Troveo

---

## 21. One-paragraph summary

**The largest assemblable, legally-defensible corpus including Indian languages and English is approximately 15 trillion unique tokens, of which around 2.8T (≈19%) will be Indic — and of that Indic portion, only about 0.3T will be naturally-occurring human-written text, because that is genuinely all that exists.** The rest must be manufactured through translation, transliteration, LLM-grounded synthesis, ASR transcription of India's 35,000 open hours of speech, and OCR of its print archive. The binding constraints are not compute or model architecture; they are **tokenizer fertility, the NonCommercial licences on India's own educational corpus, agentic-RL infrastructure, and the absence of any uncontaminated Indic evaluation set.** Three free actions — an NCERT licence request, a BharatGen data MoU, and AIKosh contributor status — are worth more than any amount of additional scraping, and should be initiated in week one.

---

*Compiled 28 July 2026. Figures marked [VENDOR CLAIM] are unaudited. Licence characterisations are research summaries, not legal opinions; obtain counsel review before any dataset enters a production training mix.*

---
---

# ADDENDUM A — Recency sweep, 28 July 2026

*Added after a targeted re-check of arXiv (via live query, not search-engine index), Indian AI press, lab model cards, and government statements. This addendum contains **material corrections** to the main document. Where it conflicts with Sections 1–21, **this addendum wins.***

## A.0 What I checked, and what I could not

**Checked directly and successfully:**
- arXiv listings queried live, sorted by announcement date, across four query angles. Coverage confirmed current through **27 July 2026** (yesterday). Abstracts pulled directly from arXiv abstract pages.
- Hugging Face model cards for `sarvamai/sarvam-105b`, `sarvam-30b`, and FP8/GGUF variants — architecture and licence read from the cards themselves.
- Indian government statements: Lok Sabha written replies (22 July 2026), DST/PIB releases, IndiaAI Mission material.
- Indian tech press: Business Standard, Tribune, MediaNama, Forbes India, Analytics India Magazine, Open Source For You.
- AI4Bharat project pages, GitHub repos, and their public X/Twitter announcement thread (surfaced via search index).

**Could not verify directly — treat with corresponding caution:**
- **X/Twitter in real time.** I have no direct X access. X content reached me only where a search engine had indexed it. **Anything announced on X in the last ~24–72 hours may be invisible to me.**
- **"Published today" (28 July 2026) specifically.** arXiv's daily announcement cycle plus search-index lag means the last ~12–24 hours are a genuine blind spot. The newest paper I confirmed is dated 27 July.
- **Private/Discord/closed-community announcements**, embargoed releases, and anything not yet on a public URL.
- **Whether any of the arXiv datasets below have actually been uploaded yet.** Several say "we will release" — that is a promise, not a download.

**On the request to be "100% confident":** I will not claim that, and you should distrust anyone who does on a $1B decision. What I can tell you precisely is *which* claims are load-bearing and how each is grounded — see §A.7. Every number below is attributed to a primary source you can independently verify in under a minute.

---

## A.1 🔴 MATERIAL CORRECTION #1 — Sarvam-105B is Apache 2.0, with full architecture disclosed

The main document treated Sarvam's models as reference points. **They are more than that: they are a licence-free starting position.**

**Confirmed from the Hugging Face model cards (`sarvamai/sarvam-105b`, `sarvamai/sarvam-30b`) — both Apache 2.0:**

| | **Sarvam-105B** | **Sarvam-30B** |
|---|---|---|
| Active params | **10.3B** | 2.4B (non-embedding) |
| Experts | **128, top-8 routing** | **128, top-6 routing** |
| Shared experts | **1** | 1 |
| Attention | **MLA-style**, decoupled QK: q_head_dim 192 (RoPE + noPE split), v_head_dim 128, head_dim **576**, hidden_size 4096 | Grouped KV (num_key_value_heads=4), 19 layers |
| FFN | intermediate 16384 / moe_intermediate 2048 | intermediate 8192 / moe_intermediate 1024 |
| Long context | **YaRN, scaling factor 40, 128K** | rope_theta **8e6**, no RoPE scaling |
| Router | **Auxiliary-loss-free balancing**, routed scaling factor 2.5 | Same |
| Pre-training | 12T tokens | 16T tokens |
| Licence | **Apache 2.0** | **Apache 2.0** |
| Access | HF, AIKosh, Sarvam API, Indus app | Same |
| Positioning | "particular strength in agentic tasks, mathematics, and coding" | "reliable coding ability, best-in-class conversational quality across Indian languages"; handles multilingual voice calls **while performing tool calls** |

Official blog: https://www.sarvam.ai/blogs/sarvam-30b-105b · Public rollout 6 March 2026.

**Why this changes the investment case.** Your plan is a 300B MoE. Sarvam-105B is a 105B MoE, Apache 2.0, trained on 12T tokens with an Indic-optimised tokenizer, already strong on agentic/coding, with the exact architectural pattern (128 experts, shared expert, aux-loss-free balancing, MLA) you would independently arrive at. Three options now exist that did not exist in the main document's framing:

1. **Upcycle.** MoE upcycling from a 105B Apache-2.0 checkpoint to ~300B is a well-trodden path and can cut pre-training compute by a large multiple.
2. **Distil.** Use Sarvam-105B as an Indic teacher for synthetic generation with **zero licence-chain contamination** — this is the clean alternative to BhashaKritika (§3.4) that the main document said you would have to build yourself.
3. **Benchmark honestly.** Any 300B you build must beat a free 105B on Indic tasks. That is now the bar, and it is public.

**Action:** before committing capital, run a 2-week evaluation of Sarvam-105B on your target task set. If it is within ~10% of your target on Indic tasks, the case for training a 300B from scratch rests entirely on coding/agentic/English headroom — not on Indic capability. Size that gap before you size the cheque.

---

## A.2 🔴 MATERIAL CORRECTION #2 — Do not use olmOCR for Devanagari

The main document recommended fine-tuning olmOCR for Indic scripts as a "highest-ROI engineering project." **A benchmark published 28 June 2026 shows that is the wrong base model.**

**"Can OCR-VLMs Read Devanagari? A Stress-Test Benchmark and Post-Correction Study" (arXiv:2606.29213)** benchmarked ten systems on Hindi/Devanagari across four synthetic degradations and 300 real printed scans:

- On **clean rendered text**, all ten cluster at chrF++ 91–98 — **synthetic text does not separate systems at all.** Any vendor benchmark using rendered text is meaningless.
- On **real scans, nine of ten collapse.** EasyOCR falls from 93.6 → **58.3**. The field spreads across a **76-point range**.
- **Strong English OCR does not predict Indic OCR.** GPT-5.5 drops to **58.5** (tying classical EasyOCR). **olmOCR-7B falls to 40.5** — the worst-in-class result, from the model behind olmOCR-Bench.
- **The winner among open models is Qwen3-VL-8B at 75.2**, which beats GPT-5.5 and **runs on a single 24 GB GPU**.
- DeepSeek-OCR has the **best median of any system** but suffers rare catastrophic repetition failures (outputs up to 71× reference length) that destroy its mean.

**Revised recommendation:** build your Indic OCR pipeline on **Qwen3-VL-8B**, report **median and catastrophic-failure-rate, never mean**, and validate exclusively on **real scans**. This correction is worth months of misdirected effort.

---

## A.3 🟢 The tokenizer finding is now quantified — and it validates the main document's #1 priority

**"The Tokenizer Tax" (arXiv:2607.24276, 27 July 2026)** — measured on FLORES-200 across six tokenizers and fourteen Indian languages:

- Under **cl100k_base** (GPT-3.5/GPT-4): Indian languages pay an **average 8.0× tokenization tax** vs English, reaching **13.0× for Malayalam**.
- This **reduces the effective context window to as little as 12%** of what English users get for equivalent semantic content.
- **Mechanism identified:** failed byte-pair merges leaving text fragmented into single-byte tokens. Merge failure correlates with tax at **Pearson r = 0.89**.
- **This is not a property of Indic scripts — it is a tokenizer design failure.** XLM-R and o200k_base reduce the average Indic tax by **73%**.

Companion work, **"BHARATI" (arXiv:2607.23319, 25 July 2026)**: morphology-aware SentencePiece BPE tokenizers on a 781MB balanced corpus across seven languages, handling Sanskrit/Tamil agglutination and sandhi. Achieves **2.6 tokens per Indian Knowledge System technical term vs 5.25 for GPT-2** and ~90% sequence-length reduction vs GPT-2/byte-level on a released 490-sentence IKS test set (measurement script published).

**Net effect on your plan:** the main document's claim that "your tokenizer decision is worth more than 2T extra tokens" is now backed by a measured 8× penalty with an identified, remediable mechanism. **Elevate this from a workstream to a gating milestone.** Nothing else should start until fertility is validated across all 22 scripts.

---

## A.4 New datasets and benchmarks published in the last 30 days

| Date | Artefact | What | Relevance |
|---|---|---|---|
| **25 Jul 2026** | **IndicTalk** (arXiv:2607.23242) | **1,328,604 event-grounded multi-turn conversations, 18 language varieties, 9 Indic languages.** Code-mixed in **both native-script and Romanized** forms. Built via news-grounding + persona-conditioned generation + automatic quality validation | **Directly serves your #1 stated task (conversation).** Nothing else at this scale exists for Indic code-mixed dialogue. ⚠️ Paper says "we will release" — **verify the data is actually up before planning around it** |
| **26 Jul 2026** | **Indic DiarBench** (arXiv:2607.23808) | **~108 hours**, **all 22 scheduled languages**, human-corrected time-aligned **speaker-attributed** transcriptions. Near-field meetings, far-field, in-the-wild. Captures code-mixing, dialect, speaker overlap | First joint diarization+ASR benchmark covering all 22. Open access. Add to your EVAL suite |
| **25 Jul 2026** | **BHARATI** (arXiv:2607.23319) | Morphology-aware tokenizers, 7 languages, + released measurement script | See §A.3 |
| **27 Jul 2026** | **The Tokenizer Tax** (arXiv:2607.24276) | Quantified Indic tokenizer penalty | See §A.3 |
| **28 Jun 2026** | **Devanagari OCR-VLM stress test** (arXiv:2606.29213) | 10-system benchmark, 300 real scans | See §A.2 |
| **27 Jun 2026** | **Conversational IndicTrans2 adaptation** (arXiv:2606.29024) | Adapts IndicTrans2-1B to conversational register across **all 21 Indic languages** using **only public data** (OpenSubtitles, BPCC-H-Daily, Tatoeba), via experience replay + model souping. **+6.2 mean conversational chrF, no FLORES regression** | Directly reusable recipe. **Note their honesty**: a blind human + multi-model check did **not** confirm perceived quality improvement — they call it register matching, not better translation. Adopt the technique *and* the scepticism |
| **29 Jun 2026** | **SCRIBE** (arXiv:2605.20712) | Diagnostic evaluation + rich transcription models for Indic ASR | Useful for the ASR-transcription pipeline in §8 |
| **28 May 2026** | **IndicKLAR** (arXiv:2605.29637) | Cross-lingual knowledge consistency, code-mixed vs native. Across nine open-weight models, **native-language accuracy gap to English reaches ~0.50**, but **code-mixed inputs close most of it — to within ~0.05 of English** with no model-level intervention | **Strategically important.** Suggests code-mixed input is a far more efficient path to Indic parity than native-script-only training. Should influence your data mixture |
| **6 Jun 2026** | **AgriGov** (arXiv:2606.08272) | Structured multilingual dataset, Indian government farmer schemes | Domain-specific; matches BharatGen's agriculture vertical |
| **16 Jun 2026** | **Darshana Graph** (arXiv:2606.18222) | Parallel commentary corpus, comparative Indian philosophy | Rare classical/IKS content |
| **Apr 2026** | **BhashaSutra** (arXiv:2604.18423) | Task-centric unified survey of Indian NLP datasets | **Use as your master index.** Most complete catalogue in existence |

---

## A.5 Corrections to figures in the main document

| Main doc said | Correct figure | Source |
|---|---|---|
| "~35,000+ hours of open Indic speech" | **BhasaAnuvaad alone is ~44,400 hours** + 17M aligned text segments, 13–14 Indian languages + English, **CC BY 4.0**. Total open Indic speech across BhasaAnuvaad + IndicVoices + Vaani + SPRING-INX + Shrutilipi + Kathbath + SYSPIN + Rasa is **well north of 100,000 hours** | arXiv:2411.04699 · github.com/AI4Bharat/BhasaAnuvaad |
| BPCC "largest public Indic parallel corpus" (unquantified) | **~230 million bitext pairs**, all 22 scheduled languages | ai4bharat.iitm.ac.in/areas/nmt |
| Samanantar "~50M pairs" | **49.7M pairs**, 11 Indic languages, **37.4M newly mined** | ai4bharat.iitm.ac.in/areas/nmt |
| BPCC licence "CC-BY-4.0" | More precise: **BPCC-Mined, NLLB-Seed, ILCI, MASSIVE and BPCC-BT are released under CC0** ("no rights reserved") — AI4Bharat waived all rights to the packaging. Even cleaner than stated | github.com/ai4bharat/IndicTrans2 |
| IndicTrans2 as latest MT | **IndicTrans3-beta** now exists with **document-level MT**; **IndicSeamless** (SeamlessM4T-v2 fine-tuned on BhasaAnuvaad) is SOTA for Indic speech-to-text translation, beating cascaded approaches on FLEURS and BhasaAnuvaad-test | AI4Bharat HF org |
| TTS coverage understated | **Rasa** — 500+ hours expressive studio speech, 20 speakers, 13 languages. **IndicParler-TTS** — first open TTS for **all 22** languages, prompt-controlled style. **IndicF5** — 11 languages. **IndicConformer** — ASR for all 22 | AI4Bharat HF org |
| Not mentioned | **FBI** and **CIA** datasets (AI4Bharat) — for building and testing multilingual LLM *evaluators*. Relevant to §12.4's "build a private eval set" recommendation | AI4Bharat HF org |

---

## A.6 The competitive and funding landscape — as stated to Parliament, 22 July 2026

From Union Minister Ashwini Vaishnaw's written Lok Sabha reply (reported by ANI, Business Standard, Tribune, and others):

- **20 indigenous sovereign AI foundation-model proposals** identified for support: **12 LLMs + 8 SLMs**.
- **237 projects** supported; **93.18 lakh (9.318 million) GPU hours** sanctioned; **15 compute service providers** empanelled.
- **27 India Data and AI Labs** established; **58 AI Centres of Excellence** approved across states.
- Named projects: Sarvam AI (30B, 105B), Gnani.ai speech-to-speech, BharatGen multilingual foundation models, Avataar AI video generation.
- The 12 funded organisations (per an earlier MeitY Lok Sabha reply): **Sarvam AI, Soket AI, Gnani AI, Gan AI, Avataar AI, IIT Bombay Consortium (BharatGen), GenLoop, Zenteiq, Intellihealth, Shodh AI, Fractal Analytics, Tech Mahindra Maker's Lab.**

**The number that should most affect your planning:** the government allocated **₹988.6 crore to the IIT Bombay-led BharatGen consortium specifically to develop a 1-trillion-parameter LLM.** BharatGen's own careers page states they are "moving from 10B-class systems to 1 Trillion across language, document/vision, and speech."

**Implication.** Your 300B sits between a free Apache-2.0 105B that already exists and a publicly-funded 1T that is being built. That is a genuinely difficult competitive position on *Indic capability alone*. It is a much stronger position if the differentiator is **agentic coding** — where, on the public evidence, no Indian effort is currently focused and where the Sarvam models, while "optimised for agentic tasks," have not published SWE-bench Verified or Terminal-Bench numbers that would rank them globally.

**A calibrating datapoint from the same week:** industry reporting cites Qwen3.6-27B at **77.2% on SWE-bench Verified**, beating a 397B MoE, running on 18GB VRAM, and matching frontier models on Terminal-Bench. *(I have not independently verified this figure against the SWE-bench leaderboard — treat as secondary reporting.)* If accurate, it means **agentic coding capability is not primarily a scale problem** — it is a data-and-environment problem, which is exactly the argument of §5.2. A 300B trained without a serious RL environment programme will lose to a well-trained 27B.

---

## A.7 What is load-bearing, and how confident I am

For a decision of this size, here is an explicit confidence ledger.

**High confidence — read from primary sources, independently verifiable in minutes:**
- Sarvam-105B/30B architecture, token counts, and **Apache 2.0** licence → the Hugging Face model cards themselves.
- Sangraha 251B/64B/24B/162B split → the IndicLLMSuite paper.
- BPCC CC0 licensing of mined/seed/BT components → the IndicTrans2 repo licence table.
- NCERT CC BY-NC-ND and DIKSHA CC BY-NC-SA → NCERT/CIET's own site.
- BhasaAnuvaad 44,400 hours, CC BY 4.0 → AI4Bharat GitHub + arXiv:2411.04699.
- The four arXiv papers in §A.2–A.4 → abstracts pulled directly from arxiv.org.
- Parliament figures (20 proposals, 237 projects, 93.18 lakh GPU hours) → multiple independent outlets reporting the same written reply.
- The ANI v. OpenAI ruling of 24 July 2026 → multiple independent legal and business outlets.

**Medium confidence — reported but not primary:**
- The ₹988.6 crore / 1-trillion-parameter BharatGen allocation → government press coverage, not a tender document I read.
- Frontier licensing deal values → all explicitly "reported," none confirmed by parties.
- The Qwen3.6-27B SWE-bench figure → single secondary source.

**Low confidence — treat as directional only:**
- **BharatGen's "20 trillion tokens / 3 petabytes"** → a CEO statement, unaudited, and "tokens" is a soft unit for a multimodal store. **Do not build a plan that depends on this being true.**
- **My own estimate that deduplicated natural Indic text totals 250–500B tokens** → an inference from Sangraha, FineWeb-2, IndicCorp and HPLT figures. **It has never been measured.** This is the single most important unverified number in the entire document, and it drives the whole corpus architecture. §19 item 2 — measuring it — should be week one, and it is a two-week job, not a six-month one.

**Known blind spots:**
- The last ~24 hours of arXiv and all real-time X/Twitter.
- Whether IndicTalk and Indic DiarBench data are actually downloadable yet.
- Anything under embargo, in private Discords, or announced at a conference in the last few days.
- Chinese and other non-English-language sources on Indic AI.

**If a $1B decision hinges on any single number here, commission an independent verification of that number.** The right use of this document is as a map of where to look and what to ask, not as a substitute for due diligence.

---

*Addendum A compiled 28 July 2026 via live arXiv query (coverage confirmed through 27 July), Hugging Face model cards, Indian government statements, and Indian technology press. Blind spots and confidence levels stated explicitly in §A.0 and §A.7.*

---
---

# ADDENDUM B — The deepest layer: the project's own priors + today's arXiv announcement
### Compiled 28 July 2026, ~afternoon IST

*Two layers of evidence the main report and Addendum A had not yet used: (1) **the papers and corpus specification inside this project** — LightningLM 0.1V, OPUS, Muennighoff's data-constrained scaling laws, Sardana & Frankle, BrahmicTokenizer-131K, Kronecker Embeddings, MUTANT — which encode hard, production-tested priors; and (2) **the cs.CL announcement of literally today**, Tuesday 28 July 2026 (140 papers, parsed live from arxiv.org's listing page). Every number below is extracted verbatim from a project document or a fetched arXiv abstract. Where this addendum conflicts with earlier sections, it wins.*

---

## B.1 🔴 CORRECTION — The corpus math in §16 was Chinchilla-brained. The project's own scaling-law paper fixes it.

The main report's central scarcity claim — "only ~0.3T naturally-occurring Indic tokens exist, therefore Indic can only be ~2% of the mix" — silently assumed **every token is seen once**. Muennighoff et al., *Scaling Data-Constrained Language Models* (arXiv:2305.16264, **in this project**), is precisely the paper that governs what happens when you can't get more unique data, and its measured results change the arithmetic:

**The measured facts (from the paper itself):**

- **Up to 4 epochs of repeated data is almost as good as fresh data.** Their 8.7B-parameter model trained for 4 epochs on 44B unique tokens finished with only **0.5% higher validation loss** than the single-epoch model on 178B unique tokens.
- **The decay constant R\*_D ≈ 15** — the point where a repeated token has lost 1/e of its value sits around **15 repetitions (16 epochs)**. Meaningful gains continue up to roughly there, then returns collapse.
- **At 40 epochs, repeating is worthless**, and some 44-epoch runs diverge mid-training.
- **Hard ceiling:** no amount of repetition beats a single epoch on **U·(1 + R\*_D) ≈ 16× the unique pool** of fresh tokens.
- **Allocation rule:** when data-constrained, **scale epochs faster than parameters** — excess parameters decay in value faster than repeated data (R\*_N < R\*_D). This is the *opposite* of what naively extending Chinchilla predicts.
- **Code buys headroom:** mixing code data "gives the ability to scale an additional 2×."
- **Caveat that saves you from a mistake:** Hernandez et al. (cited therein) found that **up-sampling just 0.1% of the corpus 100×** significantly degrades performance. Repetition works on *whole tiers*, not on tiny slivers hammered repeatedly.

**What this does to the token budget (§16 revised):**

| Quantity | Main-report framing | Corrected framing |
|---|---|---|
| Budget unit | "15T **unique** tokens" | **Unique pool × deliberate epoch schedule = effective tokens** |
| Natural Indic (~0.3T unique) | 2.0% of mix, immovable | **×4 epochs ≈ 1.2T effective at ~zero quality cost**; theoretical ceiling ~4.8T effective (16×). Indic-natural effective share rises from 2% to **~7–8% without collecting a single new token** |
| FineWeb-Edu / curated-clean tier | Seen once | 2–4 epochs is cheap and standard practice for the highest-quality tier |
| The 40–70% dedup loss (§18.1 risk) | Schedule-breaker | **Half-defused.** Dedup down to the unique pool, then *choose* repetition deliberately. Dedup ≠ see-once |
| 20T milestone | "Aggressive multi-epoch reuse" (hand-waved) | Now **quantified and principled**: 20T processed ≈ 11–13T unique + a scheduled epoch plan concentrated on the scarce, high-value tiers |

**Two honest limits.** (1) Muennighoff's experiments are English web data (C4-family) at ≤9B parameters and ≤900B tokens; **nobody has measured R\*_D for Indic text, for translated text, or for synthetic text.** Synthetic data plausibly has a *lower* repetition ceiling (less latent diversity per token). This is now the second-most-important unmeasured number after the total-unique-Indic-pool measurement of §A.7 — and it is a one-week ablation at 1.5B scale. (2) Sardana & Frankle (arXiv:2401.00448, **in this project**) supplies the complementary correction on the other axis: once expected *inference* demand is accounted for, the optimum shifts toward **smaller models trained on more tokens** than Chinchilla prescribes — which is the formal justification for the report's already-asserted "train well past compute-optimal" stance, and, for an India-population-scale deployment, it pushes hard toward maximizing active-parameter efficiency (MoE with small active count) rather than total parameters. *(Cited here at thesis level; the paper is in the project for the exact accounting.)*

---

## B.2 🔴 REVISION — Dynamic data selection: the production prior from LightningLM's own run

The main report's pipeline (§17) treated the mixture as static and phase-scheduled. The course's Session 5 material and the LightningLM 0.1V technical report (arXiv:2606.07404, **in this project**) contain a rare artifact: **a production post-mortem of running a per-iteration data selector (OPUS) at scale.** The findings should govern your architecture:

**The corpus it was run on (the closest existing blueprint to your project):**

LightningLM V4: **1,118B tokens across 33,353 shards** (~**1,254B effective** after selection dynamics), in three tiers —

| Tier | Pool | Size | Role |
|---|---|---|---|
| OPUS-eligible | D1 Web Foundation | 164.1B | Main pretraining; OPUS scores candidates, keeps ~40% |
| | D2 Web Diverse | 627.4B | |
| | D3 Code | 199.0B | |
| | D4 STEM | 49.1B | |
| **Always-ON** | AON | **78.1B** | **8% of every batch, invisible to the selector** — protects Indic data and benchmark *training* splits |
| Golden Proxy | GP | 6.8M (11 shards) | **Never trained on.** Supplies only the direction the selector steers toward |
| Dropped | B2 | 31.3B | **Cut for contamination** |

**Why the Always-ON tier exists — the single most important design fact in the whole project:** the golden proxy is **English-heavy (cosine 0.876 with the English web band)**, so OPUS **systematically under-values Indic data and would reject it** if allowed to govern everything. The report's stated principle: *"A single selector should not govern data whose value it cannot see."*

**The selector's real economics at scale:**

- OPUS was **active at the 5B stage only** — disabled at 2B (easy-token regime; candidates differed too little in utility to be worth scoring) and disabled again from 9B onward.
- At 5B, measured throughput was **30,429 wall-clock tok/s vs 36,737 ordinary tok/s** — a **~17% wall-clock tax** — *despite* amortizing each scoring pass over 10 training steps and implementing Ghost extraction, CountSketch projection (m=8192), and the per-step preconditioner as **custom kernels**.
- The originating OPUS paper reports **4.7% end-to-end overhead** — but in synchronous data-parallel training **without ZeRO-style sharding**. The integration cost with sharded parameters is where the 4.7% became ~17%. (For calibration: naive per-candidate gradient selection was **3.5× slower** — 6,875 min vs 1,985 for random — which Ghost+CountSketch reduced to 2,083 min in the native setting. OPUS defaults: buffer b=64, temperature τ=0.9, sketch m=8192.)
- The efficiency claim that motivates OPUS — **8× compute reduction on GPT-2 XL / FineWeb** — did not transfer cheaply into this production stack.

**What changes in the report:**

1. §16.3's static phase-scheduled mixture is **confirmed as the production backbone** — this is not a simplification, it is what the only in-project 120B run actually shipped with.
2. **Add an Always-ON lane to §17's pipeline: ~8% of every batch reserved for Tier-2/3 Indic scripts + agentic/world-model traces + benchmark training splits, bypassing any selector or quality classifier.** Your quality classifiers (FineWeb-Edu-style) are trained on English-proxy notions of quality and will exhibit *exactly* the same bias OPUS did. This lane is the mechanism that guarantees Bodo, Santali, Dogri and Manipuri actually train at every stage.
3. Dynamic selection is **ablation-stage tooling, not the production backbone** at 300B — and never with an English-heavy reference proxy. If you use one, build an *Indic-inclusive* golden proxy first.
4. The contamination discipline gets a precedent: LightningLM **dropped 31.3B tokens (a full pool) for contamination** and kept the proxy strictly held-out. §12.3's rules now have an existence proof at 120B scale.

---

## B.3 Tokenizer — the report's #1 priority now has two concrete, in-project recipes

Addendum A elevated the tokenizer to a gating milestone on the strength of the Tokenizer Tax paper (8.0× average Indic penalty under cl100k_base, 13.0× Malayalam, merge-failure r=0.89 — **formally announced in today's listing**, arXiv:2607.24276). The project contains the two candidate solutions, already built and audited:

**Recipe 1 — Retrofit (BrahmicTokenizer-131K, arXiv:2605.29379, in-project).** A byte-level BPE that retrofits **o200k_base** via a two-stage procedure (script-prune crop + surgical installation of Brahmic vocabulary), closing the Indic compression gap **at the 131K-vocab class while preserving English and code compression**. Its headline compression number is measured on a **27M-document public Indic corpus** whose sources (AI4Bharat + Sarvam-AI public releases) overlap the 1.045B-token audit corpus used to score surgery candidates — the authors themselves label that in-distribution and corroborate out-of-distribution on **FLORES-200 and IN22-Gen** (identical sentence sets across all 22 languages). It deliberately **declines language-aware pre-tokenization** to stay drop-in compatible and protect English/code metrics. It is the tokenizer used across the entire LightningLM family, paired with **Kronecker embeddings** (arXiv:2605.29459, in-project) which replace the ~**537M-parameter** embedding table at that vocab/width with a **33.55M-parameter** projection — negligible at 120B, but *most of a quarter of the model* at the 1.78B seed, which matters enormously for a staged-growth plan (§B.4).

**Recipe 2 — From-scratch (MUTANT-Indic, arXiv:2511.03237, in-project).** A **200K shared-vocabulary** SuperBPE-based tokenizer trained on a 10GB sample drawn from ~50GB of curated multilingual data (OLMo, Wikipedia, books, PDFs, Common Crawl, **Sangraha**; code from **Stack v2**; fastText LID → MinHash dedup → NFKC normalization → script validation, FineWeb-style).

**Decision framing for the 300B:** Retrofit preserves the English/code frontier and ecosystem compatibility and is the demonstrated-in-production choice; from-scratch at 200K (the Sarvam path — fertility 1.4–2.1) maximizes Indic fertility at the cost of re-deriving English/code performance. The Tokenizer Tax paper's finding that **o200k already removes 73% of the Indic tax vs cl100k** argues the retrofit path starts closer to the goal than the main report assumed. Either way, the gate stands: **fertility validated on all 22 scripts (including Ol Chiki and Meitei Mayek) before any other workstream commits.**

---

## B.4 The growth prior — the strongest challenge to "train 300B from scratch" in the entire evidence base

The LightningLM report's core demonstration: a **120B sparse MoE (460 routed experts, top-12 routing, 5.93B active ≈ 5% of 118.67B stored)** was **grown in four stages from a 1.78B dense seed** (→5B →9B →120B), trained end-to-end on **a single eight-GPU node**, early stages at 4K context and 9B+ at 8K — the course's stated capstone economics being ~**67 days and ~$100K-class without spot pricing** for the V4 run. Its reference list also documents the adjacent path (Grove-MoE upcycling Qwen3-30B-A3B-Base into a 33B MoE).

Combined with Addendum A's finding that **Sarvam-105B is Apache 2.0 with its full MoE recipe on the model card**, the capital question is now sharply posed:

| Path to 300B-class | Evidence base | Relative pre-training cost |
|---|---|---|
| From random init | Industry default | 1.0× (baseline) |
| **Upcycle/grow from an open checkpoint** (Sarvam-105B, or a staged seed) | LightningLM 4-stage growth (in-project, at 120B); Grove-MoE; the entire dense-to-MoE upcycling literature | **Large multiple cheaper**; inherits tokenizer + Indic competence |
| Buy capability via data quality + RL environments at smaller active size | Muennighoff allocation rule (§B.1); Qwen3.6-27B ≈ SWE-bench frontier reports (§A.6, secondary) | Cheapest per benchmark point |

For calibration of what "frontier corpus" means today (from the Session-3 notes, in-project): **Llama 4 trained on >30T tokens; Qwen3 on 36T across 119 languages.** The 15T target of §16 is respectable but not frontier-scale — which is precisely why the effective-token machinery of §B.1 and the growth machinery here matter more than raw collection.

**Revised recommendation:** the $1B question is not "which 300B do we train" but "**scratch vs grow**" — and the burden of proof now sits on scratch. Insert, before any capital commitment: a 4-week head-to-head at ~2B scale of (a) from-scratch on your mix vs (b) grown/upcycled, on identical data, judged on Indic + code held-out loss.

---

## B.5 Announced TODAY — Tuesday 28 July 2026 (cs.CL, 140 papers; abstracts fetched, not inferred from titles)

| arXiv | Paper | Why it matters to this plan |
|---|---|---|
| **2607.24717** | **DataOrchestra: Learning to Orchestrate Per-Example Curation of Pretraining Data** | The successor research line to OPUS-style selection — but for **per-example *processing***: an orchestrator decides drop / keep / clean per chunk, then selects operations from programmatic edits to LLM rewriting **with a generated instruction per rewrite**. Validated by from-scratch pretrains **0.5B–7B**, stable average gains across 11 benchmarks; also effective for math continued-pretraining. Read before finalizing §17 — it is the strongest current argument that the *cleaning* stage (not the selection stage) is where per-example adaptivity pays |
| **2607.23322** | **IKS-Instruct** — **24,795 instruction pairs**, 7 languages (en/hi/sa/ta/te/kn/ml), **41 Vedic pedagogical techniques**, **CBSE classes 6–12 aligned**; sources: Bhagavad Gita, Thirukkural, Sangam literature, Vedic texts, Vedic-math sutra demonstrations, bilingual pairs, multi-turn dialogues | Directly fills **two gaps the report flagged**: BhashaBench's weakest domain was IKS-adjacent (Ayurveda, GPT-4o 59.74%), and §10.1's proposed workaround — generate curriculum-aligned content from syllabus *structure* — has now been executed by someone, CBSE-aligned. **Licence unverified — check before ingestion** |
| **2607.24030** | **MoLGE: Mixture of Language-Group Experts** for massively multilingual ASR — dedicated experts per *cluster* of similar languages + hierarchical LoRA; studies linguistic vs data-driven grouping criteria | **Design evidence for your MoE routing.** Converges from the ASR side with BharatGen's observation that MoE layers spontaneously grouped Hindi/Marathi, and with Param-2's shared+routed expert split. Language-*group* experts (not per-language) is emerging as the pattern for 22-language capacity without dilution |
| **2607.24720** | **The Physics of Multi-Turn Long-Horizon Planning** — controlled environment isolating how planning ability is acquired across pre- and post-training | Three findings that should shape your agentic data (§5): (1) **explicit world-model construction via CoT state-transition data in *pre-training*** yields stronger long-horizon generalization; (2) atomic skills alone don't compose — **a little long-horizon data works**; (3) **suboptimal trajectories severely impair performance** because errors amplify over horizons — filter agentic traces *hard*, don't maximize volume |
| **2607.23545** | **XIH-Bench / Language Shapes Instruction-Hierarchy Compliance** — 6 languages, same- and cross-language instruction conflicts | Sharpens §11.4 with a new, named failure mode: the **"Language Boundary Effect"** (cross-language conflicts get *higher* compliance than same-language) and the finding that **language specialization makes lower-priority instructions in model-favored languages harder to override**. Translation for your threat model: **an Indic-expert model acquires a specific new jailbreak surface — injections written in its own favored languages.** Add cross-language-injection red-teaming to the 22-language safety plan |
| **2607.22859** | **PatiGonit22K** — 22,441 Bengali math word problems, simple through multi-operation, translated/annotated/culturally adapted/verified | 🟡 **Corrects the report:** §6 said Indic math/STEM is "essentially absent." Amend to *nearly* absent — this exists, though it is 4 orders of magnitude below MegaMath. The generate-and-translate strategy of §6 stands |
| 2607.24515 | Systematic analysis of LLM + transformer MT for English↔Tamil across diverse data | Add to the §7 translation evidence base |
| 2607.24604 / 2607.24223 | Agentic code repair via typed revision contracts / relevance-guided agentic corpus search | The agentic-coding literature is compounding *daily* — reinforces §5.2's conclusion that this is the live front |

*(Note on dates: the four papers in Addendum A dated 25–27 July by submission — IndicTalk, BHARATI, Indic DiarBench, Tokenizer Tax — appear in **today's formal announcement**, consistent with arXiv's weekend cycle.)*

---

## B.6 Consolidated delta — what the deepest pass changes in the report

| Section | Change |
|---|---|
| §1.2 / §16 | Re-budget in **unique + epoch-schedule → effective tokens**. Natural-Indic effective share: 2% → **7–8%** via ≤4-epoch repetition (Muennighoff), at measured ~zero loss cost. 20T milestone becomes principled, not aggressive |
| §16.3 / §17 | **Add an Always-ON lane (~8% of every batch)** for Tier-2/3 Indic + agentic/world-model traces, bypassing all selectors/classifiers — the LightningLM mechanism, with its measured justification (proxy cosine 0.876 → selector starves Indic) |
| §17 | Dynamic selection demoted to ablation tooling (OPUS ~17% wall-clock tax at 5B in a sharded stack, disabled either side of it); **DataOrchestra (today)** to be evaluated for the *cleaning* stage instead |
| §5.2 | Agentic pre-training data should include **CoT world-model / state-transition traces**; **filter suboptimal trajectories aggressively** (today's 2607.24720) |
| §6 | "Essentially absent" → "nearly absent" (PatiGonit22K, today) |
| §10.1 | IKS-Instruct (today) is a live instance of the recommended curriculum-generation strategy — evaluate its licence, and its 41-technique taxonomy as a generation scaffold |
| §11.4 / safety | Add **cross-language instruction-injection** red-teaming — the Language Boundary Effect (today's 2607.23545) is a threat model specific to Indic-specialized models |
| §A.3 / tokenizer | Two in-project candidate recipes now on the table: **retrofit o200k (BrahmicTokenizer-131K + Kronecker)** vs **from-scratch 200K (MUTANT / Sarvam-class)**; gate milestone unchanged |
| §19 / capital plan | Insert two pre-capital gates: (i) **scratch-vs-grow head-to-head at ~2B**, (ii) **Indic repetition-value ablation** (measure R\*_D for natural/translated/synthetic Indic at 1.5B — one week, second-most-important unknown in the plan) |

## B.7 Grounding statement

Everything in B.1–B.4 is extracted from documents **inside this project** (2305.16264, 2606.07404, 2602.05400, 2605.29379, 2605.29459, 2511.03237, plus local reference material) via project-knowledge retrieval — quotes and figures are verbatim from those retrievals. Everything in B.5 is from **today's cs.CL announcement**, parsed live from arxiv.org, with all six headline abstracts fetched directly before characterization. Sardana & Frankle is cited at thesis level only and flagged as such. Still unverified, unchanged from A.7: real-time X/Twitter; whether IndicTalk / IKS-Instruct data files are actually uploaded; R\*_D for Indic/synthetic text (nobody has measured it — you should be first).

*End of Addendum B.*
