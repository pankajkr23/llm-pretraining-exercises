# 04 · Decision record

Decisions the code cannot explain about itself. Each was a fork where a different choice was
defensible; the reasoning is here so a later reader can disagree with the reasoning rather than
guess at it.

The assignment these answer is `BRIEF.md`, which is a local working file — briefs are input for
the people and agents building an exercise, not public reading.

## D1 · How many strategies? Eight — and the source material names two different eights

There is no line in the source material that says "there are N strategies", so the count has to be derived.
Two lists exist, and both have exactly eight members:

| # | The pipeline map (§2, labelled "STAGE 1…8") | The closing commitments (§14) |
|---|---|---|
| 1 | Extract | *(absent)* |
| 2 | Normalize | normalization |
| — | *(never given a stage number)* | **format discipline** |
| 3 | Language ID | language validation |
| 4 | Quality filter | quality filtering |
| 5 | Deduplicate | deduplication |
| 6 | PII scrub | PII removal |
| 7 | Decontaminate | decontamination |
| 8 | Manifest | the manifest |

§14 drops **Extract**, and §2 says why: it was covered earlier, so it is taken as known and the
time goes to the stages V4 omitted altogether. In its place
§14 adds **format discipline**, the ghost-tag trap, which §2's map never numbers at all despite the
topic giving it a full section (§4) and its own widget.

So: **the answer is 8**, and the *membership* depends on whether you count the pipeline or the
commitments. The **union is 9**. Two independent counts corroborate the headline — the
`clean_text()` widget exposes exactly **8** cleaning operations, and the quality cascade has
**9** Gopher/C4 rules.

This is a real reading result, not a quibble, and the page opens with it: the reader toggles
between the two lists and watches Extract slide out as format discipline slides in.

## D2 · The example link is a *model*, not a dataset

`lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled` is a **model** repo
(`repoType: "model"`, `gated: false`, Apache-2.0) — a reasoning-distilled Qwen3.6-35B-A3B taught to
imitate Claude Opus 4.7's chain-of-thought. An early check of ours reported it missing; that check
queried the *datasets* API namespace and was simply wrong. The model card names the datasets
behind it:

- `lordx64/reasoning-distill-claude-opus-4-7-max` — raw teacher traces, pre-SFT formatting
- `lordx64/reasoning-distill-opus-4-7-max-sft` — the same traces reformatted into SFT conversations
- **~7,800 full conversations**, assistant side trained *including* `<think>…</think>`

So "a 10-100M dataset **like this**" means **a reasoning-distillation corpus carrying thinking
traces**. That is a precise target rather than a guess, and it is why corpus A below is what it is.

## D3 · We count tokens with our own tokenizer, and we never estimate

The obvious approach — pick a fertility ratio, multiply by word count — is wrong, and we started
down it before catching the problem.

Fertility is a property of *a tokenizer*, not of a corpus, and the spread across tokenizers is
enormous. From our own measurements in `03-data-collection-framework/records/fertility.json`:

| lang | ours (10k) | cl100k | o200k | gemma-4-31b | sarvam-105b | xlm-r |
|---|---|---|---|---|---|---|
| `hi` | 2.11 | 5.36 | 1.76 | 1.52 | 1.52 | 1.56 |
| `as` | 6.61 | 9.33 | 2.97 | 2.87 | 2.87 | 3.12 |
| `mni` | 7.17 | 16.50 | 16.50 | 12.18 | 2.19 | 2.15 |
| `en` | 2.04 | 1.35 | 1.32 | 1.38 | 1.38 | 1.50 |

Manipuri swings **7.6×** depending on which tokenizer is asked. Kashmiri and Konkani were never
measured at all. Quoting any single ratio silently smuggles in a tokenizer choice and presents it
as a fact about the data — exactly what `AGENTS.md` §*Reporting a measurement* forbids.

**So we tokenize.** The primary tokenizer is **ours**: `02-tokenization/web/tokenizer.json`, the
10,000-token BPE vocabulary submitted for Exercise 02. That is the right choice on the merits, not
just for continuity — it is the tokenizer this project would actually pretrain with, so *"how many
tokens does this corpus give **us**"* is the operationally correct question. A count under someone
else's 256k multilingual vocabulary answers a question we are not asking.

The same corpus is also counted under the five reference tokenizers, and the spread ships as a
finding: **"90M tokens" is not a fact about a corpus, it is a fact about a corpus and a tokenizer.**

## D4 · The corpus is the one our tokenizer can read

Measuring our S2 tokenizer against FLORES-200 dev (parallel ground truth, already on disk from
exercise 03) decided the corpus selection:

| language | script | tok/word | `[UNK]` |
|---|---|---|---|
| English | Latin | 2.04 | 0.0% |
| Hindi | Devanagari | 2.11 | 0.0% |
| Bhojpuri / Awadhi / Magahi | Devanagari | 2.30–2.39 | ~0.1% |
| Maithili | Devanagari | 2.52 | 0.0% |
| Nepali / Marathi / Sanskrit | Devanagari | 3.40–3.74 | 0–0.6% |
| Telugu | Telugu | 3.75 | 0.3% |
| **Assamese / Bengali** | Bengali | 6.61 / 6.66 | **82%** |
| **Manipuri** | Bengali | 7.17 | **84.0%** |

Exercise 02 trained on English, Hindi, Telugu and Maithili. Every Devanagari language and Telugu
therefore tokenize at 0–0.6% `[UNK]`; Bengali script does not exist in the vocabulary.

An earlier draft of this exercise chose Sangraha's **Assamese** shard, on the strength of a genuinely
good narrative — the source material names Sangraha as the corpus that got zero deduplication, and
`verified/asm` is the shape of the taught language-ID bug. That draft was wrong: a token count that
is 82% `[UNK]` is not a token count. **Assamese and Manipuri are kept as an out-of-vocabulary
probe** — deliberately *excluded* from the token budget and used only to produce the 84% figure,
which is a headline in its own right: **your vocabulary decides which data you can even use.**

The joiner-preservation branch survives the switch and is better for it: Telugu's FLORES dev file
alone carries **273 ZWNJ**, Nepali 16 ZWNJ / 35 ZWJ, Marathi 39 ZWJ. And language ID gets *harder*,
which is the point — Hindi, Maithili, Dogri, Bodo, Konkani, Bhojpuri, Awadhi, Magahi, Nepali,
Marathi and Sanskrit all share **one script**, so script detection is useless and a real
discriminator is required.

## D5 · Three corpora, because no single one exercises eight stages

Each corpus earns its place by making a stage fire that the others cannot. All are ungated
(`gated: false`, no `hf auth login`) and permissively licensed.

| | corpus | licence | ~tokens | the stage it alone exercises |
|---|---|---|---|---|
| **A** | `open-thoughts/OpenThoughts-114k` | Apache-2.0 | ~30M | **format discipline** — the only corpus with chat structure and `<think>` traces |
| **B** | `ai4bharat/sangraha` (Devanagari + Telugu, config `verified`) | CC-BY-4.0 | ~30M | **joiner preservation**, **language ID**, and the zero-dedup narrative |
| **C** | `HuggingFaceH4/stack-exchange-preferences` | CC-BY-SA-4.0 | ~30M | **PII**, and quality heuristics that actually cut |

Notes that matter:

- **A is the answer to D2** — a public reasoning-distillation corpus of the same shape as the
  mentor's example. Its lesson is that these datasets store **structured role objects**, not
  pre-rendered strings: there is no literal `<|im_start|>` or `[USER]` in the raw column. **Ghost
  tags are not found, they are *created* by naive rendering.** Stage 2b renders one conversation
  through four templates and counts the token waste.
- **B's card claims nothing about cleaning we would be contradicting.** A grep of the full
  10,925-character card for `dedup`, `PII`, `personal` and `anonym` returns **zero hits**. The
  source material's line — *"Sangraha, our Indic web crawl, had ZERO deduplication"* — stands unchallenged
  by the card. `verified/doi/data-0.parquet` row 0 is plain English, a real instance of the taught
  "the folder lied" bug found in a public corpus.
- **C is chosen for its false positives as much as its true ones.** One row group carries 98 real
  emails and 1,139 IPv4 literals — and also `2.6.21.7`, which is a Linux kernel version, and
  `10737418240`, which is a byte count. A PII chapter without false positives teaches that regexes
  work. C4 was rejected for the opposite reason: it is *defined* by having the Gopher/C4 heuristics
  already applied, so our quality stage would find nothing to cut.

## D6 · What may be published — two separate rules

These were conflated in an early draft, which made both harder to follow.

**PII: nothing real, ever.** Every interactive PII demo, on the page and in the notebook, runs on a
**hand-written synthetic document** — invented emails, phone numbers, names, an IP — which the
reader can edit freely. From the real corpus we publish **aggregate counts only**. Any corpus text
that appears anywhere is **post-scrub**, already `[EMAIL]`/`[PHONE]`. A test re-runs the full PII
scanner over every byte of `web/` and over the notebook and fails on any match. Notebook outputs
are stripped before commit for the same reason: executing a PII cell would otherwise bake real
addresses into a tracked file.

**Corpus excerpts: bounded, and only where seeing is the point.** The deduplication chapter is
unconvincing without two genuinely near-identical documents on screen — the assignment asks us to
*show* the dedup working. So: at most **12 excerpts of at most 300 characters**, emitted by
`export.py` from post-scrub text, never hand-written, each recorded with its `doc_id` and
re-derived by a test.

**Why this differs from exercise 03**, which forbids source content in `web/` absolutely: there,
the content *was* the contamination — publishing eval text would poison the thing the corpus is
measured against. Here the content is CC-licensed web text whose entire purpose in the deliverable
is to be looked at. The relaxation is bounded, attributed in `NOTICE`, and test-enforced. It is a
different rule for a different reason, not drift.

## D7 · Real versus illustrative, declared rather than implied

Two stages cannot be done honestly with the resources here, and both say so on the page rather than
quietly presenting a plausible number:

- **The FineWeb-Edu-style classifier gate** has no model behind it. It is a transparent function of
  the heuristic features, **off by default**, hatched wherever it is drawn, and its output never
  carries `provenance: "measured"`.
- **The PII name layer** is a gazetteer, not NER. No NER model has usable Maithili or Dogri support,
  so importing one would ship hundreds of megabytes of confident garbage. The source material's own widget
  says "behaviour shown via a small known list"; we match it. **No name precision or recall is
  published** — there is no gold set, and inventing one is the same sin as a fake classifier.
  Precision for the *structured* layer is published, hand-verified on 40 documents.

Decontamination is made self-contained by injecting **canary GUIDs** into a held-out slice and
proving the n-gram scan recovers them, so the stage is demonstrable without gated benchmark data.
Where the real benchmark index is absent, coverage reads `none` and the page says **UNCHECKED** —
never "clean".
