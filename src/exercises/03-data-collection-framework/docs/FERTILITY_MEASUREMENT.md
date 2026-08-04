# Fertility Measurement Protocol
### Task 2.2b — resolves Blocked B2. Produces the only `measured` numbers on the site.

**Runtime: half a day.** The tokenization itself is minutes; the work is comparator setup and normalization discipline. Do not let the setup slide — a normalization mistake silently invalidates every number downstream.

---

## 1. The comparator set

| Tokenizer | Vocab | Source | Why it's in the set |
|---|---|---|---|
| **Gemma 4** | **262,144** | HF `google/gemma-4-31b` | ★ The assignment's named target. Everything is measured against this |
| Sarvam-105B | ~200K | HF `sarvamai/sarvam-105b` (Apache 2.0) | The Indic-optimised comparator; claims fertility 1.4–2.1 |
| o200k_base | ~200K | `tiktoken` | The retrofit base for BrahmicTokenizer-131K |
| cl100k_base | ~100K | `tiktoken` | The Tokenizer Tax paper's baseline — keeps your numbers comparable to published work |
| XLM-R | 250K | HF `xlm-roberta-base` | The paper's "removes 73% of the tax" reference point |
| Your candidate | 208,896 | trained | The proposal under test |

⚠️ **Gemma models are gated on Hugging Face.** Accept the licence terms before the run or the fetch fails silently in CI. Do this first.

---

## 2. The corpora

| Corpus | Role | Coverage | Why |
|---|---|---|---|
| **IN22-Gen** | ★ Primary | **All 22 scheduled languages, n-way parallel** | **Source-original Indian content** — not translated from English. Same semantic content across languages, so tokens/word is directly comparable |
| **IN22-Conv** | Primary | Same 22 | Conversational register. Your #1 stated task is conversation, and fertility differs between registers |
| FLORES-200 devtest | Secondary | 14 Indian languages | Comparability with the Tokenizer Tax paper. Translated-from-English, so expect translationese |
| Stack v2 permissive sample | Technical | Code | Report chars/token and tokens/line, not tokens/word |
| LaTeX sample (Proof-Pile-2) | Technical | Math | Verify digits stay atomic and frequent commands stay single-token |

**The reason IN-22 leads and FLORES follows:** FLORES is translated *from* English, so it measures how a tokenizer handles translationese. IN22-Gen is source-original. If the two diverge for a language, that divergence is itself a finding worth reporting.

---

## 3. Metrics

Per language × tokenizer × corpus:

```
tokens_per_word   = total_tokens / whitespace_delimited_words
tokens_per_char   = total_tokens / unicode_codepoints      # word-boundary agnostic
expansion_ratio   = tokens / tokens_english_same_sentence  # n-way parallel makes this exact
```

**Report mean, median, and P95 for every slice.** Mean alone hides the catastrophic cases, and catastrophic cases are what break a 256K context.

**The headline metric:**

```
parity_ratio = max(fertility over Tier-A Indic) / fertility(English)
```

Target ≤ 1.5. Under cl100k it sits near 8.0.

`tokens_per_char` matters because whitespace word-counting is unfair to agglutinative languages — Malayalam and Tamil pack more morphemes per whitespace-delimited token. Reporting both prevents an artefact of the metric being mistaken for a property of the script.

---

## 4. Normalization discipline

Get these wrong and the numbers are noise:

- [ ] **Fix NFC** across all corpora and record it. NFC vs NFKC changes Indic results measurably. Do not mix.
- [ ] **Strip special tokens** — exclude BOS/EOS/pad from all counts.
- [ ] **Consistent leading whitespace.** Some tokenizers prepend a space marker (`▁`, `Ġ`). Encode identically for every comparator or the comparison is invalid.
- [ ] **Validate script per language** with GlotLID before counting. A mislabelled row corrupts one language's entire column.
- [ ] **Dual-script languages get two rows, not one:** Kashmiri (Perso-Arabic + Devanagari), Sindhi (Perso-Arabic + Devanagari), Manipuri (Bengali + Meitei Mayek), Santali (Ol Chiki). Most published fertility studies silently pick one and don't say which.

---

## 5. Output

`records/fertility.json` — every value provenance-typed:

```json
{
  "run_id": "fert-2026-08-04-a",
  "corpus_version": "IN22-Gen@<sha> + IN22-Conv@<sha> + FLORES-200-devtest",
  "normalization": "NFC, special tokens excluded, leading-space normalized",
  "measured_at": "2026-08-04T...",
  "results": [
    { "language": "mal", "script": "Mlym", "tokenizer": "gemma-4@262144",
      "corpus": "IN22-Gen",
      "tokens_per_word": { "mean": null, "median": null, "p95": null,
                           "unit": "tok/word", "provenance": "measured",
                           "source": "run:fert-2026-08-04-a" },
      "tokens_per_char": { "...": "..." },
      "expansion_vs_english": { "...": "..." },
      "sample_count": 1024 }
  ],
  "parity_ratio": { "value": null, "unit": "ratio",
                    "provenance": "measured", "source": "run:fert-2026-08-04-a" }
}
```

Every field stamps corpus version, normalization rules, sample count and run date — that's what makes `provenance: "measured"` defensible rather than decorative.

---

## 6. What makes this a contribution, not a replication

Three things, and they're worth one sentence each in the report:

1. **Measured against Gemma 4's actual 262,144 tokenizer** — the assignment's named comparator. Published work benchmarks cl100k, o200k and XLM-R. Nobody has published Gemma-4 Indic fertility.
2. **All 22 scheduled languages on source-original content.** The Tokenizer Tax paper covers 14 on FLORES (translated). IN22-Gen extends to 22 and removes translationese.
3. **Dual-script rows and P95, not just mean.** Both are standard omissions. Reporting them is cheap and immediately distinguishes the work.

---

## 7. Watch for

| Risk | Handling |
|---|---|
| Gemma gated on HF | Accept terms before the run; fail loudly, not silently |
| Bodo, Dogri, Konkani may be absent from FLORES-200 | IN-22 covers all 22 — that's why it leads |
| Tier-3 languages have tiny samples | Report `sample_count`; flag anything under 500 sentences as low-confidence |
| Dependency spread (`tiktoken` + `transformers` + `sentencepiece`) | All permissive, all fine on the Python side. `web/` stays zero-dependency |
| Result contradicts the estimate | **Publish it.** A measured number that overturns your own estimate is the most credible thing on the site — put it in `#changelog` |

---

## 8. Consequences for the TODO

- **B1 resolved** — `d_model = 6,144` confirmed. Ship as the default; keep the live input so a reviewer can test robustness.
- **B2 resolved** — this protocol runs this week. Task 2.2 fertility values ship `provenance: "measured"`.
- **Task 2.3 (vocab sweep) becomes empirical.** The peak is now derived from measured per-language fertility rather than an estimated curve. Re-run the sweep after this lands.
- **INV-4 is now satisfiable rather than aspirational** — fertility values will carry a real `tokenizer_ref` and a real run ID.
- **Report §6.3 headline changes** from *"an 8.0× tax is reported in the literature"* to *"we measured Gemma 4 at N tok/word on Malayalam against M on English."* That is a different sentence, and it is the sentence the submission should lead with.

**Do this before Phase 2 completes.** It converts the technical spine of the submission from reasoning to evidence.
