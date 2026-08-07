# 02 · Tokenization — a balanced multilingual BPE

Session 2 assignment (see [`BRIEF.md`](./BRIEF.md)). Build **one 10,000-token BPE vocabulary**
shared across India's Wikipedia article in **English, Hindi, Telugu, and a fourth language**, tuned
so all four are tokenized about equally efficiently.

```text
X(language)  = tokens / faithful units
raw score    = 1000 / (X_max − X_min)
penalty      = exp(max(0, X_hi / 1.2 − 1))
final score  = raw score / penalty
```

## Two measurements, kept side by side

This exercise has been measured two different ways, and **both are retained here** — not as old
and new, but as two things that answer different questions:

| | **v1** — our original experiments | **v2** — the graded measurement |
| --- | --- | --- |
| corpus | clipped article prose (`explaintext`) | wiki-faithful Markdown |
| denominator | whitespace **words** | **faithful units** |
| Hindi penalty | none (as designed) | yes |
| languages | en · hi · te · **ta** | en · hi · te · **mai** |
| lives in | `corpus/v1/` | `corpus/v2/` |

**Their scores can never be ranked against each other.** The same tokenizer reads ≈ 2.13 under v1
and ≈ 0.60 under v2 — four times as many atoms in the denominator, on a corpus four times the
size. Every table, the report, and the widget keep them in separate labelled sections, and
`ablate.sweep` raises if you try to sort rows from both at once.

Both run from committed corpora, offline: `uv run python -m tokenization.ablate` prints both
tables.

## v1 — our original experiments (retained, still runnable)

| experiment | spread | score |
| --- | ---: | ---: |
| Unigram · char · NFKC | 0.4813 | **2,077.90** |
| **BPE from scratch · char · NFKC** (no library) | 0.7692 | **1,300.12** |
| char BPE · NFKC (HuggingFace) | 0.8141 | 1,228.34 |
| byte BPE (baseline) | 5.2744 | 189.59 |

**The finding, which still stands:** *representation is the dominant lever*, not corpus weighting.
Byte-level BPE spends its budget rebuilding every Indic character out of three UTF-8 bytes (Tamil
≈ 6.5 tokens/word); char-level + NFKC drops every language to ~1.4–2.2 and cuts the spread ~11×.
Weighting only bites while the vocabulary is scarce — at 2k it moves the score, at 10k it is
inert — and it *over-corrects* at char level, where `balance` drags Telugu to 1.00 and pushes
English and Tamil past 1.9.

These numbers are regenerated from `corpus/v1/` on every run and pinned by
[`tests/test_v1_retained.py`](./tests/test_v1_retained.py). That guard matters because v1 and v2
share an engine: four settings v2 introduced (training from files, `[UNK]`, `min_frequency=1`,
Metaspace `prepend_scheme="never"`) each move v1's numbers, so `ablate._v1` pins all four
explicitly rather than inheriting them.

## v2 — the graded measurement

The submitted tokenizer, on the committed corpus:

| language | units | tokens | X = tokens/unit |
| --- | ---: | ---: | ---: |
| English | 186,367 | 111,875 | 0.6003 |
| **Hindi** | 88,359 | 50,672 | **0.5735** ← best |
| **Telugu** | 36,292 | 24,132 | **0.6649** ← worst |
| Maithili | 5,808 | 3,376 | 0.5813 |

```text
spread      = 0.664940 − 0.573479 = 0.091461
raw score   = 1000 / 0.091461     = 10,933.59
penalty     = 1.000000            (X_hi is nowhere near 1.2)
final score = 10,933.59
```

Reproduce it with `uv run python -m tokenization` — offline, from files in this repo.

## What a "unit" is, and why it is not a word

The denominator is a **faithful unit**: one contiguous run of letters/marks/digits, or one visible
non-space punctuation character. `भारत` is one unit; `](` is two; whitespace is never a unit.

This matters more than it sounds. Counting whitespace-separated *words* instead gives numbers four
times smaller and in a completely different band, so the two are not interchangeable — a fertility
of 0.60 and one of 2.13 can describe the same tokenizer. The report prints word counts alongside
unit counts precisely so that gap is visible rather than confusing.

Fertility below 1.0 is normal here: BPE merges frequent punctuation runs like `](` into single
tokens, so one token can cover two or three units.

The corpus is **wiki-faithful Markdown** — Wikipedia's REST HTML converted with `markdownify`,
keeping links, URLs, tables, references, image links, navboxes and categories. Not clipped prose.
The four snapshots live in [`corpus/`](./corpus/) and are committed, so training, evaluation and
every published number happen offline and identically on a fresh clone.

## The gate: reproducing the reference exactly

Before claiming any improvement, the harness has to prove it measures the same thing the reference
solution measures. `SUITE`'s first row is that recipe — HF BPE · 10k · `min_frequency=1` · NFKC ·
`Metaspace("▁", prepend_scheme="never")` · `[UNK]` · weights 3/4/4/2 — and it reproduces to the
last digit: tokens **111,390 / 51,190 / 24,428 / 4,258**, spread **0.153786**, score **6502.56**.

Getting there turned up the single most important detail in this exercise. HuggingFace splits
*files* into lines, so training from files means **no merge may span a newline**. Hand the same
trainer whole documents instead and it learns cross-line pairs, lowering every token count by
~0.6% and lifting the score to 6771 — same recipe, different number. It is not a rounding
difference and it is not visible in the config; only the exact-match gate catches it.

### Experiments on the graded corpus

All v2 rows share one corpus, so the recipe is the only thing that varies. The first row is the
reference solution reproduced exactly; everything below it is ours.
`uv run python -m tokenization.ablate` → [`artifacts/ablations.json`](./artifacts/).

| experiment | spread | score | total tokens | corpus-wide X |
| --- | ---: | ---: | ---: | ---: |
| more Telugu + Maithili (rejected) | 0.0281 | 35,604 | 192,713 | 0.6083 |
| te ×5 · mai ×6 | 0.0656 | 15,254 | 191,893 | 0.6057 |
| documents · mai ×5 | 0.0910 | 10,985 | 189,910 | 0.5994 |
| **documents · mai ×6 — submitted** | **0.0915** | **10,934** | **190,055** | **0.5999** |
| mai ×6 alone | 0.0957 | 10,445 | 191,446 | 0.6043 |
| Unigram (ablation) | 0.1138 | 8,787 | 207,782 | 0.6558 |
| documents, not lines | 0.1477 | 6,771 | 189,822 | 0.5991 |
| the reference solution (benchmark) | 0.1538 | 6,503 | 191,266 | 0.6037 |
| mai ×10 (overshoot) | 0.1577 | 6,343 | 192,249 | 0.6068 |
| BPE from scratch, no library | 0.1619 | 6,175 | 188,091 | 0.5937 |
| mai ×16 (overshoot) | 0.2353 | 4,251 | 193,299 | 0.6101 |

Two changes to the reference, each justified separately:

- **Train on documents.** Costs nothing and helps everything: fewer tokens for the same text
  *and* a smaller spread. Pure compression, no denominator games.
- **Maithili ×6.** Maithili is 1.8% of the corpus and shared Devanagari with Hindi, so it won
  almost no merges of its own and sat at the worst fertility. Spread is `max − min`, so pulling the
  *maximum* down is the honest direction. The sweep deliberately overshoots: past ×6 Maithili
  becomes the new *minimum* and the spread widens from the other end (×10 → 6,343, ×16 → 4,251).

Composed, they give the submission — the only row that beats the reference on **both** axes at
once, smaller spread *and* fewer total tokens.

### Why not the row that scores 3× higher

The `more Telugu + Maithili` row scores 35,604. We did not submit it, and the reason is measured rather than asserted.

Training and evaluation share the same four files, so corpus weighting is a knob tuned directly
against the test set. `uv run python -m tokenization.holdout` trains on 80% of each article (every
5th line held out) and scores the 20% the trainer never saw:

| config | in-sample | **held-out** | held-out corpus-wide X |
| --- | ---: | ---: | ---: |
| reference recipe | 6,503 | 3,168 | 0.6711 |
| **submission — documents · mai ×6** | 10,934 | **4,213** | **0.6674** |
| mai ×6 alone | 10,445 | 4,096 | 0.6706 |
| more Telugu + Maithili (rejected) | 35,604 | 4,103 | 0.6750 |

That row's 3.3× in-sample lead is worth **nothing** out of sample — it lands *behind* the submission
(4,103 vs 4,213) and compresses worse. That gap is the overfitting, and it is most of the headline.
The honest, transferable gain over the reference is **+33%**, not +68%.

### The fourth language

The reference chose Maithili; we also fetched Tamil with the same pipeline
(`uv run python -m tokenization.fourth_language` → [`artifacts/fourth_language.json`](./artifacts/)).

| set | spread | score | worst language |
| --- | ---: | ---: | --- |
| en/hi/te/**mai** · reference weights | 0.1477 | 6,771 | mai |
| en/hi/te/**mai** · tuned (mai ×6) | 0.0915 | 10,934 | te |
| en/hi/te/**ta** · reference weights | 0.0958 | 10,441 | te |
| en/hi/te/**ta** · tuned (te ×6) | 0.0681 | 14,690 | ta |

The scores are not comparable across sets — they are different corpora. The finding is
*structural*: Maithili's article is 5,808 units in a script Hindi already pays for, while Tamil's
is 188,367 units (larger than English) in a script nothing else uses. Swapping them moves which
language is starved, and therefore which weight is worth raising — with Maithili the binding
constraint is Maithili, with Tamil it moves to Telugu, now the smallest corpus by a factor of five.

## Faithfulness

> `decode(encode(text))` must preserve the same non-whitespace characters as `text`.

A tokenizer that quietly drops brackets or number separators posts a lovely token count while no
longer representing its input. [`tests/test_faithfulness.py`](./tests/test_faithfulness.py) runs
the rule against all four real articles, and every invariant is also run against something
deliberately broken so the guard is known to be able to fail. Three details decide whether it means
anything:

- **The baseline is post-NFKC.** The recipe normalizes before tokenizing, and NFKC genuinely
  rewrites characters (`″`→`′′`, `ⓘ`→`i`, thin spaces). Comparing against raw text fails every
  correct tokenizer, including the reference one.
- **Zero `[UNK]` is asserted, not assumed.** Unknown characters are dropped on decode, so they
  would sail straight past a round-trip check — both sides lose the same character. Train and eval
  share these files, so coverage is total; the test is what keeps that true.
- **Raw `U+2581` is banned from the corpus.** Decode turns every `▁` back into a space, so a
  genuine one in the input would be silently rewritten. There is none, so we assert its absence
  rather than escape the marker — escaping would change the token stream and with it the score.

## A criticism of the metric we just optimized

Having reproduced the score faithfully, it is worth saying what it does and does not measure.

**It rewards flatness, not quality.** `1000 / (X_max − X_min)` is maximized by making every
language equally mediocre. The Hindi penalty is meant to block that, but it only fires above
X = 1.2 and every configuration here sits near 0.6 — so on this corpus the guard is inert and the
exploit is unguarded. That is why every table above reports **corpus-wide X** next to the score:
a config that shrinks the spread while raising that number has flattened the languages rather than
improved them, and you can see it happen in the E2 rows.

**Much of what it measures is script-independent.** Punctuation alone is the majority of faithful
units in every language (en 52.0%, hi 52.3%, te 55.6%, mai 55.8%, ta 52.2%), and the "Indic"
articles are roughly half Latin letters by volume once URLs and link text are counted — 48.5% for
Hindi, 48.3% for Telugu, and 57.0% for Tamil. Markdown scaffolding is identical across languages,
so a good chunk of the convergence the score rewards is handed to the tokenizer by the corpus
format rather than earned by the merges.

## BPE from scratch (no library)

[`bpe_scratch.py`](./src/tokenization/bpe_scratch.py) implements the Sennrich/Karpathy merge loop
by hand: NFKC-normalize, split on whitespace, prefix each word with `▁`, seed the vocab with base
characters, then repeatedly merge the most frequent adjacent pair (pair statistics updated
incrementally; ties broken lexicographically, so training is deterministic). It duck-types the
slice of the HuggingFace API the pipeline uses, so it drops into the harness as `algo="bpe-scratch"`
and appears in **both** profiles.

Its two results say different things, which is a good illustration of why the profiles are kept
apart:

- **Under v1 it scores 1,300 — narrowly *beating* HuggingFace's own char-level BPE (1,228)** on the
  same recipe, behind only the different Unigram algorithm.
- **Under v2 it scores 6,175 — *below* the HuggingFace recipe (6,503).** It does produce the
  fewest total tokens of any configuration (188,091), which is worth being precise about rather
  than proud of: it splits on *all* whitespace and discards newlines entirely, so it never spends
  a token on one. On clipped prose that costs almost nothing; on Markdown, where line structure
  carries meaning, it is a real difference in what is being counted rather than a better merge
  loop.

## Widget (the reviewer deliverable)

[`web/index.html`](./web/index.html) renders `web/data.json`: the four fertilities, the score
calculation with its penalty, the full searchable vocabulary, a **download** button, and a
**paste-your-own-text encoder** that runs the real merge list in the browser.

It opens on **v2** with the reference solution marked ★ and our submission ✓, and a second tab
holds the **v1** tokenizers. Each section names the denominator it is scored in and says in words
that its numbers do not travel to the other — a `units` column header over word counts is exactly
how two measurements get quietly conflated.

The download and the encoder are the point. *"A vocab list without the actual encoding algorithm is
not enough to reproduce your score"* — so `data.json` carries the **ordered merges**, and
[`web/encoder.js`](./web/encoder.js) is the algorithm that replays them. The submitted tokenizer
also ships in HuggingFace's own format at [`web/tokenizer.json`](./web/tokenizer.json), so
`Tokenizer.from_file(...)` reproduces our counts directly. Characters outside the vocabulary render
as a visible `[UNK]` chip instead of vanishing.
[`tests/test_js_encoder.py`](./tests/test_js_encoder.py) runs corpus lines through both Python and
`node` and requires identical token streams — including a line with a literal `_`, which must never
be confused with the `▁` marker.

```bash
cd web && python3 -m http.server 8000   # http://localhost:8000
```

Live at <https://llm-pretraining-demos.vercel.app/02-tokenization/>, deployed via the repo-wide
Vercel project (see [`deploy/`](../../../deploy/)).

## Layout

```text
corpus/v1/       # committed clipped-prose snapshots — what v1 was measured on
corpus/v2/       # committed wiki-faithful Markdown + metadata — what v2 is graded on
src/tokenization/
  config.py         # the two EvalProfiles, languages, titles, vocab size, weights
  corpus.py         # load a profile's snapshot; rebuild one from Wikipedia REST HTML
  metrics.py        # units, words, fertility, spread, score, Hindi penalty, corpus-wide X
  faithfulness.py   # the round-trip rule as executable checks
  ablate.py         # Spec / train_spec / run / sweep / V1_SUITE + V2_SUITE — the one trainer
  holdout.py        # train on 80%, score the 20% never seen
  fourth_language.py# Maithili vs Tamil, same recipe
  bpe_scratch.py    # the same algorithm written by hand, no library
  widget.py         # exports web/data.json (vocab + ordered merges)
  __main__.py       # train the submission, write tokenizer.json + report.json
web/             # zero-dependency page (index.html + encoder.js + data.json)
tests/           # metric math, faithfulness on the real corpus, Python↔JS parity
artifacts/       # gitignored run outputs
```

## Run it

```bash
uv sync --all-packages
uv run python -m tokenization                  # train the submission, print + save the report
uv run python -m tokenization.ablate           # both profiles' tables, side by side
uv run python -m tokenization.holdout          # in-sample vs held-out
uv run python -m tokenization.fourth_language  # Maithili vs Tamil
uv run python -m tokenization.widget           # rebuild web/data.json
```

Re-fetching a corpus snapshot is deliberately a separate, explicit command
(`uv run python -m tokenization.corpus ta`) — the committed snapshots date from 2026-07-13 and
Wikipedia has moved on, so refetching one article silently makes it incomparable with the rest.

## Tests

```bash
uv run pytest -m "not integration"   # metric math (fast)
uv run pytest                        # + faithfulness on the real corpus, Python↔JS parity
```
