# Session 2 — Tokenization: a balanced multilingual BPE

> **In one line:** Build a single 10,000-token BPE vocabulary shared across India's Wikipedia article in four languages, tuned so every language is tokenized about equally efficiently — the smaller the gap between the best- and worst-served language, the higher the score.

## Brief

Pick India's page on Wikipedia in **English, Hindi, Telugu, and one more language of your choice**.
Ask your AI Agent to design a BPE tokenizer such that:

- You have **10,000 tokens** (your vocab) overall for all languages.
- `(Total English Vocab, say 5000 words) / (Total English tokens)` must be around **1.2 or less** — call this **X1**.
- Similarly, the ratios for Hindi (**X2**), Telugu (**X3**), and another language (**X4**).
- Sort X1, X2, X3, X4… say it's X4 (largest), X2, X3, X1 (least).
- **Your assignment score is `1000 / (X4 − X1)`** — i.e. `1000 / (max ratio − min ratio)`.

A tokenizer that serves all four languages equally well (small spread) scores high; one that's
great for English but poor for Telugu (large spread) scores low.

## What you're optimising

The graded quantity is the **spread** between the most- and least-efficiently tokenized languages.
The levers you control:

- **Vocabulary allocation** — one 10k vocab has to cover four scripts. The corpus mixture / upsampling
  weights decide how many merges each language effectively wins.
- **Pre-tokenization & normalization** — how text is split before BPE. A byte-level scheme handles every
  script uniformly.
- **Choice of the 4th language** — a script close to Hindi/Telugu vs. a Latin-script one shifts the balance.

## Definitions used here

The brief is loosely worded, so we pin the terms down (see `src/tokenization/metrics.py`):

- **corpus** = **wiki-faithful Markdown**: Wikipedia's REST HTML converted with `markdownify`,
  keeping links, URLs, tables, references, image links, navboxes and categories. Explicitly *not*
  clipped article prose — "do not report numbers from a clipped page".
- **faithful unit** = one contiguous run of Unicode letters/marks/numbers, **or** one visible
  non-space punctuation/symbol character. `भारत` is one unit; `](` is two; whitespace is never a
  unit. The `\p{M}` class is load-bearing — it keeps Indic combining marks attached to their base
  character instead of fragmenting every word at its matras.
- **tokens** = number of BPE token ids the tokenizer emits for the article.
- **ratio X** = `tokens / units` — fertility. Lower is better, and **below 1.0 is normal**, because
  BPE merges frequent punctuation runs so one token can cover two or three units.
  The brief literally writes the fraction the other way up, but its example ordering (English as
  the *least*) only holds for `tokens / units`.
- **score** = `1000 / (X_max − X_min)`, divided by a **Hindi penalty** `exp(max(0, X_hi/1.2 − 1))`
  that exists to stop you shrinking the spread by degrading the best-served language.

> Earlier revisions of this exercise counted whitespace **words** over clipped prose. That is a
> different measurement, not a worse one — the same tokenizer scores ≈ 0.60 in units and ≈ 2.13 in
> words — so numbers from the two are never comparable. `count_words` is still computed and
> reported next to the unit counts so the difference is visible rather than confusing.

## What you're submitting

1. A **widget** that shows the four ratios (X1…X4), token statistics, the calculations, and your
   self-score.
2. The widget must let a reviewer **see your tokenizer** — the full list of all tokens in the vocab —
   and **download and encode with it**. *"A vocab list without the actual encoding algorithm is not
   enough to reproduce your score"*, so the ordered **merges** and a working encoder ship too.
3. The exact tokenizer file, the build method, the corpus extraction process, per-language token
   counts, the fertility ratios, and the score calculation.
4. A public **URL** for that widget.

> **Hosting note:** these demos are served from **Vercel**; the submitted link is a Vercel URL.
> Netlify (the prior host) is decommissioned, not archived.

The Python pipeline here (fetch → train → score, plus the ablation harness) is the *engine*; it
exports the numbers and the full vocabulary that the widget renders.
