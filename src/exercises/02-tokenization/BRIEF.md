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

- **words** = number of whitespace-separated words in the article (script-agnostic; a `\w+` regex
  wrongly splits Indic words at combining vowel marks).
- **tokens** = number of BPE token ids the tokenizer emits for the article.
- **ratio X** = `tokens / words` — the average tokens per word ("fertility"). Lower is better.
  The brief literally writes `words / tokens`, but the "~1.2" target and its example ordering
  (English as the *least*) only hold for `tokens / words`: with a decent BPE, English fertility is
  ≈ 1.2 while Indic scripts are much higher. So we use `tokens / words`.

## What you're submitting

1. A **widget** that shows the four ratios (X1…X4), token statistics, the calculations, and your
   self-score.
2. The widget must let a reviewer **see your tokenizer** — the full list of all tokens in the vocab.
3. A **Netlify URL** for that widget.

> **Hosting note:** we're moving these demos to **Vercel**; the submitted link will be a Vercel URL.
> Netlify will be **decommissioned** (not archived).

The Python pipeline here (fetch → train → score, plus the ablation harness) is the *engine*; it
exports the numbers and the full vocabulary that the widget renders.
