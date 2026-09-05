# PROGRESS — 02 · Tokenization

The plan, the state, and the evidence. `REQUIREMENTS.md` (local only) holds the requirement text;
this file holds what was built from it and how far it got.

**Written retrospectively on 2026-09-05**, from the commit history, the tracked files and the test
suite — this exercise ran to completion before the convention of keeping a progress ledger existed.
Everything below cites what it was read from, and **where the record does not settle something it
says so**.

**Deliverable:** one 10,000-token BPE vocabulary shared across the *India* Wikipedia article in
several Indian languages, plus a public page that tokenizes text in the browser.

**Contract, in one line: the reference recipe is reproduced exactly before anything is claimed to
beat it.** A harness that cannot re-derive the number it is measured against is not measuring.

---

## Status

| stage | what it delivers | state |
| --- | --- | --- |
| **1 · Scaffold and a hand-written BPE** | the exercise, and byte-pair encoding implemented from scratch | **done** — `48c09ea`, `8bc5ec5`, 2026-07-09 |
| **2 · Hosting and design** | Vercel, the landing card, the shared design system and tokens | **done** — `29585d4` 2026-07-09 … `6969f92` 2026-08-05 |
| **3 · The gate** | scoring on faithful units, and the reference recipe reproduced exactly | **done** — `6dbc801`, 2026-08-06 |
| **4 · Beating it, provably** | a better recipe, and how much of the gain is real | **done** — `210fd71`, 2026-08-06 |
| **5 · Ship the tokenizer** | the artefact itself, with the published numbers guarded | **done** — `3cca3fb`, 2026-08-06 |
| **6 · Two profiles, never ranked together** | v1 retained as a real profile rather than as prose | **done** — `638243b`, 2026-08-07 |
| **7 · The rejected experiment** | the road not taken, published rather than dropped | **done** — `c766a87`, 2026-08-07 |
| **8 · The fourth language** | `mai` submitted at ×3 | **done** — `c5dc3ad`, 2026-08-07 |
| **9 · The explainer** | its own page, linked from the tool | **done** — `70f66d4`, 2026-08-07 |
| **10 · Browser tests** | the widget loaded in a browser, not merely parsed | **done** — `14d5d78`, 2026-08-06 |
| **11 · Paths and documents** | re-fetch writes where the loader reads; diagrams; module lists | **done** — `432e613`, `377b5a2`, `64c9967`, 2026-08-09 |
| **12 · Notebook** | a Colab notebook and a builder | **done** — `7f3e352`, 2026-08-24. Both **local only** |
| **13 · Confidentiality** | quoted passages paraphrased; source vocabulary removed | **done** — `0fd6556`, `09c9619`, `535ab95`, 2026-09-02 |
| **14 · Retro-fix** | link colours, the tab contract, unreadable segmented options | **done** — `dbf9783` (#131), `608dfdf` (#115), `1d80e9c` (#130), 2026-09-04 |
| **15 · Attribution** | a `NOTICE` for the CC BY-SA corpus this exercise redistributes | **done** — 2026-09-05, and it should have existed from stage 1 |
| **16 · Submit** | PK's action | **submitted** — see *Submission* below |

**41 commits touch this exercise**, from `48c09ea` (2026-07-09) to 2026-09-05. **119 tests**
collect under `src/exercises/02-tokenization`. The page returns **HTTP 200** anonymously.

## The corpus is tracked, and that is a decision with a cost

`corpus/v1/*.txt` and `corpus/v2/*.faithful.txt` are committed rather than fetched, so every
published number runs offline from bytes that cannot move underneath it. Wikipedia drifts: refetch
one article and it silently stops being comparable with the rest.

The cost is that this exercise **redistributes CC BY-SA text**, and for most of its life said so
only in a parenthetical inside `CLAUDE.md` — a file addressed to coding agents. `NOTICE` now carries
the attribution, and `tests/test_tokenization_notice.py` keeps it honest in both directions.

## What this exercise cannot establish

- **Perplexity and fertility are not comparable across tokenizers.** A tokenizer that splits more
  finely is asked an easier question per token and scores better while being no better. This
  exercise measured a real case of it.
- **Held-out scoring cannot rank recipes on this corpus.** Across the five possible 80/20 splits one
  recipe's held-out score swings 9,421 points while the recipes sit 648 points apart. Four articles
  is too little text, and `holdout.py` exists to say so rather than to be used.
- **The anti-exploit penalty is inert here.** The published Hindi penalty only fires above X = 1.2
  and everything measured sits near 0.6, so the fairness device is not doing work — which is why
  total tokens is reported beside the score.
- **One article per language is not a language.** Every number describes the *India* article, not
  Hindi or Tamil.

## Submission

**A submission was made, and the tracked record says so without dating it.** `README.md` names one
row of its own comparison table *"documents · mai ×3 — submitted"*, refers throughout to "the
submitted tokenizer", and `tests/test_submission.py` exists to assert that recipe has not moved.

What no tracked file carries is **when**, or to what. Both are PK's knowledge; this ledger will not
guess at either.

## What this record does not settle

- **Why 10,000 tokens.** The size is a requirement input, and no tracked file argues it.
- **What the rejected experiment cost to run.** `c766a87` publishes the result, not the effort.
- **The month between 2026-07-10 and 2026-08-06** is not accounted for by anything in this exercise.
- **Which of the four pinned v1 settings was decided first**, or whether they were chosen together.
