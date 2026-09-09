# Where Kronecker v2 goes next

Five open problems, three of them researched here. This document is the record: what each problem
is, what we thought would happen, what we measured, and what we now believe. It is written to be
read at whatever depth you want and stopped at any point without being left with a wrong idea.

**Every claim carries a mark saying where it came from**, because they are not equally strong:

| mark | means |
| --- | --- |
| **[measured]** | run here, on this machine, against the frozen vocabulary. Reproducible with a command in this document. |
| **[verified]** | I re-derived it myself after an agent reported it, with my own code. |
| **[reported]** | an agent reported it and I have **not** independently checked it. Treat as a lead. |
| **[reasoning]** | an argument, not a measurement. Check the argument. |

The distinction is not decoration. One research pass corrected itself twice and named two papers it
had earlier cited that **do not exist**. Nothing marked *[reported]* should be repeated as fact
without being checked first.

---

## How to read this

- **First time here, or not technical** — read *The five problems in plain words*, then the
  *In one sentence* line at the top of each problem. That is about four minutes and you will be able
  to say what we found and why it matters.
- **You want to build one of these** — read the problem's *The idea*, *What we measured* and *How to
  re-run it*. The configuration is stated exactly.
- **You want to attack the result** — read *What we expected*, *What would refute this*, and *What
  none of this establishes* at the end. The weaknesses are listed there rather than left for you to
  find.
- **You are deciding whether to fund it** — read *What I would build first* at the end. It names one
  thing, what it costs, and what would tell you it failed.

---

## The five problems, in plain words

The thing being improved: an embedding is the vector a model uses to represent a word. Normally
there is one stored vector per word, in a giant table. Kronecker embeddings **compute** the vector
from the word's *letters* instead, so the table disappears and the cost stops growing with the
vocabulary.

| # | the question | in plain words |
| --- | --- | --- |
| 1 | mathematical structure | can the embedding of "9" carry the *number* nine, so that adding two embeddings adds the numbers? |
| 2 | other kinds of data | can the same trick represent pictures and sound, not just text? |
| 3 | length | today a word longer than 32 letters gets cut off. Can that limit go away? |
| 4 | waves | can we describe each letter as a wave and add the waves together? |
| 5 | reversing it | can we go *backwards* from a vector to the word — which would let us delete the model's output layer entirely? |

Problems 4 and 5 already have answers in this exercise. Problem 5 is its central result: yes, and the
output layer goes away. Problem 4 was tried and **failed**, and problem 3's research below explains
why in a way that was not previously understood.

**The three researched here are 1, 2 and 3**, and each was also asked a second question: *does
solving it help or hurt problem 5?* That matters because problem 5 is the result we already have,
and a change that quietly breaks it is not an improvement.

---

# Problem 1 · Can an embedding do arithmetic?

> **In one sentence.** The idea does not survive contact with our own vocabulary — most numbers are
> not single words to begin with — but a narrower version of it is worth testing, and there is now a
> cheap experiment that would settle it.

## The idea

Add a few extra numbers to the end of each embedding that encode the token's numeric value, so that
adding two embeddings adds the values: the number-carrying part of `E("9") + E("9")` would be the
number-carrying part of `E("18")`.

**Why anyone would want it.** Not to make a model better at sums it has seen — it can already learn
those. The hope is **extrapolation**: a model that works on numbers far larger than any it trained
on, because it represents *quantity* rather than memorising *digit strings*.

## Why we thought it might work

The byte code cannot represent quantity at all. Train a simple probe on the numbers 0–99 and ask it
to read the value of numbers 100–999 **[verified]**:

| what the probe reads | R² | typical error |
| --- | ---: | ---: |
| the byte code as it is today | **−3.64** | 505 |
| the byte code, right-aligned | −3.83 | 509 |
| one explicit "value" number | **1.000000** | 0.0000000000001 |

An R² below zero means *worse than guessing the average*. So the byte code genuinely lacks something,
and one extra coordinate genuinely supplies it. That is the strongest argument for the idea, and it
is real.

## What we measured, and why it kills the idea as stated

**1 · Most numbers are not single words.** Our vocabulary was built by a method that glues common
letter pairs together, and it never learned most numbers **[verified]**:

| number range | is one word? | after a space, as in real text |
| --- | ---: | ---: |
| 0–99 | 96% | **38%** |
| 100–999 | 11% | 0.6% |
| 1,000–9,999 | **0.5%** | 0.5% |
| 10,000+ | **0%** | 0% |

`128` is stored as `1` + `28`. `1000` is `1` + `000`. So there is no single embedding for `128` to
carry a value — and worse, in `1234` split as `123` + `4`, the piece `123` actually means **1230**.
A value attached to the piece would be wrong by a factor of ten, for essentially every number above
a hundred.

**2 · The normalisation step destroys it.** Before use, each code is rescaled so every word has the
same overall size. Add a coordinate that grows with the number and the rescaling fights it
**[verified]**:

| the number | what the value coordinate reads | what a letter of the word reads |
| ---: | ---: | ---: |
| 0 | −0.02 | 52.25 |
| 100 | 90.51 | 0.51 |
| 1,000 | 90.51 | 0.04 |
| 10,000 | 90.51 | **−0.01** |

The value coordinate **stops changing** above about 100 — it hits a ceiling of 90.5152 — while the
letters of the word are squashed to nothing, and above 4,730 their sign actually **flips**. Exactly
backwards: the part meant to carry the number carries almost nothing, and it erases the part that
carried the word.

## Three things that are impossible, and knowing them saves work **[reasoning]**

1. **One block cannot do both adding and multiplying.** If adding two embeddings gave both `a+b` and
   `a×b`, then setting `b = 1` forces every embedding to be zero. So: two separate blocks, or one
   operation. There is no clever third option.
2. **Exact adding over an unlimited range needs an unlimited coordinate** — and it is only ever
   *one* coordinate's worth of information. Extra coordinates buy nothing.
3. **Sign, odd/even, and "remainder when divided by 7" cannot be encoded additively at all**, for a
   structural reason. That is why circular/wave encodings keep coming up, and what they cost.

There is a pretty consequence: with one linear block and one logarithmic block, `E(9) + E(9)` carries
**18 in one and 81 in the other, at the same time** — to within 0.000000000015 across 12,142 real
pairs **[reported]**. But the sum is never the embedding of a single number, so **the embedding
cannot tell you which operation was meant.** That is the context's job, not the embedding's — which
is worth knowing before building a demonstration around the identity.

## What the literature says **[reported — not verified by me]**

Two things worth chasing, and one caution.

- Approaches that make a model generalise to much longer numbers exist and are measured; the strongest
  reported result trains on 1–30 digits and works to 200. **I have not verified this citation.**
- Single-digit tokenisation is reported to beat multi-digit on both familiar and unfamiliar numbers.
  If true, our byte-level code is already in the favourable regime — there is no point where two
  neighbouring numbers suddenly share no representation.
- **The caution:** the pass that produced these corrected itself twice and named two papers that do
  not exist. Verify before citing.

## The verdict, and what survives

**As stated: dead on this vocabulary.** Not because the idea is bad, but because the objects it needs
— single tokens that are numbers — mostly do not exist here.

**What survives is sharper.** The byte code is *left-aligned*: the `9` in "9" and the `1` in "18"
land in the same slot. So decimal place value is not something the code can express, at any setting.
A **right-aligned, place-value block** would add information the code genuinely lacks — which is the
same property that made the byte-pair term in this exercise worth −0.412 nats where a neural network
added on top bought −0.002.

## The experiment that would settle it, and what would refute it

**The decisive piece is a control arm**, not the main arm. Build the numeric block twice: once real,
once with every value passed through a fixed random shuffle. Same parameters, same dimensions, same
coverage — **zero arithmetic meaning**.

- **If the idea works:** the real block beats the shuffled one on numbers it never saw during
  training, by a clear margin, and the gap is *bigger* out-of-range than in-range.
- **If it does not:** the two sit on top of each other. That result is worth publishing as-is — it
  says the block is present, provably additive, and **unused**.
- **What would make the whole thing meaningless:** building the test set only from numbers that
  happen to be single words. In the 1,000–9,999 range that is 49 numbers out of 9,000, and they are
  not a random 49 — they are mostly **years**. A model does well on them because it has seen them
  thousands of times, and the headline would read "extrapolates to four digits" while measuring
  familiarity.

---

# Problem 2 · Can the same trick do pictures and sound?

> **In one sentence.** Feeding raw picture bytes into this design cannot work, for a reason that is
> arithmetic rather than engineering — but compressing a picture first brings it back into range,
> and then one mechanism really does serve all three kinds of data.

## The hard constraint, first

Reading a word back out of its vector is a search, and the search only succeeds if the vector has
enough room. Measured: it needs roughly **10 to 12 dimensions per letter** **[reported]**.

A text word averages about 6 letters, so a few hundred dimensions is plenty — which is why this works
today. A small colour image patch of 16×16 pixels is **768 bytes**. At 10–12 dimensions each, that
needs about **9,216 dimensions**, which nobody builds.

**So raw bytes are out.** Not "expensive" — out.

## The idea that survives

Compress the patch first into a short list of codes, the way image compression already does: a patch
becomes, say, **16 codes** instead of 768 bytes. Sixteen is the same order as a text word, so it fits
the same machinery unchanged.

**And "unchanged" is meant literally.** If the codes are numbered 0–255, a compressed patch *is* a
16-letter word as far as this code is concerned. The reader, the reversal proof and the tied output
layer all apply with no modification **[reported]**.

## What that costs, in the only number that matters here **[verified]**

The whole point of this design is that the cost does not grow with the vocabulary. Here is what each
route does to that:

| route | parameter saving at a million-word vocabulary |
| --- | ---: |
| text only, as it ships | **58.3×** |
| text + images + sound, compressed first | **42.9×** *[reported]* |
| text + images + sound, raw bytes | **2.4×** *[reported]* |

The compression step is worth about **18×** of saving. That is the argument in one row.

> **A number to be careful with.** This exercise quotes two different savings — **58.3×** and
> **111.6×** — and both are correct **[verified]**. The first counts the projection *plus* the
> byte-pair block; the second counts the projection alone. Always say which.

## What we found that was not expected **[reported]**

Ask the reader to decode a vector against text, then against images, then against sound, and keep
whichever fits best. It picks the right one **every time**, and the margin is about **thirty orders
of magnitude**. So the reversal machinery identifies what kind of data it is looking at, for free —
nobody has to label it.

Two cautions in the same report: the compression codebook has to be frozen and recorded like the
vocabulary is, and the reader's guarantee is that it recovers **the compressed codes**, not the
original pixels. A text embedding does not recover "the meaning" either; it recovers the word. Say
which, always.

## The verdict

**Worth doing, and not first.** It needs a compression model we do not have, and the honest result at
our scale would be weak. Everything above is about the *code and the arithmetic*; nothing here says a
trained multimodal model would actually be better.

---

# Problem 3 · Can the length limit go away?

> **In one sentence.** Yes — and the version we found is *better than what ships today* on the
> vocabulary we already have, at the same cost, with no training required to prove it.

## What is wrong today **[verified]**

Every byte position owns a private block of 256 slots. Thirty-two positions, so byte 33 has nowhere
to go:

| positions | words that fit | width of the code |
| ---: | ---: | ---: |
| 16 | 85.3% | 4,096 |
| **32 (today)** | **94.7%** | **8,192** |
| 64 | 99.3% | 16,384 |
| 128 | 100% | 32,768 |

**533 of our 10,000 words are cut off.** Covering the longest (121 bytes) means **four times** the
code width to buy the last 5% — a hard allocation doing a soft job.

And the cost is not only width. Widening to 128 positions drops the parameter saving from **58.3× to
24.0×** **[verified]** — it eats a third of the result this exercise exists to demonstrate.

## The idea

Stop giving each position a private block. Give each position a **direction** in a shared space —
an angle on a compass rather than a drawer in a cabinet. There are unlimited directions, so there is
no length limit, and the code width does not change at all.

The price is that two directions are never exactly perpendicular, so long words get gradually harder
to read back. **Gradually** is the whole point: today it is perfect to 32 and impossible at 33.

## What we expected before measuring

There is a known limit on how well you can spread `P` directions in `d` dimensions. At our sizes it
says the design can lose **at most 15%** of its ability to tell letter order apart, no matter how
long words get **[reasoning]**. That bound is why the idea was worth measuring at all: the downside
is capped in advance.

## What we measured **[measured]**

**This section was `[reported]` and is now `[measured]`, and three of its four numbers moved.** The
scheme has since been built and run on the frozen vocabulary by a tracked producer,
`tools/measure_position_schemes.py`, writing `results/position_schemes.json`. The research pass's
figures are kept below rather than quietly replaced, because a document whose whole design is
provenance marks should show what a mark was worth when it changed.

Across the whole 10,000-word vocabulary, at the same code width as today — every byte of the word
back, in order:

| scheme | reach | words fully recovered | first reported |
| --- | --- | ---: | ---: |
| today's, cut off at 32 | 32 bytes | **94.67%** | 94.67% ✓ |
| today's alternative, folding | unlimited but **lossy** | **94.67%** | 95.58% |
| **the new one** | **unlimited** | **99.86%** | 99.74% |

**The conclusion survives and one comparison gets sharper.** Folding does not in fact beat cutting
on this question — both recover 9,467 of 10,000, because both fail every word longer than 32 bytes
and for the same reason: neither has anywhere to put the 33rd byte. The new scheme's margin over
*both* is therefore **5.19 points, not 4.16**, and the refutation clause below that reads "at or
below 95.58%" is written against a baseline that does not exist.

**The `widen to 128 positions` row is dropped rather than corrected.** Nothing here re-measured it,
and the nearest published figure — `measurements.json::d_p_128`, 99.9% — is over a sample
deliberately enriched with long words, which is a different denominator and not comparable with the
column above. What it was there to show is still true and is arithmetic rather than a measurement:
widening to 128 positions makes the code four times wider (`D = 256 · d_p`, so 8,192 → 32,768),
which is the cost the new scheme avoids paying.

**Of its 14 failures, none are the code's fault** — in every one, the true answer scores strictly
better than the answer found, so the information survived the encoding and the *search* ran out.
That was reported as 26 failures; the count moved with the recovery rate. The claim it supports did
not: `searchable` is **1.000** in both length bands where the scheme misses anything at all.

## Why the wave idea (problem 4) failed, which nobody had explained **[verified]**

The wave scheme is the *same family* as this one with a badly chosen set of directions. I measured
it: **neighbouring positions point 96% in the same direction.**

That is fatal, and it is obvious once stated — a position code exists to tell you *which* position a
letter was in, and this one makes adjacent positions nearly identical. Swapping two neighbouring
letters barely changes the code. So problem 4's failure was not bad luck, and it needed no training
run to diagnose.

It also means **the new scheme does not inherit that failure**: its neighbouring directions measure
essentially unrelated **[reported]**.

## What this does for problem 5

This is the one that *helps*:

- Today, **407 words (4%) share a code exactly** — they are indistinguishable, so "the embedding can
  be reversed" is false for them. The new scheme has **zero** collisions **[reported]**.
- It keeps the parameter saving unchanged, where widening the positions cuts it by more than half
  **[verified]**.
- It removes the length limit *and* keeps reversibility, which the folding scheme cannot: folding
  provably loses letter order past 32, and that is a theorem, not a decoder weakness.

## What would refute it **[reported]**

Stated as tests rather than hopes, and this is the part to attack:

- any two different words with identical codes;
- any failure where the answer found fits *better* than the truth — that would mean information was
  genuinely lost;
- a drop of more than one point across the 32-byte boundary, which would mean the cliff is still
  there;
- whole-vocabulary recovery at or below **94.67%**, which would mean it buys nothing over either
  shipped scheme. (This read 95.58% while folding was believed to beat cutting; it does not.)
- **a different random choice of directions giving a materially different answer** — that would mean
  the result is a property of one lucky draw rather than of the construction.

**And one thing that would NOT refute it:** a training run where it scores worse on loss. That is a
different claim on a different axis. This exercise's own history is the warning — the folding scheme
trains *better* than the truncating one while being strictly lossier.

---

## What I would build first, and why

**Problem 3, as a fourth position scheme in the comparison.**

- It is the only one of the three that **improves a result we already have** rather than adding a new
  one: 94.67% → **99.86%**, at the same cost.
- Its central claim **needs no training**, so it cannot be overturned by a change of corpus — which
  is exactly what happened to this exercise's other headline. Recovery is a property of the code and
  the vocabulary, and that is the kind of claim that survives.
- It explains problem 4's failure as a side effect, at no extra cost.
- It has a written list of five things that would refute it.

**What it costs:** a new scheme in the codec, a small generalisation of the reader, and one grid run
to check it does not hurt training. Perhaps a day.

**What would tell you to stop:** any of the five refutations above, or a different random draw of
directions moving the number materially.

---

## What none of this establishes

- **Nothing here is a trained result.** Every number in problems 2 and 3 is about the code, the
  reader and the parameter count. Whether a model *trains* better with any of it is unmeasured, and
  this exercise has already been caught by exactly that gap once.
- **One vocabulary, one machine, mostly one random draw.** The recovery figures are a property of the
  10,000-word vocabulary they were measured on and of the specific random projections used.
- **The literature review is not verified.** Every citation is marked *[reported]*, and the pass that
  produced them named two papers that do not exist.
- **Problem 1's decisive experiment has not been run.** What is established is that the idea as
  stated cannot work here, not that the narrower version does.
- **Problem 2's central claim rests on a compression model we do not have**, and the rate–distortion
  numbers behind it were measured on synthetic images, not real ones.

---

## How to re-run what is here

Everything marked **[measured]** or **[verified]** comes from this repository, with no network and
no training:

```bash
uv sync --all-packages

# how many words are cut off, and what covering them would cost
uv run python -c "from embeddings.experiment import load_vocabulary as v; \
    b=[len(t) for t in v()]; print(sum(1 for n in b if n>32), 'of', len(b), 'over 32 bytes')"

# what widening the positions costs in parameters
uv run python -c "from embeddings.budget import budget; \
    print([round(budget(1_000_000,768,d_p=d,n_buckets=8192).dense_tied / \
    budget(1_000_000,768,d_p=d,n_buckets=8192).v2_total,1) for d in (32,64,128)])"

# what wrapping actually recovers, by word length
uv run python src/exercises/07-model-embeddings-internals/tools/measure_wrap_recovery.py

# how similar neighbouring positions are under each scheme -- why the wave idea failed
uv run python -c "import numpy as np; from embeddings import codec; \
    from embeddings.config import KroneckerConfig as C; \
    g = np.asarray(codec._table_for([b'A'*64], C(d_p=32, d_model=384, positions='fourier')), float); \
    g /= np.linalg.norm(g, axis=1, keepdims=True); \
    print('adjacent similarity:', round(float(np.mean([g[p] @ g[p+1] for p in range(30)])), 4))"
```

The trained comparisons, the corpus gates, the auditors and the recovery-by-length measurement are
listed in the exercise's [README](README.md); the reasoning behind each setting is in
[DECISIONS.md](DECISIONS.md).
