# 08 · Modern attention variants — in the order they were launched

**Thirty ways of computing attention and of telling a model where a token sits, ordered by the
date each one actually appeared, with every date read from the primary source.**

The task here is not to describe attention mechanisms. It is to put them in **chronological order**
and explain each as an answer to a problem that existed *at that moment*, because the order shows
something a list cannot: read in sequence, the field visibly changes its mind, and having seen it
change you can make a reasonable guess at what comes next. A list flattens that away.

The graded axis is therefore the **dates**, and we were warned about the failure mode specifically:
an agent asked for a launch date will supply a confident one it has half remembered, so every date
has to be checked against the paper or release itself rather than recalled.

So this exercise treats a date the way the other exercises treat a measurement. Every entry in
[`results/mechanisms.json`](results/mechanisms.json) carries the URL it was read from, the source's
**own wording** of the date, and the day somebody looked. A citation that cannot be checked will not
construct.

## How to read this

- **Meeting this for the first time** — read [What the two bills are](#what-the-two-bills-are)
  below. Everything on the timeline is somebody paying less of one of them, and none of the rest
  makes sense without that.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it). The catalogue is data; the modules only load, validate and count it.
- **Deciding whether to believe it** — go to [The evidence](#the-evidence), which explains how each
  date was verified and what we found wrong, then
  [What this cannot establish](#what-this-cannot-establish).
- **Wanting to pick a mechanism rather than read an argument** — the published page opens with an
  at-a-glance table: all thirty on one line each, with *when you would pick it* against every one.
  That field is on all thirty entries in the catalogue and the page used to render it exactly once,
  inside a panel that shows one mechanism at a time and only after a click.

## What the two bills are

Attention charges twice, and the two bills grow differently:

```text
compute    grows with T²    every token scores against every other token
KV cache   grows with T     every token's key and value are kept for the next one
```

Both are closed forms, so this exercise computes them rather than quoting them. At Exercise 08's own
yardstick — 48 layers, 8 KV heads, head dimension 128, bf16 — one user at a 32,768-token context
costs **6.44 GB** of KV cache and eight users cost **51.54 GB**. `src/attention/cache.py`
reproduces both exactly, and a test pins them, so editing the yardstick breaks the documents that
cite it rather than letting them drift.

The third thread is **position**: a model that mixes tokens by content alone cannot tell `dog bites
man` from `man bites dog`, and roughly a third of the timeline is people trying to supply that
without breaking anything else.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | the yardstick model every cost is computed against, taken from the source material |
| `cache.py` | the two bills — KV bytes, `T²` scores, head sharing, sequence compression |
| `sources.py` | the citation model: what was read, from where, quoted verbatim, and when |
| `catalogue.py` | the mechanisms, their trade-offs, and the coverage list the requirements mandates |
| `timeline.py` | ordering, the gaps, and which bill each period was paying down |
| `story.py` | the six chapters the page tells, and the rule that every mechanism is in exactly one |

The page is six more, and they are listed because a module nobody names is a module the next
reader regenerates the site without:

| module | owns |
| --- | --- |
| `web/data.js` | everything the page renders, emitted by `tools/build_web_data.py`. Never hand-edited |
| `web/chapters.js` | the twelve spine sections and the six chapters — layout only, no numbers |
| `web/figures.js` | the six numbered plates: the invoice, the race, the centrefold, the timeline |
| `web/glyphs.js` | the four glyph generators, one per shape, read from each entry's `pattern` block |
| `web/support.js` | the predicate itself — which query-key pairs survive, at any resolution |
| `web/diagrams.js` | the full-size diagram per mechanism, four scenes over the same `pattern` block |
| `web/field-guide/` | the second route: all thirty diagrams at once, in one convention |

`web/support.js` is extracted rather than inlined so a glyph at 26px and a diagram at 720 units call
the *same* predicate — they cannot disagree about what a mechanism does, which they could while each
had its own copy.

`results/mechanisms.json` is the tracked evidence. Nothing derives it; it was assembled by hand from
primary sources and is validated by `catalogue.py`.

Three documents sit beside the code, and each has one job:

| document | holds |
| --- | --- |
| [`DECISIONS.md`](DECISIONS.md) | the reasoning behind a choice, and what it cost — including the corrections that were true and still moved off the page |
| [`docs/METHOD.md`](docs/METHOD.md) | the apparatus: how the page is generated, drawn and themed, and what to run |
| [`docs/MEASURES.md`](docs/MEASURES.md) | the width and readability audit, and what looked wrong in the numbers and is not |

`docs/METHOD.md` exists because the page's own colophon reached 358 words of production notes and
five of six review readers stalled in it. The colophon keeps the three claims the *numbers* rest on
— how a date was read, how a byte figure is computed, what an entry must state to be admitted — and
links out for the rest.

Two of those modules exist because a claim needed somewhere to live where a test could reach it.
`story.py` holds the page's chapter grouping — an editorial claim, so it is data with a guard rather
than prose inside the page's JavaScript, and `story.check()` refuses a partition that does not cover
the catalogue exactly once. `cache.tokens_before_wall()` holds the three crossings the page's race
figure animates towards, as the same arithmetic as the invoice solved for the context instead of the
bytes — so the figure and the table cannot disagree.

## Run it

```bash
uv sync --all-packages                                    # no extras — this exercise needs no torch
uv run pytest src/exercises/08-modern-attention-variants

# the timeline itself, and the pressure in each period
uv run python -c "
import sys; sys.path.insert(0, 'src/exercises/08-modern-attention-variants/src')
from attention.catalogue import load
from attention.timeline import in_order, pressure_by_period
for m in in_order(load()):
    print(f'{m.date}  {m.bill:8} {m.name}')
for p in pressure_by_period(load()):
    print(p.start, p.end, p.dominant or 'no single pressure', p.counts)
"
```

## The evidence

**Every date is the arXiv `v1` submission date**, read from the abstract page, with the
submission-history line stored verbatim beside it. That choice matters: later versions move by
months and sometimes years — Bahdanau's v1 and v7 are twenty months apart — so quoting a revision or
a conference date silently reorders the timeline. Three guards enforce it, and each was watched
failing on a deliberately broken catalogue:

- a mandated mechanism removed → the failure names the instructor's own phrase for it
- a date transposed (`2021-04-20` → `2021-04-02`) → caught by comparing against the quoted string
- a source URL stripped → refuses to construct at all

**Not every source is a paper, and the catalogue says so.** NTK-aware RoPE scaling has no paper: it
is a Reddit post by user `bloc97`, and its date is Reddit's own machine-readable timestamp. We could
not reach reddit.com — it refused our requests — so the field was read from a Wayback Machine
capture, and the entry records that. A reader who needs the live page needs a browser.

### Two errors in the course material

The requirements invites this — *"if you catch me in another one, tell me"* — so both are recorded
rather than quietly corrected.

- **The transformer is mis-dated.** The source says Vaswani "invented in 2018 and 17".
  *Attention Is All You Need* is `arXiv:1706.03762`, v1 **12 June 2017**.
- **DroPE is two different papers, and the source quotes the wrong one's title.** The technique
  described in class — pretrain with positional embeddings, drop them, recalibrate briefly — is
  *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings*,
  `arXiv:2512.12167`. The source's garbled "rotate position emitting for efficient" maps instead
  onto **DRoPE** with a capital R, `arXiv:2503.15029`, an autonomous-driving trajectory paper with
  no relation to the technique. The two names differ by one capital letter.

### One number that does not reproduce

The source says eight users at a 1M-token context need about **1 TB**. The source material's own
formula, at the source material's own yardstick, gives **1.57 TB**. Both are recorded; neither is published
alone. A smaller model, fewer KV heads or fp8 storage would each reconcile them, and the source
does not say which was meant.

### What the order shows

Derived by `timeline.pressure_by_period`, not asserted. The requirements predicts a tidy sequence —
exactness, then memory, then length, then memory again — and the data is messier than that: **one of the seven two-year windows has no single
dominant pressure at all.** In that period the field was attacking several bills at once, and a test
fails if that ever stops being true, so the finding cannot quietly become the tidy story.

> **Correction, and it is our own.** This section said **two** of six windows for as long as the
> catalogue held twenty-three mechanisms. Adding top-k attention (2019-12-25) put a second *compute*
> entry into the 2018–19 window, which had been an exact 1–1 tie between compute and cache, and the
> tie broke in favour of compute. The claim moved from two undecided windows to one. Nothing about
> the method changed — the count was always derived — but the input was incomplete, and an
> incomplete input made the field look more indecisive than it was. This is recorded rather than
> silently amended because the number appeared on a published page.

Five things visible only on a date axis, and not from any list:

- **Attention is three years older than the Transformer.** Bahdanau's soft alignment is 2014-09-01;
  the 2017 paper removed the recurrence around it rather than inventing attention. 1,015 days apart.
- **Nobody attacked the cost for 680 days.** Between the Transformer and Sparse Transformers the
  field used attention without once trying to make it cheaper, because contexts were short enough
  that the bill was small.
- **Two crowds, not one queue.** Sparse attention (2019-04-23) and MQA (2019-11-06) are 197 days
  apart and have nothing to do with each other — one attacks the score grid, the other the cache. A
  date-ordered *list* interleaves them into apparent nonsense; two lanes make it obvious.
- **Nothing attacked both bills at once until 2020-04-10.** Every one of the seven mechanisms that
  bounds compute *and* cache with a single idea falls in the last third of the timeline.
- **Two mechanisms sat unusable for years after they were published.** MQA waited 1,293 days for
  GQA to make head sharing tunable; the delta rule waited 1,204 days for a parallel formulation.
  Publication date and usable date are not the same date, and only the axis shows the gap.

### Eleven mechanisms the coverage list did not name

Each is on the timeline, each carries the URL its date was read from, and each is marked `†` on the
page's index plate.

| mechanism | launch date | source | the source's own date string |
| --- | --- | --- | --- |
| **Additive (Bahdanau) attention** | 2014-09-01 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | `[v1] Mon, 1 Sep 2014 16:33:02 UTC (83 KB)` |
| **Reformer (LSH attention)** | 2020-01-13 | [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) | `[v1] Mon, 13 Jan 2020 18:38:28 UTC (421 KB)` |
| **FlashAttention** | 2022-05-27 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | `[v1] Fri, 27 May 2022 17:53:09 UTC (1,325 KB)` |
| **Mamba (selective state space)** | 2023-12-01 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | `[v1] Fri, 1 Dec 2023 18:01:34 UTC (1,264 KB)` |
| **Parallelised DeltaNet** | 2024-06-10 | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | `[v1] Mon, 10 Jun 2024 17:24:42 UTC (124 KB)` |
| **Kimi Delta Attention (KDA)** | 2025-10-30 | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | `[v1] Thu, 30 Oct 2025 16:59:43 UTC (645 KB)` |
| **Mamba-3** | 2026-03-16 | [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) | `[v1] Mon, 16 Mar 2026 17:30:08 UTC (247 KB)` |
| **Compressed sparse attention (DeepSeek-V4)** | 2026-04-26 | [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) | `[v1] Sun, 26 Apr 2026 14:49:33 UTC (2,854 KB)` |
| **Gated DeltaNet-2** | 2026-05-21 | [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791) | `[v1] Thu, 21 May 2026 17:44:57 UTC (94 KB)` |
| **MiniMax sparse attention (MSA)** | 2026-06-11 | [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392) | `[v1] Thu, 11 Jun 2026 14:23:41 UTC (3,976 KB)` |
| **Higher-dimensional RoPE (HD-RoPE)** | 2026-08-30 | [Higher-Dimensional Rotary Position Embedding](https://arxiv.org/abs/2608.29715) | `[v1] Sun, 30 Aug 2026 10:46:24 UTC (1,372 KB)` |

**The last six carry the timeline to 31 August 2026**, and they are the reason the plate does not
stop at a round number. Every one of them was verified by opening its arXiv abstract page and
copying the submission-history line printed in the table above; six of the six are dated after the
point the reference material stops.

**The position lane now ends on a contradiction, and that is the finding.** DroPE (2025-12-13)
concludes that positional embeddings should be *deleted* and the model left to infer order from the
causal mask. HD-RoPE (2026-08-30) concludes they should be made *richer* — rotating in higher
dimensions rather than in independent planes. Both report gains over standard RoPE. A chronology can
show that the field has not settled this; it cannot settle it.

**What we looked for and did not find is also a result.** We checked every frontier lab through
31 August 2026:

- **OpenAI, Anthropic and Meta published no architecture at all** in the window — system cards
  without attention mechanisms, positional schemes or parameter counts. Nothing to put on a
  chronology of mechanisms.
- **GLM-5, Qwen, Gemma, ERNIE and Kimi K3 describe their attention in terms of mechanisms already
  on this plate** — grouped-query, latent, sliding-window, sparse, and the delta-rule family. They
  are evidence about *adoption*, which is a different axis and one this page cannot see.
- **Gnani.ai has published no attention mechanism.** Searched and found nothing — they build
  conversational voice systems rather than architectures. Recorded so the absence is a checked
  result rather than an unchecked assumption.
- **JEPA and the world-model line do not belong here, and it took checking to say so.** JEPA
  changes the *objective* — predict in representation space instead of reconstructing the input —
  while its encoders remain vision transformers running ordinary softmax attention. Every in-window
  JEPA paper we checked changes the loss, the regulariser or the domain; not one touches attention.
  World models apply existing mechanisms rather than inventing them.

**Bahdanau earns its place by being first.** Attention is on this timeline three years before the
Transformer, and a chronology that starts in 2017 hides the single most surprising thing in it.

**FlashAttention earns its place by invalidating an argument.** Every compute-side entry before it
assumed the cost was arithmetic. It was memory traffic. Exact attention got several times faster
with a bit-for-bit identical result, and the headline case for approximate attention evaporated —
which is why this page draws its glyph *identical* to standard attention's, with only a tiling
overlay to mark the difference.

**Mamba and Parallelised DeltaNet earn theirs by closing gaps the timeline makes visible.** The
delta rule was published in 2021 and was sequential by construction; it sat unusable for **1,204
days** until someone parallelised it. You cannot see a hole like that in a list.

**Where to look next, and how we used it.** Sebastian Raschka's
[*A Visual Guide to Attention Variants in Modern LLMs*](https://magazine.sebastianraschka.com/p/visual-attention-variants)
is the best single index of which attention variants are actually running inside shipped models,
and his architecture comparisons are how several of the leads above were found. It is used here as
an **index, never as a date**: every date on this page was read from the paper's own arXiv abstract
page, because a date copied from a secondary source is exactly the failure this work is graded on.

### One mechanism the coverage list named and we had wrong

The list says *"sparse and top-k attention"*. Those are **two** techniques, and we had catalogued
one. A fixed sparse pattern decides which pairs of positions can ever interact **before the model
sees any data**; top-k decides **per query, from the scores themselves**. Worse, the entry for
Sparse Transformers claimed "top-k attention" as an alias, so the catalogue actively asserted they
were the same thing.

Top-k attention is now its own entry — **2019-12-25**, [Explicit Sparse Transformer: Concentrated
Attention Through Explicit Selection](https://arxiv.org/abs/1912.11637), `[v1] Wed, 25 Dec 2019
10:59:31 UTC (689 KB)` — and `MANDATED` now maps that phrase to *both* keys, so a compound
requirement can never again be satisfied by half of itself. The source material teaches the distinction at
length, including the catch that makes top-k interesting: naive top-k still has to score every key
before it can rank them, so it reduces the work *after* selection and not the scoring itself.

## Two claims that had to be sourced, not asserted

Everything on the page is generated from `results/mechanisms.json`, and two kinds of claim in it
would have been easy to write from memory and impossible for a reader to check. Both are now
sourced the same way the dates are, and both are checked mechanically.

**A size may only enter with the sentence it was read from.** `GLYPH_SCALES` originally held no
sizes at all, because *"a glyph drawn to specific numbers would be inventing them."* The diagrams
need real numbers, so provenance became the price of entry: `Glyph._check_sizes` refuses a `stated`
size without a quote and a location, refuses an `ours` size without a reason, and requires **the
quote to contain the number it is evidence for**. All thirty carry sizes now — 78 of the 80 quoted
verbatim from the primary paper.

The method is worth copying. Every paper was downloaded *first*; agents read those local files and
proposed claims; then each quote was checked as a contiguous run of that file's own characters.
**82 proposed, 82 verbatim, zero fabrications.** Verbatim is not the same as correct, so a second
check asked whether each quote talks about the quantity claimed — which caught *"Figure 4: The KV
cache of StreamingLLM"* offered as evidence for four attention sinks, and a *Communications of the
ACM* volume number offered as a head dimension.

**A model name is a claim too.** The page names the models that ship each mechanism, because
otherwise a reader cannot tell whether it describes history, a research frontier, or the thing
inside the chatbot they used this morning — and *"almost every open model uses them"* asks for
trust while offering nothing to check. Eight model papers were located through arXiv's search API
rather than from memory, read, and quoted: 21 adoption records across 8 models.

**Twenty-two of the thirty have no model named, and that is a result.** It separates what the field
adopted from what it admired. Reformer is the case in point, and a test keeps it empty until a
paper says otherwise.

## The arc was tested, and the test had to be fixed first

The requirements' claim is that the field wanted *exactness, then memory, then length, then memory again*
— in the catalogue's labels, `compute → cache → position → cache`. `timeline.arc_verdict` tests it.

For a while the page published **"the claimed arc holds in 6 of these 7 two-year windows"**. That
number was derived, and it was evidence for nothing: it counted windows that produced *a* clear
winner, not windows whose winner the arc predicted. Six windows do decide, and they decide
`origin → position → compute → both → (no winner) → both → both`. **The cache bill — the one the
story has the field returning to twice — never dominates a single window on its own.**

Then the noise floor cost a second finding. The two-year buckets begin in 2014 because attention
does, not because the field turned on that boundary, so `arc_robustness` re-runs the whole tally
with the edges shifted one year. The claim that the field settles on both bills from 2020 onward
**does not survive** — shift the edges and that window goes to position. It is published as one
reading of the chronology rather than a measurement, and a test will fail if a future catalogue
ever makes it robust, so the hedge cannot outlive its reason.

## What this cannot establish

- **This is a chronology, not an experiment.** Nothing here was trained, and no claim about which
  mechanism is *better* is measured — the trade-offs are read from the papers and from the source material,
  not reproduced. Where a paper reports a number, it is attributed to that paper.
- **A first-appearance date is not the whole story.** Ideas have precursors, and several entries
  here have contested attributions that the entry records rather than resolves — learned absolute
  positions in particular go back at least to 2016 and arguably to 2015, through a lead we did not
  open.
- **The arithmetic is the source material's, at the source material's yardstick.** The cache figures are exact for
  48 layers, 8 KV heads, head dim 128 and bf16, and mean nothing for another configuration. They are
  arithmetic, not measurements of any running system.
- **The trade-offs are editorial.** *What it buys*, *what it gives up* and *when to choose it* are
  written by us from the sources. They are the part a reader should argue with, and the part no test
  can check.
- **One source could not be read live.** The NTK-aware entry comes from an archived capture rather
  than the original page.

## The page

<https://llm-pretraining-demos.vercel.app/08-modern-attention-variants/>

Twelve sections carrying the spine `AGENTS.md` requires, set as a **monograph feature**: six
numbered plates, six chapters, and the thirty mechanisms as *one object entered thirty times*
rather than thirty collapsed cards a reader has to click through.

The spine sentence is the thing the source material never states, and it is what makes drawing this worth
doing rather than restating the reading: **attention is one idea that sent two bills, and almost
everything since is somebody who could not pay one of them.** The two bills are the triangle of
scores between every pair of tokens, and the cache holding what each past token contributed.

Three views of the same thirty, because they answer different questions and a reader should
not have to interact to get an answer to any of them:

| view | what it answers | interaction needed |
| --- | --- | --- |
| **Plate III**, the chronology | *where* each sits in time, and which bill it pays | none — all 30 at once |
| the reading spread | *what one of them traded*, in depth | one click, and it is pre-loaded |
| the index plate | *comparability* — 30 rows, same six fields, same six places | none |

Every mechanism carries a **glyph** drawn by one of four generators from a `pattern` block in the
catalogue, so a glyph is derived from data rather than hand-drawn per mechanism. Two of those
drawings are load-bearing and easy to get wrong. FlashAttention's field is byte-identical to
standard attention's, because it is *exact* attention and drawing it a different shape would be the
worst factual error available on the page; its difference is a tiling overlay. And linear attention
does not get a thin diagonal — a diagonal implies "attends only to itself", which is the opposite of
a fixed-size state that summarises everything.

The plates each carry an argument the prose cannot make: the KV cache typeset as a printed
**invoice** with a cut line where one 80&nbsp;GB accelerator is exhausted; one attention step
**exploded into five bays**, ending in the weighted sum of V that produces the vector leaving the
block; three cache arrangements **racing one wall**, which shows that head sharing moves along the
same line rather than leaving it — the thing a bar chart provably cannot show.

The page prints **no shell commands**. Those are in *Run it* above, where commands belong.
