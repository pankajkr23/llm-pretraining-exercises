# 08 · Modern attention variants — in the order they were launched

**Twenty-three ways of computing attention and of telling a model where a token sits, ordered by the
date each one actually appeared, with every date read from the primary source.**

Session 8's assignment is not to describe attention mechanisms. It is to put them in **chronological
order** and explain each as an answer to a problem that existed *at that moment* — because the order
shows something a list cannot:

> "When you lay them out on a timeline you can watch the field change its mind … You cannot see that
> from a list. You can see it from a timeline, and once you see it you can guess what comes next,
> which is the whole reason I am asking."

The graded axis is therefore the **dates**, and the instructor was explicit about the failure mode:

> "Your agent will happily invent a launch date and describe a technique it has half remembered.
> Check every date against the actual paper or release."

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

## What the two bills are

Attention charges twice, and the two bills grow differently:

```text
compute    grows with T²    every token scores against every other token
KV cache   grows with T     every token's key and value are kept for the next one
```

Both are closed forms, so this exercise computes them rather than quoting them. At Session 8's own
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
| `config.py` | the yardstick model every cost is computed against, taken from the session |
| `cache.py` | the two bills — KV bytes, `T²` scores, head sharing, sequence compression |
| `sources.py` | the citation model: what was read, from where, quoted verbatim, and when |
| `catalogue.py` | the mechanisms, their trade-offs, and the coverage list the assignment mandates |
| `timeline.py` | ordering, the gaps, and which bill each period was paying down |

`results/mechanisms.json` is the tracked evidence. Nothing derives it; it was assembled by hand from
primary sources and is validated by `catalogue.py`.

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

The assignment invites this — *"if you catch me in another one, tell me"* — so both are recorded
rather than quietly corrected.

- **The transformer is mis-dated.** The transcript says Vaswani "invented in 2018 and 17".
  *Attention Is All You Need* is `arXiv:1706.03762`, v1 **12 June 2017**.
- **DroPE is two different papers, and the transcript quotes the wrong one's title.** The technique
  described in class — pretrain with positional embeddings, drop them, recalibrate briefly — is
  *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings*,
  `arXiv:2512.12167`. The transcript's garbled "rotate position emitting for efficient" maps instead
  onto **DRoPE** with a capital R, `arXiv:2503.15029`, an autonomous-driving trajectory paper with
  no relation to the technique. The two names differ by one capital letter.

### One number that does not reproduce

The transcript says eight users at a 1M-token context need about **1 TB**. The session's own
formula, at the session's own yardstick, gives **1.57 TB**. Both are recorded; neither is published
alone. A smaller model, fewer KV heads or fp8 storage would each reconcile them, and the transcript
does not say which was meant.

### What the order shows

Derived by `timeline.pressure_by_period`, not asserted. The brief predicts a tidy sequence —
*"first it wants exactness, then it wants memory back, then it wants length, then it wants memory
back again"* — and the data is messier than that in a way worth reporting: **two of the six
two-year windows have no single dominant pressure at all.** In those periods the field was attacking
several bills at once, and a test fails if that ever stops being true, so the finding cannot quietly
become the tidy story.

Two things visible only on a date axis:

- **Attention is three years older than the Transformer.** Bahdanau's soft alignment is 2014; the
  2017 paper removed the recurrence around it rather than inventing attention.
- **Nobody attacked the cost for nearly two years.** Between the Transformer and Sparse Transformers
  there are 680 days in which the field used attention without trying to make it cheaper.

## What this cannot establish

- **This is a chronology, not an experiment.** Nothing here was trained, and no claim about which
  mechanism is *better* is measured — the trade-offs are read from the papers and from the session,
  not reproduced. Where a paper reports a number, it is attributed to that paper.
- **A first-appearance date is not the whole story.** Ideas have precursors, and several entries
  here have contested attributions that the entry records rather than resolves — learned absolute
  positions in particular go back at least to 2016 and arguably to 2015, through a lead we did not
  open.
- **The arithmetic is the session's, at the session's yardstick.** The cache figures are exact for
  48 layers, 8 KV heads, head dim 128 and bf16, and mean nothing for another configuration. They are
  arithmetic, not measurements of any running system.
- **The trade-offs are editorial.** *What it buys*, *what it gives up* and *when to choose it* are
  written by us from the sources. They are the part a reader should argue with, and the part no test
  can check.
- **One source could not be read live.** The NTK-aware entry comes from an archived capture rather
  than the original page.

## Where this is going

`web/` does not exist yet. When the page lands it must be registered in **two** places in the same
change — the landing card in `deploy/vercel/index.html` and `SPINE_ENFORCED` in
`tests/test_page_spine.py` — because each guard fails in both directions and an early entry is as
red as a missing one.
