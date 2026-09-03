# RESULTS — 09 · Loss functions and output heads

**Generated from `results/*.json`. Do not edit by hand** — run `tools/render_results.py`. Every
number and every verdict here, including the words *above*, *lower* and *identical*, is read
from those files rather than written by anyone.

Configuration: `d_model` 256, 4 blocks, 4 heads, sequence 128, batch 8, vocabulary 10,001.

---

## The seven numbers

| # | what was asked | the number |
| --- | --- | --- |
| 1 | shapes, with each dimension named | logits are **39.1x** the hidden states |
| 2 | the shift, verified in strings | broken shift trains to **0.1784** against **4.1447** correct |
| 3 | padding masked | **48** of 254 contribute (206 dropped) |
| 4 | a packed boundary masked | **9.339547** masked against **9.341408** unmasked, 37 dropped |
| 5 | perplexity, untrained | **12,078.2** against a vocabulary of 10,001 |
| 6 | tied against untied head | **2,560,256** against **0** added parameters |
| 7 | peak memory, plain against chunked | **340.8 MiB** against **37.5 MiB** — **9.08x** |

### 1 · Shapes

| tensor | shape | what each dimension is |
| --- | --- | --- |
| `tokens` | `(8, 128)` | batch · position — the ids fed in |
| `hidden` | `(8, 128, 256)` | batch · position · width — one vector per position |
| `logits` | `(8, 128, 10001)` | batch · position · vocabulary — one score per token |
| `inputs` | `(8, 127)` | batch · position — last dropped, nothing follows it |
| `targets` | `(8, 127)` | batch · position — first dropped, nothing predicts it |
| `flat logits` | `(1016, 10001)` | position · vocabulary — batch folded away |
| `flat targets` | `(1016,)` | position — one correct id per position |

The logits are **39.1 times** the hidden states that produced them
— 10,001 vocabulary against 256 width. That ratio is the entire
subject of item 7. The trunk holds 5,752,576 parameters and owns **no output head**.

### 2 · The shift

At initialisation the correct shift scores 9.1768 and the off-by-one
9.4462 — within noise, because an untrained model is equally bad at both.
**Train them and the bug becomes visible in the worst possible way:** over
300 steps the broken model reaches **0.1784**
while the correct one is still at **4.1447**. The broken model's loss
is **lower** by 3.9664.

A model handed its own input as the answer learns to copy, and copying is easy. Nothing raises.

### 3 · Padding

206 of 254 positions were padding
(81.1%), leaving **48**
contributing. Padding is trivially predictable, so scoring it improves the number while the model
gets worse — the count is what makes that visible.

### 4 · The packed boundary

Two documents in one sequence, joining at position 29.
**37** positions cross a boundary and are dropped, moving the loss from
9.341408 to 9.339547.

**The difference is small and that is the finding**, not a disappointment: a handful of positions
barely moves an average, so nothing looks wrong. The gradient still asserts a continuation between
two texts with nothing to do with one another.

### 5 · Perplexity

| quantity | value |
| --- | --- |
| vocabulary | 10,001 |
| loss an untrained model must show, `ln(V)` | 9.2104 |
| loss measured | 9.3992 |
| perplexity measured | 12,078.2 |
| ratio to vocabulary | 1.208 |

Read perplexity as a count: the size of the uniform menu the model behaves as though it were
choosing from. **It is not comparable across tokenizers** — one that splits more finely is asked an
easier question at each step and scores better while being no better.

### 6 · The head

| arrangement | added parameters |
| --- | --- |
| untied | 2,560,256 |
| tied | 0 |
| untied, tying unavailable | 2,560,256 |

The head is **44.9%** of the parameters at this width, against a body of
3,145,728. Tying removes it entirely — and needs an input table with one row per token
to tie *to*, which is why the third row exists rather than being a rounding of the second.

2 dense heads cost **5,120,512** parameters together. That
is the honest price of Part 2.

### 7 · Peak memory

| path | peak above baseline | loss |
| --- | --- | --- |
| materialised | 340.78 MiB | 9.254968 |
| chunked (128 rows) | 37.53 MiB | 9.254969 |
| **ratio** | **9.08x** | losses **identical** |

4,096 rows against a 10,001 vocabulary — a logits tensor of
156.27 MiB in fp32. Baseline (an interpreter with torch loaded, and
subtracted from both) was 189.05 MiB.

**The ratio is only meaningful because the losses are identical.** Chunking is not an approximation;
a difference here would mean the two paths computed different things, not that one was cheaper.

The ratio has a noise floor, measured below rather than assumed.

---

## Part 2 — the `t+2` head

300 steps, Adam at 0.0003, batch 8 x
128 tokens.

**Corpus: this repository's own AGENTS.md, tokenized with exercise 02's BPE** — 35,941 tokens
(`sha256:19f24ce7db26e4f3`), against 307,200 token positions
consumed. That is **8.55 epochs**.

**So every loss below is a memorisation number, and saying so is not a caveat but the correct
reading.** A model that has seen the same text 8.5 times is not being measured on
its ability to generalise. Both findings survive it — each compares two models trained *identically*
on that same repeated text, so the repetition is held constant and cancels — but the absolute values
do not transfer to a run on fresh data, and a reader entitled to assume they might should be told
they cannot.

| head | final loss |
| --- | --- |
| `t+1` | 4.4537 |
| `t+2` | 5.4953 |
| sum | 9.9490 |

**Stated before the run: the further head should sit above the nearer one**, because predicting two
positions ahead is genuinely harder. It sits **above**, by +1.0416, and was higher
on **297 of 300** steps.

The losses simply add — that is the whole of multi-token prediction as an objective. The cost is
5,120,512 parameters against 2,560,256 for one head, which is the
argument against dense extra heads at a large vocabulary.

### The step count was varied before any of this was quoted

The number of steps is an arbitrary choice, so the whole run was repeated at other values. **These
are separate runs, not truncations** — the corpus builder produces exactly `steps x batch_size`
shuffled sequences, so a 60-step run sees different batches than the first 60 of a 300-step run,
and reading the short numbers off the long curve would answer a different question.

| steps | gap `t+2` minus `t+1` | further head higher | broken shift | correct shift |
| --- | --- | --- | --- | --- |
| 60 | +0.0199 | 57/60 | 3.0485 | 6.2076 |
| 150 | +0.3171 | 146/150 | 0.9144 | 5.2405 |
| 300 | +1.0416 | 297/300 | 0.1784 | 4.1447 |

The gap grows monotonically: **yes**. Every run found the further head harder:
**yes**. Every run found the broken shift lower: **yes**. So neither finding
is an artefact of where a run happened to stop.

### And the memory ratio has a noise floor

Peak resident set size is the operating system's number and it moves between runs. The same
measurement repeated 5 times gave **9.09x**, **9.10x**, **9.12x**, **9.11x**, **9.27x** — a spread of
**0.18** on a ratio of about 9. The losses agreed on every
repetition: **yes**.

**So the honest claim is "about 9x", and any comparison finer than that is reading
noise.**

---

*Every figure above, including the ones in this section, is read from `results/harness.json`,
`results/training.json` and `results/sensitivity.json`. The sensitivity numbers used to be typed
into this renderer, and one of them printed the 300-step correct shift as 4.15 while the generated
table sixty lines above read 4.1447 — the same quantity, twice, in one document. That is why they
are a run now.*
