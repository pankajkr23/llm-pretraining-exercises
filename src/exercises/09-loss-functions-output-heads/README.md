# 09 · Loss functions and output heads

**Almost everything that can go wrong between a model's output and its loss goes wrong silently, and
the worst of it makes the number look better.** This exercise builds those few lines, makes each one
observable, and then trains a model with the bug on purpose to show what it costs.

The headline: a model handed its own input as the answer trains to a loss of **0.18** while the
correct one is still at **4.14**. Nothing raises. The curve looks like a triumph.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is), which explains the
  problem before any of the machinery.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it).
- **Deciding whether to believe it** — the numbers are in **[RESULTS.md](RESULTS.md)**, generated
  from `results/*.json` and never typed by hand. Then read
  [What this cannot establish](#what-this-cannot-establish).

## What this is

A language model is never told the right answer by anybody. The answer is simply *the next token*,
so the entire training signal is produced by shifting the text against itself by one position. Three
lines of tensor code do it, and four separate things go wrong in them:

- **The shift can go the wrong way**, handing the model its own input as the answer. It then learns
  to copy, which is easy, and the loss collapses.
- **Padding can be counted.** Short sequences are padded to a common length, and padding is
  trivially predictable — so scoring it improves the number while the model gets worse.
- **A packed document boundary can be predicted across.** Packing several documents into one
  sequence is standard practice; the join between them is two texts with no relationship, and the
  gradient there asserts a continuity that does not exist.
- **The mean can use the wrong denominator** — dividing by every position rather than by the ones
  that counted, so each batch is scaled by whatever fraction of it happened to be real.

None of the four raises an exception. The defence is to make the intermediate state visible: shapes
printed, targets printed **as strings** rather than ids, and a contributing-token count beside every
loss so a number that moved can be traced to a reason.

Two costs sit at the other end of the same pipe. The **output head** is the one matrix sized by the
tokenizer rather than by the model, so its share grows as the model shrinks; tying it to the
embedding removes those parameters entirely, at the price of forcing a token's input vector and its
scoring direction to be the same thing — and tying is *unavailable* to any architecture whose input
side has no per-token rows. And the **logits tensor** it produces is larger than the hidden states
that made it by exactly the ratio of vocabulary to width, for a tensor whose only purpose is to
collapse into one number.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | every dimension a measurement is taken at, and the parameter arithmetic |
| `model.py` | a small pre-norm transformer trunk — hidden states, and no head at all |
| `tokenizer.py` | exercise 02's frozen BPE, so targets can be read as text rather than as ids |
| `shift.py` | the `t+1` and `t+k` slices, the string table, and the off-by-one kept on purpose |
| `masks.py` | padding, packed-document boundaries, and the contributing count that evidences both |
| `losses.py` | masked cross-entropy, perplexity, two kinds of chunking, and two knobs |
| `heads.py` | tied, untied and tying-unavailable heads, plus the multi-token head |
| `memory.py` | peak memory for both loss paths, measured in isolated child processes |
| `harness.py` | one run producing every number, into `results/harness.json` |
| `training.py` | the short run the two findings need, into `results/training.json` |

**The trunk owns no output head.** That split is what lets one trunk feed one head in the ordinary
case and two heads when a second predicts further ahead, without a different model for each.

`torch` is the `train` extra. `config.py` and the parameter arithmetic in `heads.py` need none of
it, so a fresh clone can price an output head without installing the wheels.

## Run it

```bash
uv sync --all-packages --extra train

uv run python -m lossheads.harness    # the seven numbers -> results/harness.json
uv run python -m lossheads.training   # the two findings  -> results/training.json
uv run python src/exercises/09-loss-functions-output-heads/tools/render_results.py

uv run pytest src/exercises/09-loss-functions-output-heads
```

The harness prints as much as it computes — items 1 to 4 are about what a reader can *see*, so
running it and reading the output is the point rather than a diagnostic. Training takes about
35 seconds on a laptop.

Without `--extra train` the tensor tests **skip** and the suite still reports green — so pass the
flag, or you are measuring less than you think.

## The evidence

**Every measured number lives in [RESULTS.md](RESULTS.md)**, generated by `tools/render_results.py`
from the two JSON files a run writes. Nothing there is typed, including the verdict words: a test
flips the data and asserts the document's *conclusion* flips with it, because byte-equality alone
would pass on a template with "above" hard-coded.

Two findings are worth stating here, and both were predicted before they were measured.

**The off-by-one is invisible until you train, and then it is unmistakable — in the wrong
direction.** At initialisation the correct and broken shifts score within noise of each other,
because an untrained model is equally bad at predicting the next token and at copying the current
one. After 300 steps the broken model is at 0.18 and the correct one at 4.14. This is why the check
is to print the strings and read them, and never to watch the number go down.

**A head predicting two positions ahead sits above one predicting the next token, and stays there.**
Predicting further out is genuinely harder. It was higher on 297 of 300 steps.

**Both were checked against a different arbitrary choice before being quoted.** The step count is
the only arbitrary thing in the run, so it was varied — at 60, 150 and 300 steps both effects grow
monotonically, so neither is an artefact of where the run stopped. The memory ratio has a noise
floor too: five repetitions of the same measurement spread from 8.89x to 9.57x, so it is reported as
"about 9x" and no finer.

Alongside those, five **equivalences** are asserted rather than described — label smoothing at
`epsilon = 0`, a z-loss weight of `0`, and both kinds of chunking all give back plain cross-entropy;
a tied head is exactly a linear map with the embedding's weights. Each is written twice, once at the
no-op setting and once away from it, because a function that ignored its argument entirely would
pass the first half.

**Three corrections came out of building this**, and they are more useful than the successes:

- Chunked cross-entropy divided by the row count rather than the contributing count, so it
  disagreed with the unchunked loss on any masked input. Every test written on unmasked input passed
  either way, which is how that ships.
- The memory measurement was first written with `tracemalloc`, which is blind to torch: it reported
  **429 bytes** for an **81,928,192-byte** logits tensor. Both paths would have come back as noise
  and the ratio would have been the quotient of two noise figures.
- A test claimed the head was "most" of a 128-wide model. It is 44.9%. It had also inherited
  whatever `n_layer` defaulted to, so it silently became a different claim when that default moved.

## What this cannot establish

**It cannot tell you which loss trains a better model.** The training run exists to show two
specific effects, not to produce a good model. Nothing here is a quality comparison.

**300 steps is not a training curve.** Both findings are bounded by that number, which is why it is
stated beside them rather than chosen quietly.

**The memory numbers are CPU peak RSS at laptop shapes**, with a measured spread of 0.69 on a ratio
of about 9. Where a figure describes a tensor too large for any accelerator, that is arithmetic
carried from a larger configuration and labelled as such — not something this exercise ran.

**Perplexity is not comparable across tokenizers, and this exercise has exactly one.** A tokenizer
that splits text more finely is asked an easier question at each step and scores better while being
no better. Any perplexity here is a within-tokenizer signal, never a scoreboard.

**The parameter arithmetic is a ratio, not a measurement.** `body_params` uses the standard
`12 · d_model²` per block, which is approximate by construction. The tests assert the *direction*
the head's share moves, deliberately, because a hard-coded percentage goes stale — as one already
did here.
