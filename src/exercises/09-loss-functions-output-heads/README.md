# 09 · Loss functions and output heads

**Almost everything that can go wrong between a model's output and its loss goes wrong silently, and
several of the bugs make the number look better.** This exercise builds the few lines in between and
makes each of them observable — the shift, the masks, the denominator, the head, and the memory the
whole thing costs.

## How to read this

- **Meeting this for the first time** — read [What this is](#what-this-is), which explains the
  problem before any of the machinery.
- **Changing the code** — start at [How the pieces fit](#how-the-pieces-fit), then
  [Run it](#run-it).
- **Deciding whether to believe it** — go to [The evidence](#the-evidence), then
  [What this cannot establish](#what-this-cannot-establish).

## What this is

A language model is never told the right answer by anybody. The answer is simply *the next token*,
so the entire training signal is produced by shifting the text against itself by one position. Three
lines of tensor code do it, and four separate things go wrong in them:

- **The shift can go the wrong way**, handing the model its own input as the answer. The loss then
  *falls*, beautifully, and the model has learned to copy.
- **Padding can be counted.** Short sequences are padded to a common length, and padding is trivially
  predictable — so scoring it makes the number better while the model gets worse.
- **A packed document boundary can be predicted across.** Packing several documents into one
  sequence is standard practice; the join between them is two texts with no relationship, and
  training on that pair asserts a continuity that does not exist.
- **The mean can use the wrong denominator** — dividing by every position rather than by the ones
  that counted, so each batch is scaled by whatever fraction of it happened to be real.

None of the four raises an exception. The only defence is to make the intermediate state visible,
which is what this exercise is: shapes printed, targets printed **as strings** rather than ids, and
a contributing-token count beside every loss so a number that moved can be traced to the reason.

Two costs sit on the other end of the same pipe. The **output head** is the one matrix sized by the
tokenizer rather than by the model, so its share of the parameters grows as the model gets smaller;
tying it to the embedding removes those parameters entirely, at the price of forcing a token's input
vector and its scoring direction to be the same thing. And the **logits tensor** it produces is
larger than the hidden states that made it by exactly the ratio of vocabulary to width — a tensor
whose only purpose is to collapse into one number.

## How the pieces fit

| module | owns |
| --- | --- |
| `config.py` | every dimension a measurement here is taken at, and the parameter arithmetic |
| `model.py` | a small pre-norm transformer trunk — hidden states, and no head at all |
| `tokenizer.py` | exercise 02's frozen BPE, so targets can be read as text rather than as ids |
| `shift.py` | the `t+1` and `t+k` slices, the string table, and the off-by-one kept on purpose |
| `masks.py` | padding, packed-document boundaries, and the contributing count that evidences both |
| `losses.py` | masked cross-entropy, perplexity, chunking, and two knobs on the objective |
| `heads.py` | tied and untied heads, and what each costs |

**The trunk owns no output head.** That split is what lets one trunk feed one head in the ordinary
case and two heads when a second predicts further ahead, without a different model for each.

`torch` is the `train` extra. `config.py` and the parameter arithmetic in `heads.py` need none of
it, so a fresh clone can price an output head without installing the wheels.

## Run it

```bash
uv sync --all-packages --extra train
uv run pytest src/exercises/09-loss-functions-output-heads
```

Without `--extra train` the tensor tests **skip** and the suite still reports green — so pass the
flag, or you are measuring less than you think. The arithmetic tests run either way, deliberately,
so the ordinary CI job is not collecting an empty file.

## The evidence

**Two kinds of claim live here, and they are graded differently.**

The first kind is an **equivalence**, asserted rather than described. Each is written twice — once
at the setting where it must be a no-op, once away from it — because a function that ignored its
argument entirely would pass the first half:

| claim | where it is pinned |
| --- | --- |
| label smoothing at `epsilon = 0` **is** cross-entropy | `test_label_smoothing_at_zero_is_plain_cross_entropy` |
| a z-loss weight of `0` **is** cross-entropy | `test_a_zero_z_loss_weight_is_plain_cross_entropy` |
| chunked cross-entropy **is** the unchunked value | `test_chunked_cross_entropy_equals_the_unchunked_value` |
| a tied head **is** a linear map with the embedding's weights | `test_a_tied_head_scores_exactly_as_a_linear_layer_with_those_weights` |
| cross-entropy cannot see a constant logit shift; z-loss can | `test_the_z_loss_penalises_logit_scale_that_the_base_loss_cannot_see` |

**One of those twins overturned its own first version, and the correction is the better result.**
Label smoothing at `epsilon = 0.1` on random logits moved the loss by **0.0003**: 4.6337 to 4.6340.
The tempting fix is a wider tolerance. The honest reading is that a near-uniform model has almost no
confidence to penalise, and penalising confidence is what smoothing *is* — so it is nearly a no-op
exactly when there is nothing to smooth. The test now asserts both halves: sharp logits move a lot,
flat ones barely move.

A second one has already been corrected the same way. An early test claimed the head was *most* of a
128-wide model. At eight blocks it is **44.9%** — 1,280,128 parameters against a body of 1,572,864.
The claim survives without the overstatement, and it is now expressed as the comparison it actually
is rather than as a percentage that has to be maintained by hand.

The second kind of claim is a **measured number from a run**, and those are not here yet: the harness
that produces them is stage 10 of `PROGRESS.md`. When they land they will be written to
`results/` and rendered from there, never typed into this file.

## What this cannot establish

**It cannot tell you which loss trains a better model.** Nothing here has trained anything yet.
Every result above is an identity between two ways of computing a number, verified on synthetic
logits — a statement that the arithmetic is right, and no statement at all about downstream quality.

**The parameter arithmetic is a ratio, not a measurement.** `body_params` uses the standard
`12 · d_model²` per block, which is approximate by construction. The tests assert the *direction*
the share moves, deliberately, because a hard-coded percentage is a number in prose and those go
stale — as one already did here.

**Perplexity is not comparable across tokenizers, and this exercise has exactly one.** A tokenizer
that splits text more finely is asked an easier question at each step and scores better while being
no better. Exercise 02 measured a real case. Any perplexity quoted here is a within-tokenizer signal
and not a scoreboard.

**Nothing is measured on a GPU.** The shapes are laptop-sized. Where a figure describes a tensor too
large for any accelerator, that is arithmetic carried from a larger configuration and labelled as
such — not something this exercise ran.
