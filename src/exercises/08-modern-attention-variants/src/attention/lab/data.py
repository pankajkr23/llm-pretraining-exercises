"""The two kinds of data the lab trains on: real text, and a recall test built to be answerable.

**Real text** is exercise 09's frozen corpus, tokenized with exercise 02's frozen tokenizer, and it
is read *through exercise 09* rather than copied: a second copy of either file would be a second
thing to keep byte-identical, and exercise 06's manifests already pin the tokenizer's hash. The
corpus is small — tens of thousands of tokens — so a run of any length reads it more than once.
`corpus_report` says how many times, and `AGENTS.md` requires that number beside any comparison:
above one epoch a loss measures memorisation as much as it measures modelling.

**Associative recall** is the test that separates the families most sharply. A sequence lists
key–value pairs, then asks for some of the keys again; the right answer to each query is the value
that key was paired with. Full attention can look the pair up directly. A fixed-size recurrent
state has to have *kept* it, and with more pairs than the state can hold it cannot have. Being
synthetic, every answer is correct by construction and chance accuracy is known exactly.

The recall layout, with `p` pairs and `q` queries:

    k1 v1 k2 v2 … kp vp  SEP  q1 a1 q2 a2 … qq aq

- id `0` is `SEP`; keys are ids `1 … n_keys`; values are ids `n_keys + 1 … n_keys + n_values`, so a
  key can never be mistaken for a value.
- The `p` keys in one sequence are **distinct** (drawn without replacement), so every query has
  exactly one right answer. Values are drawn with replacement and may repeat.
- The `q` queries are distinct keys drawn from that sequence's own pairs, in random order, and each
  `a` is the value its key was paired with.
- The model reads `inputs = sequence[:-1]` and is scored only where it is reading a query, against
  the answer that follows it. Every other target is `IGNORE`, so the loss and the accuracy are
  about recall alone, not about predicting random keys.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

#: The target value the loss skips — PyTorch's `cross_entropy` default `ignore_index`.
IGNORE = -100

#: The separator between the pairs and the queries.
SEP = 0

#: One extra token per language-model sequence, so `seq_len` inputs have `seq_len` targets.
LM_EXTRA_TOKEN = 1


def _lm_config(window: int) -> Any:
    """Exercise 09's configuration with the sequence length set to `window`.

    Exercise 09's `Config` is reused rather than a lookalike, because `_corpus` checks the
    tokenizer's ids against its `vocab_size` (10,001: the 10,000-entry vocabulary plus exercise
    09's padding row), and a copy of that number here would be a second one to keep in step.
    """
    from lossheads.config import Config

    if window < 2:
        raise ValueError(f"a language-model window needs at least two tokens, got {window}")
    return replace(Config(), seq_len=window)


def lm_vocab_size() -> int:
    """Rows the embedding and head need for the frozen corpus — exercise 09's `vocab_size`."""
    from lossheads.config import Config

    return Config().vocab_size


def lm_batches(seq_len: int, batch: int, steps: int, seed: int) -> Tensor:
    """Real-text batches, `[steps, batch, seq_len + 1]` token ids.

    Each row holds `seq_len + 1` consecutive tokens so that inputs `row[:-1]` and targets `row[1:]`
    are both `seq_len` long. The rows are exercise 09's `_corpus`: the frozen text cut into
    consecutive windows (tiled when the run needs more than the corpus holds) and shuffled with
    `seed`. The same `seed` therefore gives every variant the same batches in the same order.
    """
    from lossheads.training import _corpus

    if batch < 1 or steps < 1:
        raise ValueError("batch and steps must both be at least 1")
    window = seq_len + LM_EXTRA_TOKEN
    tokens = _corpus(_lm_config(window), steps * batch, seed)
    return tokens.reshape(steps, batch, window)


def corpus_report(seq_len: int, sequences: int) -> dict[str, Any]:
    """How much text there is, how much a run reads, and so how many epochs.

    Exercise 09's `corpus_facts`, called with the window `lm_batches` actually cuts
    (`seq_len + 1`), plus the fields that say which window that was. `epochs` is
    `tokens_consumed / corpus_tokens`; above `1.0` the report says so in `warning`.
    """
    from lossheads.training import corpus_facts

    window = seq_len + LM_EXTRA_TOKEN
    facts = dict(corpus_facts(_lm_config(window), sequences))
    facts["seq_len"] = seq_len
    facts["window"] = window
    facts["sequences"] = sequences
    if facts["epochs"] > 1.0:
        facts["warning"] = (
            f"the run reads the corpus {facts['epochs']:.2f} times over, so its losses measure "
            "memorisation as well as modelling; compare variants with each other, not with a "
            "held-out loss"
        )
    return facts


@dataclass(frozen=True)
class RecallBatches:
    """Associative-recall data; the layout is in the module docstring.

    Attributes:
        inputs: `[steps, batch, 2p + 2q]` token ids the model reads.
        targets: Same shape; the value that answers the query at that position, else `IGNORE`.
        pairs: Key–value pairs per sequence (`p`).
        queries: Queries per sequence (`q`).
        n_keys: Distinct key ids available.
        n_values: Distinct value ids available.
    """

    inputs: Tensor
    targets: Tensor
    pairs: int
    queries: int
    n_keys: int
    n_values: int

    @property
    def vocab_size(self) -> int:
        """Ids used: the separator, the keys and the values."""
        return 1 + self.n_keys + self.n_values

    @property
    def chance(self) -> float:
        """Accuracy of guessing a value uniformly — the floor any accuracy is read against."""
        return 1.0 / self.n_values


def recall_batches(
    batch: int,
    steps: int,
    pairs: int,
    queries: int | None = None,
    n_keys: int | None = None,
    n_values: int = 16,
    seed: int = 0,
) -> RecallBatches:
    """Seeded associative-recall batches.

    Args:
        batch: Sequences per step.
        steps: Number of batches.
        pairs: Key–value pairs per sequence.
        queries: Queries per sequence, at most `pairs`; defaults to `pairs`.
        n_keys: Key ids to draw from, at least `pairs`; defaults to `max(2 * pairs, 16)`.
        n_values: Value ids to draw from.
        seed: The only source of randomness; the same arguments always give the same tensors.

    Returns:
        A `RecallBatches`.
    """
    queries = pairs if queries is None else queries
    n_keys = max(2 * pairs, 16) if n_keys is None else n_keys
    if min(batch, steps, pairs, queries, n_values) < 1:
        raise ValueError("batch, steps, pairs, queries and n_values must all be at least 1")
    if queries > pairs:
        raise ValueError(f"cannot ask {queries} distinct queries about {pairs} pairs")
    if n_keys < pairs:
        raise ValueError(f"{pairs} distinct keys need at least that many key ids, got {n_keys}")

    generator = torch.Generator().manual_seed(seed)
    total = steps * batch
    # argsort of uniform noise is a uniformly random permutation; its first `pairs` columns are a
    # draw without replacement, which is what makes every key in a sequence distinct.
    keys = torch.rand(total, n_keys, generator=generator).argsort(dim=-1)[:, :pairs] + 1
    values = torch.randint(0, n_values, (total, pairs), generator=generator) + 1 + n_keys
    asked = torch.rand(total, pairs, generator=generator).argsort(dim=-1)[:, :queries]
    query_keys = keys.gather(1, asked)
    answers = values.gather(1, asked)

    length = 2 * pairs + 1 + 2 * queries
    sequence = torch.empty(total, length, dtype=torch.long)
    sequence[:, 0 : 2 * pairs : 2] = keys
    sequence[:, 1 : 2 * pairs : 2] = values
    sequence[:, 2 * pairs] = SEP
    sequence[:, 2 * pairs + 1 :: 2] = query_keys
    sequence[:, 2 * pairs + 2 :: 2] = answers

    inputs = sequence[:, :-1]
    targets = torch.full_like(inputs, IGNORE)
    # The query sits at 2p + 1 + 2i and its answer is the very next token, so that position's
    # target is the answer.
    targets[:, 2 * pairs + 1 :: 2] = answers
    shape = (steps, batch, length - 1)
    return RecallBatches(
        inputs=inputs.reshape(shape),
        targets=targets.reshape(shape),
        pairs=pairs,
        queries=queries,
        n_keys=n_keys,
        n_values=n_values,
    )
