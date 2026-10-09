"""Batches of token windows from the corpus, reproducibly.

A batch is `(batch, seq_len + 1)` ids: the model reads the first `seq_len` and predicts the last
`seq_len`. Windows start at uniformly random offsets drawn from a seeded generator, so two runs with
the same seed read exactly the same tokens in the same order — the precondition for comparing two
schedules or two learning rates on anything but noise.

Validation windows are drawn once, from the validation split, and reused for every evaluation in a
run, so a change in validation loss is a change in the model, not in the sample.
"""

import numpy as np
import torch


def window_starts(n_tokens: int, seq_len: int, count: int, seed: int) -> np.ndarray:
    """`count` window start offsets, each leaving room for `seq_len + 1` ids.

    Args:
        n_tokens: Length of the token array.
        seq_len: Positions per window, excluding the shifted target.
        count: How many windows.
        seed: Generator seed.

    Returns:
        Offsets, as int64.
    """
    span = n_tokens - (seq_len + 1)
    if span < 1:
        raise ValueError(f"{n_tokens} tokens cannot hold a window of {seq_len + 1}")
    return np.random.default_rng(seed).integers(0, span, size=count, dtype=np.int64)


def gather(tokens: np.ndarray, starts: np.ndarray, seq_len: int) -> torch.Tensor:
    """Stack the windows starting at `starts` into a `(len(starts), seq_len + 1)` int64 tensor."""
    rows = [np.asarray(tokens[s : s + seq_len + 1], dtype=np.int64) for s in starts]
    return torch.from_numpy(np.stack(rows))


class Batches:
    """An endless, seeded stream of training batches.

    Args:
        tokens: The training split's ids (a memory map is fine).
        batch: Windows per batch.
        seq_len: Positions per window.
        seed: Seed; batch `i` is a function of `(seed, i)` only.
    """

    def __init__(self, tokens: np.ndarray, batch: int, seq_len: int, seed: int) -> None:
        """Keep the split and the shape; nothing is drawn until a step asks for its batch."""
        self.tokens, self.batch, self.seq_len, self.seed = tokens, batch, seq_len, seed

    def __call__(self, step: int) -> torch.Tensor:
        """The batch for 0-based `step`. Independent of every other step's batch."""
        starts = window_starts(
            len(self.tokens), self.seq_len, self.batch, self.seed * 1_000_003 + step
        )
        return gather(self.tokens, starts, self.seq_len)


def validation_set(tokens: np.ndarray, windows: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    """A fixed set of validation windows, drawn once."""
    return gather(tokens, window_starts(len(tokens), seq_len, windows, seed), seq_len)
