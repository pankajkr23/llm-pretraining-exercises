"""The parameter arithmetic — including where this architecture stops paying for itself.

Two numbers decide whether any of this is worth doing, and both are simple enough that guessing them
is inexcusable:

    dense tied embedding      V * d          grows with the vocabulary
    Kronecker projection      D * d + 1      D = 256 * d_p, independent of V

So the crossover is exactly `V > D = 256 * d_p`. Below it this architecture COSTS parameters; above
it the saving is unbounded. At `d_p = 32` that threshold is 8,192 tokens — under any real
vocabulary, which is why the idea works at all — and at `d_p = 128`, which truncates nothing in the
repo's vocabulary, it is 32,768.

The n-gram lock-breaker is the honest complication. It is V-independent by construction (`m` is a
constant you choose), but measured quality tracks `V / m`, so holding quality as V grows means
growing `m`. Both ends of that dial are reported by `dial`, because quoting the fixed-`m` saving
while implying the grown-`m` quality would be the dishonest version of this result.
"""

from dataclasses import dataclass

from embeddings.config import BYTE_VALUES


@dataclass(frozen=True)
class Budget:
    """Parameter counts for one configuration.

    Attributes:
        dense_tied: `V * d`. One matrix serving both embedding and output head.
        v1: v1's projection plus the untied `d -> V` head it forces.
        v2: The tied Kronecker head — projection, one output scale, the `d x d` transform.
        ngram: The optional byte n-gram block, `m * d`.
    """

    dense_tied: int
    v1: int
    v2: int
    ngram: int

    @property
    def v2_total(self) -> int:
        """Everything the v2 head costs. Contains no term in `V`."""
        return self.v2 + self.ngram

    @property
    def saving(self) -> float:
        """How many times smaller the v2 head is than the dense tied baseline."""
        return self.dense_tied / self.v2_total


def budget(vocab_size: int, d_model: int, d_p: int = 32, n_buckets: int = 0) -> Budget:
    """Parameter counts for a vocabulary size and model width.

    Args:
        vocab_size: `V`. Appears only in `dense_tied` and in v1's untied head.
        d_model: Model width.
        d_p: Byte positions the code addresses; sets `D = 256 * d_p`.
        n_buckets: Hash buckets for the n-gram block; `0` omits it.
    """
    width = BYTE_VALUES * d_p
    return Budget(
        dense_tied=vocab_size * d_model,
        v1=width * d_model + vocab_size * d_model,
        v2=width * d_model + 1 + d_model * d_model,
        ngram=n_buckets * d_model,
    )


def crossover(d_p: int = 32) -> int:
    """Vocabulary size above which the bare Kronecker projection beats a dense tied embedding.

    Exactly `D = 256 * d_p`, comparing `D*d + 1` against `V*d`. Verified rather than asserted: at
    `V = 32,768` with `d_p = 128` the two are 25,165,825 and 25,165,824 — a difference of one, which
    is the output scale.

    Note what this does NOT include, because the difference is easy to quote wrongly. `Budget.v2`
    also carries the `d x d` output transform (`d^2`), and `Budget.ngram` may carry more, so the
    real break-even for a fully equipped head sits above this number. At `V = 1M, d = 768` the bare
    projection is 122.1x smaller than the dense tied baseline and the projection-plus-transform is
    111.6x. Both are true of different things; state which one you mean.
    """
    return BYTE_VALUES * d_p
