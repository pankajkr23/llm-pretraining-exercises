"""Every dimension this exercise measures against, in one place.

`AGENTS.md` asks for one `config.py` dataclass per exercise. Recording the configuration here
rather than inlining it is what makes "we reproduce the published number" checkable: change a field
and the test that reproduces it fails.

**The vocabulary size is the whole subject.** An output head is a `d_model × vocab` matrix, so it is
the one parameter block that scales with the vocabulary rather than with the model — and at small
`d_model` it rivals the entire body. Two vocabularies appear here and they do different jobs:
`vocab_size` is the one this exercise actually runs against, exercise 02's shipped BPE, because it
is a size this repository measured rather than a round number. `reference_vocab_size` and
`reference_d_model` are the course model's, carried only so the memory arithmetic can be quoted at
the scale where it stops being an abstraction.
"""

from dataclasses import dataclass

BYTES_PER_BF16 = 2
"""Bytes per element of a `bfloat16` logits tensor — the dtype the memory table is quoted in."""


@dataclass(frozen=True)
class Config:
    """The shapes and hyper-parameters every measurement in this exercise is taken at.

    Attributes:
        vocab_size: Rows in the output head. Exercise 02's shipped BPE vocabulary of 10,000,
            **plus one** — that tokenizer has no padding token, so `[PAD]` is ours and the head
            grows by exactly one row to hold it. Padding is a decision this exercise makes, not a
            property of the tokenizer it inherited.
        d_model: Model width. The head is `d_model × vocab_size` parameters.
        n_layer: Transformer blocks.
        n_head: Attention heads per block.
        seq_len: Sequence length a loss is computed over.
        batch_size: Sequences per batch.
        pad_id: Token id used to pad a short sequence. Never a prediction, never in the mean.
            It is `10_000`, one past the tokenizer's last real id, so it can never be confused
            with `[UNK]` — which is id `0` and is a genuine prediction the model may make.
        ignore_index: The value a masked target is set to, which the loss drops.
        label_smoothing: Mass moved off the target and spread across the vocabulary. `0.0`
            makes the smoothed loss *identical* to plain cross-entropy.
        z_loss_weight: Coefficient on the auxiliary penalty applied to `logsumexp(logits)`. `0.0`
            reduces to plain cross-entropy for the same reason.
        chunk_size: Rows per chunk when computing cross-entropy in pieces. Trades peak memory for
            a Python-level loop; the result must be identical to the unchunked value.
        horizons: Which future positions the heads predict. `(1, 2)` is what is asked for —
            the ordinary next-token head, plus one predicting `t+2`.
        reference_vocab_size: The course model's vocabulary, for the memory arithmetic only.
        reference_d_model: The course model's width, for the same reason.
        source: Where these values come from, so a reader can check them.
    """

    vocab_size: int = 10_001
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    seq_len: int = 128
    batch_size: int = 8
    pad_id: int = 10_000
    ignore_index: int = -100
    label_smoothing: float = 0.0
    z_loss_weight: float = 0.0
    chunk_size: int = 128
    horizons: tuple[int, ...] = (1, 2)
    reference_vocab_size: int = 131_072
    reference_d_model: int = 4_096
    source: str = (
        "vocab_size is exercise 02's shipped 10,000-entry BPE vocabulary plus one [PAD] row "
        "this exercise adds, because that tokenizer has none; the model shape is this "
        "exercise's own, chosen small enough to train on a laptop; reference_* are the course "
        "model's dimensions, carried only so the memory table can be quoted at that scale"
    )

    @property
    def head_params(self) -> int:
        """Parameters in an untied output head: one row per vocabulary entry."""
        return self.d_model * self.vocab_size

    @property
    def body_params(self) -> int:
        """A rough transformer body, for pricing the head against it.

        `12 · d_model²` per block is the standard count for attention plus a 4× MLP, and it is
        approximate on purpose — the point is the *ratio*, which is why an exact figure would imply
        a precision this comparison does not have. `count_parameters` in `model.py` reports the
        exact figure for the model actually built.
        """
        return 12 * self.d_model * self.d_model * self.n_layer

    @property
    def head_share(self) -> float:
        """Fraction of parameters the untied head accounts for."""
        return self.head_params / (self.head_params + self.body_params)

    @property
    def rows(self) -> int:
        """Token positions in one batch, before the shift drops one per sequence."""
        return self.batch_size * self.seq_len

    def logits_bytes(self, vocab_size: int | None = None) -> int:
        """Bytes a materialised `[B, T, V]` bf16 logits tensor occupies.

        This is the third bill and it lands in the last layer: the hidden states are `[B, T, D]`,
        so the logits are larger by exactly `vocab_size / d_model`. Pass
        `config.reference_vocab_size` to price the course model instead of ours.
        """
        return self.rows * (vocab_size or self.vocab_size) * BYTES_PER_BF16

    def chunked_logits_bytes(self, vocab_size: int | None = None) -> int:
        """Bytes the same computation peaks at when done `chunk_size` rows at a time."""
        return self.chunk_size * (vocab_size or self.vocab_size) * BYTES_PER_BF16
