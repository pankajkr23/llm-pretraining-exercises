"""Every dimension in one place, so no module invents its own.

`d_p` is the interesting one. It is the number of byte POSITIONS a one-hot position factor can
address, and it is simultaneously the source of v1's silent collisions (bytes past `d_p` are
discarded) and half of the code width `D = 256 * d_p`. Changing it changes what the codec can
represent and what the projection costs, and nothing else.
"""

from dataclasses import dataclass

#: Byte values. Not configurable — a byte is a byte.
BYTE_VALUES = 256


@dataclass(frozen=True)
class KroneckerConfig:
    """Dimensions of the codec and the head.

    Attributes:
        d_p: Byte positions the position factor can address. With `onehot` positions this is a hard
            truncation point; with `wrap` it is the number of slots positions fold onto.
        d_model: Model width. The codec is exactly invertible from `d_model >= 384` at `d_p = 32`.
        positions: `onehot` (v1), `wrap` (length-free, best measured loss) or `fourier`
            (length-free, measurably worse — kept because the negative result is the finding).
        n_buckets: Hash buckets for the byte n-gram block. `0` disables it. Quality tracks
            `vocab_size / n_buckets`, so this is a dial, not a constant to forget.
        znorm: Per-token z-normalisation, as v1 specifies. Affine in the code, so invertible.
    """

    d_p: int = 32
    d_model: int = 384
    positions: str = "wrap"
    n_buckets: int = 8192
    znorm: bool = True

    def __post_init__(self) -> None:
        """Reject a scheme name that does not exist, rather than silently behaving like v1."""
        if self.positions not in ("onehot", "wrap", "fourier"):
            raise ValueError(f"unknown position scheme {self.positions!r}")
        if self.d_p < 1:
            raise ValueError(f"d_p must be positive, got {self.d_p}")

    @property
    def code_width(self) -> int:
        """`D = 256 * d_p` — the width of the fixed code, independent of the vocabulary."""
        return BYTE_VALUES * self.d_p
