"""How many vocabulary tokens each position scheme makes permanently indistinguishable.

This is the concrete cost of v1's `d_p` truncation, and it is larger and more mundane than the
phrase "sovereign risk" suggests. A one-hot position factor addresses exactly `d_p` positions, so
every byte past `d_p` is DISCARDED — and two tokens sharing a `d_p`-byte prefix are then the same
vector. Not approximately: bit for bit, forever, with no error signal anywhere in training.

On the repo's frozen 10k tokenizer at `d_p = 32`: **407 tokens (4.07%) in 75 groups**. The largest
group is **83 distinct tokens** all sharing the prefix `](https://hi.wikipedia.org/wiki/`.

Counted from the byte strings themselves rather than from codes, because a v1 collision is exactly
"same first `d_p` bytes" and involves no floating point at all. `collisions_by_code` cross-checks
that against the actual vectors, which is how the two disagreeing would be caught.
"""

import numpy as np

from embeddings.codec import code
from embeddings.config import KroneckerConfig


def truncation_groups(byte_strings: list[bytes], d_p: int) -> list[list[int]]:
    """Token ids grouped by shared `d_p`-byte prefix, keeping only groups with more than one member.

    This is what `onehot` positions conflate, derived from the bytes with no codec involved.
    """
    groups: dict[bytes, list[int]] = {}
    for i, bs in enumerate(byte_strings):
        groups.setdefault(bs[:d_p], []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def colliding_tokens(byte_strings: list[bytes], d_p: int) -> int:
    """How many tokens sit in a truncation group. The headline count."""
    return sum(len(g) for g in truncation_groups(byte_strings, d_p))


def collisions_by_code(byte_strings: list[bytes], cfg: KroneckerConfig, decimals: int = 6) -> int:
    """The same count, measured from the CODES — the cross-check on `colliding_tokens`.

    Slower and scheme-general: it works for `wrap` and `fourier`, where "same prefix" is not the
    right question because neither truncates. Rounding to `decimals` before comparing keeps two
    genuinely identical codes from being separated by float noise.
    """
    seen: dict[bytes, int] = {}
    bad: set[int] = set()
    for i, bs in enumerate(byte_strings):
        key = np.round(code(bs, cfg), decimals).tobytes()
        if key in seen:
            bad.add(i)
            bad.add(seen[key])
        else:
            seen[key] = i
    return len(bad)


def cosine(a: bytes, b: bytes, cfg: KroneckerConfig) -> float:
    """Cosine between two tokens' codes. `1.0` means the codec cannot tell them apart at all."""
    u, v = code(a, cfg), code(b, cfg)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
