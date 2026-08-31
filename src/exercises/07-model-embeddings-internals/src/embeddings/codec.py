"""The forward code, three position schemes, and the analytic inverse of z-normalisation.

v1's codec, Eq. 1:

    kappa(b) = (1 / sqrt(L)) * sum_{p<L}  c_{b_p}  (x)  pos(p)

with `c` a one-hot over the 256 byte values, `pos(p)` a factor over byte positions, and then a
per-token z-normalisation over all `D = 256 * d_p` coordinates.

Everything here is sparse and never materialises a `V x D` matrix. A token contributes at most `L`
non-zero coordinates out of 8,192, so the code is not merely sparse but BLOCK-one-hot: reshaped to
`(d_p, 256)` it is a stack of one-hot rows. It *is* the byte string, which is why `decode` can
invert it exactly rather than approximately.

Three position schemes, and the choice between them is measured rather than argued:

    onehot   v1's. Orthogonal, trains well, and DISCARDS every byte past `d_p` -- which makes 407
             of the repo's 10,000 tokens permanently identical (see `collisions`).
    wrap     position `p` lands in slot `p % d_p` carrying a fixed sign from `p // d_p`. Keeps the
             block-one-hot structure, has no length limit, removes every collision, and trains
             BETTER than onehot. Its cost is that folding records a multiset rather than a sequence,
             so blind byte recovery past `d_p` is impossible -- `decode.fold_is_order_lossy`
             demonstrates that by construction.
    fourier  RoPE-style geometric frequencies. Length-free and collision-free, but the position
             vectors are not orthogonal and it trained measurably WORSE than doing nothing. Kept
             because a negative result that is deleted gets re-discovered.
"""

from dataclasses import dataclass

import numpy as np

from embeddings.config import BYTE_VALUES, KroneckerConfig

#: Fixed seed for the wrap signs. They must be identical between the run that trains a model and any
#: later run that decodes from it, so this is deliberately not configurable.
_WRAP_SEED = 1234


def wrap_signs(n_wraps: int, d_p: int) -> np.ndarray:
    """`(n_wraps, d_p)` array of +/-1, deterministic.

    Without them, position `p` and position `p + d_p` are indistinguishable and any anagram across
    the wrap boundary collides. They block the swaps whose two levels disagree in sign -- 15 of 32
    slots, measured. They do not block all of them, which is the limit `decode` documents.
    """
    rng = np.random.default_rng(_WRAP_SEED)
    return np.where(rng.random((max(n_wraps, 1), d_p)) < 0.5, -1.0, 1.0)


def fourier_positions(d_p: int, max_len: int, base: float = 10_000.0) -> np.ndarray:
    """`(max_len, d_p)` unit-norm position features on a geometric frequency ladder."""
    half = d_p // 2
    omega = base ** (-2.0 * np.arange(half) / d_p)
    ang = np.arange(max_len)[:, None] * omega[None, :]
    f = np.empty((max_len, d_p))
    f[:, 0::2] = np.cos(ang)
    f[:, 1::2] = np.sin(ang)
    return f / np.sqrt(half)


def atoms(
    bs: bytes, cfg: KroneckerConfig, table: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The non-zero coordinates of `kappa(bs)` before z-normalisation, as `(indices, values)`.

    Args:
        bs: The token's UTF-8 bytes.
        cfg: Dimensions and position scheme.
        table: Precomputed signs (`wrap`) or position features (`fourier`). Built on demand when
            omitted, which is convenient for one token and wasteful for a vocabulary.

    Returns:
        `(indices, values)` into a `D`-dimensional code. Duplicate indices are already summed, so a
        caller may scatter them directly.
    """
    d_p = cfg.d_p
    n = len(bs)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)

    if cfg.positions == "onehot":
        # Truncation lives here, and it is the single most consequential line in v1's codec.
        live = min(n, d_p)
        vals = np.frombuffer(bs[:live], dtype=np.uint8)
        idx = np.arange(live, dtype=np.int64) * BYTE_VALUES + vals
        return idx, np.full(live, 1.0 / np.sqrt(live))

    if cfg.positions == "wrap":
        signs = wrap_signs(n // d_p + 1, d_p) if table is None else table
        p = np.arange(n)
        slot = p % d_p
        idx = slot * BYTE_VALUES + np.frombuffer(bs, dtype=np.uint8)
        val = signs[p // d_p, slot] / np.sqrt(n)
        # Two positions can share a slot AND a byte; sum them rather than dropping one.
        uniq, inv = np.unique(idx, return_inverse=True)
        return uniq, np.bincount(inv, weights=val, minlength=uniq.size)

    feats = fourier_positions(d_p, n) if table is None else table[:n]
    # Byte-major here: the code is no longer one-hot per slot, so the block layout buys nothing.
    grid = np.zeros((BYTE_VALUES, d_p))
    np.add.at(grid, np.frombuffer(bs, dtype=np.uint8), feats)
    flat = (grid / np.sqrt(n)).reshape(-1)
    nz = np.flatnonzero(flat)
    return nz, flat[nz]


def code(bs: bytes, cfg: KroneckerConfig) -> np.ndarray:
    """The dense `D`-vector for one token, z-normalised when `cfg.znorm`. For tests and teaching."""
    idx, val = atoms(bs, cfg)
    x = np.zeros(cfg.code_width)
    x[idx] = val
    if not cfg.znorm:
        return x
    return (x - x.mean()) / (x.std() + 1e-12)


def znorm_stats(sum_v: np.ndarray, sum_v2: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Population mean and standard deviation of a sparse code over its `width` coordinates.

    z-normalisation is AFFINE in the code, and both moments follow from the non-zero values alone.
    That is what makes it analytically invertible and what lets the induced embedding be computed by
    a gather plus a rank-one correction, instead of by building the code.
    """
    mu = sum_v / width
    var = np.maximum(sum_v2 / width - mu**2, 1e-30)
    return mu, np.sqrt(var)


@dataclass(frozen=True)
class Encoded:
    """What `encode` produced, including the two moments needed to invert z-normalisation.

    Carrying the moments is the point. Deriving them from `lengths` alone works for `onehot`, where
    every atom is `+1/sqrt(L)`, and is WRONG for `wrap`, whose atoms are signed and whose sum
    therefore varies per token. Returning what was actually computed removes a whole class of
    scheme-specific special cases from the inverse.

    Attributes:
        h: `(V, d_model)` projected embeddings.
        lengths: Code atoms per token — `min(len(bs), d_p)` for `onehot`, `len(bs)` for `wrap`.
        sum_v: Sum of the code's non-zero values, per token.
        sum_v2: Sum of their squares, per token.
    """

    h: np.ndarray
    lengths: np.ndarray
    sum_v: np.ndarray
    sum_v2: np.ndarray


def encode(byte_strings: list[bytes], w: np.ndarray, cfg: KroneckerConfig) -> Encoded:
    """`h = znorm(kappa) @ W` for a whole vocabulary, computed sparsely.

    Never forms the `V x D` code matrix: each token touches at most `L` rows of `W`, and z-norm
    contributes a rank-one correction built from `W`'s column sums.
    """
    width, d_model = w.shape
    if width != cfg.code_width:
        raise ValueError(f"W has {width} rows, config says D = {cfg.code_width}")
    table = _table_for(byte_strings, cfg)

    n = len(byte_strings)
    raw = np.zeros((n, d_model), dtype=np.float64)
    sum_v = np.zeros(n)
    sum_v2 = np.zeros(n)
    lengths = np.zeros(n, dtype=np.int64)
    for i, bs in enumerate(byte_strings):
        idx, val = atoms(bs, cfg, table)
        lengths[i] = max(idx.size, 1)
        if idx.size:
            raw[i] = val @ w[idx]
            sum_v[i] = val.sum()
            sum_v2[i] = (val**2).sum()
    if not cfg.znorm:
        return Encoded(raw, lengths, sum_v, sum_v2)
    mu, sd = znorm_stats(sum_v, sum_v2, width)
    h = (raw - mu[:, None] * w.sum(0)[None, :]) / sd[:, None]
    return Encoded(h, lengths, sum_v, sum_v2)


def targets_from_h(enc: Encoded, w: np.ndarray, cfg: KroneckerConfig) -> np.ndarray:
    """Undo z-normalisation and the `1/sqrt(L)` scale, leaving a sum of unit-coefficient atoms.

    After this, `t_i = sum_p W[atom_p]` (signed, for `wrap`), which is exactly the problem `decode`
    solves. The inverse is analytic and exact to float precision — measured max error **7.0e-07** —
    because both z-norm moments are closed forms in the code's own non-zero values, which `Encoded`
    carries rather than re-deriving.
    """
    if cfg.znorm:
        mu, sd = znorm_stats(enc.sum_v, enc.sum_v2, w.shape[0])
        t = enc.h * sd[:, None] + mu[:, None] * w.sum(0)[None, :]
    else:
        t = enc.h
    return t * np.sqrt(enc.lengths.astype(np.float64))[:, None]


def _table_for(byte_strings: list[bytes], cfg: KroneckerConfig) -> np.ndarray | None:
    """Precompute the per-scheme lookup once for a whole vocabulary."""
    if cfg.positions == "onehot":
        return None
    max_len = max((len(b) for b in byte_strings), default=1)
    if cfg.positions == "wrap":
        return wrap_signs(max_len // cfg.d_p + 1, cfg.d_p)
    return fourier_positions(cfg.d_p, max(max_len, 1))
