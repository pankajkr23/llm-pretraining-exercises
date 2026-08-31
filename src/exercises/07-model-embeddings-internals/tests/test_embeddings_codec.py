"""The forward code, and the two properties everything downstream rests on."""

import numpy as np
import pytest
from embeddings import codec
from embeddings.config import KroneckerConfig

ONEHOT = KroneckerConfig(d_p=32, d_model=384, positions="onehot")
WRAP = KroneckerConfig(d_p=32, d_model=384, positions="wrap")


def test_the_code_is_block_one_hot_and_unit_norm_before_znorm():
    """`kappa` reshaped to `(d_p, 256)` is a stack of one-hot rows -- it IS the byte string.

    Stronger than generic sparsity, and it is what lets `decode` invert exactly rather
    than approximately, so it is worth asserting rather than assuming.
    """
    idx, val = codec.atoms(b"the", ONEHOT)
    assert idx.tolist() == [0 * 256 + ord("t"), 1 * 256 + ord("h"), 2 * 256 + ord("e")]
    assert np.allclose(val, 1 / np.sqrt(3))
    assert np.isclose(np.linalg.norm(val), 1.0)


def test_znorm_is_affine_so_the_moments_come_from_the_nonzeros_alone():
    """Must match numpy's own mean/std, or `targets_from_h` inverts the wrong map."""
    for token in (b"a", b"hello", b"a much longer token indeed"):
        _, val = codec.atoms(token, ONEHOT)
        mu, sd = codec.znorm_stats(val.sum(), (val**2).sum(), ONEHOT.code_width)
        dense = np.zeros(ONEHOT.code_width)
        dense[codec.atoms(token, ONEHOT)[0]] = val
        assert np.isclose(mu, dense.mean())
        assert np.isclose(sd, dense.std())


def test_onehot_truncates_and_wrap_does_not():
    """The single most consequential difference between the two schemes."""
    long = bytes(range(40))
    assert codec.atoms(long, ONEHOT)[0].size == 32  # bytes 32..39 discarded
    assert codec.atoms(long, WRAP)[0].size == 40  # all present, folded onto 32 slots


def test_encode_matches_the_literal_definition(sample, projection):
    """The sparse path must equal building the dense code and multiplying. Otherwise it is a
    different codec wearing the same name, and every measured number is about the wrong thing."""
    small = sample[:40]
    enc = codec.encode(small, projection, ONEHOT)
    literal = np.stack([codec.code(bs, ONEHOT) for bs in small]) @ projection
    assert np.allclose(enc.h, literal, atol=1e-7)


def test_targets_invert_znorm_exactly(sample, projection):
    """The twin of the test above: `targets_from_h` must undo what `encode` did."""
    small = sample[:40]
    enc = codec.encode(small, projection, ONEHOT)
    t = codec.targets_from_h(enc, projection, ONEHOT)
    expected = np.stack([projection[codec.atoms(bs, ONEHOT)[0]].sum(0) for bs in small])
    assert np.abs(t - expected).max() < 1e-6


def test_an_unknown_position_scheme_is_rejected():
    """A typo must fail loudly rather than silently behaving like v1."""
    with pytest.raises(ValueError, match="unknown position scheme"):
        KroneckerConfig(positions="rope")


def test_the_codec_check_can_actually_fail(sample, projection):
    """The deliberately-broken twin. A codec that ignored positions entirely would still be sparse,
    unit-norm and invertible-looking -- and would collide constantly. The equality test above has to
    reject it, or it is not testing what it claims."""
    small = sample[:40]
    enc = codec.encode(small, projection, ONEHOT)
    broken = np.stack(
        [projection[np.frombuffer(bs[:32], dtype=np.uint8).astype(np.int64)].sum(0) for bs in small]
    )
    assert not np.allclose(enc.h, broken, atol=1e-7)
