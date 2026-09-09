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


WRAP = KroneckerConfig(d_p=32, d_model=384, positions="wrap", n_buckets=0)


def _wrap_table(vocabulary):
    return codec.wrap_signs(max(len(t) for t in vocabulary) // WRAP.d_p + 2, WRAP.d_p)


def test_the_recorded_length_is_positions_not_atoms(vocabulary):
    """`atoms` merges duplicate `(slot, byte)` pairs; the `1/sqrt(L)` scale was never per atom.

    Under `wrap`, two folded positions can land on the same slot carrying the same byte, and the
    merged non-zero count is then smaller than the position count. `encode` recorded the merged
    count and `targets_from_h` undid the scale with it, so the recovered target came back multiplied
    by `sqrt(nnz / L)`.

    Measured on the frozen vocabulary before the fix: **142 of 10,000 tokens** affected, worst case
    a 65-byte token merging to 48 atoms — a **14.07%** error on every coordinate. It was invisible
    in every published recovery figure because all of them were measured under `onehot`, where each
    position owns a distinct slot and no merge is possible.
    """
    table = _wrap_table(vocabulary)
    merging = [t for t in vocabulary if codec.atoms(t, WRAP, table)[0].size != len(t)]
    assert merging, "no vocabulary token merges under wrap; this test is not exercising the case"

    enc = codec.encode(merging[:40], np.eye(256 * WRAP.d_p, WRAP.d_model), WRAP)
    assert list(enc.lengths) == [len(t) for t in merging[:40]], (
        "encode recorded the merged atom count as the length; targets_from_h then undoes the "
        "1/sqrt(L) scale with the wrong L"
    )


def test_the_recovered_target_is_exact_for_a_token_that_merges(vocabulary, projection):
    """The end-to-end consequence: `t = sum_p sign * W[atom_p]`, to float precision.

    The twin of the test above, and the one that would have caught this without anyone knowing the
    word "merge" — it compares the whole round trip against a target built by hand.
    """
    table = _wrap_table(vocabulary)
    worst = max(vocabulary, key=lambda t: len(t) - codec.atoms(t, WRAP, table)[0].size)
    assert codec.atoms(worst, WRAP, table)[0].size < len(worst), "the fixture token does not merge"

    got = codec.targets_from_h(codec.encode([worst], projection, WRAP), projection, WRAP)[0]
    expected = np.zeros(projection.shape[1])
    for position, byte in enumerate(worst):
        slot = position % WRAP.d_p
        expected += table[position // WRAP.d_p, slot] * projection[slot * 256 + byte]

    live = np.abs(expected) > 1e-9
    assert live.any()
    ratio = got[live] / expected[live]
    assert np.allclose(ratio, 1.0, atol=1e-9), (
        f"the recovered target is scaled by {ratio.min():.6f}..{ratio.max():.6f} rather than 1.0"
    )
