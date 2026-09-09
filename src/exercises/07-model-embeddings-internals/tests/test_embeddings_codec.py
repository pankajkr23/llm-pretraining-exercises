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


# --- spc: one shared space of position directions -------------------------------------------

SPC = KroneckerConfig(d_p=32, d_model=384, positions="spc", reach=128, n_buckets=0)


def test_the_spc_frame_is_fixed_by_reach_and_not_by_the_batch():
    """A token must encode to the same thing alone as it does beside a long one.

    **This is the defect that shipped in the first draft.** The frame was built at
    `position_frame(d_p, len(token))` in `atoms` and `position_frame(d_p, max_len)` in
    `_table_for`, so the directions a token used depended on the longest token it happened to be
    encoded with — the frame is optimised jointly for exactly the number of positions asked for, so
    row `p` of a 64-row frame is not row `p` of a 128-row one. Nothing would have failed: the code
    is well formed either way, and the only symptom is a decoder correlating against directions the
    encoder never used, which reads as a recovery failure.
    """
    alone = codec.atoms(b"hello", SPC)
    crowded = codec.atoms(b"hello", SPC, codec._table_for([b"hello", b"x" * 120], SPC))
    assert np.array_equal(alone[0], crowded[0])
    assert np.allclose(alone[1], crowded[1])

    short = codec.position_frame(SPC.d_p, 64)
    full = codec.position_frame(SPC.d_p, SPC.reach)
    assert not np.allclose(short, full[:64]), (
        "if a frame built for 64 positions WERE the first 64 rows of one built for 128, the bug "
        "above would have been harmless — this asserts the premise, so the test above keeps\n"
        "meaning something"
    )


def test_a_token_past_the_reach_is_refused_rather_than_truncated():
    """Silently dropping the tail is what `onehot` does, and not doing it is the whole claim."""
    narrow = KroneckerConfig(d_p=32, d_model=384, positions="spc", reach=8, n_buckets=0)
    assert codec.atoms(b"x" * 8, narrow)[0].size > 0
    with pytest.raises(ValueError, match="does not fit"):
        codec.atoms(b"x" * 9, narrow)


def test_the_frame_is_deterministic_from_the_seed_alone():
    """It is never stored, so a frame that varied per process would decode nothing it encoded."""
    a = codec.position_frame(16, 40, seed=codec.SPC_SEED)
    b = codec.position_frame(16, 40, seed=codec.SPC_SEED + 1)
    assert np.array_equal(a, codec.position_frame(16, 40))
    assert not np.allclose(a, b), "a different seed must give a different frame"


def test_repulsion_beats_a_random_frame_and_respects_the_welch_bound():
    """Both directions: it must actually help, and it must not claim the impossible.

    A frame that scored *below* the Welch bound would mean `frame_coherence` is measuring the wrong
    thing — the bound is a theorem, not a target — so the second assertion is a check on the
    instrument rather than on the frame.
    """
    reach, d_p = 128, 32
    random_frame = np.random.default_rng(1).standard_normal((reach, d_p))
    random_frame /= np.linalg.norm(random_frame, axis=1, keepdims=True)
    repelled = codec.frame_coherence(codec.position_frame(d_p, reach))

    assert repelled < codec.frame_coherence(random_frame)
    assert repelled >= codec.welch_bound(reach, d_p)


def test_the_welch_bound_is_zero_only_while_the_directions_can_be_perpendicular():
    """At or below `d_p` an orthonormal frame exists, so the bound is silent; above it, it bites."""
    assert codec.welch_bound(32, 32) == 0.0
    assert codec.welch_bound(31, 32) == 0.0
    assert codec.welch_bound(33, 32) > 0.0
    assert codec.welch_bound(256, 32) > codec.welch_bound(33, 32)


def test_spc_keeps_the_code_width_while_removing_the_length_limit():
    """The trade the scheme exists to make, asserted as arithmetic rather than described."""
    assert SPC.code_width == ONEHOT.code_width == 256 * 32
    long = bytes(range(100))
    assert codec.atoms(long, ONEHOT)[0].size == 32, "onehot discards everything past d_p"
    assert codec.atoms(long, SPC)[0].size > 32, "spc writes every position into the shared space"
