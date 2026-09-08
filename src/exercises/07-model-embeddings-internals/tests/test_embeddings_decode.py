"""Invertibility -- the claim that refutes v1's stated reason for an untied head."""

import numpy as np
import pytest
from embeddings import codec, decode
from embeddings.config import KroneckerConfig

ONEHOT = KroneckerConfig(d_p=32, d_model=384, positions="onehot")


def _truth(byte_strings: list[bytes], d_p: int) -> np.ndarray:
    out = np.full((len(byte_strings), d_p), decode.ABSENT, dtype=np.int64)
    for i, bs in enumerate(byte_strings):
        b = np.frombuffer(bs[:d_p], dtype=np.uint8)
        out[i, : len(b)] = b
    return out


def _recover(byte_strings, w, cfg):
    enc = codec.encode(byte_strings, w, cfg)
    t = codec.targets_from_h(enc, w, cfg)
    guess, resid = decode.recover(t, enc.lengths, w, cfg)
    truth = _truth(byte_strings, cfg.d_p)
    live = truth >= 0
    return guess, resid, ((guess == truth) & live).sum(1) == live.sum(1)


def test_recovery_is_exact_at_the_measured_width(sample, projection):
    """d_model=384 recovered 100.00% of 2,000 real tokens. On a 600-token sample it must not
    regress -- this is the headline claim of the whole exercise."""
    _, _, ok = _recover(sample, projection, ONEHOT)
    assert ok.mean() == 1.0, f"recovery regressed to {ok.mean():.2%}"


def test_the_residual_certifies_the_answer(sample, projection):
    """A decoder that knows when it is right is qualitatively different from one that is usually
    right. `residual == 0` must coincide with `bytes are correct`, on every token."""
    _, resid, ok = _recover(sample, projection, ONEHOT)
    assert ((resid < 1e-8) == ok).all()


def test_coordinate_descent_beats_the_matched_filter_it_starts_from(sample, projection):
    """The matched filter ignores interference between positions; this is what cancels it. If the
    two ever tie, the coordinate-descent step has stopped doing anything."""
    enc = codec.encode(sample, projection, ONEHOT)
    t = codec.targets_from_h(enc, projection, ONEHOT)
    truth = _truth(sample, ONEHOT.d_p)
    live = truth >= 0
    mf = decode.matched_filter(t, enc.lengths, projection, ONEHOT.d_p)
    mf_ok = ((mf == truth) & live).sum(1) == live.sum(1)
    _, _, cd_ok = _recover(sample, projection, ONEHOT)
    assert mf_ok.mean() < cd_ok.mean()


def test_a_narrow_projection_loses_information_it_cannot_certify(sample):
    """The twin: at d_model=128 recovery must FAIL, and the certificate must report the failures
    rather than claiming success. A guard that only ever sees the passing case is not a guard."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((ONEHOT.code_width, 128))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    cfg = KroneckerConfig(d_p=32, d_model=128, positions="onehot")
    _, resid, ok = _recover(sample, w, cfg)
    assert ok.mean() < 0.99, "d_model=128 is expected to be lossy; it recovered everything"
    assert ((resid < 1e-8) == ok).all(), "the certificate must agree even when the decode fails"


def test_folding_is_order_lossy_by_construction():
    """`wrap` buys unbounded length by folding, and folding records a multiset rather than a
    sequence. Demonstrated by exhibiting two different byte strings with the same code -- not by a
    hit rate, because this is a statement about the code and not about any decoder."""
    cfg = KroneckerConfig(d_p=32, positions="wrap")
    assert decode.fold_is_order_lossy(cfg) < 1e-9


def test_fourier_positions_are_refused_rather_than_silently_wrong(sample, projection):
    """The code is not block-one-hot under Fourier positions, so this decoder does not apply.
    Returning plausible nonsense would be worse than failing."""
    cfg = KroneckerConfig(d_p=32, d_model=384, positions="fourier")
    with pytest.raises(ValueError, match="not block-one-hot"):
        decode.recover(np.zeros((2, 384)), np.array([3, 3]), projection, cfg)


WRAP = KroneckerConfig(d_p=32, d_model=384, positions="wrap")


def test_wrapped_positions_recover_exactly_up_to_d_p_bytes(sample, projection):
    """The claim four documents make, and which no tracked code demonstrated until now.

    `README.md`, `CLAUDE.md`, `PROGRESS.md` and the published page all say round-trip recovery under
    wrap is 100% to 32 bytes. `recover` accepted `wrap` — it rejects only `fourier` — and **every
    test drove it with `onehot`**, so the claim was never exercised. It was also false of the
    shipped decoder: `matched_filter` and `block_omp` take an argmax over `W`'s raw rows, and under
    wrap an atom enters as `sign * W[row]` with half the slots carrying `-1`, which inverts the
    argmax. Measured before the fix: **47.00%** on 200 vocabulary tokens at `d_model=768`.
    """
    short = [t for t in sample if 1 <= len(t) <= WRAP.d_p]
    assert len(short) > 100, "the fixture no longer has enough short tokens to make this a test"
    _, _, ok = _recover(short, projection, WRAP)
    assert ok.mean() == 1.0, f"wrap recovery regressed to {ok.mean():.2%} at or below d_p"


def test_the_wrap_dictionary_is_signed_and_an_unsigned_one_fails(sample, projection):
    """The twin. A guard nobody has watched fail is not a guard.

    Decoding wrap against `W`'s raw rows — which is what the code did — must score far below the
    sign-aware dictionary. If the two ever agree, either the signs have stopped being applied or
    `_dictionary_for` has stopped being called, and the test above would pass for the wrong reason.
    """
    from embeddings import decode as decode_module

    short = [t for t in sample if 1 <= len(t) <= WRAP.d_p]
    signed = decode_module._dictionary_for(projection, WRAP)
    assert not np.allclose(signed, projection), "the wrap dictionary is not signed at all"

    enc = codec.encode(short, projection, WRAP)
    t = codec.targets_from_h(enc, projection, WRAP)
    truth = _truth(short, WRAP.d_p)
    live = truth >= 0
    unaware, _ = decode.coordinate_descent(
        t,
        enc.lengths,
        projection,
        WRAP.d_p,
        decode.block_omp(t, enc.lengths, projection, WRAP.d_p),
    )
    rate = float((((unaware == truth) & live).sum(1) == live.sum(1)).mean())
    assert rate < 0.9, (
        f"an unsigned dictionary recovered {rate:.2%} of wrapped tokens, so this test is not "
        "measuring what it claims and the sign-awareness above proves nothing"
    )
