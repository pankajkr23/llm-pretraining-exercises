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
