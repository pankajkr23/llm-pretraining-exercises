"""Integration: the ablation harness runs a mini-sweep offline and returns sane results."""

import pytest
from tokenization.ablate import Spec, compute_weights, run, sweep


def test_compute_weights_upsamples_smaller_corpora():
    corpora = {"big": "x" * 1000, "small": "y" * 100}
    flat = compute_weights(corpora, "flat")
    assert flat == {"big": 1.0, "small": 1.0}
    balance = compute_weights(corpora, "balance")
    assert balance["small"] > balance["big"] == 1.0


@pytest.mark.integration
def test_mini_sweep_runs_and_ranks():
    corpora = {
        "en": "india is a country in south asia " * 40,
        "hi": "भारत एक देश है " * 40,
    }
    words = {c: len(t.split()) for c, t in corpora.items()}
    specs = [
        Spec("bpe", "byte", None, 400, "flat", "byte"),
        Spec("bpe", "char", "NFC", 400, "flat", "char"),
    ]
    results = sweep(specs, corpora, words)
    assert len(results) == 2
    assert all(r.error is None for r in results)
    assert all(r.vocab_actual <= 400 for r in results)
    # sorted best-first
    assert results[0].score >= results[1].score


@pytest.mark.integration
def test_bad_spec_is_recorded_not_raised():
    corpora = {"en": "hello world " * 20}
    words = {"en": len(corpora["en"].split())}
    bad = run(Spec(algo="nope", label="bad"), corpora, words)
    assert bad.error is not None
