"""The schedules do what their docstrings say, at every boundary that a run depends on."""

import math

import pytest
from optimizers.schedules import constant, cosine, decay_start, warmup_factor, wsd, wsd_branch

TOTAL, PEAK, WARMUP = 300, 1e-3, 30


def test_warmup_rises_linearly_and_reaches_peak_on_its_last_step() -> None:
    assert warmup_factor(0, 10) == pytest.approx(0.1)
    assert warmup_factor(9, 10) == 1.0
    assert warmup_factor(10, 10) == 1.0
    assert warmup_factor(5, 0) == 1.0, "zero warmup means full rate from the first step"
    rates = [constant(s, TOTAL, PEAK, warmup=WARMUP) for s in range(WARMUP)]
    assert all(b > a for a, b in zip(rates, rates[1:], strict=False))
    assert rates[-1] == PEAK


def test_cosine_starts_at_peak_after_warmup_and_ends_exactly_at_floor() -> None:
    assert cosine(WARMUP, TOTAL, PEAK, warmup=WARMUP) == pytest.approx(PEAK)
    assert cosine(TOTAL - 1, TOTAL, PEAK, warmup=WARMUP, floor=1e-5) == pytest.approx(1e-5)
    middle = WARMUP + (TOTAL - 1 - WARMUP) / 2
    assert cosine(int(middle), TOTAL, PEAK, warmup=WARMUP) == pytest.approx(PEAK / 2, rel=0.02)


def test_cosine_never_rises_after_warmup() -> None:
    rates = [cosine(s, TOTAL, PEAK, warmup=WARMUP) for s in range(WARMUP, TOTAL)]
    assert all(b <= a for a, b in zip(rates, rates[1:], strict=False))


def test_cosine_stopped_at_200_of_300_is_still_well_above_its_floor() -> None:
    """The comparison's premise: a cosine shaped for 300 has not finished decaying at 200."""
    at_200 = cosine(199, TOTAL, PEAK, warmup=WARMUP)
    assert 0.1 * PEAK < at_200 < 0.5 * PEAK


def test_wsd_holds_peak_until_the_decay_and_reaches_floor_at_the_end() -> None:
    start = decay_start(TOTAL, 0.2)
    assert start == 240
    assert all(wsd(s, TOTAL, PEAK, warmup=WARMUP) == PEAK for s in range(WARMUP, start))
    assert wsd(start, TOTAL, PEAK, warmup=WARMUP) < PEAK
    assert wsd(TOTAL - 1, TOTAL, PEAK, warmup=WARMUP, floor=2e-5) == pytest.approx(2e-5)


def test_wsd_at_step_200_of_300_has_not_decayed_at_all() -> None:
    assert wsd(199, TOTAL, PEAK, warmup=WARMUP, decay_fraction=0.2) == PEAK


@pytest.mark.parametrize("shape", ["linear", "cosine"])
def test_both_decay_shapes_fall_monotonically(shape: str) -> None:
    rates = [
        wsd(s, TOTAL, PEAK, warmup=WARMUP, shape=shape)
        for s in range(decay_start(TOTAL, 0.2), TOTAL)
    ]
    assert all(b < a for a, b in zip(rates, rates[1:], strict=False))


def test_a_branch_decays_from_peak_to_floor_over_exactly_its_own_length() -> None:
    first = wsd_branch(200, 200, 30, PEAK)
    last = wsd_branch(229, 200, 30, PEAK, floor=1e-5)
    assert first < PEAK
    assert last == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        wsd_branch(230, 200, 30, PEAK)
    with pytest.raises(ValueError):
        wsd_branch(199, 200, 30, PEAK)


@pytest.mark.parametrize(
    "call",
    [
        lambda: cosine(300, TOTAL, PEAK),
        lambda: cosine(-1, TOTAL, PEAK),
        lambda: cosine(0, TOTAL, 0.0),
        lambda: cosine(0, TOTAL, PEAK, floor=2 * PEAK),
        lambda: wsd(0, TOTAL, PEAK, warmup=280, decay_fraction=0.2),
        lambda: wsd(0, TOTAL, PEAK, decay_fraction=0.0),
        lambda: wsd(0, TOTAL, PEAK, shape="step"),
    ],
)
def test_invalid_schedules_are_refused_not_silently_clamped(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_every_rate_is_finite_and_positive_across_the_run() -> None:
    for fn in (constant, cosine, wsd):
        for s in range(TOTAL):
            rate = fn(s, TOTAL, PEAK, warmup=WARMUP, floor=1e-6)
            assert math.isfinite(rate) and rate > 0
