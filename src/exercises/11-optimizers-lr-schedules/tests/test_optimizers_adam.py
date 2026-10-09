"""Adam written out by hand: the arithmetic, the bias-correction gap, and when it closes.

Torch-free. The comparison with `torch.optim.Adam` is in `test_optimizers_torch.py`.
"""

import math

import pytest
from optimizers.adam import (
    adam_by_hand,
    largest_difference,
    steps_until_within,
    uncorrected_over_corrected,
)


def test_the_first_step_moves_the_weight_by_exactly_eta_whatever_the_gradient() -> None:
    """At t=1 bias correction makes m̂ = g and v̂ = g², so the step is η·sign(g) (ε aside)."""
    for g in (1e-6, 0.3, -42.0):
        step = adam_by_hand(1.0, [g], lr=1e-3, eps=0.0)[0]
        assert step.update == pytest.approx(-1e-3 * math.copysign(1.0, g))
        assert step.m_hat == pytest.approx(g)
        assert step.v_hat == pytest.approx(g * g)


def test_each_quantity_follows_its_recurrence() -> None:
    grads = [0.5, -0.2, 0.8, 0.0, -0.6]
    b1, b2 = 0.9, 0.999
    steps = adam_by_hand(0.7, grads, beta1=b1, beta2=b2)
    m = v = 0.0
    for t, (s, g) in enumerate(zip(steps, grads, strict=True), start=1):
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        assert (s.m, s.v) == pytest.approx((m, v))
        assert s.m_hat == pytest.approx(m / (1 - b1**t))
        assert s.v_hat == pytest.approx(v / (1 - b2**t))
    assert steps[-1].w == pytest.approx(0.7 + sum(s.update for s in steps))


def test_a_constant_gradient_is_recovered_exactly_by_bias_correction() -> None:
    steps = adam_by_hand(0.0, [0.25] * 50)
    assert all(s.m_hat == pytest.approx(0.25) for s in steps)
    assert all(s.v_hat == pytest.approx(0.0625) for s in steps)


def test_switching_correction_off_multiplies_the_step_by_the_closed_form_ratio() -> None:
    grads = [0.25] * 30
    on = adam_by_hand(0.0, grads, eps=0.0)
    off = adam_by_hand(0.0, grads, eps=0.0, bias_correction=False)
    for t, (a, b) in enumerate(zip(on, off, strict=True), start=1):
        assert b.update / a.update == pytest.approx(uncorrected_over_corrected(t))


def test_the_ratio_is_3_16_at_step_one_and_rises_before_it_falls() -> None:
    assert uncorrected_over_corrected(1) == pytest.approx(math.sqrt(1000) / 10)
    first_20 = [uncorrected_over_corrected(t) for t in range(1, 21)]
    peak = max(range(20), key=first_20.__getitem__)
    assert 0 < peak < 19, "the peak is inside the first twenty steps, not at either end"
    assert min(first_20) > 3, (
        "within twenty steps the uncorrected step is never close to the corrected one"
    )


def test_the_gap_closes_only_after_thousands_of_steps() -> None:
    within = {tol: steps_until_within(tol) for tol in (0.10, 0.05, 0.01)}
    assert within[0.10] < within[0.05] < within[0.01]
    assert within[0.01] > 1000
    t = within[0.01]
    assert uncorrected_over_corrected(t) - 1 <= 0.01 < uncorrected_over_corrected(t - 1) - 1
    assert all(uncorrected_over_corrected(s) - 1 <= 0.01 for s in range(t, t + 2000, 97))


def test_the_closing_step_matches_the_closed_form_for_the_second_moment() -> None:
    """Long after m has recovered, the ratio is 1/√(1 − β₂ᵗ); solving it for 1% gives the step."""
    target = math.log(1 - 1 / 1.01**2) / math.log(0.999)
    assert abs(steps_until_within(0.01) - target) <= 1


def test_comparing_runs_of_different_length_is_refused() -> None:
    with pytest.raises(ValueError):
        largest_difference(adam_by_hand(0, [1.0]), adam_by_hand(0, [1.0, 2.0]))


@pytest.mark.parametrize(
    "bad", [lambda: uncorrected_over_corrected(0), lambda: steps_until_within(0.0)]
)
def test_invalid_inputs_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        bad()
