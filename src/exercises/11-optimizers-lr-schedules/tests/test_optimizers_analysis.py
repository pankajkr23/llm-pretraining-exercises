"""The sweep reader and the settling detector recover answers planted in synthetic data.

Both are measurement code, so both are tested the way `AGENTS.md` asks of a guard: on inputs where
the right answer is known, including inputs built to fool them.
"""

import math

import numpy as np
import pytest
from optimizers.ratios import settles_at, smooth
from optimizers.sweep import find_minimum, fit_power_law, predict

LRS = [1.25e-4 * 2**i for i in range(7)]


def _bowl(centre: float) -> list[float]:
    return [3.0 + 0.2 * (math.log2(lr) - math.log2(centre)) ** 2 for lr in LRS]


@pytest.mark.parametrize("centre", [3e-4, 1e-3, 2.2e-3])
def test_a_parabola_finds_a_minimum_between_grid_points(centre: float) -> None:
    found = find_minimum(LRS, _bowl(centre))
    assert found.interior
    assert found.lr == pytest.approx(centre, rel=1e-6)
    assert found.loss == pytest.approx(3.0, abs=1e-9)


def test_a_minimum_at_the_edge_of_the_grid_is_flagged_not_extrapolated() -> None:
    found = find_minimum(LRS, _bowl(1e-5))
    assert not found.interior
    assert found.lr == LRS[0]


def test_a_diverged_run_counts_as_worse_than_any_finite_one() -> None:
    losses = _bowl(1e-3)
    losses[-1] = float("nan")
    losses[-2] = float("inf")
    assert find_minimum(LRS, losses).lr == pytest.approx(1e-3, rel=1e-6)


def test_every_run_diverging_is_an_error_not_a_minimum() -> None:
    with pytest.raises(ValueError):
        find_minimum(LRS, [float("nan")] * len(LRS))


def test_unsorted_learning_rates_are_refused() -> None:
    with pytest.raises(ValueError):
        find_minimum(LRS[::-1], _bowl(1e-3))


def test_a_power_law_is_recovered_exactly() -> None:
    widths = [256, 512, 1024]
    law = fit_power_law(widths, [0.768 / w for w in widths])
    assert law.exponent == pytest.approx(-1.0)
    assert law.at(4096) == pytest.approx(0.768 / 4096)


def test_the_prediction_reports_its_own_uncertainty() -> None:
    widths = [256, 512, 1024]
    seeds = [[3e-3, 1.5e-3, 7.5e-4], [3.3e-3, 1.4e-3, 8e-4]]
    out = predict(widths, seeds, 4096)
    assert out["extrapolation"] == 4
    assert len(out["per_seed"]) == 2
    assert out["seed_spread"] >= 1
    assert min(out["per_seed"]) <= out["prediction"] <= max(out["per_seed"]) * 1.0001


def test_a_flat_minimum_across_widths_predicts_the_same_rate() -> None:
    out = predict([256, 512, 1024], [[1e-3] * 3, [1e-3] * 3], 4096)
    assert out["exponent"] == pytest.approx(0.0, abs=1e-12)
    assert out["prediction"] == pytest.approx(1e-3)


def test_smoothing_is_a_trailing_mean() -> None:
    assert list(smooth(np.array([1.0, 3.0, 5.0, 7.0]), 2)) == [1.0, 2.0, 4.0, 6.0]


def test_settling_is_found_where_a_ramp_meets_its_plateau() -> None:
    curve = np.concatenate([np.linspace(0.0, 1.0, 100), np.ones(200)])
    step = settles_at(curve, band=0.1, window=1, tail=50)
    assert 88 <= step <= 92, "a linear ramp to 1.0 enters the ±10% band at 0.9 of the ramp"


def test_a_curve_still_moving_at_the_end_has_not_settled() -> None:
    assert settles_at(np.linspace(0.0, 1.0, 300), band=0.05, window=5, tail=50) is None


def test_settling_needs_enough_steps_to_judge() -> None:
    with pytest.raises(ValueError):
        settles_at(np.ones(20), tail=50, window=10)
