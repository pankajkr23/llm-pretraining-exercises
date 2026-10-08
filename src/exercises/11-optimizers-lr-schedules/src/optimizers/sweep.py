"""Reading a learning-rate sweep: each width's minimum, and what they predict at a new width.

A sweep tries learning rates that double (a straight line on a log axis), so the minimum is found
on `log₂ η`. **The best grid point is not the minimum.** The minimum is placed by a parabola through
the best point and its two neighbours, which locates it between grid points; at the edge of the grid
there is no neighbour on one side, and the result is flagged rather than extrapolated, because a
minimum at the edge of the range says only that the range was too narrow.

The prediction at a wider model fits a power law, `η*(width) = a · width^b`, as a straight line in
log–log, and evaluates it at the target width. In the standard parametrization the exponent is
expected near −1 (the best rate roughly halves when width doubles); in muP near 0 (the minimum stays
put — the reason muP exists). How far to trust the prediction is reported as the spread of the
predictions made from each seed separately, and as how far beyond the widest measured width it
reaches.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Minimum:
    """Where one sweep's loss is lowest.

    Attributes:
        lr: The estimated best learning rate.
        loss: The parabola's value there (or the best point's loss at an edge).
        interior: False when the best point is at either end of the grid.
    """

    lr: float
    loss: float
    interior: bool


def find_minimum(lrs: list[float], losses: list[float]) -> Minimum:
    """The minimum of one sweep, between grid points where a parabola can place it.

    Non-finite losses (a diverged run) are treated as worse than any finite one.

    Args:
        lrs: Learning rates, increasing.
        losses: Final loss at each.

    Returns:
        The minimum.
    """
    x = np.log2(np.asarray(lrs, dtype=float))
    y = np.asarray(losses, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("learning rates must be strictly increasing")
    safe = np.where(np.isfinite(y), y, np.inf)
    i = int(np.argmin(safe))
    if not np.isfinite(safe[i]):
        raise ValueError("every run diverged")
    if i == 0 or i == len(x) - 1 or not np.all(np.isfinite(safe[i - 1 : i + 2])):
        return Minimum(lr=float(2 ** x[i]), loss=float(y[i]), interior=False)
    a, b, c = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
    if a <= 0:  # three points not convex: the best grid point is the honest answer
        return Minimum(lr=float(2 ** x[i]), loss=float(y[i]), interior=True)
    vertex = -b / (2 * a)
    return Minimum(lr=float(2**vertex), loss=float(np.polyval([a, b, c], vertex)), interior=True)


@dataclass(frozen=True)
class PowerLaw:
    """`η* = coefficient · width ** exponent`, fitted in log–log.

    Attributes:
        coefficient: a.
        exponent: b.
    """

    coefficient: float
    exponent: float

    def at(self, width: float) -> float:
        """The predicted best learning rate at `width`."""
        return self.coefficient * width**self.exponent


def fit_power_law(widths: list[int], lrs: list[float]) -> PowerLaw:
    """Least squares through `(log width, log η*)`. Two points are an exact line; more are a fit."""
    if len(widths) < 2:
        raise ValueError("a power law needs at least two widths")
    slope, intercept = np.polyfit(np.log(widths), np.log(lrs), 1)
    return PowerLaw(coefficient=float(math.exp(intercept)), exponent=float(slope))


def predict(
    widths: list[int], minima_by_seed: list[list[float]], target: int
) -> dict[str, float | list[float]]:
    """Predict the best learning rate at `target` from each seed's minima, and from their mean.

    Args:
        widths: The swept widths.
        minima_by_seed: For each seed, the best η at each width (same order as `widths`).
        target: The width to predict for.

    Returns:
        The mean-curve prediction and exponent, each seed's prediction, the spread between them as
        a ratio (max/min), and how many times wider the target is than the widest width measured.
    """
    per_seed = [fit_power_law(widths, m).at(target) for m in minima_by_seed]
    mean_minima = np.exp(np.mean(np.log(np.asarray(minima_by_seed)), axis=0))
    law = fit_power_law(widths, list(mean_minima))
    return {
        "prediction": law.at(target),
        "exponent": law.exponent,
        "per_seed": per_seed,
        "seed_spread": max(per_seed) / min(per_seed),
        "extrapolation": target / max(widths),
    }
