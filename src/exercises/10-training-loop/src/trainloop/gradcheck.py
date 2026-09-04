"""Verify one gradient by hand, and find the range of nudges where hand and autograd agree.

`backward()` reports a derivative. A derivative is a claim about what happens to the loss when a
weight moves a little, and it is checkable: move the weight a little, see what the loss did, divide.
If the two disagree, something between the model and the scalar is wrong.

**The central difference is the right estimator here.** A one-sided difference
`(f(w+h) - f(w)) / h` carries an error proportional to `h`; the central form
`(f(w+h) - f(w-h)) / 2h` cancels the first-order term and leaves an error proportional to `h²`.
For the same `h` it is dramatically closer, which is what "agree to several decimals" needs.

**And `h` has a floor as well as a ceiling, so a single value is not an answer.** Too large and the
function's curvature shows up in the estimate; too small and `f(w+h)` and `f(w-h)` differ in bits
that float arithmetic does not keep, so the subtraction is dominated by rounding error. There is a
window in between, and `sweep` finds it rather than picking one value that happened to work.

**In fp32 that window is narrower than people expect**, which is itself the finding: this is why a
gradient check is run in float64 wherever it is affordable, and why a check that agrees at exactly
one `h` should be distrusted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch


@dataclass(frozen=True)
class GradientCheck:
    """One weight, one nudge size, and the two numbers being compared.

    Attributes:
        epsilon: How far the weight was moved in each direction.
        analytic: What `backward()` reported for this weight.
        numeric: `(loss(w + eps) - loss(w - eps)) / (2 * eps)`.
        matching_digits: How many decimal digits the two share. The requirements ask for
            "several", so this is the number that answers it.
    """

    epsilon: float
    analytic: float
    numeric: float
    matching_digits: float

    @property
    def absolute_error(self) -> float:
        """How far apart they are."""
        return abs(self.analytic - self.numeric)

    @property
    def relative_error(self) -> float:
        """The same, scaled by the gradient's own size — the number that is comparable."""
        denominator = max(abs(self.analytic), abs(self.numeric), 1e-30)
        return self.absolute_error / denominator


def _matching_digits(a: float, b: float) -> float:
    """Decimal digits two numbers agree on, as `-log10(relative error)`.

    Infinite when they are bit-identical, which is why the caller caps it rather than printing it.
    """
    import math

    denominator = max(abs(a), abs(b), 1e-30)
    error = abs(a - b) / denominator
    return float("inf") if error == 0 else -math.log10(error)


def check_one_weight(
    loss_at: Callable[[], torch.Tensor],
    weight: torch.Tensor,
    index: tuple[int, ...],
    epsilon: float = 1e-3,
) -> GradientCheck:
    """Compare autograd's gradient for one element against a central difference.

    Args:
        loss_at: Recomputes the loss with the model's current weights. Called three times.
        weight: The parameter tensor holding the element under test.
        index: Which element.
        epsilon: How far to move it in each direction.

    Returns:
        A `GradientCheck`.

    Raises:
        ValueError: When `weight.grad` is absent — the caller must run `backward()` first, and a
            silent zero here would look like perfect disagreement rather than a missing step.
    """
    import torch

    if weight.grad is None:
        raise ValueError(
            "weight.grad is None: run loss.backward() before checking, or the analytic side of "
            "this comparison does not exist yet"
        )
    analytic = float(weight.grad[index])

    with torch.no_grad():
        original = weight[index].clone()

        weight[index] = original + epsilon
        up = float(loss_at())

        weight[index] = original - epsilon
        down = float(loss_at())

        weight[index] = original

    numeric = (up - down) / (2 * epsilon)
    return GradientCheck(epsilon, analytic, numeric, _matching_digits(analytic, numeric))


def sweep(
    loss_at: Callable[[], torch.Tensor],
    weight: torch.Tensor,
    index: tuple[int, ...],
    epsilons: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
) -> list[GradientCheck]:
    """Run the check across several nudge sizes, so the agreement window is visible.

    Quoting one epsilon is quoting the one that worked. The shape of this list — agreement rising
    as `epsilon` falls, then *falling again* once float error dominates — is the actual finding.
    """
    return [check_one_weight(loss_at, weight, index, eps) for eps in epsilons]


def best(checks: list[GradientCheck]) -> GradientCheck:
    """The nudge size where the two agreed most closely."""
    return min(checks, key=lambda c: c.relative_error)


def report(checks: list[GradientCheck]) -> str:
    """The sweep as a table, with the window named rather than implied."""
    lines = [
        f"    {'epsilon':>10}  {'autograd':>14}  {'central diff':>14}  {'rel. error':>12}  "
        f"{'digits':>7}",
        f"    {'-' * 10}  {'-' * 14}  {'-' * 14}  {'-' * 12}  {'-' * 7}",
    ]
    for check in checks:
        digits = (
            "exact" if check.matching_digits == float("inf") else f"{check.matching_digits:.1f}"
        )
        lines.append(
            f"    {check.epsilon:>10.0e}  {check.analytic:>14.9f}  {check.numeric:>14.9f}  "
            f"{check.relative_error:>12.2e}  {digits:>7}"
        )
    winner = best(checks)
    lines += [
        "",
        f"    Closest agreement at epsilon = {winner.epsilon:.0e}: "
        f"{winner.matching_digits:.1f} matching decimal digits.",
        "",
        "    Read the column, not the row. Agreement improves as epsilon falls — the central",
        "    difference's error goes as epsilon squared — and then gets WORSE again, because",
        "    loss(w+h) and loss(w-h) stop differing in bits the float type keeps, so the",
        "    subtraction becomes rounding noise. There is a window, not a best value, and a",
        "    check that agrees at exactly one epsilon has not been verified; it has been fitted.",
    ]
    return "\n".join(lines)
