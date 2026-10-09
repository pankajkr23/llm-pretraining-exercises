"""The update-to-weight ratio, per layer, and the step at which warmup stops moving it.

For a weight matrix `W` and the change one optimiser step makes to it, `ΔW`:

    ratio = ‖ΔW‖ / ‖W‖           (Frobenius norms, `W` taken before the step)

It is the fraction by which one step changes the layer. Adam moves every weight by roughly η
regardless of its gradient's size, so at initialisation — weights of size about `1/√fan_in` — a
full-sized step changes a wide layer by a large fraction. Warmup holds η down while that is true.
A healthy ratio sits near 10⁻³ through most of a run.

**"The step at which warmup stops changing it" is measured, not assumed to be the end of warmup.**
`settles_at` smooths each layer's curve and reports the first step after which it stays within a
band around its own later level. If the ratio settles before warmup ends, warmup was longer than it
needed to be for that layer; if after, the ratio was still being driven by something else.

Only matrices are tracked: biases and norm gains start at 0 and 1, where the ratio is either
undefined or dominated by the constant.
"""

import numpy as np
import torch


class RatioMeter:
    """Snapshots a model's matrices before a step and measures each one's relative change after.

    Usage, around every optimiser step::

        meter.before()
        optimizer.step()
        ratios = meter.after()  # {parameter name: ‖ΔW‖/‖W‖}

    Args:
        model: The model whose 2-D parameters are tracked.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Track every 2-D parameter of `model`."""
        self.tracked = {n: p for n, p in model.named_parameters() if p.ndim == 2}
        self._snapshot: dict[str, torch.Tensor] = {}

    def before(self) -> None:
        """Copy every tracked matrix as it is now."""
        self._snapshot = {n: p.detach().clone() for n, p in self.tracked.items()}

    @torch.no_grad()
    def after(self) -> dict[str, float]:
        """`‖W_after − W_before‖ / ‖W_before‖` for every tracked matrix."""
        if not self._snapshot:
            raise RuntimeError("after() called without a matching before()")
        ratios = {}
        for name, p in self.tracked.items():
            old = self._snapshot[name]
            ratios[name] = float(torch.linalg.vector_norm(p - old) / torch.linalg.vector_norm(old))
        self._snapshot = {}
        return ratios


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """A trailing moving average; the first `window − 1` entries average what exists so far."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    cumulative = np.cumsum(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = (cumulative[i] - (cumulative[lo - 1] if lo else 0.0)) / (i - lo + 1)
    return out


def settles_at(
    values: np.ndarray, band: float = 0.1, window: int = 10, tail: int = 50
) -> int | None:
    """The first step from which the smoothed curve stays within `band` of its late level.

    The late level is the median of the smoothed curve over its last `tail` steps. A curve that
    never settles — still rising or falling at the end — returns None rather than a guess.

    Args:
        values: One value per step.
        band: Relative half-width of the band, e.g. 0.1 for ±10%.
        window: Smoothing window in steps.
        tail: Steps at the end that define the late level.

    Returns:
        A 0-based step index, or None.
    """
    s = smooth(values, window)
    if len(s) < tail + window:
        raise ValueError(f"need at least {tail + window} steps, got {len(s)}")
    level = float(np.median(s[-tail:]))
    inside = np.abs(s - level) <= band * abs(level)
    if not inside[-tail:].all():
        return None
    outside = np.flatnonzero(~inside)
    return int(outside[-1] + 1) if outside.size else 0
