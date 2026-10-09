"""Learning-rate schedules as pure functions of the step: warmup, cosine and WSD.

Every schedule here maps a **0-based step index** to the learning rate used *at* that step, and
needs nothing but the arithmetic below — no torch, no optimizer — so each can be plotted, tested and
reasoned about on its own.

- **Warmup** raises the rate linearly over the first `warmup` steps: step 0 runs at `peak/warmup`,
  step `warmup - 1` at `peak`. Early gradients are strongly correlated, so Adam takes near-full
  steps on every weight at once; warmup keeps those first updates small relative to the weights.
- **Cosine** follows half a cosine from `peak` down to `floor`, reaching `floor` exactly at the last
  step. `total` is an input to every rate it returns, so **a cosine run commits to its budget up
  front**: halt it partway and the rate never got low enough to finish the job.
- **WSD** — warmup, stable, decay — holds `peak` flat and decays only over the final
  `decay_fraction` of the run. The stable phase has no built-in end, which is its point: a
  checkpoint taken anywhere in it can be *branched* and decayed separately (`wsd_branch`), so one
  run yields finished models at several budgets.

So the practical difference is about planning: one schedule has to know its budget on day one, the
other can defer that decision to the branch.
"""

import math


def _check(step: int, total: int, peak: float, warmup: int, floor: float) -> None:
    if not 0 <= step < total:
        raise ValueError(f"step must lie in [0, {total}), got {step}")
    if peak <= 0 or floor < 0 or floor > peak:
        raise ValueError(f"need 0 <= floor <= peak and peak > 0; got peak={peak}, floor={floor}")
    if not 0 <= warmup < total:
        raise ValueError(f"warmup must lie in [0, {total}), got {warmup}")


def warmup_factor(step: int, warmup: int) -> float:
    """The fraction of peak used at `step` during linear warmup, and 1.0 once it is over.

    Args:
        step: 0-based step.
        warmup: Warmup length in steps; 0 disables warmup.

    Returns:
        `(step + 1) / warmup` while `step < warmup`, else 1.0.
    """
    if warmup <= 0 or step >= warmup:
        return 1.0
    return (step + 1) / warmup


def constant(step: int, total: int, peak: float, warmup: int = 0, floor: float = 0.0) -> float:
    """Warmup, then `peak` for the rest of the run."""
    _check(step, total, peak, warmup, floor)
    return peak * warmup_factor(step, warmup)


def cosine(step: int, total: int, peak: float, warmup: int = 0, floor: float = 0.0) -> float:
    """Warmup, then half a cosine from `peak` at the end of warmup to `floor` at step `total - 1`.

    Args:
        step: 0-based step.
        total: The run length this schedule is shaped for. Decided before the first step.
        peak: The highest rate.
        warmup: Warmup steps.
        floor: The rate at the last step.

    Returns:
        The learning rate at `step`.
    """
    _check(step, total, peak, warmup, floor)
    if step < warmup:
        return peak * warmup_factor(step, warmup)
    span = max(1, total - 1 - warmup)
    progress = min(1.0, (step - warmup) / span)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def decay_start(total: int, decay_fraction: float) -> int:
    """The first step of WSD's decay: the last `decay_fraction` of the run, at least one step."""
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError(f"decay_fraction must lie in (0, 1], got {decay_fraction}")
    return total - max(1, round(decay_fraction * total))


DECAY_SHAPES = ("linear", "cosine")


def _check_shape(shape: str) -> None:
    """Refuse an unknown decay shape at the call, not only once a run reaches the decay."""
    if shape not in DECAY_SHAPES:
        raise ValueError(f"shape must be one of {DECAY_SHAPES}, got {shape!r}")


def _decay(progress: float, peak: float, floor: float, shape: str) -> float:
    if shape == "linear":
        return peak + (floor - peak) * progress
    if shape == "cosine":
        return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"shape must be 'linear' or 'cosine', got {shape!r}")


def wsd(
    step: int,
    total: int,
    peak: float,
    warmup: int = 0,
    floor: float = 0.0,
    decay_fraction: float = 0.2,
    shape: str = "linear",
) -> float:
    """Warmup, a flat `peak`, then a decay to `floor` over the last `decay_fraction` of the run.

    Args:
        step: 0-based step.
        total: The run length.
        peak: The stable rate.
        warmup: Warmup steps.
        floor: The rate at step `total - 1`.
        decay_fraction: Share of the run spent decaying.
        shape: `"linear"` or `"cosine"` decay.

    Returns:
        The learning rate at `step`.
    """
    _check(step, total, peak, warmup, floor)
    _check_shape(shape)
    start = decay_start(total, decay_fraction)
    if start < warmup:
        raise ValueError(f"decay would start at {start}, inside the {warmup}-step warmup")
    if step < warmup:
        return peak * warmup_factor(step, warmup)
    if step < start:
        return peak
    return _decay((step - start + 1) / (total - start), peak, floor, shape)


def wsd_branch(
    step: int,
    branch_at: int,
    decay_steps: int,
    peak: float,
    floor: float = 0.0,
    shape: str = "linear",
) -> float:
    """The rate on a decay branch taken from a WSD checkpoint at `branch_at`.

    The branch continues from the checkpoint's weights and optimizer state and decays from `peak` to
    `floor` over `decay_steps`, reaching `floor` at step `branch_at + decay_steps - 1`. This is the
    move cosine cannot make: its decay is already spent by the time it is stopped.

    Args:
        step: 0-based global step, at or after `branch_at`.
        branch_at: The checkpoint's step: the first step of the branch.
        decay_steps: How long the branch decays.
        peak: The stable rate the checkpoint was trained at.
        floor: The final rate.
        shape: `"linear"` or `"cosine"`.

    Returns:
        The learning rate at `step`.
    """
    if decay_steps < 1:
        raise ValueError(f"decay_steps must be at least 1, got {decay_steps}")
    _check_shape(shape)
    if not branch_at <= step < branch_at + decay_steps:
        raise ValueError(f"step {step} is outside the branch [{branch_at}, +{decay_steps})")
    return _decay((step - branch_at + 1) / decay_steps, peak, floor, shape)


SCHEDULES = {"constant": constant, "cosine": cosine, "wsd": wsd}
