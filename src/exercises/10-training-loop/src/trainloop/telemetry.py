"""Per-step traces, and the search for a step where the gradient moved before the loss did.

**The gradient norm is the earlier signal.** The loss is an average over a whole batch, so a change
in what the model is doing has to be large enough to move that average before it is visible. The
gradient norm is a direct measurement of how hard the optimiser is currently pushing, and it moves
first — which is why a run that logs only the loss finds out about its problems late.

**Item 4 asks for one step where that happened, and it is a search rather than a claim.** The
honest procedure is to log both traces over a real run and look. If no such step exists, that is the
result and it gets reported: a manufactured example would be worse than reporting nothing.

**What "moved before" has to mean, precisely.** A step where the gradient norm changed by more than
the loss did, relative to each trace's own typical step-to-step movement — because the two are in
different units and a raw comparison would only be measuring which number happens to be larger. The
threshold is a parameter with a default, not a constant hidden in a function, so it can be varied
and the finding checked against that variation.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Trace:
    """Per-step traces from one run.

    Attributes:
        steps: Step indices, from 1.
        loss: Combined loss at each step.
        grad_norm: Global gradient norm, before clipping, at each step.
        clipped: Whether the gradient was clipped at that step.
        seconds: Wall-clock seconds the step took, for MFU.
        tokens: Valid token positions the step consumed.
    """

    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    clipped: list[bool] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)

    def record(
        self, step: int, loss: float, grad_norm: float, clipped: bool, seconds: float, tokens: int
    ) -> None:
        """Append one step."""
        self.steps.append(step)
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.clipped.append(clipped)
        self.seconds.append(seconds)
        self.tokens.append(tokens)

    def as_dict(self) -> dict[str, object]:
        """Plain data — no tensors, no devices. See exercise 09 on why that matters."""
        return {
            "steps": self.steps,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "clipped": self.clipped,
            "seconds": self.seconds,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class LeadingStep:
    """A step where the gradient norm moved and the loss did not.

    Attributes:
        step: Which step.
        grad_move: How far the gradient norm moved, in units of its own typical step.
        loss_move: The same for the loss.
        grad_norm: The gradient norm at that step.
        loss: The loss at that step.
    """

    step: int
    grad_move: float
    loss_move: float
    grad_norm: float
    loss: float

    @property
    def lead(self) -> float:
        """How much further the gradient moved than the loss, in shared units."""
        return self.grad_move - self.loss_move


def _typical_move(series: list[float]) -> float:
    """Median absolute step-to-step change — the scale a move is measured against.

    Median rather than mean, because a single large jump is exactly what is being looked for and it
    should not be allowed to inflate the yardstick used to find it.
    """
    moves = [abs(b - a) for a, b in zip(series, series[1:], strict=False)]
    positive = [m for m in moves if m > 0]
    return statistics.median(positive) if positive else 1.0


def find_leading_steps(
    trace: Trace, threshold: float = 3.0, loss_ceiling: float = 1.0
) -> list[LeadingStep]:
    """Steps where the gradient norm moved sharply and the loss barely moved.

    Args:
        trace: A completed run.
        threshold: How many typical gradient-norm steps the move must exceed.
        loss_ceiling: How many typical loss steps the loss must stay *within* to count as
            "did not move".

    Returns:
        Every qualifying step, in order. **Possibly empty**, which is a real answer.
    """
    if len(trace.steps) < 3:
        return []

    grad_scale = _typical_move(trace.grad_norm)
    loss_scale = _typical_move(trace.loss)

    found: list[LeadingStep] = []
    for i in range(1, len(trace.steps)):
        grad_move = abs(trace.grad_norm[i] - trace.grad_norm[i - 1]) / grad_scale
        loss_move = abs(trace.loss[i] - trace.loss[i - 1]) / loss_scale
        if grad_move >= threshold and loss_move <= loss_ceiling:
            found.append(
                LeadingStep(
                    step=trace.steps[i],
                    grad_move=grad_move,
                    loss_move=loss_move,
                    grad_norm=trace.grad_norm[i],
                    loss=trace.loss[i],
                )
            )
    return found


def robustness(trace: Trace) -> dict[str, int]:
    """How many steps qualify under several thresholds.

    The threshold is an arbitrary choice, so the finding is quoted alongside what happens when it
    moves. A result that exists at exactly one threshold is a result about the threshold.
    """
    return {
        f"threshold={t}": len(find_leading_steps(trace, threshold=t))
        for t in (2.0, 2.5, 3.0, 4.0, 5.0)
    }


def save(trace: Trace, extra: dict[str, object], path: Path | None = None) -> Path:
    """Write the trace and whatever else the run produced.

    Separate from the run for the reason exercise 05 paid for: three experiments trained to
    completion and died in their final statement, one losing fifteen trained models.
    """
    path = path or (RESULTS / "run.json")
    path.parent.mkdir(exist_ok=True)
    payload: dict[str, object] = {"trace": trace.as_dict()}
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
