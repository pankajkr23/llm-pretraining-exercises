"""Every dimension this exercise measures against, in one place.

Exercise 09 built the target path — the shift, the masks, the loss — and this one builds the step
around it. So the model shape here **is** exercise 09's, reused rather than restated, and what is
added is everything a step needs that scoring alone does not: an optimiser, gradient accumulation,
and the arithmetic behind a utilisation figure.

**The micro-batch lengths are the most important field in this file.** Gradient accumulation is
only wrong when the micro-batches hold *different* numbers of real tokens; make them equal and the
bug's error is exactly zero, so a demonstration built on equal lengths silently proves nothing.
`micro_batch_tokens` is therefore deliberately uneven, and recorded here rather than chosen inside
a function, so a reader can see that the unevenness was a decision.
"""

from dataclasses import dataclass, field

from lossheads.config import Config as LossConfig


@dataclass(frozen=True)
class Config:
    """The shapes, hyper-parameters and hardware facts every measurement here is taken at.

    Attributes:
        model: Exercise 09's configuration — width, depth, vocabulary, sequence length. Reused
            rather than restated, so the two exercises cannot disagree about the model.
        steps: Optimiser steps in a logged run. Every claim about "over training" is bounded by it.
        learning_rate: Adam's.
        seed: Fixed, so every number here is reproducible.
        accumulation: Micro-batches combined into one optimiser step.
        micro_batch_tokens: Valid tokens in each micro-batch. **Deliberately uneven** — see the
            module docstring. The default reproduces the worked example the requirements give.
        micro_batch_losses: The per-micro-batch average losses that go with those token counts,
            for the arithmetic demonstration that needs no model at all.
        grad_clip: Norm the gradient is clipped to, or `None` for no clipping.
        device_peak_flops: A **fallback** peak, used only when no measured one is supplied.
            `mfu.measured_peak_flops` is what the harness actually uses, because a configured peak
            can describe a processor the run never touched — which is exactly what happened here,
            and reported 39.13%.
        device_name: What that fallback peak belongs to. A peak with no device attached is not a
            source, and a peak attached to the wrong device is worse than none.
    """

    model: LossConfig = field(default_factory=LossConfig)
    steps: int = 200
    learning_rate: float = 3e-4
    seed: int = 10
    accumulation: int = 3
    micro_batch_tokens: tuple[int, ...] = (4, 4, 2)
    micro_batch_losses: tuple[float, ...] = (2.0, 2.0, 5.0)
    grad_clip: float | None = 1.0
    device_peak_flops: float = 3.5e12
    device_name: str = (
        "Apple M-series GPU via MPS, ~3.5 TFLOP/s fp32 — an order-of-magnitude figure for this "
        "class of chip, not a vendor-published dense peak, and the MFU below is only as good as it"
    )

    @property
    def tokens_per_step(self) -> int:
        """Token positions consumed by one optimiser step, across all micro-batches."""
        return self.accumulation * self.model.batch_size * self.model.seq_len

    @property
    def micro_batches_are_uneven(self) -> bool:
        """Whether the configured micro-batches can expose the accumulation bug at all.

        Equal token counts make the wrong reduction exactly right, so a run configured that way
        would report a gap of zero and look like a clean bill of health.
        """
        return len(set(self.micro_batch_tokens)) > 1
