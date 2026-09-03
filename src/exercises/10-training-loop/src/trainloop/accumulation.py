"""Two ways of combining micro-batch losses, one of which is wrong — and hid for years.

Gradient accumulation exists because the batch you want will not fit in the memory you have. Split
it into micro-batches, compute each one's loss, combine them, and take a single optimiser step. The
whole technique rests on the combination being equivalent to having run the big batch at once.

**The wrong combination averages the averages.** It gives a micro-batch holding two real tokens
exactly the same vote as one holding four, so short micro-batches are over-weighted in proportion to
how short they are. The right combination sums the losses and divides by the total token count, so
every token weighs the same.

**What makes this worth a module rather than a footnote is how it hid.** A bug of this shape lived
inside every major training framework until 2024. The error vanishes entirely when the micro-batches
carry the same number of real tokens, and a hand-built test case almost always does — so the fault
was invisible to precisely the checks that would have caught it. Curves looked ordinary. Nothing
raised.

So `Config.micro_batch_tokens` is deliberately uneven, `micro_batches_are_uneven` exists to assert
that it is, and `compare` refuses to report a gap from an even configuration rather than reporting
a reassuring zero.

The worked case: micro-batches of **4, 4 and 2** valid tokens with average losses **2.0, 2.0 and
5.0** combine correctly to **2.6000** and wrongly to **3.0000** — **15.4%** out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

from .config import Config


class EvenMicroBatchesError(ValueError):
    """Raised when a comparison is asked for on micro-batches that cannot expose the bug.

    A gap of zero from equal token counts is not evidence that the reduction is correct — it is
    evidence that the experiment could not tell. Reporting it as the former is the failure this
    whole module is about, so it is refused rather than returned.
    """


@dataclass(frozen=True)
class Combination:
    """Both reductions of the same micro-batch losses, and the gap between them.

    Attributes:
        token_counts: Valid tokens per micro-batch.
        losses: Average loss per micro-batch.
        correct: Total loss over total tokens — every token weighted equally.
        wrong: The mean of the per-micro-batch means.
        uneven: Whether the token counts differ at all.
    """

    token_counts: tuple[int, ...]
    losses: tuple[float, ...]
    correct: float
    wrong: float
    uneven: bool

    @property
    def absolute_gap(self) -> float:
        """How far the wrong reduction is from the right one."""
        return self.wrong - self.correct

    @property
    def relative_gap(self) -> float:
        """The same, as a fraction of the correct value — the number worth quoting."""
        return self.absolute_gap / self.correct if self.correct else 0.0

    def __str__(self) -> str:
        """The comparison, in the form the requirements ask for it."""
        rows = "\n".join(
            f"    micro-batch {i + 1}: {tokens:>3} valid tokens, average loss {loss:.4f}"
            for i, (tokens, loss) in enumerate(zip(self.token_counts, self.losses, strict=True))
        )
        weighted = " + ".join(
            f"{t}({loss:.1f})" for t, loss in zip(self.token_counts, self.losses, strict=True)
        )
        plain = " + ".join(f"{loss:.1f}" for loss in self.losses)
        return (
            f"{rows}\n"
            f"\n    correct  ({weighted}) / {sum(self.token_counts)} = {self.correct:.4f}"
            f"\n    wrong    ({plain}) / {len(self.losses)} = {self.wrong:.4f}"
            f"\n    gap      {self.absolute_gap:+.4f}  ({self.relative_gap:+.1%})"
        )


def combine_correctly(losses: tuple[float, ...], token_counts: tuple[int, ...]) -> float:
    """Total loss over total tokens. Every token carries the same weight.

    Args:
        losses: Average loss per micro-batch.
        token_counts: Valid tokens per micro-batch.

    Returns:
        The batch's mean loss.

    Raises:
        ValueError: When the two sequences differ in length, or no tokens are present.
    """
    if len(losses) != len(token_counts):
        raise ValueError(f"{len(losses)} losses against {len(token_counts)} token counts")
    total_tokens = sum(token_counts)
    if total_tokens == 0:
        raise ValueError("no valid tokens across any micro-batch")
    return sum(loss * n for loss, n in zip(losses, token_counts, strict=True)) / total_tokens


def combine_wrongly(losses: tuple[float, ...]) -> float:
    """The mean of the means. **This is the bug**, kept as a named function so it can be measured.

    It ignores the token counts entirely, which is why it is right when they happen to be equal and
    wrong in proportion to how much they differ.
    """
    if not losses:
        raise ValueError("no micro-batches")
    return sum(losses) / len(losses)


def compare(config: Config | None = None) -> Combination:
    """Both reductions on the configured micro-batches.

    Raises:
        EvenMicroBatchesError: When the token counts are equal, so the comparison could not detect
            the bug even if it were present.
    """
    config = config or Config()
    if not config.micro_batches_are_uneven:
        raise EvenMicroBatchesError(
            f"micro_batch_tokens is {config.micro_batch_tokens}, which is even. Averaging the "
            "averages is exactly correct when every micro-batch holds the same number of valid "
            "tokens, so this comparison would report a gap of zero — which says the experiment "
            "was blind, not that the reduction is right."
        )
    return Combination(
        token_counts=config.micro_batch_tokens,
        losses=config.micro_batch_losses,
        correct=combine_correctly(config.micro_batch_losses, config.micro_batch_tokens),
        wrong=combine_wrongly(config.micro_batch_losses),
        uneven=True,
    )


def accumulate(
    micro_batches: list[tuple[torch.Tensor, torch.Tensor]],
    forward: object,
    correct: bool = True,
    config: Config | None = None,
) -> tuple[torch.Tensor, int]:
    """Run a real accumulation over real micro-batches, either reduction.

    This is the version that produces the two *curves* the requirements ask to be plotted together.
    The arithmetic above shows the bug in four lines; this shows what it does to a training run.

    Args:
        micro_batches: `(logits, targets)` pairs, targets already masked.
        forward: Callable taking `(logits, targets)` and returning `(summed loss, token count)`.
        correct: `True` sums and divides once; `False` averages the per-micro-batch averages.
        config: Unused today, taken so a caller does not have to change when it is.

    Returns:
        `(combined loss, total valid tokens)`.
    """
    import torch

    del config  # accepted for signature stability; nothing here reads it yet

    total_loss = torch.zeros(())
    total_tokens = 0
    means: list[torch.Tensor] = []

    for logits, targets in micro_batches:
        summed, count = forward(logits, targets)
        total_tokens += count
        if correct:
            total_loss = total_loss + summed
        else:
            means.append(summed / count if count else summed * 0.0)

    if correct:
        return (total_loss / total_tokens if total_tokens else total_loss), total_tokens
    return (torch.stack(means).mean() if means else total_loss), total_tokens
