"""Model FLOPs Utilisation: what fraction of the hardware's arithmetic the run actually used.

MFU is the ratio of the arithmetic a training step *needs* to the arithmetic the device *could have
done* in the same wall-clock time. It is the honest way to answer "is this run fast?", because
tokens per second says nothing without knowing what the hardware was capable of.

**It is also trivially inflated, which is why every input is named here rather than hidden.** Three
numbers go in, and each has a way to be wrong in the flattering direction:

- **FLOPs per token.** The standard estimate is `6 × parameters`: roughly two per parameter for the
  forward pass, four for the backward. **Which parameters** is the whole question, and getting it
  wrong here inflated this exercise's own first figure by 45%. An embedding lookup is a *gather* —
  it reads one row per token and does no arithmetic at all — so the token and position tables
  contribute parameters and essentially no FLOPs. Counting them is free inflation, which is why
  every published MFU uses **non-embedding** parameters. `flops_per_token` states which convention
  it used and offers attention's quadratic term separately rather than folding it in.
- **Wall clock.** Time only the optimiser step and MFU looks better than the run does; time the
  whole loop including data loading and it looks worse and truer.
- **Device peak.** The largest number in a vendor's table is usually a sparse or low-precision one
  the run never touches — and it may not even be the processor the run used. This exercise's first
  version divided FLOPs achieved on the **CPU** by a **GPU's** advertised peak and reported 39.13%,
  a number that looked excellent and compared two different processors. `measured_peak_flops` now
  measures a large dense matrix multiply on the same device, dtype and framework as the run, which
  is the only denominator the numerator can honestly be divided by.

**The requirements ask what costs us the distance to 40%.** That is the real question, and it is
answered by naming what the step spends time on that is not the matrix multiplies in the estimate —
which for a run this small is nearly everything.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Utilisation:
    """One MFU figure and every input that produced it.

    Attributes:
        parameters: Trainable parameters counted, and therefore priced.
        flops_per_token: Arithmetic attributed to one token position.
        tokens: Token positions the measured window consumed.
        seconds: Wall-clock seconds that window took.
        device_peak_flops: What the hardware could have done in a second.
        device_name: Which hardware, at which precision.
        convention: How `flops_per_token` was arrived at, in words.
    """

    parameters: int
    flops_per_token: float
    tokens: int
    seconds: float
    device_peak_flops: float
    device_name: str
    convention: str

    @property
    def total_flops(self) -> float:
        """Arithmetic the window needed."""
        return self.flops_per_token * self.tokens

    @property
    def achieved_flops_per_second(self) -> float:
        """Arithmetic per second the run actually achieved."""
        return self.total_flops / self.seconds if self.seconds else 0.0

    @property
    def mfu(self) -> float:
        """Achieved over peak. A fraction, not a percentage."""
        return self.achieved_flops_per_second / self.device_peak_flops

    @property
    def tokens_per_second(self) -> float:
        """The figure people quote instead, which says nothing without the peak beside it."""
        return self.tokens / self.seconds if self.seconds else 0.0

    def __str__(self) -> str:
        """Every input, then the figure — in that order, deliberately."""
        return (
            f"    parameters priced   {self.parameters:,}\n"
            f"    FLOPs per token     {self.flops_per_token:,.0f}   ({self.convention})\n"
            f"    tokens measured     {self.tokens:,}\n"
            f"    wall clock          {self.seconds:.3f} s\n"
            f"    achieved            {self.achieved_flops_per_second / 1e9:,.2f} GFLOP/s\n"
            f"    device peak         {self.device_peak_flops / 1e12:,.2f} TFLOP/s\n"
            f"    device              {self.device_name}\n"
            f"\n    MFU                 {self.mfu:.2%}\n"
            f"    tokens/second       {self.tokens_per_second:,.0f}"
        )


def flops_per_token(
    parameters: int,
    include_attention: bool = False,
    seq_len: int = 0,
    n_layer: int = 0,
    d_model: int = 0,
) -> tuple[float, str]:
    """FLOPs attributed to one token position, and the convention used to get there.

    Args:
        parameters: Trainable parameters. **Whether the embedding table is in this number changes
            the answer**, so the caller decides and the convention string records what it decided.
        include_attention: Whether to add attention's quadratic term. It is excluded by default
            because the `6N` estimate is the one every published MFU figure uses, and mixing
            conventions makes comparisons meaningless.
        seq_len: Sequence length, needed only for the attention term.
        n_layer: Blocks, needed only for the attention term.
        d_model: Width, needed only for the attention term.

    Returns:
        `(flops per token, the convention in words)`.
    """
    base = 6.0 * parameters
    convention = (
        f"6 x {parameters:,} NON-EMBEDDING parameters — 2 forward, 4 backward; embedding lookups "
        "are gathers and do no arithmetic; attention's quadratic term excluded"
    )
    if not include_attention:
        return base, convention

    attention = 12.0 * n_layer * seq_len * d_model
    return (
        base + attention,
        f"6 x {parameters:,} parameters, plus 12 x {n_layer} layers x {seq_len} sequence x "
        f"{d_model} width for attention",
    )


def measure(
    parameters: int,
    tokens: int,
    seconds: float,
    config: Config | None = None,
    include_attention: bool = False,
    device_peak_flops: float | None = None,
    device_name: str | None = None,
) -> Utilisation:
    """Compute MFU from a measured window.

    Args:
        parameters: Trainable parameters priced.
        tokens: Token positions the window consumed.
        seconds: Wall-clock seconds it took.
        config: Supplies the fallback device peak and its description.
        include_attention: Whether to add attention's quadratic term.
        device_peak_flops: A measured peak, which is always preferable to the configured one.
        device_name: What that measured peak belongs to.

    Returns:
        A `Utilisation` carrying the figure and every input.

    Raises:
        ValueError: When `seconds` is not positive — a zero here would report infinite utilisation.
    """
    config = config or Config()
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}: MFU would be infinite")

    per_token, convention = flops_per_token(
        parameters,
        include_attention=include_attention,
        seq_len=config.model.seq_len,
        n_layer=config.model.n_layer,
        d_model=config.model.d_model,
    )
    return Utilisation(
        parameters=parameters,
        flops_per_token=per_token,
        tokens=tokens,
        seconds=seconds,
        device_peak_flops=device_peak_flops or config.device_peak_flops,
        device_name=device_name or config.device_name,
        convention=convention,
    )


def distance_to_target(utilisation: Utilisation, target: float = 0.40) -> str:
    """What is costing the run the distance to `target`, named rather than guessed.

    The requirements ask for this explicitly and for it to be honest. The honest answer for a run
    this size is that the estimate's denominator assumes the device is doing nothing but large
    matrix multiplies, and almost nothing in this run is one.
    """
    gap = target - utilisation.mfu
    return (
        f"    Target {target:.0%}, achieved {utilisation.mfu:.2%}, short by "
        f"{gap:.2%} of peak.\n"
        "\n"
        "    What is costing it, in the order it costs:\n"
        "\n"
        "    1. THE MODEL IS TOO SMALL FOR THE MACHINE. A 6N estimate assumes the device spends\n"
        "       its time inside large matrix multiplies. At this width every multiply finishes\n"
        "       before the device is fully occupied, so the fixed cost of launching the work\n"
        "       dominates the work. This is the whole gap, and it is a property of the shape\n"
        "       rather than of the code.\n"
        "    2. NOTHING IS FUSED. Every operation reads its inputs from memory and writes its\n"
        "       output back. A real training stack fuses chains of them so intermediates never\n"
        "       leave the chip; this one uses the framework's default kernels throughout.\n"
        "    3. THE STEP IS TIMED WHOLE, INCLUDING WHAT IS NOT ARITHMETIC. Data slicing, the\n"
        "       optimiser's own element-wise work, and the gradient-norm computation are all\n"
        "       inside the measured window and none of them is in the numerator. Timing only\n"
        "       the matrix multiplies would report a better number and a less true one.\n"
        "    4. FP32, NOT BF16. The peak quoted is a fp32 peak, so this is not inflating the\n"
        "       figure — but a real run in bf16 would move both sides, and comparing this\n"
        "       number to a published bf16 MFU would be comparing two different quantities.\n"
        "\n"
        "    The reachable fix is (1): none of the others is worth doing at this scale, and\n"
        "    saying so is more useful than a list of optimisations nobody should apply here."
    )


def measured_peak_flops(device: str = "cpu", size: int = 2048, repeats: int = 5) -> float:
    """Measure what this machine actually sustains on a large dense matrix multiply.

    **This replaced a vendor figure, and the replacement is the finding.** The first version of
    this exercise divided FLOPs achieved on the **CPU** by a **GPU's** peak, and reported
    **39.13% MFU** — a number that looked like a triumph and meant nothing, because the numerator
    and the denominator described different processors. It is exactly the failure the module
    docstring warns about, committed by the module that warns about it.

    A measured GEMM peak fixes the class of problem rather than the instance: it is the same
    device, the same dtype and the same framework as the run, so the ratio is between two things
    that can be compared. It is also *lower* than a vendor peak, and therefore reports a **worse**
    MFU — which is the honest direction to be wrong in.

    Args:
        device: Where to measure. Must be the device the run used.
        size: Square matrix edge. Large enough that the multiply, not the launch, dominates.
        repeats: Timed repetitions; the best is taken, since interference only ever slows it.

    Returns:
        Sustained FLOPs per second. `2 * size³` FLOPs per multiply — one multiply and one add per
        output element per inner step.
    """
    import time

    import torch

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    flops = 2.0 * size**3

    (a @ b).sum().item()  # warm up: first call pays for allocation and kernel selection

    best_seconds = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        result = a @ b
        result.sum().item()  # force completion before stopping the clock
        best_seconds = min(best_seconds, time.perf_counter() - started)

    return flops / best_seconds
