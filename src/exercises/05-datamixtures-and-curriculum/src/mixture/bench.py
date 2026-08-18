"""Measure what this machine actually does, across model sizes.

`proxy.py` refuses to name a throughput for hardware nobody measured. This is how that figure stops
being unknown, and it exists as a module rather than a one-off command because the number it
produces goes into the specification: a figure quoted from a terminal scrollback nobody can rerun
is no better than the estimate it replaced.

**Sweep without gaps.** Exercise 02 learned this the expensive way -- a weight sweep that went
2 -> 5 -> 6 confidently named x6 the optimum, and filling in x3 and x4 moved it to x3. A throughput
curve has the same trap: two points cannot tell you where a crossover is, and the interesting
question here is exactly a crossover. On Apple silicon, small models run *faster on the CPU*,
because at that size the work per kernel is too small to cover the cost of dispatching it. Where
that stops being true is a measurement, not a guess.

Every figure is reported in TFLOP/s under the same `6ND` approximation `proxy.py` prices the ladder
with, so a measured rate and an estimated one can be compared without converting units.
"""

import json
import platform
from dataclasses import asdict, dataclass

import torch

from mixture.model import ModelConfig
from mixture.train import ARTIFACTS, TrainConfig, train


@dataclass(frozen=True)
class BenchPoint:
    """One (model size, device) measurement.

    Attributes:
        device: What ran, as torch reports it.
        params: Model parameters.
        layers: Blocks.
        width: Residual width.
        context: Sequence length.
        batch: Sequences per step.
        tokens_per_second: Measured rate.
        tflops: Measured rate under `6ND`.
    """

    device: str
    params: int
    layers: int
    width: int
    context: int
    batch: int
    tokens_per_second: float
    tflops: float


# Sizes to sweep. Chosen to be dense enough to locate a crossover rather than straddle it, and to
# stop where a step would take long enough that a laptop sweep becomes a laptop job.
LADDER: tuple[tuple[int, int, int], ...] = (
    (2, 128, 2),
    (4, 256, 4),
    (6, 384, 6),
    (8, 512, 8),
    (10, 640, 10),
    (12, 768, 12),
)


def available_devices() -> tuple[str, ...]:
    """Devices worth measuring on this machine.

    Returns:
        Device strings, always including CPU because it is the baseline the accelerator has to
        beat, and it is not obvious at small sizes that it does.
    """
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return tuple(devices)


def measure(
    layers: int,
    width: int,
    heads: int,
    device: str,
    steps: int = 20,
    batch: int = 16,
    context: int = 256,
) -> BenchPoint:
    """Time one configuration.

    Args:
        layers: Blocks.
        width: Residual width.
        heads: Attention heads.
        device: Device string.
        steps: Steps to time. Kept small; the first steps include one-off allocation, which is
            why `train` times only the forward/backward/step region.
        batch: Sequences per step.
        context: Sequence length.

    Returns:
        The measurement.
    """
    model_config = ModelConfig(layers=layers, width=width, heads=heads, context=context)
    config = TrainConfig(
        arm=f"bench-{device}-{layers}x{width}",
        shares={"web": 0.4, "indic": 0.3, "code": 0.3},
        steps=steps,
        batch=batch,
        log_every=steps + 1,
    )
    _, record = train(config, model_config, device=device)
    throughput = record.throughput
    return BenchPoint(
        device=throughput["device"],
        params=throughput["params"],
        layers=layers,
        width=width,
        context=context,
        batch=batch,
        tokens_per_second=throughput["tokens_per_second"],
        tflops=throughput["tflops"],
    )


def sweep(steps: int = 20, batch: int = 16) -> list[BenchPoint]:
    """Measure every ladder rung on every available device.

    Args:
        steps: Steps per point.
        batch: Sequences per step.

    Returns:
        Every measurement, in sweep order.
    """
    points: list[BenchPoint] = []
    for device in available_devices():
        for layers, width, heads in LADDER:
            points.append(measure(layers, width, heads, device, steps=steps, batch=batch))
    return points


def crossover(points: list[BenchPoint]) -> dict[str, object]:
    """Where the accelerator overtakes the CPU, if it does.

    Args:
        points: Output of `sweep`.

    Returns:
        The smallest parameter count at which a non-CPU device is faster, the best rate seen on
        each device, and a note when no crossover was observed inside the swept range.
    """
    by_device: dict[str, dict[int, float]] = {}
    for point in points:
        kind = point.device.split(":")[0]
        by_device.setdefault(kind, {})[point.params] = point.tflops

    cpu = by_device.get("cpu", {})
    result: dict[str, object] = {
        "best_tflops": {kind: max(rates.values()) for kind, rates in by_device.items()},
        "crossover_params": None,
        "note": "",
    }

    for kind, rates in by_device.items():
        if kind == "cpu":
            continue
        beats = sorted(params for params, rate in rates.items() if rate > cpu.get(params, 0.0))
        if beats:
            result["crossover_params"] = beats[0]
            result["note"] = f"{kind} overtakes cpu at {beats[0]:,} parameters"
        else:
            result["note"] = (
                f"{kind} did not overtake cpu anywhere in the swept range "
                f"({min(rates):,}-{max(rates):,} parameters); at these sizes the accelerator is "
                "the slower choice"
            )
    return result


def main() -> None:
    """Run the sweep and print the curve."""
    points = sweep()
    print(f"{'device':<10}{'layers':>7}{'width':>7}{'params':>12}{'tok/s':>12}{'TFLOP/s':>10}")
    for point in points:
        print(
            f"{point.device.split(':')[0]:<10}{point.layers:>7}{point.width:>7}"
            f"{point.params:>12,}{point.tokens_per_second:>12,.0f}{point.tflops:>10.3f}"
        )

    summary = crossover(points)
    print()
    for kind, best in summary["best_tflops"].items():
        print(f"  best on {kind}: {best:.3f} TFLOP/s")
    print(f"  {summary['note']}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "throughput.json"
    path.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "points": [asdict(point) for point in points],
                "summary": summary,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {path.name}")


if __name__ == "__main__":
    main()
