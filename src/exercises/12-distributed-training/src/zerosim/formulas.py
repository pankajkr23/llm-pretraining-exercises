"""What each stage *should* cost, derived by hand — the predictions the measurements are held to.

Nothing here reads a measurement. These are the hand calculations, written as code so that a test
can put each one beside the number the simulator actually measured and fail if they differ.

**Memory: bytes per weight on one device, at N devices** (`bf16-mixed`; activations excluded):

| stage | weights | gradients | master + m + v | total | N = 8 | N → ∞ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (DP) | 2 | 2 | 12 | 16 | 16 | 16 |
| 1 | 2 | 2 | 12 / N | 4 + 12/N | 5.5 | 4 |
| 2 | 2 | 2 / N | 12 / N | 2 + 14/N | 3.75 | 2 |
| 3 | 2 / N | 2 / N | 12 / N | 16/N | 2 | 0 |

The last column is the **floor**: what stays replicated however many devices are added. It is why
data parallelism and ZeRO-1 cannot fit a large enough model at any N — their floor is 16 and 4 bytes
per weight, and adding devices never removes it.

**Communication: bytes each device sends per step** (ring collectives; `P` = one copy of the
weights in the gradient dtype, i.e. `numel × 2` in bf16):

| stage | collectives per step | bytes sent per device |
| --- | --- | --- |
| 0 | all-reduce(grads) = reduce-scatter + all-gather | 2 · P · (N−1)/N |
| 1 | reduce-scatter(grads) + all-gather(updated weights) | 2 · P · (N−1)/N |
| 2 | the same, the reduce-scatter now unit by unit during backward | 2 · P · (N−1)/N |
| 3 | all-gather(weights) in forward + again in backward + reduce-scatter(grads) | 3 · P · (N−1)/N |

**ZeRO-1 and -2 are free in communication; ZeRO-3 costs 50% more.** Stage 0's all-reduce already is
a reduce-scatter followed by an all-gather. Stages 1 and 2 run the same two phases and keep the
intermediate shard instead of throwing it away. Stage 3 must gather the weights a second time,
because it freed them after the forward pass.

Every byte count is computed from **padded** element counts, because padding is real memory and
real traffic (`flat.py`).
"""

from fractions import Fraction

from .precision import recipe

CATEGORY_OF = {
    "params": "param_bytes",
    "grads": "grad_bytes",
    "master": "master_bytes",
    "adam_m": "moment_bytes",
    "adam_v": "moment_bytes",
}
"""Which `Recipe` field prices each persistent ledger category."""

SHARDED_FROM = {"params": 3, "grads": 2, "master": 1, "adam_m": 1, "adam_v": 1}
"""The first stage at which each category is split 1/N across devices instead of replicated."""


def bytes_per_weight(stage: int, world_size: int, mode: str) -> dict[str, Fraction]:
    """Bytes per weight on one device, per persistent category, plus `"total"`. Exact fractions."""
    prices = recipe(mode)
    out: dict[str, Fraction] = {}
    for category, field in CATEGORY_OF.items():
        per_weight = Fraction(getattr(prices, field))
        out[category] = per_weight / world_size if stage >= SHARDED_FROM[category] else per_weight
    out["total"] = sum(out.values(), Fraction(0))
    return out


def floor_bytes_per_weight(stage: int, mode: str) -> Fraction:
    """The replicated part: bytes per weight that no number of devices removes."""
    prices = recipe(mode)
    return sum(
        (
            Fraction(getattr(prices, field))
            for category, field in CATEGORY_OF.items()
            if stage < SHARDED_FROM[category]
        ),
        Fraction(0),
    )


def bytes_per_device(stage: int, world_size: int, mode: str, padded_numel: int) -> dict[str, int]:
    """`bytes_per_weight × padded_numel`, per category — the exact number the ledger must show.

    Raises:
        ValueError: If `padded_numel` does not divide by `world_size`; such a buffer cannot be
            sharded evenly and the measurement would not be comparable.
    """
    if padded_numel % world_size:
        raise ValueError(f"{padded_numel} elements do not shard evenly across {world_size}")
    out = {}
    for category, value in bytes_per_weight(stage, world_size, mode).items():
        exact = value * padded_numel
        if exact.denominator != 1:
            raise AssertionError(f"{category} came to a fractional byte count: {exact}")
        out[category] = int(exact)
    return out


def comm_bytes_per_step(stage: int, world_size: int, payload_bytes: int) -> dict[str, Fraction]:
    """Bytes ONE device sends per step, split by collective, plus `"total"`.

    Args:
        stage: 0–3.
        world_size: `N`.
        payload_bytes: `P`, one full copy of the (padded) weights in the gradient dtype.
    """
    one_pass = Fraction(payload_bytes * (world_size - 1), world_size)
    gathers = {0: 1, 1: 1, 2: 1, 3: 2}[stage]
    out = {"reduce_scatter": one_pass, "all_gather": gathers * one_pass}
    out["total"] = out["reduce_scatter"] + out["all_gather"]
    return out


def comm_multiple(stage: int) -> int:
    """How many `P · (N−1)/N` a device sends per step: 2 for stages 0–2, 3 for stage 3."""
    return 3 if stage == 3 else 2


def ladder(
    params: int,
    world_sizes: tuple[int, ...],
    mode: str,
    card_bytes: int,
) -> list[dict[str, object]]:
    """GiB per device for a hypothetical model of `params` weights, every stage, every N.

    Returns:
        One row per `(stage, N)`: exact bytes, GiB, whether it fits `card_bytes`, and the floor.
        Activations are excluded, so "fits" here is necessary, not sufficient.
    """
    rows = []
    for stage in (0, 1, 2, 3):
        floor = floor_bytes_per_weight(stage, mode) * params
        for n in world_sizes:
            per_weight = bytes_per_weight(stage, n, mode)["total"]
            total = per_weight * params
            rows.append(
                {
                    "stage": stage,
                    "world_size": n,
                    "bytes_per_weight": float(per_weight),
                    "bytes": float(total),
                    "gib": float(total) / 2**30,
                    "fits": total <= card_bytes,
                    "floor_bytes": float(floor),
                    "never_fits": floor > card_bytes,
                }
            )
    return rows
