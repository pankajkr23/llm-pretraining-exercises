"""Demand against supply, priced three ways, with a verdict per lane.

This is the module the assignment's warning is aimed at: quietly handing a large share to a lane
with almost no real data behind it is exactly the wishful accounting the work exists to prevent.
So no share here is allowed to stand without an answer to "out of
what?".

Every lane is priced in three currencies, because they say different things:

- **demand** — `share x run_tokens`. What the schedule asks for.
- **seen tokens** — what compute is billed on. Linear in epochs: four passes cost four times.
- **worth tokens** — what those passes are *worth* as fresh text. Sub-linear, and bounded.
  `dataframework.mix.worth_tokens` carries the fit (Muennighoff et al., JMLR v26 2025, Eq. 18) and
  the ceiling: no schedule extracts more than **16.4x** a unique pool, however many passes it runs.

The ceiling is what turns "this lane is thin" into "this lane is impossible". A lane can pass every
epoch threshold and still be asking a pool for more than repetition can ever yield from it.

Two corrections to the raw inventory are applied here, both argued in the functions that apply
them:

- **`supervised_ratio`** discounts a lane whose data is used with a loss mask. Session 5 §6 is
  explicit that in an agentic trajectory only the assistant's own tokens are supervised, so the
  agentic lane's raw token count is not the quantity a loss can see.
- **`double_counted`** removes text a lane claims that another lane already counted. The
  long-context lane is 60% re-counted code.
"""

from dataclasses import dataclass, field

from dataframework.mix import (
    EPOCHS_HALF_LIFE,
    EPOCHS_NEAR_FREE,
    EPOCHS_WORTHLESS,
    WORTH_CEILING_MULTIPLE,
    seen_tokens,
    worth_tokens,
)

from mixture import benchmarks, inventory
from mixture.config import Config

# Verdicts, worst last. The thresholds are the published points on the repetition curve, not
# preferences — see `dataframework.mix.GUARDRAIL_BASIS` for what each one is evidenced by.
VERDICTS = ("surplus", "covered", "repeat", "strained", "worthless", "impossible")


@dataclass(frozen=True)
class Correction:
    """An adjustment to a lane's raw inventory total, with the argument for it.

    Attributes:
        lane: The lane adjusted.
        kind: `double-count` or `supervision`.
        factor: Multiplier applied to raw supply.
        because: Why, in one sentence a reviewer can disagree with.
        provenance: `derived` where it follows from figures in the inventory, `estimated` where a
            session statement had to be turned into a number.
    """

    lane: str
    kind: str
    factor: float
    because: str
    provenance: str


def double_counted() -> dict[str, Correction]:
    """Lanes whose inventory rows are packings of text another lane already counted.

    The long-context slot lists two rows. *Repo-packed code (32K+)*, 60B, is described by the
    inventory itself as *"packed from code corpora"* — it is the code lane's tokens arranged into
    longer sequences, not additional text. *Book-length corpora (packed)*, 40B, is books and long
    documents, which no other lane in this inventory carries: the four web rows are all crawl
    (DCLM, FineWeb-Edu, D2 Web-Diverse, D1 Web-Foundation).

    So the honest unique contribution of the long-context slot is **40B, not 100B**, and the
    consequence is structural rather than arithmetic: a slot that is 60% re-counted code is a
    **sequence-length schedule**, not a lane with a budget. `lanes.py` acts on that.

    Returns:
        Lane key to the correction applied to it.
    """
    rows = {row.name: row for row in inventory.DATASETS if row.lane == "long_context"}
    packed_code = rows["Repo-packed code (32K+)"].tokens or 0.0
    total = sum(row.tokens or 0.0 for row in rows.values())
    unique = total - packed_code
    return {
        "long_context": Correction(
            lane="long_context",
            kind="double-count",
            factor=unique / total if total else 1.0,
            because=(
                f"{packed_code / 1e9:.0f}B of the slot's {total / 1e9:.0f}B is repo-packed code, "
                "which the inventory says is packed from the code corpora already counted in the "
                f"code lane; only the {unique / 1e9:.0f}B of packed books is text no other lane "
                "holds"
            ),
            provenance="derived",
        )
    }


# The notes' own words for how much of an agentic trajectory carries loss. Turned into a number
# below, with the arithmetic shown rather than the conclusion asserted.
_SUPERVISED_TOKENS_PER_TRAJECTORY = (200.0, 500.0)  # "a few hundred supervised tokens"


def supervised_ratio(lane: str) -> Correction | None:
    """How much of a lane's raw supply a loss can actually see.

    In **pre-training** the loss is on every token, so for web, code, STEM, Indic and reasoning
    text the ratio is 1.0 and no discount applies. The distinction matters for one lane.

    Agentic trajectories belong in the anneal and post-training stages: they are scarce, costly and
    among the most valuable Tier A data there is, so they are protected for annealing rather than
    spent early. There the masking rule of §6 applies:
    only the assistant's own tokens are supervised. And it sizes the result: *"A whole run yields
    only a few hundred supervised tokens."*

    Three inventory rows are long-trajectory data and turn that phrase into a ratio. SWE-Gym holds
    150M tokens across 2,400 samples (62,500 per trajectory), OpenHands rollouts 90M across 10,000
    (9,000), and SWE-smith 120M across 26,000 (4,615). A few hundred supervised tokens against
    those is somewhere between 0.3% and 10.8%.

    This is **estimated**, not measured — it converts a phrase into a range, and the range is wide
    because the phrase is. Two things keep it from being load-bearing. The generous end is the one
    applied, so the lane is judged at its most favourable. And the agentic verdict does not depend
    on the discount at all: 627M raw tokens against a 40B demand is already 3.9x the repetition
    ceiling, so the lane is impossible before any masking is considered. The discount changes the
    size of the hole, not whether there is one.

    Args:
        lane: Lane key.

    Returns:
        The correction for that lane, or None where the loss sees every token.
    """
    if lane != "agentic":
        return None

    trajectory_rows = [
        row
        for row in inventory.DATASETS
        if row.lane == lane
        and row.tokens
        and row.samples
        and row.name.startswith(("SWE-", "OpenHands"))
    ]
    ratios = [
        supervised / ((row.tokens or 0.0) / (row.samples or 1.0))
        for row in trajectory_rows
        for supervised in _SUPERVISED_TOKENS_PER_TRAJECTORY
    ]

    low, high = min(ratios), max(ratios)
    names = ", ".join(row.name for row in trajectory_rows)
    return Correction(
        lane=lane,
        kind="supervision",
        # The generous end, deliberately. A bound argued against yourself is the one that survives
        # a reviewer: if the lane fails even at the most favourable supervision ratio, where in the
        # range the truth sits stops mattering.
        factor=high,
        because=(
            f"only the assistant's own tokens are supervised (§6); at {low:.1%}-{high:.1%} "
            f"supervised per trajectory, derived from the token-per-sample counts of {names} "
            "against the session's 'few hundred supervised tokens'. The generous end is applied, "
            "and the lane is impossible without the discount anyway"
        ),
        provenance="estimated",
    )


@dataclass(frozen=True)
class LaneVerdict:
    """One lane, priced against what exists.

    Attributes:
        lane: Lane key.
        share: Its share of the run.
        demand: `share x run_tokens`.
        raw_supply: Itemised sum from the inventory, before corrections.
        supply: After double-count and supervision corrections.
        epochs: `demand / supply`, the passes the schedule implies.
        seen: Tokens the run processes — what compute is billed on.
        worth: What those passes are worth as fresh text.
        ceiling: The most repetition can ever extract from this pool.
        shortfall: `demand - ceiling` where demand exceeds it; 0 otherwise.
        verdict: One of `VERDICTS`.
        corrections: Adjustments applied, each with its argument.
        benchmark_keys: The benchmarks this lane funds.
    """

    lane: str
    share: float
    demand: float
    raw_supply: float
    supply: float
    epochs: float
    seen: float
    worth: float
    ceiling: float
    shortfall: float
    verdict: str
    corrections: tuple[Correction, ...] = ()
    benchmark_keys: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fundable(self) -> bool:
        """Whether the lane can be filled from data that exists, with repetition allowed.

        Returns:
            False when demand exceeds what infinite repetition of the pool could ever be worth.
        """
        return self.verdict != "impossible"


def _verdict(epochs: float, demand: float, ceiling: float) -> str:
    """Classify a lane against the published points on the repetition curve.

    Args:
        epochs: Passes the schedule implies.
        demand: Tokens the schedule asks for.
        ceiling: `supply x WORTH_CEILING_MULTIPLE`.

    Returns:
        One of `VERDICTS`.
    """
    if demand > ceiling:
        return "impossible"
    if epochs > EPOCHS_WORTHLESS:
        return "impossible"
    if epochs > EPOCHS_HALF_LIFE:
        return "worthless"
    if epochs > EPOCHS_NEAR_FREE:
        return "strained"
    if epochs > 1.0:
        return "repeat"
    if epochs > 0.5:
        return "covered"
    return "surplus"


def evaluate_lane(lane: str, share: float, config: Config) -> LaneVerdict:
    """Price one lane against the supply that exists for it.

    Args:
        lane: Lane key.
        share: Its share of the run, as a fraction.
        config: Thresholds and run size.

    Returns:
        The lane's verdict, with every correction applied recorded alongside it.
    """
    raw = inventory.lane_supply(lane).counted_tokens
    corrections: list[Correction] = []

    correction = double_counted().get(lane)
    if correction is not None:
        corrections.append(correction)
    correction = supervised_ratio(lane)
    if correction is not None:
        corrections.append(correction)

    supply = raw
    for applied in corrections:
        supply *= applied.factor

    demand = share * config.run_tokens
    epochs = demand / supply if supply else float("inf")
    ceiling = supply * WORTH_CEILING_MULTIPLE

    lane_benchmarks = benchmarks.by_lane().get(lane, ())

    notes: list[str] = []
    if supply and demand > ceiling:
        notes.append(
            f"demand is {demand / ceiling:.1f}x the {WORTH_CEILING_MULTIPLE:.1f}x ceiling on this "
            f"pool: no schedule reaches it, so the gap must be generated or the share must fall"
        )
        # Stated separately because a reviewer's first move against an impossible verdict is to
        # attack whichever correction produced it. Re-running the ceiling test on the uncorrected
        # inventory total shows whether the verdict needs the corrections at all.
        raw_ceiling = raw * WORTH_CEILING_MULTIPLE
        if raw and demand > raw_ceiling:
            notes.append(
                f"and it survives dropping every correction: {raw / 1e9:.2f}B raw would still "
                f"cap at {raw_ceiling / 1e9:.1f}B, which is {demand / raw_ceiling:.1f}x short"
            )

    return LaneVerdict(
        lane=lane,
        share=share,
        demand=demand,
        raw_supply=raw,
        supply=supply,
        epochs=epochs,
        seen=seen_tokens(supply, epochs) if supply else 0.0,
        worth=worth_tokens(supply, epochs) if supply else 0.0,
        ceiling=ceiling,
        shortfall=max(0.0, demand - ceiling),
        verdict=_verdict(epochs, demand, ceiling),
        corrections=tuple(corrections),
        benchmark_keys=tuple(b.key for b in lane_benchmarks),
        notes=tuple(notes),
    )


def evaluate(shares: dict[str, float], config: Config | None = None) -> dict[str, LaneVerdict]:
    """Price a whole mixture.

    Args:
        shares: Lane key to its share of the run. Lanes at zero are still evaluated, because a lane
            deliberately set to zero is a decision that has to survive review like any other.
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        Lane key to its verdict.
    """
    config = config or Config()
    return {lane: evaluate_lane(lane, share, config) for lane, share in shares.items()}
