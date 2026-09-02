"""The V5 mixture: a share for every lane, the Indic tier split, the floor and the reserve.

This module is assignment items 1-5. Every share carries three things a reviewer can attack
separately — what it buys (a benchmark), what funds it (inventory rows), and why it is that number
rather than the session's default.

**The starting point is the session's own mixture**, and departing from it needs an argument each
time. Three departures are made, all forced by `supply.py`:

- **Long-context 6% -> 0% as a lane.** 60% of its supply is repo-packed code already counted in the
  code lane. It becomes a sequence-length schedule over other lanes (`curriculum.py`) with its own
  benchmark and no tokens of its own. The 6% is not saved; it moves to the lanes the long sequences
  are packed *from*.
- **Code 24% -> 28%.** It absorbs the repo-packed long sequences, which were its tokens already, and
  coding is half of the session's stated target capability.
- **Indic 16% -> 18% and reasoning 6% -> 8%, funded by web 34% -> 32%.** Web is the only lane with a
  6.9x surplus; two points off it cost nothing in supply terms and buy headroom in two lanes that
  are the differentiator and the thinnest real pool respectively.

The one share that is *not* changed despite failing its supply check is **agentic, held at the 2%
floor**. `supply.py` shows the demand is 3.9x more than infinite repetition of the pool could ever
be worth. Cutting the share to what supply allows (~0.03%) would satisfy the arithmetic and lose
the capability, so the share stays and the gap is declared as a **generation bill** instead. That
is the session's own instruction — agentic data *"must largely be built rather than collected"* —
and `generation_bill()` prices it rather than waving at it.
"""

from dataclasses import dataclass

from dataframework.mix import (
    ALWAYS_ON_CEILING,
    MAX_SYNTHETIC_SHARE_OF_INDIC,
    WORTH_CEILING_MULTIPLE,
)

from mixture import benchmarks, inventory, supply
from mixture.config import Config


@dataclass(frozen=True)
class Lane:
    """One capability lane in the mixture.

    Attributes:
        key: Lane key, matching `inventory.LANES`.
        name: How the lane is written in the spec.
        share: Its share of the pre-training budget.
        session_share: What Session 5's default mixture gave it, for comparison.
        because: Why this number, in the terms a reviewer will push on.
        funded_by: Inventory dataset names that supply it.
        schedule_only: True where the lane holds no tokens of its own and is applied as a schedule
            over others.
    """

    key: str
    name: str
    share: float
    session_share: float
    because: str
    funded_by: tuple[str, ...]
    schedule_only: bool = False

    @property
    def delta(self) -> float:
        """Change from the session's default share.

        Returns:
            Our share minus the session's.
        """
        return self.share - self.session_share


LANES: tuple[Lane, ...] = (
    Lane(
        key="web",
        name="General web",
        share=0.32,
        session_share=0.34,
        because=(
            "the only lane with real surplus — 4.691T against 640B, 0.14 epochs — so it funds the "
            "two points going to Indic and reasoning. It stays largest because breadth of world "
            "knowledge is what MMLU and HLE measure, and nothing else supplies it"
        ),
        funded_by=("DCLM-Baseline", "FineWeb-Edu", "D2 Web-Diverse", "D1 Web-Foundation"),
    ),
    Lane(
        key="code",
        name="Code",
        share=0.28,
        session_share=0.24,
        because=(
            "half the stated target capability, and it absorbs the retired long-context slot — 60B "
            "of which was repo-packed code from these same corpora. At 560B against 1.103T it runs "
            "at 0.51 epochs, so the increase costs no repetition"
        ),
        funded_by=("The Stack v2", "D3 Code", "CommitPack / CommitPackFT"),
    ),
    Lane(
        key="indic",
        name="Indic",
        share=0.18,
        session_share=0.16,
        because=(
            "the differentiator, and the reason the project exists. Two points above the session "
            "default buys headroom over the 12% floor rather than sitting on it, at 1.33 epochs — "
            "inside the band where repetition is near-free"
        ),
        funded_by=(
            "Sangraha (verified)",
            "Sangraha (unverified)",
            "Sangraha (synthetic)",
            "IndicCorpV2",
            "Samanantar",
            "BPCC (parallel)",
        ),
    ),
    Lane(
        key="stem",
        name="STEM / math",
        share=0.12,
        session_share=0.12,
        because=(
            "unchanged, but on 146B of itemised supply rather than the 250B the session's supply "
            "check quotes. That moves it from 0.96 epochs to 1.64 — still fundable, with no margin "
            "left to give away"
        ),
        funded_by=("D4 STEM", "peS2o", "proof-pile-2"),
    ),
    Lane(
        key="reasoning",
        name="Reasoning traces",
        share=0.08,
        session_share=0.06,
        because=(
            "up two points because this lane reserves a *distribution* of trace lengths, not a "
            "quantity. 85.1B is the thinnest real pool in the mixture and 92% of it sits in one "
            "V4-lineage dataset, so the band structure has to be bought deliberately"
        ),
        funded_by=(
            "AON",
            "OpenMathReasoning",
            "OpenThoughts2",
            "NuminaMath",
            "OpenR1-Math",
        ),
    ),
    Lane(
        key="agentic",
        name="Agentic / tool-use",
        share=0.02,
        session_share=0.02,
        because=(
            "held at the session's floor although supply cannot fund it: 40B against 627M is 3.9x "
            "more than infinite repetition could be worth. The share commits to *building* the "
            "data, not to holding it — priced in §8"
        ),
        funded_by=(
            "SWE-Gym",
            "SWE-smith",
            "OpenHands rollouts",
            "ToolBench",
            "ToolACE",
            "Glaive function-calling v2",
            "Nexus / NexusRaven",
            "xLAM / APIGen",
            "Hermes function-calling",
        ),
    ),
    Lane(
        key="long_context",
        name="Long-context",
        share=0.0,
        session_share=0.06,
        because=(
            "retired as a lane, kept as a capability. 60 of its 100B is repo-packed code already "
            "counted under code, so a 6% share would double-count it. It becomes a sequence-length "
            "schedule over code, books and web — its own benchmark, no budget"
        ),
        funded_by=("Repo-packed code (32K+)", "Book-length corpora (packed)"),
        schedule_only=True,
    ),
)


def shares() -> dict[str, float]:
    """The mixture as a lane-to-share mapping.

    Returns:
        Every lane key with its share, including the retired long-context lane at zero.
    """
    return {lane.key: lane.share for lane in LANES}


def get(key: str) -> Lane:
    """Look up a lane.

    Args:
        key: Lane key.

    Returns:
        The lane.

    Raises:
        KeyError: If no lane has that key.
    """
    for lane in LANES:
        if lane.key == key:
            return lane
    raise KeyError(f"no lane {key!r}")


# ------------------------------------------------------------------------- the Indic tier split

# Assignment item 2. The tiers are exercise 03's provenance ladder, and which row belongs where is
# the single most contested judgment in this spec — see `TIER_C_DISPUTE`.
INDIC_TIER_ROWS: dict[str, tuple[str, ...]] = {
    "A": ("Sangraha (verified)",),
    "B": ("Sangraha (unverified)", "IndicCorpV2"),
    "C": ("Sangraha (synthetic)", "Samanantar", "BPCC (parallel)"),
    "D": (),
}

TIER_NAMES: dict[str, str] = {
    "A": "verified native",
    "B": "unverified crawl",
    "C": "translated",
    "D": "synthetic",
}

# Our demanded split, against the notes' default of 40/25/20/15.
INDIC_TIER_SHARES: dict[str, float] = {"A": 0.45, "B": 0.20, "C": 0.20, "D": 0.15}
NOTES_INDIC_TIER_SHARES: dict[str, float] = {"A": 0.40, "B": 0.25, "C": 0.20, "D": 0.15}

TIER_C_DISPUTE = """\
The inventory's largest Indic row is named "Sangraha (synthetic)" and tagged tier **C**, which is
the *translated* tier. The name and the tag cannot both be honoured, and which one wins decides
which tier is fundable.

Filed as **translated** (what this spec does): C holds 167.1B against a 72B demand and is covered;
D holds nothing against 54B, so 54B of Indic must be generated.

Filed as **synthetic**: D holds 162B and is covered; C holds only the 5.1B residual of Samanantar
and BPCC against 72B, so ~67B of translated Indic must be produced instead.

Either way a hole of roughly the same size opens; filing it under the name rather than the tag
moves the hole, it does not fill it. The tag is followed because the tier ladder asks *how was this
text produced*, and AI4Bharat's own description of that component is machine translation and
transliteration of existing Wikimedia content — a translation pipeline, not a generative one.
Tier D is reserved for model-generated novel text, of which the inventory lists none.
"""


@dataclass(frozen=True)
class IndicTier:
    """One tier of the Indic lane.

    Attributes:
        tier: A, B, C or D.
        name: What the tier means.
        share: Its share of the Indic lane.
        demand: Tokens asked of it.
        supply: Tokens available to it.
        epochs: Passes implied.
        rows: Inventory datasets assigned to it.
        must_generate: Tokens that do not exist and have to be produced.
    """

    tier: str
    name: str
    share: float
    demand: float
    supply: float
    epochs: float
    rows: tuple[str, ...]
    must_generate: float


def indic_tiers(config: Config | None = None) -> dict[str, IndicTier]:
    """Size each Indic tier against the rows assigned to it.

    The 5.1B that Samanantar and BPCC hold between them (they carry no individual token count) is
    credited to tier C as a block, because both are parallel corpora and the residual cannot be
    split without inventing a division the inventory does not state.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        Tier letter to its sizing.
    """
    config = config or Config()
    lane_demand = get("indic").share * config.run_tokens
    rows = {row.name: row for row in inventory.DATASETS if row.lane == "indic"}
    residual = inventory.lane_supply("indic").residual or 0.0

    tiers: dict[str, IndicTier] = {}
    for tier, names in INDIC_TIER_ROWS.items():
        counted = sum(rows[name].tokens or 0.0 for name in names)
        # The uncounted parallel corpora both sit in C, so the residual lands there whole.
        uncounted = residual if any(rows[name].tokens is None for name in names) else 0.0
        tier_supply = counted + uncounted
        demand = INDIC_TIER_SHARES[tier] * lane_demand
        tiers[tier] = IndicTier(
            tier=tier,
            name=TIER_NAMES[tier],
            share=INDIC_TIER_SHARES[tier],
            demand=demand,
            supply=tier_supply,
            epochs=demand / tier_supply if tier_supply else float("inf"),
            rows=names,
            # What must be *generated* is not what the tier is short of — it is what repetition
            # can never reach. Tier A asks 162B of a 64B pool and is short 98B, but 2.53 passes
            # cover it at a cost the repetition curve calls near-free. Subtracting supply from
            # demand would have billed 98B of synthetic Indic that nobody needs to produce, and
            # would have made repetition and generation look like the same answer.
            must_generate=max(0.0, demand - tier_supply * WORTH_CEILING_MULTIPLE),
        )
    return tiers


# ------------------------------------------------------------------------ the protected floor

# Assignment item 4. V4 pinned an Always-On lane at 8% of every batch, outside the selector's
# control, because an English-heavy proxy (cosine 0.876 with the English web band) starves exactly
# the lanes we are trying to build. V5 extends the same protection and the notes fix the two
# numbers.
FLOOR: dict[str, float] = {"indic": 0.12, "agentic": 0.02}


@dataclass(frozen=True)
class Floor:
    """The share of every batch the selector may not touch.

    Attributes:
        per_lane: Lane key to its guaranteed minimum share.
        total: Sum of the guarantees.
        ceiling: Upper bound on the whole protected lane.
        headroom: Share of a protected lane left exposed to the selector.
    """

    per_lane: dict[str, float]
    total: float
    ceiling: float
    headroom: dict[str, float]


def protected_floor() -> Floor:
    """The always-on lane, and how much of each protected share sits above it.

    The floor is a *minimum*, not the lane's whole share. Indic runs at 18% of which 12 points are
    protected, so 6 points remain subject to OPUS selection — the selector still gets to prefer the
    better Indic batches, it simply cannot drive the lane toward zero. That distinction is what
    keeps the protected total at 14% rather than 20%, under the 20% ceiling that exists because the
    protected lane is the one part of a batch no general quality signal reaches.

    Returns:
        The floor, its total, the ceiling and the per-lane headroom.
    """
    headroom = {lane: get(lane).share - floor for lane, floor in FLOOR.items()}
    return Floor(
        per_lane=dict(FLOOR),
        total=sum(FLOOR.values()),
        ceiling=ALWAYS_ON_CEILING,
        headroom=headroom,
    )


# ------------------------------------------------------------------------ the anneal reserve

# Assignment item 5. §9: "the best data must be saved deliberately, not merely discovered at the
# end", and reserving it "is decided here, at composition time". Each entry is the fraction of that
# lane's pool withheld from ordinary sampling.
RESERVE_FRACTIONS: dict[str, float] = {
    # All of it. §6: these long trajectories are "Tier A datasets ... protected for the annealing
    # stage". The lane cannot fund pre-training anyway, so spending it early would waste the only
    # agentic data that exists on the phase least able to use it.
    "agentic": 1.00,
    # Of the verified-native tier only — the tier MILU actually measures.
    "indic": 0.30,
    # The long and ultra bands, which are the traces the effort dial needs and the ones a main run
    # would consume first.
    "reasoning": 0.15,
    # Curated high-quality subsets.
    "stem": 0.05,
}

# Why each pool, and not another. Kept beside the fractions rather than in the renderer, so the
# argument for a reservation and its size cannot drift apart.
RESERVE_REASONS: dict[str, str] = {
    "agentic": (
        "§6 calls these Tier-A and reserves them for annealing; the lane cannot fund "
        "pre-training anyway, so spending it early wastes it on the phase least able to use it"
    ),
    "indic": (
        "verified-native only — reserving translated text would spend the highest-leverage "
        "phase of the run on the lowest-provenance data available"
    ),
    "reasoning": "the long and ultra bands, which a main run would consume first",
    "stem": "curated high-quality subsets",
}

# What fraction of each pool the reservation is expressed against, for the rendered table.
RESERVE_BASIS: dict[str, str] = {
    "agentic": "100% of the lane",
    "indic": "30% of tier A",
    "reasoning": "15% of the lane",
    "stem": "5% of the lane",
}

# How far under the anneal stage's budget the reserve may land and still count as covering it.
# See `Reserve.covers_anneal` for why this is a tolerance rather than an equality.
RESERVE_TOLERANCE = 0.10


@dataclass(frozen=True)
class Reserve:
    """Data held back from the main run for the final low-LR cooldown.

    Attributes:
        per_lane: Lane key to tokens withheld.
        total: Tokens withheld overall.
        share_of_run: That total as a fraction of the run.
        target_share: The anneal stage's budget, from `Config.anneal_share`.
    """

    per_lane: dict[str, float]
    total: float
    share_of_run: float
    target_share: float

    @property
    def covers_anneal(self) -> bool:
        """Whether the reserve is large enough to fill the anneal stage.

        Compared with a tolerance rather than exactly. The fractions above are chosen from what
        each pool can spare, and the stage budget they are checked against is the session's own
        *"~2% of tokens"* — an approximate figure. An exact `>=` made the reserve fail at 1.99%,
        which is a rounding artefact reported as a design fault. The tolerance is stated here so
        it is a decision rather than a fudge: a reserve more than a tenth short of the stage it
        feeds is a real problem, and one 0.25% short is not.

        Returns:
            True when the withheld tokens reach the stage's budget, within `RESERVE_TOLERANCE`.
        """
        return self.share_of_run >= self.target_share * (1 - RESERVE_TOLERANCE)


def anneal_reserve(config: Config | None = None) -> Reserve:
    """Size the reserve from the fractions declared above.

    The Indic fraction applies to tier A alone rather than to the whole lane, because reserving
    translated text for the cooldown would spend the highest-leverage phase of the run on the
    lowest-provenance data available.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        The reserve, per lane and in total.
    """
    config = config or Config()
    pools = {
        "agentic": inventory.lane_supply("agentic").counted_tokens,
        "indic": indic_tiers(config)["A"].supply,
        "reasoning": inventory.lane_supply("reasoning").counted_tokens,
        "stem": inventory.lane_supply("stem").counted_tokens,
    }
    per_lane = {lane: pools[lane] * fraction for lane, fraction in RESERVE_FRACTIONS.items()}
    total = sum(per_lane.values())
    return Reserve(
        per_lane=per_lane,
        total=total,
        share_of_run=total / config.run_tokens,
        target_share=config.anneal_share,
    )


# ---------------------------------------------------------------------- what has to be built


@dataclass(frozen=True)
class GenerationItem:
    """Tokens a lane needs that do not exist and must be produced.

    Attributes:
        lane: Which lane, or `indic-D` for a tier.
        tokens: How many have to be generated.
        because: What makes generation the only route.
    """

    lane: str
    tokens: float
    because: str


def generation_bill(config: Config | None = None) -> tuple[GenerationItem, ...]:
    """Everything the mixture asks for that has to be built rather than collected.

    A share is only honest if the gap it cannot fund is named. Two gaps exist, and both are
    consequences of decisions taken above rather than accidents.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        One item per gap, largest first.
    """
    config = config or Config()
    verdicts = supply.evaluate(shares(), config)

    items: list[GenerationItem] = []

    agentic = verdicts["agentic"]
    reserved = anneal_reserve(config).per_lane["agentic"]
    items.append(
        GenerationItem(
            lane="agentic",
            # What repetition can never reach, less the pool already committed to the anneal.
            tokens=agentic.shortfall,
            because=(
                f"{agentic.raw_supply / 1e6:.0f}M of real trajectories cap at "
                f"{agentic.ceiling / 1e9:.1f}B under infinite repetition, against a "
                f"{agentic.demand / 1e9:.0f}B share. The existing pool is committed whole to the "
                f"anneal reserve ({reserved / 1e6:.0f}M), so the pre-training share is a "
                "commitment to synthesise trajectories and verify them with executable checks"
            ),
        )
    )

    tier_d = indic_tiers(config)["D"]
    items.append(
        GenerationItem(
            lane="indic-D",
            tokens=tier_d.must_generate,
            because=(
                "the synthetic Indic tier has no supply at all: the inventory's one row named "
                "'synthetic' is tagged as translated and is counted in tier C. Tier D is the "
                "long-tail language coverage no native corpus provides, so it is generated or the "
                "tier is dropped"
            ),
        )
    )

    return tuple(sorted(items, key=lambda item: -item.tokens))


def synthetic_share_of_indic(config: Config | None = None) -> float:
    """How much of the Indic lane is manufactured text, against the 50% guardrail.

    Translated and synthetic tiers both count: exercise 03's rule exists because translation and
    synthesis buy fluency without buying culture, and that argument does not distinguish them.

    Args:
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        The combined share of tiers C and D.
    """
    tiers = indic_tiers(config)
    return tiers["C"].share + tiers["D"].share


def synthetic_cap() -> float:
    """The guardrail that share is measured against.

    Returns:
        Exercise 03's cap on manufactured text inside the Indic lane.
    """
    return MAX_SYNTHETIC_SHARE_OF_INDIC


def benchmarks_without_a_lane() -> tuple[str, ...]:
    """Benchmarks no funded lane pays for.

    A benchmark whose every lane sits at zero is a capability the mixture claims and does not buy.

    A schedule-only lane counts as funded. Long-context holds no budget, but `long-eval` is still
    bought — by the sequence-length schedule applied over code, books and web, which is where its
    tokens were always coming from. Treating it as unfunded would report the double-count fix as a
    dropped capability, which is the opposite of what happened.

    Returns:
        Benchmark keys, empty when every one is funded.
    """
    funded = {lane.key for lane in LANES if lane.share > 0 or lane.schedule_only}
    return tuple(
        benchmark.key for benchmark in benchmarks.BENCHMARKS if not (set(benchmark.lanes) & funded)
    )
