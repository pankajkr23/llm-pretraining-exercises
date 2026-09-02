"""The Session 5 dataset inventory, transcribed as data rather than quoted as prose.

Session 5 §4 makes the argument this module implements: *"A shopping list only works when the data
actually exists... each capability slot must be sized against the actual datasets that can feed
it."* So the lane supplies used everywhere downstream are **summed from these rows**, never quoted
from the session's slot headlines.

That distinction is not pedantry — it caught two things.

**One.** The session's two widgets disagree with each other. Its inventory groups general web and
STEM into a single slot of 4.8T, while its supply check prices them separately at 4.5T and 250B.
Summing the named datasets gives 4.691T of web and 146B of STEM. The STEM gap is the one that
matters: 146B against a 240B demand is a lane that cannot be funded from unique text, where 250B
against 240B is a lane with margin. `supply.py` uses the itemised figure and says so.

**Two.** Two Indic rows — Samanantar and BPCC — are listed with no token count at all. The slot
headline (276B) exceeds the four rows that do carry counts (270.9B) by 5.1B, which is what those
two must hold between them. That residual is recorded as a residual. Splitting it across the two
rows would have produced two plausible numbers that nobody measured, which is the exact failure
this exercise is written against.

Every row's figures are typed by where they came from:

- ``confirmed``  the session states it is confirmed from its own sources (Sangraha and V4 rows)
- ``approximate`` the session's own caveat: *"Dataset sizes are approximate and being verified"*
- ``unstated``   the inventory lists the dataset but gives no figure
"""

from dataclasses import dataclass, field

# Lane keys. Seven, matching the seven bands of the notes' mixture composer, except that the
# inventory's single "General web & STEM" slot is split in two — the composer gives web and STEM
# separate shares (34% and 12%), so they cannot share one supply pool without one of them borrowing
# the other's headroom.
LANES = ("web", "code", "stem", "indic", "reasoning", "long_context", "agentic")

# The notes flag these as confirmed from its own sources rather than approximate:
# "Sangraha and V4 numbers are confirmed from our sources."
_CONFIRMED_SOURCES = ("AI4Bharat", "V4 run (confirmed)", "V4 corpus")


@dataclass(frozen=True)
class Dataset:
    """One row of the Session 5 inventory.

    Attributes:
        name: The dataset as the inventory names it.
        source: Who publishes it.
        lane: Which capability slot it feeds.
        samples: Approximate sample count, or None where the inventory gives none.
        tokens: Approximate token count, or None where the inventory gives none.
        licence: Licence as stated, or None where unstated.
        tier: Session 3 provenance tier (A/B/C/D), or None where the inventory leaves it blank.
        provenance: How much the figures can be leaned on — see the module docstring.
        note: Anything about the row that changes how it may be counted.
    """

    name: str
    source: str
    lane: str
    samples: float | None = None
    tokens: float | None = None
    licence: str | None = None
    tier: str | None = None
    provenance: str = "approximate"
    note: str = ""


def _row(
    name: str,
    source: str,
    lane: str,
    samples: float | None = None,
    tokens: float | None = None,
    licence: str | None = None,
    tier: str | None = None,
    note: str = "",
) -> Dataset:
    """Build a row, deriving its provenance from the source the session named.

    Args:
        name: Dataset name.
        source: Publisher, as the inventory gives it.
        lane: Capability slot.
        samples: Sample count, or None.
        tokens: Token count, or None.
        licence: Licence string, or None.
        tier: Provenance tier, or None.
        note: Free-text caveat.

    Returns:
        The row, with `provenance` set to `unstated` when it carries no token count, `confirmed`
        when the session vouches for the source, and `approximate` otherwise.
    """
    if tokens is None:
        provenance = "unstated"
    elif source in _CONFIRMED_SOURCES:
        provenance = "confirmed"
    else:
        provenance = "approximate"
    return Dataset(name, source, lane, samples, tokens, licence, tier, provenance, note)


# --------------------------------------------------------------------------------- the inventory

DATASETS: tuple[Dataset, ...] = (
    # ---- Code · inventory slot headline 1.1T -----------------------------------------------
    _row(
        "The Stack v2",
        "BigCode / Software Heritage",
        "code",
        samples=600e6,
        tokens=900e9,
        licence="permissive + opt-out",
        tier="B",
    ),
    _row(
        "D3 Code",
        "V4 run (confirmed)",
        "code",
        samples=250e6,
        tokens=199e9,
        licence="V4 lineage",
    ),
    _row(
        "CommitPack / CommitPackFT",
        "BigCode",
        "code",
        samples=4e6,
        tokens=4e9,
        licence="mixed / permissive",
    ),
    # ---- Agentic & tool-use · inventory slot headline 627M ---------------------------------
    # The inventory itself flags this slot: "SLOT RUNS THIN". Nine datasets and the whole lane is
    # smaller than a single code dataset's rounding error.
    _row(
        "ToolBench",
        "OpenBMB",
        "agentic",
        samples=120e3,
        tokens=80e6,
        licence="Apache-2.0",
        tier="D",
        note="single-turn API calls; short examples, so samples overstate its weight in tokens",
    ),
    _row(
        "Glaive function-calling v2",
        "Glaive",
        "agentic",
        samples=113e3,
        tokens=50e6,
    ),
    _row(
        "ToolACE",
        "ToolACE",
        "agentic",
        samples=110e3,
        tokens=60e6,
        tier="A/D",
        note="the inventory gives two tiers for one row; treated as unresolved, not as A",
    ),
    _row(
        "xLAM / APIGen",
        "Salesforce",
        "agentic",
        samples=60e3,
        tokens=25e6,
        licence="CC-BY (mixed)",
    ),
    _row(
        "Nexus / NexusRaven",
        "Nexusflow",
        "agentic",
        samples=40e3,
        tokens=30e6,
        licence="CC-BY-4.0",
        tier="A",
    ),
    _row(
        "SWE-smith",
        "SWE-smith",
        "agentic",
        samples=26e3,
        tokens=120e6,
        licence="task licenses",
    ),
    _row(
        "Hermes function-calling",
        "NousResearch",
        "agentic",
        samples=15e3,
        tokens=22e6,
    ),
    _row(
        "OpenHands rollouts",
        "All-Hands / OpenHands",
        "agentic",
        samples=10e3,
        tokens=90e6,
        licence="mixed",
        note="long trajectories: 9,000 tokens per sample against ToolBench's 667",
    ),
    _row(
        "SWE-Gym",
        "SWE-Gym",
        "agentic",
        samples=2.4e3,
        tokens=150e6,
        note=(
            "the extreme of the same point: 2,400 samples carry 150M tokens, 62,500 per sample, "
            "and it is the largest agentic dataset in the inventory by tokens while being the "
            "smallest but one by samples"
        ),
    ),
    # ---- Reasoning & math · inventory slot headline 85.1B ----------------------------------
    _row(
        "AON",
        "V4 corpus",
        "reasoning",
        samples=40e6,
        tokens=78e9,
        note="92% of the lane's tokens sit in this one V4-lineage set",
    ),
    _row(
        "OpenMathReasoning",
        "NVIDIA",
        "reasoning",
        samples=3.2e6,
        tokens=2e9,
    ),
    _row(
        "OpenThoughts2",
        "OpenThoughts",
        "reasoning",
        samples=1.1e6,
        tokens=3e9,
        note="exercise 04 cleaned OpenThoughts-114k, an earlier release of this line",
    ),
    _row(
        "NuminaMath",
        "Numina",
        "reasoning",
        samples=860e3,
        tokens=500e6,
        licence="Apache / CC-BY",
    ),
    _row(
        "OpenR1-Math",
        "Hugging Face",
        "reasoning",
        samples=220e3,
        tokens=1.6e9,
        note="R1-distilled",
    ),
    # ---- Long-context · inventory slot headline 100B ---------------------------------------
    # Both rows are packings of text counted elsewhere. See `supply.py` for what that costs the
    # lane, which is all of it.
    _row(
        "Repo-packed code (32K+)",
        "packed from code corpora",
        "long_context",
        samples=1.5e6,
        tokens=60e9,
        licence="permissive",
        note="packed from the code corpora above — not additional text",
    ),
    _row(
        "Book-length corpora (packed)",
        "books + long docs",
        "long_context",
        samples=400e3,
        tokens=40e9,
        licence="mixed / public-domain",
        note="packed long documents; the only long-context row that is not re-counted code",
    ),
    # ---- Indic · inventory slot headline 276B ----------------------------------------------
    _row(
        "Sangraha (synthetic)",
        "AI4Bharat",
        "indic",
        tokens=162e9,
        tier="C",
        note=(
            "the inventory names this row 'synthetic' and tiers it C, which is the translated "
            "tier — see `lanes.py`, where the name and the tag cannot both be honoured"
        ),
    ),
    _row(
        "Samanantar",
        "AI4Bharat",
        "indic",
        samples=49.7e6,
        licence="CC0 / CC-BY",
        note="no token count in the inventory; parallel sentence pairs",
    ),
    _row(
        "Sangraha (verified)",
        "AI4Bharat",
        "indic",
        tokens=64e9,
        note="the verified-native tier — the scarcest and most valuable Indic text in the lane",
    ),
    _row(
        "BPCC (parallel)",
        "AI4Bharat",
        "indic",
        note="no sample or token count in the inventory",
    ),
    _row(
        "Sangraha (unverified)",
        "AI4Bharat",
        "indic",
        samples=15e6,
        tokens=24e9,
    ),
    _row(
        "IndicCorpV2",
        "AI4Bharat",
        "indic",
        samples=10e6,
        tokens=20.9e9,
        licence="CC-BY / mixed",
    ),
    # ---- General web ------------------------------------------------------------------------
    _row(
        "DCLM-Baseline",
        "DataComp-LM",
        "web",
        samples=2.6e9,
        tokens=2.6e12,
        licence="mixed / CommonCrawl",
    ),
    _row(
        "FineWeb-Edu",
        "FineWeb-Edu",
        "web",
        samples=1.3e9,
        tokens=1.3e12,
        licence="ODC-By",
    ),
    _row(
        "D2 Web-Diverse",
        "V4 corpus",
        "web",
        samples=780e6,
        tokens=627e9,
    ),
    _row(
        "D1 Web-Foundation",
        "V4 corpus",
        "web",
        samples=200e6,
        tokens=164e9,
    ),
    # ---- STEM / math --------------------------------------------------------------------------
    _row(
        "D4 STEM",
        "V4 corpus",
        "stem",
        tokens=49e9,
    ),
    _row(
        "peS2o",
        "AI2",
        "stem",
        tokens=42e9,
    ),
    _row(
        "proof-pile-2",
        "EleutherAI",
        "stem",
        samples=5e6,
        tokens=55e9,
    ),
)


# The slot totals the notes print above its own rows, kept so the itemised sums can be checked
# against them rather than silently replacing them. Where the two disagree, the disagreement is the
# finding.
NOTES_SLOT_HEADLINES: dict[str, float] = {
    "code": 1.1e12,
    "agentic": 627e6,
    "reasoning": 85.1e9,
    "long_context": 100e9,
    "indic": 276e9,
    # One headline covering two lanes, which is the discrepancy documented above.
    "web+stem": 4.8e12,
}

# The notes' supply-check widget prices the same lanes again, and not identically.
NOTES_SUPPLY_CHECK: dict[str, float] = {
    "code": 1.1e12,
    "agentic": 0.63e9,
    "reasoning": 85e9,
    "long_context": 100e9,
    "indic": 276e9,
    "web": 4.5e12,
    "stem": 250e9,
}


@dataclass(frozen=True)
class LaneSupply:
    """What one lane's rows add up to, and what they fail to add up to.

    Attributes:
        lane: Lane key.
        rows: The datasets feeding it.
        counted_tokens: Sum over rows that carry a token count.
        rows_without_tokens: How many rows carry none.
        headline: The slot total the session printed, where it printed one for this lane alone.
        residual: headline − counted_tokens, when a headline exists. What the uncounted rows must
            hold between them, stated as one number because the inventory does not divide it.
    """

    lane: str
    rows: tuple[Dataset, ...]
    counted_tokens: float
    rows_without_tokens: int
    headline: float | None = None
    residual: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def lane_supply(lane: str) -> LaneSupply:
    """Sum one lane from its rows, and report what the rows do not cover.

    Args:
        lane: One of `LANES`.

    Returns:
        The lane's counted supply, the count of rows with no figure, and the residual against the
        session's headline where one exists for this lane on its own.

    Raises:
        ValueError: If `lane` is not a known lane.
    """
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")

    rows = tuple(row for row in DATASETS if row.lane == lane)
    counted = sum(row.tokens for row in rows if row.tokens is not None)
    missing = sum(1 for row in rows if row.tokens is None)
    headline = NOTES_SLOT_HEADLINES.get(lane)
    residual = None if headline is None else headline - counted

    notes: list[str] = []
    if missing:
        names = ", ".join(row.name for row in rows if row.tokens is None)
        notes.append(
            f"{missing} row(s) carry no token count ({names}); "
            + (
                f"the slot headline leaves {residual / 1e9:.1f}B for them between it and the rows "
                "that do, and the inventory does not say how it divides"
                if residual is not None
                else "and no slot headline exists to bound them"
            )
        )
    return LaneSupply(lane, rows, counted, missing, headline, residual, tuple(notes))


def all_supply() -> dict[str, LaneSupply]:
    """Every lane summed from its rows.

    Returns:
        Lane key to its `LaneSupply`.
    """
    return {lane: lane_supply(lane) for lane in LANES}


def headline_disagreements() -> list[dict[str, object]]:
    """Where the session's two widgets price the same lane differently.

    The inventory prints a slot total above its rows; the supply check prints a supply figure for
    the same lane. They are not always the same number, and neither always equals the rows.

    Returns:
        One entry per lane, each with the itemised sum, both quoted figures, and the gap that
        matters — itemised against the supply check, which is the figure the session's own demand
        arithmetic is compared to.
    """
    findings: list[dict[str, object]] = []
    for lane, supply in all_supply().items():
        quoted = NOTES_SUPPLY_CHECK.get(lane)
        if quoted is None:
            continue
        gap = supply.counted_tokens - quoted
        if abs(gap) <= 0.02 * quoted:
            continue
        findings.append(
            {
                "lane": lane,
                "itemised": supply.counted_tokens,
                "supply_check": quoted,
                "slot_headline": supply.headline,
                "gap": gap,
                "relative": gap / quoted,
            }
        )
    return findings


def humanise(value: float | None) -> str:
    """Format a token count at the scale it is best read in.

    Args:
        value: A token count, or None where none exists.

    Returns:
        A short string in T/B/M, or an em dash for None.
    """
    if value is None:
        return "—"
    if abs(value) >= 1e12:
        return f"{value / 1e12:.3f}T"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.1f}B"
    return f"{value / 1e6:.0f}M"


def main() -> None:
    """Print the itemised lane supplies beside the two figures the session quotes for them."""
    header = (
        f"{'lane':<13} {'itemised':>10} {'supply check':>13} {'headline':>10} {'gap':>10}  rows"
    )
    print(header)
    print("-" * len(header))
    for lane, supply in all_supply().items():
        quoted = NOTES_SUPPLY_CHECK.get(lane)
        gap = None if quoted is None else supply.counted_tokens - quoted
        print(
            f"{lane:<13} {humanise(supply.counted_tokens):>10} {humanise(quoted):>13} "
            f"{humanise(supply.headline):>10} {humanise(gap):>10}  {len(supply.rows)}"
        )

    findings = headline_disagreements()
    print(f"\nlanes where the itemised sum and the supply check differ by over 2%: {len(findings)}")
    for finding in findings:
        print(
            f"  {finding['lane']:<13} itemised {humanise(finding['itemised']):>8} "
            f"vs supply check {humanise(finding['supply_check']):>8} "
            f"({finding['relative']:+.1%})"
        )

    for supply in all_supply().values():
        for note in supply.notes:
            print(f"\n{supply.lane}: {note}")


if __name__ == "__main__":
    main()
