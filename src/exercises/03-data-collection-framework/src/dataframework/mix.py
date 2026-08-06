"""Compose a data mix and check it against the framework's rules.

The mix is where every earlier decision becomes a number. Three ideas do the work:

**Seen tokens are not worth tokens** (rule R1). A pool read four times costs four times the compute
and is not worth four times the text. This module reports both, because an earlier version of it
reported only the product — `unique x epochs` — under the name "effective tokens", and that one
word did real damage: it let a compute-side quantity stand in for a data-side claim, and the page
went on to advertise a figure that no schedule can reach. See `worth_tokens` for the arithmetic and
the citation.

**Some tiers must be protected from the selectors** (rule R2). Quality classifiers are
English-biased -- the Atlas records a golden-proxy cosine of 0.876 with the English web band -- so a
per-iteration selector left to itself will quietly starve Tier-2 and Tier-3 Indic. The Always-ON
lane reserves a fixed share of every batch that no selector may touch.

**Repetition is the weakest of the three answers to a small pool** (rule R3). Collecting is best and
slowest; rephrasing is what the frontier actually does; re-reading is what you do with what is left.
The guardrails below bound the third rather than recommending it.
"""

import math
from typing import Any

# ---------------------------------------------------------------- rule R1

# The decay constant fitted by Muennighoff et al., "Scaling Data-Constrained Language Models",
# JMLR v26 (2025), Eq. 18. Their R*_D is the "half-life" of repeated data: at R_D = R*_D a repeated
# token is worth 1 - 1/e of a fresh one.
REPETITION_DECAY = 15.4

# What repetition is worth at infinity, as a multiple of the unique pool: 1 + R*_D. The paper puts
# it plainly -- "if we repeat data, we will not get a better loss than could be obtained with a
# single epoch on U_D + U_D.R*_D fresh tokens". No schedule beats this, so any budget claiming more
# from a pool of a given size is claiming something unreachable rather than merely unevidenced.
WORTH_CEILING_MULTIPLE = 1 + REPETITION_DECAY

# Three points on that curve, all from the same paper, all meaning different things. They used to be
# two constants named "advised" and "hard", and the second conflated 16 epochs (the half-life) with
# 16x the pool (the asymptote) -- two different sixteens.
EPOCHS_NEAR_FREE = 4  # "up to 4 epochs of repeated data yields negligible changes to loss"
EPOCHS_HALF_LIFE = 16  # R*_D ~= 15 repetitions: repeats have lost 1/e of their value
EPOCHS_WORTHLESS = 40  # the paper's own Figure 1: "At 40 epochs, repeating is worthless"

MIN_TIER_SHARE = 0.01  # a tier below 1% is noise, and repeating it is wasted compute

# Rule R2.
ALWAYS_ON_SHARE = 0.08

# And an upper bound. The lane means "not the English-trained scorer", not "no scorer" -- every tier
# in it still passes a purpose-built check. But the lane is the one part of the mixture no general
# quality signal reaches, so it has to stay a reserved minority rather than grow by accretion each
# time a tier looks hard to filter.
ALWAYS_ON_CEILING = 0.20

# Composition guardrail: past this, the "Indic" tier is mostly manufactured text.
MAX_SYNTHETIC_SHARE_OF_INDIC = 0.50


def seen_tokens(unique_tokens: float, epochs: float) -> float:
    """Tokens the model actually processes -- what compute is billed on.

    Linear, and correctly so: reading a pool four times is four times the forward passes. This is
    the honest use of the product that used to be called `effective_tokens`.

    Args:
        unique_tokens: Distinct tokens available.
        epochs: Times the pool is read.

    Returns:
        Tokens processed.
    """
    return unique_tokens * epochs


def worth_tokens(unique_tokens: float, epochs: float) -> float:
    """What those passes are worth, as an equivalent quantity of fresh text.

    Muennighoff et al., JMLR v26 (2025), Eq. 18:

        D' = U_D + R*_D . U_D . (1 - e^(-R_D / R*_D)),  R_D = epochs - 1

    Four passes are worth 3.73x the pool, not 4x; sixteen are worth 10.6x, not 16x; and no number of
    passes exceeds `WORTH_CEILING_MULTIPLE`. Measured on English web text (C4, OSCAR) at up to 9B
    parameters and 900B tokens, so it is the best available number and not a measurement of this
    corpus -- ATLAS (ICLR 2026) fits the same curve shape per language and finds Hindi's tail bends
    upward sooner.

    Args:
        unique_tokens: Distinct tokens available.
        epochs: Times the pool is read.

    Returns:
        The equivalent quantity of fresh text.
    """
    if epochs <= 1:
        return unique_tokens * max(epochs, 0)
    repetitions = epochs - 1
    decayed = REPETITION_DECAY * (1 - math.exp(-repetitions / REPETITION_DECAY))
    return unique_tokens * (1 + decayed)


def compose(tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Compose tiers into a mix and compute its shares.

    Args:
        tiers: Each with `name`, `unique_tokens`, `epochs`, and optionally `is_indic`,
            `is_synthetic` and `always_on`.

    Returns:
        The composed mix: per-tier seen tokens and shares, plus totals and the Indic breakdown.
    """
    composed: list[dict[str, Any]] = []
    for tier in tiers:
        unique = tier.get("unique_tokens") or 0
        # `or 1` rather than a default, so an explicit None reads as "unstated" instead of raising
        # a TypeError three frames later in the arithmetic.
        epochs = tier.get("epochs")
        epochs = 1 if epochs is None else epochs
        composed.append(
            {
                **tier,
                "seen_tokens": seen_tokens(unique, epochs),
                "worth_tokens": worth_tokens(unique, epochs),
            }
        )

    total_seen = sum(row["seen_tokens"] for row in composed) or 1.0
    total_unique = sum(row.get("unique_tokens", 0) for row in composed)
    total_worth = sum(row["worth_tokens"] for row in composed)

    for row in composed:
        row["share"] = row["seen_tokens"] / total_seen

    indic = [row for row in composed if row.get("is_indic")]
    indic_seen = sum(row["seen_tokens"] for row in indic)
    natural_indic_seen = sum(row["seen_tokens"] for row in indic if not row.get("is_synthetic"))
    natural_indic_worth = sum(row["worth_tokens"] for row in indic if not row.get("is_synthetic"))

    return {
        "tiers": composed,
        "total_seen_tokens": total_seen,
        "total_unique_tokens": total_unique,
        "total_worth_tokens": total_worth,
        "indic_share": indic_seen / total_seen,
        "natural_indic_share": natural_indic_seen / total_seen,
        # The same slice priced both ways. Shipping only the first overstates the scarcest tier on
        # the page, which is exactly the mistake this pair of numbers exists to stop.
        "natural_indic_worth_share": natural_indic_worth / total_seen,
        "synthetic_share_of_indic": (
            (indic_seen - natural_indic_seen) / indic_seen if indic_seen else 0.0
        ),
        "always_on_share": sum(row["share"] for row in composed if row.get("always_on")),
    }


def check(mix: dict[str, Any]) -> list[dict[str, str]]:
    """Check a composed mix against the guardrails.

    Errors mean the mix is not buildable; warnings mean it is buildable but someone should look.

    Args:
        mix: Output of `compose`.

    Returns:
        One entry per finding, each with `level`, `tier` and `message`.
    """
    findings: list[dict[str, str]] = []

    for row in mix["tiers"]:
        name = row.get("name", "<unnamed>")
        # `compose` normalises its own copy, but the original key survives in the row, so an
        # explicit None would reach the comparisons below and raise rather than be reported.
        epochs = row.get("epochs")
        epochs = 1 if epochs is None else epochs

        # Priced both ways, so a schedule can be judged on what it buys rather than on how many
        # times it reads. A tier can sit under every epoch threshold and still be paying for passes
        # that return almost nothing.
        unique = row.get("unique_tokens", 0)
        efficiency = (row["worth_tokens"] / row["seen_tokens"]) if row["seen_tokens"] else 1.0

        if epochs > EPOCHS_WORTHLESS:
            findings.append(
                {
                    "level": "error",
                    "tier": name,
                    "message": (
                        f"{epochs} epochs is past {EPOCHS_WORTHLESS}, where the measured value of "
                        "another pass is zero; this is compute spent to learn nothing."
                    ),
                }
            )
        elif epochs > EPOCHS_HALF_LIFE:
            findings.append(
                {
                    "level": "warning",
                    "tier": name,
                    "message": (
                        f"{epochs} epochs is past the {EPOCHS_HALF_LIFE}-epoch half-life; these "
                        f"passes are worth {efficiency:.0%} of what they cost."
                    ),
                }
            )
        elif epochs > EPOCHS_NEAR_FREE:
            findings.append(
                {
                    "level": "warning",
                    "tier": name,
                    "message": (
                        f"{epochs} epochs is past the ~{EPOCHS_NEAR_FREE}-epoch point where "
                        f"repetition is near-free; these passes are worth {efficiency:.0%} of "
                        "what they cost."
                    ),
                }
            )

        # A tier with no stated pool cannot be checked against a ceiling expressed as a multiple
        # of that pool. It used to skip the check in silence, which reads identically to passing it.
        if not unique:
            findings.append(
                {
                    "level": "warning",
                    "tier": name,
                    "message": (
                        "no unique-token pool stated, so the repetition ceiling cannot be checked "
                        "for this tier — it is unassessed, not clean."
                    ),
                }
            )

        # A negative or absent schedule is a malformed plan, not a small one. `epochs` defaulting
        # to 1 is deliberate and fine; a negative count produced a negative budget that still
        # reported buildable.
        if epochs < 0:
            findings.append(
                {
                    "level": "error",
                    "tier": name,
                    "message": (
                        f"{epochs} epochs is not a schedule; a pool cannot be read a negative "
                        "number of times."
                    ),
                }
            )

        # The one that no epoch count catches: a plan can ask a pool for more than repetition can
        # ever yield from it, and arithmetic will happily print the answer.
        if unique and row["seen_tokens"] > unique * WORTH_CEILING_MULTIPLE:
            findings.append(
                {
                    "level": "error",
                    "tier": name,
                    "message": (
                        f"schedule asks {row['seen_tokens'] / unique:.1f}x the unique pool, above "
                        f"the {WORTH_CEILING_MULTIPLE:.1f}x that infinite repetition can be worth; "
                        "no number of passes reaches this."
                    ),
                }
            )

        if row["share"] < MIN_TIER_SHARE and epochs > 1:
            findings.append(
                {
                    "level": "warning",
                    "tier": name,
                    "message": (
                        f"share {row['share']:.2%} is under {MIN_TIER_SHARE:.0%} yet it is "
                        f"repeated {epochs}x — repeating a rounding error."
                    ),
                }
            )

    if mix["synthetic_share_of_indic"] > MAX_SYNTHETIC_SHARE_OF_INDIC:
        findings.append(
            {
                "level": "warning",
                "tier": "indic",
                "message": (
                    f"{mix['synthetic_share_of_indic']:.0%} of the Indic tier is synthetic. "
                    "Translation and synthesis buy fluency; they do not buy culture."
                ),
            }
        )

    if mix["always_on_share"] > ALWAYS_ON_CEILING:
        findings.append(
            {
                "level": "warning",
                "tier": "always-on",
                "message": (
                    f"the Always-ON lane is {mix['always_on_share']:.1%}, above the "
                    f"{ALWAYS_ON_CEILING:.0%} ceiling; the lane bypasses the general quality "
                    "scorer, so it has to stay a reserved minority rather than grow by accretion."
                ),
            }
        )

    if mix["always_on_share"] < ALWAYS_ON_SHARE:
        findings.append(
            {
                "level": "warning",
                "tier": "always-on",
                "message": (
                    f"the Always-ON lane is {mix['always_on_share']:.1%}, below the reserved "
                    f"{ALWAYS_ON_SHARE:.0%}; English-biased selectors will starve the thin tiers."
                ),
            }
        )

    return findings


def is_buildable(findings: list[dict[str, str]]) -> bool:
    """Whether a mix has no blocking findings.

    Args:
        findings: Output of `check`.

    Returns:
        False if any finding is an error.
    """
    return not any(finding["level"] == "error" for finding in findings)
