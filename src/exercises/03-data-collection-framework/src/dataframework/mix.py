"""Compose a data mix and check it against the framework's rules.

The mix is where every earlier decision becomes a number. Two ideas do the work:

**Effective tokens = unique pool x epochs** (rule R1). Budgeting as though every token is seen once
is the "Chinchilla-brained" error the Atlas corrects in Addendum B: repetition is nearly free up to
~4 epochs, so a 300B unique Indic pool read 4 times contributes 1.2T *seen* tokens. That reframing
lifts the natural-Indic share from ~2% to ~8% without collecting a single new document — which is
why the guardrails below are about epochs, not just volume.

**Some tiers must be protected from the selectors** (rule R2). Quality classifiers are
English-biased — the Atlas records a golden-proxy cosine of 0.876 with the English web band — so a
per-iteration selector left to itself will quietly starve Tier-2 and Tier-3 Indic. The Always-ON
lane reserves a fixed share of every batch that no selector may touch.
"""

from typing import Any

# Rule R1 guardrails.
MAX_EPOCHS_HARD = 16  # beyond ~16x the unique pool, repetition stops paying (R*_D ~ 15)
MAX_EPOCHS_ADVISED = 4  # up to ~4 epochs is near-free; past that, returns decay
MIN_TIER_SHARE = 0.01  # a tier below 1% is noise, and repeating it is wasted compute

# Rule R2.
ALWAYS_ON_SHARE = 0.08

# Composition guardrail: past this, the "Indic" tier is mostly manufactured text.
MAX_SYNTHETIC_SHARE_OF_INDIC = 0.50


def effective_tokens(unique_tokens: float, epochs: float) -> float:
    """Compute seen tokens from a unique pool and an epoch schedule (R1).

    Args:
        unique_tokens: Distinct tokens available.
        epochs: Times the pool is read.

    Returns:
        Tokens the model actually sees.
    """
    return unique_tokens * epochs


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
        seen = effective_tokens(tier.get("unique_tokens", 0), tier.get("epochs", 1))
        composed.append({**tier, "seen_tokens": seen})

    total_seen = sum(row["seen_tokens"] for row in composed) or 1.0
    total_unique = sum(row.get("unique_tokens", 0) for row in composed)

    for row in composed:
        row["share"] = row["seen_tokens"] / total_seen

    indic = [row for row in composed if row.get("is_indic")]
    indic_seen = sum(row["seen_tokens"] for row in indic)
    natural_indic_seen = sum(row["seen_tokens"] for row in indic if not row.get("is_synthetic"))

    return {
        "tiers": composed,
        "total_seen_tokens": total_seen,
        "total_unique_tokens": total_unique,
        "indic_share": indic_seen / total_seen,
        "natural_indic_share": natural_indic_seen / total_seen,
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
        epochs = row.get("epochs", 1)

        if epochs > MAX_EPOCHS_HARD:
            findings.append(
                {
                    "level": "error",
                    "tier": name,
                    "message": (
                        f"{epochs} epochs exceeds the hard ceiling of {MAX_EPOCHS_HARD}; "
                        "past ~16x the unique pool, repetition stops buying anything."
                    ),
                }
            )
        elif epochs > MAX_EPOCHS_ADVISED:
            findings.append(
                {
                    "level": "warning",
                    "tier": name,
                    "message": (
                        f"{epochs} epochs is past the ~{MAX_EPOCHS_ADVISED}-epoch point where "
                        "repetition is near-free; returns decay from here."
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
