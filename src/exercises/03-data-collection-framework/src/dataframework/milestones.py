"""Turn the 5T / 10T / 15T / 20T ladder into complete, checked mix configurations.

The ladder is the report's central budget argument, and each rung has to be a real mix rather than a
headline: a milestone you cannot compose is not a plan. Each preset is built from the same tier
shape, scaled to its target, and run through the same guardrails as any other mix — so a rung that
only reaches its number by reading a thin pool sixteen times says so.
"""

from typing import Any

from .mix import check, compose, is_buildable

# Tier shape shared by every rung, expressed as shares of the seen-token budget. Values follow the
# recommended 15T mix in `docs/DECISIONS.md`; the naturalness and epoch structure is what varies.
TIER_SHAPE: tuple[dict[str, Any], ...] = (
    {"name": "english-web-hq", "share": 0.200, "epochs": 1, "capabilities": ["knowledge"]},
    {"name": "code", "share": 0.147, "epochs": 1, "capabilities": ["code", "agentic-coding"]},
    {"name": "math-stem", "share": 0.080, "epochs": 1, "capabilities": ["math-reasoning"]},
    {
        "name": "indic-natural",
        "share": 0.080,
        "epochs": 4,
        "is_indic": True,
        "always_on": True,
        "capabilities": ["indic-language"],
    },
    {
        "name": "indic-synthetic",
        "share": 0.153,
        "epochs": 1,
        "is_indic": True,
        "is_synthetic": True,
        "capabilities": ["indic-language"],
    },
    {
        "name": "india-context-english",
        "share": 0.050,
        "epochs": 1,
        "capabilities": ["india-context"],
    },
    {"name": "agentic-traces", "share": 0.053, "epochs": 1, "capabilities": ["agentic-coding"]},
    {"name": "general-web", "share": 0.237, "epochs": 1, "capabilities": ["knowledge"]},
)


def build_preset(target_seen_tokens: float) -> dict[str, Any]:
    """Build one milestone as a composed, checked mix.

    Args:
        target_seen_tokens: The rung's seen-token budget.

    Returns:
        The composed mix plus its findings and buildability.
    """
    tiers = []
    for tier in TIER_SHAPE:
        seen = target_seen_tokens * tier["share"]
        tiers.append(
            {key: value for key, value in tier.items() if key != "share"}
            | {"unique_tokens": seen / tier["epochs"]}
        )

    mix = compose(tiers)
    findings = check(mix)
    return {
        "target_seen_tokens": target_seen_tokens,
        "mix": mix,
        "findings": findings,
        "buildable": is_buildable(findings),
    }


def build_all(milestones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build every rung of the ladder.

    Args:
        milestones: Records from `records/milestones.json`, each with `id` and `tokens`.

    Returns:
        One entry per rung, carrying the source record and the composed preset.
    """
    presets = []
    for record in milestones:
        tokens = record.get("tokens")
        if not tokens:
            continue
        preset = build_preset(float(tokens))
        presets.append(
            {
                "id": record.get("id"),
                "recommended": bool(record.get("recommended")),
                "verdict": record.get("verdict"),
                "feasibility": record.get("feasibility"),
                **preset,
            }
        )
    return presets
