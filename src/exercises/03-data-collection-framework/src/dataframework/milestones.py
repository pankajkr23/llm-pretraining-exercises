"""Turn the milestone ladder into complete, checked mix configurations.

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
    {
        "name": "english-web-hq",
        "share": 0.200,
        "epochs": 1,
        "capabilities": ["knowledge"],
        "sources": "FineWeb-Edu · Nemotron-CC",
        "why": (
            "Filtered English web is where the model learns to reason at all. It is the cheapest "
            "capability in the mix and the one an India-first model is least likely to be judged "
            "on — but skimping here degrades everything downstream, including Indic reasoning."
        ),
    },
    {
        "name": "code",
        "share": 0.147,
        "epochs": 1,
        "capabilities": ["code", "agentic-coding"],
        "sources": "The Stack v2 (permissive subset)",
        "why": (
            "Code is a primary capability, not a garnish, and it transfers: models trained on more "
            "code reason better in prose. The permissive subset only, because the licence position "
            "on the rest is unresolved."
        ),
    },
    {
        "name": "math-stem",
        "share": 0.080,
        "epochs": 1,
        "capabilities": ["math-reasoning"],
        "sources": "Proof-Pile-2 · OpenWebMath · MegaMath",
        "why": (
            "Reasoning-dense text buys more per token than anything else in the mix. Almost none "
            "of it exists in Indian languages, which is itself one of the findings."
        ),
    },
    {
        "name": "indic-natural",
        "share": 0.080,
        "epochs": 4,
        "is_indic": True,
        "always_on": True,
        "capabilities": ["indic-language"],
        "sources": "Sangraha (verified) · IndicCorp v2 · Varta",
        "why": (
            "The only tier carrying what the languages actually know. It is also the scarcest, "
            "which is why it is read four times and why it sits in the protected lane — an "
            "English-biased quality filter scores it as noise and drops it."
        ),
    },
    {
        "name": "indic-synthetic",
        "share": 0.153,
        "epochs": 1,
        "is_indic": True,
        "is_synthetic": True,
        "capabilities": ["indic-language"],
        "sources": "translation · transliteration · LLM generation",
        "why": (
            "Manufactured text fills the volume the natural pool cannot reach. It buys fluency and "
            "reasoning transfer; it does not buy culture, and counting it as natural Indic is the "
            "commonest way to overstate a corpus."
        ),
    },
    {
        "name": "india-context-english",
        "share": 0.050,
        "epochs": 1,
        "capabilities": ["india-context"],
        "sources": "Indian law · government · news · encyclopaedic",
        "why": (
            "English text about India. It teaches the worldview without paying the tokenizer tax — "
            "and it is the capability the benchmark set can least reliably detect."
        ),
    },
    {
        "name": "agentic-traces",
        "share": 0.053,
        "epochs": 1,
        "always_on": True,
        "capabilities": ["agentic-coding"],
        "sources": "verified tool-use and SWE trajectories",
        "why": (
            "Multi-step tool use has to be trained on, not prompted into existence. Traces are "
            "scarce and filtered hard, so they share the protected lane."
        ),
    },
    {
        "name": "general-web",
        "share": 0.237,
        "epochs": 1,
        "capabilities": ["knowledge"],
        "sources": "FineWeb · CommonCrawl derivatives",
        "why": (
            "Breadth. The least interesting tier per token and the largest by volume, which is the "
            "usual shape of a pre-training corpus."
        ),
    },
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
        # Only rungs the page draws become presets. The research's earlier staging (5T/10T/15T) is
        # the argument for how you reach the seed, not a budget on offer — and composing a full mix
        # for each of them costs the index budget several KB to say nothing the page renders.
        if not record.get("ladder"):
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
