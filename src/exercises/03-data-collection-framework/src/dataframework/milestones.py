"""Turn the milestone ladder into complete, checked mix configurations.

The ladder is the report's central budget argument, and each rung has to be a real mix rather than a
headline: a milestone you cannot compose is not a plan. Each preset is built from the same tier
shape, scaled to its target, and run through the same guardrails as any other mix — so a rung that
only reaches its number by reading a thin pool sixteen times says so.
"""

from typing import Any

from .mix import check, compose, is_buildable

# Tier shape shared by every rung, expressed as shares of the seen-token budget. Shares follow
# `docs/DECISIONS.md`; the naturalness and epoch structure is what varies between rungs.
#
# `kind` is a second lens on the same numbers: skills are the tiers that teach the model to *do*
# something (code, maths, tool use), knowledge the tiers that teach it what is *true*. A reader who
# cannot hold ten tiers in mind can hold two, and the split is worth knowing on its own — a corpus
# that is 90% knowledge produces a model that recites.
TIER_SHAPE: tuple[dict[str, Any], ...] = (
    {
        "name": "english-web-hq",
        "kind": "knowledge",
        "share": 0.150,
        "epochs": 1,
        "capabilities": ["knowledge"],
        "sources": "FineWeb-Edu · Nemotron-CC",
        "why": (
            "Filtered English web is where the model learns to reason at all, and the capability "
            "an India-first model is least likely to be judged on. It gave up five points to the "
            "skills tiers on the argument that code teaches reasoning too — which is a bet, and "
            "the tier to watch if general reasoning regresses."
        ),
    },
    {
        "name": "code",
        "kind": "skills",
        "share": 0.200,
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
        "kind": "skills",
        "share": 0.120,
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
        "kind": "knowledge",
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
        "kind": "knowledge",
        "share": 0.143,
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
        "kind": "knowledge",
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
        "kind": "skills",
        "share": 0.080,
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
        "name": "indic-knowledge-systems",
        "kind": "knowledge",
        "share": 0.030,
        "epochs": 2,
        "is_indic": True,
        "always_on": True,
        "capabilities": ["india-context", "knowledge"],
        "sources": "Ayurveda and Siddha · Jyotish and Panchang · NDLI and DLI scans",
        "why": (
            "Indian knowledge systems, and the one tier no frontier model has any reason to build. "
            "Most of it exists only as scanned pages in institutional archives, so it is an OCR "
            "problem before it is a data problem — which is exactly why nobody else has it and why "
            "it is worth the collection cost."
        ),
    },
    {
        "name": "indic-civilizational",
        "kind": "knowledge",
        "share": 0.020,
        "epochs": 2,
        "is_indic": True,
        "always_on": True,
        "capabilities": ["india-context", "knowledge"],
        "sources": "Vedic corpus · classical Sanskrit · Upanishads · Dharmashastra",
        "why": (
            "The civilizational literature an India-first model is expected to reason from rather "
            "than merely quote. It is small, largely public-domain, and mostly digitised badly. "
            "Reading it twice costs almost nothing; not having it at all is the difference between "
            "a model that knows India and one that has read about it."
        ),
    },
    {
        "name": "general-web",
        "kind": "knowledge",
        "share": 0.127,
        "epochs": 1,
        "capabilities": ["knowledge"],
        "sources": "FineWeb · CommonCrawl derivatives",
        "why": (
            "Breadth — everyday knowledge and common sense, and the least interesting tier per "
            "token. It used to be the largest here, and gave up seven points to code, maths and "
            "tool use, "
            "because the assignment names those as primary capabilities and a mixture is zero-sum."
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
            # Demand, not supply, and the name now says so. This field used to be `unique_tokens`,
            # which reads as an inventory and is computed from a share we chose: at 16.8T the
            # indic-natural tier came out at 336B, and chapter 2 printed that as "every verified
            # corpus anyone has assembled, added together". The catalogue can commit 84.9B. The
            # number a tier holds is in `sourcing`, never here. Correction X22.
            | {"unique_tokens_required": seen / tier["epochs"]}
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
