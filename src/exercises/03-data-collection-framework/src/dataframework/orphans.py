"""Find mix tiers whose removal no benchmark would notice.

An orphan tier is one you are paying for and cannot evaluate: delete it, and every instrument you
own reports the same score. That is not an argument to cut it — culture and coverage matter whether
or not a benchmark sees them — but it *is* an argument you must make out loud, because the honest
version of "this tier is worth 300B tokens" includes "and here is why no number will show it".

Attaching the token count and rupee cost to the warning is the point: an orphan tier costing
₹2 crore is a different conversation from one costing ₹200.
"""

from typing import Any

from .coverage import capabilities_for
from .fertility import training_cost


def detectors_for(tier: dict[str, Any], benchmarks: list[dict[str, Any]]) -> list[str]:
    """Find benchmarks that would register this tier's removal.

    Args:
        tier: A mix tier, optionally carrying `capabilities`.
        benchmarks: Benchmark records.

    Returns:
        Names of benchmarks covering any of the tier's capabilities.
    """
    wanted = set(tier.get("capabilities") or [])
    if not wanted:
        return []
    return [
        benchmark.get("name")
        for benchmark in benchmarks
        if wanted & set(capabilities_for(benchmark))
    ]


def find_orphans(
    mix: dict[str, Any],
    benchmarks: list[dict[str, Any]],
    *,
    n_params: float = 40e9,
) -> list[dict[str, Any]]:
    """Report every tier no benchmark can detect, priced.

    Args:
        mix: Output of `mix.compose`.
        benchmarks: Benchmark records.
        n_params: Model size, for pricing the tier's compute.

    Returns:
        One entry per orphan tier, with its share, token count and cost.
    """
    orphans: list[dict[str, Any]] = []
    for tier in mix["tiers"]:
        if detectors_for(tier, benchmarks):
            continue
        cost = training_cost(tier["seen_tokens"], n_params)
        orphans.append(
            {
                "tier": tier.get("name"),
                "seen_tokens": tier["seen_tokens"],
                "share": tier["share"],
                "gpu_hours": cost["gpu_hours"],
                "usd": cost["usd"],
                "inr": cost["inr"],
                "message": (
                    f"no benchmark would detect removing {tier.get('name')} "
                    f"({tier['share']:.1%} of the mix, ~${cost['usd']:,.0f} of compute)"
                ),
            }
        )
    return orphans
