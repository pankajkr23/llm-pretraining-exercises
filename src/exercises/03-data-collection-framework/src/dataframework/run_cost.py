"""Price the training run against the budget that is actually shipping.

`records/cost.json` carried this as a static block: 6 x 40e9 x 15e12, worked once by hand against a
15T budget and never touched again. The recommended budget has since moved to 16.8T, so the chapter
that asks what it costs was answering for a run nobody proposes — 2.50M H100-hours where
the plan implies 2.80M, and $5.0M where it implies $5.6M. Twelve percent low, silently.

This is the second time this shape of bug has turned up in this package; `vocab_trade.py` exists
because the vocabulary trade had gone stale the same way. A derivation that is copied rather than
computed goes stale without anybody editing it, which is what makes it worth a module rather than a
comment.

The three constants are the ones `docs/DECISIONS.md` §4.4 already uses, so nothing new is assumed.
"""

from typing import Any

from .vocab_trade import (
    FLOPS_PER_PARAM_TOKEN,
    H100_FLOPS,
    H100_USD_PER_HOUR,
    SECONDS_PER_HOUR,
    USD_TO_INR,
)


def price_run(params: float, seen_tokens: float) -> dict[str, Any]:
    """Work out what one full pass of the schedule costs in compute.

    Args:
        params: Model parameters.
        seen_tokens: The budget's seen-token total — what compute is billed on, so the seen figure
            rather than what those tokens are worth.

    Returns:
        The derivation as ordered steps, plus the inputs it was computed from.
    """
    flops = FLOPS_PER_PARAM_TOKEN * params * seen_tokens
    gpu_hours = flops / H100_FLOPS / SECONDS_PER_HOUR
    usd = gpu_hours * H100_USD_PER_HOUR

    return {
        "params": params,
        "tokens": seen_tokens,
        "steps": [
            {
                "label": "training FLOPs",
                "expression": f"6 x {params:.3g} x {seen_tokens:.3g}",
                "value": flops,
                "unit": "FLOPs",
            },
            {
                "label": "H100-hours",
                "expression": f"{flops:.3g} / {H100_FLOPS:.1e} / 3600",
                "value": round(gpu_hours),
                "unit": "H100-hours",
            },
            {
                "label": "compute cost",
                "expression": f"{gpu_hours:.3g} x {H100_USD_PER_HOUR:.2f}",
                "value": round(usd),
                "unit": "USD",
                "inr": round(usd * USD_TO_INR),
            },
        ],
        "recomputed": True,
    }
