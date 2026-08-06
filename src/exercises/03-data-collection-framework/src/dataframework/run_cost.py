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

# One scale, used everywhere this page prices something it estimated rather than observed.
#
# A figure like "$5,600,000" reads as though somebody costed it. What is behind it is 6ND — solid —
# multiplied by two assumptions that are not: a sustained 4.0e14 FLOP/s, which is an MFU guess that
# moves +/-30% between real runs, and a $2.00 list rate that this project's own record calls a list
# price, "negotiated well below it" at reservation scale and below that again on spot. Those
# compound to a band several times wide, and seven significant figures claim a precision nobody has.
#
# The order of magnitude is what the decision turns on, so the order of magnitude is what is shown.
COST_BANDS: tuple[tuple[float, str], ...] = (
    (1e4, "$"),  # thousands
    (1e5, "$$"),  # tens of thousands
    (1e6, "$$$"),  # hundreds of thousands
    (1e7, "$$$$"),  # millions
    (1e8, "$$$$$"),  # tens of millions
)

COST_BAND_LEGEND = "$ thousands · $$ tens of thousands · $$$ hundreds of thousands · $$$$ millions"


def cost_band(usd: float) -> str:
    """Bracket a dollar figure to its order of magnitude.

    Args:
        usd: The estimated cost.

    Returns:
        A run of dollar signs, one per order of magnitude above a thousand.
    """
    for ceiling, band in COST_BANDS:
        if usd < ceiling:
            return band
    return COST_BANDS[-1][1] + "$"


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
                # The figure stays in the record and the page renders the band. Keeping the number
                # is what makes the arithmetic auditable and lets each fork be priced by its share;
                # showing seven significant figures is what claims a precision nobody has. The
                # hours above are printed as a number because 6ND is arithmetic and throughput is
                # their only soft term; the money adds a list price on top of that.
                "label": "compute cost",
                "expression": f"{gpu_hours:.3g} x {H100_USD_PER_HOUR:.2f}",
                "value": round(usd),
                "unit": "USD",
                "inr": round(usd * USD_TO_INR),
                "band": cost_band(usd),
                "band_legend": COST_BAND_LEGEND,
            },
        ],
        "recomputed": True,
    }
