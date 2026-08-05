"""Price the vocabulary decision against the mixture that is actually shipping.

`docs/DECISIONS.md` §4.4 works this arithmetic once, by hand, against a 15T budget with a 25.3%
India slice. Both of those have since changed, and the hand-computed result — 869B tokens, 144,800
H100-hours, ₹2.5 crore — was still being shipped as though it followed from the current plan. A
derivation that is copied rather than computed goes stale silently, which is the failure this
module exists to prevent: the numbers below are recomputed from the mix on every export.

One input cannot be computed and is not pretended otherwise. The fertility improvement a larger
vocabulary buys — about 2.4 to about 1.85 tokens per word — describes the **candidate** tokenizer,
and nobody has trained it. §4.4 promised that task 2.2b would replace both figures with
observations; 2.2b measured five *existing* tokenizers, which cannot say anything about a
tokenizer that does not exist. That promise was unfulfillable and this module says so rather than
inheriting it.
"""

from typing import Any

# Training FLOPs per token for a dense model of N parameters, the standard 6ND accounting.
FLOPS_PER_PARAM_TOKEN = 6

# Sustained throughput assumed for an H100 in BF16, and the list price per hour. Both are planning
# figures: large reservations are negotiated well below list and spot capacity below that again.
H100_FLOPS = 4.0e14
H100_USD_PER_HOUR = 2.00
USD_TO_INR = 86.2

SECONDS_PER_HOUR = 3600

# The one input that is not measurable. Recorded as an assumption with its own provenance rather
# than folded into the arithmetic where a reader cannot see it.
FERTILITY_GAIN = 0.229
FERTILITY_GAIN_NOTE = (
    "Indic fertility from about 2.4 to about 1.85 tokens per word. An estimate about a tokenizer "
    "nobody has trained, so it cannot be confirmed by the fertility run — that run measures "
    "tokenizers that exist. It is the single unmeasured input to this result."
)


def price_vocab_trade(
    params: float,
    seen_tokens: float,
    indic_share: float,
    compute_cost_share: float,
    fertility_gain: float = FERTILITY_GAIN,
) -> dict[str, Any]:
    """Work out what a larger vocabulary saves, from the mixture actually being shipped.

    Args:
        params: Model parameters.
        seen_tokens: The budget's seen-token total.
        indic_share: The Indian-language share of that budget, as a fraction.
        compute_cost_share: Extra forward compute the larger vocabulary costs, as a fraction.
        fertility_gain: Fractional reduction in Indic tokens per word.

    Returns:
        The derivation as ordered steps, plus the return multiple.
    """
    token_saving_share = indic_share * fertility_gain
    tokens_saved = seen_tokens * token_saving_share
    flops_saved = FLOPS_PER_PARAM_TOKEN * params * tokens_saved
    gpu_hours = flops_saved / H100_FLOPS / SECONDS_PER_HOUR
    usd = gpu_hours * H100_USD_PER_HOUR

    return {
        "steps": [
            {
                "label": "token saving",
                "expression": f"{indic_share:.3f} x {fertility_gain:.3f}",
                "value": round(token_saving_share, 4),
                "unit": "share",
                "as_tokens": round(tokens_saved),
            },
            {
                "label": "FLOPs saved",
                "expression": f"6 x {params:.3g} x {tokens_saved:.3g}",
                "value": flops_saved,
                "unit": "FLOPs",
            },
            {
                "label": "GPU-hours",
                "expression": f"{flops_saved:.3g} / {H100_FLOPS:.1e} / 3600",
                "value": round(gpu_hours),
                "unit": "H100-hours",
            },
            {
                "label": "cost saved",
                "expression": f"{gpu_hours:.3g} x {H100_USD_PER_HOUR:.2f}",
                "value": round(usd),
                "unit": "USD",
                "inr": round(usd * USD_TO_INR),
            },
        ],
        # The return on the compute the bigger vocabulary costs. Both sides are shares of the same
        # budget, so they divide directly.
        "return_multiple": round(token_saving_share / compute_cost_share, 2)
        if compute_cost_share
        else None,
        "unmeasured_input": {
            "value": fertility_gain,
            "unit": "share",
            "provenance": "estimated",
            "source": FERTILITY_GAIN_NOTE,
        },
        "recomputed": True,
    }
