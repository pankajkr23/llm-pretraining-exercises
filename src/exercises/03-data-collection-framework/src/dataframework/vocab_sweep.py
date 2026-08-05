"""Find the vocabulary size where two opposing costs cross.

Growing the vocabulary cuts fertility — more Indic sequences become single tokens, so every corpus
shrinks in tokens — but it grows the softmax and embedding matrices, which cost compute on every
forward pass. One curve falls, the other rises, and the answer is where the net is deepest.

The sweep emits **the whole curve, not just the peak**. A reader who only sees "V = 208,896" has to
take it on trust; a reader who sees the curve can watch it be flat-ish for 60K either side and judge
for themselves how much the precise number matters.
"""

from typing import Any

from .fertility import D_MODEL_DEFAULT

# The sweep range from `docs/TODO.md` task 2.3.
V_MIN = 64_000
V_MAX = 320_000
V_STEP = 8_000

# Fertility improves with vocabulary, but with diminishing returns: each doubling buys less than the
# last. Modelled as a log-law anchored on a measured reference point, so the shape comes from data
# rather than taste. Until task 2.2b runs there is no measured anchor — see `sweep`.
DEFAULT_LAYERS = 48


def softmax_cost_fraction(vocab_size: int, d_model: int, n_layers: int = DEFAULT_LAYERS) -> float:
    """Estimate the share of forward-pass compute spent on the vocabulary projection.

    Args:
        vocab_size: Candidate vocabulary size.
        d_model: Model width.
        n_layers: Transformer layers.

    Returns:
        The vocabulary projection's share of per-token forward FLOPs, in [0, 1).
    """
    # Per token: 2 * d_model * V for the output projection, against ~12 * d_model^2 per layer.
    vocab_flops = 2 * d_model * vocab_size
    body_flops = 12 * d_model * d_model * n_layers
    return vocab_flops / (vocab_flops + body_flops)


def fertility_at(vocab_size: int, reference_vocab: int, reference_fertility: float) -> float:
    """Project fertility at a candidate vocabulary from one measured reference point.

    Args:
        vocab_size: Candidate vocabulary size.
        reference_vocab: Vocabulary size the reference fertility was measured at.
        reference_fertility: Measured tokens per word at `reference_vocab`.

    Returns:
        Projected tokens per word, floored at 1.0 — a token can never be shorter than a word.
    """
    if vocab_size <= 0 or reference_vocab <= 0:
        return reference_fertility
    # Log-law: equal multiplicative steps in V buy equal absolute fertility gains.
    import math

    scale = math.log(vocab_size / reference_vocab) / math.log(2)
    # Each doubling of V removes ~8% of the *excess* fertility above 1.0.
    excess = max(reference_fertility - 1.0, 0.0)
    return max(1.0, 1.0 + excess * (0.92**scale))


def sweep(
    reference_vocab: int,
    reference_fertility: float,
    *,
    d_model: int = D_MODEL_DEFAULT,
    n_layers: int = DEFAULT_LAYERS,
    v_min: int = V_MIN,
    v_max: int = V_MAX,
    v_step: int = V_STEP,
) -> list[dict[str, float]]:
    """Sweep vocabulary size and compute the net benefit at each point.

    Args:
        reference_vocab: Vocabulary the reference fertility was measured at.
        reference_fertility: Measured tokens per word at that vocabulary.
        d_model: Model width.
        n_layers: Transformer layers.
        v_min: Smallest vocabulary to try.
        v_max: Largest vocabulary to try.
        v_step: Step between candidates.

    Returns:
        One row per candidate, each with `vocab_size`, `fertility`, `softmax_cost`,
        `token_saving` and `net_benefit`.
    """
    baseline_fertility = fertility_at(v_min, reference_vocab, reference_fertility)
    baseline_cost = softmax_cost_fraction(v_min, d_model, n_layers)

    curve: list[dict[str, float]] = []
    for vocab_size in range(v_min, v_max + 1, v_step):
        fertility = fertility_at(vocab_size, reference_vocab, reference_fertility)
        cost = softmax_cost_fraction(vocab_size, d_model, n_layers)
        # Fewer tokens per word is a proportional saving on the entire training run.
        token_saving = (baseline_fertility - fertility) / baseline_fertility
        curve.append(
            {
                "vocab_size": vocab_size,
                "fertility": round(fertility, 4),
                "softmax_cost": round(cost, 6),
                "token_saving": round(token_saving, 6),
                "net_benefit": round(token_saving - (cost - baseline_cost), 6),
            }
        )
    return curve


def find_peak(curve: list[dict[str, float]]) -> dict[str, float] | None:
    """Pick the point of greatest net benefit.

    Args:
        curve: Output of `sweep`.

    Returns:
        The best row, or `None` for an empty curve.
    """
    return max(curve, key=lambda row: row["net_benefit"]) if curve else None


def round_to_multiple(value: int, multiple: int = 128) -> int:
    """Round a vocabulary size up to a hardware-friendly multiple.

    Tensor cores want the vocabulary dimension to be a multiple of 128; the recommended
    `V = 208,896` is `1,632 x 128` for exactly this reason.

    Args:
        value: Raw vocabulary size.
        multiple: Alignment.

    Returns:
        The next multiple at or above `value`.
    """
    return -(-value // multiple) * multiple


def summarise(curve: list[dict[str, float]]) -> dict[str, Any]:
    """Summarise a sweep for export.

    Args:
        curve: Output of `sweep`.

    Returns:
        The peak, the hardware-aligned recommendation, and the full curve.
    """
    peak = find_peak(curve)
    return {
        "peak": peak,
        "recommended_vocab": round_to_multiple(int(peak["vocab_size"])) if peak else None,
        "curve": curve,
    }
