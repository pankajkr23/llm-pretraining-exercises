"""Tokenizer fertility — the exchange rate between a script and a compute bill.

Fertility is tokens-per-word. When an English-centric BPE splits Malayalam into 13 tokens where it
splits English into 1.2, every Malayalam sentence costs ~10x more to train on. That tax is why the
Atlas calls the tokenizer decision worth more than 2T extra tokens: it is a multiplier on every
other data decision, and unlike corpus size you cannot buy your way out of it later.

**No fertility number is invented here.** Ground rule 8 requires measurement, so this module
provides the arithmetic and a measurement harness, and reports `unknown` until a real run has
happened (`docs/FERTILITY_MEASUREMENT.md`, task 2.2b). `estimated` is not an option for fertility:
a plausible-looking tax that nobody measured is exactly the failure INV-4 exists to prevent.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

from .models import Value

# Training cost model. All overridable, because each is a modelling choice a reviewer may dispute.
FLOPS_PER_TOKEN_PER_PARAM = 6  # forward + backward, the standard approximation
H100_EFFECTIVE_FLOPS = 4.0e14  # ~40% MFU of 989e12 bf16
USD_PER_GPU_HOUR = 2.0
USD_TO_INR = 88.0
D_MODEL_DEFAULT = 6144  # confirmed for the 40B target (docs/OPEN.md)

# The headline target: the worst-served Tier-A Indic language may cost at most 1.5x English.
PARITY_TARGET = 1.5


def tokens_per_word(token_count: int, word_count: int) -> float:
    """Compute fertility.

    Args:
        token_count: Tokens the tokenizer emitted.
        word_count: Whitespace-separated words in the same text.

    Returns:
        Tokens per word, or 0.0 when there are no words.
    """
    return token_count / word_count if word_count else 0.0


def parity_ratio(worst_indic_fertility: float, english_fertility: float) -> float:
    """Compute the headline parity ratio.

    Args:
        worst_indic_fertility: Fertility of the worst-served Tier-A Indic language.
        english_fertility: Fertility of English on the same tokenizer.

    Returns:
        The ratio; `PARITY_TARGET` or below is the goal.
    """
    return worst_indic_fertility / english_fertility if english_fertility else 0.0


def meets_parity(ratio: float) -> bool:
    """Whether a parity ratio clears the target.

    Args:
        ratio: The parity ratio.

    Returns:
        True when it is at or below `PARITY_TARGET`.
    """
    return 0 < ratio <= PARITY_TARGET


def tokens_for_corpus(word_count: int, fertility: float) -> int:
    """Convert a corpus size in words into the tokens a tokenizer will actually produce.

    Args:
        word_count: Words in the corpus.
        fertility: Tokens per word for that tokenizer and language.

    Returns:
        Token count, rounded.
    """
    return round(word_count * fertility)


def training_cost(
    token_count: float,
    n_params: float,
    *,
    flops_per_token_per_param: int = FLOPS_PER_TOKEN_PER_PARAM,
    effective_flops: float = H100_EFFECTIVE_FLOPS,
    usd_per_gpu_hour: float = USD_PER_GPU_HOUR,
    usd_to_inr: float = USD_TO_INR,
) -> dict[str, float]:
    """Price a token count as compute.

    Args:
        token_count: Tokens to train on.
        n_params: Model parameter count.
        flops_per_token_per_param: FLOPs per token per parameter.
        effective_flops: Achieved FLOP/s per accelerator.
        usd_per_gpu_hour: Rental price.
        usd_to_inr: Conversion rate.

    Returns:
        A mapping of `tokens`, `flops`, `gpu_hours`, `usd` and `inr`.
    """
    flops = flops_per_token_per_param * n_params * token_count
    gpu_hours = flops / effective_flops / 3600 if effective_flops else 0.0
    usd = gpu_hours * usd_per_gpu_hour
    return {
        "tokens": token_count,
        "flops": flops,
        "gpu_hours": gpu_hours,
        "usd": usd,
        "inr": usd * usd_to_inr,
    }


def cost_delta(
    word_count: int,
    baseline_fertility: float,
    improved_fertility: float,
    n_params: float,
    **kwargs: Any,
) -> dict[str, float]:
    """Price the saving from a better tokenizer.

    This is the number that makes the tokenizer argument concrete: the same corpus, tokenized twice,
    differing by however many GPU-hours the fertility gap is worth.

    Args:
        word_count: Words in the corpus.
        baseline_fertility: Tokens per word before.
        improved_fertility: Tokens per word after.
        n_params: Model parameter count.
        **kwargs: Overrides forwarded to `training_cost`.

    Returns:
        The saving in each unit; negative values mean the change costs more.
    """
    before = training_cost(tokens_for_corpus(word_count, baseline_fertility), n_params, **kwargs)
    after = training_cost(tokens_for_corpus(word_count, improved_fertility), n_params, **kwargs)
    return {key: before[key] - after[key] for key in before}


def measure(
    encode: Callable[[str], list[int]],
    corpus: dict[str, str],
    *,
    tokenizer_ref: str,
    run_id: str,
) -> dict[str, Any]:
    """Measure fertility per language for one real tokenizer.

    Args:
        encode: Turns text into token ids — a real tokenizer, not a stand-in.
        corpus: Language code to text.
        tokenizer_ref: Identifier of the tokenizer measured, e.g. a HF repo id.
        run_id: Identifier of this measurement run (INV-4 requires both).

    Returns:
        Language code to a serialised `Value` carrying `provenance: "measured"`.

    Raises:
        ValueError: If `tokenizer_ref` or `run_id` is blank — an unattributable measurement is
            indistinguishable from an annotation, which is what INV-4 forbids.
    """
    if not tokenizer_ref.strip() or not run_id.strip():
        raise ValueError("a measured fertility needs both a tokenizer_ref and a run_id (INV-4)")

    results: dict[str, Any] = {}
    for language, text in corpus.items():
        words = len(text.split())
        value = Value(
            value=round(tokens_per_word(len(encode(text)), words), 4),
            unit="tokens/word",
            provenance="measured",
            source=f"{tokenizer_ref}@{run_id}",
        )
        results[language] = dataclasses.asdict(value)
    return results


def unmeasured(languages: list[str], reason: str) -> dict[str, Any]:
    """Report fertility as unknown, for languages no run has covered yet.

    The honest placeholder while task 2.2b is outstanding. Ground rule 8 rules out `estimated` for
    fertility, so an unmeasured language says so rather than carrying a plausible number.

    Args:
        languages: Language codes.
        reason: Why no measurement exists.

    Returns:
        Language code to a serialised unknown `Value`.
    """
    return {
        language: dataclasses.asdict(Value.unknown("tokens/word", source=reason))
        for language in languages
    }
