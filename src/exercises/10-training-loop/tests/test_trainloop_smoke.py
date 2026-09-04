"""The parts of exercise 10 that need no `torch`, tested where CI will actually run them.

`floats.py` is pure arithmetic and `accumulation.py`'s comparison is four numbers, so both belong
here rather than behind the `train` extra — which means the ordinary CI job runs real assertions
about this exercise rather than collecting a file with nothing in it.

The bit patterns are cross-checked against `torch`'s own casts in the gated twin. A decomposition
that agrees only with itself proves nothing.
"""

import pytest
from trainloop.accumulation import (
    EvenMicroBatchesError,
    combine_correctly,
    combine_wrongly,
    compare,
)
from trainloop.config import Config
from trainloop.floats import BF16, FP8_E4M3, FP32, decompose


def test_the_configured_micro_batches_can_actually_expose_the_bug() -> None:
    """The single most important property in this exercise's configuration.

    Averaging the averages is **exactly correct** when every micro-batch holds the same number of
    valid tokens. A demonstration built on even micro-batches reports a gap of zero and reads as a
    clean bill of health, which is how this bug survived inside every major training framework.
    """
    assert Config().micro_batches_are_uneven, (
        "micro_batch_tokens is even, so the accumulation comparison would report a gap of zero — "
        "which says the experiment was blind, not that the reduction is right"
    )


def test_the_comparison_refuses_an_even_configuration() -> None:
    """The twin. A gap of zero must be an error here, never a result."""
    with pytest.raises(EvenMicroBatchesError, match="even"):
        compare(Config(micro_batch_tokens=(4, 4, 4)))


def test_the_two_reductions_reproduce_the_worked_example() -> None:
    """4, 4 and 2 valid tokens at losses 2.0, 2.0 and 5.0: 2.6000 correctly, 3.0000 wrongly."""
    losses, tokens = (2.0, 2.0, 5.0), (4, 4, 2)
    assert combine_correctly(losses, tokens) == pytest.approx(2.6)
    assert combine_wrongly(losses) == pytest.approx(3.0)

    combination = compare()
    assert combination.absolute_gap == pytest.approx(0.4)
    assert combination.relative_gap == pytest.approx(0.4 / 2.6)


def test_the_two_reductions_agree_when_the_token_counts_are_equal() -> None:
    """The property that made the bug invisible, asserted rather than described."""
    losses, tokens = (2.0, 3.0, 7.0), (5, 5, 5)
    assert combine_correctly(losses, tokens) == pytest.approx(combine_wrongly(losses))


def test_zero_tokens_is_refused_rather_than_divided_by() -> None:
    """A batch with nothing valid in it has no mean, and returning one would invent a number."""
    with pytest.raises(ValueError, match="no valid tokens"):
        combine_correctly((1.0, 2.0), (0, 0))


@pytest.mark.parametrize(
    ("fmt", "bits", "hex_pattern"),
    [
        (FP32, "0 01111011 10011001100110011001101", "0x3DCCCCCD"),
        (BF16, "0 01111011 1001101", "0x3DCD"),
        (FP8_E4M3, "0 0011 101", "0x1D"),
    ],
)
def test_zero_point_one_decomposes_to_the_expected_pattern(fmt, bits, hex_pattern) -> None:
    """The bit patterns, built from arithmetic. Cross-checked against torch in the gated twin."""
    taken = decompose(0.1, fmt)
    assert taken.bits == bits
    assert taken.hex == hex_pattern


def test_fewer_mantissa_bits_means_more_error_holding_one_tenth() -> None:
    """One tenth repeats in binary, so no format holds it — the question is only how much it misses.

    Asserted as an ordering rather than as three literals: if the rounding ever regresses to
    truncation, the ordering survives and the literals would not have caught it anyway.
    """
    assert (
        abs(decompose(0.1, FP8_E4M3).relative_error)
        > abs(decompose(0.1, BF16).relative_error)
        > abs(decompose(0.1, FP32).relative_error)
        > 0
    ), "fewer mantissa bits must mean more error, and no format holds 0.1 exactly"


def test_e4m3_reaches_448_because_it_reserves_no_infinity() -> None:
    """The number that shows the format made a different trade, derived from the field widths."""
    assert FP8_E4M3.largest_normal == 448.0
    assert not FP8_E4M3.has_infinity
    assert FP32.has_infinity and BF16.has_infinity


def test_bf16_keeps_fp32s_range_and_spends_the_saving_on_precision() -> None:
    """Why bf16 replaced fp16 for training, as arithmetic rather than as a claim."""
    assert BF16.exponent_bits == FP32.exponent_bits
    assert BF16.smallest_normal == FP32.smallest_normal
    assert BF16.mantissa_bits < FP32.mantissa_bits
    assert BF16.decimal_digits < FP32.decimal_digits


def test_a_value_that_overflows_the_format_is_refused() -> None:
    """Silently returning infinity would hide exactly the property this module exists to show."""
    with pytest.raises(ValueError, match="overflows"):
        decompose(1e5, FP8_E4M3)
    with pytest.raises(ValueError, match="not finite"):
        decompose(float("inf"), FP32)
