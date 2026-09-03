"""What 0.1 looks like in fp32, bf16 and fp8 E4M3 — bit by bit, built rather than looked up.

A fixed-width slot has to serve numbers many orders of magnitude apart, and counting in binary
cannot span that. So the bits are divided into three jobs: one **sign** bit, a handful of
**exponent** bits fixing the magnitude, and the remaining **mantissa** bits selecting which value of
that magnitude. Scientific notation does the same thing — in `6.02 × 10²³` the exponent picks the
scale and the leading digits pick the value within it.

Every format here divides the same bits differently, and the trade never changes: **exponent bits
buy range, mantissa bits buy detail.** More of one is always less of the other.

**0.1 is a good number to ask about because no binary format can hold it.** One tenth is
`0.0001100110011…` repeating in binary, exactly as one third is `0.333…` repeating in decimal — so
every format below stores something slightly else, and the question is only *how much* else.

**This module builds the bit patterns from arithmetic rather than reading them out of the machine**,
which is the point of the exercise. `verify.py`'s tests then check each one against what torch's own
cast produces, because a decomposition that agrees with itself proves nothing. It needs no torch to
run — only to be checked.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    """One floating-point format, described by how it divides its bits.

    Attributes:
        name: What to call it.
        exponent_bits: Bits given to range.
        mantissa_bits: Bits given to precision.
        has_infinity: Whether the all-ones exponent is reserved for infinity and NaN. **fp8 E4M3
            does not reserve it**, which is why it reaches 448 rather than stopping much lower —
            the format spends that encoding on numbers instead.
    """

    name: str
    exponent_bits: int
    mantissa_bits: int
    has_infinity: bool = True

    @property
    def total_bits(self) -> int:
        """Sign plus exponent plus mantissa."""
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def bias(self) -> int:
        """What is subtracted from the stored exponent to get the real one."""
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def max_exponent_field(self) -> int:
        """The largest exponent field that still means a normal number."""
        top = (1 << self.exponent_bits) - 1
        return top - 1 if self.has_infinity else top

    @property
    def largest_normal(self) -> float:
        """The biggest finite value the format holds.

        Derived from the field widths, not typed. E4M3's 448 falls out of `has_infinity=False`,
        which is the whole reason that flag exists rather than being implied.
        """
        mantissa = 2.0 - 2.0 ** (-self.mantissa_bits)
        if not self.has_infinity:
            # The all-ones exponent with an all-ones mantissa is the NaN encoding, so the largest
            # finite value is one mantissa step below it.
            mantissa = 2.0 - 2.0 ** (1 - self.mantissa_bits)
        return mantissa * 2.0 ** (self.max_exponent_field - self.bias)

    @property
    def smallest_normal(self) -> float:
        """The smallest value held without losing precision to a subnormal representation."""
        return 2.0 ** (1 - self.bias)

    @property
    def decimal_digits(self) -> float:
        """Roughly how many decimal digits the mantissa resolves."""
        import math

        return (self.mantissa_bits + 1) * math.log10(2)


FP32 = Format("fp32", exponent_bits=8, mantissa_bits=23)
BF16 = Format("bf16", exponent_bits=8, mantissa_bits=7)
FP8_E4M3 = Format("fp8 E4M3", exponent_bits=4, mantissa_bits=3, has_infinity=False)

FORMATS = (FP32, BF16, FP8_E4M3)


@dataclass(frozen=True)
class Decomposition:
    """One value, held in one format, taken apart.

    Attributes:
        fmt: The format it was stored in.
        asked: The value we asked the format to hold.
        sign: 0 for positive, 1 for negative.
        exponent_field: The exponent bits as an integer, before the bias is removed.
        mantissa_field: The mantissa bits as an integer.
        stored: What the bit pattern actually represents — **not** what was asked for.
    """

    fmt: Format
    asked: float
    sign: int
    exponent_field: int
    mantissa_field: int
    stored: float

    @property
    def bits(self) -> str:
        """The whole pattern, grouped `sign exponent mantissa`."""
        exponent = format(self.exponent_field, f"0{self.fmt.exponent_bits}b")
        mantissa = format(self.mantissa_field, f"0{self.fmt.mantissa_bits}b")
        return f"{self.sign} {exponent} {mantissa}"

    @property
    def hex(self) -> str:
        """The pattern as one hexadecimal integer."""
        packed = (
            (self.sign << (self.fmt.exponent_bits + self.fmt.mantissa_bits))
            | (self.exponent_field << self.fmt.mantissa_bits)
            | self.mantissa_field
        )
        return f"0x{packed:0{(self.fmt.total_bits + 3) // 4}X}"

    @property
    def unbiased_exponent(self) -> int:
        """The real exponent: the field minus the bias."""
        return self.exponent_field - self.fmt.bias

    @property
    def error(self) -> float:
        """How far the stored value is from the one asked for."""
        return self.stored - self.asked

    @property
    def relative_error(self) -> float:
        """The same, as a fraction of the value asked for."""
        return self.error / self.asked if self.asked else 0.0

    def working(self) -> str:
        """The construction, written out — which is what the requirements ask to see."""
        significand = 1 + self.mantissa_field / (1 << self.fmt.mantissa_bits)
        return (
            f"    bits            {self.bits}   ({self.hex})\n"
            f"    exponent field  {self.exponent_field} - bias {self.fmt.bias} = "
            f"{self.unbiased_exponent}\n"
            f"    mantissa field  {self.mantissa_field}/{1 << self.fmt.mantissa_bits}, so the "
            f"significand is 1 + {self.mantissa_field}/{1 << self.fmt.mantissa_bits} = "
            f"{significand!r}\n"
            f"    value           {significand!r} x 2^{self.unbiased_exponent} = {self.stored!r}\n"
            f"    error           {self.error:+.10g}  ({self.relative_error:+.4%})"
        )


def _fp32_fields(value: float) -> tuple[int, int, int]:
    """Sign, exponent field and mantissa field of `value` as an IEEE binary32.

    `struct` is the one shortcut taken here, and only to get at the machine's own fp32 bits — every
    other format below is derived from these by rounding, which is the part worth doing by hand.
    """
    (packed,) = struct.unpack(">I", struct.pack(">f", value))
    return packed >> 31, (packed >> 23) & 0xFF, packed & 0x7F_FFFF


def _round_to_nearest_even(mantissa: int, drop: int) -> tuple[int, bool]:
    """Drop the lowest `drop` bits of `mantissa`, rounding half to even.

    Round-to-nearest-even is what every format here uses, and it matters: truncating instead would
    bias every conversion downwards, and the bias would accumulate over a training run rather than
    cancelling.

    Returns:
        `(rounded mantissa, whether it overflowed into the next exponent)`.
    """
    if drop <= 0:
        return mantissa, False

    kept = mantissa >> drop
    remainder = mantissa & ((1 << drop) - 1)
    half = 1 << (drop - 1)

    if remainder > half or (remainder == half and kept & 1):
        kept += 1

    overflow = kept >= (1 << (mantissa.bit_length() - drop)) if mantissa else False
    return kept, overflow


def decompose(value: float, fmt: Format) -> Decomposition:
    """Take `value` apart as `fmt` would hold it, rounding to nearest even.

    Args:
        value: The number to store. Finite, and within the format's range.
        fmt: Which format to store it in.

    Returns:
        A `Decomposition` carrying the bits and what they actually mean.

    Raises:
        ValueError: When the value is not finite, or overflows the format — silently returning
            infinity would hide exactly the property this module exists to show.
    """
    import math

    if not math.isfinite(value):
        raise ValueError(f"{value} is not finite; this module is about how finite values are held")
    if abs(value) > fmt.largest_normal:
        raise ValueError(
            f"{value} overflows {fmt.name}, whose largest finite value is {fmt.largest_normal}"
        )

    sign, fp32_exponent, fp32_mantissa = _fp32_fields(value)

    if fmt is FP32 or fmt.mantissa_bits >= 23:
        exponent_field, mantissa_field = fp32_exponent, fp32_mantissa
    else:
        drop = 23 - fmt.mantissa_bits
        mantissa_field, carried = _round_to_nearest_even(fp32_mantissa, drop)
        exponent_field = fp32_exponent - 127 + fmt.bias
        if mantissa_field >= (1 << fmt.mantissa_bits):
            mantissa_field = 0
            exponent_field += 1
        if carried:  # pragma: no cover - defensive; the branch above already normalises
            exponent_field += 1

    significand = 1 + mantissa_field / (1 << fmt.mantissa_bits)
    stored = (-1) ** sign * significand * 2.0 ** (exponent_field - fmt.bias)
    return Decomposition(fmt, value, sign, exponent_field, mantissa_field, stored)


def report(value: float = 0.1) -> str:
    """The three decompositions and the recommendation, as the requirements ask for them."""
    lines = [
        f"  {value!r} cannot be held exactly by any binary format: one tenth is",
        "  0.0001100110011... repeating in binary, as one third is 0.333... in decimal.",
        "  So each format below stores something else, and the question is how much else.",
        "",
    ]
    for fmt in FORMATS:
        taken = decompose(value, fmt)
        lines += [
            f"  {fmt.name}  —  1 sign + {fmt.exponent_bits} exponent + "
            f"{fmt.mantissa_bits} mantissa = {fmt.total_bits} bits, bias {fmt.bias}",
            taken.working(),
            "",
        ]
    lines += [
        "  Which one would I train in? bf16, and the reason is one column of the table:",
        "",
        f"  {'format':<10} {'exponent':>9} {'mantissa':>9} {'smallest normal':>18} "
        f"{'decimal digits':>15}",
    ]
    for fmt in FORMATS:
        lines.append(
            f"  {fmt.name:<10} {fmt.exponent_bits:>9} {fmt.mantissa_bits:>9} "
            f"{fmt.smallest_normal:>18.3g} {fmt.decimal_digits:>15.1f}"
        )
    lines += [
        "",
        "  bf16 keeps fp32's EIGHT exponent bits and spends the savings entirely out of the",
        "  mantissa. That is why it replaced fp16 for training: it has fp32's range, so a",
        "  gradient that underflowed in fp16 does not underflow here, and no loss-scaling",
        "  machinery is needed to keep small values alive. It pays in precision, and gradient",
        "  descent tolerates imprecision far better than it tolerates zeros.",
        "",
        "  fp8 E4M3 is a different decision, not a further step along the same one. Four",
        "  exponent bits give it a much narrower range, and it does not reserve the all-ones",
        f"  exponent for infinity — which is why it reaches {FP8_E4M3.largest_normal:g} rather",
        "  than stopping lower. It is a format for weights and activations under a scaling",
        "  scheme that keeps values inside that range, not a drop-in replacement for bf16.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
