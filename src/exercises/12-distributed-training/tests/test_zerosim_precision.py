"""The byte prices `precision.py` writes as integers are the element sizes of the real dtypes.

`precision.py` is torch-free so the formulas can be checked in the ordinary CI job; this is the
other half, run where torch is installed, so a recipe cannot claim bf16 is 2 bytes while the stages
store something else.
"""

import pytest

torch = pytest.importorskip("torch")

from zerosim.precision import FP32_BYTES, RECIPES, recipe  # noqa: E402


@pytest.mark.parametrize("mode", sorted(RECIPES))
def test_every_byte_price_is_the_real_element_size(mode: str) -> None:
    r = recipe(mode)
    assert torch.empty((), dtype=r.torch_dtype("param")).element_size() == r.param_bytes
    assert torch.empty((), dtype=r.torch_dtype("grad")).element_size() == r.grad_bytes
    assert torch.empty((), dtype=torch.float32).element_size() == FP32_BYTES


def test_mixed_precision_really_is_bfloat16() -> None:
    assert recipe("bf16-mixed").torch_dtype("param") is torch.bfloat16
    assert recipe("fp32").torch_dtype("param") is torch.float32


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown precision mode"):
        recipe("fp16")
