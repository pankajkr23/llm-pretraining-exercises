"""What one weight costs to train, by precision recipe — the 16 bytes, itemised.

Two recipes, kept as data rather than as branches scattered through the stages:

| recipe | weight | gradient | fp32 master copy | Adam m | Adam v | total |
| --- | --- | --- | --- | --- | --- | --- |
| `bf16-mixed` | 2 (bf16) | 2 (bf16) | 4 | 4 | 4 | 16 |
| `fp32` | 4 | 4 | — (the weight *is* the master) | 4 | 4 | 16 |

**Both total 16, and that is a coincidence of the arithmetic, not a law.** Mixed precision does not
shrink the training state; it shrinks activations and buys speed. What differs is *which* of the 16
bytes ZeRO can shard at each stage, so the per-stage formulas in `formulas.py` differ between the
two recipes even though the totals agree.

**This module needs no `torch`**, so the formulas and the memory ladder can be checked in the
ordinary CI job. The byte sizes are therefore written here as integers — and
`tests/test_zerosim_precision.py` asserts each one against the element size of the real torch dtype
it names, so a recipe cannot claim bf16 is 2 bytes while the stages store fp32.
"""

from dataclasses import dataclass

FP32_BYTES = 4
"""Bytes per fp32 element. Asserted against `torch.float32` in the precision tests."""


@dataclass(frozen=True)
class Recipe:
    """The dtypes one precision mode stores, and therefore what each category costs per weight.

    Attributes:
        mode: `"bf16-mixed"` or `"fp32"`.
        param_dtype: Name of the torch dtype the weights are stored and computed in.
        param_bytes: Its element size.
        grad_dtype: Name of the torch dtype gradients are produced and communicated in.
        grad_bytes: Its element size.
        has_master: Whether a separate fp32 copy of the weights exists. In `fp32` it does not: the
            optimiser updates the weights themselves.
    """

    mode: str
    param_dtype: str
    param_bytes: int
    grad_dtype: str
    grad_bytes: int
    has_master: bool

    @property
    def master_bytes(self) -> int:
        """Bytes per weight for the fp32 master copy; 0 when the weights are themselves fp32."""
        return FP32_BYTES if self.has_master else 0

    @property
    def moment_bytes(self) -> int:
        """Bytes per weight for ONE Adam moment. There are two, `m` and `v`, always fp32."""
        return FP32_BYTES

    @property
    def optimizer_bytes(self) -> int:
        """Bytes per weight ZeRO-1 shards: the master copy (if any) plus both moments."""
        return self.master_bytes + 2 * self.moment_bytes

    @property
    def total_bytes(self) -> int:
        """All of it: what plain data parallelism replicates on every device."""
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes

    def torch_dtype(self, which: str):
        """The real `torch.dtype` for `"param"` or `"grad"`. The one method here needing torch."""
        import torch

        return getattr(torch, self.param_dtype if which == "param" else self.grad_dtype)


RECIPES = {
    "bf16-mixed": Recipe("bf16-mixed", "bfloat16", 2, "bfloat16", 2, has_master=True),
    "fp32": Recipe("fp32", "float32", FP32_BYTES, "float32", FP32_BYTES, has_master=False),
}


def recipe(mode: str) -> Recipe:
    """The recipe for `mode`.

    Raises:
        ValueError: For any other mode, naming the two that exist.
    """
    if mode not in RECIPES:
        raise ValueError(f"unknown precision mode {mode!r}; expected one of {sorted(RECIPES)}")
    return RECIPES[mode]
