"""Small operations more than one family needs, written once so the families cannot disagree.

- `attend` is softmax attention written out step by step — scores, scale, mask, softmax,
  weighted sum — so a reader can see every stage, and so the lab's own reference is not a library
  call. `tests/test_attention_lab_core.py` checks it against PyTorch's
  `scaled_dot_product_attention`.
- `rope_angles` / `apply_rope` rotate **adjacent pairs** of dimensions, `(x1, x2), (x3, x4)`,
  which is how the RoPE paper (arXiv:2104.09864) writes the rotation, with the frequency of pair i
  set by `base ** (-2 i / d)` for i counted from zero. Many libraries rotate the two *halves* of
  the vector instead; that is a different pairing, and the lab uses the paper's. The equation
  numbers this follows are recorded, once verified, in `docs/ATTENTION_LAB.md`.
- `causal_mask` and `repeat_kv` are the two pieces of bookkeeping every attention variant shares.

Shapes throughout: `[batch, heads, tokens, head_dim]`.
"""

import math

import torch
from torch import Tensor


def float64_device(t: Tensor) -> torch.device:
    """Where float64 arithmetic for `t` can run: its own device, unless that is Apple's MPS.

    MPS has no float64, so there the precise constants are computed on the CPU and the caller moves
    the result. Every other device — CPU, CUDA, and the shape-only `meta` device — keeps its own.
    """
    return torch.device("cpu") if t.device.type == "mps" else t.device


def causal_mask(queries: int, keys: int, offset: int = 0, device=None) -> Tensor:
    """Boolean `[queries, keys]`: True where query i may read key j.

    `offset` is how many keys precede the first query — during decoding, the length of the cache.
    """
    rows = torch.arange(queries, device=device)[:, None] + offset
    cols = torch.arange(keys, device=device)[None, :]
    return cols <= rows


def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor | None = None,
    bias: Tensor | None = None,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Softmax attention, one visible step at a time.

    Args:
        q: `[batch, heads, queries, d]`.
        k: `[batch, heads, keys, d]`.
        v: `[batch, heads, keys, d_v]`.
        allowed: Boolean mask broadcastable to `[batch, heads, queries, keys]`; False cells are
            removed before the softmax.
        bias: Additive bias broadcastable to the score grid (ALiBi uses this).
        scale: Multiplier on the scores; defaults to `1/sqrt(d)`.

    Returns:
        The output `[batch, heads, queries, d_v]` and the weights `[batch, heads, queries, keys]`.
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # A row with no allowed key is all -inf and would be NaN; it reads nothing instead.
    weights = torch.nan_to_num(weights, nan=0.0)
    return weights @ v, weights


def repeat_kv(x: Tensor, groups: int) -> Tensor:
    """Share each key/value head with `groups` query heads: `[b, kv, t, d]` to `[b, kv*g, t, d]`."""
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


def rope_angles(positions: Tensor, dim: int, base: float) -> Tensor:
    """Angles `m * θ_i` for each position m and pair i: `[tokens, dim // 2]`."""
    if dim % 2:
        raise ValueError(f"RoPE rotates pairs, so the rotated width must be even, not {dim}")
    # Computed in float64 on the CPU and moved by the caller: Apple's MPS backend has no float64,
    # and the lab runs on a laptop GPU. The notebook found this; the CPU-only tests could not.
    where = float64_device(positions)
    positions = positions.detach().to(where)
    i = torch.arange(dim // 2, dtype=torch.float64, device=where)
    theta = base ** (-2.0 * i / dim)
    return positions.to(torch.float64)[:, None] * theta[None, :]


def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Rotate adjacent pairs of the last dimension of `x` by `angles` (`[tokens, dim // 2]`)."""
    cos = torch.cos(angles).to(device=x.device, dtype=x.dtype)
    sin = torch.sin(angles).to(device=x.device, dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def split_heads(x: Tensor, heads: int) -> Tensor:
    """`[b, t, heads*d] -> [b, heads, t, d]`."""
    b, t, width = x.shape
    return x.view(b, t, heads, width // heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """`[b, heads, t, d] -> [b, t, heads*d]`."""
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)
