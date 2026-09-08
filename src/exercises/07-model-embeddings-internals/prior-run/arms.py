"""The four arms, so the output-side question is settled by measurement rather than argument.

    A  dense      nn.Embedding + tied head          (the control everyone actually ships)
    B  v1         Kronecker in, dense UNTIED head   (what the paper proposes)
    C  v2-tied    Kronecker in, head tied to E=KW   (zero extra parameters)
    D  v2-byte    Kronecker in, byte-factorised head (independent of V entirely)

The efficient trick that makes C and D practical is that the code is a *gather*, not a matmul.

    kappa_v @ W  =  (1/sqrt(L)) * sum_p W[p, b_p]

so the induced embedding for the whole vocabulary is an embedding lookup into `W` reshaped to
(d_p*256, d) at index `p*256 + b`, summed over positions. z-norm is affine, so it contributes a
rank-one correction using colsum(W). No V x D matrix is ever built.
"""

import numpy as np
import torch
import torch.nn.functional as F

DC = 256


def byte_index_table(token_bytes: list[bytes], d_p: int):
    """(V, d_p) flat indices into a (d_p*256, d) table, plus a validity mask and lengths."""
    v = len(token_bytes)
    idx = torch.zeros(v, d_p, dtype=torch.long)
    mask = torch.zeros(v, d_p)
    lengths = torch.zeros(v)
    for i, bs in enumerate(token_bytes):
        n = min(len(bs), d_p)
        lengths[i] = max(n, 1)
        for p in range(n):
            idx[i, p] = p * DC + bs[p]
            mask[i, p] = 1.0
    return idx, mask, lengths


class Kronecker(torch.nn.Module):
    """The v1 codec as a gather, plus the induced embedding matrix E for a tied head."""

    def __init__(self, token_bytes: list[bytes], d_model: int, d_p: int = 32, znorm: bool = True):
        super().__init__()
        self.d_p, self.D, self.znorm = d_p, d_p * DC, znorm
        idx, mask, lengths = byte_index_table(token_bytes, d_p)
        self.register_buffer("idx", idx)
        self.register_buffer("mask", mask)
        self.register_buffer("lengths", lengths)
        self.W = torch.nn.Parameter(torch.randn(self.D, d_model) / np.sqrt(self.D))

    def induced(self) -> torch.Tensor:
        """E = K W, computed by gather. Never materialises the V x D code matrix."""
        rows = F.embedding(self.idx, self.W)               # (V, d_p, d)
        raw = (rows * self.mask.unsqueeze(-1)).sum(1)      # (V, d)
        raw = raw / self.lengths.sqrt().unsqueeze(-1)      # the 1/sqrt(L)
        if not self.znorm:
            return raw
        # z-norm is affine in the code: (kappa - mu)/sigma, mu and sigma set by L alone.
        L = self.lengths.unsqueeze(-1)
        mu = (L / L.sqrt()) / self.D
        var = ((1 / L.sqrt() - mu) ** 2 * L + mu**2 * (self.D - L)) / self.D
        return (raw - mu * self.W.sum(0)) / var.sqrt()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.induced())


def params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
