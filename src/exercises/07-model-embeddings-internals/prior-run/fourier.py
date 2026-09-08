"""Problem #4 — a real Fourier alternative, and why it is the same idea as #5 rather than a rival.

v1 factorises (byte value) x (byte POSITION AS A ONE-HOT). A one-hot over positions can address
exactly `d_p` of them, which is the entire origin of the 32-byte cap and of the silent collisions
the exercise calls the sovereign risk: two tokens agreeing on their first 32 bytes get identical
codes forever.

Replace the one-hot position factor with a FOURIER position factor and the cap disappears:

    kappa_F(b) = (1/sqrt(L)) * sum_p  c_{b_p} (x) f(p)

    f(p) = [cos(w_1 p), sin(w_1 p), ..., cos(w_m p), sin(w_m p)],   w_j = base^(-2j/d_p)

`f(p)` is defined for every p, so there is no maximum length, no truncation, and no cap-induced
collision. The dimension D = 256 * d_p is unchanged, the code is still a Kronecker product of a
value factor and a position factor, and — the point — it is still a fixed code feeding one shared
projection, so everything built for the tied output head in `arms.py` applies unchanged.

That is why #4 and #5 are one architecture: Fourier positions make it LENGTH-free, the tied head
makes it VOCABULARY-free.

The cost is real and is measured below: one-hot positions are orthogonal, Fourier positions are not,
so recovery is harder. This file measures both sides of that trade rather than asserting either.
"""

import numpy as np

DC = 256


def fourier_positions(d_p: int, max_len: int, base: float = 10_000.0) -> np.ndarray:
    """(max_len, d_p) position features. RoPE-style geometric frequencies."""
    half = d_p // 2
    j = np.arange(half)
    omega = base ** (-2.0 * j / d_p)
    p = np.arange(max_len)[:, None]
    ang = p * omega[None, :]
    f = np.empty((max_len, d_p))
    f[:, 0::2] = np.cos(ang)
    f[:, 1::2] = np.sin(ang)
    return f / np.sqrt(half)  # unit norm per position


def code_fourier(bs: bytes, d_p: int, F: np.ndarray, znorm: bool = True) -> np.ndarray:
    """The Fourier-position code. Note: NO truncation — every byte contributes."""
    L = len(bs)
    grid = np.zeros((DC, d_p))
    for p in range(L):
        grid[bs[p]] += F[p]
    v = (grid / np.sqrt(max(L, 1))).reshape(-1)
    if znorm:
        v = (v - v.mean()) / (v.std() + 1e-12)
    return v


def code_onehot(bs: bytes, d_p: int, znorm: bool = True) -> np.ndarray:
    """v1's code, for comparison. Truncates at d_p."""
    L = min(len(bs), d_p)
    grid = np.zeros((d_p, DC))
    for p in range(L):
        grid[p, bs[p]] = 1.0
    v = (grid / np.sqrt(max(L, 1))).reshape(-1)
    if znorm:
        v = (v - v.mean()) / (v.std() + 1e-12)
    return v
