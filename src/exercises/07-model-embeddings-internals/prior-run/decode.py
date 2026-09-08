"""Better decoders than the matched filter.

The matched filter ignores interference: `h` is a SUM of `L` rows of `W`, so correlating against
one row lets the other `L-1` leak in. Coordinate descent cancels that leak iteratively, which is
why it should beat the filter exactly where the filter is weakest — long tokens.
"""

import numpy as np

DC = 256


def raw_code(bs: bytes, d_p: int):
    """The pre-z-norm code and the effective length."""
    L = min(len(bs), d_p)
    g = np.zeros((d_p, DC))
    for p in range(L):
        g[p, bs[p]] = 1.0
    if L:
        g /= np.sqrt(L)
    return g.reshape(-1), L


def encode(byte_strings, W, d_p, znorm=True):
    """Encode a list of byte strings to hidden vectors, returning the lengths too."""
    out = np.zeros((len(byte_strings), W.shape[1]))
    lengths = []
    for i, bs in enumerate(byte_strings):
        v, L = raw_code(bs, d_p)
        lengths.append(L)
        if znorm:
            v = (v - v.mean()) / (v.std() + 1e-12)
        out[i] = v @ W
    return out, lengths


def _unznorm(h, W, L, D):
    """Undo the z-norm. It is affine in the sparse vector, with mu and sigma set by L alone."""
    mu = (L / np.sqrt(L)) / D
    sig = np.sqrt(((1 / np.sqrt(L) - mu) ** 2 * L + mu**2 * (D - L)) / D)
    return sig * h + mu * W.sum(axis=0)


def decode_cd(h, W, d_p, lengths, znorm=True, iters=12):
    """Coordinate descent — iterative interference cancellation.

    `iters=0` gives the plain matched filter, so the two are comparable on one code path.
    """
    D = W.shape[0]
    Wg = W.reshape(d_p, DC, -1)
    out = []
    for i in range(h.shape[0]):
        L = lengths[i]
        if L == 0:
            out.append(b"")
            continue
        hi = _unznorm(h[i], W, L, D) if znorm else h[i]
        hi = hi * np.sqrt(L)  # undo 1/sqrt(L): now hi is a plain sum of L rows

        b = np.empty(L, dtype=int)
        for p in range(L):
            b[p] = int(Wg[p].dot(hi).argmax())

        if iters:
            recon = np.zeros_like(hi)
            for p in range(L):
                recon += Wg[p, b[p]]
            for _ in range(iters):
                changed = False
                for p in range(L):
                    resid = hi - recon + Wg[p, b[p]]
                    nb = int(Wg[p].dot(resid).argmax())
                    if nb != b[p]:
                        recon = recon - Wg[p, b[p]] + Wg[p, nb]
                        b[p] = nb
                        changed = True
                if not changed:
                    break
        out.append(bytes(b.tolist()))
    return out
