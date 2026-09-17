"""The delta-rule family: a fixed-size memory that *overwrites* instead of only adding.

**The problem.** Linear attention (`attention.lab.linear`) only ever adds `v kᵀ` to its state.
Two writes under similar keys pile up, and nothing is ever removed, so the memory blurs as the
context grows.

**What changes in code.** Before writing, the layer reads what the memory currently returns for
the new key, `v_old = S k`, and writes only the *difference*:
`S ← S + β (v − S k) kᵀ = S (I − β k kᵀ) + β v kᵀ`. That is one step of gradient descent on
`½‖S k − v‖²` — the Widrow-Hoff delta rule — with a learned step size `β`. With `β = 1` and a unit
key, the key now retrieves exactly `v`, and keys orthogonal to it are untouched.

**The trade-off.** The update is no longer a plain sum, so the cumulative-sum trick of linear
attention does not apply. Training needs a different parallel form (the WY / UT representation of
a product of Householder-like matrices), which is more work per chunk. Each later variant adds a
way to *forget* on top of *overwrite*:

- `delta_rule` — Schlag, Irie & Schmidhuber, arXiv:2102.11174v3, Eq 20–25: `v̄ = W φ(k)`,
  `β = σ(W_β x)`, `W ← W + β (v − v̄) ⊗ φ(k)`, `y = W φ(q)`, with **sum normalisation** (Eq 29) and
  a choice of feature map: `elu + 1` (Eq 30) or DPFP-ν (Eq 33–37). That paper gives only the
  recurrence; the `parallel` mode here uses DeltaNet's chunkwise form below, which applies because
  the update is the same `W (I − β φ φᵀ) + β v φᵀ` with `φ` the normalised feature.
- `deltanet_parallel` — Yang et al., arXiv:2406.06484v6. Keys and queries are
  `L2Norm(SiLU(·))`, and training uses the chunkwise WY/UT form (Eq 3, 5–12):
  `T = (I + strictLower(diag(β) K Kᵀ))⁻¹ diag(β)`, `W = T K`, `U = T V`,
  `S ← S + (U − W Sᵀ)ᵀ K`, `O = Q Sᵀ + (Q Kᵀ ⊙ M)(U − W Sᵀ)` (rows are tokens). A short causal
  convolution (kernel stated in App A.1) runs before the activation; its last `kernel − 1`
  inputs are part of the decode state.
- `gated_deltanet` — Yang, Kautz & Hatamizadeh, arXiv:2412.06464v3, Eq 10:
  `S ← S (α (I − β k kᵀ)) + β v kᵀ`, with a data-dependent decay `α ∈ (0, 1)`. The paper says only
  "We use Mamba2’s parameterization for α"; the formula implemented is
  `α = exp(−exp(A_log) · softplus(W_α x + dt_bias))`, written from that formula alone (the
  official code is under a non-commercial licence and was not copied). The chunkwise form follows
  §3.3: `Ũ = [I + strictLower(diag(β)(Γ ⊙ K Kᵀ))]⁻¹ diag(β) V` with `Γ_ri = γ_r / γ_i`, and the
  end-of-chunk state `γ_C S + (Ũ − ←W Sᵀ)ᵀ →K` exactly as printed. **One deviation, stated:** the
  printed output line uses the plain causal mask `M` and `←W` (weights `γ_i`); expanding
  `S^r = γ_r S P^r + G^r` gives weight `γ_r / γ_i` on the `Ũ` term and `γ_r` on the `W` term, and
  that is what `delta_rule_chunkwise` computes. The chunkwise-equals-recurrent test is the arbiter.
- `kda` — Kimi Delta Attention, arXiv:2510.26692v2, Eq 1: the decay becomes **channel-wise**,
  `S ← (I − β k kᵀ) Diag(α) S + β k vᵀ`, `o = Sᵀ q`, with the chunkwise form of Eq 2–9
  (`M = (I + StrictTril(Diag(β)(Γ⊙K)(K/Γ)ᵀ))⁻¹ Diag(β)`, `W = M(Γ⊙K)`, `U = M V`). The pairwise
  decay ratios are computed as `exp(log γ_r − log γ_i)` so nothing is divided by a small number.
  The decay gate is the one in the fla library (MIT), `g = −exp(A_log) · softplus(z + dt_bias)`,
  `α = exp(g)`, with `z` a low-rank projection of rank `head_dim` (§4). Two Kimi K3 changes
  (arXiv:2607.24653v2 §2.1.1) are parameters: `decay_floor < 0` switches to K3's bounded form
  `g = g_min · sigmoid(exp(A) · z)` (its Eq 5, `g_min = −5`, `A` initialised to 0), and
  `output_gate="full-rank"` replaces the low-rank output gate with a full-rank projection (its
  Eq 6). The pseudo-code in the Kimi Linear appendix also scales `q` by `d_k^(−1/2)`; Eq 1 and
  Eq 10 do not, and neither does this module.
- `gated_deltanet2` — Hatamizadeh, Choi & Kautz, arXiv:2605.22791v1, Eq 8–12: the scalar `β` is
  split into a channel-wise **erase** gate `b ∈ [0,1]^{d_k}` and **write** gate `w ∈ [0,1]^{d_v}`,
  `S ← (I − k (b⊙k)ᵀ) D S + k (w⊙v)ᵀ`, `o = Sᵀ q`, with `D = Diag(exp(g))`,
  `g = −exp(a) ⊙ softplus(W_f x + δ)` (`a` per head, `δ` per key channel, App C.1). The chunkwise
  form is §3.3 Eq 18–25, implemented as printed. With `b = w = β·1` it is KDA's rule.

**Orientation.** Linear attention, the delta rule, DeltaNet and Gated DeltaNet keep
`S ∈ R^{d_v × d_k}` and read `o = S q`. KDA and Gated DeltaNet-2 keep `S ∈ R^{d_k × d_v}` and read
`o = Sᵀ q`. Each module here keeps its own paper's orientation; the two are transposes, which is how
the tests compare them.

Every module offers both forms through `mode`: a full pass uses the chunkwise (or parallel) form,
and decoding one token at a time uses the recurrence, so the generic stepwise test is also a
chunkwise-equals-recurrent test.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812 - the conventional alias

from attention.lab import hparams, sources
from attention.lab.base import Mixer, MixerSpec
from attention.lab.linear import elu_plus_one
from attention.lab.ops import merge_heads, split_heads
from attention.lab.registry import register

# --- kernels: S ∈ R^{d_v × d_k}, o = S q (delta rule, DeltaNet, Gated DeltaNet) -----------------


def delta_rule_recurrent(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, s: Tensor, log_alpha: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """The (gated) delta rule one token at a time: `S ← S α (I − β k kᵀ) + β v kᵀ`, `o = S q`.

    arXiv:2406.06484v6 Eq 3 when `log_alpha` is None; arXiv:2412.06464v3 Eq 10 otherwise.

    Args:
        q: `[batch, heads, tokens, d_k]`.
        k: `[batch, heads, tokens, d_k]`.
        v: `[batch, heads, tokens, d_v]`.
        beta: `[batch, heads, tokens]` writing strength.
        s: `[batch, heads, d_v, d_k]` state before the first token.
        log_alpha: `[batch, heads, tokens]` log of the scalar decay, or None for no decay.

    Returns:
        Outputs `[batch, heads, tokens, d_v]` and the state after the last token.
    """
    outs = []
    for t in range(q.shape[2]):
        if log_alpha is not None:
            s = torch.exp(log_alpha[:, :, t, None, None]) * s
        kt = k[:, :, t]
        read = torch.einsum("bhvk,bhk->bhv", s, kt)
        s = s + beta[:, :, t, None, None] * (v[:, :, t] - read)[..., :, None] * kt[..., None, :]
        outs.append(torch.einsum("bhvk,bhk->bhv", s, q[:, :, t]))
    return torch.stack(outs, dim=2), s


def _strict_lower(size: int, like: Tensor) -> Tensor:
    return torch.ones(size, size, dtype=like.dtype, device=like.device).tril(-1)


def delta_rule_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    s: Tensor,
    chunk: int,
    log_alpha: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The (gated) delta rule, chunk by chunk, through the UT transform.

    Without decay this is arXiv:2406.06484v6 Eq 8–12. With decay it is arXiv:2412.06464v3 §3.3:
    within a chunk `γ_r = Π_{i≤r} α_i` and `Γ_ri = γ_r / γ_i`;
    `Ũ = [I + strictLower(diag(β)(Γ ⊙ K Kᵀ))]⁻¹ diag(β) V`, and `W` is the *ungated*
    `[I + strictLower(diag(β) K Kᵀ)]⁻¹ diag(β) K` (the product of `(I − β k kᵀ)` does not see
    `α`, because a scalar commutes out of it). Then, for row `r` of the chunk,

        o_r = γ_r (S q_r − Σ_{i≤r} (k_iᵀ q_r) S w_i) + Σ_{i≤r} (γ_r/γ_i)(k_iᵀ q_r) ũ_i
        S'  = γ_C (S − Σ_i S w_i k_iᵀ) + Σ_i (γ_C/γ_i) ũ_i k_iᵀ.

    The state line is the paper's; the output line is re-derived (see the module docstring).

    Args:
        q: `[batch, heads, tokens, d_k]`.
        k: `[batch, heads, tokens, d_k]`.
        v: `[batch, heads, tokens, d_v]`.
        beta: `[batch, heads, tokens]`.
        s: `[batch, heads, d_v, d_k]`.
        chunk: Tokens per chunk `C`; the last chunk may be shorter.
        log_alpha: `[batch, heads, tokens]` or None.

    Returns:
        Outputs `[batch, heads, tokens, d_v]` and the state after the last token.
    """
    outs = []
    for start in range(0, q.shape[2], chunk):
        qc, kc, vc, bc = (t[:, :, start : start + chunk] for t in (q, k, v, beta))
        size = qc.shape[2]
        lower = _strict_lower(size, qc)
        causal = lower + torch.eye(size, dtype=qc.dtype, device=qc.device)
        if log_alpha is None:
            gam = torch.ones_like(bc)
            ratio = torch.ones_like(causal)
        else:
            cum = log_alpha[:, :, start : start + chunk].cumsum(dim=-1)
            gam = torch.exp(cum)
            diff = cum[..., :, None] - cum[..., None, :]
            ratio = torch.exp(torch.where(causal.bool(), diff, torch.zeros_like(diff)))
        kk = kc @ kc.transpose(-2, -1)
        eye = torch.eye(size, dtype=qc.dtype, device=qc.device)
        u_lhs = eye + bc[..., :, None] * ratio * kk * lower
        u = torch.linalg.solve_triangular(u_lhs, bc[..., None] * vc, upper=False)
        w_lhs = eye + bc[..., :, None] * kk * lower
        w = torch.linalg.solve_triangular(w_lhs, bc[..., None] * kc, upper=False)
        s_t = s.transpose(-2, -1)
        sq, sw = qc @ s_t, w @ s_t
        scores = (qc @ kc.transpose(-2, -1)) * causal
        outs.append(gam[..., None] * (sq - scores @ sw) + (scores * ratio) @ u)
        tail = ratio[..., -1, :, None] * kc
        s = gam[..., -1, None, None] * (s - sw.transpose(-2, -1) @ kc) + u.transpose(-2, -1) @ tail
    return torch.cat(outs, dim=2), s


# --- kernels: S ∈ R^{d_k × d_v}, o = Sᵀ q (KDA, Gated DeltaNet-2) --------------------------------


def kda_recurrent(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, s: Tensor, log_alpha: Tensor
) -> tuple[Tensor, Tensor]:
    """arXiv:2510.26692v2 Eq 1: `S ← (I − β k kᵀ) Diag(α) S + β k vᵀ`, `o = Sᵀ q`.

    Args:
        q: `[batch, heads, tokens, d_k]`.
        k: `[batch, heads, tokens, d_k]`.
        v: `[batch, heads, tokens, d_v]`.
        beta: `[batch, heads, tokens]`.
        s: `[batch, heads, d_k, d_v]`.
        log_alpha: `[batch, heads, tokens, d_k]` channel-wise log decay.

    Returns:
        Outputs `[batch, heads, tokens, d_v]` and the state after the last token.
    """
    outs = []
    for t in range(q.shape[2]):
        s = torch.exp(log_alpha[:, :, t, :, None]) * s
        kt = k[:, :, t]
        read = torch.einsum("bhkv,bhk->bhv", s, kt)
        s = s + beta[:, :, t, None, None] * kt[..., :, None] * (v[:, :, t] - read)[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", s, q[:, :, t]))
    return torch.stack(outs, dim=2), s


def _pairwise_decay(cum: Tensor, causal: Tensor) -> Tensor:
    """`exp(cum_r − cum_i)` per channel for `i ≤ r`, zero above the diagonal: `[..., C, C, d_k]`."""
    diff = cum[..., :, None, :] - cum[..., None, :, :]
    keep = causal.bool()[..., None]
    return torch.where(keep, torch.exp(torch.where(keep, diff, torch.zeros_like(diff))), 0.0)


def kda_chunkwise(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, s: Tensor, log_alpha: Tensor, chunk: int
) -> tuple[Tensor, Tensor]:
    """arXiv:2510.26692v2 Eq 6–9, chunk by chunk.

    With `γ_r` the cumulative decay from the chunk start (`Γ^{1→C}` has rows `γ_r`):
    `G_ri = (γ_r ⊙ k_r)ᵀ (k_i / γ_i)`, `M = (I + StrictTril(Diag(β) G))⁻¹ Diag(β)`,
    `W = M (Γ ⊙ K)`, `U = M V`, `S' = Diag(γ_C) S + (Γ^{i→C} ⊙ K)ᵀ (U − W S)`,
    `O = (Γ ⊙ Q) S + Tril((Γ ⊙ Q)(K/Γ)ᵀ)(U − W S)`. The ratios `γ_r / γ_i` are formed in log
    space.
    """
    outs = []
    for start in range(0, q.shape[2], chunk):
        qc, kc, vc, bc = (t[:, :, start : start + chunk] for t in (q, k, v, beta))
        size = qc.shape[2]
        lower = _strict_lower(size, qc)
        causal = lower + torch.eye(size, dtype=qc.dtype, device=qc.device)
        cum = log_alpha[:, :, start : start + chunk].cumsum(dim=2)
        gam = torch.exp(cum)
        pair = _pairwise_decay(cum, causal)
        g = torch.einsum("bhrc,bhic,bhric->bhri", kc, kc, pair)
        lhs = torch.eye(size, dtype=qc.dtype, device=qc.device) + bc[..., :, None] * g * lower
        rhs_w = bc[..., None] * (gam * kc)
        w = torch.linalg.solve_triangular(lhs, rhs_w, upper=False)
        u = torch.linalg.solve_triangular(lhs, bc[..., None] * vc, upper=False)
        pseudo = u - w @ s
        scores = torch.einsum("bhrc,bhic,bhric->bhri", qc, kc, pair)
        outs.append((gam * qc) @ s + scores @ pseudo)
        tail = pair[:, :, -1] * kc
        s = gam[:, :, -1, :, None] * s + tail.transpose(-2, -1) @ pseudo
    return torch.cat(outs, dim=2), s


def gdn2_recurrent(
    q: Tensor, k: Tensor, v: Tensor, b: Tensor, w: Tensor, s: Tensor, log_alpha: Tensor
) -> tuple[Tensor, Tensor]:
    """arXiv:2605.22791v1 Eq 8–9: `S̄ = D S`, `r = S̄ᵀ(b⊙k)`, `S = S̄ + k (w⊙v − r)ᵀ`, `o = Sᵀ q`.

    Args:
        q: `[batch, heads, tokens, d_k]`.
        k: `[batch, heads, tokens, d_k]`.
        v: `[batch, heads, tokens, d_v]`.
        b: `[batch, heads, tokens, d_k]` erase gate.
        w: `[batch, heads, tokens, d_v]` write gate.
        s: `[batch, heads, d_k, d_v]`.
        log_alpha: `[batch, heads, tokens, d_k]`, the log of the diagonal of `D`.

    Returns:
        Outputs `[batch, heads, tokens, d_v]` and the state after the last token.
    """
    outs = []
    for t in range(q.shape[2]):
        decayed = torch.exp(log_alpha[:, :, t, :, None]) * s
        kt = k[:, :, t]
        read = torch.einsum("bhkv,bhk->bhv", decayed, b[:, :, t] * kt)
        s = decayed + kt[..., :, None] * (w[:, :, t] * v[:, :, t] - read)[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", s, q[:, :, t]))
    return torch.stack(outs, dim=2), s


def gdn2_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    s: Tensor,
    log_alpha: Tensor,
    chunk: int,
) -> tuple[Tensor, Tensor]:
    """arXiv:2605.22791v1 §3.3 Eq 18–25, chunk by chunk.

    `T = tril(Ē K̄ᵀ, −1)` with `Ē = γ ⊙ (B ⊙ K)`, `K̄ = γ⁻¹ ⊙ K`; `A = (I + T)⁻¹`; `Y = A Ē`,
    `U = A Z` with `Z = W ⊙ V`; `S' = Diag(γ_C) S + K_tailᵀ (U − Y S)`;
    `O = Q_γ S + A_qk (U − Y S)` with `(A_qk)_rs = 1[r≥s] q_rᵀ Diag(γ_r/γ_s) k_s`. The products
    `Ē K̄ᵀ` and `A_qk` are formed from `γ_r/γ_s` in log space rather than from `γ⁻¹`.
    """
    outs = []
    for start in range(0, q.shape[2], chunk):
        qc, kc, vc, bc, wc = (t[:, :, start : start + chunk] for t in (q, k, v, b, w))
        size = qc.shape[2]
        lower = _strict_lower(size, qc)
        causal = lower + torch.eye(size, dtype=qc.dtype, device=qc.device)
        cum = log_alpha[:, :, start : start + chunk].cumsum(dim=2)
        gam = torch.exp(cum)
        pair = _pairwise_decay(cum, causal)
        tri = torch.einsum("bhrc,bhsc,bhrsc->bhrs", bc * kc, kc, pair) * lower
        lhs = torch.eye(size, dtype=qc.dtype, device=qc.device) + tri
        y = torch.linalg.solve_triangular(lhs, gam * (bc * kc), upper=False)
        u = torch.linalg.solve_triangular(lhs, wc * vc, upper=False)
        residual = u - y @ s
        a_qk = torch.einsum("bhrc,bhsc,bhrsc->bhrs", qc, kc, pair)
        outs.append((gam * qc) @ s + a_qk @ residual)
        k_tail = pair[:, :, -1] * kc
        s = gam[:, :, -1, :, None] * s + k_tail.transpose(-2, -1) @ residual
    return torch.cat(outs, dim=2), s


# --- feature maps (arXiv:2102.11174v3 §5) ---------------------------------------------------------


def dpfp(x: Tensor, nu: int) -> Tensor:
    """DPFP-ν (Eq 37): `φ_{iν}(k) = r([k; −k])_i · r([k; −k])_{i+ν}` for every shift up to `ν`.

    The output has `2 · d_key · ν` components; indices wrap around, as the paper's appendix does.
    """
    doubled = F.relu(torch.cat([x, -x], dim=-1))
    return torch.cat([doubled * doubled.roll(-shift, dims=-1) for shift in range(1, nu + 1)], -1)


def sum_normalise(phi: Tensor, floor: float = 1e-6) -> Tensor:
    """Eq 29: divide a feature vector by the sum of its (non-negative) components.

    `floor` only guards the all-zero vector DPFP can produce; it is ours, not the paper's.
    """
    return phi / phi.sum(dim=-1, keepdim=True).clamp_min(floor)


# --- a short causal convolution with a decode cache -----------------------------------------------


class ShortConv(nn.Module):
    """Depthwise causal 1-D convolution over tokens, carrying its last `kernel − 1` inputs.

    Starting from a zero cache is the same as zero left-padding, so a full pass and a token-by-token
    pass agree exactly.
    """

    def __init__(self, channels: int, kernel: int) -> None:
        """One filter of width `kernel` per channel."""
        super().__init__()
        self.channels, self.kernel = channels, kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)

    def empty(self, batch: int, device: torch.device | None, dtype: torch.dtype | None) -> Tensor:
        """The cache before any token: `[batch, channels, kernel − 1]` zeros."""
        return torch.zeros(batch, self.channels, self.kernel - 1, device=device, dtype=dtype)

    def forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        """Convolve `x` (`[batch, tokens, channels]`) after the cached inputs."""
        joined = torch.cat([cache, x.transpose(1, 2)], dim=-1)
        out = F.conv1d(joined, self.conv.weight, groups=self.channels)
        return out.transpose(1, 2), joined[..., joined.shape[-1] - (self.kernel - 1) :]


def _check_mode(mode: str, allowed: tuple[str, ...]) -> None:
    if mode not in allowed:
        raise ValueError(f"mode must be one of {allowed}, not {mode!r}")


def _paper_text(name: str) -> str:
    return f"not stated in the {name} text read; recorded here as our own choice"


# --- the delta rule (Schlag et al.) ---------------------------------------------------------------

SCHLAG_URL = "https://arxiv.org/html/2102.11174v3"
SCHLAG_TITLE = "Linear Transformers Are Secretly Fast Weight Programmers"

sources.add(
    sources.LabSource(
        id="delta_rule.d_model",
        value=128,
        quote="we set the model dimension (same for key, value, and query) D to 128",
        where="§6.3 Language Modelling Experiments (small configuration), arXiv:2102.11174v3",
        url=SCHLAG_URL,
        title=SCHLAG_TITLE,
        unit="dimensions",
    )
)
sources.add(
    sources.LabSource(
        id="delta_rule.heads",
        value=8,
        quote="H is set to 8",
        where="§6.3 Language Modelling Experiments, arXiv:2102.11174v3",
        url=SCHLAG_URL,
        title=SCHLAG_TITLE,
        unit="heads",
    )
)


class DeltaRule(Mixer):
    """The fast weight programmer with the delta update rule and sum-normalised features.

    Args:
        d_model: Width of the residual stream.
        heads: Number of heads; each projects to `d_model // heads` dimensions.
        feature_map: `elu` (Eq 30) or `dpfp` (Eq 37).
        nu: DPFP's capacity hyperparameter ν (ignored for `elu`).
        chunk_size: Chunk length of the `parallel` form.
        mode: `recurrent` (Eq 20–25) or `parallel` (DeltaNet's chunkwise form) for a full pass.
        context: Training length the source used; recorded, not enforced.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        feature_map: str = "elu",
        nu: int = 1,
        chunk_size: int = 4,
        mode: str = "parallel",
        context: int | None = None,
    ) -> None:
        """Build the projections."""
        super().__init__()
        _check_mode(mode, ("recurrent", "parallel"))
        if feature_map not in ("elu", "dpfp"):
            raise ValueError(f"feature_map must be 'elu' or 'dpfp', not {feature_map!r}")
        if d_model % heads:
            raise ValueError(f"d_model {d_model} is not divisible by heads {heads}")
        self.heads, self.head_dim = heads, d_model // heads
        self.feature_map, self.nu, self.chunk_size, self.mode = feature_map, nu, chunk_size, mode
        self.context = context
        self.d_dot = self.head_dim * (2 * nu if feature_map == "dpfp" else 1)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def phi(self, x: Tensor) -> Tensor:
        """The chosen feature map followed by sum normalisation (Eq 29)."""
        mapped = elu_plus_one(x) if self.feature_map == "elu" else dpfp(x, self.nu)
        return sum_normalise(mapped)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """`W^(0) = 0`: one `d_value × d_dot` fast weight matrix per head."""
        shape = (batch, self.heads, self.head_dim, self.d_dot)
        return {"W": torch.zeros(shape, device=device, dtype=dtype), "pos": 0}

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` and return the output and the state after its last token."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        q = self.phi(split_heads(self.q_proj(x), self.heads))
        k = self.phi(split_heads(self.k_proj(x), self.heads))
        v = split_heads(self.v_proj(x), self.heads)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        if self.mode == "parallel":
            out, fast = delta_rule_chunkwise(q, k, v, beta, state["W"], self.chunk_size)
        else:
            out, fast = delta_rule_recurrent(q, k, v, beta, state["W"])
        return self.out_proj(merge_heads(out)), {"W": fast, "pos": state["pos"] + x.shape[1]}


_LAB = "lab scale, small enough to run every check on a laptop CPU"

register(
    MixerSpec(
        name="delta_rule",
        family="recurrent",
        summary=(
            "Replaces the purely additive write with the delta rule: read the value stored under "
            "the key, then move it toward the new value by a learned write strength."
        ),
        parent="linear_attention",
        covers="delta_rule",
        source="catalogue:delta_rule",
        checked_against=(
            "arXiv:2102.11174v3 Eq 20-25 (update rule), Eq 29 (sum normalisation), Eq 30 (elu+1), "
            "Eq 37 (DPFP); parallel mode uses arXiv:2406.06484v6 Eq 8-12"
        ),
        factory=DeltaRule,
        lab=(
            hparams.ours("d_model", 32, "Width of the residual stream.", _LAB),
            hparams.ours(
                "heads",
                4,
                "Number of heads.",
                "lab scale, four heads of eight dimensions each keep the tests fast",
            ),
            hparams.ours(
                "feature_map",
                "dpfp",
                "Feature map applied to keys and queries.",
                "the paper's own proposal, so the lab exercises it by default",
            ),
            hparams.ours(
                "nu",
                1,
                "DPFP capacity hyperparameter.",
                "the smallest value the paper allows, keeping the fast weight matrix small",
            ),
            hparams.ours(
                "chunk_size",
                5,
                "Chunk length of the parallel form.",
                "deliberately not a divisor of the test lengths so a ragged chunk runs",
            ),
            hparams.ours(
                "mode",
                "parallel",
                "Which form computes a full pass.",
                "the parallel form is the training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "delta_rule.d_model", "Model dimension D."),
            hparams.from_lab_source("heads", "delta_rule.heads", "Heads H."),
            hparams.from_catalogue("context", "delta_rule", "context", "Training context L."),
            hparams.ours(
                "feature_map",
                "dpfp",
                "Feature map applied to keys and queries.",
                "the language model table compares several maps; DPFP shown",
            ),
            hparams.ours("nu", 1, "DPFP capacity hyperparameter.", _paper_text("delta rule")),
            hparams.ours(
                "chunk_size",
                64,
                "Chunk length of the parallel form.",
                "the paper has no parallel form; DeltaNet's usual chunk length reused",
            ),
            hparams.ours(
                "mode",
                "recurrent",
                "Which form computes a full pass.",
                "the paper computes the rule token by token with a custom kernel",
            ),
        ),
        state_growth="constant",
    )
)


# --- DeltaNet (Yang et al. 2024) ------------------------------------------------------------------

DELTANET_URL = "https://arxiv.org/html/2406.06484v6"
DELTANET_TITLE = "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"

sources.add(
    sources.LabSource(
        id="deltanet_parallel.conv_kernel",
        value=4,
        quote="the kernel size for convolution layers is set at 4",
        where="Appendix A.1 Hyperparameters, arXiv:2406.06484v6",
        url=DELTANET_URL,
        title=DELTANET_TITLE,
        unit="tokens",
    )
)


class DeltaNet(Mixer):
    """DeltaNet: the delta rule with L2-normalised SiLU keys and a chunkwise parallel form.

    Args:
        d_model: Width of the residual stream.
        heads: Number of heads.
        head_dim: Width of each head (q, k and v alike).
        chunk_size: Chunk length `C` of the chunkwise form.
        short_conv: Run a short causal convolution on q, k and v before the activation.
        conv_kernel: Width of that convolution.
        mode: `chunkwise` or `recurrent` for a full pass.
        context: Training length the source used; recorded, not enforced.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        chunk_size: int,
        short_conv: bool = True,
        conv_kernel: int = 4,
        mode: str = "chunkwise",
        context: int | None = None,
    ) -> None:
        """Build the projections, the optional convolutions and the output norm."""
        super().__init__()
        _check_mode(mode, ("chunkwise", "recurrent"))
        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
        self.short_conv, self.context = short_conv, context
        inner = heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        if short_conv:
            self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
        self.beta_proj = nn.Linear(d_model, heads)
        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """A zero state matrix per head, plus the convolution caches."""
        d = self.head_dim
        state: dict[str, Any] = {
            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
            "pos": 0,
        }
        if self.short_conv:
            for name, conv in self.convs.items():
                state[f"conv_{name}"] = conv.empty(batch, device, dtype)
        return state

    def _qkv(self, x: Tensor, state: dict[str, Any], new: dict[str, Any]) -> list[Tensor]:
        """Projection, optional short convolution, SiLU; L2 norm on q and k."""
        out = []
        for name, proj in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)):
            h = proj(x)
            if self.short_conv:
                h, new[f"conv_{name}"] = self.convs[name](h, state[f"conv_{name}"])
            h = split_heads(F.silu(h), self.heads)
            out.append(F.normalize(h, dim=-1) if name in "qk" else h)
        return out

    def _log_alpha(self, x: Tensor) -> Tensor | None:
        """No decay in DeltaNet."""
        return None

    def _output(self, o: Tensor, x: Tensor) -> Tensor:
        """Normalise each head, then project (§3 "normalization before output projection")."""
        return self.out_proj(merge_heads(self.norm(o)))

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` and return the output and the state after its last token."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
        q, k, v = self._qkv(x, state, new)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        log_alpha = self._log_alpha(x)
        if self.mode == "chunkwise":
            o, new["S"] = delta_rule_chunkwise(
                q, k, v, beta, state["S"], self.chunk_size, log_alpha
            )
        else:
            o, new["S"] = delta_rule_recurrent(q, k, v, beta, state["S"], log_alpha)
        return self._output(o, x), new


_CHUNK_NOTE = "the paper says the chunk is usually 64 or 128; 64 shown"

register(
    MixerSpec(
        name="deltanet_parallel",
        family="recurrent",
        summary=(
            "Keeps the delta rule but uses L2-normalised SiLU keys and trains it in parallel "
            "over the sequence with a chunkwise WY/UT form; adds a short causal convolution."
        ),
        parent="delta_rule",
        covers="deltanet_parallel",
        source="catalogue:deltanet_parallel",
        checked_against=(
            "arXiv:2406.06484v6 Eq 3 (recurrence), Eq 5-12 (chunkwise WY/UT form), §3.3 feature "
            "map and normalisation, Appendix A.1 (convolution kernel)"
        ),
        factory=DeltaNet,
        lab=(
            hparams.ours("d_model", 32, "Width of the residual stream.", _LAB),
            hparams.ours(
                "heads",
                4,
                "Number of heads.",
                "lab scale, four heads of eight dimensions each keep the tests fast",
            ),
            hparams.ours(
                "head_dim",
                8,
                "Width of each head.",
                "lab scale, four heads of eight dimensions match the residual width",
            ),
            hparams.ours(
                "chunk_size",
                5,
                "Chunk length of the chunkwise form.",
                "deliberately not a divisor of the test lengths so a ragged chunk runs",
            ),
            hparams.ours(
                "short_conv",
                True,
                "Short causal convolution on q, k, v.",
                "on, so the tests exercise the convolution cache during decoding",
            ),
            hparams.ours(
                "conv_kernel",
                4,
                "Width of the short convolution.",
                "the paper's value, kept at lab scale because it is already small",
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the chunkwise form is the training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.ours(
                "d_model",
                2048,
                "Width of the residual stream.",
                "the width of the paper's kernel speed benchmark; not read for its language models",
            ),
            hparams.ours(
                "heads",
                16,
                "Number of heads.",
                _paper_text("DeltaNet") + ", sixteen times 128 is 2048",
            ),
            hparams.from_catalogue("head_dim", "deltanet_parallel", "headDim", "Head dimension."),
            hparams.ours("chunk_size", 64, "Chunk length of the chunkwise form.", _CHUNK_NOTE),
            hparams.ours(
                "short_conv",
                True,
                "Short causal convolution on q, k, v.",
                "the paper's language models use convolution layers of stated width",
            ),
            hparams.from_lab_source(
                "conv_kernel", "deltanet_parallel.conv_kernel", "Convolution kernel size."
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the paper trains with the chunkwise parallel form",
            ),
        ),
        state_growth="constant",
    )
)


# --- Gated DeltaNet -------------------------------------------------------------------------------


def _inverse_softplus(y: Tensor) -> Tensor:
    return y + torch.log(-torch.expm1(-y))


class GatedDeltaNet(DeltaNet):
    """DeltaNet with a data-dependent scalar decay per head (Eq 10) and a SiLU output gate.

    Extra args beyond `DeltaNet`:
        output_gate: Multiply the normalised output by `SiLU(W_g x)` before the projection (§3.4).
    """

    def __init__(self, *args: Any, output_gate: bool = True, **kwargs: Any) -> None:
        """Add the decay parameters and the output gate."""
        super().__init__(*args, **kwargs)
        d_model = self.q_proj.in_features
        self.alpha_proj = nn.Linear(d_model, self.heads, bias=False)
        # Initialisation ranges are ours: A in [1, 16] and softplus(dt_bias) log-uniform in
        # [0.001, 0.1], the ranges the sourcing notes record for this parameterisation.
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1))).clamp_min(
            1e-4
        )
        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
        self.output_gate = output_gate
        if output_gate:
            self.gate_proj = nn.Linear(d_model, self.heads * self.head_dim, bias=False)

    def _log_alpha(self, x: Tensor) -> Tensor:
        """`log α = −exp(A_log) · softplus(W_α x + dt_bias)` (Mamba2's parameterisation)."""
        logits = self.alpha_proj(x) + self.dt_bias
        return (-torch.exp(self.A_log) * F.softplus(logits)).transpose(1, 2)

    def _output(self, o: Tensor, x: Tensor) -> Tensor:
        """Norm, then the SiLU output gate, then the projection."""
        h = merge_heads(self.norm(o))
        if self.output_gate:
            h = h * F.silu(self.gate_proj(x))
        return self.out_proj(h)


GDN_URL = "https://arxiv.org/html/2412.06464v3"

register(
    MixerSpec(
        name="gated_deltanet",
        family="recurrent",
        summary=(
            "Adds a data-dependent scalar decay alpha to DeltaNet's transition, so the memory can "
            "be cleared quickly as well as edited precisely, plus a SiLU output gate."
        ),
        parent="deltanet_parallel",
        covers="gated_deltanet",
        source="catalogue:gated_deltanet",
        checked_against=(
            "arXiv:2412.06464v3 Eq 10 (gated delta rule), §3.3 (chunkwise form; output line "
            "re-derived), §3.4 (block: conv, SiLU, L2, output gate); alpha from the Mamba2 "
            "parameterisation formula"
        ),
        factory=GatedDeltaNet,
        lab=(
            hparams.ours("d_model", 32, "Width of the residual stream.", _LAB),
            hparams.ours(
                "heads",
                4,
                "Number of heads.",
                "lab scale, four heads of eight dimensions each keep the tests fast",
            ),
            hparams.ours(
                "head_dim",
                8,
                "Width of each head.",
                "lab scale, four heads of eight dimensions match the residual width",
            ),
            hparams.ours(
                "chunk_size",
                5,
                "Chunk length of the chunkwise form.",
                "deliberately not a divisor of the test lengths so a ragged chunk runs",
            ),
            hparams.ours(
                "short_conv",
                True,
                "Short causal convolution on q, k, v.",
                "on, as in the paper's block design, exercising the cache",
            ),
            hparams.ours(
                "conv_kernel",
                4,
                "Width of the short convolution.",
                "DeltaNet's stated width, reused because this paper's text omits it",
            ),
            hparams.ours(
                "output_gate",
                True,
                "SiLU output gate.",
                "on, as in the paper's block design figure and its caption",
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the chunkwise form is the training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.ours(
                "d_model", 2048, "Width of the residual stream.", _paper_text("Gated DeltaNet")
            ),
            hparams.ours("heads", 16, "Number of heads.", _paper_text("Gated DeltaNet")),
            hparams.from_catalogue("head_dim", "gated_deltanet", "headDim", "Head dimension."),
            hparams.from_catalogue("context", "gated_deltanet", "context", "Training length."),
            hparams.ours(
                "chunk_size",
                64,
                "Chunk length of the chunkwise form.",
                _paper_text("Gated DeltaNet") + "; DeltaNet's usual 64",
            ),
            hparams.ours(
                "short_conv",
                True,
                "Short causal convolution on q, k, v.",
                "the paper's block design names a short convolution on each path",
            ),
            hparams.ours(
                "conv_kernel",
                4,
                "Width of the short convolution.",
                _paper_text("Gated DeltaNet") + "; DeltaNet's 4 reused",
            ),
            hparams.ours(
                "output_gate",
                True,
                "SiLU output gate.",
                "the paper's block design caption names a SiLU output gate",
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the paper trains with its hardware-efficient chunkwise algorithm",
            ),
        ),
        state_growth="constant",
    )
)


# --- Kimi Delta Attention -------------------------------------------------------------------------

KIMI_LINEAR_URL = "https://arxiv.org/html/2510.26692v2"
KIMI_LINEAR_TITLE = "Kimi Linear: An Expressive, Efficient Attention Architecture"
KIMI_LINEAR_CONFIG = (
    "https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/resolve/main/config.json"
)
KIMI_K3_URL = "https://arxiv.org/html/2607.24653v2"
KIMI_K3_CONFIG = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json"

sources.add(
    sources.LabSource(
        id="kda.output_gate",
        value="low-rank",
        quote="the output gate adopts a low-rank parameterization similar to the forget gate",
        where="§4 Neural Parameterization, arXiv:2510.26692v2",
        url=KIMI_LINEAR_URL,
        title=KIMI_LINEAR_TITLE,
    )
)
sources.add(
    sources.LabSource(
        id="kda.heads",
        value=32,
        quote='"num_heads": 32',
        where="config.json linear_attn_config of Kimi-Linear-48B-A3B-Instruct (main branch)",
        url=KIMI_LINEAR_CONFIG,
        title="Kimi-Linear-48B-A3B-Instruct config.json",
        unit="heads",
    )
)
sources.add(
    sources.LabSource(
        id="kda.conv_kernel",
        value=4,
        quote='"short_conv_kernel_size": 4',
        where="config.json linear_attn_config of Kimi-Linear-48B-A3B-Instruct (main branch)",
        url=KIMI_LINEAR_CONFIG,
        title="Kimi-Linear-48B-A3B-Instruct config.json",
        unit="tokens",
    )
)
sources.add(
    sources.LabSource(
        id="kda.d_model",
        value=2304,
        quote='"hidden_size": 2304',
        where="config.json of Kimi-Linear-48B-A3B-Instruct (main branch)",
        url=KIMI_LINEAR_CONFIG,
        title="Kimi-Linear-48B-A3B-Instruct config.json",
        unit="dimensions",
    )
)
sources.add(
    sources.LabSource(
        id="kda.k3_decay",
        value="scaled sigmoid",
        quote="Kimi K3 instead uses a scaled sigmoid to bound the log-decay from below",
        where="§2.1.1 Kimi Delta Attention, Lower-bounded decay, arXiv:2607.24653v2",
        url=KIMI_K3_URL,
        title="Kimi K3 technical report",
    )
)
sources.add(
    sources.LabSource(
        id="kda.k3_decay_floor",
        value=-5.0,
        quote='"gate_lower_bound": -5.0',
        where="config.json linear_attn_config of Kimi-K3 (main branch)",
        url=KIMI_K3_CONFIG,
        title="Kimi-K3 config.json",
    )
)
sources.add(
    sources.LabSource(
        id="kda.k3_output_gate",
        value="full-rank",
        quote="to an input-dependent full-rank projection",
        where="§2.1.1 Kimi Delta Attention, Full-rank gate, arXiv:2607.24653v2",
        url=KIMI_K3_URL,
        title="Kimi K3 technical report",
    )
)


class KimiDeltaAttention(Mixer):
    """KDA: the gated delta rule with a channel-wise decay, in the `S ∈ R^{d_k × d_v}` orientation.

    Args:
        d_model: Width of the residual stream.
        heads: Number of heads.
        head_dim: `d_k = d_v`, also the rank of the low-rank decay and gate projections.
        chunk_size: Chunk length `C` of the chunkwise form.
        conv_kernel: Width of the short convolution on q, k, v.
        decay_floor: `0.0` gives Kimi Linear's unbounded `−exp(A)·softplus(·)`; a negative value
            `g_min` gives Kimi K3's bounded `g_min · sigmoid(exp(A) · z)`.
        output_gate: `low-rank` (Kimi Linear, Eq 10) or `full-rank` (Kimi K3, Eq 6).
        mode: `chunkwise` or `recurrent` for a full pass.
        context: Training length the source used; recorded, not enforced.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        chunk_size: int,
        conv_kernel: int = 4,
        decay_floor: float = 0.0,
        output_gate: str = "low-rank",
        mode: str = "chunkwise",
        context: int | None = None,
    ) -> None:
        """Build the projections, convolutions, gates and norm."""
        super().__init__()
        _check_mode(mode, ("chunkwise", "recurrent"))
        if output_gate not in ("low-rank", "full-rank"):
            raise ValueError(f"output_gate must be 'low-rank' or 'full-rank', not {output_gate!r}")
        if decay_floor > 0:
            raise ValueError("decay_floor is a log-decay bound and must be <= 0")
        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
        self.decay_floor, self.context = decay_floor, context
        inner = heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
        self.alpha_down = nn.Linear(d_model, head_dim, bias=False)
        self.alpha_up = nn.Linear(head_dim, inner, bias=False)
        self.beta_proj = nn.Linear(d_model, heads, bias=False)
        # K3 initialises A to zero; the Kimi Linear init range is ours, as for Gated DeltaNet.
        k3 = decay_floor < 0
        self.A_log = nn.Parameter(
            torch.zeros(heads) if k3 else torch.empty(heads).uniform_(1, 16).log()
        )
        dt = torch.exp(torch.empty(inner).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(_inverse_softplus(dt.clamp_min(1e-4)))
        if output_gate == "low-rank":
            self.gate = nn.Sequential(
                nn.Linear(d_model, head_dim, bias=False), nn.Linear(head_dim, inner, bias=True)
            )
        else:
            self.gate = nn.Linear(d_model, inner, bias=True)
        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """A zero `d_k × d_v` matrix per head, plus the convolution caches."""
        d = self.head_dim
        state: dict[str, Any] = {
            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
            "pos": 0,
        }
        for name, conv in self.convs.items():
            state[f"conv_{name}"] = conv.empty(batch, device, dtype)
        return state

    def log_alpha(self, x: Tensor) -> Tensor:
        """Channel-wise log decay `[batch, heads, tokens, d_k]`."""
        z = split_heads(self.alpha_up(self.alpha_down(x)) + self.dt_bias, self.heads)
        scale = torch.exp(self.A_log).view(1, -1, 1, 1)
        if self.decay_floor < 0:
            return self.decay_floor * torch.sigmoid(scale * z)
        return -scale * F.softplus(z)

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` and return the output and the state after its last token."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
        paths = {}
        for name in "qkv":
            h, new[f"conv_{name}"] = self.convs[name](
                getattr(self, f"{name}_proj")(x), state[f"conv_{name}"]
            )
            paths[name] = split_heads(F.silu(h), self.heads)
        q, k, v = F.normalize(paths["q"], dim=-1), F.normalize(paths["k"], dim=-1), paths["v"]
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        log_alpha = self.log_alpha(x)
        if self.mode == "chunkwise":
            o, new["S"] = kda_chunkwise(q, k, v, beta, state["S"], log_alpha, self.chunk_size)
        else:
            o, new["S"] = kda_recurrent(q, k, v, beta, state["S"], log_alpha)
        h = torch.sigmoid(self.gate(x)) * merge_heads(self.norm(o))
        return self.out_proj(h), new


#: The two Kimi K3 changes, as build overrides: `get("kda").build("paper", **K3_OVERRIDES)`.
#: Each value is read from `sources`, so the override carries its provenance.
K3_OVERRIDES = {
    "decay_floor": sources.LAB_SOURCES["kda.k3_decay_floor"].value,
    "output_gate": sources.LAB_SOURCES["kda.k3_output_gate"].value,
}

register(
    MixerSpec(
        name="kda",
        family="recurrent",
        summary=(
            "Makes Gated DeltaNet's decay channel-wise (one rate per key dimension), with a "
            "low-rank decay projection and a low-rank sigmoid output gate."
        ),
        parent="gated_deltanet",
        covers="kda",
        source="catalogue:kda",
        checked_against=(
            "arXiv:2510.26692v2 Eq 1 (recurrence), Eq 2-9 (chunkwise form), §4 and Eq 10 "
            "(parameterisation, output gate); decay gate formula from fla (MIT); Kimi K3 options "
            "from arXiv:2607.24653v2 §2.1.1 Eq 5-6"
        ),
        factory=KimiDeltaAttention,
        lab=(
            hparams.ours("d_model", 32, "Width of the residual stream.", _LAB),
            hparams.ours(
                "heads",
                4,
                "Number of heads.",
                "lab scale, four heads of eight dimensions each keep the tests fast",
            ),
            hparams.ours(
                "head_dim",
                8,
                "Key and value dimension per head.",
                "lab scale, four heads of eight dimensions match the residual width",
            ),
            hparams.ours(
                "chunk_size",
                5,
                "Chunk length of the chunkwise form.",
                "deliberately not a divisor of the test lengths so a ragged chunk runs",
            ),
            hparams.ours(
                "conv_kernel",
                4,
                "Width of the short convolution.",
                "the released model's value, already small enough for the lab",
            ),
            hparams.ours(
                "decay_floor",
                0.0,
                "Zero for Kimi Linear's decay; negative for K3's.",
                "the lab default is the Kimi Linear form this entry covers",
            ),
            hparams.ours(
                "output_gate",
                "low-rank",
                "Rank of the sigmoid output gate.",
                "the lab default is the Kimi Linear form this entry covers",
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the chunkwise form is the training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "kda.d_model", "Hidden size of the released model."),
            hparams.from_lab_source("heads", "kda.heads", "KDA heads in the released model."),
            hparams.from_catalogue("head_dim", "kda", "headDim", "Key and value head dimension."),
            hparams.from_catalogue("chunk_size", "kda", "chunk", "Chunk length C."),
            hparams.from_lab_source("conv_kernel", "kda.conv_kernel", "Short convolution width."),
            hparams.ours(
                "decay_floor",
                0.0,
                "Zero for Kimi Linear's decay; negative for K3's.",
                "Kimi Linear's decay is unbounded; zero selects that form",
            ),
            hparams.from_lab_source("output_gate", "kda.output_gate", "Rank of the output gate."),
            hparams.from_catalogue("context", "kda", "context", "Pre-training context window."),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the paper trains with its hardware-efficient chunkwise algorithm",
            ),
        ),
        state_growth="constant",
    )
)


# --- Gated DeltaNet-2 -----------------------------------------------------------------------------

GDN2_URL = "https://arxiv.org/html/2605.22791v1"

sources.add(
    sources.LabSource(
        id="gated_deltanet2.d_model",
        value=2048,
        quote="Since d model = 2048",
        where="Appendix E.1 Training, arXiv:2605.22791v1",
        url=GDN2_URL,
        title="Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention",
        unit="dimensions",
    )
)


class GatedDeltaNet2(Mixer):
    """Gated DeltaNet-2: separate channel-wise erase and write gates on a channel-wise decay.

    Args:
        d_model: Width of the residual stream.
        heads: Number of heads.
        head_dim: `d_k = d_v` per head.
        chunk_size: Chunk length `C` of the chunkwise form.
        conv_kernel: Width of the short convolution on q, k, v.
        mode: `chunkwise` or `recurrent` for a full pass.
        context: Training length the source used; recorded, not enforced.
        state_size: `d_k` as the source states it; must equal `head_dim` when given.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        chunk_size: int,
        conv_kernel: int = 4,
        mode: str = "chunkwise",
        context: int | None = None,
        state_size: int | None = None,
    ) -> None:
        """Build the projections, convolutions, gates and norm."""
        super().__init__()
        _check_mode(mode, ("chunkwise", "recurrent"))
        if state_size is not None and state_size != head_dim:
            raise ValueError("this module uses d_k = d_v = head_dim; state_size must match")
        self.heads, self.head_dim, self.chunk_size, self.mode = heads, head_dim, chunk_size, mode
        self.context = context
        inner = heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.convs = nn.ModuleDict({n: ShortConv(inner, conv_kernel) for n in "qkv"})
        self.erase_proj = nn.Linear(d_model, inner)
        self.write_proj = nn.Linear(d_model, inner)
        self.decay_proj = nn.Linear(d_model, inner, bias=False)
        # `a` per key head and `δ` per key channel (App C.1); initial ranges are ours.
        self.a = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(inner).uniform_(math.log(1e-3), math.log(1e-1)))
        self.delta = nn.Parameter(_inverse_softplus(dt.clamp_min(1e-4)))
        self.gate_proj = nn.Linear(d_model, inner, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """A zero `d_k × d_v` matrix per head, plus the convolution caches."""
        d = self.head_dim
        state: dict[str, Any] = {
            "S": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
            "pos": 0,
        }
        for name, conv in self.convs.items():
            state[f"conv_{name}"] = conv.empty(batch, device, dtype)
        return state

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` and return the output and the state after its last token."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        new: dict[str, Any] = {"pos": state["pos"] + x.shape[1]}
        paths = {}
        for name in "qkv":
            h, new[f"conv_{name}"] = self.convs[name](
                getattr(self, f"{name}_proj")(x), state[f"conv_{name}"]
            )
            paths[name] = split_heads(F.silu(h), self.heads)
        q, k, v = F.normalize(paths["q"], dim=-1), F.normalize(paths["k"], dim=-1), paths["v"]
        erase = split_heads(torch.sigmoid(self.erase_proj(x)), self.heads)
        write = split_heads(torch.sigmoid(self.write_proj(x)), self.heads)
        logits = split_heads(self.decay_proj(x) + self.delta, self.heads)
        log_alpha = -torch.exp(self.a).view(1, -1, 1, 1) * F.softplus(logits)
        if self.mode == "chunkwise":
            o, new["S"] = gdn2_chunkwise(
                q, k, v, erase, write, state["S"], log_alpha, self.chunk_size
            )
        else:
            o, new["S"] = gdn2_recurrent(q, k, v, erase, write, state["S"], log_alpha)
        h = merge_heads(self.norm(o)) * F.silu(self.gate_proj(x))
        return self.out_proj(h), new


register(
    MixerSpec(
        name="gated_deltanet2",
        family="recurrent",
        summary=(
            "Splits the delta rule's scalar write strength into a channel-wise erase gate on the "
            "key and a channel-wise write gate on the value, over a channel-wise decay."
        ),
        parent="gated_deltanet",
        covers="gated_deltanet2",
        source="catalogue:gated_deltanet2",
        checked_against=(
            "arXiv:2605.22791v1 Eq 8-12 (recurrence and gates), §3.3 Eq 18-25 (chunkwise form), "
            "§3.5 and App C.1 (block and parameter shapes)"
        ),
        factory=GatedDeltaNet2,
        lab=(
            hparams.ours("d_model", 32, "Width of the residual stream.", _LAB),
            hparams.ours(
                "heads",
                4,
                "Number of heads.",
                "lab scale, four heads of eight dimensions each keep the tests fast",
            ),
            hparams.ours(
                "head_dim",
                8,
                "Key and value dimension per head.",
                "lab scale, four heads of eight dimensions match the residual width",
            ),
            hparams.ours(
                "chunk_size",
                5,
                "Chunk length of the chunkwise form.",
                "deliberately not a divisor of the test lengths so a ragged chunk runs",
            ),
            hparams.ours(
                "conv_kernel",
                4,
                "Width of the short convolution.",
                "the value the Gated DeltaNet family uses elsewhere, already small",
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the chunkwise form is the training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "gated_deltanet2.d_model", "Model dimension."),
            hparams.from_catalogue("heads", "gated_deltanet2", "heads", "Heads H."),
            hparams.from_catalogue("head_dim", "gated_deltanet2", "headDim", "d_v per head."),
            hparams.from_catalogue("state_size", "gated_deltanet2", "stateSize", "d_k per head."),
            hparams.from_catalogue("chunk_size", "gated_deltanet2", "chunk", "Chunk length C."),
            hparams.from_catalogue("context", "gated_deltanet2", "context", "Training length."),
            hparams.ours(
                "conv_kernel",
                4,
                "Width of the short convolution.",
                _paper_text("Gated DeltaNet-2") + "; DeltaNet's 4 reused",
            ),
            hparams.ours(
                "mode",
                "chunkwise",
                "Which form computes a full pass.",
                "the paper trains with its fused chunkwise kernels",
            ),
        ),
        state_growth="constant",
    )
)
