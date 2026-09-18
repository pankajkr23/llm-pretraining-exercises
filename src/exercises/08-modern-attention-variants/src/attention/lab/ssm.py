"""The recurrent family: selective state space models, Mamba and Mamba-3.

**The problem they solve.** Softmax attention keeps every past key and value, so its memory grows
with every token and each new token reads all of them. A state space model (SSM) instead folds the
past into a fixed-size hidden state `h` and updates it once per token, like an RNN, so decoding
costs the same at token 10 as at token 100,000. The older SSMs (S4) used the *same* update for
every token, which made them fast but unable to decide what to keep. Mamba's change is to make the
update depend on the token itself — the model can choose, token by token, to remember or to forget.

**`mamba` — the S6 selective SSM block (arXiv:2312.00752v2, Algorithm 2, §3.2–3.6).** Relative to
`linear_attention`, which adds `k v^T` to a matrix state with no forgetting, each channel `d` of
the expanded input keeps `N` numbers and updates them with its own decay:

    h_t = Ā_t ⊙ h_{t-1} + B̄_t x_t,     y_t = C_t · h_t + D x_t                (Eq 2, per channel)
    Ā_t = exp(Δ_t A)                                                           (Eq 4)
    B̄_t = (Δ_t A)^{-1} (exp(Δ_t A) - 1) · Δ_t B_t      "zoh", the paper's rule   (Eq 4)
    B̄_t = Δ_t B_t                                      "euler", what the released code does

`A` is diagonal (`[channels, N]`), so every product above is elementwise. The *selection* is that
`B_t = Linear_N(x_t)`, `C_t = Linear_N(x_t)` and
`Δ_t = softplus(Parameter + Linear_D(Linear_R(x_t)))` are computed from the token (§3.2, and the
low-rank form of §3.6). Around that core, the block of §3.4 expands the width by `E`, runs a short
causal depthwise convolution and SiLU on one branch, gates the SSM output with `SiLU(z)` from the
other branch, and projects back.

Where this file and the reference code differ, the paper wins and the difference is stated:

- `A` is a plain parameter initialised to S4D-Real, `A_n = -(n+1)` with `n` counted from zero. The
  official code (state-spaces/mamba, Apache-2.0) stores `A_log = log(1..N)` and uses
  `A = -exp(A_log)`: the same starting values, but it keeps `A` negative during training, which a
  plain parameter does not guarantee.
- `Δ`'s bias starts at `softplus^{-1}(Uniform([0.001, 0.1]))` as §3.6 says; the code samples the
  same range log-uniformly.
- `discretization="zoh"` is the default because Eq 4 is what the paper states. Mamba-3's Table 1
  footnote records that the released implementation uses `B̄ = ΔB` instead; `"euler"` is that.
- The convolution width (4), the low-rank width rule `ceil(d_model / 16)` and the skip `D`
  (initialised to ones) are not in the paper; they come from the official code and are marked so.

**`mamba3` — SISO Mamba-3 (arXiv:2603.15569v1).** Relative to `mamba`, four things change:

1. *Scalar decay per head, Mamba-2 style* (§2.2, Eq 1): `A_t` is one number per head, and — per
   Remark 1 — data-dependent. The state of each head is a matrix `[P, N]`: `h_t = α_t h_{t-1} +
   γ_t B_t x_t^T` with `α_t = exp(Δ_t A_t)`.
2. *Exponential-trapezoidal discretization* (Proposition 1, Eq 5): the input integral averages the
   two ends of the step instead of holding the right one,

       h_t = α_t h_{t-1} + (1 - λ_t) Δ_t α_t B_{t-1} x_{t-1} + λ_t Δ_t B_t x_t,   λ_t = σ(u_t)

   (λ_t's form is Appendix A.3). λ_t = 1 is exactly Mamba-2's exponential-Euler update. The layer
   must therefore remember the previous token's `B_{t-1}` and `x_{t-1}`, which it keeps in its
   state — still constant size.
3. *Complex state as rotations* (Proposition 2, Eq 9): the state's `N` channels are `N/2` pairs,
   and each step rotates pair `i` by the data-dependent angle `Δ_t θ_t[i]` before decaying it.
   Generalising Eq 9 to the trapezoidal rule gives the update this file's scan implements,

       h_t = α_t R_t (h_{t-1} + (1 - λ_t) Δ_t B_{t-1} x_{t-1}^T) + λ_t Δ_t B_t x_t^T,

   and Proposition 4 (Eq 11) shows the same outputs come from rotating `B` and `C` by the
   *cumulative* angle instead — "the RoPE trick" — which is what the parallel form here does.
   Both are implemented; the tests prove they agree.
4. *BCNorm and B/C biases* (§3.4): RMSNorm on the `B` and `C` projections, then learnable
   head-specific, channel-wise biases initialised to ones (Appendix F). The short convolution is
   **off by default**, because the paper finds bias plus the trapezoidal rule make it redundant
   (Table 5a); `use_conv=True` puts it back on the `x` branch only.

Not implemented: MIMO (Appendix C) — this is the SISO layer the paper uses by default. Not stated
in the paper, and therefore our choice: how `A_t` is produced (`-softplus` of a projection), that
`θ_t` is one projection shared by every head (the per-head `Δ_t` still makes angles per head),
that `λ_t` is per head, the `Δ` bias initialisation (taken from the parent), and the skip `D`
(kept from the parent Mamba block; the Mamba-3 paper does not mention one).

**The trade-off.** State is constant: the memory after 100 tokens and after 100,000 is the same.
The price is that the past is compressed — a recurrent layer cannot look up an exact token the way
attention can, which is the recall gap hybrid models exist to close.

Two paths per variant, selectable with `mode`:

- `"scan"` — the recurrence, one token at a time. This is the reference. Mamba's Algorithm 2
  itself says the time-varying SSM is computed by recurrence (scan) only, so it is `mamba`'s
  default.
- `"parallel"` — every token at once, from the cumulative sum of the log-decays: the decay from
  step `s` to step `t` is `exp(S_t - S_s)`. It is quadratic in memory, which is fine at lab scale
  and is how the lab checks the scan against an independent calculation. It is `mamba3`'s default,
  because Mamba-3's layer is computed in that masked (SSD-style) form.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn.functional import silu, softplus

from attention.lab import hparams, ops
from attention.lab.base import Mixer, MixerSpec
from attention.lab.registry import register
from attention.lab.sources import LabSource, add

DISCRETIZATIONS = ("zoh", "euler")
MODES = ("scan", "parallel")

#: Mamba-3's B and C biases start at one (Appendix F of arXiv:2603.15569v1).
BC_BIAS_INIT = 1.0

add(
    LabSource(
        id="mamba.convKernel",
        value=4,
        quote="d_conv=4,",
        where=(
            "Mamba.__init__ default arguments, mamba_ssm/modules/mamba_simple.py, official "
            "state-spaces/mamba code (Apache-2.0) at commit e9594ce; the Mamba paper does not "
            "state the convolution width"
        ),
        url=(
            "https://raw.githubusercontent.com/state-spaces/mamba/"
            "e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py"
        ),
        title="state-spaces/mamba: mamba_ssm/modules/mamba_simple.py",
        unit="tokens",
    )
)
add(
    LabSource(
        id="mamba.dModel",
        value=1024,
        quote="We use a model dimension of D = 1024 and state dimension N = 16",
        where="E.5 Efficiency Benchmark (Scan Operation), arXiv:2312.00752v2",
        url="https://arxiv.org/html/2312.00752v2",
        title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        unit="dimensions",
    )
)


# --- shared pieces -------------------------------------------------------------------------------


def _inverse_softplus(y: Tensor) -> Tensor:
    """The `z` with `softplus(z) == y`, for `y > 0`."""
    return y + torch.log(-torch.expm1(-y))


def _delta_bias(width: int, dt_min: float, dt_max: float) -> Tensor:
    """`softplus^{-1}(Uniform([dt_min, dt_max]))`, the Δ initialisation of Mamba §3.6."""
    if not 0 < dt_min < dt_max:
        raise ValueError(f"need 0 < dt_min < dt_max, got {dt_min}, {dt_max}")
    dt = dt_min + (dt_max - dt_min) * torch.rand(width)
    return _inverse_softplus(dt)


def _decay_grid(log_decay: Tensor, time_dim: int) -> Tensor:
    """`exp(S_t - S_s)` for `s <= t` and 0 above the diagonal, with `S` the cumulative log-decay.

    `log_decay` has time on axis `time_dim`; the result inserts a second time axis right after it,
    so a `[b, T, ...]` input gives `[b, T, T, ...]` indexed `[b, t, s, ...]`.
    """
    cumulative = log_decay.cumsum(time_dim)
    later = cumulative.unsqueeze(time_dim + 1)  # S_t
    earlier = cumulative.unsqueeze(time_dim)  # S_s
    steps = log_decay.shape[time_dim]
    allowed = ops.causal_mask(steps, steps, device=log_decay.device)
    allowed = allowed.view(*allowed.shape, *([1] * (log_decay.dim() - time_dim - 1)))
    gap = torch.where(allowed, later - earlier, torch.full_like(later - earlier, -math.inf))
    return gap.exp()


class _CausalConv(nn.Module):
    """A depthwise causal convolution whose left context is carried in the state between calls.

    With an all-zero starting buffer it equals the reference code's `padding=k-1` then truncate.
    """

    def __init__(self, width: int, kernel: int) -> None:
        super().__init__()
        if kernel < 1:
            raise ValueError(f"conv_kernel must be at least 1, got {kernel}")
        self.width, self.kernel = width, kernel
        self.conv = nn.Conv1d(width, width, kernel, groups=width, bias=True)

    def empty(self, batch: int, device=None, dtype=None) -> Tensor:
        """The buffer before any token: `[batch, width, kernel - 1]` zeros."""
        return torch.zeros(batch, self.width, self.kernel - 1, device=device, dtype=dtype)

    def forward(self, x: Tensor, buffer: Tensor) -> tuple[Tensor, Tensor]:
        """Convolve `x` (`[b, T, width]`) after `buffer`; return the output and the new buffer."""
        seq = torch.cat([buffer, x.transpose(1, 2)], dim=2)
        out = self.conv(seq).transpose(1, 2)
        return out, seq[:, :, seq.shape[2] - (self.kernel - 1) :]


# --- mamba: the S6 selective scan ----------------------------------------------------------------


def zoh_input_factor(delta_a: Tensor) -> Tensor:
    """`(exp(z) - 1) / z` elementwise, the ZOH factor on `ΔB` for a diagonal `A` (Eq 4).

    For diagonal `A`, `(ΔA)^{-1}(exp(ΔA) - I) · ΔB` is `ΔB` times this factor per element. It tends
    to 1 as `z → 0`, which is why ZOH and Euler agree for a small step.
    """
    small = delta_a.abs() < 1e-8
    safe = torch.where(small, torch.ones_like(delta_a), delta_a)
    return torch.where(small, 1.0 + delta_a / 2.0, torch.expm1(safe) / safe)


def discretize(
    delta: Tensor, a: Tensor, b_in: Tensor, discretization: str
) -> tuple[Tensor, Tensor]:
    """Ā and B̄ for every token and channel (Algorithm 2, line 7).

    Args:
        delta: Step sizes `[batch, T, channels]`.
        a: The diagonal state matrix `[channels, N]`.
        b_in: Input projections `[batch, T, N]`.
        discretization: `"zoh"` (Eq 4) or `"euler"` (`B̄ = ΔB`, the released code).

    Returns:
        `Ā` and `B̄`, each `[batch, T, channels, N]`.
    """
    if discretization not in DISCRETIZATIONS:
        raise ValueError(f"discretization must be one of {DISCRETIZATIONS}")
    delta_a = delta.unsqueeze(-1) * a
    b_bar = delta.unsqueeze(-1) * b_in.unsqueeze(2)
    if discretization == "zoh":
        b_bar = b_bar * zoh_input_factor(delta_a)
    return delta_a.exp(), b_bar


def selective_scan(
    x: Tensor,
    delta: Tensor,
    a: Tensor,
    b_in: Tensor,
    c_out: Tensor,
    skip: Tensor,
    h: Tensor,
    discretization: str,
) -> tuple[Tensor, Tensor]:
    """The recurrence of Eq 2, one token at a time — the reference.

    Args:
        x: `[batch, T, channels]`.
        delta: `[batch, T, channels]`.
        a: `[channels, N]`.
        b_in: `[batch, T, N]`.
        c_out: `[batch, T, N]`.
        skip: The skip `D`, `[channels]`.
        h: The starting state `[batch, channels, N]`.
        discretization: `"zoh"` or `"euler"`.

    Returns:
        The output `[batch, T, channels]` and the state after the last token.
    """
    a_bar, b_bar = discretize(delta, a, b_in, discretization)
    outputs = []
    for t in range(x.shape[1]):
        h = a_bar[:, t] * h + b_bar[:, t] * x[:, t, :, None]
        outputs.append((h * c_out[:, t, None, :]).sum(-1) + skip * x[:, t])
    return torch.stack(outputs, dim=1), h


def selective_parallel(
    x: Tensor,
    delta: Tensor,
    a: Tensor,
    b_in: Tensor,
    c_out: Tensor,
    skip: Tensor,
    h: Tensor,
    discretization: str,
) -> tuple[Tensor, Tensor]:
    """Eq 2 unrolled: `h_t = exp(S_t) h_0 + Σ_{s≤t} exp(S_t - S_s) B̄_s x_s`, all tokens at once.

    `S_t = A · Σ_{r≤t} Δ_r` is the cumulative log-decay per channel and state entry. Same
    arguments and returns as `selective_scan`; memory is quadratic in `T`.
    """
    _, b_bar = discretize(delta, a, b_in, discretization)
    log_decay = delta.unsqueeze(-1) * a  # Δ_t A, [b, T, channels, N]
    grid = _decay_grid(log_decay, time_dim=1)  # [b, t, s, channels, N]
    inputs = b_bar * x.unsqueeze(-1)  # [b, s, channels, N]
    states = (grid * inputs.unsqueeze(1)).sum(2) + log_decay.cumsum(1).exp() * h.unsqueeze(1)
    y = (states * c_out.unsqueeze(2)).sum(-1) + skip * x
    return y, states[:, -1]


class Mamba(Mixer):
    """The Mamba block around the S6 selective SSM (§3.4, Algorithm 2)."""

    def __init__(
        self,
        d_model: int,
        state_size: int,
        expansion: int,
        conv_kernel: int,
        dt_rank: int,
        dt_min: float,
        dt_max: float,
        discretization: str = "zoh",
        mode: str = "scan",
        context: int | None = None,
    ) -> None:
        """Build the block.

        Args:
            d_model: Model width `D`.
            state_size: `N`, numbers of state per expanded channel.
            expansion: `E`; the SSM runs at width `E · D`.
            conv_kernel: Width of the causal depthwise convolution.
            dt_rank: `R` in `s_Δ(x) = Linear_D(Linear_R(x))`.
            dt_min: Lower end of Δ's initial range.
            dt_max: Upper end of Δ's initial range.
            discretization: `"zoh"` (Eq 4) or `"euler"`.
            mode: `"scan"` or `"parallel"`.
            context: Training context length; recorded only, a recurrent layer has no limit.
        """
        super().__init__()
        if discretization not in DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {DISCRETIZATIONS}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.d_model, self.state_size = d_model, state_size
        self.inner = expansion * d_model
        self.dt_rank, self.discretization, self.mode = dt_rank, discretization, mode
        self.context = context
        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
        self.conv = _CausalConv(self.inner, conv_kernel)
        self.x_proj = nn.Linear(self.inner, dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.inner, bias=True)
        with torch.no_grad():
            self.dt_proj.bias.copy_(_delta_bias(self.inner, dt_min, dt_max))
        s4d_real = -(torch.arange(state_size, dtype=torch.float32) + 1.0)
        self.a = nn.Parameter(s4d_real.repeat(self.inner, 1))
        self.skip = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """Zero SSM state `[batch, E·D, N]`, an empty convolution buffer, and no tokens seen."""
        return {
            "h": torch.zeros(batch, self.inner, self.state_size, device=device, dtype=dtype),
            "conv": self.conv.empty(batch, device=device, dtype=dtype),
            "pos": 0,
        }

    def forward(
        self, x: Tensor, state: dict | None = None, memory: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        """Run the block over `x` (`[batch, T, d_model]`), continuing from `state`."""
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        branch, gate = self.in_proj(x).chunk(2, dim=-1)
        branch, conv_buffer = self.conv(branch, state["conv"])
        branch = silu(branch)
        low, b_in, c_out = self.x_proj(branch).split(
            [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        delta = softplus(self.dt_proj(low))  # τ_Δ(Parameter + s_Δ(x)); the bias is the Parameter
        run = selective_scan if self.mode == "scan" else selective_parallel
        y, h = run(branch, delta, self.a, b_in, c_out, self.skip, state["h"], self.discretization)
        out = self.out_proj(y * silu(gate))
        return out, {"h": h, "conv": conv_buffer, "pos": state["pos"] + x.shape[1]}


# --- mamba3: exponential-trapezoidal, rotating state ----------------------------------------------


def rotate_pairs(v: Tensor, angles: Tensor) -> Tensor:
    """Rotate adjacent pairs of `v`'s last axis by `angles` (`R(θ)` of Proposition 2).

    A thin wrapper over `ops.apply_rope`, which rotates `(v1, v2), (v3, v4), ...` by
    `[[cos, -sin], [sin, cos]]`; `angles` broadcasts against `v[..., ::2]`.
    """
    return ops.apply_rope(v, angles)


def _shift_in(previous: Tensor, seq: Tensor) -> Tensor:
    """`[previous, seq_0, ..., seq_{T-2}]` along axis 2: each token's predecessor."""
    return torch.cat([previous.unsqueeze(2), seq[:, :, :-1]], dim=2)


def trapezoidal_scan(
    x: Tensor,
    b_in: Tensor,
    c_out: Tensor,
    log_alpha: Tensor,
    delta: Tensor,
    lam: Tensor,
    angles: Tensor,
    skip: Tensor,
    state: dict,
) -> tuple[Tensor, dict]:
    """Mamba-3's recurrence with the rotation applied to the state directly — the reference.

    `h_t = α_t R_t (h_{t-1} + (1-λ_t) Δ_t B_{t-1} x_{t-1}^T) + λ_t Δ_t B_t x_t^T`, `y_t = h_t C_t`.
    With `angles = 0` this is Eq 5; with `λ = 1` it is Eq 9.

    Args:
        x: `[batch, heads, T, P]`.
        b_in: `[batch, heads, T, N]`, after BCNorm and bias.
        c_out: `[batch, heads, T, N]`, after BCNorm and bias.
        log_alpha: `Δ_t A_t`, `[batch, heads, T]`.
        delta: `Δ_t`, `[batch, heads, T]`.
        lam: `λ_t`, `[batch, heads, T]`.
        angles: `Δ_t θ_t`, `[batch, heads, T, N/2]`.
        skip: `D`, `[heads]`.
        state: `h` `[batch, heads, P, N]`, `b_prev` `[batch, heads, N]`, `x_prev`
            `[batch, heads, P]`.

    Returns:
        The output `[batch, heads, T, P]` and the new `h`, `b_prev`, `x_prev`.
    """
    h, b_prev, x_prev = state["h"], state["b_prev"], state["x_prev"]
    outputs = []
    for t in range(x.shape[2]):
        alpha = log_alpha[:, :, t].exp()[..., None, None]
        previous = x_prev.unsqueeze(-1) * b_prev.unsqueeze(-2)
        current = x[:, :, t].unsqueeze(-1) * b_in[:, :, t].unsqueeze(-2)
        carried = h + ((1.0 - lam[:, :, t]) * delta[:, :, t])[..., None, None] * previous
        h = alpha * rotate_pairs(carried, angles[:, :, t, None, :])
        h = h + (lam[:, :, t] * delta[:, :, t])[..., None, None] * current
        outputs.append((h * c_out[:, :, t, None, :]).sum(-1) + skip[:, None] * x[:, :, t])
        b_prev, x_prev = b_in[:, :, t], x[:, :, t]
    return torch.stack(outputs, dim=2), {"h": h, "b_prev": b_prev, "x_prev": x_prev}


def trapezoidal_parallel(
    x: Tensor,
    b_in: Tensor,
    c_out: Tensor,
    log_alpha: Tensor,
    delta: Tensor,
    lam: Tensor,
    angles: Tensor,
    skip: Tensor,
    state: dict,
) -> tuple[Tensor, dict]:
    """Proposition 4 (Eq 11): the same layer as a decay-masked linear attention with the RoPE trick.

    `B` and `C` are rotated by the cumulative angle `-Φ_t` (`Φ_t = Σ_{i≤t} Δ_i θ_i`, counted from
    the start of this call, so the carried state needs no phase), the scalar decays become a
    `[T, T]` mask `exp(S_t - S_s)`, and the trapezoidal rule adds a second key built from each
    token's predecessor. The returned state is rotated back by `+Φ_T` into the frame the scan
    keeps. Same arguments and returns as `trapezoidal_scan`.
    """
    h0 = state["h"]
    phase = angles.cumsum(2)
    phase_before = torch.cat([torch.zeros_like(phase[:, :, :1]), phase[:, :, :-1]], dim=2)
    query = rotate_pairs(c_out, -phase)  # (Π R_i^T) C_t
    key = rotate_pairs(b_in, -phase) * (lam * delta).unsqueeze(-1)  # γ_t (Π R_i^T) B_t
    beta = (1.0 - lam) * delta * log_alpha.exp()
    b_before = _shift_in(state["b_prev"], b_in)
    x_before = _shift_in(state["x_prev"], x)
    key_before = rotate_pairs(b_before, -phase_before) * beta.unsqueeze(-1)
    grid = _decay_grid(log_alpha, time_dim=2)  # [b, heads, t, s]
    from_start = log_alpha.cumsum(2).exp()  # Π_{r≤t} α_r

    def read(keys: Tensor, values: Tensor) -> Tensor:
        return torch.einsum("bhtn,bhsn,bhts,bhsp->bhtp", query, keys, grid, values)

    y = read(key, x) + read(key_before, x_before)
    y = y + from_start.unsqueeze(-1) * torch.einsum("bhtn,bhpn->bhtp", query, h0)
    y = y + skip[:, None, None] * x

    last = grid[:, :, -1]  # decay from each s to the last token
    h_rot = from_start[:, :, -1, None, None] * h0
    h_rot = h_rot + torch.einsum("bhs,bhsn,bhsp->bhpn", last, key, x)
    h_rot = h_rot + torch.einsum("bhs,bhsn,bhsp->bhpn", last, key_before, x_before)
    h = rotate_pairs(h_rot, phase[:, :, -1, None, :])
    return y, {"h": h, "b_prev": b_in[:, :, -1], "x_prev": x[:, :, -1]}


class Mamba3(Mixer):
    """SISO Mamba-3: Mamba-2's layer with the trapezoidal rule, rotations, BCNorm and B/C biases."""

    def __init__(
        self,
        d_model: int,
        state_size: int,
        expansion: int,
        head_dim: int,
        conv_kernel: int,
        dt_min: float,
        dt_max: float,
        use_conv: bool = False,
        trapezoid: bool = True,
        rotate: bool = True,
        mode: str = "parallel",
        context: int | None = None,
    ) -> None:
        """Build the layer.

        Args:
            d_model: Model width `D`.
            state_size: `N` per head; even, because the rotation acts on pairs.
            expansion: The layer runs at width `expansion · D`.
            head_dim: `P`, the width of each head's input.
            conv_kernel: Width of the optional short convolution.
            dt_min: Lower end of Δ's initial range.
            dt_max: Upper end of Δ's initial range.
            use_conv: Put the short causal convolution and its SiLU back on the `x` branch.
            trapezoid: `λ_t = σ(u_t)` when True; `λ_t = 1` (exponential-Euler) when False.
            rotate: Rotate the state by `Δ_t θ_t` when True; a real-valued state when False.
            mode: `"parallel"` (RoPE trick) or `"scan"` (direct rotation).
            context: Training context length; recorded only.
        """
        super().__init__()
        inner = expansion * d_model
        if state_size % 2:
            raise ValueError(f"state_size must be even to rotate pairs, got {state_size}")
        if inner % head_dim:
            raise ValueError(f"expanded width {inner} is not a multiple of head_dim {head_dim}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.d_model, self.state_size, self.head_dim = d_model, state_size, head_dim
        self.inner, self.heads = inner, inner // head_dim
        self.trapezoid, self.rotate, self.mode, self.context = trapezoid, rotate, mode, context
        # x, z, B, C, Δ, A, u (for λ), θ — all projected from the token.
        self.sizes = [inner, inner, state_size, state_size, *[self.heads] * 3, state_size // 2]
        self.in_proj = nn.Linear(d_model, sum(self.sizes), bias=False)
        self.conv = _CausalConv(inner, conv_kernel) if use_conv else None
        self.dt_bias = nn.Parameter(_delta_bias(self.heads, dt_min, dt_max))
        self.b_norm = nn.RMSNorm(state_size)
        self.c_norm = nn.RMSNorm(state_size)
        self.b_bias = nn.Parameter(torch.full((self.heads, state_size), BC_BIAS_INIT))
        self.c_bias = nn.Parameter(torch.full((self.heads, state_size), BC_BIAS_INIT))
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """Zero state `[batch, heads, P, N]`, zero previous `B` and `x`, and no tokens seen."""
        kw = {"device": device, "dtype": dtype}
        state = {
            "h": torch.zeros(batch, self.heads, self.head_dim, self.state_size, **kw),
            "b_prev": torch.zeros(batch, self.heads, self.state_size, **kw),
            "x_prev": torch.zeros(batch, self.heads, self.head_dim, **kw),
            "pos": 0,
        }
        if self.conv is not None:
            state["conv"] = self.conv.empty(batch, **kw)
        return state

    def _heads(self, v: Tensor) -> Tensor:
        """`[b, T, heads, ...] -> [b, heads, T, ...]`."""
        return v.transpose(1, 2)

    def forward(
        self, x: Tensor, state: dict | None = None, memory: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        """Run the layer over `x` (`[batch, T, d_model]`), continuing from `state`."""
        batch, steps, _ = x.shape
        if state is None:
            state = self.init_state(batch, device=x.device, dtype=x.dtype)
        branch, gate, b_raw, c_raw, dt, a_raw, u, theta = self.in_proj(x).split(self.sizes, -1)
        new_state = {"pos": state["pos"] + steps}
        if self.conv is not None:
            branch, new_state["conv"] = self.conv(branch, state["conv"])
            branch = silu(branch)
        delta = softplus(dt + self.dt_bias)  # [b, T, heads]
        a_t = -softplus(a_raw)  # data-dependent, negative (Remark 1)
        lam = torch.sigmoid(u) if self.trapezoid else torch.ones_like(u)
        angles = delta.unsqueeze(-1) * theta.unsqueeze(2)  # Δ_t θ_t, [b, T, heads, N/2]
        if not self.rotate:
            angles = torch.zeros_like(angles)
        b_in = self.b_norm(b_raw).unsqueeze(2) + self.b_bias  # [b, T, heads, N]
        c_out = self.c_norm(c_raw).unsqueeze(2) + self.c_bias
        heads_x = branch.view(batch, steps, self.heads, self.head_dim)
        run = trapezoidal_parallel if self.mode == "parallel" else trapezoidal_scan
        y, ssm_state = run(
            self._heads(heads_x),
            self._heads(b_in),
            self._heads(c_out),
            self._heads(delta * a_t),
            self._heads(delta),
            self._heads(lam),
            self._heads(angles),
            self.skip,
            state,
        )
        y = y.transpose(1, 2).reshape(batch, steps, self.inner)
        out = self.out_proj(y * silu(gate))
        return out, {**ssm_state, **new_state}


# --- registrations -------------------------------------------------------------------------------

_DT_NOTE = (
    "Mamba §3.6 (arXiv:2312.00752v2) states Δ starts in Uniform([0.001, 0.1]), but only inside an "
    "equation the quote gate cannot match as prose, so it is recorded as ours"
)
_DT_NOTE_M3 = (
    "the Mamba-3 paper does not state Δ's initial range; we reuse the parent Mamba's §3.6 range, "
    "which appears only inside an equation"
)


def _mamba_common() -> tuple:
    return (
        hparams.from_catalogue("expansion", "mamba", "expansion", "E: the SSM runs at E·d_model."),
        hparams.from_lab_source(
            "conv_kernel", "mamba.convKernel", "Width of the causal depthwise convolution."
        ),
        hparams.ours("dt_min", 0.001, "Lower end of Δ's initial range.", _DT_NOTE),
        hparams.ours("dt_max", 0.1, "Upper end of Δ's initial range.", _DT_NOTE),
        hparams.ours(
            "discretization",
            "zoh",
            "How B̄ is formed: Eq 4's zero-order hold, or Euler (B̄ = ΔB) as the code does.",
            "Eq 4 is what the paper states; the released code uses Euler instead",
        ),
        hparams.ours(
            "mode",
            "scan",
            "Recurrent scan, or the all-tokens-at-once cumulative-decay form.",
            "Algorithm 2 says the time-varying SSM is computed by scan only",
        ),
    )


register(
    MixerSpec(
        name="mamba",
        family="recurrent",
        summary=(
            "Replaces linear attention's never-forgetting matrix state with a per-channel diagonal "
            "SSM whose decay Δ and read/write vectors B and C are computed from each token."
        ),
        parent="linear_attention",
        covers="mamba",
        source="catalogue:mamba",
        checked_against=(
            "arXiv:2312.00752v2 Eq 1-2, Eq 4 (ZOH), Algorithm 2 (S6), §3.2 (s_B, s_C, s_Δ, "
            "τ_Δ = softplus), §3.4 (block: expansion E, conv, SiLU gate), §3.6 (S4D-Real "
            "A_n = -(n+1), Δ init, low-rank s_Δ); conv width, dt rank and D from "
            "state-spaces/mamba at e9594ce"
        ),
        factory=Mamba,
        lab=(
            hparams.ours("d_model", 32, "Model width.", "lab scale, small enough for a laptop CPU"),
            hparams.ours(
                "state_size", 8, "N, state numbers per channel.", "lab scale, half the paper's N"
            ),
            hparams.ours(
                "dt_rank",
                2,
                "R, width of the low-rank Δ projection.",
                "the code's ceil(d_model / 16) rule gives 2 at this width",
            ),
            hparams.ours(
                "context",
                256,
                "Longest sequence used; recorded, not enforced.",
                "the longest sequence the generic lab tests feed",
            ),
            *_mamba_common(),
        ),
        paper=(
            hparams.from_lab_source("d_model", "mamba.dModel", "Model width D in §E.5."),
            hparams.from_catalogue("state_size", "mamba", "stateSize", "N, state per channel."),
            hparams.ours(
                "dt_rank",
                64,
                "R, width of the low-rank Δ projection.",
                "not stated in the paper; the official code's ceil(d_model / 16) rule",
            ),
            hparams.from_catalogue("context", "mamba", "context", "Training context length."),
            *_mamba_common(),
        ),
        state_growth="constant",
    )
)


def _mamba3_common() -> tuple:
    return (
        hparams.from_catalogue(
            "expansion", "mamba3", "expansion", "The layer runs at expansion·d_model."
        ),
        hparams.from_lab_source(
            "conv_kernel", "mamba.convKernel", "Width of the optional short convolution."
        ),
        hparams.ours("dt_min", 0.001, "Lower end of Δ's initial range.", _DT_NOTE_M3),
        hparams.ours("dt_max", 0.1, "Upper end of Δ's initial range.", _DT_NOTE_M3),
        hparams.ours(
            "use_conv",
            False,
            "Whether the short causal convolution is kept.",
            "off, because Table 5a finds bias plus trapezoid make it redundant",
        ),
        hparams.ours(
            "trapezoid",
            True,
            "λ_t = σ(u_t) (Eq 5) when on; λ_t = 1, Mamba-2's rule, when off.",
            "on, because the exponential-trapezoidal rule is the Mamba-3 default",
        ),
        hparams.ours(
            "rotate",
            True,
            "Rotate the state by Δ_t θ_t (complex state) when on.",
            "on, because the complex state is the Mamba-3 default",
        ),
        hparams.ours(
            "mode",
            "parallel",
            "RoPE-trick parallel form, or the direct-rotation scan.",
            "the layer is computed in the masked parallel form",
        ),
    )


register(
    MixerSpec(
        name="mamba3",
        family="recurrent",
        summary=(
            "Keeps Mamba's selective SSM but with a scalar data-dependent decay per head, the "
            "exponential-trapezoidal update, a rotating (complex) state, BCNorm and B/C biases, "
            "and no short convolution by default."
        ),
        parent="mamba",
        covers="mamba3",
        source="catalogue:mamba3",
        checked_against=(
            "arXiv:2603.15569v1 §2.2 Eq 1 and Remark 1 (scalar, data-dependent A_t), Proposition 1 "
            "Eq 5 (exponential-trapezoidal), Appendix A.3 (λ_t = σ(u_t)), Proposition 2 Eq 9 "
            "(rotating state, used by the scan), Proposition 4 Eq 11 (RoPE trick, used by the "
            "parallel form), §3.4 (BCNorm, B/C biases, conv removed), Appendix F (biases start at "
            "one); SISO only, MIMO (Appendix C) not implemented"
        ),
        factory=Mamba3,
        lab=(
            hparams.ours("d_model", 32, "Model width.", "lab scale, small enough for a laptop CPU"),
            hparams.ours(
                "state_size",
                8,
                "N per head; even, since the rotation acts on pairs.",
                "lab scale, four rotating pairs per head",
            ),
            hparams.ours(
                "head_dim",
                16,
                "P, input width per head.",
                "lab scale, giving four heads at twice d_model",
            ),
            hparams.ours(
                "context",
                256,
                "Longest sequence used; recorded, not enforced.",
                "the longest sequence the generic lab tests feed",
            ),
            *_mamba3_common(),
        ),
        paper=(
            hparams.ours(
                "d_model",
                1024,
                "Model width.",
                "the Mamba-3 paper states no width we can quote; chosen so heads divide evenly",
            ),
            hparams.from_catalogue("state_size", "mamba3", "stateSize", "N per head."),
            hparams.from_catalogue("head_dim", "mamba3", "headDim", "P, input width per head."),
            hparams.from_catalogue("context", "mamba3", "context", "Training context length."),
            *_mamba3_common(),
        ),
        state_growth="constant",
    )
)
