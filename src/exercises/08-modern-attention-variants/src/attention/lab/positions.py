"""The positions family: how attention learns *where* a token is.

**The problem.** Softmax attention on its own does not see word order. It scores every query
against every key and adds up the values, and that sum is the same whichever order the keys come
in. A causal mask tells a token which tokens came before it, but not how far back each one was.
Every variant here puts position back in, and they differ in where they put it and what it costs
when a model has to read further than it was trained on.

**What changes in the code, variant by variant** (each is one small hook on the same causal
multi-head attention with a KV cache, `_PositionAttention`):

| variant | where position enters | the change from its parent |
| --- | --- | --- |
| `sinusoidal` | added to the input, before the q/k/v projections | a fixed sine/cosine table |
| `learned_absolute` | added to the input | the table is trained, and has a fixed size |
| `rope` | q and k are rotated by an angle that grows with position | no addition; a rotation |
| `alibi` | a penalty added to the scores, growing with distance | no rotation; one slope per head |
| `ntk_aware` | RoPE with a larger base | every frequency slows, the fastest barely |
| `yarn` | RoPE, per-frequency interpolation and a temperature | slow frequencies are squeezed |
| `drope` | RoPE that can be switched off after training | off means no position at all |
| `hd_rope` | RoPE whose rotation mixes four dimensions at once | 4D blocks instead of 2D pairs |

**The trade-off.** Adding a table to the input is cheap but ties the model to positions it has
seen: a learned table simply has no row for position 2,000 if it was built with 1,024. Rotating
q and k makes the score depend only on the *distance* between two tokens, which is why RoPE became
the default; but a rotation the model never saw during training still confuses it, and
`ntk_aware`, `yarn` and `drope` are three answers to that. ALiBi avoids the rotation altogether and
pays with a fixed bias towards nearby tokens.

**What the lab keeps between tokens.** A KV cache (keys already rotated, where the variant
rotates) plus `pos`, the number of tokens seen. Positions are absolute: token `t` is always at
position `pos + t`, so decoding one token at a time gives the same result as one call over the
whole sequence. `init_state(..., start=n)` begins counting at `n` with an empty cache, which is how
the tests shift every position by the same amount.

**Where each formula comes from** is in each spec's `checked_against`, and the open questions are
written next to the code they affect: YaRN's Eq 17 writes the ratio with the new base, and HD-RoPE's
Algorithm 2 does not match its own Eq 13 (see `apply_hd_rope`).
"""

import math

import torch
from torch import Tensor, nn

from attention.lab import ops
from attention.lab.base import Mixer, MixerSpec
from attention.lab.hparams import from_catalogue, from_lab_source, ours
from attention.lab.registry import register
from attention.lab.sources import LabSource, add

# --- the shared attention ------------------------------------------------------------------------


class _PositionAttention(Mixer):
    """Causal multi-head attention with a KV cache, and four hooks where position can enter.

    The hooks, each an identity here, are what a position variant overrides:

    - `embed_positions(x, positions)` — before the projections (sinusoidal, learned).
    - `rotate(x, positions)` — on q and k after the projections (the RoPE family).
    - `score_bias(q_positions, k_positions)` — added to the scores (ALiBi).
    - `score_scale()` — the multiplier on q·k; `None` means `1/sqrt(head_dim)`.

    Args:
        d_model: Width of the token vectors.
        heads: Number of attention heads; `d_model` must divide evenly.
        head_dim: Optional; when given it must equal `d_model // heads`. It exists so a
            paper-scale spec can state the head width it was read with, and have it checked.
    """

    def __init__(self, d_model: int, heads: int, head_dim: int | None = None) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model} is not divisible by {heads} heads")
        if head_dim is not None and head_dim != d_model // heads:
            raise ValueError(f"head_dim {head_dim} != d_model // heads = {d_model // heads}")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    # hooks ----------------------------------------------------------------------------------------

    def embed_positions(self, x: Tensor, positions: Tensor) -> Tensor:
        """Add position information to the input; the parent adds none."""
        return x

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate `[b, h, t, d]` by position; the parent does not rotate."""
        return x

    def score_bias(self, q_positions: Tensor, k_positions: Tensor) -> Tensor | None:
        """An additive `[heads, queries, keys]` bias on the scores; the parent adds none."""
        return None

    def score_scale(self) -> float | None:
        """The multiplier on q·k; `None` is the usual `1/sqrt(head_dim)`."""
        return None

    # the contract ---------------------------------------------------------------------------------

    def init_state(
        self,
        batch: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        start: int = 0,
    ) -> dict:
        """An empty KV cache, with the position counter at `start` (0 unless shifting)."""
        dtype = dtype or self.q_proj.weight.dtype
        empty = torch.zeros(batch, self.heads, 0, self.head_dim, device=device, dtype=dtype)
        return {"k": empty, "v": empty.clone(), "pos": int(start)}

    def forward(
        self, x: Tensor, state: dict | None = None, memory: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        """Attend causally over the cache plus `x`; return the output and the grown cache."""
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, device=x.device, dtype=x.dtype)
        start = state["pos"]
        positions = torch.arange(start, start + tokens, device=x.device)
        x = self.embed_positions(x, positions)
        q = self.rotate(ops.split_heads(self.q_proj(x), self.heads), positions)
        k = self.rotate(ops.split_heads(self.k_proj(x), self.heads), positions)
        v = ops.split_heads(self.v_proj(x), self.heads)
        k = torch.cat([state["k"], k], dim=2)
        v = torch.cat([state["v"], v], dim=2)
        cached = state["k"].shape[2]
        keys = k.shape[2]
        # The cache holds the `keys` most recent tokens, the last `tokens` of which are new.
        k_positions = torch.arange(start + tokens - keys, start + tokens, device=x.device)
        allowed = ops.causal_mask(tokens, keys, offset=cached, device=x.device)
        bias = self.score_bias(positions, k_positions)
        if bias is not None:
            bias = bias.to(device=q.device, dtype=q.dtype)
        y, _ = ops.attend(q, k, v, allowed=allowed, bias=bias, scale=self.score_scale())
        new_state = {"k": k, "v": v, "pos": start + tokens}
        return self.o_proj(ops.merge_heads(y)), new_state


# --- 1. sinusoidal --------------------------------------------------------------------------------


def sinusoidal_table(positions: Tensor, d_model: int, base: float) -> Tensor:
    """`PE(pos, 2i) = sin(pos / base^(2i/d))`, `PE(pos, 2i+1) = cos(...)`: `[tokens, d_model]`.

    Computed in float64. arXiv:1706.03762v7 §3.5, with `base = 10000` there.
    """
    if d_model % 2:
        raise ValueError(
            f"the sinusoidal table pairs sin and cos, so d_model must be even: {d_model}"
        )
    # Computed in float64 on the CPU and moved by the caller: Apple's MPS backend has no float64,
    # and the lab runs on a laptop GPU. The notebook found this; the CPU-only tests could not.
    where = ops.float64_device(positions)
    positions = positions.detach().to(where)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float64, device=where)
    angles = positions.to(torch.float64)[:, None] / base ** (two_i / d_model)[None, :]
    table = torch.empty(len(positions), d_model, dtype=torch.float64, device=where)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


class SinusoidalAttention(_PositionAttention):
    """Adds the fixed sine/cosine table to the input before q, k and v are projected.

    Nothing is learned and nothing is limited: any position has a row. The paper chose it because
    `PE(pos + k)` is a fixed linear function of `PE(pos)` for any offset `k`, which
    `tests/test_attention_lab_positions.py` checks numerically.
    """

    def __init__(self, d_model: int, heads: int, base: float, head_dim: int | None = None) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, head_dim)
        self.base = base

    def embed_positions(self, x: Tensor, positions: Tensor) -> Tensor:
        """`x + PE(positions)`."""
        table = sinusoidal_table(positions, self.d_model, self.base)
        return x + table.to(device=x.device, dtype=x.dtype)


# --- 2. learned absolute --------------------------------------------------------------------------


class LearnedAbsoluteAttention(_PositionAttention):
    """Adds a trained row per position, `e_j = w_j + p_j` (arXiv:1705.03122v3 §3.1).

    The table has `max_positions` rows and no row beyond them: asking for position
    `max_positions` is an error, which is the length limit this design carries. In the source the
    embeddings feed a convolutional model with a single dot-product attention (§3.3); the lab puts
    the same table in front of its multi-head attention so the position schemes can be compared.

    Args:
        d_model: Width of the token vectors and of each table row.
        heads: Number of attention heads.
        max_positions: Rows in the table.
        init_std: Standard deviation of the normal initialisation of the table.
        head_dim: Optional check on `d_model // heads`.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        max_positions: int,
        init_std: float,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, head_dim)
        self.max_positions = max_positions
        self.table = nn.Embedding(max_positions, d_model)
        nn.init.normal_(self.table.weight, mean=0.0, std=init_std)

    def embed_positions(self, x: Tensor, positions: Tensor) -> Tensor:
        """`x + table[positions]`; refuses a position the table has no row for."""
        if len(positions) and int(positions[-1]) >= self.max_positions:
            raise ValueError(
                f"position {int(positions[-1])} is beyond the {self.max_positions} rows a "
                "learned table has; this is the length limit of learned positions"
            )
        return x + self.table(positions).to(x.dtype)


# --- 3. RoPE --------------------------------------------------------------------------------------


class RopeAttention(_PositionAttention):
    """Rotates each adjacent pair of q and k dimensions by `position × θ_i` (arXiv:2104.09864v5).

    Eq 14: `f(x_m, m) = R^d_{Θ,m} W x_m`; Eq 15 makes `R` block-diagonal over the pairs
    `(1,2), (3,4), …` with `θ_i = base^(−2(i−1)/d)`; Eq 34 is the efficient form `ops.apply_rope`
    computes. Because every rotation is orthogonal, `q_m·k_n` depends on `n − m` only (Eq 16).

    Args:
        d_model: Width of the token vectors.
        heads: Number of heads; `d_model // heads` must be even.
        base: `b` in `θ_i = b^(−2(i−1)/d)`.
        context: The context length the source trained at. Recorded for the cost tables; the
            layer does not read it.
        head_dim: Optional check on `d_model // heads`.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        base: float,
        context: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, head_dim)
        if self.head_dim % 2:
            raise ValueError(f"RoPE rotates pairs, so head_dim must be even, not {self.head_dim}")
        self.base = base
        self.context = context

    def angles(self, positions: Tensor) -> Tensor:
        """`[tokens, head_dim // 2]` angles `m × θ_i`."""
        return ops.rope_angles(positions, self.head_dim, self.base)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate every adjacent pair by its angle (Eq 34)."""
        return ops.apply_rope(x, self.angles(positions))


# --- 4. ALiBi -------------------------------------------------------------------------------------

ALIBI_CODE = (
    "https://github.com/ofirpress/attention_with_linear_biases/blob/"
    "4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742-L752"
)


def alibi_slopes(heads: int) -> list[float]:
    """One slope per head.

    For a power-of-two head count this is the paper's rule (arXiv:2108.12409v2 §3): the geometric
    sequence that starts at `2^(−8/n)` and uses that same value as its ratio, so eight heads get
    `1/2, 1/4, …, 1/256`.

    **For any other head count the paper states no rule.** This follows the authors' official code
    (MIT licence) at `ALIBI_CODE`, whose comment says the paper only trained power-of-two head
    counts: take the slopes for the largest power of two below `n`, then fill the rest with every
    other slope of the sequence for twice that power. That is the code's rule, not the paper's.
    """
    if heads < 1:
        raise ValueError(f"need at least one head, not {heads}")

    def power_of_two(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]

    if heads & (heads - 1) == 0:
        return power_of_two(heads)
    closest = 2 ** math.floor(math.log2(heads))
    return power_of_two(closest) + alibi_slopes(2 * closest)[0::2][: heads - closest]


class AlibiAttention(_PositionAttention):
    """No rotation and no table: the score for query `i` and key `j` gets `−m·(i − j)` added.

    arXiv:2108.12409v2 §3: `softmax(q_i K^T + m·[−(i−1), …, −2, −1, 0])`, with `m` a slope fixed
    per head before training. Footnote 10: the bias is *not* multiplied by the `1/sqrt(d_k)`
    scale, which is why it goes into `ops.attend`'s `bias`, applied after the scale.

    Args:
        d_model: Width of the token vectors.
        heads: Number of heads; each gets its own slope from `alibi_slopes`.
        slope_rule: Only the paper's geometric sequence is implemented.
        trained_length: Length the source trained on. Recorded, not read by the layer.
        extended_length: Length the source evaluated at. Recorded, not read by the layer.
        head_dim: Optional check on `d_model // heads`.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        slope_rule: str = "geometric sequence",
        trained_length: int | None = None,
        extended_length: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, head_dim)
        if slope_rule != "geometric sequence":
            raise ValueError(
                f"only the paper's geometric slope sequence is implemented: {slope_rule}"
            )
        self.trained_length = trained_length
        self.extended_length = extended_length
        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
        self.slopes = torch.tensor(alibi_slopes(heads), dtype=torch.float64)

    def score_bias(self, q_positions: Tensor, k_positions: Tensor) -> Tensor:
        """`[heads, queries, keys]` of `−m_h · (i − j)`."""
        where = ops.float64_device(q_positions)
        distance = (q_positions[:, None] - k_positions[None, :]).to(where, torch.float64)
        return -self.slopes.to(where)[:, None, None] * distance[None]


# --- 5. NTK-aware ---------------------------------------------------------------------------------


def ntk_base(base: float, scale: float, dim: int) -> float:
    """`b' = b · s^(|D| / (|D| − 2))` (arXiv:2309.00071v2 Definition 1, Eq 16).

    Chosen (App. A.1) so the lowest frequency, at the last pair, is divided by exactly `s` — as
    position interpolation would — while the highest, at the first pair, is unchanged.
    """
    if dim <= 2:
        raise ValueError(f"NTK-aware scaling needs a rotated width above 2, not {dim}")
    return base * scale ** (dim / (dim - 2))


class NtkAwareAttention(RopeAttention):
    """RoPE with a larger base, so long positions turn slowly without squeezing short ones.

    `g(m) = m` (Eq 14) and `h(θ_d) = b'^(−2d/|D|)` (Eq 15) with `b'` from `ntk_base`. With
    `scale = 1` the base is unchanged and this is exactly `rope`.

    Args:
        d_model, heads, base, context, head_dim: As for `RopeAttention`.
        scale: `s`, how many times longer than trained the model is asked to read.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        base: float,
        scale: float,
        context: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, base, context=context, head_dim=head_dim)
        self.scale = scale
        self.scaled_base = ntk_base(base, scale, self.head_dim)

    def angles(self, positions: Tensor) -> Tensor:
        """RoPE's angles with `b'` in place of `b`."""
        return ops.rope_angles(positions, self.head_dim, self.scaled_base)


# --- 6. YaRN --------------------------------------------------------------------------------------


def yarn_ramp(r: Tensor, alpha: float, beta: float) -> Tensor:
    """`γ(r)`: 0 below `α`, 1 above `β`, linear between (arXiv:2309.00071v2 Eq 18)."""
    return torch.clamp((r - alpha) / (beta - alpha), 0.0, 1.0)


def yarn_frequencies(
    dim: int, base: float, scale: float, original_length: int, alpha: float, beta: float
) -> Tensor:
    """NTK-by-parts frequencies `h(θ_d)` for `d = 0 … |D|/2 − 1` (Eq 13, 17, 18, 20), float64.

    `h(θ_d) = (1 − γ(r(d))) θ_d / s + γ(r(d)) θ_d`, with `r(d) = L / λ_d` and `λ_d = 2π / θ_d`.

    **Open question.** The paper's Eq 17 writes `λ_d` with the NTK base `b'`, although YaRN never
    changes the base and Eq 13 defines `λ_d` with `b`. The lab uses the original base `b`, which
    is the reading under which Eq 13 and Eq 17 agree; it has not been checked against the
    authors' code in this run.
    """
    d = torch.arange(dim // 2, dtype=torch.float64)
    theta = base ** (-2.0 * d / dim)
    wavelength = 2 * math.pi / theta
    gamma = yarn_ramp(original_length / wavelength, alpha, beta)
    return (1 - gamma) * theta / scale + gamma * theta


def yarn_attention_factor(scale: float) -> float:
    """`sqrt(1/t) = 0.1 ln(s) + 1` (Eq 22), the factor q and k are each multiplied by."""
    return 0.1 * math.log(scale) + 1.0


class YarnAttention(NtkAwareAttention):
    """NTK-by-parts interpolation plus a softmax temperature (Definition 3).

    Frequencies whose wavelength is short next to the trained length `L` are left alone,
    frequencies whose wavelength is long are divided by `s` (as position interpolation would), and
    the ones in between are blended by the ramp `γ`. Both q and k are then multiplied by
    `sqrt(1/t)` so the scores are divided by `t` (Eq 21) without touching the attention code —
    the "length scaling" trick of §3.4. With `scale = 1` every frequency is unchanged and the
    factor is 1, so this is exactly `rope`.

    Args:
        d_model, heads, base, head_dim: As for `RopeAttention`.
        scale: `s = L' / L` (Eq 11).
        original_length: `L`, the context the model was trained at.
        extended_length: `L'`; must equal `s × L`.
        alpha, beta: The ramp's bounds (Eq 18).
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        base: float,
        scale: float,
        original_length: int,
        extended_length: int,
        alpha: float,
        beta: float,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, base, scale, context=original_length, head_dim=head_dim)
        if extended_length != scale * original_length:
            raise ValueError(f"s = L'/L (Eq 11): {extended_length} != {scale} x {original_length}")
        self.original_length = original_length
        self.extended_length = extended_length
        self.alpha = alpha
        self.beta = beta
        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
        self.frequencies = yarn_frequencies(
            self.head_dim, base, scale, original_length, alpha, beta
        )
        self.attention_factor = yarn_attention_factor(scale)

    def angles(self, positions: Tensor) -> Tensor:
        """`m × h(θ_d)`."""
        where = ops.float64_device(positions)
        return positions.to(where, torch.float64)[:, None] * self.frequencies.to(where)[None, :]

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate, then multiply by `sqrt(1/t)`."""
        return ops.apply_rope(x, self.angles(positions)) * self.attention_factor


# --- 7. DroPE -------------------------------------------------------------------------------------


class DropeAttention(RopeAttention):
    """RoPE during pretraining, then no positional embedding at all (arXiv:2512.12167v1 §5).

    DroPE takes a model trained with RoPE, removes the rotation from every layer, and
    recalibrates briefly. `rope_on=True` is the model before the drop and equals `rope`;
    `rope_on=False` is the model after it: attention with no positional information, whose
    output is unchanged if every position is shifted by the same amount, or if the earlier tokens
    are reordered.

    With the rotation off, a logit scale `β = 1 + c · ln(s)` multiplies the scores, where
    `s = C_test / C_train` (App. C.2: "a single scalar logit scale (equivalently, the inverse
    temperature)", fitted as `c = 0.412` for the from-scratch model and `c = 0.103` for
    SmolLM-DroPE). With `extension = 1` it is 1. The paper also adds QK-norm for the SmolLM runs
    and uses grouped KV heads; the lab leaves both out, since neither is about position.

    Args:
        d_model, heads, base, head_dim: As for `RopeAttention`.
        rope_on: Whether the rotation is applied.
        extension: `s`, the context extension factor the logit scale is set for.
        logit_coefficient: `c` in `β = 1 + c ln(s)`.
        trained_length: `C_train`. Recorded, not read by the layer.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        base: float,
        rope_on: bool,
        extension: float,
        logit_coefficient: float,
        trained_length: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, base, context=trained_length, head_dim=head_dim)
        self.rope_on = rope_on
        self.extension = extension
        self.logit_coefficient = logit_coefficient
        self.trained_length = trained_length
        self.logit_scale = 1.0 + logit_coefficient * math.log(extension)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """RoPE's rotation, or nothing once it has been dropped."""
        return super().rotate(x, positions) if self.rope_on else x

    def score_scale(self) -> float | None:
        """`β / sqrt(head_dim)` without the rotation; the usual scale with it."""
        if self.rope_on:
            return None
        return self.logit_scale / math.sqrt(self.head_dim)


# --- 8. HD-RoPE -----------------------------------------------------------------------------------


def apply_hd_rope(x: Tensor, angles: Tensor) -> Tensor:
    """HD-RoPE's 4D rotation, **Algorithm 2 as printed** in arXiv:2608.29715v1.

    `x` is split into four equal chunks `x1 … x4` (a chunk split, not interleaved), then

        t1 = (−x2,  x1,  x4, −x3)
        t2 = ( x3,  x4,  x2, −x1)
        t3 = ( x4, −x3, −x1, −x2)
        out = x·cos + t1·(1/3)·sin + t2·(2/3)·sin + t3·(2/3)·sin

    Each 4D block `(x1[i], x2[i], x3[i], x4[i])` turns by one angle `angles[:, i]`, because Eq 16
    uses the same θ in both 2×2 blocks of a 4D rotation. Algorithm 2 does not say how `freqs` is
    laid out; one shared angle per block is the only layout consistent with Eq 16.

    What we computed (not claims of the paper; `tests/test_attention_lab_positions.py` checks the
    first two): the printed transform is `I cos + G sin` with `G` skew-symmetric and `G² = −I`, so
    it is orthogonal with determinant 1 at every position, and `R(m)ᵀ R(n) = R(n − m)`, which is
    the relative-position property. It is **not** `Q₄ᵀ R Q₄` with the printed `C₄` (Eq 13, 19,
    21): that product, with one θ per block, is again a rotation in two fixed planes with no
    mixing. It **is** `Q₄ R Q₄ᵀ`. We also found `det(Q₄) = −1`, so `Q₄` is not in `SO(4)` as
    Eq 13 states. Which of the two the authors ran is an open question; no code is published.

    Args:
        x: `[..., tokens, d]` with `d` divisible by 4.
        angles: `[tokens, d // 4]`.
    """
    d = x.shape[-1]
    if d % 4:
        raise ValueError(f"HD-RoPE rotates 4D blocks, so the width must divide by 4, not {d}")
    cos = torch.cos(angles).to(device=x.device, dtype=x.dtype)
    sin = torch.sin(angles).to(device=x.device, dtype=x.dtype)
    x1, x2, x3, x4 = x.chunk(4, dim=-1)
    t1 = torch.cat([-x2, x1, x4, -x3], dim=-1)
    t2 = torch.cat([x3, x4, x2, -x1], dim=-1)
    t3 = torch.cat([x4, -x3, -x1, -x2], dim=-1)
    cos4 = torch.cat([cos] * 4, dim=-1)
    sin4 = torch.cat([sin] * 4, dim=-1)
    return x * cos4 + t1 * (sin4 / 3) + t2 * (2 * sin4 / 3) + t3 * (2 * sin4 / 3)


class HdRopeAttention(RopeAttention):
    """RoPE whose rotation mixes four dimensions per frequency instead of two.

    **Frequencies are our choice.** The paper divides the head into `d/4` blocks with one θ each
    (Eq 21) but does not state their values. The lab uses `θ_i = base^(−4i/d)`, `i = 0 … d/4 − 1`
    — RoPE's schedule spread over `d/4` blocks so the fastest is 1 and the slowest is near
    `1/base`, as in RoPE.

    Args:
        d_model, heads, base, head_dim: As for `RopeAttention`; `head_dim` must divide by 4.
        extended_length: The long-context length the source continued pretraining at. Recorded,
            not read by the layer.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        base: float,
        extended_length: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        """Build the layer; the arguments are described on the class."""
        super().__init__(d_model, heads, base, head_dim=head_dim)
        if self.head_dim % 4:
            raise ValueError(f"HD-RoPE needs head_dim divisible by 4, not {self.head_dim}")
        self.extended_length = extended_length

    def angles(self, positions: Tensor) -> Tensor:
        """`[tokens, head_dim // 4]` angles, one per 4D block."""
        return ops.rope_angles(positions, self.head_dim // 2, self.base)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Algorithm 2 as printed."""
        return apply_hd_rope(x, self.angles(positions))


# --- sources the catalogue does not carry ---------------------------------------------------------

_ALIBI = "https://arxiv.org/html/2108.12409v2"
_ALIBI_TITLE = (
    "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
)
_ROPE = "https://arxiv.org/html/2104.09864v5"
_ROPE_TITLE = "RoFormer: Enhanced Transformer with Rotary Position Embedding"
_YARN = "https://arxiv.org/html/2309.00071v2"
_YARN_TITLE = "YaRN: Efficient Context Window Extension of Large Language Models"
_DROPE = "https://arxiv.org/html/2512.12167v1"
_DROPE_TITLE = "Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings"
_HD = "https://arxiv.org/html/2608.29715v1"
_HD_TITLE = "Higher-Dimensional Rotary Position Embedding"

_SLOPES_QUOTE = (
    "In general, for n heads, our set of slopes is the geometric sequence that starts at"
)
_ALPHA_BETA_QUOTE = "for the Llama family of models, good values for α and β are α = 1 and β = 32"
for _source in (
    LabSource(
        "alibi.slope_rule",
        "geometric sequence",
        _SLOPES_QUOTE,
        "S3 Attention with Linear Biases (ALiBi), arXiv:2108.12409v2",
        _ALIBI,
        _ALIBI_TITLE,
    ),
    LabSource(
        "alibi.d_model",
        1024,
        "The model has 16 transformer layers of dimension 1024 , with 8 heads",
        "S2.1 Background and Experimental Setup (WikiText-103 model), arXiv:2108.12409v2",
        _ALIBI,
        _ALIBI_TITLE,
        "dimensions",
    ),
    LabSource(
        "rope.d_model",
        768,
        "12 layer char-based PerFormer with 768 dimensions and 12 heads",
        "S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5",
        _ROPE,
        _ROPE_TITLE,
        "dimensions",
    ),
    LabSource(
        "rope.heads",
        12,
        "12 layer char-based PerFormer with 768 dimensions and 12 heads",
        "S4.4.1 Implementation details (Performer with RoPE, Enwik8), arXiv:2104.09864v5",
        _ROPE,
        _ROPE_TITLE,
        "heads",
    ),
    LabSource(
        "yarn.alpha",
        1,
        _ALPHA_BETA_QUOTE,
        "S3.2 Definition 2 (NTK-by-parts), arXiv:2309.00071v2",
        _YARN,
        _YARN_TITLE,
    ),
    LabSource(
        "yarn.beta",
        32,
        _ALPHA_BETA_QUOTE,
        "S3.2 Definition 2 (NTK-by-parts), arXiv:2309.00071v2",
        _YARN,
        _YARN_TITLE,
    ),
    LabSource(
        "yarn.original_length",
        4096,
        "Since the original maximal context length of Llama 2 is 4096",
        "B.3 Dynamic scaling on models without any fine-tuning, arXiv:2309.00071v2",
        _YARN,
        _YARN_TITLE,
        "tokens",
    ),
    LabSource(
        "drope.logit_coefficient",
        0.103,
        "for SmolLM-DroPE the optimal scale is β ⋆ = 1 + 0.103",
        "C.2 Evaluation (Long-context evaluations), arXiv:2512.12167v1",
        _DROPE,
        _DROPE_TITLE,
    ),
    LabSource(
        "drope.d_model",
        960,
        "Hidden size 896 960",
        "Table 4, SmolLM column (Appendix C.1), arXiv:2512.12167v1",
        _DROPE,
        _DROPE_TITLE,
        "dimensions",
    ),
    LabSource(
        "drope.heads",
        15,
        "Number of attention heads 14 15",
        "Table 4, SmolLM column (Appendix C.1), arXiv:2512.12167v1",
        _DROPE,
        _DROPE_TITLE,
        "heads",
    ),
    LabSource(
        "hd_rope.d_model",
        2048,
        "Hidden Dim. 1536 2048",
        "Table 6, 1.3B column (A.3.1 Model Architecture), arXiv:2608.29715v1",
        _HD,
        _HD_TITLE,
        "dimensions",
    ),
    LabSource(
        "hd_rope.heads",
        32,
        "Heads 24 32 KV Heads",
        "Table 6, 1.3B column (A.3.1 Model Architecture), arXiv:2608.29715v1",
        _HD,
        _HD_TITLE,
        "heads",
    ),
    LabSource(
        "hd_rope.base",
        10000,
        "including θ = 10 k and θ = 500 k",
        "S4.1 Experiment Setup (Baselines and evaluations), arXiv:2608.29715v1",
        _HD,
        _HD_TITLE,
    ),
):
    add(_source)

# --- registrations --------------------------------------------------------------------------------

_LAB_WIDTH = ours(
    "d_model", 32, "Width of the token vectors.", "lab scale, small enough to run on a laptop"
)
_LAB_HEADS = ours(
    "heads",
    2,
    "Number of attention heads.",
    "lab scale; gives a head width of 16, even and divisible by 4",
)
register(
    MixerSpec(
        name="sinusoidal",
        family="position",
        summary="Adds a fixed sine/cosine table to the input before q, k and v are projected.",
        parent="standard_attention",
        covers="sinusoidal",
        source="catalogue:sinusoidal",
        checked_against=(
            "§3.5 Positional Encoding (PE(pos,2i) = sin, PE(pos,2i+1) = cos), arXiv:1706.03762v7"
        ),
        factory=SinusoidalAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            from_catalogue("base", "sinusoidal", "base", "The longest wavelength is base × 2π."),
        ),
        paper=(
            from_catalogue("d_model", "sinusoidal", "dims", "Width of the token vectors."),
            from_catalogue("heads", "sinusoidal", "heads", "Number of attention heads."),
            from_catalogue("head_dim", "sinusoidal", "headDim", "Width of each head."),
            from_catalogue("base", "sinusoidal", "base", "The longest wavelength is base × 2π."),
        ),
    )
)

register(
    MixerSpec(
        name="learned_absolute",
        family="position",
        summary="Trains the table instead, at a fixed size, so no position beyond it exists.",
        parent="sinusoidal",
        covers="learned_absolute",
        source="catalogue:learned_absolute",
        checked_against=(
            "§3.1 Position Embeddings (e = w + p), arXiv:1705.03122v3 (PDF; no HTML version exists)"
        ),
        factory=LearnedAbsoluteAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            ours(
                "max_positions",
                512,
                "Rows in the position table.",
                "lab scale; covers the 256 tokens the generic memory test reads",
            ),
            ours(
                "init_std",
                0.1,
                "Standard deviation of the table's normal initialisation.",
                "the source's §3.5 says 0.1 for all embeddings, read from the PDF by eye; "
                "the quote verifier cannot read a PDF, so it is recorded as ours",
            ),
        ),
        paper=(
            from_catalogue("d_model", "learned_absolute", "dims", "Width of the embeddings."),
            ours(
                "heads",
                1,
                "Number of attention heads.",
                "the source's §3.3 attention is one dot product per decoder layer, "
                "read from the PDF",
            ),
            ours(
                "max_positions",
                1024,
                "Rows in the position table.",
                "not stated in the source; a table size we chose for the paper-scale shape",
            ),
            ours(
                "init_std",
                0.1,
                "Standard deviation of the table's normal initialisation.",
                "the source's §3.5 says 0.1 for all embeddings, read from the PDF by eye; "
                "the quote verifier cannot read a PDF, so it is recorded as ours",
            ),
        ),
    )
)

register(
    MixerSpec(
        name="rope",
        family="position",
        summary="Rotates q and k by a position-dependent angle instead of adding to the input.",
        parent="sinusoidal",
        covers="rope",
        source="catalogue:rope",
        checked_against=(
            "Eq 14, Eq 15 (adjacent pairs) and the efficient form Eq 34, arXiv:2104.09864v5"
        ),
        factory=RopeAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            from_catalogue("base", "rope", "base", "b in θ_i = b^(−2(i−1)/d)."),
        ),
        paper=(
            from_lab_source("d_model", "rope.d_model", "Width of the token vectors."),
            from_lab_source("heads", "rope.heads", "Number of attention heads."),
            from_catalogue("base", "rope", "base", "b in θ_i = b^(−2(i−1)/d)."),
            from_catalogue(
                "context", "rope", "context", "Context length of the source's long run."
            ),
        ),
    )
)

register(
    MixerSpec(
        name="alibi",
        family="position",
        summary="Drops the rotation; adds a per-head score penalty proportional to distance.",
        parent="rope",
        covers="alibi",
        source="catalogue:alibi",
        checked_against=(
            "§3 bias softmax(q_i K^T + m·[−(i−1), …, 0]), the slope rule, and footnote 10, "
            f"arXiv:2108.12409v2; non-power-of-two slopes from the official code {ALIBI_CODE}"
        ),
        factory=AlibiAttention,
        lab=(
            _LAB_WIDTH,
            ours(
                "heads",
                4,
                "Number of attention heads, one slope each.",
                "lab scale; a power of two, so the paper's own rule applies",
            ),
            from_lab_source("slope_rule", "alibi.slope_rule", "How each head's slope is set."),
        ),
        paper=(
            from_lab_source("d_model", "alibi.d_model", "Width of the token vectors."),
            from_catalogue("heads", "alibi", "heads", "Number of attention heads, one slope each."),
            from_lab_source("slope_rule", "alibi.slope_rule", "How each head's slope is set."),
            from_catalogue("trained_length", "alibi", "trainedLength", "Length trained on."),
            from_catalogue("extended_length", "alibi", "extendedLength", "Length evaluated at."),
        ),
    )
)

register(
    MixerSpec(
        name="ntk_aware",
        family="position",
        summary="Raises RoPE's base so the slowest frequency is divided by s and the fastest kept.",
        parent="rope",
        covers="ntk_aware",
        source="catalogue:ntk_aware",
        checked_against="Definition 1, Eq 14–16, and App. A.1, arXiv:2309.00071v2",
        factory=NtkAwareAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            from_catalogue("base", "ntk_aware", "base", "The original RoPE base b."),
            ours(
                "scale",
                4,
                "s, the context extension factor.",
                "lab scale; a modest extension that changes every frequency",
            ),
        ),
        paper=(
            # Neither NTK-aware scaling nor YaRN states a model width. The width shown is the
            # RoPE paper's own (verified), because these schemes change only RoPE's frequencies.
            # An earlier draft used a width recalled from memory; that is not allowed here.
            from_lab_source("d_model", "rope.d_model", "Width of the token vectors (RoPE paper)."),
            from_lab_source("heads", "rope.heads", "Number of attention heads (RoPE paper)."),
            from_catalogue("base", "ntk_aware", "base", "The original RoPE base b."),
            ours(
                "scale",
                16,
                "s, the context extension factor.",
                "not stated for NTK-aware itself; set to YaRN's s for comparison",
            ),
        ),
    )
)

register(
    MixerSpec(
        name="yarn",
        family="position",
        summary="Interpolates only slow frequencies (a ramp from α to β) and sharpens the softmax.",
        parent="ntk_aware",
        covers="yarn",
        source="catalogue:yarn",
        checked_against=(
            "Eq 11, 13, 17–20 (Definition 2) and Eq 21–22 (Definition 3), arXiv:2309.00071v2; "
            "r(d) uses the original base b although Eq 17 prints b'"
        ),
        factory=YarnAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            from_catalogue("base", "yarn", "base", "The original RoPE base b."),
            ours(
                "scale",
                4,
                "s = L'/L.",
                "lab scale; a modest extension that changes every frequency",
            ),
            ours(
                "original_length",
                16,
                "L, the trained context.",
                "lab scale, so the ramp falls inside a 16-wide head",
            ),
            ours("extended_length", 64, "L' = s × L.", "follows from the lab's s and L by Eq 11"),
            from_lab_source(
                "alpha", "yarn.alpha", "Below this ratio r, a frequency is fully interpolated."
            ),
            from_lab_source("beta", "yarn.beta", "Above this ratio r, a frequency is left alone."),
        ),
        paper=(
            # Neither NTK-aware scaling nor YaRN states a model width. The width shown is the
            # RoPE paper's own (verified), because these schemes change only RoPE's frequencies.
            # An earlier draft used a width recalled from memory; that is not allowed here.
            from_lab_source("d_model", "rope.d_model", "Width of the token vectors (RoPE paper)."),
            from_lab_source("heads", "rope.heads", "Number of attention heads (RoPE paper)."),
            from_catalogue("base", "yarn", "base", "The original RoPE base b."),
            from_catalogue("scale", "yarn", "extension", "s = L'/L."),
            from_lab_source("original_length", "yarn.original_length", "L, the trained context."),
            from_catalogue("extended_length", "yarn", "extendedLength", "L' = s × L."),
            from_lab_source(
                "alpha", "yarn.alpha", "Below this ratio r, a frequency is fully interpolated."
            ),
            from_lab_source("beta", "yarn.beta", "Above this ratio r, a frequency is left alone."),
        ),
    )
)

register(
    MixerSpec(
        name="drope",
        family="position",
        summary="Switches RoPE off after pretraining (no position signal) and adds a logit scale.",
        parent="rope",
        covers="drope",
        source="catalogue:drope",
        checked_against=(
            "§5 (dropping positional embeddings) and App. C.2 (logit scale), arXiv:2512.12167v1"
        ),
        factory=DropeAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            from_catalogue("base", "drope", "base", "The RoPE base before the drop."),
            ours(
                "rope_on",
                False,
                "Whether the rotation is still applied.",
                "the lab shows the model after the drop",
            ),
            from_catalogue("extension", "drope", "extension", "s, the context extension factor."),
            from_lab_source(
                "logit_coefficient", "drope.logit_coefficient", "c in β = 1 + c ln(s)."
            ),
            ours(
                "trained_length",
                16,
                "C_train, the trained context.",
                "lab scale, recorded only; the layer does not read it",
            ),
        ),
        paper=(
            from_lab_source("d_model", "drope.d_model", "Width of the token vectors."),
            from_lab_source("heads", "drope.heads", "Number of query heads."),
            from_catalogue("head_dim", "drope", "dims", "Width of each head."),
            from_catalogue("base", "drope", "base", "The RoPE base before the drop."),
            ours(
                "rope_on",
                False,
                "Whether the rotation is still applied.",
                "DroPE is the model after the drop",
            ),
            from_catalogue("extension", "drope", "extension", "s, the context extension factor."),
            from_lab_source(
                "logit_coefficient", "drope.logit_coefficient", "c in β = 1 + c ln(s)."
            ),
            from_catalogue(
                "trained_length", "drope", "trainedLength", "C_train, the trained context."
            ),
        ),
    )
)

register(
    MixerSpec(
        name="hd_rope",
        family="position",
        summary="Replaces 2D pair rotations with 4D rotations mixing four dims per frequency.",
        parent="rope",
        covers="hd_rope",
        source="catalogue:hd_rope",
        checked_against=(
            "Algorithm 2 as printed (4D, Paley-I), with Eq 16 for one θ per block, "
            "arXiv:2608.29715v1; "
            "Algorithm 2 equals Q4 R Q4^T, not the Q4^T R Q4 of Eq 13 and 21"
        ),
        factory=HdRopeAttention,
        lab=(
            _LAB_WIDTH,
            _LAB_HEADS,
            from_lab_source(
                "base", "hd_rope.base", "The RoPE base the frequencies are spread from."
            ),
        ),
        paper=(
            from_lab_source("d_model", "hd_rope.d_model", "Width of the token vectors."),
            from_lab_source("heads", "hd_rope.heads", "Number of query heads."),
            from_catalogue("head_dim", "hd_rope", "dims", "Width of each head."),
            from_lab_source(
                "base", "hd_rope.base", "The RoPE base the frequencies are spread from."
            ),
            from_catalogue(
                "extended_length",
                "hd_rope",
                "extendedLength",
                "Length of the long-context continued pretraining.",
            ),
        ),
    )
)
