"""The linear family: attention whose memory is one fixed-size matrix instead of a growing cache.

**The problem.** Softmax attention compares a query with every key it has seen, so a decoder keeps
every key and value (the KV cache) and each new token costs time proportional to the context. That
cache is what makes long contexts expensive.

**What changes in code.** Linear attention replaces `softmax(q·k)` by a product of feature maps
`φ(q)ᵀφ(k)`. Once the similarity factorises, the sum over the past can be done *before* the query
arrives: the layer keeps `S = Σ φ(k_j) v_jᵀ` (one `d_k × d_v` matrix per head) and reads it with
`φ(q)`. The KV cache becomes a constant-size state, and decoding one token costs the same at
position 10 as at position 10 million.

**The trade-off.** A `d_k × d_v` matrix can hold only about `d_k` independent associations. Adding
more superimposes them, so retrieval of one exact earlier token degrades as the context grows —
the reason the delta rule (`attention.lab.delta`) and hybrids with a few softmax layers exist.

Two variants live here:

- `linear_attention` — Katharopoulos et al., "Transformers are RNNs", arXiv:2006.16236v3. The
  feature map is `φ(x) = elu(x) + 1` (Eq 7). The **parallel** form is the causal cumulative sum of
  Eq 10–12, `V'_i = φ(Q_i)ᵀ S_i / φ(Q_i)ᵀ Z_i`; the **recurrent** form is the RNN of Eq 16–20,
  which updates `s_i` and `z_i` one token at a time. The state holds both `s` and `z`. The paper
  writes rows (`Q = x W_Q`); with column vectors the state is `S ∈ R^{d_k × d_v}` read as `Sᵀ φ(q)`.
  The residual `+ x_i` and the feed-forward `f_l` of Eq 20 belong to the transformer block, not to
  the mixer, so they are not applied here; an output projection is added, as in any multi-head
  layer.
- `lightning_attention` — MiniMax-01, arXiv:2501.08313v1 §2.2.1. The paper calls it an I/O-aware
  implementation of TransNormer: linear attention without a denominator, `O = Norm(Q(KᵀV))`,
  computed by **tiling** (Eq 7–9, Algorithm 1). Inside a block of `B` tokens the masked "left
  product" `[(Q Kᵀ) ⊙ M] V` is used; across blocks the running `KV` matrix carries the past
  (`KV ← KV + Kᵀ V`). The **recurrent** form is Eq 5, `kv_t = kv_{t−1} + k_t v_tᵀ`.

  The paper sections read omit normalisation, SiLU and gating "for analytical tractability", and
  **state no decay at all**. The per-head decay used here is **taken from the released code**
  (`modeling_minimax_text_01.py` on the MiniMax-Text-01 model page, code licence MIT), not from the
  paper: `kv_t = λ kv_{t−1} + k_t v_tᵀ` with `λ = exp(−slope)`, the slopes following the
  ALiBi-style geometric rule `start = 2^(−2^(−(log2 n − 3)))`, `slope_i = start^(i+1)`, scaled per
  layer by `1 − layer_index / (num_layers − 1) + 1e−5`. Also from that code: SiLU on the q/k/v
  projection, no `1/√d` scale and no q/k normalisation, an RMSNorm over the concatenated heads, a
  sigmoid output gate computed from the layer input, then the output projection. The tiled pass
  with decay weights the intra-block mask by `λ^(t−s)` and the carried `KV` by `λ^(j+1)` for the
  `j`-th token of a block — the same algebra the code uses, re-derived here. This file implements
  those equations; it does not copy the code.

Orientation note for the whole lab: linear attention, DeltaNet and Gated DeltaNet write the state as
`S ∈ R^{d_v × d_k}` read by `S q`; KDA and Gated DeltaNet-2 write `S ∈ R^{d_k × d_v}` read by
`Sᵀ q`. The two are transposes of one another; each module below keeps its own paper's convention.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812 - the conventional alias

from attention.lab import hparams, sources
from attention.lab.base import Mixer, MixerSpec
from attention.lab.ops import merge_heads, split_heads
from attention.lab.registry import register

MINIMAX_URL = "https://arxiv.org/html/2501.08313v1"
MINIMAX_TITLE = "MiniMax-01: Scaling Foundation Models with Lightning Attention"

# --- small shared pieces --------------------------------------------------------------------------


def elu_plus_one(x: Tensor) -> Tensor:
    """The feature map of arXiv:2006.16236 Eq 7: `elu(x) + 1`, always positive."""
    return F.elu(x) + 1.0


def cumulative_linear_attention(
    q: Tensor, k: Tensor, v: Tensor, s0: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Causal linear attention with no denominator and no decay, by cumulative sum.

    `o_t = (s0 + Σ_{j≤t} k_j v_jᵀ)ᵀ q_t` — the "right product" of arXiv:2501.08313 Eq 5 written
    as one cumulative sum rather than a loop. A deliberately plain second implementation that the
    tiled lightning pass is tested against.

    Args:
        q: `[batch, heads, tokens, d_k]`, already feature-mapped.
        k: `[batch, heads, tokens, d_k]`, already feature-mapped.
        v: `[batch, heads, tokens, d_v]`.
        s0: `[batch, heads, d_k, d_v]` state before the first token, or None for zeros.

    Returns:
        The outputs `[batch, heads, tokens, d_v]` and the final state.
    """
    outer = torch.einsum("bhtk,bhtv->bhtkv", k, v).cumsum(dim=2)
    if s0 is not None:
        outer = outer + s0.unsqueeze(2)
    out = torch.einsum("bhtk,bhtkv->bhtv", q, outer)
    return out, outer[:, :, -1]


def alibi_style_slopes(heads: int) -> list[float]:
    """The geometric per-head slopes the MiniMax-Text-01 code uses for its decay.

    For a power of two `n`, `start = 2^(−2^(−(log2 n − 3)))` and slope `i` is `start^(i+1)`. For
    other head counts the code takes the slopes of the nearest lower power of two and fills the
    rest with every other slope of the next power of two; the same rule is implemented here.
    """

    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(heads).is_integer():
        return power_of_two(heads)
    lower = 2 ** math.floor(math.log2(heads))
    return power_of_two(lower) + alibi_style_slopes(2 * lower)[0::2][: heads - lower]


# --- linear attention (Katharopoulos et al.) -----------------------------------------------------


class LinearAttention(Mixer):
    """Causal linear attention with `φ = elu + 1` and the `S_i / Z_i` normalisation.

    Args:
        d_model: Width of the residual stream.
        heads: Number of heads.
        head_dim: Width of each head (q, k and v alike).
        mode: `parallel` (cumulative sums, Eq 10–12) or `recurrent` (the RNN, Eq 16–20). A full
            pass uses `mode`; both give the same numbers.
        context: The sequence length the source trained at. Recorded for shapes and cost; the
            layer itself has no length limit.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        mode: str = "parallel",
        context: int | None = None,
    ) -> None:
        """Build the projections."""
        super().__init__()
        if mode not in ("parallel", "recurrent"):
            raise ValueError(f"mode must be 'parallel' or 'recurrent', not {mode!r}")
        self.heads, self.head_dim, self.mode, self.context = heads, head_dim, mode, context
        inner = heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """`s_0 = 0` and `z_0 = 0` (Eq 16–17)."""
        d = self.head_dim
        return {
            "s": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype),
            "z": torch.zeros(batch, self.heads, d, device=device, dtype=dtype),
            "pos": 0,
        }

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` and return the output and the state after its last token."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        q = elu_plus_one(split_heads(self.q_proj(x), self.heads))
        k = elu_plus_one(split_heads(self.k_proj(x), self.heads))
        v = split_heads(self.v_proj(x), self.heads)
        if self.mode == "parallel":
            out, s, z = self._parallel(q, k, v, state["s"], state["z"])
        else:
            out, s, z = self._recurrent(q, k, v, state["s"], state["z"])
        new_state = {"s": s, "z": z, "pos": state["pos"] + x.shape[1]}
        return self.out_proj(merge_heads(out)), new_state

    @staticmethod
    def _parallel(
        q: Tensor, k: Tensor, v: Tensor, s0: Tensor, z0: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Eq 10–12: `S_i`, `Z_i` as cumulative sums, then one division per position."""
        numerator, s = cumulative_linear_attention(q, k, v, s0)
        z = k.cumsum(dim=2) + z0.unsqueeze(2)
        denominator = (q * z).sum(-1, keepdim=True)
        return numerator / denominator, s, z[:, :, -1]

    @staticmethod
    def _recurrent(
        q: Tensor, k: Tensor, v: Tensor, s: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Eq 16–20, one token at a time."""
        outs = []
        for t in range(q.shape[2]):
            s = s + k[:, :, t, :, None] * v[:, :, t, None, :]
            z = z + k[:, :, t]
            numerator = torch.einsum("bhk,bhkv->bhv", q[:, :, t], s)
            denominator = (q[:, :, t] * z).sum(-1, keepdim=True)
            outs.append(numerator / denominator)
        return torch.stack(outs, dim=2), s, z


register(
    MixerSpec(
        name="linear_attention",
        family="linear",
        summary=(
            "Replaces softmax(q·k) with elu+1 feature maps so the past folds into one fixed matrix "
            "S and a normaliser Z, turning the KV cache into a constant-size recurrent state."
        ),
        parent="standard_attention",
        covers="linear_attention",
        source="catalogue:linear_attention",
        checked_against=(
            "arXiv:2006.16236v3 Eq 7 (feature map), Eq 10-12 (causal cumulative form), "
            "Eq 16-20 (RNN form)"
        ),
        factory=LinearAttention,
        lab=(
            hparams.ours(
                "d_model",
                32,
                "Width of the residual stream.",
                "lab scale, small enough to run every check on a laptop CPU",
            ),
            hparams.ours(
                "heads",
                4,
                "Number of attention heads.",
                "lab scale, four heads of eight dimensions each keep the tests fast",
            ),
            hparams.ours(
                "head_dim",
                8,
                "Width of each head.",
                "lab scale, four heads of eight dimensions match the residual width",
            ),
            hparams.ours(
                "mode",
                "parallel",
                "Which form computes a full pass.",
                "the parallel cumulative form is the training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.from_catalogue("heads", "linear_attention", "heads", "Heads per layer."),
            hparams.from_catalogue(
                "context", "linear_attention", "context", "Sequence length of the MNIST run."
            ),
            hparams.from_catalogue(
                "head_dim", "linear_attention", "headDim", "Dimensions per head."
            ),
            hparams.ours(
                "d_model",
                256,
                "Width of the residual stream.",
                "heads times head dimension, eight times thirty-two, as the catalogue states both",
            ),
            hparams.ours(
                "mode",
                "parallel",
                "Which form computes a full pass.",
                "the paper trains with the parallel form and samples with the RNN form",
            ),
        ),
        state_growth="constant",
    )
)


# --- lightning attention (MiniMax-01) -------------------------------------------------------------

sources.add(
    sources.LabSource(
        id="lightning_attention.definition",
        value="TransNormer",
        quote="represents an I/O-aware, optimized implementation of TransNormer",
        where="§2.2.1 Lightning Attention, arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
    )
)
sources.add(
    sources.LabSource(
        id="lightning_attention.block_size",
        value=256,
        quote=(
            "Each input within the batch is padded to ensure that its length is a multiple of the "
            "predefined block size, which is set to 256"
        ),
        where="§3.2.2 Improved Linear Attention Sequence Parallelism (varlen), arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
        unit="tokens",
    )
)
sources.add(
    sources.LabSource(
        id="lightning_attention.hybrid_ratio",
        value=7,
        # "transformber" is the source's own spelling; the quote is copied as printed.
        quote="a transformber block with softmax attention is positioned after every 7 "
        "transnormer blocks",
        where="§2 Model Architecture, arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
        unit="lightning layers per softmax layer",
    )
)
sources.add(
    sources.LabSource(
        id="lightning_attention.layers",
        value=80,
        quote="of linear attention, leading to a total of 80 layers",
        where="§2 Model Architecture, arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
        unit="layers",
    )
)
_HEADS_QUOTE = "Each attention module is composed of 64 heads, each with a head dimension of 128"
sources.add(
    sources.LabSource(
        id="lightning_attention.heads",
        value=64,
        quote=_HEADS_QUOTE,
        where="§2 Model Architecture, arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
        unit="heads",
    )
)
sources.add(
    sources.LabSource(
        id="lightning_attention.head_dim",
        value=128,
        quote=_HEADS_QUOTE,
        where="§2 Model Architecture, arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
        unit="dimensions",
    )
)
sources.add(
    sources.LabSource(
        id="lightning_attention.d_model",
        value=6144,
        quote="The model’s hidden size is configured to 6144",
        where="§2 Model Architecture, arXiv:2501.08313v1",
        url=MINIMAX_URL,
        title=MINIMAX_TITLE,
        unit="dimensions",
    )
)


class LightningAttention(Mixer):
    """TransNormer-style linear attention computed by tiling, with the released code's decay.

    Args:
        d_model: Width of the residual stream.
        heads: Number of heads.
        head_dim: Width of each head (q, k and v alike).
        block_size: Tokens per tile `B` in the tiled pass (Algorithm 1).
        decay: Apply the per-head decay `λ = exp(−slope)` found in the released code. False gives
            `λ = 1`, the paper's Eq 5 exactly.
        layer_index: Zero-based position of this layer, which scales the slopes.
        num_layers: Layers in the model the slopes are scaled across (at least 2).
        mode: `tiled` (Algorithm 1) or `recurrent` (Eq 5) for a full pass.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        block_size: int,
        decay: bool = True,
        layer_index: int = 0,
        num_layers: int = 2,
        mode: str = "tiled",
    ) -> None:
        """Build the projections and the fixed decay."""
        super().__init__()
        if mode not in ("tiled", "recurrent"):
            raise ValueError(f"mode must be 'tiled' or 'recurrent', not {mode!r}")
        if num_layers < 2 or not 0 <= layer_index < num_layers:
            raise ValueError("need num_layers >= 2 and 0 <= layer_index < num_layers")
        self.heads, self.head_dim, self.block_size, self.mode = heads, head_dim, block_size, mode
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(d_model, 3 * inner, bias=False)
        self.output_gate = nn.Linear(d_model, inner, bias=False)
        self.norm = nn.RMSNorm(inner, eps=1e-6)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        scale = 1 - layer_index / (num_layers - 1) + 1e-5
        slopes = torch.tensor(alibi_style_slopes(heads), dtype=torch.float64) * scale
        ratio = torch.exp(-slopes) if decay else torch.ones(heads, dtype=torch.float64)
        # A plain CPU attribute, not a buffer: a float64 buffer cannot move to Apple's MPS device.
        self.ratio = ratio

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """`kv_0 = 0` (Eq 5)."""
        d = self.head_dim
        return {"kv": torch.zeros(batch, self.heads, d, d, device=device, dtype=dtype), "pos": 0}

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Mix `x` and return the output and the state after its last token."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        q, k, v = (
            split_heads(part, self.heads) for part in F.silu(self.qkv_proj(x)).chunk(3, dim=-1)
        )
        ratio = self.ratio.to(device=x.device, dtype=x.dtype)
        if self.mode == "tiled":
            out, kv = self.tiled(q, k, v, state["kv"], ratio, self.block_size)
        else:
            out, kv = self.recurrent(q, k, v, state["kv"], ratio)
        out = self.norm(merge_heads(out))
        out = torch.sigmoid(self.output_gate(x)) * out
        return self.out_proj(out), {"kv": kv, "pos": state["pos"] + x.shape[1]}

    @staticmethod
    def recurrent(
        q: Tensor, k: Tensor, v: Tensor, kv: Tensor, ratio: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Eq 5 with decay: `kv_t = λ kv_{t−1} + k_t v_tᵀ`, `o_t = kv_tᵀ q_t`."""
        lam = ratio.view(1, -1, 1, 1)
        outs = []
        for t in range(q.shape[2]):
            kv = lam * kv + k[:, :, t, :, None] * v[:, :, t, None, :]
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], kv))
        return torch.stack(outs, dim=2), kv

    @staticmethod
    def tiled(
        q: Tensor, k: Tensor, v: Tensor, kv: Tensor, ratio: Tensor, block_size: int
    ) -> tuple[Tensor, Tensor]:
        """Algorithm 1: masked left product inside each block, carried `KV` across blocks.

        For the `j`-th token of a block (`j = 0 … m−1`), the state after it is
        `λ^(j+1) KV + Σ_{s≤j} λ^(j−s) k_s v_sᵀ`, so the inter-block term is weighted by
        `λ^(j+1)` and the intra-block mask by `λ^(j−s)`. With `λ = 1` both weights are 1 and this
        is the paper's Eq 7–9 exactly.
        """
        log_lam = torch.log(ratio).view(1, -1, 1, 1)
        outs = []
        for start in range(0, q.shape[2], block_size):
            qb, kb, vb = (t[:, :, start : start + block_size] for t in (q, k, v))
            m = qb.shape[2]
            j = torch.arange(m, device=q.device, dtype=q.dtype)
            gap = j[:, None] - j[None, :]
            intra_weight = torch.where(gap >= 0, torch.exp(log_lam * gap), 0.0)
            intra = ((qb @ kb.transpose(-2, -1)) * intra_weight) @ vb
            carry = torch.exp(log_lam * (j + 1).view(1, 1, m, 1))
            inter = carry * (qb @ kv)
            outs.append(intra + inter)
            key_weight = torch.exp(log_lam * (m - 1 - j).view(1, 1, m, 1))
            kv = torch.exp(log_lam * m) * kv + (kb * key_weight).transpose(-2, -1) @ vb
        return torch.cat(outs, dim=2), kv


register(
    MixerSpec(
        name="lightning_attention",
        family="linear",
        summary=(
            "Drops the linear-attention denominator for an output RMSNorm and sigmoid gate, uses "
            "SiLU features, decays the state per head, and computes the full pass in tiles."
        ),
        parent="linear_attention",
        covers=None,
        source="lab:lightning_attention.definition",
        checked_against=(
            "arXiv:2501.08313v1 §2.2.1 Eq 4-9 and Algorithm 1 (tiling), Eq 5 (recurrence); "
            "decay, SiLU, RMSNorm and output gate from the released MiniMax-Text-01 modeling code "
            "(MIT), which the paper sections read do not state"
        ),
        factory=LightningAttention,
        lab=(
            hparams.ours(
                "d_model",
                32,
                "Width of the residual stream.",
                "lab scale, small enough to run every check on a laptop CPU",
            ),
            hparams.ours(
                "heads",
                4,
                "Number of attention heads.",
                "lab scale, and a power of two so the slope rule takes its simple branch",
            ),
            hparams.ours(
                "head_dim",
                8,
                "Width of each head.",
                "lab scale, four heads of eight dimensions match the residual width",
            ),
            hparams.ours(
                "block_size",
                5,
                "Tokens per tile.",
                "deliberately not a divisor of the "
                "test lengths so a ragged last tile is always exercised",
            ),
            hparams.ours(
                "decay",
                True,
                "Apply the per-head decay from the released code.",
                "the released model decays its state; the paper text omits it",
            ),
            hparams.ours(
                "layer_index",
                0,
                "Zero-based layer position; scales the slopes.",
                "the first layer has the strongest decay in the released code",
            ),
            hparams.ours(
                "num_layers",
                2,
                "Layers the slope scaling spans.",
                "the smallest value the slope scaling formula accepts without dividing by zero",
            ),
            hparams.ours(
                "mode",
                "tiled",
                "Which form computes a full pass.",
                "the tiled form is the paper's training path; decoding uses the recurrence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "lightning_attention.d_model", "Hidden size."),
            hparams.from_lab_source("heads", "lightning_attention.heads", "Heads per layer."),
            hparams.from_lab_source(
                "head_dim", "lightning_attention.head_dim", "Dimensions per head."
            ),
            hparams.from_lab_source(
                "block_size", "lightning_attention.block_size", "Tokens per tile."
            ),
            hparams.ours(
                "decay",
                True,
                "Apply the per-head decay from the released code.",
                "present in the released code, not stated in the paper sections read",
            ),
            hparams.ours(
                "layer_index",
                0,
                "Zero-based layer position; scales the slopes.",
                "any layer of the eighty; the first is shown by default",
            ),
            hparams.from_lab_source(
                "num_layers", "lightning_attention.layers", "Layers in the model."
            ),
            hparams.ours(
                "mode",
                "tiled",
                "Which form computes a full pass.",
                "the paper trains with the tiled form of Algorithm 1",
            ),
        ),
        state_growth="constant",
    )
)
