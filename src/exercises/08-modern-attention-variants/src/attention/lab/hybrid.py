"""The hybrid family: a few full-attention layers among many fixed-memory layers.

**The problem.** A linear or delta-rule layer (`attention.lab.linear`, `attention.lab.delta`) keeps
a fixed-size state, so its memory and its cost per token do not grow with the context. The price is
recall: a matrix of fixed size cannot hold every earlier token exactly, and long-context retrieval
suffers. A softmax layer recalls exactly, but its KV cache grows with every token. A hybrid keeps
mostly fixed-memory layers and places a full-attention layer every few layers, so the stack still
has an exact lookup somewhere while most of its layers stay constant-size.

**What changes in code, relative to the parents.** Nothing inside any layer. A hybrid is a `Mixer`
that owns an ordered list of existing mixers and applies them in turn, each as a pre-norm residual
step, `h ← h + layer(norm(h))`, carrying one state per layer. The layout is a string with one
letter per layer:

- `kda_hybrid` — Kimi Linear (arXiv:2510.26692v2 §4, "Hybrid model architecture"): blocks of three
  KDA layers (`K`) and one full MLA layer (`M`). The source applies **no position encoding** to the
  MLA layers; `NoPEMultiHeadLatentAttention` below is that layer. `mla.py` has no switch for it, so
  it is a thin subclass here: the same projections and the same decoupled query/key dimensions as
  the parent, never rotated. That keeps the parent's cache layout, and it is also what the released
  Kimi Linear modelling code does (read, not copied: it splits off the `qk_rope_head_dim` part and
  concatenates it without rotation).
- `lightning_hybrid` — MiniMax-01 (arXiv:2501.08313v1 §2): seven lightning-attention layers (`L`)
  and one softmax layer (`A`). The source's softmax layers use grouped-query attention with RoPE on
  half of each head; `PartialRoPEGroupedAttention` is `GroupedAttention` with the first
  `rotary_dim` dimensions of every query and key rotated. Each lightning layer is built with its
  position in the whole stack, because the released code scales its decay by layer depth.
- `kimi_k3_stack` — Kimi K3 (arXiv:2607.24653v2 §2.1): blocks of three KDA layers with K3's two
  changes (lower-bounded decay, full-rank output gate; `delta.K3_OVERRIDES`) and one **Gated MLA**
  layer (`G`), plus one more Gated MLA layer at the very end. Gated MLA is the NoPE MLA with an
  input-dependent, channel-wise sigmoid gate on the heads' output before the output projection
  (its Eq 7, `y = W_o[σ(W_g x) ⊙ õ]`).

The layout is `pattern × blocks + final`, where `pattern` is `ratio` fixed-memory letters followed
by one full-attention letter (or any string a learner passes), and `blocks` is whatever fills
`num_layers` after `final`. At paper scale that reproduces each released layout: 23 blocks plus a
final `G` is Kimi K3's 93 layers, six blocks plus `KKM` is Kimi Linear's 27, and ten blocks is
MiniMax-01's 80.

**The hybrid's output is the sum of its layers' residual updates**, not the final residual stream,
so a hybrid can sit inside `TinyDecoder`'s own `x + mixer(norm(x))` without counting its input
twice. A one-layer hybrid is therefore exactly `layer(norm(x))`, which the family tests check.

**The trade-off.** Memory is the sum of the layers' states: the fixed-memory part is constant and
the full-attention part grows, so a 3:1 stack keeps a quarter of the KV cache a full-attention
stack of the same depth would. The ratio is a quality/throughput choice the sources settled by
experiment, and it is a parameter here so it can be varied.

**What is ours, and what is not implemented.**

- Only the mixers are stacked: the sources pair every layer with an MoE feed-forward network, and
  MiniMax-01 uses post-norm with scaled residuals. The norm here is an `RMSNorm` per layer, our
  choice for a pre-norm stack.
- The lab MLA always compresses queries through `q_rank`; the released Kimi Linear config has no
  query compression, and the paper-scale `mla_q_rank` for `kda_hybrid` says so.
- `PartialRoPEGroupedAttention` rotates adjacent pairs (the lab's `ops.apply_rope`); the released
  MiniMax code rotates the first `rotary_dim` dimensions with the half-split pairing. It keeps
  un-rotated keys in its cache and rotates them when it reads them, which gives the same numbers
  as caching rotated keys and holds the same bytes.
- Gated MLA's gate has no bias, as Eq 7 is written.
- Kimi K3's **Attention Residuals** (cross-layer attention over earlier block outputs, arXiv
  2603.15031) are not implemented: that paper was not read for this module, and it changes the
  residual stream rather than the mixer, which is outside what a `Mixer` can express.

Followed: arXiv:2510.26692v2 §4 (hybrid layout, NoPE); arXiv:2607.24653v2 §2.1 and §2.1.2 Eq 7
(K3 layout, Gated MLA); arXiv:2501.08313v1 §2 (MiniMax-01 layout, GQA, partial RoPE).
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from attention.lab import hparams, ops
from attention.lab.base import Mixer, MixerSpec
from attention.lab.core import GroupedAttention
from attention.lab.delta import (
    KIMI_K3_CONFIG,
    KIMI_K3_URL,
    KIMI_LINEAR_CONFIG,
    KIMI_LINEAR_TITLE,
    KIMI_LINEAR_URL,
    KimiDeltaAttention,
)
from attention.lab.linear import MINIMAX_TITLE, MINIMAX_URL, LightningAttention
from attention.lab.mla import MultiHeadLatentAttention
from attention.lab.registry import register
from attention.lab.sources import LabSource, add

# --- sources --------------------------------------------------------------------------------------

_KL_WHERE = "§4 The Kimi Linear Model Architecture, Hybrid model architecture, arXiv:2510.26692v2"
_KL_CONFIG_TITLE = "Kimi-Linear-48B-A3B-Instruct config.json"
_KL_CONFIG_WHERE = "config.json of Kimi-Linear-48B-A3B-Instruct (main branch)"
_K3_TITLE = "Kimi K3 technical report"
_K3_CONFIG_TITLE = "Kimi-K3 config.json"
_K3_CONFIG_WHERE = "config.json text_config of Kimi-K3 (main branch)"
_MM_CONFIG = "https://huggingface.co/MiniMaxAI/MiniMax-Text-01/resolve/main/config.json"
_MM_CONFIG_TITLE = "MiniMax-Text-01 config.json"
_MM_CONFIG_WHERE = "config.json of MiniMax-Text-01 (main branch)"
_MM_WHERE = "§2 Model Architecture, arXiv:2501.08313v1"


def _config(id_: str, value: Any, quote: str, where: str, url: str, title: str, unit: str) -> None:
    add(LabSource(id_, value, quote, where, url, title, unit))


add(
    LabSource(
        "kda_hybrid.ratio",
        3,
        "Empirically, a uniform 3:1 ratio, i.e., repeating 3 KDA layers to 1 full MLA layer, "
        "provided the best quality",
        _KL_WHERE,
        KIMI_LINEAR_URL,
        KIMI_LINEAR_TITLE,
        "KDA layers per MLA layer",
    )
)
add(
    LabSource(
        "kda_hybrid.nope",
        "NoPE",
        "In Kimi Linear, we apply NoPE to all full attention (MLA) layers.",
        "§4 The Kimi Linear Model Architecture, No Position Encoding (NoPE) for MLA Layers, "
        "arXiv:2510.26692v2",
        KIMI_LINEAR_URL,
        KIMI_LINEAR_TITLE,
    )
)
for _id, _value, _quote, _unit in (
    ("kda_hybrid.layers", 27, '"num_hidden_layers": 27', "layers"),
    ("kda_hybrid.mla_heads", 32, '"num_attention_heads": 32', "heads"),
    ("kda_hybrid.mla_head_dim", 128, '"qk_nope_head_dim": 128', "dimensions"),
    ("kda_hybrid.mla_kv_rank", 512, '"kv_lora_rank": 512', "dimensions"),
    ("kda_hybrid.mla_rope_dim", 64, '"qk_rope_head_dim": 64', "dimensions"),
):
    _config(_id, _value, _quote, _KL_CONFIG_WHERE, KIMI_LINEAR_CONFIG, _KL_CONFIG_TITLE, _unit)

add(
    LabSource(
        "kimi_k3.layout",
        3,
        "Each block contains 3 KDA layers followed by 1 Gated MLA layer, giving a 3 : 1 mixing "
        "ratio.",
        "§2.1 Hybrid Attention, arXiv:2607.24653v2",
        KIMI_K3_URL,
        _K3_TITLE,
        "KDA layers per Gated MLA layer",
    )
)
add(
    LabSource(
        "kimi_k3.final_layer",
        "Gated MLA",
        "An additional Gated MLA layer is placed at the end of the backbone, ensuring that the "
        "final layer always performs global attention",
        "§2.1 Hybrid Attention, arXiv:2607.24653v2",
        KIMI_K3_URL,
        _K3_TITLE,
    )
)
add(
    LabSource(
        "kimi_k3.mla_output_gate",
        "full-rank",
        "augments MLA with an input-dependent, channel-wise full-rank output gate",
        "§2.1.2 Gated MLA, arXiv:2607.24653v2",
        KIMI_K3_URL,
        _K3_TITLE,
    )
)
add(
    LabSource(
        "kimi_k3.nope",
        "NoPE",
        "applies No Position Encoding (NoPE) to all MLA layers",
        "§2.1.2 Gated MLA, arXiv:2607.24653v2",
        KIMI_K3_URL,
        _K3_TITLE,
    )
)
for _id, _value, _quote, _unit in (
    ("kimi_k3.layers", 93, '"num_hidden_layers": 93', "layers"),
    ("kimi_k3.d_model", 7168, '"hidden_size": 7168', "dimensions"),
    ("kimi_k3.kda_heads", 96, '"num_heads": 96', "heads"),
    ("kimi_k3.kda_head_dim", 128, '"head_dim": 128', "dimensions"),
    ("kimi_k3.conv_kernel", 4, '"short_conv_kernel_size": 4', "tokens"),
    ("kimi_k3.mla_heads", 96, '"num_attention_heads": 96', "heads"),
    ("kimi_k3.mla_head_dim", 128, '"qk_nope_head_dim": 128', "dimensions"),
    ("kimi_k3.mla_kv_rank", 512, '"kv_lora_rank": 512', "dimensions"),
    ("kimi_k3.mla_q_rank", 1536, '"q_lora_rank": 1536', "dimensions"),
    ("kimi_k3.mla_rope_dim", 64, '"qk_rope_head_dim": 64', "dimensions"),
):
    _config(_id, _value, _quote, _K3_CONFIG_WHERE, KIMI_K3_CONFIG, _K3_CONFIG_TITLE, _unit)

add(
    LabSource(
        "lightning_hybrid.rope_base",
        10000,
        "is applied to half of the attention head dimension, with a base frequency set to 10,000",
        _MM_WHERE,
        MINIMAX_URL,
        MINIMAX_TITLE,
    )
)
for _id, _value, _quote, _unit in (
    ("lightning_hybrid.kv_heads", 8, '"num_key_value_heads": 8', "heads"),
    ("lightning_hybrid.rotary_dim", 64, '"rotary_dim": 64', "dimensions"),
):
    _config(_id, _value, _quote, _MM_CONFIG_WHERE, _MM_CONFIG, _MM_CONFIG_TITLE, _unit)


# --- the full-attention layers the hybrids use ----------------------------------------------------

POSITIONS = ("NoPE", "RoPE")


class NoPEMultiHeadLatentAttention(MultiHeadLatentAttention):
    """MLA with no position encoding: the decoupled query and key dimensions are never rotated.

    Everything else — projections, cache, scale — is the parent's. With `position="RoPE"` the layer
    rotates as the parent does, which is how the tests prove this class and the parent agree.

    Args:
        position: `NoPE` (Kimi Linear, Kimi K3) or `RoPE` (the parent, DeepSeek-V2).
        **kwargs: The parent's arguments.
    """

    def __init__(self, position: str = "NoPE", **kwargs: Any) -> None:
        """Build the parent's projections and remember whether to rotate."""
        super().__init__(**kwargs)
        if position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, not {position!r}")
        self.position = position

    def head_outputs(self, x: Tensor, state: dict | None) -> tuple[Tensor, dict]:
        """The heads' merged output `õ` (before the output projection) and the new cache."""
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        pos = state["pos"]
        if self.position == "RoPE":
            positions = torch.arange(pos, pos + tokens, device=x.device)
            angles = ops.rope_angles(positions, self.rope_dim, self.rope_base)

            def place(t: Tensor) -> Tensor:
                return ops.apply_rope(t, angles)

        else:

            def place(t: Tensor) -> Tensor:
                return t

        c_kv = torch.cat([state["c_kv"], self.w_dkv(x)], dim=1)
        k_rope = torch.cat([state["k_rope"], place(self.w_kr(x))], dim=1)
        keys = c_kv.shape[1]
        k_content = ops.split_heads(self.w_uk(c_kv), self.heads)
        v_content = ops.split_heads(self.w_uv(c_kv), self.heads)
        shared = k_rope[:, None].expand(batch, self.heads, keys, self.rope_dim)
        k = torch.cat([k_content, shared], dim=-1)
        c_q = self.w_dq(x)
        q_content = ops.split_heads(self.w_uq(c_q), self.heads)
        q_rope = place(ops.split_heads(self.w_qr(c_q), self.heads))
        q = torch.cat([q_content, q_rope], dim=-1)
        allowed = ops.causal_mask(tokens, keys, offset=keys - tokens, device=x.device)
        scale = (self.head_dim + self.rope_dim) ** -0.5
        out, _ = ops.attend(q, k, v_content, allowed=allowed, scale=scale)
        return ops.merge_heads(out), {"c_kv": c_kv, "k_rope": k_rope, "pos": pos + tokens}

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Attend over the cache and project the heads back to `d_model`."""
        o, state = self.head_outputs(x, state)
        return self.w_o(o), state


class GatedNoPEMultiHeadLatentAttention(NoPEMultiHeadLatentAttention):
    """Kimi K3's Gated MLA: `y = W_o[σ(W_g x) ⊙ õ]` (arXiv:2607.24653v2 Eq 7), NoPE by default.

    `W_g` is a full-rank, bias-free map from the layer input to one gate per head channel.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Build the NoPE MLA and the full-rank gate projection."""
        super().__init__(**kwargs)
        d_model = self.w_dkv.in_features
        self.w_g = nn.Linear(d_model, self.heads * self.head_dim, bias=False)

    def gate(self, x: Tensor) -> Tensor:
        """`σ(W_g x)`: one value in (0, 1) per token and head channel."""
        return torch.sigmoid(self.w_g(x))

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Gate the heads' output channel by channel, then project."""
        o, state = self.head_outputs(x, state)
        return self.w_o(self.gate(x) * o), state


class PartialRoPEGroupedAttention(GroupedAttention):
    """Grouped-query attention with RoPE on the first `rotary_dim` dimensions of each head.

    The parent caches keys before rotation; this class rotates the whole cache at read time, with
    the queries at positions `offset …` and the keys at `0 …`. That is the same arithmetic as
    rotating each key once when it is written.

    Args:
        d_model: Width of a token.
        n_heads: Query heads.
        n_kv_heads: Key/value heads; must divide `n_heads`.
        head_dim: Width of one head.
        rotary_dim: Leading dimensions of each head that are rotated; even, at most `head_dim`.
        rope_base: RoPE frequency base.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        rope_base: float,
    ) -> None:
        """Build the parent's projections and remember the rotated width."""
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        if rotary_dim % 2 or not 0 <= rotary_dim <= head_dim:
            raise ValueError(f"rotary_dim must be even and at most {head_dim}, not {rotary_dim}")
        self.rotary_dim, self.rope_base = rotary_dim, rope_base

    def _rotate(self, x: Tensor, first: int) -> Tensor:
        positions = torch.arange(first, first + x.shape[2], device=x.device)
        angles = ops.rope_angles(positions, self.rotary_dim, self.rope_base)
        turned = ops.apply_rope(x[..., : self.rotary_dim], angles)
        return torch.cat([turned, x[..., self.rotary_dim :]], dim=-1)

    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
        """Rotate the leading dimensions of the queries and every cached key, then attend."""
        if self.rotary_dim:
            q, k = self._rotate(q, offset), self._rotate(k, 0)
        return super().mix(q, k, v, offset)


# --- the stack ------------------------------------------------------------------------------------

#: What each layout letter builds.
LETTERS = {
    "K": "Kimi Delta Attention (fixed-size state)",
    "M": "MLA without position encoding (KV cache)",
    "G": "Gated MLA without position encoding (KV cache)",
    "L": "lightning attention (fixed-size state)",
    "A": "grouped-query softmax attention with partial RoPE (KV cache)",
}

#: Builds the mixer for one letter, given its index in the stack and the stack's depth.
LayerBuilder = Callable[[int, int], Mixer]


def layout(ratio: int, pattern: str, final: str, num_layers: int, linear: str, full: str) -> str:
    """One letter per layer: `block × blocks + final`, filling exactly `num_layers`.

    Args:
        ratio: Fixed-memory layers per full-attention layer; the block is `linear × ratio + full`.
        pattern: A block given letter by letter; when non-empty it is used and `ratio` is not.
        final: Layers appended after the last block.
        num_layers: Total depth.
        linear: The fixed-memory letter.
        full: The full-attention letter.

    Raises:
        ValueError: When the blocks cannot fill `num_layers` exactly.
    """
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0, not {ratio}")
    block = pattern or linear * ratio + full
    room = num_layers - len(final)
    if room < 0 or room % len(block):
        raise ValueError(
            f"{num_layers} layers cannot be {block!r} repeated plus {final!r}; change num_layers, "
            "ratio, pattern or final so the blocks fit exactly"
        )
    return block * (room // len(block)) + final


class HybridStack(Mixer):
    """An ordered list of mixers, each applied as `h ← h + layer(norm(h))`.

    Args:
        d_model: Width of the residual stream.
        letters: One layout letter per layer.
        builders: Letter → function building that layer from `(index, depth)`.
    """

    def __init__(self, d_model: int, letters: str, builders: dict[str, LayerBuilder]) -> None:
        """Build every layer and one RMSNorm in front of each."""
        super().__init__()
        unknown = sorted(set(letters) - set(builders))
        if unknown:
            raise ValueError(
                f"letters {unknown} are not layers of this hybrid; use {sorted(builders)}"
            )
        if not letters:
            raise ValueError("a hybrid needs at least one layer")
        self.letters = letters
        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in letters)
        self.layers = nn.ModuleList(builders[c](i, len(letters)) for i, c in enumerate(letters))

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """Every layer's empty state, in order."""
        states = [layer.init_state(batch, device=device, dtype=dtype) for layer in self.layers]
        return {"layers": states, "pos": 0}

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Run the layers in order; return the summed residual updates and every layer's state."""
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        h, total, states = x, torch.zeros_like(x), []
        for norm, layer, layer_state in zip(self.norms, self.layers, state["layers"], strict=True):
            update, layer_state = layer(norm(h), layer_state)
            h, total = h + update, total + update
            states.append(layer_state)
        return total, {"layers": states, "pos": state["pos"] + x.shape[1]}

    def state_bytes(self, state: Any) -> int:
        """The sum of what each layer reports keeping."""
        return sum(
            layer.state_bytes(s) for layer, s in zip(self.layers, state["layers"], strict=True)
        )

    def bytes_by_letter(self, state: Any) -> dict[str, int]:
        """State bytes grouped by layout letter, e.g. the KDA part and the MLA part."""
        out: dict[str, int] = {}
        for letter, layer, s in zip(self.letters, self.layers, state["layers"], strict=True):
            out[letter] = out.get(letter, 0) + layer.state_bytes(s)
        return out


def _mla_builder(letter: str, d_model: int, kw: dict[str, Any]) -> LayerBuilder:
    cls = GatedNoPEMultiHeadLatentAttention if letter == "G" else NoPEMultiHeadLatentAttention

    def build(index: int, depth: int) -> Mixer:
        return cls(
            d_model=d_model,
            heads=kw["mla_heads"],
            head_dim=kw["mla_head_dim"],
            kv_rank=kw["mla_kv_rank"],
            q_rank=kw["mla_q_rank"],
            rope_dim=kw["mla_rope_dim"],
            rope_base=kw["mla_rope_base"],
            position=kw["mla_position"],
        )

    return build


def _kda_builder(d_model: int, kw: dict[str, Any]) -> LayerBuilder:
    def build(index: int, depth: int) -> Mixer:
        return KimiDeltaAttention(
            d_model,
            kw["kda_heads"],
            kw["kda_head_dim"],
            kw["kda_chunk_size"],
            conv_kernel=kw["kda_conv_kernel"],
            decay_floor=kw.get("kda_decay_floor", 0.0),
            output_gate=kw.get("kda_output_gate", "low-rank"),
        )

    return build


def kda_hybrid(
    d_model: int, num_layers: int, ratio: int, pattern: str = "", final: str = "", **kw: Any
) -> HybridStack:
    """Kimi Linear: `K`×ratio + `M`, repeated; `kw` holds the `kda_*` and `mla_*` sizes."""
    letters = layout(ratio, pattern, final, num_layers, "K", "M")
    return HybridStack(
        d_model, letters, {"K": _kda_builder(d_model, kw), "M": _mla_builder("M", d_model, kw)}
    )


def kimi_k3_stack(
    d_model: int,
    num_layers: int,
    ratio: int,
    pattern: str = "",
    final: str = "G",
    mla_output_gate: str = "full-rank",
    **kw: Any,
) -> HybridStack:
    """Kimi K3: `K`×ratio + `G`, repeated, then a final `G`; `mla_output_gate="none"` ungates it."""
    if mla_output_gate not in ("full-rank", "none"):
        raise ValueError(f"mla_output_gate must be 'full-rank' or 'none', not {mla_output_gate!r}")
    letters = layout(ratio, pattern, final, num_layers, "K", "G")
    full = "G" if mla_output_gate == "full-rank" else "M"
    return HybridStack(
        d_model, letters, {"K": _kda_builder(d_model, kw), "G": _mla_builder(full, d_model, kw)}
    )


def lightning_hybrid(
    d_model: int, num_layers: int, ratio: int, pattern: str = "", final: str = "", **kw: Any
) -> HybridStack:
    """MiniMax-01: `L`×ratio + `A`, repeated; `kw` holds the `lin_*` and `attn_*` sizes."""
    letters = layout(ratio, pattern, final, num_layers, "L", "A")

    def lightning(index: int, depth: int) -> Mixer:
        # The released code scales the decay by depth, and its formula needs at least two layers.
        return LightningAttention(
            d_model,
            kw["lin_heads"],
            kw["lin_head_dim"],
            kw["lin_block_size"],
            decay=kw["lin_decay"],
            layer_index=index,
            num_layers=max(depth, 2),
        )

    def softmax(index: int, depth: int) -> Mixer:
        return PartialRoPEGroupedAttention(
            d_model,
            kw["attn_heads"],
            kw["attn_kv_heads"],
            kw["attn_head_dim"],
            kw["attn_rotary_dim"],
            kw["attn_rope_base"],
        )

    return HybridStack(d_model, letters, {"L": lightning, "A": softmax})


# --- registrations --------------------------------------------------------------------------------

_LAB = "lab scale, small enough to run every check on a laptop CPU"
_D_MODEL = "Width of the residual stream."
_NUM_LAYERS = "Total depth; the blocks repeat to fill it after the final layers."
_RATIO = "Fixed-memory layers per full-attention layer in one block."
_PATTERN = "One block, letter by letter; empty means build it from the ratio."
_FINAL = "Layers appended after the last block."
_PATTERN_NOTE = "empty, so the block is built from the sourced ratio; set it to try other layouts"


def _lab(name: str, value: Any, meaning: str, note: str = _LAB):
    return hparams.ours(name, value, meaning, note)


def _mla_lab(position_source: str) -> tuple:
    return (
        _lab("mla_heads", 4, "MLA attention heads."),
        _lab("mla_head_dim", 8, "MLA content query, key and value width per head."),
        _lab("mla_kv_rank", 12, "Width of MLA's cached key/value latent."),
        _lab("mla_q_rank", 16, "Width of MLA's query latent."),
        _lab("mla_rope_dim", 4, "Width of MLA's decoupled query/key part (unrotated under NoPE)."),
        hparams.ours(
            "mla_rope_base",
            10000.0,
            "RoPE base, used only when mla_position is RoPE.",
            "the conventional base; unused under NoPE, which the source applies",
        ),
        hparams.from_lab_source(
            "mla_position", position_source, "Position encoding of the MLA layers."
        ),
    )


def _kda_lab() -> tuple:
    return (
        _lab("kda_heads", 4, "KDA heads."),
        _lab("kda_head_dim", 8, "KDA key and value width per head."),
        hparams.ours(
            "kda_chunk_size",
            5,
            "Chunk length of KDA's chunkwise form.",
            "deliberately not a divisor of the test lengths so a ragged chunk runs",
        ),
        hparams.ours(
            "kda_conv_kernel",
            4,
            "Width of KDA's short convolution.",
            "the released models' value, already small enough for the lab",
        ),
    )


register(
    MixerSpec(
        name="kda_hybrid",
        family="hybrid",
        summary=(
            "Stacks KDA layers with a full MLA layer after every three, the MLA layers carrying no "
            "position encoding, so most layers keep a fixed state and a quarter keep a KV cache."
        ),
        parent="kda",
        covers=None,
        source="lab:kda_hybrid.ratio",
        checked_against=(
            "arXiv:2510.26692v2 §4 'Hybrid model architecture' (uniform 3:1 KDA-to-MLA layers) and "
            "'No Position Encoding (NoPE) for MLA Layers'; layout of the released config"
        ),
        factory=kda_hybrid,
        lab=(
            _lab("d_model", 32, _D_MODEL),
            hparams.ours("num_layers", 8, _NUM_LAYERS, "lab scale, two blocks of the 3:1 pattern"),
            hparams.from_lab_source("ratio", "kda_hybrid.ratio", _RATIO),
            hparams.ours("pattern", "", _PATTERN, _PATTERN_NOTE),
            hparams.ours("final", "", _FINAL, "none at lab scale, so the layout is uniform blocks"),
            *_kda_lab(),
            *_mla_lab("kda_hybrid.nope"),
        ),
        paper=(
            hparams.from_lab_source("d_model", "kda.d_model", "Hidden size of the released model."),
            hparams.from_lab_source("num_layers", "kda_hybrid.layers", "Layers in the model."),
            hparams.from_lab_source("ratio", "kda_hybrid.ratio", _RATIO),
            hparams.ours("pattern", "", _PATTERN, _PATTERN_NOTE),
            hparams.ours(
                "final",
                "KKM",
                _FINAL,
                "the released config lists full attention at layers 4 to 24 by fours and 27, "
                "so after six blocks the last three layers are KDA, KDA, MLA",
            ),
            hparams.from_lab_source("kda_heads", "kda.heads", "KDA heads."),
            hparams.from_catalogue("kda_head_dim", "kda", "headDim", "KDA head dimension."),
            hparams.from_catalogue("kda_chunk_size", "kda", "chunk", "KDA chunk length."),
            hparams.from_lab_source("kda_conv_kernel", "kda.conv_kernel", "Short conv width."),
            hparams.from_lab_source("mla_heads", "kda_hybrid.mla_heads", "MLA heads."),
            hparams.from_lab_source(
                "mla_head_dim", "kda_hybrid.mla_head_dim", "MLA content width per head."
            ),
            hparams.from_lab_source("mla_kv_rank", "kda_hybrid.mla_kv_rank", "MLA latent width."),
            hparams.ours(
                "mla_q_rank",
                2304,
                "Width of MLA's query latent.",
                "the released config has no query compression; the lab MLA always has one, so "
                "the hidden size is used, a rank that compresses nothing",
            ),
            hparams.from_lab_source(
                "mla_rope_dim", "kda_hybrid.mla_rope_dim", "MLA decoupled query/key width."
            ),
            hparams.ours(
                "mla_rope_base",
                10000.0,
                "RoPE base, used only when mla_position is RoPE.",
                "the released config's value, unused because its MLA layers use NoPE",
            ),
            hparams.from_lab_source(
                "mla_position", "kda_hybrid.nope", "Position encoding of the MLA layers."
            ),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="kimi_k3_stack",
        family="hybrid",
        summary=(
            "Kimi K3's layout: blocks of three K3 KDA layers and one Gated MLA layer (NoPE, "
            "channel-wise sigmoid output gate), plus a final Gated MLA layer."
        ),
        parent="kda_hybrid",
        covers=None,
        source="lab:kimi_k3.layout",
        checked_against=(
            "arXiv:2607.24653v2 §2.1 (3:1 blocks, final Gated MLA layer), §2.1.1 Eq 5-6 (K3 KDA, "
            "via delta.K3_OVERRIDES), §2.1.2 Eq 7 (Gated MLA, NoPE); Attention Residuals (§2.2) "
            "not implemented"
        ),
        factory=kimi_k3_stack,
        lab=(
            _lab("d_model", 32, _D_MODEL),
            hparams.ours(
                "num_layers", 9, _NUM_LAYERS, "lab scale, two blocks plus the final Gated MLA"
            ),
            hparams.from_lab_source("ratio", "kimi_k3.layout", _RATIO),
            hparams.ours("pattern", "", _PATTERN, _PATTERN_NOTE),
            hparams.ours(
                "final",
                "G",
                _FINAL,
                "one Gated MLA layer, as lab:kimi_k3.final_layer states for the backbone's end",
            ),
            *_kda_lab(),
            hparams.from_lab_source("kda_decay_floor", "kda.k3_decay_floor", "K3 log-decay bound."),
            hparams.from_lab_source("kda_output_gate", "kda.k3_output_gate", "K3 KDA output gate."),
            *_mla_lab("kimi_k3.nope"),
            hparams.from_lab_source(
                "mla_output_gate", "kimi_k3.mla_output_gate", "Gated MLA's output gate."
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "kimi_k3.d_model", "Hidden size."),
            hparams.from_lab_source("num_layers", "kimi_k3.layers", "Layers in the model."),
            hparams.from_lab_source("ratio", "kimi_k3.layout", _RATIO),
            hparams.ours("pattern", "", _PATTERN, _PATTERN_NOTE),
            hparams.ours(
                "final",
                "G",
                _FINAL,
                "one Gated MLA layer, as lab:kimi_k3.final_layer states for the backbone's end",
            ),
            hparams.from_lab_source("kda_heads", "kimi_k3.kda_heads", "KDA heads."),
            hparams.from_lab_source("kda_head_dim", "kimi_k3.kda_head_dim", "KDA head dimension."),
            hparams.from_catalogue("kda_chunk_size", "kda", "chunk", "KDA chunk length."),
            hparams.from_lab_source("kda_conv_kernel", "kimi_k3.conv_kernel", "Short conv width."),
            hparams.from_lab_source("kda_decay_floor", "kda.k3_decay_floor", "K3 log-decay bound."),
            hparams.from_lab_source("kda_output_gate", "kda.k3_output_gate", "K3 KDA output gate."),
            hparams.from_lab_source("mla_heads", "kimi_k3.mla_heads", "MLA heads."),
            hparams.from_lab_source(
                "mla_head_dim", "kimi_k3.mla_head_dim", "MLA content width per head."
            ),
            hparams.from_lab_source("mla_kv_rank", "kimi_k3.mla_kv_rank", "MLA latent width."),
            hparams.from_lab_source("mla_q_rank", "kimi_k3.mla_q_rank", "MLA query latent width."),
            hparams.from_lab_source(
                "mla_rope_dim", "kimi_k3.mla_rope_dim", "MLA decoupled query/key width."
            ),
            hparams.ours(
                "mla_rope_base",
                10000.0,
                "RoPE base, used only when mla_position is RoPE.",
                "not read from the K3 sources; unused because its MLA layers use NoPE",
            ),
            hparams.from_lab_source(
                "mla_position", "kimi_k3.nope", "Position encoding of the MLA layers."
            ),
            hparams.from_lab_source(
                "mla_output_gate", "kimi_k3.mla_output_gate", "Gated MLA's output gate."
            ),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="lightning_hybrid",
        family="hybrid",
        summary=(
            "Stacks lightning-attention layers with a grouped-query softmax layer (RoPE on half of "
            "each head) after every seven, as MiniMax-01 does."
        ),
        parent="lightning_attention",
        covers=None,
        source="lab:lightning_attention.hybrid_ratio",
        checked_against=(
            "arXiv:2501.08313v1 §2 Model Architecture (a softmax block after every 7 transnormer "
            "blocks, 80 layers, GQA, RoPE on half the head dimension with base 10,000); layout, "
            "key/value heads and rotary width from the released config"
        ),
        factory=lightning_hybrid,
        lab=(
            _lab("d_model", 32, _D_MODEL),
            hparams.ours("num_layers", 8, _NUM_LAYERS, "lab scale, one block of the 7:1 pattern"),
            hparams.from_lab_source("ratio", "lightning_attention.hybrid_ratio", _RATIO),
            hparams.ours("pattern", "", _PATTERN, _PATTERN_NOTE),
            hparams.ours("final", "", _FINAL, "the source's layout has no extra final layers"),
            _lab("lin_heads", 4, "Lightning attention heads."),
            _lab("lin_head_dim", 8, "Lightning attention width per head."),
            hparams.ours(
                "lin_block_size",
                5,
                "Tokens per tile in the lightning pass.",
                "deliberately not a divisor of the test lengths so a ragged tile runs",
            ),
            hparams.ours(
                "lin_decay",
                True,
                "Apply the per-head decay from the released code.",
                "the released model decays its state; the paper text omits it",
            ),
            _lab("attn_heads", 4, "Softmax query heads."),
            _lab("attn_kv_heads", 2, "Softmax key/value heads (groups)."),
            _lab("attn_head_dim", 8, "Softmax width per head."),
            hparams.ours(
                "attn_rotary_dim",
                4,
                "Leading dimensions of each softmax head that RoPE rotates.",
                "half of the lab head width, the fraction the source states",
            ),
            hparams.from_lab_source(
                "attn_rope_base", "lightning_hybrid.rope_base", "RoPE base of the softmax layers."
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "lightning_attention.d_model", "Hidden size."),
            hparams.from_lab_source("num_layers", "lightning_attention.layers", "Layers."),
            hparams.from_lab_source("ratio", "lightning_attention.hybrid_ratio", _RATIO),
            hparams.ours("pattern", "", _PATTERN, _PATTERN_NOTE),
            hparams.ours("final", "", _FINAL, "the source's layout has no extra final layers"),
            hparams.from_lab_source("lin_heads", "lightning_attention.heads", "Heads per layer."),
            hparams.from_lab_source(
                "lin_head_dim", "lightning_attention.head_dim", "Dimensions per head."
            ),
            hparams.from_lab_source(
                "lin_block_size", "lightning_attention.block_size", "Tokens per tile."
            ),
            hparams.ours(
                "lin_decay",
                True,
                "Apply the per-head decay from the released code.",
                "present in the released code, not stated in the paper sections read",
            ),
            hparams.from_lab_source("attn_heads", "lightning_attention.heads", "Query heads."),
            hparams.from_lab_source(
                "attn_kv_heads", "lightning_hybrid.kv_heads", "Key/value heads."
            ),
            hparams.from_lab_source(
                "attn_head_dim", "lightning_attention.head_dim", "Dimensions per head."
            ),
            hparams.from_lab_source(
                "attn_rotary_dim", "lightning_hybrid.rotary_dim", "Rotated dimensions per head."
            ),
            hparams.from_lab_source(
                "attn_rope_base",
                "lightning_hybrid.rope_base",
                "RoPE base stated in §2 (the released config's rope_theta differs).",
            ),
        ),
        state_growth="grows",
    )
)
