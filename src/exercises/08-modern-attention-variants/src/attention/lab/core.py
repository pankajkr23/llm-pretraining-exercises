"""The origin and the full-attention family: where attention starts, and how it first got cheaper.

**The problem.** A translation model built from one recurrent network squeezes a whole sentence
into one fixed-size vector. Bahdanau attention (arXiv:1409.0473) removed that bottleneck: at every
output step the decoder scores *every* encoder state, turns the scores into weights with a softmax,
and reads a weighted average. Scaled dot-product attention (arXiv:1706.03762) kept that idea, made
the score a dot product, and pointed it at the sequence itself.

**What each variant changes, at code level.**

- `bahdanau_attention` — the origin. The score is a small network, `v^T tanh(W s + U h)`, not a dot
  product, and the queries read an *encoder memory* rather than earlier tokens. Nothing is kept
  between steps, so its state is only a position counter.
- `standard_attention` — `softmax(Q K^T / sqrt(d_k)) V` with several heads and a causal mask. At
  decode time every token's key and value are kept: the **KV cache**, which grows with every token.
  No position information is added here; positions are the position family's job.
- `mqa` — every query head reads **one** key/value head. The cache shrinks by the number of heads;
  the price is capacity, since all heads now look up the same keys.
- `gqa` — the query heads are split into groups and each group shares one key/value head. With one
  group it is `mqa`; with a group per head it is `standard_attention`. `GroupedAttention` is the one
  class behind all three, so those two identities hold by construction and are tested anyway.
- `flashattention` — *the same numbers*, computed differently. The full pass goes through
  `torch.nn.functional.scaled_dot_product_attention`, which uses a fused kernel where one exists.
  With `tiled=True` it instead runs Algorithm 1 of arXiv:2205.14135v2 in plain PyTorch: keys and
  values in blocks, queries in blocks, and a running maximum `m` and normaliser `l` per query so a
  row's softmax is assembled block by block without ever holding the full score grid. That loop is
  the teaching point; on a CPU it is slower, not faster, because the saving is in GPU memory reads.

**The trade-off.** The first three spend memory to keep every key and value, and MQA/GQA buy that
memory back with sharing. FlashAttention changes neither the result nor the cache — only where the
score grid lives while it is being computed.

Every linear map here has no bias term. The sources write the projections as matrices, and a bias
would be a choice the sources do not make.
"""

import math
from typing import Any

import torch
from torch import Tensor, nn

from attention.lab import hparams, ops
from attention.lab.base import Mixer, MixerSpec
from attention.lab.registry import register
from attention.lab.sources import LabSource, add

#: PyTorch's fused attention entry point; it picks a fused kernel where one exists.
sdpa = torch.nn.functional.scaled_dot_product_attention

# --- sources the catalogue does not carry ---------------------------------------------------------

_BAHDANAU = "Neural Machine Translation by Jointly Learning to Align and Translate"
_MQA = "Fast Transformer Decoding: One Write-Head is All You Need"
_FLASH = "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
_T5_XXL = "google/t5-v1_1-xxl config.json (the public T5.1.1 XXL checkpoint)"
_T5_XXL_URL = "https://huggingface.co/google/t5-v1_1-xxl/raw/main/config.json"
_T5_WHERE = (
    "config.json of the public T5.1.1 XXL checkpoint; the GQA paper uptrains T5.1.1 XXL "
    "(S3.1 Experimental setup) and does not print its head count"
)

add(
    LabSource(
        id="bahdanau_attention.d_model",
        value=1000,
        quote="Its decoder has 1000 hidden units.",
        where="S4.2 Models, arXiv:1409.0473v7",
        url="https://arxiv.org/html/1409.0473v7",
        title=_BAHDANAU,
        unit="hidden units",
    )
)
add(
    LabSource(
        id="bahdanau_attention.d_align",
        value=1000,
        quote="The number of hidden units in the alignment model n ′ is 1000.",
        where="Appendix A.2.3 Model Size, arXiv:1409.0473v7",
        url="https://arxiv.org/html/1409.0473v7",
        title=_BAHDANAU,
        unit="hidden units",
    )
)
add(
    LabSource(
        id="mqa.d_model",
        value=1024,
        quote="we use an encoder-decoder Transformer model with 6 layers, using d m o d e l = 1024",
        where="S4.1 Experimental Setup, arXiv:1911.02150v1",
        url="https://arxiv.org/html/1911.02150v1",
        title=_MQA,
        unit="dimensions",
    )
)
add(
    LabSource(
        id="mqa.n_heads",
        value=8,
        quote="h = 8 , d k = d v = 128 , learned positional embeddings",
        where="S4.1 Experimental Setup, arXiv:1911.02150v1",
        url="https://arxiv.org/html/1911.02150v1",
        title=_MQA,
        unit="heads",
    )
)
add(
    LabSource(
        id="mqa.head_dim",
        value=128,
        quote="h = 8 , d k = d v = 128 , learned positional embeddings",
        where="S4.1 Experimental Setup, arXiv:1911.02150v1",
        url="https://arxiv.org/html/1911.02150v1",
        title=_MQA,
        unit="dimensions",
    )
)
add(
    LabSource(
        id="gqa.d_model",
        value=4096,
        quote='"d_model": 4096',
        where=_T5_WHERE,
        url=_T5_XXL_URL,
        title=_T5_XXL,
        unit="dimensions",
    )
)
add(
    LabSource(
        id="gqa.n_heads",
        value=64,
        quote='"num_heads": 64',
        where=_T5_WHERE,
        url=_T5_XXL_URL,
        title=_T5_XXL,
        unit="heads",
    )
)
add(
    LabSource(
        id="gqa.head_dim",
        value=64,
        quote='"d_kv": 64',
        where=_T5_WHERE,
        url=_T5_XXL_URL,
        title=_T5_XXL,
        unit="dimensions",
    )
)
add(
    LabSource(
        id="flashattention.n_heads",
        value=8,
        quote="8 heads of dimension 64, and batch size 128",
        where="Appendix E.5 Full Benchmarking Results, 'Setup', arXiv:2205.14135v1",
        url="https://arxiv.org/html/2205.14135v1",
        title=_FLASH,
        unit="heads",
    )
)


# --- the origin: additive attention over an encoder memory ----------------------------------------


class BahdanauAttention(Mixer):
    """Additive attention: each query scores every memory slot with a one-layer network.

    Follows arXiv:1409.0473v7 Eq. 5–6 and Appendix A.1.2:
    `e_ij = v^T tanh(W s_{i-1} + U h_j)`, `alpha_ij = softmax_j(e_ij)`, `c_i = sum_j alpha_ij h_j`.
    Here the query `x_t` plays the part of the decoder state `s_{i-1}`, and the context vector
    `c_t` is projected back to `d_model` so the layer returns the lab's usual shape — in the paper
    `c_i` is fed into the decoder's recurrent update instead, which is outside this layer.

    Args:
        d_model: Width of a query (the decoder state, `n` in the paper).
        d_memory: Width of one memory slot (an annotation `h_j`, `2n` in the paper).
        d_align: Hidden width of the scoring network (`n'` in the paper).
    """

    def __init__(self, d_model: int, d_memory: int, d_align: int) -> None:
        """Build the three scoring matrices and the output projection."""
        super().__init__()
        self.w = nn.Linear(d_model, d_align, bias=False)
        self.u = nn.Linear(d_memory, d_align, bias=False)
        self.v = nn.Linear(d_align, 1, bias=False)
        self.out = nn.Linear(d_memory, d_model, bias=False)

    def align(self, x: Tensor, memory: Tensor) -> Tensor:
        """The alignment weights `[batch, queries, slots]`; each row sums to one."""
        # U h_j does not depend on the query, which is why the paper says it can be precomputed.
        keys = self.u(memory)[:, None, :, :]
        scores = self.v(torch.tanh(self.w(x)[:, :, None, :] + keys)).squeeze(-1)
        return torch.softmax(scores, dim=-1)

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Read the memory once per query; the only state is how many queries were seen."""
        if memory is None:
            raise ValueError("Bahdanau attention is cross-attention and needs `memory`")
        state = self.init_state(x.shape[0]) if state is None else state
        context = self.align(x, memory) @ memory
        return self.out(context), {"pos": state["pos"] + x.shape[1]}

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """Nothing is carried between queries, so the state is a counter and holds no tensor."""
        return {"pos": 0}


# --- the full-attention family --------------------------------------------------------------------


class GroupedAttention(Mixer):
    """Causal multi-head self-attention whose query heads share key/value heads in groups.

    `n_kv_heads == n_heads` is multi-head attention (arXiv:1706.03762v7 §3.2.2),
    `n_kv_heads == 1` is multi-query attention (arXiv:1911.02150v1 §3), and anything that divides
    `n_heads` in between is grouped-query attention (arXiv:2305.13245v1 §2.2).

    The KV cache holds keys and values at `n_kv_heads`, before they are repeated for the query
    heads — that is where the saving is.

    Args:
        d_model: Width of a token.
        n_heads: Query heads.
        n_kv_heads: Key/value heads; must divide `n_heads`.
        head_dim: Width of one head's query, key and value.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        """Build the four projections."""
        super().__init__()
        if n_kv_heads < 1 or n_heads % n_kv_heads:
            raise ValueError(f"n_kv_heads={n_kv_heads} must divide n_heads={n_heads}")
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
        """Attention over the whole cache; `offset` is how many keys precede the first query."""
        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
        return ops.attend(q, k, v, allowed=allowed)[0]

    def forward(
        self, x: Tensor, state: Any = None, memory: Tensor | None = None
    ) -> tuple[Tensor, Any]:
        """Project, append to the cache, attend causally, merge the heads."""
        if state is None:
            state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        q = ops.split_heads(self.q(x), self.n_heads)
        k = torch.cat([state["k"], ops.split_heads(self.k(x), self.n_kv_heads)], dim=2)
        v = torch.cat([state["v"], ops.split_heads(self.v(x), self.n_kv_heads)], dim=2)
        groups = self.n_heads // self.n_kv_heads
        y = self.mix(q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), state["pos"])
        return self.o(ops.merge_heads(y)), {"k": k, "v": v, "pos": state["pos"] + x.shape[1]}

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """An empty cache: zero tokens of keys and values at `n_kv_heads`."""
        dtype = self.q.weight.dtype if dtype is None else dtype
        device = self.q.weight.device if device is None else device
        empty = torch.zeros(batch, self.n_kv_heads, 0, self.head_dim, dtype=dtype, device=device)
        return {"k": empty, "v": empty.clone(), "pos": 0}


def tiled_attention(q: Tensor, k: Tensor, v: Tensor, allowed: Tensor, block: int) -> Tensor:
    """Softmax attention assembled block by block: Algorithm 1 of arXiv:2205.14135v2.

    The outer loop walks key/value blocks `K_j, V_j`, the inner loop query blocks `Q_i`. For each
    pair only the `B_r x B_c` score block `S_ij` exists. Each query row keeps a running maximum
    score `m` and a running sum `l` of `exp(score - m)`; when a new block arrives the row's output
    is rescaled to the new normaliser and the block's contribution is added (lines 9–13). The
    paper's Algorithm 1 writes `S = Q K^T`; the `1/sqrt(d)` scale (its Algorithm 2's `tau`) is
    applied here so the result equals the lab's `ops.attend`. One block size is used for both
    loops; the paper derives `B_c` and `B_r` from the SRAM size, which a CPU does not have.

    Args:
        q: `[batch, heads, queries, d]`.
        k: `[batch, heads, keys, d]`.
        v: `[batch, heads, keys, d_v]`.
        allowed: Boolean mask broadcastable to `[batch, heads, queries, keys]`.
        block: Rows and columns per block.

    Returns:
        `[batch, heads, queries, d_v]`.
    """
    if block < 1:
        raise ValueError(f"block must be at least 1, not {block}")
    b, h, n_q, d = q.shape
    n_k = k.shape[2]
    allowed = allowed.broadcast_to(b, h, n_q, n_k)
    scale = 1.0 / math.sqrt(d)
    starts = range(0, n_q, block)
    out = [q.new_zeros(b, h, min(block, n_q - i), v.shape[-1]) for i in starts]
    ell = [q.new_zeros(b, h, min(block, n_q - i)) for i in starts]
    top = [q.new_full((b, h, min(block, n_q - i)), float("-inf")) for i in starts]
    for j in range(0, n_k, block):
        k_j, v_j = k[:, :, j : j + block], v[:, :, j : j + block]
        for r, i in enumerate(starts):
            s_ij = (q[:, :, i : i + block] @ k_j.transpose(-2, -1)) * scale
            s_ij = s_ij.masked_fill(~allowed[:, :, i : i + block, j : j + block], float("-inf"))
            m_blk = s_ij.amax(dim=-1)
            # A row with every key in this block masked contributes nothing (exp(-inf) = 0).
            p_blk = torch.exp(s_ij - torch.where(m_blk.isfinite(), m_blk, 0.0)[..., None])
            l_blk = p_blk.sum(dim=-1)
            m_new = torch.maximum(top[r], m_blk)
            anchor = torch.where(m_new.isfinite(), m_new, 0.0)
            keep, add_ = torch.exp(top[r] - anchor), torch.exp(m_blk - anchor)
            l_new = keep * ell[r] + add_ * l_blk
            summed = (ell[r] * keep)[..., None] * out[r] + add_[..., None] * (p_blk @ v_j)
            safe = torch.where(l_new > 0, l_new, 1.0)
            out[r] = torch.where((l_new > 0)[..., None], summed / safe[..., None], 0.0)
            ell[r], top[r] = l_new, m_new
    return torch.cat(out, dim=2)


class FlashAttention(GroupedAttention):
    """Exact attention through a fused kernel, or through the paper's tiled loop.

    Same weights, same cache and same output as `GroupedAttention`; only `mix` differs.

    Args:
        d_model: Width of a token.
        n_heads: Query heads (each its own key/value head, as in the parent).
        head_dim: Width of one head.
        tiled: Use `tiled_attention` (Algorithm 1) instead of the fused call.
        block: Block size for the tiled loop.
    """

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, tiled: bool = False, block: int = 4
    ) -> None:
        """Build the parent's projections and remember how to compute the grid."""
        super().__init__(d_model, n_heads, n_heads, head_dim)
        self.tiled, self.block = tiled, block

    def mix(self, q: Tensor, k: Tensor, v: Tensor, offset: int) -> Tensor:
        """The fused call for a whole causal pass, or the tiled loop when asked for it."""
        if self.tiled:
            allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
            return tiled_attention(q, k, v, allowed, self.block)
        if offset == 0 and q.shape[2] == k.shape[2]:
            return sdpa(q, k, v, is_causal=True)
        # With a cache, `is_causal` would align the mask to the top-left corner, which is wrong.
        allowed = ops.causal_mask(q.shape[2], k.shape[2], offset, device=q.device)
        return sdpa(q, k, v, attn_mask=allowed)


def _standard(d_model: int, n_heads: int, head_dim: int) -> GroupedAttention:
    return GroupedAttention(d_model, n_heads, n_heads, head_dim)


def _mqa(d_model: int, n_heads: int, head_dim: int) -> GroupedAttention:
    return GroupedAttention(d_model, n_heads, 1, head_dim)


def _gqa(d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> GroupedAttention:
    return GroupedAttention(d_model, n_heads, n_kv_heads, head_dim)


# --- registrations --------------------------------------------------------------------------------

_LAB_WIDTH = "lab scale, small enough to run every test on a laptop CPU in seconds"
_D_MODEL = "Width of one token vector."
_HEADS = "How many query heads attend in parallel."
_HEAD_DIM = "Width of one head's query, key and value."

register(
    MixerSpec(
        name="bahdanau_attention",
        family="origin",
        summary=(
            "The first learned attention: a small network scores every encoder state for each "
            "decoder query, and the query reads their softmax-weighted average."
        ),
        parent=None,
        covers="bahdanau_attention",
        source="catalogue:bahdanau_attention",
        checked_against=(
            "arXiv:1409.0473v7 Eq. 5 and 6 (context vector, softmax weights, e_ij = a(s, h)) and "
            "Appendix A.1.2 (a = v^T tanh(W s + U h))"
        ),
        factory=BahdanauAttention,
        lab=(
            hparams.ours("d_model", 32, "Width of a query (the decoder state).", _LAB_WIDTH),
            hparams.ours("d_memory", 24, "Width of one encoder memory slot.", _LAB_WIDTH),
            hparams.ours("d_align", 16, "Hidden width of the scoring network.", _LAB_WIDTH),
        ),
        paper=(
            hparams.from_lab_source(
                "d_model", "bahdanau_attention.d_model", "Width of the decoder state."
            ),
            hparams.ours(
                "d_memory",
                2000,
                "Width of one annotation, the concatenated forward and backward encoder states.",
                "not printed as a number; twice the stated 1000 encoder units per direction, "
                "because each annotation concatenates both directions (S3.2)",
            ),
            hparams.from_lab_source(
                "d_align", "bahdanau_attention.d_align", "Hidden width of the alignment model."
            ),
        ),
        cross=True,
        state_growth="constant",
    )
)

register(
    MixerSpec(
        name="standard_attention",
        family="full",
        summary=(
            "Scaled dot-product attention over the sequence itself, split into heads, with a "
            "causal mask and a KV cache that keeps every key and value."
        ),
        parent=None,
        covers="standard_attention",
        source="catalogue:standard_attention",
        checked_against=(
            "arXiv:1706.03762v7 §3.2.1 Eq. 1 (softmax(QK^T/sqrt(d_k))V), §3.2.2 (MultiHead with "
            "W^O) and §3.2.3 (masking illegal connections with -inf)"
        ),
        factory=_standard,
        lab=(
            hparams.ours("d_model", 32, _D_MODEL, _LAB_WIDTH),
            hparams.ours("n_heads", 4, _HEADS, _LAB_WIDTH),
            hparams.ours("head_dim", 8, _HEAD_DIM, _LAB_WIDTH),
        ),
        paper=(
            hparams.from_catalogue("d_model", "sinusoidal", "dims", _D_MODEL),
            hparams.from_catalogue("n_heads", "standard_attention", "heads", _HEADS),
            hparams.from_catalogue("head_dim", "standard_attention", "headDim", _HEAD_DIM),
        ),
    )
)

register(
    MixerSpec(
        name="mqa",
        family="full",
        summary=(
            "Every query head reads one shared key/value head, so the KV cache is n_heads "
            "times smaller."
        ),
        parent="standard_attention",
        covers="mqa",
        source="catalogue:mqa",
        checked_against=(
            "arXiv:1911.02150v1 §3 Multi-Query Attention (the heads share a single set of keys "
            "and values)"
        ),
        factory=_mqa,
        lab=(
            hparams.ours("d_model", 32, _D_MODEL, _LAB_WIDTH),
            hparams.ours("n_heads", 4, _HEADS, _LAB_WIDTH),
            hparams.ours("head_dim", 8, _HEAD_DIM, _LAB_WIDTH),
        ),
        paper=(
            hparams.from_lab_source("d_model", "mqa.d_model", _D_MODEL),
            hparams.from_lab_source("n_heads", "mqa.n_heads", _HEADS),
            hparams.from_lab_source("head_dim", "mqa.head_dim", _HEAD_DIM),
        ),
    )
)

register(
    MixerSpec(
        name="gqa",
        family="full",
        summary=(
            "Query heads are split into n_kv_heads groups and each group shares one key/value "
            "head, between MQA (one group) and multi-head attention (a group per head)."
        ),
        parent="mqa",
        covers="gqa",
        source="catalogue:gqa",
        checked_against="arXiv:2305.13245v1 §2.2 Grouped-query attention",
        factory=_gqa,
        lab=(
            hparams.ours("d_model", 32, _D_MODEL, _LAB_WIDTH),
            hparams.ours("n_heads", 4, _HEADS, _LAB_WIDTH),
            hparams.ours(
                "n_kv_heads",
                2,
                "Key/value heads, one per group.",
                "lab scale; two groups is strictly between MQA and multi-head attention",
            ),
            hparams.ours("head_dim", 8, _HEAD_DIM, _LAB_WIDTH),
        ),
        paper=(
            hparams.from_lab_source("d_model", "gqa.d_model", _D_MODEL),
            hparams.from_lab_source("n_heads", "gqa.n_heads", _HEADS),
            hparams.from_catalogue(
                "n_kv_heads", "gqa", "kvHeads", "Key/value heads, one per group."
            ),
            hparams.from_lab_source("head_dim", "gqa.head_dim", _HEAD_DIM),
        ),
    )
)

register(
    MixerSpec(
        name="flashattention",
        family="full",
        summary=(
            "The same exact attention as its parent, computed by a fused kernel or by tiling the "
            "score grid into blocks so the full grid is never held at once."
        ),
        parent="standard_attention",
        covers="flashattention",
        source="catalogue:flashattention",
        checked_against=(
            "arXiv:2205.14135v2 Algorithm 1 (block loop, running m and l, lines 9–13) and "
            "Theorem 1 (returns softmax(QK^T)V exactly)"
        ),
        factory=FlashAttention,
        lab=(
            hparams.ours("d_model", 32, _D_MODEL, _LAB_WIDTH),
            hparams.ours("n_heads", 4, _HEADS, _LAB_WIDTH),
            hparams.ours("head_dim", 8, _HEAD_DIM, _LAB_WIDTH),
            hparams.ours(
                "tiled",
                True,
                "Run the paper's tiled loop instead of the fused kernel.",
                "on so the generic tests exercise the Algorithm 1 loop, the teaching point",
            ),
            hparams.ours(
                "block",
                5,
                "Rows and columns per block in the tiled loop.",
                "lab choice; five does not divide twelve, so ragged final blocks get tested",
            ),
        ),
        paper=(
            hparams.ours(
                "d_model",
                512,
                _D_MODEL,
                "not stated as a model width; follows from the stated 8 heads of dimension 64",
            ),
            hparams.from_lab_source("n_heads", "flashattention.n_heads", _HEADS),
            hparams.from_catalogue("head_dim", "flashattention", "headDim", _HEAD_DIM),
            hparams.ours(
                "tiled",
                False,
                "Run the paper's tiled loop instead of the fused kernel.",
                "at full scale the fused kernel is the practical path; the loop is for reading",
            ),
            hparams.ours(
                "block",
                256,
                "Rows and columns per block in the tiled loop.",
                "not stated as a default; the paper derives block sizes from SRAM size M and "
                "width d, and 256 is where its Figure 2 runtime stops improving",
            ),
        ),
    )
)
