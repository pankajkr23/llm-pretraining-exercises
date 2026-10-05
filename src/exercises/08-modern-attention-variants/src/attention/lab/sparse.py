"""The sparse family: attention in which each query reads only some of the earlier tokens.

**The problem.** Full attention compares every query with every earlier key, so its compute grows
with the square of the context, and its cache holds every earlier token. Most of those comparisons
end up with almost no weight. Every variant here chooses, per query, a *subset* of keys and runs an
ordinary softmax over that subset only. They differ in how the subset is chosen:

| variant | the subset is | chosen by | cache |
| --- | --- | --- | --- |
| `sliding_window` | the last `window` tokens | position | bounded |
| `sparse_attention` | a fixed stride/block pattern | position | grows |
| `topk_attention` | the `topk` highest-scoring keys | the full score row | grows |
| `reformer` | keys hashed into the query's bucket | random rotations of the content | grows |
| `attention_sinks` | the first `sinks` tokens plus a window | position | bounded |
| `nsa` | compressed blocks, top blocks, and a window | a learned score per block | grows |
| `deepseek_csa` | top compressed entries, and a window | a separate learned indexer | grows |
| `msa` | top token blocks per KV group | a separate learned indexer | grows |

**What changes in code, relative to the parent.** Every variant computes queries, keys and values
the way its parent does and calls the same `ops.attend`; what changes is the boolean `allowed` mask
handed to it (and, for the learned selectors, the extra projections that decide that mask). So the
sparsity is always visible as a mask, and the tests can compare it with sets written out by hand
from the papers' definitions.

**The trade-off.** A token outside the subset contributes nothing. A positional pattern can miss
the one far token that matters; a content-based selector can pick the wrong one, and every learned
selector adds parameters and a non-differentiable top-k. The lab computes each subset as an
explicit mask, which shows the mechanism exactly but spends full quadratic compute — it measures
*what* each variant reads, not how fast a kernel would read it.

Positions are absolute throughout: `pos` in the state is the number of tokens already seen, and the
cache records enough to know each cached key's position, so one call over a sequence and one call
per token build identical masks.
"""

import math

import torch
from torch import Tensor, nn

from attention.lab import hparams, ops
from attention.lab.base import Mixer, MixerSpec
from attention.lab.registry import register
from attention.lab.sources import LabSource, add

_LAB_NOTE = "lab scale, small enough to run on a laptop"


def _lab(name: str, value, meaning: str, note: str = _LAB_NOTE):
    return hparams.ours(name, value, meaning, note)


def _positions(start: int, count: int, device) -> Tensor:
    return torch.arange(start, start + count, device=device)


# --- a shared base: projections plus a cache of keys and values -----------------------------------


class _CachedAttention(Mixer):
    """Q/K/V/O projections, a key/value cache, and a mask chosen by the subclass.

    Subclasses implement `allowed(q_pos, k_pos)`, a boolean `[queries, keys]` or
    `[heads, queries, keys]` mask over absolute positions. `keep` bounds the cache to the most
    recent keys; `None` keeps every key.
    """

    keep: int | None = None

    def __init__(self, d_model: int, heads: int, context: int | None = None) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model} is not divisible by heads {heads}")
        self.heads, self.head_dim, self.context = heads, d_model // heads, context
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """An empty key/value cache."""
        dtype = dtype or self.w_q.weight.dtype
        empty = torch.zeros(batch, self.heads, 0, self.head_dim, device=device, dtype=dtype)
        return {"k": empty, "v": empty.clone(), "pos": 0}

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Queries, keys and values, split into heads."""
        return (
            ops.split_heads(self.w_q(x), self.heads),
            ops.split_heads(self.w_k(x), self.heads),
            ops.split_heads(self.w_v(x), self.heads),
        )

    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Which absolute key positions each absolute query position may read."""
        raise NotImplementedError

    def mix(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Attention of `q` over `k, v` under `allowed`; subclasses may change the scores."""
        out, _ = ops.attend(q, k, v, allowed=self.allowed(q_pos, k_pos))
        return out

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Append this chunk to the cache, attend under the variant's mask, trim the cache."""
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        pos, tokens = state["pos"], x.shape[1]
        if self.context is not None and pos + tokens > self.context:
            raise ValueError(f"{pos + tokens} tokens exceed the context of {self.context}")
        q, k, v = self.project(x)
        k = torch.cat([state["k"], k], dim=2)
        v = torch.cat([state["v"], v], dim=2)
        k_pos = _positions(pos + tokens - k.shape[2], k.shape[2], x.device)
        q_pos = _positions(pos, tokens, x.device)
        y = self.w_o(ops.merge_heads(self.mix(q, k, v, q_pos, k_pos)))
        if self.keep is not None:
            k, v = k[:, :, -self.keep :], v[:, :, -self.keep :]
        return y, {"k": k, "v": v, "pos": pos + tokens}


# --- sliding window (Longformer) ------------------------------------------------------------------


def window_mask(q_pos: Tensor, k_pos: Tensor, window: int) -> Tensor:
    """True where key j is one of the last `window` positions up to and including query i."""
    i, j = q_pos[:, None], k_pos[None, :]
    return (j <= i) & (j > i - window)


class SlidingWindowAttention(_CachedAttention):
    """Each query reads itself and the `window - 1` tokens before it; the cache keeps `window`.

    Longformer (arXiv:2004.05150v2 §3.1) defines the window for a bidirectional encoder: a window
    of size w covers w/2 tokens on each side. A causal decoder can only look back, so this lab's
    `window` counts the keys a query reads, the query itself included. That is our reading, not a
    definition from the paper.
    """

    def __init__(self, d_model: int, heads: int, window: int) -> None:
        """Standard attention whose mask is a band of width `window`."""
        super().__init__(d_model, heads)
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = self.keep = window

    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """The band of the last `window` positions."""
        return window_mask(q_pos, k_pos, self.window)


# --- factorized patterns (Sparse Transformer) -----------------------------------------------------

_ST_URL = "https://arxiv.org/html/1904.10509v1"
_ST_TITLE = "Generating Long Sequences with Sparse Transformers"
_ST_WHERE = "§7.2 Text, arXiv:1904.10509v1"
_ST_RECIPE = "We used a stride of 128, c = 32 , and merged the factorized attention heads."
_ST_MODEL = (
    "We used 30-layer fixed Sparse Transformers with 8 heads, d = 512, and a dropout rate of 0.40 ."
)
add(LabSource("sparse_attention.c", 32, _ST_RECIPE, _ST_WHERE, _ST_URL, _ST_TITLE, "positions"))
add(LabSource("sparse_attention.combine", "merged", _ST_RECIPE, _ST_WHERE, _ST_URL, _ST_TITLE))
add(LabSource("sparse_attention.pattern", "fixed", _ST_MODEL, _ST_WHERE, _ST_URL, _ST_TITLE))
add(LabSource("sparse_attention.d_model", 512, _ST_MODEL, _ST_WHERE, _ST_URL, _ST_TITLE, "dims"))

PATTERNS = ("strided", "fixed")
COMBINES = ("merged", "separate")


def factorized_masks(
    pattern: str, q_pos: Tensor, k_pos: Tensor, stride: int, c: int
) -> tuple[Tensor, Tensor]:
    """The two index sets `A^(1)` and `A^(2)` of §4.3, as boolean `[queries, keys]` masks.

    Both are subsets of `{j : j <= i}` (§4.2), so causality is applied to each.

    - strided: `A1 = {t, ..., i}` with `t = max(0, i - l)`; `A2 = {j : (i - j) mod l = 0}`.
    - fixed: `A1 = {j : floor(j / l) = floor(i / l)}`; `A2 = {j : j mod l ∈ {t, ..., l}}` with
      `t = l - c`.

    **An ambiguity in the fixed pattern, resolved one way.** `j mod l` never equals `l`, so with
    0-indexed positions the set `{l - c, ..., l}` is in effect `{l - c, ..., l - 1}`, which has `c`
    members — matching the paper's remark that the pattern costs `c` times the strided one. The
    paper's worked example ("positions 120-128" for stride 128, c = 8) names nine positions, which
    reads as a 1-indexed count; this lab uses the 0-indexed formula as written.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}, not {pattern!r}")
    i, j = q_pos[:, None], k_pos[None, :]
    causal = j <= i
    if pattern == "strided":
        first = (j >= i - stride) & causal
        second = ((i - j) % stride == 0) & causal
    else:
        first = (
            torch.div(j, stride, rounding_mode="floor")
            == torch.div(i, stride, rounding_mode="floor")
        ) & causal
        second = (j % stride >= stride - c) & causal
    return first, second


class FactorizedSparseAttention(_CachedAttention):
    """Multi-head attention whose heads read a factorized strided or fixed pattern.

    `combine="merged"` gives every head the union `A1 ∪ A2` (§5.1 Eq 7, used inside multi-head
    attention as Eq 8 allows). `combine="separate"` gives even heads `A1` and odd heads `A2`, the
    other reading of Eq 8. The interleaved form (Eq 6) alternates patterns across residual blocks,
    which is a property of a stack rather than of one layer, so it is not offered here. Nor is the
    paper's refinement of giving different heads different sub-blocks of length c.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        pattern: str,
        stride: int,
        c: int,
        combine: str,
        context: int,
    ) -> None:
        """Standard attention with a factorized mask, for sequences up to `context` tokens."""
        super().__init__(d_model, heads, context)
        if combine not in COMBINES:
            raise ValueError(f"combine must be one of {COMBINES}, not {combine!r}")
        if not 1 <= c <= stride:
            raise ValueError("c must be between 1 and the stride")
        self.pattern, self.stride, self.c, self.combine = pattern, stride, c, combine

    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """`[heads, queries, keys]`: the merged union, or alternating sets per head."""
        first, second = factorized_masks(self.pattern, q_pos, k_pos, self.stride, self.c)
        if self.combine == "merged":
            return (first | second).expand(self.heads, -1, -1)
        return torch.stack([first if h % 2 == 0 else second for h in range(self.heads)])


def _sparse_attention(**kw) -> FactorizedSparseAttention:
    return FactorizedSparseAttention(**kw)


# --- top-k (Explicit Sparse Transformer) ----------------------------------------------------------


class TopKAttention(_CachedAttention):
    """Keep, in each score row, only the entries at least as large as the k-th largest.

    Eq 1: `P = Q K^T / sqrt(d)`. Eq 2: `M(P, k)_ij = P_ij` if `P_ij >= t_i` (the k-th largest value
    of row i), else `-inf` — so ties with `t_i` are all kept. Eq 3–4: softmax, then a weighted sum
    of values. The paper treats `t_i` as a constant for back-propagation, so it is detached.

    **Not stated in the paper: the order of causal masking and top-k.** This lab masks the future
    first and then takes the top k of what remains, so a query with fewer than k earlier tokens
    reads all of them. The paper describes a single head; this lab applies the rule in each head.
    """

    def __init__(self, d_model: int, heads: int, topk: int) -> None:
        """Standard attention with a per-row top-k mask."""
        super().__init__(d_model, heads)
        if topk < 1:
            raise ValueError("topk must be at least 1")
        self.topk = topk

    def allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Causality only; the top-k part depends on the scores and is applied in `mix`."""
        return k_pos[None, :] <= q_pos[:, None]

    def mix(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Eq 1–4, with the causal mask applied before the selection."""
        causal = self.allowed(q_pos, k_pos)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # Eq 1
        scores = scores.masked_fill(~causal, float("-inf"))
        kth = min(self.topk, k.shape[2])
        threshold = scores.detach().topk(kth, dim=-1).values[..., -1:]  # t_i, a constant
        keep = causal & (scores >= threshold)  # Eq 2, ties kept
        out, _ = ops.attend(q, k, v, allowed=keep)  # Eq 3–4
        return out


# --- LSH attention (Reformer) ---------------------------------------------------------------------

_RF_URL = "https://arxiv.org/html/2001.04451v2"
_RF_TITLE = "Reformer: The Efficient Transformer"
_RF_QUOTE = (
    "All experiments have d m o d e l = 1024 , d f f = 4096 , n h e a d s = 8 , and a total batch "
    "size of 8 sequences."
)
_RF_WHERE = "§5 Experiments, arXiv:2001.04451v2"
add(LabSource("reformer.d_model", 1024, _RF_QUOTE, _RF_WHERE, _RF_URL, _RF_TITLE, "dims"))
add(LabSource("reformer.heads", 8, _RF_QUOTE, _RF_WHERE, _RF_URL, _RF_TITLE, "heads"))


def lsh_buckets(x: Tensor, rotations: Tensor) -> Tensor:
    """`h(x) = argmax([xR ; -xR])` for each round: `[..., rounds]` bucket ids.

    `x` is `[batch, heads, tokens, d_k]`; `rotations` is `[heads, rounds, d_k, b/2]`, or has a last
    dimension of 0 for the one-bucket case, where every vector hashes to bucket 0.
    """
    if rotations.shape[-1] == 0:
        return torch.zeros(*x.shape[:-1], rotations.shape[1], dtype=torch.long, device=x.device)
    projected = torch.einsum("bhtd,hrdc->bhtrc", x, rotations.to(x.dtype))
    return torch.cat([projected, -projected], dim=-1).argmax(dim=-1)


class LSHAttention(_CachedAttention):
    """Shared-QK attention restricted to keys that hash into the query's bucket in some round.

    - Shared QK (§2): one projection makes the queries, and keys are `k_j = q_j / ||q_j||`, so a
      key always hashes like its own query. Values have their own projection.
    - Hash (§2): a fixed random matrix `R` of shape `[d_k, b/2]` per round, and
      `h(x) = argmax([xR ; -xR])`. The matrices are a seeded buffer, so a stepwise decode hashes
      exactly as the full pass does.
    - Attention set (Eq 4, 6): `P_i` is every earlier key that shares the query's bucket in at
      least one of `n_rounds` rounds. Appendix A (Eq 12–16) shows the multi-round computation,
      with its `N_ij` correction, equals one softmax over that union, which is what runs here.
    - Self (§2, Eq 16): a token does not attend to itself unless it has no other target. The
      paper does this with a large finite penalty on the diagonal (`self_penalty`), so a query
      with other targets gives itself a weight that underflows to zero, and a query with none
      still reads itself.

    **Not reproduced: sorting and chunking (Eq 5).** Sorting by bucket and letting chunks of `m`
    queries attend within and one chunk back is how the paper *batches* Eq 4; it can drop a
    same-bucket key when a bucket overflows its chunk. This lab evaluates Eq 4 exactly, so it shows
    what LSH attention computes rather than how fast. The scores are scaled by `1/sqrt(d_k)`, which
    the paper omits from its equations only for clarity.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        n_buckets: int,
        n_rounds: int,
        self_penalty: float,
        seed: int,
    ) -> None:
        """Shared-QK attention with fixed random rotations."""
        super().__init__(d_model, heads)
        if n_buckets != 1 and n_buckets % 2:
            raise ValueError("n_buckets must be even (R has b/2 columns), or 1")
        del self.w_k  # shared QK: the query projection makes the keys
        self.n_buckets, self.n_rounds, self.self_penalty = n_buckets, n_rounds, self_penalty
        generator = torch.Generator().manual_seed(seed)
        rotations = torch.randn(heads, n_rounds, self.head_dim, n_buckets // 2, generator=generator)
        self.register_buffer("rotations", rotations)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Queries, unit-norm keys from the same projection, and values."""
        q = ops.split_heads(self.w_q(x), self.heads)
        k = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        return q, k, ops.split_heads(self.w_v(x), self.heads)

    def mix(self, q: Tensor, k: Tensor, v: Tensor, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Eq 4 and 6 as a mask, with the self penalty of Eq 16 as a bias."""
        q_hash = lsh_buckets(q, self.rotations)  # [b, h, Q, rounds]
        k_hash = lsh_buckets(k, self.rotations)  # [b, h, K, rounds]
        same = (q_hash[:, :, :, None, :] == k_hash[:, :, None, :, :]).any(dim=-1)
        causal = k_pos[None, :] <= q_pos[:, None]
        is_self = (k_pos[None, :] == q_pos[:, None]).to(q.dtype)
        out, _ = ops.attend(q, k, v, allowed=same & causal, bias=-self.self_penalty * is_self)
        return out


def _reformer(**kw) -> LSHAttention:
    return LSHAttention(**kw)


# --- attention sinks (StreamingLLM) ---------------------------------------------------------------

_SL_URL = "https://arxiv.org/html/2309.17453v4"
_SL_TITLE = "Efficient Streaming Language Models with Attention Sinks"
add(
    LabSource(
        "attention_sinks.window",
        1020,
        "The perplexity is restored when we reintroduce the initial four tokens alongside the "
        "recent 1020 tokens (4+1020).",
        "Table 1 caption, arXiv:2309.17453v4",
        _SL_URL,
        _SL_TITLE,
        "tokens",
    )
)


def in_cache_positions(
    q_pos: Tensor, k_pos: Tensor, sinks: int, window: int
) -> tuple[Tensor, Tensor, Tensor]:
    """What each query reads under StreamingLLM, and the positions RoPE uses for it.

    A query at absolute position i reads the first `sinks` tokens and the last `window` tokens up
    to itself. Positions are assigned *within that set* (§3.2): the visible keys are numbered 0, 1,
    2, ... in order, and the query takes the last number.

    Returns:
        `visible` (`[queries, keys]` bool), `key_rank` (`[queries, keys]`, meaningful where
        visible) and `query_rank` (`[queries]`).
    """
    i, j = q_pos[:, None], k_pos[None, :]
    visible = (j <= i) & ((j < sinks) | (j > i - window))
    key_rank = visible.long().cumsum(dim=-1) - 1
    query_rank = visible.sum(dim=-1) - 1
    return visible, key_rank, query_rank


class SinkAttention(Mixer):
    """A window plus a few permanent initial tokens, with RoPE applied at positions in the cache.

    The cache holds the first `sinks` tokens and the last `window` tokens, **before** rotation;
    every step rotates the keys at their current place in the cache (§3.2), so the same token can
    carry different positions at different steps. Because the rotation depends on the query, the
    full pass rotates the keys once per query — the same arithmetic a streaming decode performs.

    The parent (`sliding_window`) uses no position encoding. This variant adds RoPE because
    StreamingLLM's method is defined by how positions are assigned inside the cache.
    """

    def __init__(self, d_model: int, heads: int, sinks: int, window: int, rope_base: float):
        """Projections, and the sizes of the two parts of the cache."""
        super().__init__()
        if d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model / heads must be a whole, even number for RoPE")
        self.heads, self.head_dim = heads, d_model // heads
        self.sinks, self.window, self.rope_base = sinks, window, rope_base
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """An empty cache of un-rotated keys, values, and their absolute positions."""
        dtype = dtype or self.w_q.weight.dtype
        empty = torch.zeros(batch, self.heads, 0, self.head_dim, device=device, dtype=dtype)
        return {
            "k": empty,
            "v": empty.clone(),
            "k_pos": torch.zeros(0, dtype=torch.long, device=device),
            "pos": 0,
        }

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Attend with in-cache positions, then evict everything but the sinks and the window."""
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        pos = state["pos"]
        q = ops.split_heads(self.w_q(x), self.heads)
        k = torch.cat([state["k"], ops.split_heads(self.w_k(x), self.heads)], dim=2)
        v = torch.cat([state["v"], ops.split_heads(self.w_v(x), self.heads)], dim=2)
        k_pos = torch.cat([state["k_pos"], _positions(pos, tokens, x.device)])
        q_pos = _positions(pos, tokens, x.device)

        visible, key_rank, query_rank = in_cache_positions(q_pos, k_pos, self.sinks, self.window)
        key_angles = ops.rope_angles(key_rank.clamp_min(0).flatten(), self.head_dim, self.rope_base)
        key_angles = key_angles.view(tokens, len(k_pos), -1)
        # One rotated copy of the cached keys per query: [b, h, queries, keys, d].
        k_rot = ops.apply_rope(k[:, :, None].expand(-1, -1, tokens, -1, -1), key_angles)
        q_rot = ops.apply_rope(q, ops.rope_angles(query_rank, self.head_dim, self.rope_base))
        v_rep = v[:, :, None].expand(-1, -1, tokens, -1, -1)
        out, _ = ops.attend(q_rot[:, :, :, None], k_rot, v_rep, allowed=visible[:, None, :])
        y = self.w_o(ops.merge_heads(out[:, :, :, 0]))

        end = pos + tokens
        keep = (k_pos < self.sinks) | (k_pos >= end - self.window)
        return y, {"k": k[:, :, keep], "v": v[:, :, keep], "k_pos": k_pos[keep], "pos": end}


# --- shared pieces of the learned block selectors -------------------------------------------------


def _topk_mask(scores: Tensor, k: int) -> Tensor:
    """True at the `k` largest finite entries of the last dimension (all finite ones if fewer)."""
    size = scores.shape[-1]
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    if k <= 0 or size == 0:
        return chosen
    index = scores.topk(min(k, size), dim=-1).indices
    chosen.scatter_(-1, index, True)
    return chosen & (scores > float("-inf"))


def _rope_tail(x: Tensor, positions: Tensor, rope_dim: int, base: float) -> Tensor:
    """Rotate the last `rope_dim` dimensions of `x` at `positions`; leave the rest alone."""
    if rope_dim == 0:
        return x
    angles = ops.rope_angles(positions, rope_dim, base)
    head, tail = x[..., : x.shape[-1] - rope_dim], x[..., x.shape[-1] - rope_dim :]
    return torch.cat([head, ops.apply_rope(tail, angles)], dim=-1)


# --- NSA ------------------------------------------------------------------------------------------

_NSA_URL = "https://arxiv.org/html/2502.11089v2"
_NSA_TITLE = "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
_NSA_WHERE = "§4.1 Pretraining Setup, arXiv:2502.11089v2"
_NSA_GQA = "For GQA, we set the number of groups to 4, with a total of 64 attention heads."
_NSA_HEAD = (
    "For each head, the hidden dimensions of the query, key, and value are configured as "
    "d q = d k = 192 and d v = 128 , respectively."
)
_NSA_FORCED = (
    "selected block count n = 16 (including fixed activating the 1 initial block and 2 local "
    "blocks)"
)
for _id, _value, _quote, _unit in (
    ("nsa.d_model", 2560, "The model consists of 30 layers with a hidden dimension of 2560.", "d"),
    ("nsa.heads", 64, _NSA_GQA, "heads"),
    ("nsa.kv_heads", 4, _NSA_GQA, "groups"),
    ("nsa.head_dim", 192, _NSA_HEAD, "dims"),
    ("nsa.value_dim", 128, _NSA_HEAD, "dims"),
    ("nsa.initial_blocks", 1, _NSA_FORCED, "blocks"),
    ("nsa.local_blocks", 2, _NSA_FORCED, "blocks"),
):
    add(LabSource(_id, _value, _quote, _NSA_WHERE, _NSA_URL, _NSA_TITLE, _unit))


def nsa_compressed_visible(q_pos: Tensor, blocks: int, block: int, stride: int) -> Tensor:
    """`[queries, blocks]`: compressed block i is visible once all its tokens have been seen.

    Eq 7 builds, for a query that has seen t tokens (1-indexed), the blocks
    `φ(k_{id+1 : id+l})` for `0 <= i <= floor((t - l) / d)` — that is, `i·d + l <= t`. With a
    0-indexed query position p, t = p + 1, so block i is visible when `i·d + l <= p + 1`.
    """
    i = torch.arange(blocks, device=q_pos.device)[None, :]
    return i * stride + block <= q_pos[:, None] + 1


def nsa_selection_map(selected_blocks: int, compressed: int, block: int, stride: int, sel: int):
    """The matrix of Eq 9: `p_slc[j] = Σ_m Σ_n p_cmp[(l'/d)·j - m - n]`, as `[sel_blocks, cmp]`.

    Indices are read 0-based as written; an index outside `[0, compressed)` contributes nothing.
    When `l' = l = d` this is the identity, as the paper says.
    """
    out = torch.zeros(selected_blocks, compressed)
    for j in range(selected_blocks):
        for m in range(sel // stride):
            for n in range(block // stride):
                index = (sel // stride) * j - m - n
                if 0 <= index < compressed:
                    out[j, index] += 1
    return out


_BRANCHES = ("cmp", "slc", "win")


class _BlockCompressor(nn.Module):
    """φ of Eq 7: a learnable MLP with intra-block position encoding, one block to one vector.

    **φ's architecture is not stated in the paper.** This lab adds a learned position embedding to
    each of the `block` vectors, flattens them, and applies Linear → GELU → Linear.
    """

    def __init__(self, dim: int, block: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(block, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(block * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, blocks: Tensor) -> Tensor:
        """`[..., n, block, dim] -> [..., n, dim]`."""
        return self.mlp((blocks + self.position).flatten(-2))


class NativeSparseAttention(Mixer):
    """NSA: three gated branches — compressed blocks, selected blocks, a sliding window.

    Eq 5: `o = Σ_c g_c · Attn(q, K_c, V_c)` over c ∈ {cmp, slc, win}, where each gate is a sigmoid
    of an MLP of the token (**the gate MLP's architecture is not stated**; this lab uses
    Linear → GELU → Linear, one gate per branch per token). The query projection is shared and each
    branch has its own keys and values (§3.3.3). Keys and values are GQA-shared.

    - **cmp** (Eq 7): blocks of `block` keys, `stride` apart, each compressed by φ. Block i is
      visible once complete (see `nsa_compressed_visible`).
    - **slc** (Eq 8–12): the compression branch's attention weights score the compressed blocks
      (this lab reuses those weights, including their `1/sqrt(d)`, as "intermediate attention
      scores"; Eq 8 prints the softmax without a scale). Eq 9 maps them onto selection blocks of
      `sel_block` tokens, Eq 10 sums them over the heads of a KV group, and the top `selected`
      blocks per group are read token by token, up to the query. `initial_blocks` and the
      `local_blocks` most recent blocks are always selected and count towards `selected`; the
      paper does not define "local" further, and this lab takes it to mean the block holding the
      query and the ones just before it.
    - **win** (§3.3.3): the last `window` tokens, the query included.

    NSA's sections read here do not describe a position encoding, and this lab adds none.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        value_dim: int,
        block: int,
        stride: int,
        sel_block: int,
        selected: int,
        initial_blocks: int,
        local_blocks: int,
        window: int,
        context: int,
    ) -> None:
        """Projections for three branches, φ for keys and values, and the gate."""
        super().__init__()
        if heads % kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if block % stride or sel_block % stride or block > sel_block:
            raise ValueError("Eq 9 needs d | l, d | l' and l <= l'")
        if initial_blocks + local_blocks > selected:
            raise ValueError("the forced blocks count towards `selected`")
        self.heads, self.kv_heads, self.head_dim, self.value_dim = (
            heads,
            kv_heads,
            head_dim,
            value_dim,
        )
        self.block, self.stride, self.sel_block, self.selected = block, stride, sel_block, selected
        self.initial_blocks, self.local_blocks = initial_blocks, local_blocks
        self.window, self.context = window, context
        self.w_q = nn.Linear(d_model, heads * head_dim, bias=False)
        self.w_k = nn.ModuleDict(
            {c: nn.Linear(d_model, kv_heads * head_dim, bias=False) for c in _BRANCHES}
        )
        self.w_v = nn.ModuleDict(
            {c: nn.Linear(d_model, kv_heads * value_dim, bias=False) for c in _BRANCHES}
        )
        self.phi_k = _BlockCompressor(head_dim, block)
        self.phi_v = _BlockCompressor(value_dim, block)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(_BRANCHES))
        )
        self.w_o = nn.Linear(heads * value_dim, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """Empty caches for each branch, and an empty tail of not-yet-compressed keys."""
        dtype = dtype or self.w_q.weight.dtype

        def empty(width: int) -> Tensor:
            return torch.zeros(batch, self.kv_heads, 0, width, device=device, dtype=dtype)

        return {
            "cmp_k": empty(self.head_dim),
            "cmp_v": empty(self.value_dim),
            "tail_k": empty(self.head_dim),
            "tail_v": empty(self.value_dim),
            "slc_k": empty(self.head_dim),
            "slc_v": empty(self.value_dim),
            "win_k": empty(self.head_dim),
            "win_v": empty(self.value_dim),
            "pos": 0,
        }

    def _kv(self, x: Tensor, branch: str) -> tuple[Tensor, Tensor]:
        return (
            ops.split_heads(self.w_k[branch](x), self.kv_heads),
            ops.split_heads(self.w_v[branch](x), self.kv_heads),
        )

    def _compress(self, state: dict, k: Tensor, v: Tensor, end: int) -> tuple[Tensor, ...]:
        """Extend the compressed cache with every block completed by `end` tokens."""
        done = state["cmp_k"].shape[2]
        tail_k = torch.cat([state["tail_k"], k], dim=2)
        tail_v = torch.cat([state["tail_v"], v], dim=2)
        tail_start = done * self.stride
        total = (end - self.block) // self.stride + 1 if end >= self.block else 0
        new_k, new_v = [state["cmp_k"]], [state["cmp_v"]]
        if total > done:
            starts = torch.arange(done, total, device=k.device) * self.stride - tail_start
            index = starts[:, None] + torch.arange(self.block, device=k.device)[None, :]
            new_k.append(self.phi_k(tail_k[:, :, index]))
            new_v.append(self.phi_v(tail_v[:, :, index]))
        keep_from = total * self.stride - tail_start
        return (
            torch.cat(new_k, dim=2),
            torch.cat(new_v, dim=2),
            tail_k[:, :, keep_from:],
            tail_v[:, :, keep_from:],
        )

    def select_blocks(self, weights: Tensor, q_pos: Tensor, tokens: int) -> Tensor:
        """Eq 9–11 plus the forced blocks: `[batch, kv_heads, queries, selection blocks]`."""
        n_sel = -(-tokens // self.sel_block)
        mapping = nsa_selection_map(
            n_sel, weights.shape[-1], self.block, self.stride, self.sel_block
        ).to(weights)
        p_slc = weights @ mapping.T  # Eq 9
        batch, _, queries, _ = p_slc.shape
        p_group = p_slc.view(batch, self.kv_heads, -1, queries, n_sel).sum(dim=2)  # Eq 10
        starts = torch.arange(n_sel, device=q_pos.device) * self.sel_block
        valid = starts[None, :] <= q_pos[:, None]
        own = torch.div(q_pos, self.sel_block, rounding_mode="floor")[:, None]
        index = torch.arange(n_sel, device=q_pos.device)[None, :]
        forced = (index < self.initial_blocks) | (
            (index <= own) & (index > own - self.local_blocks)
        )
        scores = p_group.masked_fill(forced & valid, float("inf"))
        scores = scores.masked_fill(~valid, float("-inf"))
        return _topk_mask(scores, self.selected)  # Eq 11

    def branch_outputs(self, x: Tensor, state: dict | None = None) -> tuple[dict, dict]:
        """Each branch's attention output `[batch, heads, tokens, value_dim]`, and the new state."""
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        pos = state["pos"]
        end = pos + tokens
        if end > self.context:
            raise ValueError(f"{end} tokens exceed the context of {self.context}")
        groups = self.heads // self.kv_heads
        q = ops.split_heads(self.w_q(x), self.heads)
        q_pos = _positions(pos, tokens, x.device)
        outs, new = {}, {"pos": end}

        # cmp: compress completed blocks, attend to the visible ones (Eq 7).
        cmp_k, cmp_v, tail_k, tail_v = self._compress(state, *self._kv(x, "cmp"), end)
        new.update(cmp_k=cmp_k, cmp_v=cmp_v, tail_k=tail_k, tail_v=tail_v)
        seen = nsa_compressed_visible(q_pos, cmp_k.shape[2], self.block, self.stride)
        outs["cmp"], weights = ops.attend(
            q, ops.repeat_kv(cmp_k, groups), ops.repeat_kv(cmp_v, groups), allowed=seen
        )

        # slc: score blocks from the compression weights, read the chosen blocks' tokens.
        slc_k, slc_v = self._kv(x, "slc")
        slc_k = torch.cat([state["slc_k"], slc_k], dim=2)
        slc_v = torch.cat([state["slc_v"], slc_v], dim=2)
        new.update(slc_k=slc_k, slc_v=slc_v)
        chosen = self.select_blocks(weights, q_pos, end)
        k_pos = _positions(0, end, x.device)
        token_block = torch.div(k_pos, self.sel_block, rounding_mode="floor")
        readable = chosen[..., token_block] & (k_pos[None, :] <= q_pos[:, None])
        outs["slc"], _ = ops.attend(
            q,
            ops.repeat_kv(slc_k, groups),
            ops.repeat_kv(slc_v, groups),
            allowed=readable.repeat_interleave(groups, dim=1),
        )

        # win: the last `window` tokens.
        win_k, win_v = self._kv(x, "win")
        win_k = torch.cat([state["win_k"], win_k], dim=2)
        win_v = torch.cat([state["win_v"], win_v], dim=2)
        w_pos = _positions(end - win_k.shape[2], win_k.shape[2], x.device)
        outs["win"], _ = ops.attend(
            q,
            ops.repeat_kv(win_k, groups),
            ops.repeat_kv(win_v, groups),
            allowed=window_mask(q_pos, w_pos, self.window),
        )
        new.update(win_k=win_k[:, :, -self.window :], win_v=win_v[:, :, -self.window :])
        return outs, new

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Eq 5: the gated sum of the three branches, then the output projection."""
        outs, new = self.branch_outputs(x, state)
        gates = torch.sigmoid(self.gate(x))  # [b, tokens, 3]
        mixed = sum(gates[:, None, :, c, None] * outs[name] for c, name in enumerate(_BRANCHES))
        return self.w_o(ops.merge_heads(mixed)), new


def _nsa(**kw) -> NativeSparseAttention:
    return NativeSparseAttention(**kw)


# --- DeepSeek-V4 CSA ------------------------------------------------------------------------------

_V4_URL = "https://arxiv.org/html/2606.19348v1"
_V4_TITLE = "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
_V4_WHERE = "§4.2.1 Model Setups, DeepSeek-V4-Pro, arXiv:2606.19348v1"
_V4_CSA = (
    "For CSA, we set the compression rate m to 4, the number of indexer query heads n h I to 64, "
    "the indexer head dimension c I to 128"
)
_V4_GROUPS = (
    "The number of output projection groups g is set to 16, and the dimension of each "
    "intermediate attention output d g is set to 1024."
)
for _id, _value, _quote, _unit in (
    (
        "deepseek_csa.d_model",
        7168,
        "We set the number of Transformer layers to 61 and the hidden dimension d to 7168.",
        "dims",
    ),
    ("deepseek_csa.m", 4, _V4_CSA, "tokens per entry"),
    ("deepseek_csa.indexer_heads", 64, _V4_CSA, "heads"),
    ("deepseek_csa.indexer_dim", 128, _V4_CSA, "dims"),
    ("deepseek_csa.q_rank", 1536, "and the query compression dimension d c to 1536.", "dims"),
    ("deepseek_csa.groups", 16, _V4_GROUPS, "groups"),
    ("deepseek_csa.group_dim", 1024, _V4_GROUPS, "dims"),
):
    add(LabSource(_id, _value, _quote, _V4_WHERE, _V4_URL, _V4_TITLE, _unit))


class _OverlapCompressor(nn.Module):
    """Eq 9–12: two KV streams, softmax-weighted over 2m tokens, one entry per m tokens.

    Entry i mixes stream a over tokens `[m·i, m·(i+1) - 1]` with stream b over
    `[m·(i-1), m·i - 1]`; the softmax runs per channel over all 2m elements, with learnable
    positional biases `B_a, B_b`. For i = 0 the b half is padded with `-inf` weights and zero
    entries, as the paper states. Entry i therefore exists once token `m·(i+1) - 1` has been seen.
    """

    def __init__(self, d_model: int, dim: int, m: int) -> None:
        super().__init__()
        self.m, self.dim = m, dim
        self.w_kv_a = nn.Linear(d_model, dim, bias=False)
        self.w_kv_b = nn.Linear(d_model, dim, bias=False)
        self.w_z_a = nn.Linear(d_model, dim, bias=False)
        self.w_z_b = nn.Linear(d_model, dim, bias=False)
        self.b_a = nn.Parameter(torch.zeros(m, dim))
        self.b_b = nn.Parameter(torch.zeros(m, dim))

    def streams(self, x: Tensor) -> Tensor:
        """`[batch, tokens, 4, dim]`: C_a, Z_a, C_b, Z_b for each token."""
        return torch.stack([self.w_kv_a(x), self.w_z_a(x), self.w_kv_b(x), self.w_z_b(x)], dim=2)

    def entries(self, tail: Tensor, tail_start: int, first: int, last: int) -> Tensor:
        """Entries `first .. last-1`, from per-token streams starting at token `tail_start`."""
        m = self.m
        out = []
        for i in range(first, last):
            a = tail[:, m * i - tail_start : m * (i + 1) - tail_start]
            if i == 0:
                c_b = torch.zeros_like(a[:, :, 2])
                z_b = torch.full_like(a[:, :, 3], float("-inf"))
            else:
                b = tail[:, m * (i - 1) - tail_start : m * i - tail_start]
                c_b, z_b = b[:, :, 2], b[:, :, 3]
            logits = torch.cat([a[:, :, 1] + self.b_a, z_b + self.b_b], dim=1)  # Eq 11
            weights = torch.softmax(logits, dim=1)  # over the 2m rows, per channel
            values = torch.cat([a[:, :, 0], c_b], dim=1)
            out.append((weights * values).sum(dim=1))  # Eq 12
        if not out:
            return tail.new_zeros(tail.shape[0], 0, self.dim)
        return torch.stack(out, dim=1)

    def update(self, x: Tensor, entries: Tensor, tail: Tensor, end: int) -> tuple[Tensor, Tensor]:
        """Append every entry completed by `end` tokens; keep the tail the next entry needs."""
        done = entries.shape[1]
        tail = torch.cat([tail, self.streams(x)], dim=1)
        tail_start = max(0, self.m * (done - 1))
        total = end // self.m
        entries = torch.cat([entries, self.entries(tail, tail_start, done, total)], dim=1)
        keep_from = max(0, self.m * (total - 1)) - tail_start
        return entries, tail[:, keep_from:]


def csa_visible(q_pos: Tensor, entries: int, m: int) -> Tensor:
    """`[queries, entries]`: entry s is visible to query t when `s < floor(t / m)`.

    Eq 16 states the condition with t as the query's position. Read 0-indexed, it keeps a query
    away from the entry of its own block even when the query is that block's last token, which is
    what §2.3.3 says ("a query cannot access information from other tokens within its own
    compressed block"), so this lab reads it that way.
    """
    s = torch.arange(entries, device=q_pos.device)[None, :]
    return s < torch.div(q_pos, m, rounding_mode="floor")[:, None]


class CompressedSparseAttention(Mixer):
    """DeepSeek-V4's CSA: a lightning indexer picks compressed entries; MQA reads them.

    Followed (arXiv:2606.19348v1): compression Eq 9–12 (see `_OverlapCompressor`); the indexer's
    keys use the same compression with their own weights; low-rank queries shared by the indexer
    and the core attention (Eq 13, 14, 18); index scores `I_{t,s} = Σ_h w_{t,h} ReLU(q_{t,h} ·
    K_s)` (Eq 15–16); the top-k visible entries (Eq 17); MQA in which each entry is both key and
    value (Eq 19); grouped output projection (§2.3.1); per-head RMSNorm on queries and entries,
    partial RoPE on the last `rope_dim` dimensions with the output rotated back by the query's
    position, a sliding-window branch of `window` uncompressed tokens, and learnable sink logits
    added to each head's softmax denominator (Eq 27) (§2.3.3).

    **Our choices where the text does not say:**

    - The RoPE position of a compressed entry: this lab uses the position of the entry's last
      a-stream token, `m·(s+1) - 1`.
    - The "position −i" used to rotate the output is read as the query's own position.
    - The window's uncompressed entries come from their own projection, and pass through the same
      RMSNorm as the compressed entries. The window includes the query.
    - The core attention scale is `1/sqrt(head_dim)`; sink logits start at zero.
    - The indexer uses no position encoding.

    The sections read state how CSA and HCA are interleaved across layers but not HCA's place in
    a single layer; HCA is not reproduced here. The indexer's top-k is not differentiable, so its
    weights get no gradient from the language-model loss; the paper's indexer training is not
    reproduced.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        q_rank: int,
        m: int,
        indexer_heads: int,
        indexer_dim: int,
        topk: int,
        window: int,
        groups: int,
        group_dim: int,
        rope_dim: int,
        rope_base: float,
        attention_sink: bool,
        context: int,
    ) -> None:
        """Compressors, low-rank queries, indexer, window projection and grouped output."""
        super().__init__()
        if heads % groups:
            raise ValueError("heads must be a multiple of groups")
        if rope_dim % 2 or rope_dim > head_dim:
            raise ValueError("rope_dim must be even and at most head_dim")
        self.heads, self.head_dim, self.m = heads, head_dim, m
        self.indexer_heads, self.indexer_dim, self.topk = indexer_heads, indexer_dim, topk
        self.window, self.groups, self.rope_dim, self.rope_base = (
            window,
            groups,
            rope_dim,
            rope_base,
        )
        self.context = context
        self.kv = _OverlapCompressor(d_model, head_dim, m)
        self.index_kv = _OverlapCompressor(d_model, indexer_dim, m)
        self.w_dq = nn.Linear(d_model, q_rank, bias=False)  # Eq 13
        self.w_iuq = nn.Linear(q_rank, indexer_heads * indexer_dim, bias=False)  # Eq 14
        self.w_w = nn.Linear(d_model, indexer_heads, bias=False)  # Eq 15
        self.w_uq = nn.Linear(q_rank, heads * head_dim, bias=False)  # Eq 18
        self.w_win = nn.Linear(d_model, head_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.kv_norm = nn.RMSNorm(head_dim)
        self.sink = nn.Parameter(torch.zeros(heads)) if attention_sink else None
        self.w_group = nn.ModuleList(
            nn.Linear(head_dim * heads // groups, group_dim, bias=False) for _ in range(groups)
        )
        self.w_o = nn.Linear(groups * group_dim, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """No entries, no tails, no window."""
        dtype = dtype or self.w_dq.weight.dtype

        def empty(*shape: int) -> Tensor:
            return torch.zeros(batch, *shape, device=device, dtype=dtype)

        return {
            "entries": empty(0, self.head_dim),
            "tail": empty(0, 4, self.head_dim),
            "index_keys": empty(0, self.indexer_dim),
            "index_tail": empty(0, 4, self.indexer_dim),
            "win": empty(0, self.head_dim),
            "pos": 0,
        }

    def select(self, x: Tensor, index_keys: Tensor, q_pos: Tensor) -> Tensor:
        """Eq 13–17: `[batch, queries, entries]`, True for the chosen visible entries."""
        batch, tokens, _ = x.shape
        q_idx = self.w_iuq(self.w_dq(x)).view(batch, tokens, self.indexer_heads, -1)
        w_idx = self.w_w(x)
        dots = torch.einsum("bthc,bsc->bths", q_idx, index_keys)
        scores = (w_idx[..., None] * torch.relu(dots)).sum(dim=2)  # Eq 16
        visible = csa_visible(q_pos, index_keys.shape[1], self.m)
        return _topk_mask(scores.masked_fill(~visible, float("-inf")), self.topk)  # Eq 17

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Select entries, attend over them and the window, project by groups."""
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        pos = state["pos"]
        end = pos + tokens
        if end > self.context:
            raise ValueError(f"{end} tokens exceed the context of {self.context}")
        q_pos = _positions(pos, tokens, x.device)
        entries, tail = self.kv.update(x, state["entries"], state["tail"], end)
        index_keys, index_tail = self.index_kv.update(
            x, state["index_keys"], state["index_tail"], end
        )
        win = torch.cat([state["win"], self.w_win(x)], dim=1)
        chosen = self.select(x, index_keys, q_pos)

        # Core attention: one KV head, queries per head (Eq 18–19), keys = values.
        q = self.q_norm(ops.split_heads(self.w_uq(self.w_dq(x)), self.heads))
        q = _rope_tail(q, q_pos, self.rope_dim, self.rope_base)
        entry_pos = (torch.arange(entries.shape[1], device=x.device) + 1) * self.m - 1
        win_pos = _positions(end - win.shape[1], win.shape[1], x.device)
        kv = torch.cat(
            [
                _rope_tail(self.kv_norm(entries), entry_pos, self.rope_dim, self.rope_base),
                _rope_tail(self.kv_norm(win), win_pos, self.rope_dim, self.rope_base),
            ],
            dim=1,
        )[:, None]
        allowed = torch.cat(
            [chosen, window_mask(q_pos, win_pos, self.window).expand(batch, -1, -1)], -1
        )
        allowed = allowed[:, None]
        bias = None
        if self.sink is not None:  # Eq 27: a key scoring z'_h whose value is zero
            kv = torch.cat([kv, kv.new_zeros(batch, 1, 1, self.head_dim)], dim=2)
            allowed = torch.cat([allowed, allowed.new_ones(batch, 1, tokens, 1)], dim=-1)
            bias = torch.zeros(1, self.heads, 1, kv.shape[2], dtype=x.dtype, device=x.device)
            bias[..., -1] = self.sink.to(x.dtype)[None, :, None]
        out, _ = ops.attend(
            q, kv, kv, allowed=allowed, bias=bias, scale=1.0 / math.sqrt(self.head_dim)
        )
        out = _rope_tail(out, -q_pos, self.rope_dim, self.rope_base)

        # Grouped output projection.
        per_head = out.transpose(1, 2)  # [b, tokens, heads, c]
        chunks = per_head.chunk(self.groups, dim=2)
        mids = [proj(chunk.flatten(-2)) for proj, chunk in zip(self.w_group, chunks, strict=True)]
        y = self.w_o(torch.cat(mids, dim=-1))
        return y, {
            "entries": entries,
            "tail": tail,
            "index_keys": index_keys,
            "index_tail": index_tail,
            "win": win[:, -self.window :],
            "pos": end,
        }


def _csa(**kw) -> CompressedSparseAttention:
    return CompressedSparseAttention(**kw)


# --- MiniMax Sparse Attention ---------------------------------------------------------------------

_MSA_URL = "https://arxiv.org/html/2606.13392v2"
_MSA_TITLE = "MiniMax Sparse Attention"
add(
    LabSource(
        "msa.d_model",
        3072,
        "The model uses a 200K-token vocabulary and hidden size d model = 3072 .",
        "§5.1 Setup, Model Structure, arXiv:2606.13392v2",
        _MSA_URL,
        _MSA_TITLE,
        "dims",
    )
)


def msa_block_scores(scores: Tensor, q_pos: Tensor, k_pos: Tensor, block: int) -> Tensor:
    """Eq 6: a block's score is the max token score over its causally visible tokens.

    `scores` is `[..., queries, keys]`; the result is `[..., queries, blocks]`, with `-inf` for a
    block that has no visible token.
    """
    visible = k_pos[None, :] <= q_pos[:, None]
    masked = scores.masked_fill(~visible, float("-inf"))
    blocks = int(k_pos.max().item()) // block + 1 if len(k_pos) else 0
    token_block = torch.div(k_pos, block, rounding_mode="floor")
    out = masked.new_full((*masked.shape[:-1], blocks), float("-inf"))
    index = token_block.expand(*masked.shape[:-1], -1)
    return out.scatter_reduce(-1, index, masked, reduce="amax")


class MiniMaxSparseAttention(Mixer):
    """MSA: per KV group, a light index branch picks k token blocks; exact attention reads them.

    Followed (arXiv:2606.13392v2 §3.1): one index query head per GQA group and one index key head
    shared by all groups (Eq 5), computed from a stop-gradient copy of the input (Eq 11); token
    scores `S = Q_idx K_idx^T / sqrt(d_idx)`; block scores are the max over the block's causally
    visible tokens, `-inf` for a block with none (Eq 6); the top `selected` blocks per group
    (Eq 7), shared by the group's query heads; and exact softmax attention over the visible tokens
    of those blocks, with the group's KV head (Eq 8). `index_alignment_loss` is the KL objective
    that trains the index branch (Eq 9–10).

    **The local block, read one way.** §3.1 says the block containing the query is always
    included, and this lab does that, counting it towards `selected`. Appendix C.2 says the final
    recipe "only forces the special incomplete self block", i.e. not when the query ends its
    block. The two readings differ only for the last token of a block.

    **Our choices where the text does not say:** `d_idx`; that RoPE rotates the *last* `rope_dim`
    dimensions of each head (the source gives the width, not which dimensions); and that the index
    branch uses no position encoding. Block boundaries follow Eq 4, 0-indexed: block b holds tokens
    `[b·B_k, (b+1)·B_k)`.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        block: int,
        selected: int,
        index_dim: int,
        rope_dim: int,
        rope_base: float,
        context: int,
    ) -> None:
        """GQA projections plus the two index projections of Eq 5."""
        super().__init__()
        if heads % kv_heads:
            raise ValueError("heads must be a multiple of kv_heads")
        if rope_dim % 2 or rope_dim > head_dim:
            raise ValueError("rope_dim must be even and at most head_dim")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.block, self.selected, self.index_dim = block, selected, index_dim
        self.rope_dim, self.rope_base, self.context = rope_dim, rope_base, context
        self.w_q = nn.Linear(d_model, heads * head_dim, bias=False)
        self.w_k = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.w_v = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.w_q_idx = nn.Linear(d_model, kv_heads * index_dim, bias=False)
        self.w_k_idx = nn.Linear(d_model, index_dim, bias=False)
        self.w_o = nn.Linear(heads * head_dim, d_model, bias=False)

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """Empty main and index caches."""
        dtype = dtype or self.w_q.weight.dtype
        empty = torch.zeros(batch, self.kv_heads, 0, self.head_dim, device=device, dtype=dtype)
        return {
            "k": empty,
            "v": empty.clone(),
            "k_idx": torch.zeros(batch, 1, 0, self.index_dim, device=device, dtype=dtype),
            "pos": 0,
        }

    def _step(self, x: Tensor, state: dict | None) -> tuple[dict, dict]:
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        pos = state["pos"]
        end = pos + tokens
        if end > self.context:
            raise ValueError(f"{end} tokens exceed the context of {self.context}")
        q_pos = _positions(pos, tokens, x.device)
        k_pos = _positions(0, end, x.device)
        q = _rope_tail(
            ops.split_heads(self.w_q(x), self.heads), q_pos, self.rope_dim, self.rope_base
        )
        k = _rope_tail(
            ops.split_heads(self.w_k(x), self.kv_heads), q_pos, self.rope_dim, self.rope_base
        )
        k = torch.cat([state["k"], k], dim=2)
        v = torch.cat([state["v"], ops.split_heads(self.w_v(x), self.kv_heads)], dim=2)
        detached = x.detach()  # Eq 11
        q_idx = ops.split_heads(self.w_q_idx(detached), self.kv_heads)
        k_idx = torch.cat([state["k_idx"], ops.split_heads(self.w_k_idx(detached), 1)], dim=2)
        token_scores = (q_idx @ k_idx.transpose(-2, -1)) / math.sqrt(self.index_dim)
        block_scores = msa_block_scores(token_scores, q_pos, k_pos, self.block)  # Eq 6
        own = torch.div(q_pos, self.block, rounding_mode="floor")
        is_own = torch.arange(block_scores.shape[-1], device=x.device)[None, :] == own[:, None]
        chosen = _topk_mask(block_scores.masked_fill(is_own, float("inf")), self.selected)
        token_block = torch.div(k_pos, self.block, rounding_mode="floor")
        readable = chosen[..., token_block] & (k_pos[None, :] <= q_pos[:, None])
        work = {"q": q, "k": k, "v": v, "token_scores": token_scores, "readable": readable}
        return work, {"k": k, "v": v, "k_idx": k_idx, "pos": end}

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Eq 5–8: select blocks per group, then exact attention over their visible tokens."""
        work, new = self._step(x, state)
        groups = self.heads // self.kv_heads
        out, _ = ops.attend(
            work["q"],
            ops.repeat_kv(work["k"], groups),
            ops.repeat_kv(work["v"], groups),
            allowed=work["readable"].repeat_interleave(groups, dim=1),
        )
        return self.w_o(ops.merge_heads(out)), new

    def index_alignment_loss(self, x: Tensor) -> Tensor:
        """Eq 9–10 over a whole sequence: KL(teacher ‖ index distribution) on the selected tokens.

        The teacher averages the group's per-head main-branch distributions and is detached, and
        the index branch sees a detached input, so this loss reaches only the index projections.
        """
        work, _ = self._step(x, None)
        groups = self.heads // self.kv_heads
        readable = work["readable"]  # [b, kv, N, N]
        k = ops.repeat_kv(work["k"], groups)
        main = (work["q"] @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        main = main.detach().masked_fill(~readable.repeat_interleave(groups, dim=1), -math.inf)
        batch, _, n, _ = main.shape
        teacher = torch.softmax(main, dim=-1).view(batch, self.kv_heads, groups, n, n).mean(2)
        student = torch.log_softmax(
            work["token_scores"].masked_fill(~readable, -math.inf), dim=-1
        ).masked_fill(~readable, 0.0)  # an unread token contributes nothing, and no NaN gradient
        terms = teacher * (torch.log(teacher.clamp_min(1e-300)) - student)
        kl = torch.where(readable, terms, torch.zeros_like(terms)).sum(dim=-1)
        return kl.mean()


def _msa(**kw) -> MiniMaxSparseAttention:
    return MiniMaxSparseAttention(**kw)


# --- registrations --------------------------------------------------------------------------------

_BASE_NOTE = "the conventional RoPE base; not read from this variant's source here"


register(
    MixerSpec(
        name="sliding_window",
        family="sparse",
        summary="Each query reads only the last `window` tokens, so the cache stops growing.",
        parent="standard_attention",
        covers="sliding_window",
        source="catalogue:sliding_window",
        checked_against="arXiv:2004.05150v2 §3.1 (sliding window), in a causal reading",
        factory=lambda **kw: SlidingWindowAttention(**kw),
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "number of attention heads"),
            _lab(
                "window",
                4,
                "keys each query reads, itself included",
                "lab scale, a third of the generic test's twelve tokens so the band is visible",
            ),
        ),
        paper=(
            _lab(
                "d_model",
                768,
                "width of the token vectors",
                "not read from Longformer here; a BERT-base width chosen only to size the layer",
            ),
            _lab(
                "heads",
                12,
                "number of attention heads",
                "not read from Longformer here; a "
                "BERT-base head count chosen only to size the layer",
            ),
            _lab(
                "window",
                512,
                "keys each query reads",
                "Longformer states 'We use sliding "
                "window attention with window size of 512' for its masked-LM pretraining (§5), "
                "where the window is bidirectional, so it is not a causal key count",
            ),
        ),
        state_growth="bounded",
    )
)

register(
    MixerSpec(
        name="sparse_attention",
        family="sparse",
        summary=(
            "Each head reads a factorized strided or fixed pattern of earlier positions, merged "
            "into one set, instead of every earlier position."
        ),
        parent="standard_attention",
        covers="sparse_attention",
        source="catalogue:sparse_attention",
        checked_against="arXiv:1904.10509v1 §4.2–4.3 (strided and fixed sets), §5.1 Eq 7–8",
        factory=_sparse_attention,
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "number of attention heads"),
            _lab(
                "pattern",
                "fixed",
                "which factorization",
                "lab default, the pattern the source uses for text",
            ),
            _lab(
                "stride",
                4,
                "the stride l",
                "lab scale, close to the square root of the generic test's length",
            ),
            _lab(
                "c",
                1,
                "summary positions per block in the fixed pattern",
                "lab scale, one summary cell per block of four",
            ),
            _lab(
                "combine",
                "merged",
                "how the two sets meet in a head",
                "lab default, the combination the source reports for text",
            ),
            _lab(
                "context",
                512,
                "longest sequence the layer accepts",
                "lab scale, above the generic tests' longest sequence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "sparse_attention.d_model", "width of the model"),
            hparams.from_catalogue("heads", "sparse_attention", "heads", "number of heads"),
            hparams.from_lab_source("pattern", "sparse_attention.pattern", "factorization"),
            hparams.from_catalogue("stride", "sparse_attention", "stride", "the stride l"),
            hparams.from_lab_source("c", "sparse_attention.c", "summary positions per block"),
            hparams.from_lab_source("combine", "sparse_attention.combine", "head combination"),
            hparams.from_catalogue(
                "context", "sparse_attention", "context", "training context length"
            ),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="topk_attention",
        family="sparse",
        summary=(
            "Each query keeps only its `topk` highest-scoring earlier keys and gives every other "
            "key zero weight."
        ),
        parent="standard_attention",
        covers="topk_attention",
        source="catalogue:topk_attention",
        checked_against="arXiv:1912.11637v1 §2 Eq 1–4 and Appendix A.3 (threshold is a constant)",
        factory=lambda **kw: TopKAttention(**kw),
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "number of attention heads"),
            _lab(
                "topk",
                3,
                "keys kept per query row",
                "lab scale, a quarter of the generic test's twelve tokens so selection is active",
            ),
        ),
        paper=(
            _lab(
                "d_model",
                512,
                "width of the token vectors",
                "not read from this source here; "
                "a Transformer-base width chosen only to size the layer",
            ),
            _lab(
                "heads",
                8,
                "number of attention heads",
                "not read from this source here; a "
                "Transformer-base head count chosen only to size the layer",
            ),
            hparams.from_catalogue("topk", "topk_attention", "topk", "keys kept per query row"),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="reformer",
        family="sparse",
        summary=(
            "Queries and keys share one projection and each query reads only earlier keys that "
            "hash into its bucket in some round, never itself unless nothing else is visible."
        ),
        parent="standard_attention",
        covers="reformer",
        source="catalogue:reformer",
        checked_against=(
            "arXiv:2001.04451v2 §2 (hash, shared QK, Eq 4, Eq 6, self mask), Appendix A Eq 12–16; "
            "Eq 5 chunking not reproduced"
        ),
        factory=_reformer,
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 2, "number of attention heads"),
            _lab(
                "n_buckets",
                4,
                "hash buckets per round",
                "lab scale, a few buckets so some pairs are excluded",
            ),
            _lab(
                "n_rounds",
                2,
                "hash rounds whose buckets are united",
                "lab scale, more than one so the union matters",
            ),
            _lab(
                "self_penalty",
                1e5,
                "score subtracted from attention to oneself",
                "Appendix A "
                "Eq 16 prints 10^5 inside an equation, which the quote checker cannot read",
            ),
            _lab(
                "seed",
                0,
                "seed for the fixed random rotations",
                "any fixed seed; the source only requires the rotations be random and fixed",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "reformer.d_model", "width of the model"),
            hparams.from_lab_source("heads", "reformer.heads", "number of heads"),
            _lab(
                "n_buckets",
                32,
                "hash buckets per round",
                "not stated in the source; chosen as a power of two for sizing only",
            ),
            hparams.from_catalogue("n_rounds", "reformer", "hashes", "hash rounds"),
            _lab(
                "self_penalty",
                1e5,
                "score subtracted from attention to oneself",
                "Appendix A "
                "Eq 16 prints 10^5 inside an equation, which the quote checker cannot read",
            ),
            _lab(
                "seed",
                0,
                "seed for the fixed random rotations",
                "any fixed seed; the source only requires the rotations be random and fixed",
            ),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="attention_sinks",
        family="sparse",
        summary=(
            "Keeps the first `sinks` tokens alongside the window and assigns RoPE positions "
            "within that cache rather than in the original text."
        ),
        parent="sliding_window",
        covers="attention_sinks",
        source="catalogue:attention_sinks",
        checked_against="arXiv:2309.17453v4 §3.2 (rolling cache, positions within the cache)",
        factory=lambda **kw: SinkAttention(**kw),
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "number of attention heads"),
            hparams.from_catalogue("sinks", "attention_sinks", "sinks", "initial tokens kept"),
            _lab(
                "window",
                4,
                "recent tokens kept, the query included",
                "lab scale, so sinks plus window is shorter than the generic test",
            ),
            _lab("rope_base", 10000.0, "RoPE frequency base", _BASE_NOTE),
        ),
        paper=(
            _lab(
                "d_model",
                5120,
                "width of the token vectors",
                "not read from StreamingLLM; a size chosen only to shape the layer",
            ),
            _lab(
                "heads",
                40,
                "number of attention heads",
                "not read from StreamingLLM; a head count chosen only to shape the layer",
            ),
            hparams.from_catalogue("sinks", "attention_sinks", "sinks", "initial tokens kept"),
            hparams.from_lab_source(
                "window", "attention_sinks.window", "recent tokens kept in the 4+1020 cache"
            ),
            _lab("rope_base", 10000.0, "RoPE frequency base", _BASE_NOTE),
        ),
        state_growth="bounded",
    )
)

register(
    MixerSpec(
        name="nsa",
        family="sparse",
        summary=(
            "Replaces one attention with three gated branches over GQA keys: compressed blocks, "
            "the top-scoring token blocks, and a sliding window."
        ),
        parent="gqa",
        covers="nsa",
        source="catalogue:nsa",
        checked_against="arXiv:2502.11089v2 §3.2 Eq 5, §3.3.1 Eq 7, §3.3.2 Eq 8–12, §3.3.3",
        factory=_nsa,
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "query heads"),
            _lab("kv_heads", 2, "key/value heads (GQA groups)"),
            _lab("head_dim", 8, "query and key width per head"),
            _lab(
                "value_dim",
                6,
                "value width per head",
                "lab scale, different from head_dim as in the source",
            ),
            _lab("block", 2, "compression block length l"),
            _lab("stride", 1, "stride d between compression blocks"),
            _lab("sel_block", 2, "selection block length l'"),
            _lab(
                "selected",
                4,
                "selection blocks per query, forced ones included",
                "lab scale, leaves two free choices among six blocks of the generic test",
            ),
            _lab("initial_blocks", 1, "initial blocks always selected"),
            _lab("local_blocks", 1, "most recent blocks always selected"),
            _lab("window", 3, "sliding-window tokens"),
            _lab(
                "context",
                512,
                "longest sequence the layer accepts",
                "lab scale, above the generic tests' longest sequence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "nsa.d_model", "hidden dimension"),
            hparams.from_lab_source("heads", "nsa.heads", "attention heads"),
            hparams.from_lab_source("kv_heads", "nsa.kv_heads", "GQA groups"),
            hparams.from_lab_source("head_dim", "nsa.head_dim", "query and key width per head"),
            hparams.from_lab_source("value_dim", "nsa.value_dim", "value width per head"),
            hparams.from_catalogue("block", "nsa", "compressBlock", "compression block length"),
            hparams.from_catalogue("stride", "nsa", "compressStride", "compression stride"),
            hparams.from_catalogue("sel_block", "nsa", "blockSize", "selection block length"),
            hparams.from_catalogue("selected", "nsa", "selected", "selected blocks per query"),
            hparams.from_lab_source("initial_blocks", "nsa.initial_blocks", "forced initial"),
            hparams.from_lab_source("local_blocks", "nsa.local_blocks", "forced local blocks"),
            hparams.from_catalogue("window", "nsa", "window", "sliding-window tokens"),
            hparams.from_catalogue("context", "nsa", "context", "longest trained context"),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="deepseek_csa",
        family="sparse",
        summary=(
            "Compresses every m tokens into one shared key-value entry, lets a lightning indexer "
            "pick the top entries, and runs MQA over them plus a window."
        ),
        parent="nsa",
        covers="deepseek_csa",
        source="catalogue:deepseek_csa",
        checked_against=(
            "arXiv:2606.19348v1 §2.3.1 Eq 9–19 and grouped output projection; §2.3.3 (RMSNorm, "
            "partial RoPE, window branch, sink logits Eq 27)"
        ),
        factory=_csa,
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "query heads"),
            _lab("head_dim", 8, "width of each head and of each entry"),
            _lab("q_rank", 16, "query latent width"),
            _lab(
                "m",
                2,
                "tokens per compressed entry",
                "lab scale, so the generic test builds several entries",
            ),
            _lab("indexer_heads", 2, "indexer query heads"),
            _lab("indexer_dim", 4, "indexer head width"),
            _lab(
                "topk",
                2,
                "compressed entries per query",
                "lab scale, fewer than the entries a late query can see so selection is active",
            ),
            _lab("window", 3, "uncompressed recent tokens"),
            _lab("groups", 2, "output projection groups"),
            _lab("group_dim", 8, "intermediate width per group"),
            _lab("rope_dim", 4, "rotated dimensions at the end of each vector"),
            _lab("rope_base", 10000.0, "RoPE frequency base", _BASE_NOTE),
            _lab(
                "attention_sink",
                True,
                "learnable sink logits in each head",
                "§2.3.3 describes them; a true or false value cannot be matched against a quote",
            ),
            _lab(
                "context",
                512,
                "longest sequence the layer accepts",
                "lab scale, above the generic tests' longest sequence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "deepseek_csa.d_model", "hidden dimension"),
            hparams.from_catalogue("heads", "deepseek_csa", "heads", "query heads"),
            hparams.from_catalogue("head_dim", "deepseek_csa", "headDim", "head dimension c"),
            hparams.from_lab_source("q_rank", "deepseek_csa.q_rank", "query latent width"),
            hparams.from_lab_source("m", "deepseek_csa.m", "tokens per compressed entry"),
            hparams.from_lab_source(
                "indexer_heads", "deepseek_csa.indexer_heads", "indexer query heads"
            ),
            hparams.from_lab_source("indexer_dim", "deepseek_csa.indexer_dim", "indexer width"),
            hparams.from_catalogue("topk", "deepseek_csa", "topk", "entries per query"),
            hparams.from_catalogue("window", "deepseek_csa", "window", "window tokens"),
            hparams.from_lab_source("groups", "deepseek_csa.groups", "output groups"),
            hparams.from_lab_source("group_dim", "deepseek_csa.group_dim", "width per group"),
            hparams.from_catalogue("rope_dim", "deepseek_csa", "dims", "rotated dimensions"),
            _lab("rope_base", 10000.0, "RoPE frequency base", _BASE_NOTE),
            _lab(
                "attention_sink",
                True,
                "learnable sink logits in each head",
                "§2.3.3 describes them; a true or false value cannot be matched against a quote",
            ),
            hparams.from_catalogue("context", "deepseek_csa", "context", "supported context"),
        ),
        state_growth="grows",
    )
)

register(
    MixerSpec(
        name="msa",
        family="sparse",
        summary=(
            "Adds a light index branch to GQA that picks, per KV group, the top blocks of tokens "
            "by their best token score, and reads only those blocks exactly."
        ),
        parent="gqa",
        covers="msa",
        source="catalogue:msa",
        checked_against="arXiv:2606.13392v2 §2.3 Eq 4, §3.1 Eq 5–8, §3.2 Eq 9–11",
        factory=_msa,
        lab=(
            _lab("d_model", 32, "width of the token vectors"),
            _lab("heads", 4, "query heads"),
            _lab("kv_heads", 2, "key/value heads (GQA groups)"),
            _lab("head_dim", 8, "width per head"),
            _lab(
                "block", 2, "tokens per block B_k", "lab scale, so the generic test has six blocks"
            ),
            _lab(
                "selected",
                2,
                "blocks per query and group, the own block included",
                "lab scale, fewer than the blocks a late query can see",
            ),
            _lab(
                "index_dim",
                4,
                "index head width d_idx",
                "not stated in the source; lab scale width for the index heads",
            ),
            _lab("rope_dim", 4, "rotated dimensions at the end of each head"),
            _lab("rope_base", 10000.0, "RoPE frequency base", _BASE_NOTE),
            _lab(
                "context",
                512,
                "longest sequence the layer accepts",
                "lab scale, above the generic tests' longest sequence",
            ),
        ),
        paper=(
            hparams.from_lab_source("d_model", "msa.d_model", "hidden size"),
            hparams.from_catalogue("heads", "msa", "heads", "query heads"),
            hparams.from_catalogue("kv_heads", "msa", "kvHeads", "key/value heads"),
            hparams.from_catalogue("head_dim", "msa", "headDim", "width per head"),
            hparams.from_catalogue("block", "msa", "blockSize", "tokens per block"),
            hparams.from_catalogue("selected", "msa", "selected", "blocks per query and group"),
            _lab(
                "index_dim",
                128,
                "index head width d_idx",
                "not stated in the source; set to the head width only to size the layer",
            ),
            hparams.from_catalogue("rope_dim", "msa", "dims", "RoPE dimension"),
            _lab("rope_base", 10000.0, "RoPE frequency base", _BASE_NOTE),
            hparams.from_catalogue("context", "msa", "context", "context of the speed claim"),
        ),
        state_growth="grows",
    )
)
