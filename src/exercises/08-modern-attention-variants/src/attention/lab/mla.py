"""The cache family, part two: multi-head latent attention (MLA), from DeepSeek-V2.

**The problem.** A decoder keeps every earlier token's keys and values, and that cache is what
limits how long a context fits in memory. Grouped-query attention shrinks it by letting query heads
share key/value heads, and pays for it in quality. MLA asks a different question: instead of
keeping fewer heads, can a layer keep one *small vector* per token and rebuild every head's key and
value from it?

**What changes in code, relative to the parent (`gqa`).**

- The layer projects each token down to a latent `c_kv` of width `kv_rank` (Eq 9), and derives
  every head's key and value from that latent with two up-projections (Eq 10–11). The cache holds
  `c_kv`, not keys and values.
- Queries go through their own down/up pair (Eq 12–13). This saves activation memory in training
  and does nothing for the cache, which the source says in as many words.
- Rotary position embedding cannot be applied to keys rebuilt from a cached latent without
  rebuilding them every step, so position rides on a separate, **decoupled** part: each head gets
  `rope_dim` extra query dimensions (Eq 14), and all heads share **one** extra rotated key of width
  `rope_dim`, computed from the token itself (Eq 15). That shared key is cached too.
- Each head's query and key are the concatenation of the two parts (Eq 16–17), attention is scaled
  by `1/sqrt(head_dim + rope_dim)` and reads the *un-rotated* values (Eq 18), and an output
  projection mixes the heads (Eq 19).

So the cache per token is `kv_rank + rope_dim` numbers per layer, whatever the number of heads —
`tests/test_attention_lab_mla.py` checks exactly that, from the tensors the layer really keeps.

**The trade-off.** Every decode step spends an up-projection per cached token to rebuild keys and
values (the source notes the up-projections could be absorbed into neighbouring matrices; this lab
computes them explicitly, which is the same function). The latent is a low-rank bottleneck, so the
keys and values of different heads are no longer independent.

What this module does not reproduce: the source's appendix adds normalisations after the latents,
with settings the sections read here do not state, so none are applied. The RoPE base is not taken
from DeepSeek-V2 here; see the `rope_base` parameter.

Followed: arXiv:2405.04434v5, §2.1.2 (Eq 9–13) and §2.1.3 (Eq 14–19); sizes from §3.1.2.
"""

import math

import torch
from torch import Tensor, nn

from attention.lab import hparams, ops
from attention.lab.base import Mixer, MixerSpec
from attention.lab.registry import register
from attention.lab.sources import LabSource, add

_V2_URL = "https://arxiv.org/html/2405.04434v5"
_V2_TITLE = "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
_HEADS_QUOTE = (
    "In MLA, we set the number of attention heads n h to 128 and the per-head dimension d h to 128."
)
_RANKS_QUOTE = (
    "The KV compression dimension d c is set to 512, and the query compression dimension d c ′ is "
    "set to 1536."
)
_WHERE = "§3.1.2 Hyper-Parameters, Model Hyper-Parameters, arXiv:2405.04434v5"

add(LabSource("mla.heads", 128, _HEADS_QUOTE, _WHERE, _V2_URL, _V2_TITLE, "heads"))
add(LabSource("mla.head_dim", 128, _HEADS_QUOTE, _WHERE, _V2_URL, _V2_TITLE, "dimensions"))
add(LabSource("mla.kv_rank", 512, _RANKS_QUOTE, _WHERE, _V2_URL, _V2_TITLE, "dimensions"))
add(LabSource("mla.q_rank", 1536, _RANKS_QUOTE, _WHERE, _V2_URL, _V2_TITLE, "dimensions"))
add(
    LabSource(
        "mla.rope_dim",
        64,
        "For the decoupled queries and key, we set the per-head dimension d h R to 64.",
        _WHERE,
        _V2_URL,
        _V2_TITLE,
        "dimensions",
    )
)
add(
    LabSource(
        "mla.d_model",
        5120,
        "We set the number of Transformer layers to 60 and the hidden dimension to 5120.",
        _WHERE,
        _V2_URL,
        _V2_TITLE,
        "dimensions",
    )
)


class MultiHeadLatentAttention(Mixer):
    """MLA: cache a small latent plus one shared rotated key per token.

    Args:
        d_model: Width of the token vectors (`d` in the source).
        heads: Number of attention heads (`n_h`).
        head_dim: Width of each head's content query, key and value (`d_h`).
        kv_rank: Width of the cached key/value latent (`d_c`).
        q_rank: Width of the query latent (`d_c'`).
        rope_dim: Width of each head's decoupled rotary query, and of the shared rotary key
            (`d_h^R`). Must be even, because RoPE rotates pairs.
        rope_base: The RoPE frequency base.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        kv_rank: int,
        q_rank: int,
        rope_dim: int,
        rope_base: float,
    ) -> None:
        """Build the down/up projections of Eq 9–15 and the output projection of Eq 19."""
        super().__init__()
        if rope_dim % 2:
            raise ValueError(f"rope_dim must be even, not {rope_dim}")
        self.heads, self.head_dim, self.rope_dim = heads, head_dim, rope_dim
        self.kv_rank, self.rope_base = kv_rank, rope_base
        self.w_dkv = nn.Linear(d_model, kv_rank, bias=False)  # Eq 9
        self.w_uk = nn.Linear(kv_rank, heads * head_dim, bias=False)  # Eq 10
        self.w_uv = nn.Linear(kv_rank, heads * head_dim, bias=False)  # Eq 11
        self.w_dq = nn.Linear(d_model, q_rank, bias=False)  # Eq 12
        self.w_uq = nn.Linear(q_rank, heads * head_dim, bias=False)  # Eq 13
        self.w_qr = nn.Linear(q_rank, heads * rope_dim, bias=False)  # Eq 14
        self.w_kr = nn.Linear(d_model, rope_dim, bias=False)  # Eq 15
        self.w_o = nn.Linear(heads * head_dim, d_model, bias=False)  # Eq 19

    def init_state(
        self, batch: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> dict:
        """An empty cache: no latents and no rotated keys yet."""
        dtype = dtype or self.w_dkv.weight.dtype
        return {
            "c_kv": torch.zeros(batch, 0, self.kv_rank, device=device, dtype=dtype),
            "k_rope": torch.zeros(batch, 0, self.rope_dim, device=device, dtype=dtype),
            "pos": 0,
        }

    def forward(self, x: Tensor, state: dict | None = None, memory: Tensor | None = None):
        """Attend over every cached token, rebuilding keys and values from the latents."""
        batch, tokens, _ = x.shape
        if state is None:
            state = self.init_state(batch, x.device, x.dtype)
        pos = state["pos"]
        positions = torch.arange(pos, pos + tokens, device=x.device)
        angles = ops.rope_angles(positions, self.rope_dim, self.rope_base)

        # What gets cached: the latent (Eq 9) and the one rotated key shared by all heads (Eq 15).
        c_kv = torch.cat([state["c_kv"], self.w_dkv(x)], dim=1)
        k_rope = torch.cat([state["k_rope"], ops.apply_rope(self.w_kr(x), angles)], dim=1)
        keys = c_kv.shape[1]

        # Keys and values for every cached token, rebuilt from the latent (Eq 10, 11, 17).
        k_content = ops.split_heads(self.w_uk(c_kv), self.heads)
        v_content = ops.split_heads(self.w_uv(c_kv), self.heads)
        shared = k_rope[:, None].expand(batch, self.heads, keys, self.rope_dim)
        k = torch.cat([k_content, shared], dim=-1)

        # Queries through their own latent (Eq 12, 13), with a rotated part per head (Eq 14, 16).
        c_q = self.w_dq(x)
        q_content = ops.split_heads(self.w_uq(c_q), self.heads)
        q_rope = ops.apply_rope(ops.split_heads(self.w_qr(c_q), self.heads), angles)
        q = torch.cat([q_content, q_rope], dim=-1)

        allowed = ops.causal_mask(tokens, keys, offset=keys - tokens, device=x.device)
        scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)  # Eq 18
        out, _ = ops.attend(q, k, v_content, allowed=allowed, scale=scale)
        y = self.w_o(ops.merge_heads(out))  # Eq 19
        return y, {"c_kv": c_kv, "k_rope": k_rope, "pos": pos + tokens}


def _mla(**kw) -> MultiHeadLatentAttention:
    return MultiHeadLatentAttention(**kw)


_ROPE_BASE_NOTE = (
    "the conventional RoPE base; this module does not read a base from DeepSeek-V2's text"
)

register(
    MixerSpec(
        name="mla",
        family="cache",
        summary=(
            "Caches one low-rank latent and one shared rotated key per token instead of per-head "
            "keys and values, rebuilding every head's key and value from the latent."
        ),
        parent="gqa",
        covers="mla",
        source="catalogue:mla",
        checked_against="arXiv:2405.04434v5 §2.1.2 Eq 9–13 and §2.1.3 Eq 14–19",
        factory=_mla,
        lab=(
            hparams.ours(
                "d_model",
                32,
                "width of the token vectors",
                "lab scale, small enough to run on a laptop",
            ),
            hparams.ours(
                "heads",
                4,
                "number of attention heads",
                "lab scale, a few heads so per-head structure is visible",
            ),
            hparams.ours(
                "head_dim",
                8,
                "width of each head's content query, key and value",
                "lab scale, d_model divided by the number of heads",
            ),
            hparams.ours(
                "kv_rank",
                12,
                "width of the cached key/value latent",
                "lab scale, smaller than heads times head_dim so compression is real",
            ),
            hparams.ours(
                "q_rank",
                16,
                "width of the query latent",
                "lab scale, between kv_rank and heads times head_dim as in the source's ordering",
            ),
            hparams.ours(
                "rope_dim",
                4,
                "width of the decoupled rotary query per head and of the shared rotary key",
                "lab scale, half of head_dim as in the source's ratio",
            ),
            hparams.ours("rope_base", 10000.0, "RoPE frequency base", _ROPE_BASE_NOTE),
        ),
        paper=(
            hparams.from_lab_source("d_model", "mla.d_model", "width of the token vectors"),
            hparams.from_lab_source("heads", "mla.heads", "number of attention heads"),
            hparams.from_lab_source(
                "head_dim", "mla.head_dim", "width of each head's content query, key and value"
            ),
            hparams.from_lab_source("kv_rank", "mla.kv_rank", "width of the cached latent"),
            hparams.from_lab_source("q_rank", "mla.q_rank", "width of the query latent"),
            hparams.from_lab_source(
                "rope_dim", "mla.rope_dim", "width of the decoupled rotary query and key"
            ),
            hparams.ours("rope_base", 10000.0, "RoPE frequency base", _ROPE_BASE_NOTE),
        ),
        state_growth="grows",
    )
)
