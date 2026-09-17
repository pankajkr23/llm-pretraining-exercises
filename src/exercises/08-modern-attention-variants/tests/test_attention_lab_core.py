"""The identities the origin and full-attention variants must satisfy, beyond the generic contract.

- The lab's explicit `ops.attend` is PyTorch's `scaled_dot_product_attention`, written out.
- Grouped-query attention with a group per head is multi-head attention, and with one group is
  multi-query attention — and, more usefully, a GQA layer equals a multi-head layer whose key and
  value projections are copied once per group member, which is what "shares" means.
- The KV cache each variant keeps is exactly the bytes `attention.cache.kv_cache_bytes` prices.
- FlashAttention's tiled loop and its fused path both return exactly what `ops.attend` returns.
- Bahdanau attention's weights over the memory sum to one, and a query's output depends only on
  that query.

Each identity was broken on purpose before it was trusted; the scratch scripts that did so are not
part of the repository.
"""

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.cache import kv_cache_bytes  # noqa: E402
from attention.config import BYTES_PER_NUMBER, Yardstick  # noqa: E402
from attention.lab import core, ops, registry  # noqa: E402
from attention.lab.base import as_kwargs  # noqa: E402

F64 = torch.float64
NAMES = ("bahdanau_attention", "standard_attention", "mqa", "gqa", "flashattention")


def _qkv(seed: int = 0, batch: int = 2, heads: int = 3, queries: int = 9, keys: int = 9):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, queries, 8, generator=g, dtype=F64)
    k = torch.randn(batch, heads, keys, 8, generator=g, dtype=F64)
    v = torch.randn(batch, heads, keys, 6, generator=g, dtype=F64)
    return q, k, v


def _x(width: int, tokens: int = 11, seed: int = 3, batch: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, tokens, width, generator=g, dtype=F64)


def _build(name: str, **overrides):
    torch.manual_seed(0)
    return registry.get(name).build("lab", **overrides).double().eval()


# --- the reference itself -------------------------------------------------------------------------


def test_attend_is_scaled_dot_product_attention_with_a_causal_mask() -> None:
    q, k, v = _qkv()
    ours = ops.attend(q, k, v, allowed=ops.causal_mask(9, 9))[0]
    theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(ours, theirs)


def test_attend_matches_the_library_when_queries_follow_a_cache() -> None:
    q, k, v = _qkv(queries=3, keys=10)
    allowed = ops.causal_mask(3, 10, offset=7)
    ours = ops.attend(q, k, v, allowed=allowed)[0]
    theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    torch.testing.assert_close(ours, theirs)


# --- MHA, GQA and MQA are one family --------------------------------------------------------------


def test_gqa_with_a_group_per_head_is_standard_attention() -> None:
    standard = _build("standard_attention")
    heads = as_kwargs(registry.get("gqa").lab)["n_heads"]
    gqa = _build("gqa", n_kv_heads=heads)
    gqa.load_state_dict(standard.state_dict())
    x = _x(as_kwargs(registry.get("standard_attention").lab)["d_model"])
    with torch.no_grad():
        torch.testing.assert_close(gqa(x)[0], standard(x)[0])


def test_gqa_with_one_group_is_mqa() -> None:
    mqa = _build("mqa")
    gqa = _build("gqa", n_kv_heads=1)
    gqa.load_state_dict(mqa.state_dict())
    x = _x(as_kwargs(registry.get("mqa").lab)["d_model"])
    with torch.no_grad():
        torch.testing.assert_close(gqa(x)[0], mqa(x)[0])


def test_gqa_equals_standard_attention_with_its_key_value_heads_copied_per_group() -> None:
    """Sharing means exactly this: each query head reads the key/value head of its own group."""
    gqa = _build("gqa")
    kw = as_kwargs(registry.get("gqa").lab)
    heads, groups, hd = kw["n_heads"], kw["n_heads"] // kw["n_kv_heads"], kw["head_dim"]
    assert 1 < groups < heads, "the lab GQA must sit strictly between MQA and MHA"
    standard = _build("standard_attention")
    state = gqa.state_dict()
    for name in ("k.weight", "v.weight"):
        rows = state[name].view(kw["n_kv_heads"], hd, -1)
        state[name] = rows.repeat_interleave(groups, dim=0).reshape(heads * hd, -1)
    standard.load_state_dict(state)
    x = _x(kw["d_model"])
    with torch.no_grad():
        torch.testing.assert_close(gqa(x)[0], standard(x)[0])


@pytest.mark.parametrize("name", ["standard_attention", "gqa", "mqa", "flashattention"])
@pytest.mark.parametrize("tokens", [1, 13, 40])
def test_the_kv_cache_holds_exactly_what_the_cost_model_prices(name: str, tokens: int) -> None:
    kw = as_kwargs(registry.get(name).lab)
    torch.manual_seed(0)
    mixer = registry.get(name).build("lab").float()
    batch = 3
    x = torch.randn(batch, tokens, kw["d_model"])
    with torch.no_grad():
        _, state = mixer(x)
    kv_heads = kw.get("n_kv_heads", {"mqa": 1}.get(name, kw["n_heads"]))
    yardstick = Yardstick(
        layers=1,
        kv_heads=kv_heads,
        query_heads=kw["n_heads"],
        head_dim=kw["head_dim"],
        dtype="fp32",
    )
    assert state["k"].element_size() == BYTES_PER_NUMBER["fp32"]
    assert mixer.state_bytes(state) == kv_cache_bytes(yardstick, context=tokens, batch=batch)


def test_mqa_keeps_a_cache_n_heads_times_smaller_than_standard_attention() -> None:
    x = _x(32, tokens=20)
    with torch.no_grad():
        standard = _build("standard_attention")
        mqa = _build("mqa")
        big = standard.state_bytes(standard(x)[1])
        small = mqa.state_bytes(mqa(x)[1])
    assert big == small * as_kwargs(registry.get("mqa").lab)["n_heads"]


# --- FlashAttention computes the same numbers -----------------------------------------------------


@pytest.mark.parametrize("block", [1, 2, 4, 5, 9, 32])
@pytest.mark.parametrize(("queries", "keys", "offset"), [(9, 9, 0), (3, 10, 7), (1, 6, 5)])
def test_the_tiled_loop_equals_attend(block: int, queries: int, keys: int, offset: int) -> None:
    q, k, v = _qkv(queries=queries, keys=keys)
    allowed = ops.causal_mask(queries, keys, offset)
    expected = ops.attend(q, k, v, allowed=allowed)[0]
    torch.testing.assert_close(core.tiled_attention(q, k, v, allowed, block), expected)


def test_the_tiled_loop_reads_nothing_for_a_row_with_no_allowed_key() -> None:
    q, k, v = _qkv()
    allowed = ops.causal_mask(9, 9).clone()
    allowed[4] = False
    expected = ops.attend(q, k, v, allowed=allowed)[0]
    got = core.tiled_attention(q, k, v, allowed, 4)
    torch.testing.assert_close(got, expected)
    assert torch.count_nonzero(got[:, :, 4]) == 0


@pytest.mark.parametrize("tiled", [True, False])
def test_flashattention_equals_standard_attention_with_the_same_weights(tiled: bool) -> None:
    standard = _build("standard_attention")
    flash = _build("flashattention", tiled=tiled)
    flash.load_state_dict(standard.state_dict())
    x = _x(32)
    with torch.no_grad():
        torch.testing.assert_close(flash(x)[0], standard(x)[0])
        # Decoding after a prefix exercises the fused path's explicit mask, not `is_causal`.
        _, prefix = flash(x[:, :6])
        torch.testing.assert_close(flash(x[:, 6:], prefix)[0], standard(x)[0][:, 6:])


# --- Bahdanau attention ---------------------------------------------------------------------------


def _bahdanau_inputs(seed: int = 5):
    kw = as_kwargs(registry.get("bahdanau_attention").lab)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 6, kw["d_model"], generator=g, dtype=F64)
    memory = torch.randn(2, 7, kw["d_memory"], generator=g, dtype=F64)
    return x, memory


def test_bahdanau_weights_over_the_memory_sum_to_one_per_query() -> None:
    mixer = _build("bahdanau_attention")
    x, memory = _bahdanau_inputs()
    with torch.no_grad():
        weights = mixer.align(x, memory)
    assert weights.shape == (2, 6, 7)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 6, dtype=F64))
    assert bool((weights >= 0).all())


def test_bahdanau_output_for_a_query_ignores_every_other_query() -> None:
    mixer = _build("bahdanau_attention")
    x, memory = _bahdanau_inputs()
    changed = x.clone()
    changed[:, [0, 1, 3, 4, 5]] = torch.randn_like(changed[:, [0, 1, 3, 4, 5]])
    with torch.no_grad():
        after, before = mixer(changed, memory=memory)[0], mixer(x, memory=memory)[0]
    torch.testing.assert_close(after[:, 2], before[:, 2])


def test_bahdanau_output_is_the_projected_weighted_average_of_the_memory() -> None:
    mixer = _build("bahdanau_attention")
    x, memory = _bahdanau_inputs()
    with torch.no_grad():
        weights = mixer.align(x, memory)
        expected = torch.einsum("bqs,bsd->bqd", weights, memory) @ mixer.out.weight.T
        torch.testing.assert_close(mixer(x, memory=memory)[0], expected)


def test_bahdanau_refuses_to_run_without_a_memory() -> None:
    with pytest.raises(ValueError, match="memory"):
        _build("bahdanau_attention")(_bahdanau_inputs()[0])


# --- every variant builds at the scale its source states ------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_variant_builds_at_paper_scale(name: str) -> None:
    with torch.device("meta"):
        mixer = registry.get(name).build("paper")
    assert sum(p.numel() for p in mixer.parameters()) > 0


def test_flashattention_paper_width_is_the_stated_heads_times_head_width() -> None:
    kw = as_kwargs(registry.get("flashattention").paper)
    assert kw["d_model"] == kw["n_heads"] * kw["head_dim"]


def test_standard_attention_paper_width_agrees_with_its_heads() -> None:
    kw = as_kwargs(registry.get("standard_attention").paper)
    assert kw["d_model"] == kw["n_heads"] * kw["head_dim"]
