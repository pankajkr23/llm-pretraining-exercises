"""The sparse family's identities, and its masks against sets written out by hand from the papers.

Each variant chooses which earlier keys a query reads. Two kinds of check follow:

- **the sets themselves** — the index sets of the Sparse Transformer, NSA's and CSA's block
  visibility, StreamingLLM's worked example — written out by hand from each paper's definition;
- **the limits** — widen the subset until it holds every earlier key and the variant must become
  ordinary causal attention over its own projections, computed here with `ops.attend`.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="the attention lab needs the train extra")

from attention.lab import ops, registry, sparse  # noqa: E402

T = 12


def _build(name: str, **overrides):
    torch.manual_seed(0)
    return registry.get(name).build("lab", **overrides).double().eval()


def _x(tokens: int = T, batch: int = 2, width: int = 32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, tokens, width, generator=generator, dtype=torch.float64)


def _sets(mask) -> list[set[int]]:
    return [set(torch.nonzero(row).flatten().tolist()) for row in mask]


def _pos(n: int):
    return torch.arange(n)


def _rope_tail(x, positions, rope_dim, base):
    """RoPE on the last `rope_dim` dimensions, written here so the module cannot grade itself."""
    angles = ops.rope_angles(positions, rope_dim, base)
    keep = x.shape[-1] - rope_dim
    return torch.cat([x[..., :keep], ops.apply_rope(x[..., keep:], angles)], dim=-1)


def _causal_reference(mixer, x, allowed=None):
    """Plain attention over the mixer's own projections."""
    q, k, v = mixer.project(x)
    tokens = x.shape[1]
    allowed = ops.causal_mask(tokens, tokens) if allowed is None else allowed
    out, _ = ops.attend(q, k, v, allowed=allowed)
    return mixer.w_o(ops.merge_heads(out))


# --- sliding_window -------------------------------------------------------------------------------


def test_the_window_is_the_last_w_positions_including_the_query() -> None:
    mask = sparse.window_mask(_pos(6), _pos(6), 3)
    assert _sets(mask) == [{0}, {0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}]


def test_a_window_as_long_as_the_sequence_is_full_causal_attention() -> None:
    mixer = _build("sliding_window", window=T)
    x = _x()
    with torch.no_grad():
        y, _ = mixer(x)
        torch.testing.assert_close(y, _causal_reference(mixer, x))


def test_the_window_cache_keeps_exactly_w_keys() -> None:
    mixer = _build("sliding_window", window=5)
    with torch.no_grad():
        _, state = mixer(_x())
    assert state["k"].shape[2] == 5


# --- sparse_attention -----------------------------------------------------------------------------

# §4.3, strided, l = 3, n = 8: A1 = {max(0, i-3), ..., i}; A2 = {j <= i : (i - j) mod 3 = 0}.
STRIDED_A1 = [{0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6},
              {4, 5, 6, 7}]  # fmt: skip
STRIDED_A2 = [{0}, {1}, {2}, {0, 3}, {1, 4}, {2, 5}, {0, 3, 6}, {1, 4, 7}]
# §4.3, fixed, l = 3, c = 1, n = 8: A1 = same block of 3; A2 = {j <= i : j mod 3 = 2}.
FIXED_A1 = [{0}, {0, 1}, {0, 1, 2}, {3}, {3, 4}, {3, 4, 5}, {6}, {6, 7}]
FIXED_A2 = [set(), set(), {2}, {2}, {2}, {2, 5}, {2, 5}, {2, 5}]


def test_the_strided_sets_match_the_paper_definition() -> None:
    first, second = sparse.factorized_masks("strided", _pos(8), _pos(8), stride=3, c=1)
    assert _sets(first) == STRIDED_A1
    assert _sets(second) == STRIDED_A2


def test_the_fixed_sets_match_the_paper_definition() -> None:
    first, second = sparse.factorized_masks("fixed", _pos(8), _pos(8), stride=3, c=1)
    assert _sets(first) == FIXED_A1
    assert _sets(second) == FIXED_A2


def test_the_fixed_pattern_keeps_c_summary_positions_per_block() -> None:
    """With l = 4, c = 2 the summary cells are positions 2, 3 of each block (0-indexed)."""
    _, second = sparse.factorized_masks("fixed", _pos(12), _pos(12), stride=4, c=2)
    assert _sets(second)[11] == {2, 3, 6, 7, 10, 11}


def test_merged_heads_read_the_union_and_separate_heads_alternate() -> None:
    q, k = _pos(8), _pos(8)
    first, second = sparse.factorized_masks("fixed", q, k, stride=3, c=1)
    merged = _build("sparse_attention", stride=3, c=1).allowed(q, k)
    assert all(torch.equal(merged[h], first | second) for h in range(merged.shape[0]))
    separate = _build("sparse_attention", stride=3, c=1, combine="separate").allowed(q, k)
    assert torch.equal(separate[0], first) and torch.equal(separate[1], second)


@pytest.mark.parametrize("pattern", ["strided", "fixed"])
def test_a_stride_as_long_as_the_sequence_is_full_causal_attention(pattern) -> None:
    mixer = _build("sparse_attention", pattern=pattern, stride=T)
    x = _x()
    with torch.no_grad():
        y, _ = mixer(x)
        torch.testing.assert_close(y, _causal_reference(mixer, x))


# --- topk_attention -------------------------------------------------------------------------------


def test_topk_as_large_as_the_sequence_is_full_causal_attention() -> None:
    mixer = _build("topk_attention", topk=T)
    x = _x()
    with torch.no_grad():
        y, _ = mixer(x)
        torch.testing.assert_close(y, _causal_reference(mixer, x))


def test_a_query_with_fewer_than_k_earlier_tokens_reads_all_of_them() -> None:
    """The causal mask is applied before the selection, so early rows are ordinary rows."""
    topk = 5
    mixer = _build("topk_attention", topk=topk)
    x = _x()
    with torch.no_grad():
        y, _ = mixer(x)
        reference = _causal_reference(mixer, x)
    torch.testing.assert_close(y[:, :topk], reference[:, :topk])
    assert not torch.allclose(y[:, topk:], reference[:, topk:])


def test_each_row_keeps_exactly_k_keys_when_scores_are_distinct() -> None:
    mixer = _build("topk_attention", topk=3)
    x = _x(tokens=8)
    q, k, _ = mixer.project(x)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    causal = ops.causal_mask(8, 8)
    kept = []

    def spy(q_, k_, v_, allowed=None, **kw):
        kept.append(allowed)
        return original(q_, k_, v_, allowed=allowed, **kw)

    original = sparse.ops.attend
    sparse.ops.attend = spy
    try:
        with torch.no_grad():
            mixer(x)
    finally:
        sparse.ops.attend = original
    counts = kept[0].sum(dim=-1)
    expected = torch.clamp(torch.arange(8) + 1, max=3).expand_as(counts)
    assert torch.equal(counts, expected)
    # and the kept keys are the three highest scores among the visible ones
    masked = scores.masked_fill(~causal, float("-inf"))
    third = masked.topk(3, dim=-1).values[..., -1:]
    assert torch.equal(kept[0][..., 3:, :], (masked >= third)[..., 3:, :])


# --- reformer -------------------------------------------------------------------------------------


def test_keys_are_unit_queries_and_hash_like_them() -> None:
    mixer = _build("reformer")
    q, k, _ = mixer.project(_x())
    torch.testing.assert_close(k.norm(dim=-1), torch.ones_like(k.norm(dim=-1)))
    buckets = sparse.lsh_buckets(q, mixer.rotations)
    assert torch.equal(buckets, sparse.lsh_buckets(k, mixer.rotations))
    assert buckets.min() >= 0 and buckets.max() < mixer.n_buckets


def test_one_bucket_is_causal_attention_without_self_except_for_the_first_token() -> None:
    """With a single bucket every earlier key shares it; §2's self rule is all that remains."""
    mixer = _build("reformer", n_buckets=1)
    x = _x()
    i, j = _pos(T)[:, None], _pos(T)[None, :]
    expected = (j < i) | ((i == 0) & (j == 0))
    with torch.no_grad():
        y, _ = mixer(x)
        torch.testing.assert_close(y, _causal_reference(mixer, x, allowed=expected))


def test_the_attention_set_is_the_union_of_the_rounds() -> None:
    """Eq 6: a key is readable if it shares the query's bucket in any round."""
    mixer = _build("reformer", n_rounds=3)
    x = _x()
    q, k, v = mixer.project(x)
    qh, kh = sparse.lsh_buckets(q, mixer.rotations), sparse.lsh_buckets(k, mixer.rotations)
    union = torch.zeros(2, mixer.heads, T, T, dtype=torch.bool)
    for r in range(3):
        union |= qh[..., :, None, r] == kh[..., None, :, r]
    allowed = union & ops.causal_mask(T, T)
    bias = -mixer.self_penalty * torch.eye(T, dtype=torch.float64)
    out, _ = ops.attend(q, k, v, allowed=allowed, bias=bias)
    with torch.no_grad():
        y, _ = mixer(x)
    torch.testing.assert_close(y, mixer.w_o(ops.merge_heads(out)))


# --- attention_sinks ------------------------------------------------------------------------------


def test_the_papers_worked_example_assigns_positions_within_the_cache() -> None:
    """§3.2: cache [0, 1, 2, 3, 6, 7, 8] decoding token 9 -> positions [0, ..., 7]."""
    visible, key_rank, query_rank = sparse.in_cache_positions(
        torch.tensor([9]), _pos(10), sinks=4, window=4
    )
    assert _sets(visible) == [{0, 1, 2, 3, 6, 7, 8, 9}]
    assert key_rank[0, visible[0]].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert query_rank.tolist() == [7]


def _rope_reference(mixer, x, q_positions, k_positions, allowed):
    q = ops.split_heads(mixer.w_q(x), mixer.heads)
    k = ops.split_heads(mixer.w_k(x), mixer.heads)
    v = ops.split_heads(mixer.w_v(x), mixer.heads)
    q = ops.apply_rope(q, ops.rope_angles(q_positions, mixer.head_dim, mixer.rope_base))
    k = ops.apply_rope(k, ops.rope_angles(k_positions, mixer.head_dim, mixer.rope_base))
    out, _ = ops.attend(q, k, v, allowed=allowed)
    return mixer.w_o(ops.merge_heads(out))


def test_sinks_plus_window_covering_the_sequence_is_rope_causal_attention() -> None:
    mixer = _build("attention_sinks", window=T - 4)
    x = _x()
    with torch.no_grad():
        y, _ = mixer(x)
        reference = _rope_reference(mixer, x, _pos(T), _pos(T), ops.causal_mask(T, T))
    torch.testing.assert_close(y, reference)


def test_a_late_query_reads_sinks_and_window_at_positions_counted_in_the_cache() -> None:
    sinks, window = 4, 4
    mixer = _build("attention_sinks", sinks=sinks, window=window)
    x = _x()
    kept = [0, 1, 2, 3, 8, 9, 10, 11]  # what the last query reads
    with torch.no_grad():
        y, _ = mixer(x)
        reference = _rope_reference(mixer, x[:, kept], _pos(8), _pos(8), ops.causal_mask(8, 8))
    torch.testing.assert_close(y[:, -1], reference[:, -1])


def test_the_sink_cache_is_bounded_to_sinks_plus_window() -> None:
    mixer = _build("attention_sinks")
    with torch.no_grad():
        _, state = mixer(_x(tokens=30))
    lab = mixer.sinks + mixer.window
    assert state["k_pos"].tolist() == [*range(mixer.sinks), *range(30 - mixer.window, 30)]
    assert state["k"].shape[2] == lab


# --- nsa ------------------------------------------------------------------------------------------


def test_a_compressed_block_is_visible_once_complete() -> None:
    """Eq 7 with l = 4, d = 2: block i covers tokens [2i, 2i + 4) and needs 2i + 4 <= t."""
    mask = sparse.nsa_compressed_visible(_pos(10), blocks=4, block=4, stride=2)
    expected = [set(), set(), set(), {0}, {0}, {0, 1}, {0, 1}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2, 3}]
    assert _sets(mask) == expected


def test_the_selection_map_follows_eq_9() -> None:
    """l = 2, d = 1, l' = 2: p_slc[j] sums p_cmp[2j - m - n] for m, n in {0, 1}."""
    mapping = sparse.nsa_selection_map(3, 5, block=2, stride=1, sel=2)
    expected = [[1, 0, 0, 0, 0], [1, 2, 1, 0, 0], [0, 0, 1, 2, 1]]
    assert mapping.tolist() == expected
    identity = sparse.nsa_selection_map(4, 4, block=4, stride=4, sel=4)
    assert torch.equal(identity, torch.eye(4))


def _nsa_reference(mixer, x, branch):
    q = ops.split_heads(mixer.w_q(x), mixer.heads)
    k, v = mixer._kv(x, branch)
    groups = mixer.heads // mixer.kv_heads
    out, _ = ops.attend(
        q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), allowed=ops.causal_mask(T, T)
    )
    return out


def test_selecting_every_block_makes_the_selection_branch_causal_attention() -> None:
    mixer = _build("nsa", selected=T)
    x = _x()
    with torch.no_grad():
        outs, _ = mixer.branch_outputs(x)
        torch.testing.assert_close(outs["slc"], _nsa_reference(mixer, x, "slc"))


def test_a_window_as_long_as_the_sequence_makes_the_window_branch_causal_attention() -> None:
    mixer = _build("nsa", window=T)
    x = _x()
    with torch.no_grad():
        outs, _ = mixer.branch_outputs(x)
        torch.testing.assert_close(outs["win"], _nsa_reference(mixer, x, "win"))


def test_with_only_forced_blocks_the_selection_is_the_initial_and_local_blocks() -> None:
    mixer = _build("nsa", selected=2, initial_blocks=1, local_blocks=1)
    weights = torch.rand(1, mixer.heads, T, 11, dtype=torch.float64)
    chosen = mixer.select_blocks(weights, _pos(T), T)  # six blocks of two tokens
    expected = [{0} if i < 2 else {0, i // 2} for i in range(T)]
    for group in range(mixer.kv_heads):
        assert _sets(chosen[0, group]) == expected


def test_the_compression_branch_is_silent_before_the_first_complete_block() -> None:
    mixer = _build("nsa", block=4, stride=2, sel_block=4)
    with torch.no_grad():
        outs, _ = mixer.branch_outputs(_x())
    assert torch.equal(outs["cmp"][:, :, :3], torch.zeros_like(outs["cmp"][:, :, :3]))
    assert outs["cmp"][:, :, 3].abs().sum() > 0


# --- deepseek_csa ---------------------------------------------------------------------------------


def test_an_entry_is_visible_only_after_the_querys_own_block() -> None:
    """Eq 16, s < floor(t / m), m = 2."""
    mask = sparse.csa_visible(_pos(7), entries=3, m=2)
    assert _sets(mask) == [set(), set(), {0}, {0}, {0, 1}, {0, 1}, {0, 1, 2}]


def test_entry_i_is_built_from_tokens_m_i_minus_1_to_m_i_plus_1() -> None:
    """Eq 11–12: entry i mixes stream b over [m(i-1), mi) and stream a over [mi, m(i+1))."""
    mixer = _build("deepseek_csa")
    compressor, m = mixer.kv, mixer.m
    x = _x(tokens=10, batch=1)
    empty = torch.zeros(1, 0, mixer.head_dim, dtype=torch.float64)
    tail = torch.zeros(1, 0, 4, mixer.head_dim, dtype=torch.float64)
    with torch.no_grad():
        base, _ = compressor.update(x, empty, tail, 10)
        for token in range(10):
            changed = x.clone()
            changed[:, token] += 1.0
            moved, _ = compressor.update(changed, empty, tail, 10)
            touched = {
                i for i in range(base.shape[1]) if not torch.allclose(base[:, i], moved[:, i])
            }
            owners = {i for i in range(base.shape[1]) if m * (i - 1) <= token < m * (i + 1)}
            assert touched == owners, token


def test_topk_covering_every_entry_selects_every_visible_entry() -> None:
    mixer = _build("deepseek_csa", topk=T)
    x = _x()
    q_pos = _pos(T)
    with torch.no_grad():
        _, state = mixer(x)
        chosen = mixer.select(x, state["index_keys"], q_pos)
    visible = sparse.csa_visible(q_pos, state["index_keys"].shape[1], mixer.m)
    assert torch.equal(chosen, visible.expand_as(chosen))


def _randomise_indexer(mixer, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in (mixer.index_kv, mixer.w_iuq, mixer.w_w):
            for p in module.parameters():
                p.copy_(torch.randn(p.shape, generator=generator, dtype=p.dtype))


@pytest.mark.parametrize(("topk", "invariant"), [(T, True), (1, False)])
def test_with_every_entry_selected_the_indexer_has_no_effect(topk, invariant) -> None:
    mixer = _build("deepseek_csa", topk=topk)
    x = _x()
    with torch.no_grad():
        before, _ = mixer(x)
        _randomise_indexer(mixer, seed=5)
        after, _ = mixer(x)
    assert torch.allclose(before, after) is invariant


def _csa_output(mixer, core):
    per_head = core.transpose(1, 2)
    chunks = per_head.chunk(mixer.groups, dim=2)
    mids = [proj(c.flatten(-2)) for proj, c in zip(mixer.w_group, chunks, strict=True)]
    return mixer.w_o(torch.cat(mids, dim=-1))


def test_no_entries_and_a_full_window_is_causal_mqa_over_the_window() -> None:
    mixer = _build("deepseek_csa", topk=0, window=T, attention_sink=False)
    x = _x()
    pos = _pos(T)
    rope = mixer.rope_dim, mixer.rope_base
    with torch.no_grad():
        y, _ = mixer(x)
        q = mixer.q_norm(ops.split_heads(mixer.w_uq(mixer.w_dq(x)), mixer.heads))
        q = _rope_tail(q, pos, *rope)
        kv = _rope_tail(mixer.kv_norm(mixer.w_win(x)), pos, *rope)[:, None]
        core, _ = ops.attend(
            q, kv, kv, allowed=ops.causal_mask(T, T), scale=1 / math.sqrt(mixer.head_dim)
        )
        reference = _csa_output(mixer, _rope_tail(core, -pos, *rope))
    torch.testing.assert_close(y, reference)


def test_sink_logits_add_to_the_denominator() -> None:
    """Eq 27: a very large sink takes all the weight; a very small one takes none."""
    with_sink = _build("deepseek_csa")
    without = _build("deepseek_csa", attention_sink=False)
    x = _x()
    with torch.no_grad():
        with_sink.sink.fill_(1e4)
        swamped, _ = with_sink(x)
        with_sink.sink.fill_(-1e4)
        silent, _ = with_sink(x)
        plain, _ = without(x)
    torch.testing.assert_close(swamped, torch.zeros_like(swamped))
    torch.testing.assert_close(silent, plain)


# --- msa ------------------------------------------------------------------------------------------


def test_a_block_scores_its_best_visible_token_and_a_future_block_scores_minus_infinity() -> None:
    """Eq 6 with B_k = 2 on four keys."""
    scores = torch.tensor([[5.0, 1.0, 9.0, 9.0], [5.0, 7.0, 9.0, 9.0], [0.0, 1.0, 2.0, 9.0]])
    q_pos, k_pos = torch.tensor([0, 1, 2]), _pos(4)
    blocks = sparse.msa_block_scores(scores, q_pos, k_pos, block=2)
    inf = float("inf")
    assert blocks.tolist() == [[5.0, -inf], [7.0, -inf], [1.0, 2.0]]


def test_with_one_block_a_query_reads_its_own_block_up_to_itself() -> None:
    mixer = _build("msa", selected=1)
    with torch.no_grad():
        work, _ = mixer._step(_x(), None)
    expected = [set(range(2 * (i // 2), i + 1)) for i in range(T)]
    for group in range(mixer.kv_heads):
        assert _sets(work["readable"][0, group]) == expected


def test_selecting_every_block_is_causal_gqa_attention() -> None:
    mixer = _build("msa", selected=T)
    x = _x()
    pos = _pos(T)
    groups = mixer.heads // mixer.kv_heads
    rope = mixer.rope_dim, mixer.rope_base
    with torch.no_grad():
        y, _ = mixer(x)
        q = _rope_tail(ops.split_heads(mixer.w_q(x), mixer.heads), pos, *rope)
        k = _rope_tail(ops.split_heads(mixer.w_k(x), mixer.kv_heads), pos, *rope)
        v = ops.split_heads(mixer.w_v(x), mixer.kv_heads)
        out, _ = ops.attend(
            q, ops.repeat_kv(k, groups), ops.repeat_kv(v, groups), allowed=ops.causal_mask(T, T)
        )
    torch.testing.assert_close(y, mixer.w_o(ops.merge_heads(out)))


def test_the_alignment_loss_trains_only_the_index_projections() -> None:
    """Eq 9–11: the teacher and the index input are detached."""
    mixer = _build("msa")
    loss = mixer.index_alignment_loss(_x().requires_grad_())
    assert torch.isfinite(loss) and loss >= -1e-12
    loss.backward()
    for name, p in mixer.named_parameters():
        reached = p.grad is not None and bool(p.grad.abs().sum() > 0)
        assert reached == name.startswith(("w_q_idx", "w_k_idx")), name
