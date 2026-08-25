"""The model, tested for the things the data system depends on.

Not architecture quality — that is not what this session is about. What is tested here is that the
boundaries `masks.py` and `pack.py` construct **survive the attention kernel**, because that is the
one place they can be silently discarded, and that a checkpoint restores what it saved.

torch is an optional extra, so every test here skips without it. That is deliberate and it is why
the rest of the system is numpy: CI installs no torch and still verifies almost everything.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra, not a base dependency")

from trainingdata import masks, model, spec  # noqa: E402


def _small() -> "model.ModelConfig":
    """A model small enough to train inside a unit test.

    The vocabulary stays the real one. Shrinking it would put `EOS` and `PAD` outside the embedding
    table, and the fixtures feed both — the sentinels are part of the model's vocabulary by
    construction, which is the reason `MODEL_VOCAB_SIZE` exists at all.

    Returns:
        The config.
    """
    return model.ModelConfig(d_model=32, n_layer=2, n_head=2, d_ff=64)


def _built(seed: int = 0) -> "model.TinyGPT":
    """A model initialised from an explicit generator.

    Args:
        seed: Generator seed.

    Returns:
        The model.
    """
    return model.TinyGPT(_small(), generator=torch.Generator().manual_seed(seed))


def _batch(lengths: list[int], window: int, tokens: list[int]):
    """Pack one window by hand and return everything `forward` needs.

    Args:
        lengths: Document lengths within the window.
        window: Window size.
        tokens: Token ids, `window` of them.

    Returns:
        `(tokens, additive_mask, positions, segments)` as batched tensors.
    """
    segments = masks.segment_ids(lengths, window)
    return (
        torch.tensor([tokens], dtype=torch.long),
        torch.tensor(masks.additive_mask(segments)).unsqueeze(0).unsqueeze(0),
        torch.tensor(masks.position_ids(segments)).unsqueeze(0),
        segments,
    )


# --- the leak, proven through the real kernel ----------------------------------------------------


def test_a_later_document_cannot_see_an_earlier_one() -> None:
    """**The test the whole boundary apparatus exists for, in the direction that matters.**

    Causality alone already stops document A from seeing document B — A comes first. So the leak is
    entirely in the *other* direction: B attending back into A, learning that unrelated text is a
    natural continuation. A version of this test that only checked A's logits passed even when the
    block-diagonal mask was replaced with plain `is_causal=True`, which is how that hole was found.

    Bit-exact equality is the right bar because this is an *input* claim, not a float-arithmetic
    one: if B's logits are computed from anything belonging to A, they change.
    """
    net = _built().eval()
    same_b = [11, 12, 13, spec.EOS]

    with torch.no_grad():
        a = net(*_batch([4, 4], 8, [5, 6, 7, spec.EOS] + same_b)[:3])
        b = net(*_batch([4, 4], 8, [41, 42, 43, spec.EOS] + same_b)[:3])

    assert torch.equal(a[0, 4:], b[0, 4:]), (
        "rewriting document A changed document B's logits — the block-diagonal mask did not reach "
        "the attention kernel, and the model is learning cross-document continuations"
    )
    assert not torch.equal(a[0, :4], b[0, :4]), "document A's own logits did not change at all"


def test_an_earlier_document_cannot_see_a_later_one() -> None:
    """The causal half of the same claim.

    Weaker than the test above — plain causality gives it for free — but it fails if the mask is
    ever built non-causal, which the block-diagonal construction makes easy to get wrong.
    """
    net = _built().eval()
    same_a = [5, 6, 7, spec.EOS]

    with torch.no_grad():
        a = net(*_batch([4, 4], 8, same_a + [11, 12, 13, spec.EOS])[:3])
        b = net(*_batch([4, 4], 8, same_a + [41, 42, 43, spec.EOS])[:3])

    assert torch.equal(a[0, :4], b[0, :4])


def test_padding_cannot_influence_a_real_token() -> None:
    """Padding in the softmax dilutes every real weight, and the dilution looks like nothing."""
    net = _built().eval()
    with torch.no_grad():
        a = net(*_batch([4], 8, [5, 6, 7, spec.EOS] + [spec.PAD] * 4)[:3])
        b = net(*_batch([4], 8, [5, 6, 7, spec.EOS] + [9, 9, 9, 9])[:3])
    assert torch.equal(a[0, :4], b[0, :4])


def test_the_additive_mask_never_produces_nan() -> None:
    """A fully-masked row of `-inf` becomes `nan` after softmax, and one `nan` poisons every
    gradient it touches. `masks.NEG` is a large finite negative for exactly this reason."""
    net = _built()
    tokens, mask, positions, _ = _batch([2], 8, [5, spec.EOS] + [spec.PAD] * 6)
    logits = net(tokens, mask, positions)
    logits.sum().backward()
    assert torch.isfinite(logits).all(), "the forward pass produced nan or inf"
    for name, parameter in net.named_parameters():
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"


# --- RoPE, which is what makes the packing offsets usable ---------------------------------------


def test_attention_depends_on_the_gap_between_positions_not_their_absolute_value() -> None:
    """**Why RoPE and not a learned position table.**

    A 5,000-token document chopped into 512-token windows reaches position 4,999. A table sized to
    the window cannot represent that, and clamping would corrupt exactly the continuations
    `pack.py` exists to get right. RoPE has no table: shifting every position by a constant leaves
    the attention scores unchanged, so an offset of 4,999 is not a special case.
    """
    net = _built().eval()
    tokens, mask, positions, _ = _batch([8], 8, [5, 6, 7, 8, 9, 10, 11, spec.EOS])
    with torch.no_grad():
        near = net(tokens, mask, positions)
        far = net(tokens, mask, positions + 4_000)
    assert torch.allclose(near, far, atol=1e-4), (
        "logits moved when every position shifted by a constant — attention is reading absolute "
        "position, so a document continuing past the window would be corrupted"
    )


def test_a_position_far_past_the_window_is_not_a_special_case() -> None:
    """The failure a learned table would produce: an index error, or worse, a silent clamp."""
    net = _built().eval()
    tokens, mask, positions, _ = _batch([4], 4, [5, 6, 7, spec.EOS])
    with torch.no_grad():
        logits = net(tokens, mask, positions + 100_000)
    assert torch.isfinite(logits).all()


def test_rope_is_a_rotation_and_preserves_length() -> None:
    """If it scaled as well as rotated, position would leak into magnitude."""
    positions = torch.arange(6).unsqueeze(0)
    cos, sin = model.rope_tables(positions, head_dim=8, theta=10_000.0)
    x = torch.randn(1, 2, 6, 8, generator=torch.Generator().manual_seed(3))
    rotated = model.apply_rope(x, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_position_zero_is_the_identity_rotation() -> None:
    """Otherwise the first token of every document is rotated by an arbitrary amount."""
    cos, sin = model.rope_tables(torch.zeros(1, 4, dtype=torch.long), head_dim=8, theta=10_000.0)
    x = torch.randn(1, 2, 4, 8, generator=torch.Generator().manual_seed(4))
    assert torch.allclose(model.apply_rope(x, cos, sin), x, atol=1e-6)


# --- the loss ------------------------------------------------------------------------------------


def test_the_loss_grades_only_the_masked_positions() -> None:
    """The count returned must be the number of graded positions, not the window size.

    The caller weights by it; a wrong count silently reweights every step of the run.
    """
    logits = torch.zeros(1, 6, spec.MODEL_VOCAB_SIZE)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    loss_mask = torch.tensor([[True, True, False, True, False, False]])
    _, count = model.cross_entropy(logits, tokens, loss_mask)
    assert count == 3


def test_the_loss_mask_is_applied_after_the_shift() -> None:
    """An off-by-one here grades the token *after* each excluded one.

    Invisible in the loss curve, and it grades a padding token at every document boundary. The
    proof: excluding position 0 must change the loss, and excluding the final position must not —
    the final position is already dropped by the shift.
    """
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(1, 5, spec.MODEL_VOCAB_SIZE, generator=generator)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    everything = torch.ones(1, 5, dtype=torch.bool)

    base, base_n = model.cross_entropy(logits, tokens, everything)

    without_first = everything.clone()
    without_first[0, 0] = False
    changed, changed_n = model.cross_entropy(logits, tokens, without_first)
    assert changed_n == base_n - 1
    assert not torch.isclose(changed, base)

    without_last = everything.clone()
    without_last[0, 4] = False
    same, same_n = model.cross_entropy(logits, tokens, without_last)
    assert same_n == base_n, "the final position was graded — the mask was applied before the shift"
    assert torch.isclose(same, base)


def test_a_fully_masked_window_returns_zero_without_breaking_the_graph() -> None:
    """A window of pure padding is reachable at the tail of a shard.

    Returning a bare `0.0` would detach the graph and `backward()` would raise; returning `nan`
    would poison every gradient in the accumulation group.
    """
    net = _built()
    tokens, mask, positions, _ = _batch([1], 4, [spec.EOS] + [spec.PAD] * 3)
    logits = net(tokens, mask, positions)
    loss, count = model.cross_entropy(logits, tokens, torch.zeros(1, 4, dtype=torch.bool))
    assert count == 0
    assert float(loss.detach()) == 0.0
    loss.backward()  # must not raise


# --- initialisation ------------------------------------------------------------------------------


def test_the_same_generator_seed_builds_the_same_weights() -> None:
    """A run that cannot rebuild its own starting point cannot be resumed or forked."""
    a, b = _built(seed=7), _built(seed=7)
    for (name, left), (_, right) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(left, right), f"{name} differs between two identically-seeded builds"


def test_different_seeds_build_different_weights() -> None:
    """The control: without it, an initialiser that zeroed everything would pass the test above."""
    a, b = _built(seed=7), _built(seed=8)
    assert not torch.equal(a.embed.weight, b.embed.weight)


def test_building_the_model_does_not_disturb_the_global_rng() -> None:
    """Seeding globally from a constructor changes results anywhere else in the process.

    Including in whatever ran next in the test suite, which is how this kind of bug gets found
    three files away from where it lives.
    """
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    _built(seed=99)
    assert torch.equal(torch.randn(4), before), "building the model advanced or reseeded global RNG"


def test_the_output_head_is_tied_to_the_embedding() -> None:
    """Untied, the vocabulary is paid for twice — which dominates a model this small."""
    net = _built()
    assert net.head.weight is net.embed.weight
    untied = sum(p.numel() for p in net.parameters()) + net.embed.weight.numel()
    assert net.parameter_count < untied


@pytest.mark.parametrize(
    ("d_model", "n_head", "match"),
    [(32, 5, "not divisible"), (6, 2, "odd")],
)
def test_an_impossible_shape_is_refused_before_anything_is_allocated(
    d_model: int, n_head: int, match: str
) -> None:
    """The odd-head-width case is the sharp one: RoPE rotates pairs, so it cannot split an odd
    width, and the failure would otherwise appear as a shape error deep inside attention."""
    with pytest.raises(ValueError, match=match):
        model.TinyGPT(model.ModelConfig(d_model=d_model, n_head=n_head))


# --- the ML-native integration test ---------------------------------------------------------------


@pytest.mark.integration
def test_the_model_overfits_a_single_batch() -> None:
    """If loss does not collapse on one batch, nothing else measured on this model means anything.

    The repo's standard check, and the cheapest possible proof that gradients reach every parameter
    and the optimizer is connected to them.
    """
    net = _built(seed=11)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
    tokens, mask, positions, segments = _batch([8], 8, [5, 6, 7, 8, 9, 10, 11, spec.EOS])
    loss_mask = torch.tensor(masks.loss_mask(segments, np.asarray(tokens[0]))).unsqueeze(0)

    first = last = None
    for step in range(200):
        optimizer.zero_grad()
        total, count = model.cross_entropy(net(tokens, mask, positions), tokens, loss_mask)
        loss = total / count
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())

    # 9.21 is ln(10_002): an untrained model is uniform over its vocabulary, and starting anywhere
    # below that would mean the test began from a model that already knew something.
    assert first > 9.0, f"the model started near-converged ({first:.3f}); the test proves nothing"
    assert last < 0.05, f"loss did not collapse on seven tokens: {first:.3f} -> {last:.3f}"


@pytest.mark.integration
def test_a_checkpoint_round_trips_through_the_state_dict() -> None:
    """Resume restores from a file. If the round trip is lossy, every resumed run is a new run."""
    net = _built(seed=13)
    tokens, mask, positions, _ = _batch([8], 8, [5, 6, 7, 8, 9, 10, 11, spec.EOS])
    with torch.no_grad():
        before = net(tokens, mask, positions)

    restored = model.TinyGPT(_small(), generator=torch.Generator().manual_seed(999))
    assert not torch.equal(restored(tokens, mask, positions), before), "the models started equal"

    restored.load_state_dict(net.state_dict())
    with torch.no_grad():
        assert torch.equal(restored(tokens, mask, positions), before)


@pytest.mark.integration
def test_a_packed_window_trains_without_the_documents_bleeding() -> None:
    """The leak test, but after gradient descent rather than at initialisation.

    A mask that leaked only under some weight configuration would pass the eval-mode test and fail
    here — and the run would look healthy the whole time.
    """
    net = _built(seed=17)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
    tokens, mask, positions, segments = _batch([4, 4], 8, [5, 6, 7, spec.EOS, 11, 12, 13, spec.EOS])
    loss_mask = torch.tensor(masks.loss_mask(segments, np.asarray(tokens[0]))).unsqueeze(0)

    for _ in range(20):
        optimizer.zero_grad()
        total, count = model.cross_entropy(net(tokens, mask, positions), tokens, loss_mask)
        (total / count).backward()
        optimizer.step()

    # Document A is rewritten, document B is not; B's logits must be untouched. Checking A's would
    # be free under plain causality and would prove nothing -- see the eval-mode test above.
    other = torch.tensor([[41, 42, 43, spec.EOS, 11, 12, 13, spec.EOS]], dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(
            net(tokens, mask, positions)[0, 4:], net(other, mask, positions)[0, 4:]
        ), "documents bled into each other only after training — the mask is weight-dependent"
