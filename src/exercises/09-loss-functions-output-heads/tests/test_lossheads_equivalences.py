"""The equivalences are the lesson, so they are assertions rather than prose.

Each function in `losses.py` reduces to plain cross-entropy at one setting of its own knob, and each
reduction is a fact worth holding: label smoothing at `epsilon = 0`, a z-loss weight of `0`, and
chunking at any block size all give back the base. A paragraph claiming that is worth less than an
assertion that goes red when it stops being true.

**Every equivalence is written twice** — once showing it holds at the no-op setting, once showing
the knob actually *does* something away from it. Without the second half a function that ignored its
argument entirely would pass the first, which is the "guard that cannot fail" shape this repository
keeps finding.

`torch` is the `train` extra, so this file skips without it and is registered in
`OPTIONAL_DEPENDENCY_GATES` and in the `train` CI job. A gated file in neither runs **nowhere**
while every gate stays green, which has already cost this repository 46 tests once.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra: uv sync --extra train")

from lossheads.config import Config  # noqa: E402
from lossheads.heads import (  # noqa: E402
    make_tied_head,
    make_untied_head,
    tied_head_params,
    untied_head_params,
)
from lossheads.losses import (  # noqa: E402
    chunked_cross_entropy,
    cross_entropy,
    cross_entropy_with_z_loss,
    label_smoothed_cross_entropy,
    z_loss,
)

TOLERANCE = 1e-6


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A small scoring problem with a row count that is NOT a multiple of any chunk size used.

    259 rows is deliberate: chunking bugs that average the chunk means rather than weighting by row
    count only show up when the final block is short, and a round number hides them.
    """
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(259, 64, generator=generator)
    targets = torch.randint(0, 64, (259,), generator=generator)
    return logits, targets


def test_label_smoothing_at_zero_is_plain_cross_entropy(batch) -> None:
    """The no-op setting must give back the base exactly."""
    logits, targets = batch
    assert torch.allclose(
        label_smoothed_cross_entropy(logits, targets, epsilon=0.0),
        cross_entropy(logits, targets),
        atol=TOLERANCE,
    )


def test_label_smoothing_bites_on_a_confident_model_and_barely_moves_a_uniform_one(batch) -> None:
    """Its twin — and the first version of it was measuring the wrong thing.

    Driven on the random logits of the fixture, `epsilon=0.1` moved the loss by **0.0003**: 4.6337
    to 4.6340. That is not a tolerance to widen. Random logits are near-uniform, and on a uniform
    model the target's own log-probability and the mean log-probability are nearly equal — so
    moving mass between them changes almost nothing.

    **That is the technique, not an obstacle to testing it.** Label smoothing is a penalty on
    confidence, so it is close to a no-op precisely when there is no confidence to penalise. The
    test therefore asserts both halves: sharp logits move a lot, flat ones barely move. A version
    checking only "the number changed" would have passed on noise and taught nothing.
    """
    _, targets = batch
    flat = torch.zeros(259, 64)
    sharp = torch.zeros(259, 64).scatter_(1, targets.unsqueeze(-1), 20.0)

    sharp_shift = (
        label_smoothed_cross_entropy(sharp, targets, epsilon=0.1) - cross_entropy(sharp, targets)
    ).abs()
    flat_shift = (
        label_smoothed_cross_entropy(flat, targets, epsilon=0.1) - cross_entropy(flat, targets)
    ).abs()

    assert sharp_shift > 0.1, f"smoothing did nothing to a confident model: {sharp_shift}"
    assert flat_shift < 1e-5, f"smoothing moved a uniform model, which it should not: {flat_shift}"


def test_a_zero_z_loss_weight_is_plain_cross_entropy(batch) -> None:
    """Same shape of claim, different knob."""
    logits, targets = batch
    assert torch.allclose(
        cross_entropy_with_z_loss(logits, targets, weight=0.0),
        cross_entropy(logits, targets),
        atol=TOLERANCE,
    )


def test_the_z_loss_penalises_logit_scale_that_the_base_loss_cannot_see(batch) -> None:
    """The property the term exists for, asserted rather than described.

    Softmax is shift-invariant, so adding a constant to every logit leaves cross-entropy **exactly**
    unchanged — which is why the raw scale is free to drift. The z-loss is what notices.
    """
    logits, targets = batch
    shifted = logits + 5.0

    assert torch.allclose(
        cross_entropy(shifted, targets), cross_entropy(logits, targets), atol=TOLERANCE
    ), "a constant shift changed cross-entropy, so softmax shift-invariance is not holding"
    assert z_loss(shifted) > z_loss(logits), "the z-loss did not notice a shift of every logit"


def test_chunked_cross_entropy_equals_the_unchunked_value(batch) -> None:
    """Chunking changes where the memory goes and nothing else.

    Driven at several block sizes, including ones that do not divide the row count and one larger
    than the batch, because the failure mode is entirely about the short final block.
    """
    logits, targets = batch
    expected = cross_entropy(logits, targets)
    for chunk in (1, 7, 128, 259, 1024):
        assert torch.allclose(
            chunked_cross_entropy(logits, targets, chunk_size=chunk), expected, atol=TOLERANCE
        ), f"chunk_size={chunk} disagreed with the unchunked loss"


def test_chunking_rejects_a_nonsense_block_size(batch) -> None:
    """A zero or negative chunk size silently returns 0.0 rows' worth of loss if unchecked."""
    logits, targets = batch
    for bad in (0, -1):
        with pytest.raises(ValueError, match="chunk_size"):
            chunked_cross_entropy(logits, targets, chunk_size=bad)


def test_a_tied_head_shares_storage_with_the_embedding_rather_than_copying_it() -> None:
    """Equality after construction proves nothing — a copy is equal too, and then drifts.

    If the two were separate tensors, one gradient step would move them apart and the head would
    quietly stop being tied while every equality assertion still passed at step zero.
    """
    embedding = torch.nn.Embedding(64, 16)
    head = make_tied_head(embedding)

    assert head.weight is embedding.weight, "the tied head holds a copy, not the same tensor"
    with torch.no_grad():
        embedding.weight[0, 0] += 1.0
    assert head.weight[0, 0] == embedding.weight[0, 0], (
        "editing the embedding did not move the head"
    )


def test_a_tied_head_scores_exactly_as_a_linear_layer_with_those_weights() -> None:
    """The equivalence that makes tying checkable rather than merely plausible."""
    embedding = torch.nn.Embedding(64, 16)
    head = make_tied_head(embedding)
    hidden = torch.randn(5, 16)

    assert torch.allclose(head(hidden), hidden @ embedding.weight.T, atol=TOLERANCE)


def test_tying_removes_the_head_from_the_parameter_count() -> None:
    """The saving, stated as arithmetic rather than as a claim."""
    config = Config()
    assert untied_head_params(config.d_model, config.vocab_size) == config.head_params
    assert tied_head_params(config.d_model, config.vocab_size) == 0

    untied = make_untied_head(config.d_model, 64)
    embedding = torch.nn.Embedding(64, config.d_model)
    tied = make_tied_head(embedding)
    assert untied.weight is not embedding.weight
    assert tied.weight is embedding.weight


def test_the_head_is_most_of_a_small_model_and_the_share_is_derived(batch) -> None:
    """The claim this exercise exists to make, computed rather than typed.

    At the default width the head is a large minority of the parameters; widen the model and the
    body grows quadratically while the head grows linearly, so the share falls. Asserting the
    *direction* rather than a fixed percentage is deliberate — a hard-coded figure would be a number
    in prose, which is the failure this repository has paid for most often.
    """
    narrow = Config(d_model=256)
    wide = Config(d_model=2048)

    assert narrow.head_share > wide.head_share, (
        "widening the model did not reduce the head's share, so the arithmetic is wrong"
    )
    assert 0.0 < wide.head_share < narrow.head_share < 1.0
