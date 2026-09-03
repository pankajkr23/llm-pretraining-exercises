"""Guards for the parts of the harness that a green suite would otherwise not cover.

The harness prints, and printing is not testable in a way worth the effort. What **is** worth
pinning is everything the printing rests on: that the masks drop what they claim, that the two
memory paths compute the same loss, that the multi-token heads are genuinely independent, and that
the training log's summary answers the question the documents ask it.

Every claim is written twice where a twin is possible — once at the setting where it must hold, once
where it must not — because a guard nobody has watched fail is not a guard.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch is the `train` extra: uv sync --extra train")

from lossheads.config import Config  # noqa: E402
from lossheads.heads import head_costs, make_multi_token_head, multi_head_params  # noqa: E402
from lossheads.losses import (  # noqa: E402
    chunked_projection_cross_entropy,
    contributing,
    cross_entropy,
    perplexity,
)
from lossheads.masks import (  # noqa: E402
    NO_DOCUMENT,
    keep_non_padding,
    keep_within_document,
    masked_targets,
    pack_documents,
    pad_sequences,
)
from lossheads.model import build_trunk, count_parameters  # noqa: E402
from lossheads.shift import shift_for_horizon, shift_for_next_token, shift_wrong_way  # noqa: E402
from lossheads.training import TrainingLog, save, train  # noqa: E402

SMALL = Config(vocab_size=97, d_model=32, n_layer=2, n_head=2, seq_len=24, batch_size=2, pad_id=96)
"""Tiny everywhere, for the pure-arithmetic guards that never touch the real tokenizer."""

TRAINABLE = Config(d_model=32, n_layer=2, n_head=2, seq_len=24, batch_size=2)
"""Tiny except for the vocabulary, which stays at the tokenizer's size.

Shrinking `vocab_size` for speed does not shrink the tokenizer, so the corpus produces ids the
embedding cannot look up. `_corpus` now refuses that with a message naming the cause; the first
version of this test hit it as a bare `IndexError` from inside torch.
"""


def test_the_trunk_produces_the_shape_the_harness_prints() -> None:
    """Item 1 is a claim about shapes, so the shapes are asserted rather than described."""
    trunk = build_trunk(SMALL)
    tokens = torch.randint(0, SMALL.vocab_size, (SMALL.batch_size, SMALL.seq_len))
    hidden = trunk(tokens)
    assert hidden.shape == (SMALL.batch_size, SMALL.seq_len, SMALL.d_model)
    assert count_parameters(trunk) > 0


def test_the_trunk_owns_no_output_head() -> None:
    """The split the whole exercise rests on, asserted rather than trusted.

    If the trunk grew a head, one trunk could not feed two of them, and Part 2 would need a
    different model — which is exactly the coupling this design exists to avoid.
    """
    trunk = build_trunk(SMALL)
    names = [n for n, _ in trunk.named_parameters()]
    assert not any("head" in n for n in names), f"the trunk grew an output head: {names}"


def test_padding_masking_changes_the_contributing_count() -> None:
    """Item 3. The count is the deliverable, so the count is what is asserted."""
    tokens = pad_sequences([[1, 2, 3, 4], [5, 6]], SMALL.seq_len, SMALL)
    inputs, targets = shift_for_next_token(tokens)
    keep, report = keep_non_padding(inputs, targets, SMALL)

    assert report.total == targets.numel()
    assert report.dropped > 0, "nothing was masked, so this proves nothing"
    assert report.contributing == contributing(masked_targets(targets, keep, SMALL), SMALL)


def test_padding_masking_drops_nothing_when_there_is_no_padding() -> None:
    """The twin. A mask that always drops something would pass the test above."""
    tokens = torch.randint(0, SMALL.pad_id, (SMALL.batch_size, SMALL.seq_len))
    inputs, targets = shift_for_next_token(tokens)
    _, report = keep_non_padding(inputs, targets, SMALL)
    assert report.dropped == 0, "positions were masked in a batch with no padding at all"


def test_the_boundary_mask_drops_the_join_and_every_padding_position() -> None:
    """Item 4, asserted against a hand-counted expectation rather than the implementation's own sum.

    **The first version of this test was tautological and it hid a real bug.** It computed
    `crossings = (owners[:, :-1] != owners[:, 1:]).sum()` and asserted `report.dropped == crossings`
    — which is the same expression `keep_within_document` used, so it held for every input. The
    implementation kept every pad-to-pad pair (`-1 == -1` is `True`) and this test agreed with it.

    So the expectation is now written out. Six real tokens in twelve slots: positions 0-2 are
    document 0, 3-5 are document 1, 6-11 are padding. Of the eleven `t -> t+1` pairs, exactly two
    stay: 0->1 and 1->2 inside document 0, and 3->4 and 4->5 inside document 1 — four. Everything
    else either crosses the join or touches padding.
    """
    tokens, owners = pack_documents([[1, 2, 3], [4, 5, 6]], 12, SMALL)
    keep, report = keep_within_document(owners, horizon=1)

    assert keep.shape == (1, 11)
    assert report.contributing == 4, (
        f"expected the four within-document pairs, got {report.contributing}: {keep.tolist()}"
    )
    assert report.dropped == 7


def test_the_boundary_mask_keeps_no_pair_that_touches_padding() -> None:
    """The twin, stated as the property rather than as a count — it is what actually went wrong."""
    import torch

    _, owners = pack_documents([[1, 2, 3], [4, 5, 6]], 12, SMALL)
    keep, _ = keep_within_document(owners, horizon=1)

    touches_padding = (owners[:, :-1] == NO_DOCUMENT) | (owners[:, 1:] == NO_DOCUMENT)
    assert not bool((keep & touches_padding).any()), (
        "a pair touching padding survived the boundary mask; two padding positions compare EQUAL, "
        "so an implementation testing only `source == destination` keeps every one of them"
    )
    assert torch.is_tensor(keep)


def test_the_boundary_mask_must_be_built_for_the_horizon_it_is_used_at() -> None:
    """The near-miss that survives review: a t+1 mask leaves one crossing pair per join at t+2."""
    _, owners = pack_documents([[1, 2, 3, 4], [5, 6, 7, 8]], 8, SMALL)
    _, near = keep_within_document(owners, horizon=1)
    _, far = keep_within_document(owners, horizon=2)
    assert far.dropped > near.dropped, (
        "a horizon-2 mask dropped no more than a horizon-1 mask, so it is not crossing-aware"
    )


def test_the_two_memory_paths_compute_the_same_loss() -> None:
    """Item 7's precondition. Without this the ratio is a comparison of two different things."""
    generator = torch.Generator().manual_seed(9)
    hidden = torch.randn(101, SMALL.d_model, generator=generator)
    weight = torch.randn(SMALL.vocab_size, SMALL.d_model, generator=generator)
    targets = torch.randint(0, SMALL.vocab_size, (101,), generator=generator)

    materialised = cross_entropy(hidden @ weight.T, targets, SMALL)
    for chunk in (1, 7, 32, 101, 512):
        chunked = chunked_projection_cross_entropy(hidden, weight, targets, chunk, SMALL)
        assert torch.allclose(chunked, materialised, atol=1e-5), f"chunk_size={chunk} disagreed"


def test_projection_chunking_respects_the_ignore_index() -> None:
    """The bug that survives every test written on unmasked input: the wrong denominator."""
    generator = torch.Generator().manual_seed(9)
    hidden = torch.randn(64, SMALL.d_model, generator=generator)
    weight = torch.randn(SMALL.vocab_size, SMALL.d_model, generator=generator)
    targets = torch.randint(0, SMALL.vocab_size, (64,), generator=generator)
    targets[::3] = SMALL.ignore_index

    expected = cross_entropy(hidden @ weight.T, targets, SMALL)
    got = chunked_projection_cross_entropy(hidden, weight, targets, 7, SMALL)
    assert torch.allclose(got, expected, atol=1e-5), (
        "chunked projection divided by the row count rather than the contributing count"
    )


def test_head_costs_reports_the_third_case_and_marks_it_unavailable() -> None:
    """Item 6's third row: tying needs rows to tie to, and some architectures have none."""
    with_rows = head_costs(SMALL, embedding_has_rows=True)
    without = head_costs(SMALL, embedding_has_rows=False)

    assert [c.arrangement for c in with_rows] == ["untied", "tied", "untied, tying unavailable"]
    assert with_rows[1].available and not without[1].available
    assert with_rows[1].added_params == 0
    assert without[2].added_params == SMALL.d_model * SMALL.vocab_size


def test_multi_token_heads_are_independent_and_priced_as_such() -> None:
    """Part 2's cost, and the fact that the heads share a trunk and nothing else."""
    config = Config(vocab_size=97, d_model=32, n_layer=2, n_head=2, horizons=(1, 2))
    heads = make_multi_token_head(config)
    hidden = torch.randn(2, 5, config.d_model)
    out = heads(hidden)

    assert set(out) == {1, 2}
    assert out[1].shape == (2, 5, config.vocab_size)
    assert not torch.allclose(out[1], out[2]), "both heads produced the same scores"
    assert multi_head_params(config) == 2 * config.d_model * config.vocab_size


def test_a_horizon_below_one_is_refused() -> None:
    """Horizon zero is a head predicting the token it was given. Refused rather than scored."""
    for bad in ((0,), (1, 0), ()):
        with pytest.raises(ValueError):
            make_multi_token_head(Config(horizons=bad))


def test_the_shift_at_horizon_two_drops_two_positions() -> None:
    """Part 2's data side is one slice, so the slice is what gets pinned."""
    tokens = torch.arange(20).reshape(2, 10)
    inputs, targets = shift_for_horizon(tokens, 2)
    assert inputs.shape == targets.shape == (2, 8)
    assert int(targets[0, 0]) == int(inputs[0, 0]) + 2


def test_a_horizon_reaching_past_the_sequence_is_refused() -> None:
    """Every position would be dropped, and the loss would be taken over nothing."""
    tokens = torch.arange(20).reshape(2, 10)
    for bad in (10, 11):
        with pytest.raises(ValueError, match="horizon"):
            shift_for_horizon(tokens, bad)


def test_perplexity_of_a_uniform_model_is_the_vocabulary_size() -> None:
    """Item 5's anchor, computed rather than asserted."""
    logits = torch.zeros(32, SMALL.vocab_size)
    targets = torch.randint(0, SMALL.vocab_size, (32,))
    assert perplexity(cross_entropy(logits, targets, SMALL)) == pytest.approx(
        SMALL.vocab_size, rel=1e-4
    )


def test_the_off_by_one_returns_the_input_as_its_own_target() -> None:
    """The deliberate bug, pinned so it cannot silently stop being the bug."""
    tokens = torch.arange(20).reshape(2, 10)
    inputs, targets = shift_wrong_way(tokens)
    assert torch.equal(inputs, targets)


def test_the_training_log_saves_before_a_long_run_is_attempted(tmp_path) -> None:
    """The last line of a long job, exercised first.

    Exercise 05 trained three experiments to completion and died writing their results — one lost
    fifteen trained models to a `json` failure in its final statement. Two steps costs seconds.
    """
    import json

    log = train(TRAINABLE, steps=2)
    path = save(log, tmp_path / "probe.json")
    data = json.loads(path.read_text())

    assert data["steps"] == [1, 2]
    assert set(data["by_horizon"]) == {"1", "2"}
    assert len(data["correct_shift"]) == len(data["broken_shift"]) == 2
    assert "summary" in data


def test_the_summary_answers_the_question_the_documents_ask_it() -> None:
    """Every published claim reads a field from here, so the fields have to exist and mean this."""
    log = TrainingLog(
        steps=[1, 2, 3],
        by_horizon={"1": [9.0, 8.0, 7.0], "2": [9.1, 8.3, 7.6]},
        correct_shift=[9.0, 8.0, 7.0],
        broken_shift=[9.0, 5.0, 2.0],
    )
    s = log.summary()
    assert s["further_head_is_harder"] is True
    assert s["gap"] == pytest.approx(0.6)
    assert s["steps_where_further_head_was_higher"] == 3
    assert s["broken_shift_is_lower"] is True


def test_the_summary_says_so_when_the_expected_finding_does_not_hold() -> None:
    """The twin. A summary that always reports the expected answer is not a measurement."""
    log = TrainingLog(
        steps=[1, 2],
        by_horizon={"1": [9.0, 8.0], "2": [8.5, 7.0]},
        correct_shift=[9.0, 8.0],
        broken_shift=[9.0, 9.5],
    )
    s = log.summary()
    assert s["further_head_is_harder"] is False
    assert s["broken_shift_is_lower"] is False
    assert s["steps_where_further_head_was_higher"] == 0
