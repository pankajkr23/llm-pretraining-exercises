"""Masks for packed sequences — where implementations get it wrong quietly.

The leak this file exists to catch has no symptom. If document B can attend to document A, nothing
crashes, the loss curve looks fine, and the model learns that unrelated text is a natural
continuation. So every claim here is asserted on the mask itself rather than on a downstream number.
"""

import numpy as np
import pytest
from trainingdata import masks, spec


def test_segment_ids_label_each_document_and_mark_padding() -> None:
    """Padding is `-1`, not a segment.

    A real id would make padding attend to itself and look like a document to every check below.
    """
    seg = masks.segment_ids([3, 2], window=8)
    assert seg.tolist() == [0, 0, 0, 1, 1, -1, -1, -1]


def test_documents_that_do_not_fit_are_refused() -> None:
    """Silently truncating would drop tokens the ledger records as consumed."""
    with pytest.raises(ValueError, match="do not fit"):
        masks.segment_ids([5, 5], window=8)


def test_position_ids_restart_at_each_document() -> None:
    """**The claim that makes packing invisible to the model.**

    Continuous positions would tell the model a document starting at offset 3 is 3 tokens into
    something. It is not — it is the start of its own text, and at inference it will be at 0.
    """
    seg = masks.segment_ids([3, 4], window=8)
    assert masks.position_ids(seg).tolist() == [0, 1, 2, 0, 1, 2, 3, 0]


def test_no_position_exceeds_its_own_document_length() -> None:
    """A stronger form of the same claim, over a ragged pack."""
    lengths = [5, 1, 9, 2]
    seg = masks.segment_ids(lengths, window=20)
    pos = masks.position_ids(seg)
    for i, length in enumerate(lengths):
        assert pos[seg == i].max() == length - 1


def test_a_document_cannot_attend_to_another_document() -> None:
    """**The leak with no symptom.**

    Nothing crashes if this is wrong. The loss curve looks fine. The model learns that unrelated
    text is a natural continuation — which is exactly what the source warns about, and why
    boundaries exist at all.
    """
    seg = masks.segment_ids([3, 3], window=6)
    allow = masks.attention_mask(seg)
    for i in range(6):
        for j in range(6):
            if allow[i, j]:
                assert seg[i] == seg[j], (
                    f"position {i} (doc {seg[i]}) may attend to {j} (doc {seg[j]})"
                )


def test_attention_is_causal_within_a_document() -> None:
    """A token may see its own past, itself, and nothing after it."""
    seg = masks.segment_ids([4], window=4)
    allow = masks.attention_mask(seg)
    assert allow.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_padding_attends_to_nothing_and_nothing_attends_to_padding() -> None:
    """Otherwise padding participates in the softmax and dilutes every real weight."""
    seg = masks.segment_ids([2], window=5)
    allow = masks.attention_mask(seg)
    pad = seg < 0
    assert not allow[pad].any(), "a padding row may attend somewhere"
    assert not allow[:, pad].any(), "something may attend to padding"


def test_every_real_token_can_at_least_attend_to_itself() -> None:
    """A row of all-False would produce a uniform-or-nan softmax over nothing.

    This is the twin of the two tests above: without it, a mask that forbade *everything* would pass
    both of them and break training silently.
    """
    seg = masks.segment_ids([3, 2], window=8)
    allow = masks.attention_mask(seg)
    for i in np.flatnonzero(seg >= 0):
        assert allow[i].any(), f"position {i} may attend to nothing at all"
        assert allow[i, i], f"position {i} cannot attend to itself"


def test_the_additive_mask_uses_a_finite_negative_not_minus_inf() -> None:
    """`-inf` on a fully-masked row gives `nan` after softmax, and one `nan` poisons the gradients.

    A large finite negative underflows to zero weight instead, which is the same thing numerically
    without the failure mode.
    """
    seg = masks.segment_ids([2], window=4)
    add = masks.additive_mask(seg)
    assert np.isfinite(add).all()
    assert add.dtype == np.float32
    assert set(np.unique(add).tolist()) == {0.0, masks.NEG}
    assert (add == 0.0).tolist() == masks.attention_mask(seg).tolist()


def test_the_last_token_of_a_document_earns_no_loss() -> None:
    """Next-token prediction has no target for it.

    Its "next token" is the first token of an unrelated document, and grading that would teach the
    cross-document continuation the block-diagonal mask exists to prevent.
    """
    seg = masks.segment_ids([3, 2], window=6)
    tokens = np.array([10, 11, spec.EOS, 20, spec.EOS, spec.PAD], dtype=np.int64)
    keep = masks.loss_mask(seg, tokens)
    assert keep.tolist() == [True, True, False, True, False, False]


def test_padding_never_earns_loss() -> None:
    """The source's warning: asked to predict padding 4,000 times the model does it effortlessly.

    The loss goes to zero and means nothing.
    """
    seg = masks.segment_ids([2], window=6)
    tokens = np.array([5, 6, spec.PAD, spec.PAD, spec.PAD, spec.PAD], dtype=np.int64)
    assert not masks.loss_mask(seg, tokens)[2:].any()


def test_a_pad_token_inside_a_document_still_earns_no_loss() -> None:
    """The case the segment check alone does NOT cover.

    `segments >= 0` excludes padding *positions*, so it looks like the token check is redundant —
    and a mutation removing it survived until this test existed. It is not redundant: a shard could
    carry a `PAD` id inside a real document, through a packing bug or a corrupt shard, and grading
    it would teach the model to predict padding mid-sentence.
    """
    seg = masks.segment_ids([5], window=5)
    tokens = np.array([10, spec.PAD, 12, spec.PAD, 14], dtype=np.int64)
    keep = masks.loss_mask(seg, tokens)

    assert keep.tolist() == [True, False, True, False, False], (
        "a PAD id inside a document was graded — the token check is load-bearing, not redundant"
    )


def test_context_spans_are_excluded_from_loss() -> None:
    """SFT and agentic data: the prompt is context, only the answer is graded.

    Loss is not taken on the question, only on the answer.
    """
    seg = masks.segment_ids([8], window=8)
    tokens = np.arange(8, dtype=np.int64)
    keep = masks.loss_mask(seg, tokens, context_spans=[(0, 4)])
    assert keep.tolist() == [False, False, False, False, True, True, True, False]


def test_utilization_reports_what_the_window_actually_holds() -> None:
    """Padding is visible as a number rather than inferred."""
    seg = masks.segment_ids([3, 3], window=8)
    assert masks.utilization(seg) == pytest.approx(6 / 8)


def test_loss_utilization_is_lower_than_packing_utilization() -> None:
    """**The number a raw throughput figure hides.**

    A loader can report high tokens-per-second while most of those tokens are padding or context.
    Loss utilisation is the honest one, and it is always the smaller of the two.
    """
    seg = masks.segment_ids([4, 4], window=10)
    tokens = np.arange(10, dtype=np.int64)
    tokens[8:] = spec.PAD
    keep = masks.loss_mask(seg, tokens, context_spans=[(0, 2)])

    assert masks.utilization(seg) == pytest.approx(8 / 10)
    assert masks.loss_utilization(keep) < masks.utilization(seg)


def test_a_single_document_filling_the_window_is_the_simple_case() -> None:
    """Plain pretraining: everything graded except the final token, and full utilisation."""
    seg = masks.segment_ids([8], window=8)
    tokens = np.arange(8, dtype=np.int64)
    assert masks.utilization(seg) == 1.0
    assert masks.loss_mask(seg, tokens).sum() == 7
    assert masks.attention_mask(seg).tolist() == np.tril(np.ones((8, 8), dtype=bool)).tolist()
