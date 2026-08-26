"""Packing a span into a window — and the window-edge error that has no symptom.

Concat-and-chop cuts every `sequence_length` tokens without regard for documents, so most windows
open part-way through one. Numbering that leading fragment from 0 is the naive implementation, it
looks entirely correct, and it tells the model that the middle of a document is the start of one.
"""

import numpy as np
import pytest
from trainingdata import masks, pack, spec


def _stream(*lengths: int) -> np.ndarray:
    """A token stream of documents with the given lengths, each terminated by EOS.

    Token values are `100 + document index`, so which document a token came from is readable at a
    glance in a failure message.

    Args:
        *lengths: Total length of each document, EOS included.

    Returns:
        The concatenated stream.
    """
    parts = []
    for i, length in enumerate(lengths):
        body = np.full(length - 1, 100 + i, dtype=np.int64)
        parts.append(np.concatenate([body, [spec.EOS]]))
    return np.concatenate(parts)


# --- the document index ------------------------------------------------------------------------


def test_documents_are_found_from_the_eos_separators() -> None:
    """No side file: the boundaries are already in the data, so they cannot fall out of sync."""
    index = pack.DocIndex(_stream(4, 3, 5))
    assert index.count == 3
    assert [f.length for f in index.fragments(0, 12)] == [4, 3, 5]


def test_an_unterminated_tail_is_still_a_document() -> None:
    """The common case, not an exotic one.

    `split()` cuts shards at a fixed token count, never at a document boundary, so most shards end
    mid-document. Treating that as an error would refuse most shards.
    """
    tokens = np.concatenate([_stream(4), np.full(3, 200, dtype=np.int64)])
    index = pack.DocIndex(tokens)
    assert index.count == 2
    assert [f.length for f in index.fragments(0, 7)] == [4, 3]


def test_a_stream_with_no_eos_at_all_is_one_document() -> None:
    """Degenerate but reachable — a shard cut from the middle of a very long document."""
    index = pack.DocIndex(np.full(10, 42, dtype=np.int64))
    assert index.count == 1
    assert index.fragments(0, 10) == [
        pack.Fragment(doc_index=0, shard_start=0, shard_end=10, offset=0, complete=True)
    ]


@pytest.mark.parametrize(
    ("position", "expected"),
    [(0, 0), (3, 0), (4, 1), (6, 1), (7, 2), (11, 2)],
)
def test_a_position_maps_to_the_document_containing_it(position: int, expected: int) -> None:
    """Including both sides of every boundary — an off-by-one here mislabels a whole fragment."""
    assert pack.DocIndex(_stream(4, 3, 5)).document_at(position) == expected


def test_a_position_outside_the_shard_is_refused() -> None:
    """`searchsorted` would happily return an index past the end and mislabel silently."""
    index = pack.DocIndex(_stream(4))
    with pytest.raises(ValueError, match="outside a shard"):
        index.document_at(4)
    with pytest.raises(ValueError, match="outside a shard"):
        index.document_at(-1)


# --- fragments ---------------------------------------------------------------------------------


def test_fragments_cover_the_span_exactly_once() -> None:
    """The conservation law. A gap drops tokens the ledger records as consumed; an overlap
    double-counts them."""
    index = pack.DocIndex(_stream(4, 3, 5))
    fragments = index.fragments(2, 10)
    assert sum(f.length for f in fragments) == 8
    covered = [p for f in fragments for p in range(f.shard_start, f.shard_end)]
    assert covered == list(range(2, 10)), "fragments do not tile the span"


def test_a_fragment_opening_mid_document_carries_its_true_offset() -> None:
    """**The window-edge case.**

    The span starts at 2, two tokens into document 0. Offset 2 is what lets positions continue
    instead of restarting — and `complete=False` records that its document runs past the window.
    """
    (fragment,) = pack.DocIndex(_stream(8)).fragments(2, 6)
    assert fragment.offset == 2
    assert fragment.complete is False


def test_a_whole_document_inside_a_window_starts_at_zero_and_is_complete() -> None:
    """The other half of the same claim — otherwise a constant offset would pass the test above."""
    fragments = pack.DocIndex(_stream(4, 3, 5)).fragments(0, 7)
    assert [(f.offset, f.complete) for f in fragments] == [(0, True), (0, True)]


def test_a_span_reaching_past_the_shard_is_refused() -> None:
    """Numpy would return a short slice and the window would silently hold fewer tokens."""
    index = pack.DocIndex(_stream(4))
    with pytest.raises(ValueError, match="reaches past"):
        index.fragments(0, 9)


def test_an_empty_span_is_refused() -> None:
    """A zero-length window is not a window, and every count downstream would divide by zero."""
    with pytest.raises(ValueError, match="empty span"):
        pack.DocIndex(_stream(4)).fragments(3, 3)


# --- the window --------------------------------------------------------------------------------


def test_positions_continue_across_a_window_edge_rather_than_restarting() -> None:
    """**The claim this module exists for.**

    One 12-token document, chopped into three 4-token windows. The naive implementation numbers
    every window `0,1,2,3` — telling the model three times that it is reading the start of a
    document. Nothing crashes and no loss curve looks wrong.
    """
    tokens = _stream(12)
    index = pack.DocIndex(tokens)
    seen = [pack.build_window(index, tokens, s, s + 4).positions.tolist() for s in (0, 4, 8)]

    assert seen == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], (
        "positions restarted at a window edge — the model is being told the middle of a document "
        "is the start of one"
    )


def test_positions_restart_at_a_real_document_boundary() -> None:
    """The twin. Continuing *through* a boundary would be the opposite error, equally silent."""
    tokens = _stream(3, 5)
    window = pack.build_window(pack.DocIndex(tokens), tokens, 0, 8)
    assert window.positions.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


def test_a_window_spanning_a_boundary_does_both_at_once() -> None:
    """Offsets are per-segment, not one value for the window.

    Segment 0 is the tail of document 0 and continues from 2; segment 1 begins document 1 at 0.
    """
    tokens = _stream(4, 4)
    window = pack.build_window(pack.DocIndex(tokens), tokens, 2, 8)
    assert [(f.offset, f.complete) for f in window.fragments] == [(2, True), (0, True)]
    assert window.positions.tolist() == [2, 3, 0, 1, 2, 3]


def test_documents_in_a_packed_window_still_cannot_see_each_other() -> None:
    """The boundary guarantee has to survive packing, or packing removed it."""
    tokens = _stream(3, 5)
    window = pack.build_window(pack.DocIndex(tokens), tokens, 0, 8)
    allow = masks.attention_mask(window.segments)
    for i in range(8):
        for j in range(8):
            if allow[i, j]:
                assert window.segments[i] == window.segments[j]


def test_a_short_span_is_padded_and_the_padding_is_accounted_for() -> None:
    """Padding must be visible as a number, never inferred from a gap in the counts."""
    tokens = _stream(4)
    window = pack.build_window(pack.DocIndex(tokens), tokens, 0, 4, window=8)
    assert window.tokens[4:].tolist() == [spec.PAD] * 4
    assert window.segments[4:].tolist() == [-1] * 4
    assert window.pad_tokens == 4
    assert window.pack_utilization == pytest.approx(0.5)
    assert not window.loss[4:].any(), "padding earned loss"


def test_a_span_larger_than_the_window_is_refused() -> None:
    """Truncating would drop tokens the plan says were consumed, and nothing would say so."""
    tokens = _stream(16)
    with pytest.raises(ValueError, match="does not fit"):
        pack.build_window(pack.DocIndex(tokens), tokens, 0, 16, window=8)


def test_loss_tokens_are_fewer_than_real_tokens() -> None:
    """Each document's last token has no target, so the two counts can never be equal."""
    tokens = _stream(4, 4)
    window = pack.build_window(pack.DocIndex(tokens), tokens, 0, 8)
    assert window.pad_tokens == 0
    assert window.loss_tokens == 6, "one ungraded final token per document, not more or fewer"


# --- the hashes replay re-derives ---------------------------------------------------------------


def test_the_four_hashes_are_recorded_separately() -> None:
    """One digest of everything would say 'replay disagrees' without saying about what.

    A mask bug and a token bug need different fixes, so they get different hashes.
    """
    tokens = _stream(4, 4)
    hashes = pack.build_window(pack.DocIndex(tokens), tokens, 0, 8).hashes()
    assert set(hashes) == {
        "microbatch_hash",
        "loss_mask_hash",
        "position_ids_hash",
        "segment_ids_hash",
    }
    assert len(set(hashes.values())) == 4, "two of the four arrays hashed identically"


def test_the_same_window_hashes_the_same_way_twice() -> None:
    """Replay compares a recomputed hash against a recorded one; an unstable hash makes that
    comparison meaningless in the direction that matters — it would fail on correct data."""
    tokens = _stream(6, 6)
    index = pack.DocIndex(tokens)
    first = pack.build_window(index, tokens, 0, 12).hashes()
    second = pack.build_window(pack.DocIndex(tokens), tokens, 0, 12).hashes()
    assert first == second


def test_one_changed_token_changes_the_window_hash() -> None:
    """The smallest edit must be visible, or the hash protects nothing."""
    tokens = _stream(8)
    before = pack.build_window(pack.DocIndex(tokens), tokens, 0, 8).hashes()["microbatch_hash"]
    tampered = tokens.copy()
    tampered[3] += 1
    after = pack.build_window(pack.DocIndex(tampered), tampered, 0, 8).hashes()["microbatch_hash"]
    assert before != after


def test_the_same_numbers_in_a_different_dtype_hash_differently() -> None:
    """`int32` and `int64` holding the same integers are not the same tensor.

    A hash over raw bytes alone would call a `(2,4)` array equal to a `(4,2)` one as well, so shape
    goes into the digest too.
    """
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    assert pack.hash_array(np.array(values, dtype=np.int32)) != pack.hash_array(
        np.array(values, dtype=np.int64)
    )
    flat = np.array(values, dtype=np.int64)
    assert pack.hash_array(flat.reshape(2, 4)) != pack.hash_array(flat.reshape(4, 2))


def test_a_fortran_ordered_array_hashes_like_its_contents() -> None:
    """A transposed microbatch is Fortran-ordered, and this is where the byte order actually bites.

    A 1-D strided view is neither C- nor F-contiguous, so `order="A"` falls back to C and looks
    correct; only a 2-D F-ordered array separates the two. Written after `order="A"` survived the
    strided-view test below — that test alone did not make the argument load-bearing.
    """
    values = np.arange(6, dtype=np.int64).reshape(2, 3)
    assert pack.hash_array(np.asfortranarray(values)) == pack.hash_array(values)


def test_a_non_contiguous_view_hashes_like_its_contents() -> None:
    """A sliced array is a view with strides.

    `tobytes()` on one would otherwise depend on how it was produced rather than on what it holds.
    """
    base = np.arange(16, dtype=np.int64)
    assert pack.hash_array(base[::2]) == pack.hash_array(np.arange(0, 16, 2, dtype=np.int64))


# --- the offsets argument itself ----------------------------------------------------------------


def test_offsets_must_have_one_entry_per_segment() -> None:
    """Zero-filling a short list would restart exactly the fragments the argument exists to fix."""
    segments = masks.segment_ids([3, 3], window=6)
    with pytest.raises(ValueError, match="one per segment"):
        masks.position_ids(segments, offsets=[5])


def test_padding_does_not_count_as_a_segment_needing_an_offset() -> None:
    """Off-by-one bait: two documents plus padding is two offsets, not three."""
    segments = masks.segment_ids([2, 2], window=6)
    assert masks.position_ids(segments, offsets=[7, 0]).tolist() == [7, 8, 0, 1, 0, 0]


# --- context spans -----------------------------------------------------------------------------


def test_context_spans_are_clipped_to_the_span_and_translated_to_the_window() -> None:
    """The caller knows about shards; `masks.loss_mask` knows only about the window in front of it.

    A shard-relative range handed straight through would mask the wrong positions — and on a window
    taken from the middle of a shard, it would usually mask nothing at all and look like it worked.
    """
    tokens = _stream(200)
    index = pack.DocIndex(tokens)
    # A span covering shard tokens 64..128; the context range 100..150 overlaps its second half.
    window = pack.build_window(index, tokens, 64, 128, context_spans=((100, 150),))
    assert window.context_spans == ((36, 64),), "the range was not translated to window coordinates"
    assert not window.loss[36:64].any(), "the context range still earned loss"
    assert window.loss[:36].any(), "masking spilled outside the context range"


def test_a_context_span_outside_the_window_is_ignored() -> None:
    """Every shard's spans are handed to every window cut from it, so most do not apply."""
    tokens = _stream(200)
    window = pack.build_window(pack.DocIndex(tokens), tokens, 0, 64, context_spans=((150, 180),))
    assert window.context_spans == ()
    assert window.loss.sum() > 0


def test_a_window_with_no_context_spans_grades_everything_it_did_before() -> None:
    """The control: passing an empty tuple must not change the mask at all."""
    tokens = _stream(8, 8)
    index = pack.DocIndex(tokens)
    plain = pack.build_window(index, tokens, 0, 16)
    explicit = pack.build_window(index, tokens, 0, 16, context_spans=())
    assert np.array_equal(plain.loss, explicit.loss)
    assert plain.hashes() == explicit.hashes()
