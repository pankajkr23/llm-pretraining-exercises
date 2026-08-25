"""The consumption ledger — the artifact every later claim is read out of.

If the ledger is wrong, replay is wrong, the audit is wrong, and both agree with each other because
they read the same wrong file. So the tests here are about *tamper evidence* and *crash survival*
rather than about round-tripping a dataclass.
"""

import dataclasses
import json

import pytest
from trainingdata import ledger


def _fields(**overrides) -> dict:
    """A complete set of `append` arguments, so each test overrides only what it is about.

    Args:
        **overrides: Fields to replace.

    Returns:
        Keyword arguments for `LedgerWriter.append`.
    """
    base = {
        "attempt": 0,
        "global_step": 0,
        "accum": 0,
        "flat": 0,
        "checkpoint_id": None,
        "samples": (ledger.PackedSample("shardA", 0, 512, "web", 511),),
        "tokens": 512,
        "loss_tokens": 511,
        "pad_tokens": 0,
        "pack_util": 1.0,
        "stage": "warmup",
        "lane_mix": {"web": 512},
        "attention_policy": "block-diagonal-causal",
        "position_policy": "restart-per-document",
        "pack_policy": "concat-and-chop",
        "opus_decision_id": None,
        "microbatch_hash": "b2:" + "a" * 32,
        "loss_mask_hash": "b2:" + "b" * 32,
        "position_ids_hash": "b2:" + "c" * 32,
        "segment_ids_hash": "b2:" + "d" * 32,
        "tokenizer_sha256": "sha256:" + "e" * 64,
        "plan_digest": "0123456789abcdef",
    }
    return {**base, **overrides}


def _writer(tmp_path, *, rank: int = 0, segment: int = 0) -> ledger.LedgerWriter:
    """An opened writer over a temporary directory.

    Args:
        tmp_path: pytest's temporary directory.
        rank: Worker id.
        segment: Segment number.

    Returns:
        The writer, with its segment file already claimed.
    """
    writer = ledger.LedgerWriter(
        directory=tmp_path / "ledger",
        run_id="run-test",
        branch_id="main",
        rank=rank,
        segment=segment,
    )
    writer.open()
    return writer


# --- the chain ---------------------------------------------------------------------------------


def test_the_first_event_starts_the_chain_at_genesis(tmp_path) -> None:
    """A chain needs a defined start, or the first line's `prev` is unverifiable by construction."""
    writer = _writer(tmp_path)
    event = writer.append(**_fields())
    assert event.prev == ledger.GENESIS


def test_each_event_carries_the_previous_one_s_hash(tmp_path) -> None:
    """This is the whole mechanism: the file is a chain, not a pile of independent lines."""
    writer = _writer(tmp_path)
    first = writer.append(**_fields(flat=0))
    second = writer.append(**_fields(flat=1))
    assert second.prev == first.chain_hash()
    assert second.seq == first.seq + 1


def test_a_clean_file_verifies(tmp_path) -> None:
    """The control. Without it, a `verify_chain` that returned False for everything would 'pass'."""
    writer = _writer(tmp_path)
    for i in range(5):
        writer.append(**_fields(flat=i, global_step=i))
    ok, message = ledger.verify_chain(ledger.read_segment(writer.path))
    assert ok, message
    assert "5 events" in message


def test_altering_one_field_breaks_the_chain_from_there_on(tmp_path) -> None:
    """**The claim the ledger exists to make.**

    An append-only file you can edit is not append-only. Here the edit is the one someone would
    actually make — changing a recorded count to match a number they wish they had reported.
    """
    writer = _writer(tmp_path)
    for i in range(5):
        writer.append(**_fields(flat=i, global_step=i))

    lines = writer.path.read_text().splitlines()
    doctored = json.loads(lines[2])
    doctored["loss_tokens"] = 999_999
    lines[2] = json.dumps(doctored, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    writer.path.write_text("\n".join(lines) + "\n")

    ok, message = ledger.verify_chain(ledger.read_segment(writer.path))
    assert not ok, "a doctored loss count verified clean — the ledger proves nothing"
    assert "event 3" in message, (
        f"the break should surface at the FIRST line whose prev no longer matches, which is the "
        f"one after the edit; got: {message}"
    )


def test_deleting_a_line_breaks_the_chain(tmp_path) -> None:
    """Removing an inconvenient event is the other half of tampering, and it must be as loud."""
    writer = _writer(tmp_path)
    for i in range(4):
        writer.append(**_fields(flat=i, global_step=i))

    lines = writer.path.read_text().splitlines()
    del lines[1]
    writer.path.write_text("\n".join(lines) + "\n")

    ok, message = ledger.verify_chain(ledger.read_segment(writer.path))
    assert not ok, "a deleted event went unnoticed"
    assert "seq" in message or "prev" in message


def test_reordering_two_lines_breaks_the_chain(tmp_path) -> None:
    """Order is part of the record: 'what was consumed when' is the question being answered."""
    writer = _writer(tmp_path)
    for i in range(4):
        writer.append(**_fields(flat=i, global_step=i))

    lines = writer.path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    writer.path.write_text("\n".join(lines) + "\n")

    ok, _ = ledger.verify_chain(ledger.read_segment(writer.path))
    assert not ok, "two events swapped places and the chain still verified"


def test_appending_a_forged_event_breaks_the_chain(tmp_path) -> None:
    """A forger who does not know the chain cannot extend it.

    They *could* recompute every hash forward — the chain is not a signature. What it buys is that
    tampering can never be local, which is the honest claim and the one stated in the docs.
    """
    writer = _writer(tmp_path)
    writer.append(**_fields())

    forged = json.loads(writer.path.read_text().splitlines()[0])
    forged["seq"] = 1
    forged["flat"] = 77
    with writer.path.open("a") as handle:
        handle.write(json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n")

    ok, _ = ledger.verify_chain(ledger.read_segment(writer.path))
    assert not ok


def test_an_empty_ledger_verifies_vacuously(tmp_path) -> None:
    """A run that crashed before its first event has an empty ledger, not a broken one."""
    ok, _ = ledger.verify_chain([])
    assert ok


# --- serialisation -----------------------------------------------------------------------------


def test_every_field_survives_the_round_trip(tmp_path) -> None:
    """A field that does not round-trip is a field replay cannot use, silently."""
    writer = _writer(tmp_path)
    written = writer.append(
        **_fields(
            samples=(
                ledger.PackedSample("shardA", 0, 300, "web", 299, pass_no=1),
                ledger.PackedSample("shardB", 512, 724, "code", 211, pass_no=2),
            ),
            checkpoint_id="ckpt-000040",
            opus_decision_id="dec-17",
            replayed_from=3,
        )
    )
    (back,) = ledger.read_segment(writer.path)
    assert back == written


def test_the_pass_number_survives_so_a_re_read_is_visible(tmp_path) -> None:
    """Without it a second epoch is invisible and the repeated-pass effect cannot be measured."""
    writer = _writer(tmp_path)
    writer.append(**_fields(samples=(ledger.PackedSample("s", 0, 8, "web", 7, pass_no=3),)))
    (back,) = ledger.read_segment(writer.path)
    assert back.samples[0].pass_no == 3
    assert isinstance(back.samples[0], ledger.PackedSample)


def test_a_future_schema_version_is_refused_rather_than_misread(tmp_path) -> None:
    """Reading v2 as v1 would map fields onto the wrong names and produce confident nonsense."""
    writer = _writer(tmp_path)
    writer.append(**_fields())
    payload = json.loads(writer.path.read_text().splitlines()[0])
    payload["v"] = ledger.VERSION + 1
    writer.path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ledger.LedgerCorruptionError, match="schema"):
        ledger.read_segment(writer.path)


def test_the_canonical_form_does_not_depend_on_key_order() -> None:
    """The digest must depend on the values, not on how the dict happened to be built.

    Otherwise the same event hashes two ways and the chain breaks for no reason.
    """
    event = ledger.ConsumeEvent(
        v=ledger.VERSION,
        seq=0,
        prev=ledger.GENESIS,
        run_id="r",
        branch_id="main",
        segment=0,
        rank=0,
        **_fields(),
    )
    reparsed = ledger.ConsumeEvent.from_json(json.loads(event.canonical()))
    assert reparsed.canonical() == event.canonical()
    assert reparsed.chain_hash() == event.chain_hash()


def test_two_events_differing_in_one_field_hash_differently() -> None:
    """A hash that collapsed a field would let that field be edited freely."""
    common = {
        "v": ledger.VERSION,
        "seq": 0,
        "prev": ledger.GENESIS,
        "run_id": "r",
        "branch_id": "main",
        "segment": 0,
        "rank": 0,
    }
    a = ledger.ConsumeEvent(**common, **_fields(pad_tokens=0))
    b = ledger.ConsumeEvent(**common, **_fields(pad_tokens=1))
    assert a.chain_hash() != b.chain_hash()


# --- one file per (branch, rank, segment) ------------------------------------------------------


def test_a_segment_cannot_be_claimed_twice(tmp_path) -> None:
    """`O_EXCL`. Two processes appending to one file interleave into an unreadable record."""
    _writer(tmp_path, rank=0, segment=0)
    duplicate = ledger.LedgerWriter(
        directory=tmp_path / "ledger", run_id="run-test", branch_id="main", rank=0, segment=0
    )
    with pytest.raises(FileExistsError):
        duplicate.open()


def test_appending_without_opening_is_refused(tmp_path) -> None:
    """The hole `O_EXCL` alone leaves.

    `append` opens in `"a"` mode, which creates the file — so a writer that never called `open()`
    would silently bypass the exclusive claim and could interleave with the process that holds it.
    """
    writer = ledger.LedgerWriter(
        directory=tmp_path / "ledger", run_id="run-test", branch_id="main", rank=0, segment=0
    )
    with pytest.raises(ledger.LedgerCorruptionError, match="O_EXCL"):
        writer.append(**_fields())
    assert not writer.path.exists(), "the refused append created the file anyway"


def test_each_rank_writes_its_own_file(tmp_path) -> None:
    """No shared writer means no lock, and no lock means none held when a process dies."""
    writers = [_writer(tmp_path, rank=r) for r in range(4)]
    for r, writer in enumerate(writers):
        writer.append(**_fields(flat=r))
    paths = {w.path for w in writers}
    assert len(paths) == 4
    for writer in writers:
        assert len(ledger.read_segment(writer.path)) == 1


def test_segments_are_sorted_numerically_not_lexically(tmp_path) -> None:
    """`rank10` sorts before `rank2` as text.

    A lexical sort would silently reorder the audit's view of a run once a cluster passes ten ranks
    — and would look perfectly correct on the four-rank run we test on.
    """
    directory = tmp_path / "ledger"
    for rank in (0, 2, 10):
        for segment in (0, 3, 11):
            writer = ledger.LedgerWriter(
                directory=directory, run_id="r", branch_id="main", rank=rank, segment=segment
            )
            writer.open()

    found = [p.name for p in ledger.segments_for(directory, "main")]
    assert found == [
        "main.rank0.seg0.jsonl",
        "main.rank0.seg3.jsonl",
        "main.rank0.seg11.jsonl",
        "main.rank2.seg0.jsonl",
        "main.rank2.seg3.jsonl",
        "main.rank2.seg11.jsonl",
        "main.rank10.seg0.jsonl",
        "main.rank10.seg3.jsonl",
        "main.rank10.seg11.jsonl",
    ]


def test_segments_of_another_branch_are_not_picked_up(tmp_path) -> None:
    """A fork writes beside its parent. Mixing them would attribute one run's data to another."""
    directory = tmp_path / "ledger"
    for branch in ("main", "fork-a"):
        ledger.LedgerWriter(
            directory=directory, run_id="r", branch_id=branch, rank=0, segment=0
        ).open()
    assert [p.name for p in ledger.segments_for(directory, "main")] == ["main.rank0.seg0.jsonl"]


def test_reading_a_branch_orders_events_as_the_run_consumed_them(tmp_path) -> None:
    """Files are per-rank; the run interleaved them. An audit reads run order, not file order."""
    for rank in range(3):
        writer = _writer(tmp_path, rank=rank)
        for step in range(2):
            writer.append(**_fields(global_step=step, accum=0, flat=step * 3 + rank))

    events = ledger.read_branch(tmp_path / "ledger", "main")
    assert [(e.global_step, e.rank) for e in events] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def test_next_segment_never_reuses_a_number(tmp_path) -> None:
    """A resumed process must not reopen the file its predecessor was writing."""
    directory = tmp_path / "ledger"
    directory.mkdir()
    assert ledger.next_segment(directory, "main", 0) == 0
    ledger.LedgerWriter(directory=directory, run_id="r", branch_id="main", rank=0, segment=0).open()
    assert ledger.next_segment(directory, "main", 0) == 1
    assert ledger.next_segment(directory, "main", 1) == 0, "ranks number their segments separately"


def test_reading_a_missing_segment_is_empty_not_an_error(tmp_path) -> None:
    """A rank that crashed before writing has no file, and that is a legitimate state."""
    assert ledger.read_segment(tmp_path / "absent.jsonl") == []
    assert ledger.segments_for(tmp_path / "nothing-here", "main") == []


# --- crash survival ----------------------------------------------------------------------------


def test_an_event_is_on_disk_before_append_returns(tmp_path) -> None:
    """Buffered in the process, an event is lost by exactly the crash it exists to survive."""
    writer = _writer(tmp_path)
    writer.append(**_fields())
    assert writer.path.read_text().count("\n") == 1


def test_a_torn_final_line_is_dropped_and_the_rest_still_verifies(tmp_path) -> None:
    """**The crash case.**

    `append` fsyncs, so a completed event has landed — but the kill can arrive mid-`write`, and a
    write that large is not atomic. The tail is then a prefix of an event rather than an event.
    """
    writer = _writer(tmp_path)
    for i in range(3):
        writer.append(**_fields(flat=i, global_step=i))
    whole = json.dumps({"v": 1, "seq": 3, "prev": "b2:aaa", "run_id": "run-test"})
    with writer.path.open("a") as handle:
        handle.write(whole[: len(whole) // 2])  # the kill landed mid-write

    with pytest.raises(ledger.LedgerCorruptionError):
        ledger.read_segment(writer.path)

    assert ledger.drop_torn_tail(writer.path) is True
    events = ledger.read_segment(writer.path)
    assert len(events) == 3
    ok, message = ledger.verify_chain(events)
    assert ok, message


def test_dropping_a_torn_tail_is_a_no_op_on_a_clean_file(tmp_path) -> None:
    """Repair must never remove a real event. This is the mutation that would lose data quietly."""
    writer = _writer(tmp_path)
    for i in range(3):
        writer.append(**_fields(flat=i))
    before = writer.path.read_bytes()
    assert ledger.drop_torn_tail(writer.path) is False
    assert writer.path.read_bytes() == before


def test_a_complete_last_line_without_a_newline_is_kept(tmp_path) -> None:
    """The boundary case that separates 'interrupted' from 'merely unterminated'.

    A JSON object ends in `}`; truncation always removes it, so a torn line never parses. A line
    that parses is complete data and dropping it would discard a consumed batch.
    """
    writer = _writer(tmp_path)
    writer.append(**_fields())
    writer.path.write_text(writer.path.read_text().rstrip("\n"))
    assert ledger.drop_torn_tail(writer.path) is False
    assert len(ledger.read_segment(writer.path)) == 1


def test_damage_in_the_middle_is_raised_not_repaired(tmp_path) -> None:
    """An unparseable line that is not last cannot be an interrupted write.

    Silently repairing it would hide real corruption behind a routine crash-recovery path.
    """
    writer = _writer(tmp_path)
    for i in range(3):
        writer.append(**_fields(flat=i))
    lines = writer.path.read_text().splitlines()
    lines[1] = "{not json at all"
    writer.path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ledger.LedgerCorruptionError, match="corruption"):
        ledger.drop_torn_tail(writer.path)


# --- the cut -----------------------------------------------------------------------------------


def test_truncating_discards_only_the_tail_and_leaves_a_valid_chain(tmp_path) -> None:
    """What resume does with the checkpoint's recorded cut.

    Events after the cut belong to work the checkpoint does not contain; replaying them would
    double-count, and keeping them would make the ledger disagree with the weights.
    """
    writer = _writer(tmp_path)
    for i in range(6):
        writer.append(**_fields(flat=i, global_step=i))

    assert ledger.truncate_to(writer.path, 4) == 2
    events = ledger.read_segment(writer.path)
    assert [e.flat for e in events] == [0, 1, 2, 3]
    ok, message = ledger.verify_chain(events)
    assert ok, message


def test_truncating_to_zero_empties_the_segment(tmp_path) -> None:
    """A rank that had written nothing at checkpoint time cuts to zero, not to one."""
    writer = _writer(tmp_path)
    writer.append(**_fields())
    assert ledger.truncate_to(writer.path, 0) == 1
    assert ledger.read_segment(writer.path) == []
    assert writer.path.read_text() == ""


def test_truncating_past_the_end_is_refused(tmp_path) -> None:
    """The checkpoint and this ledger would not belong to the same run.

    Silently keeping everything would let a mismatched pair resume and look healthy.
    """
    writer = _writer(tmp_path)
    writer.append(**_fields())
    with pytest.raises(ValueError, match="do not belong to the same run"):
        ledger.truncate_to(writer.path, 5)


def test_truncating_to_a_negative_count_is_refused(tmp_path) -> None:
    """`events[:-1]` would drop the last event instead of raising — an off-by-everything."""
    writer = _writer(tmp_path)
    writer.append(**_fields())
    with pytest.raises(ValueError, match="non-negative"):
        ledger.truncate_to(writer.path, -1)
    assert len(ledger.read_segment(writer.path)) == 1


def test_the_writer_reports_the_length_a_checkpoint_records(tmp_path) -> None:
    """The cut is a vector over ranks. A scalar cannot cut R files that stopped at R points."""
    writer = _writer(tmp_path)
    assert writer.length == 0
    for i in range(3):
        writer.append(**_fields(flat=i))
    assert writer.length == 3
    assert writer.length == len(ledger.read_segment(writer.path))


def test_a_re_chained_file_with_a_gap_in_seq_is_caught(tmp_path) -> None:
    """The tamperer the `prev` check alone does NOT stop.

    The chain is not a signature: anyone who can edit the file can also recompute every hash after
    their edit, and then every `prev` matches again. What they must *also* do is renumber, and the
    sequence numbers are what expose them when they do not — this is the attack that removes an
    inconvenient event and repairs the links behind it.

    Written after a mutation removing the `seq` check survived the whole file: it is not redundant
    with the `prev` check, it covers the case the `prev` check cannot.
    """
    writer = _writer(tmp_path)
    for i in range(4):
        writer.append(**_fields(flat=i, global_step=i))

    kept = [e for e in ledger.read_segment(writer.path) if e.seq != 1]  # remove one event...
    lines, prev = [], ledger.GENESIS
    for event in kept:  # ...and repair every link behind it, leaving seq at 0, 2, 3
        relinked = dataclasses.replace(event, prev=prev)
        lines.append(relinked.canonical())
        prev = relinked.chain_hash()
    writer.path.write_text("".join(line + "\n" for line in lines))

    events = ledger.read_segment(writer.path)
    assert [e.prev for e in events][0] == ledger.GENESIS
    ok, message = ledger.verify_chain(events)
    assert not ok, "a re-chained ledger with an event removed verified clean"
    assert "out of order" in message, message
