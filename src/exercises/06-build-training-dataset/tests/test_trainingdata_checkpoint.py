"""Checkpoints and the cut — the torch-free half, which is the half that has to be auditable.

A checkpoint's sidecar is read by `verify.py`, which never imports the producer and never installs
torch. So everything here runs without it, and CI covers all of it.
"""

import json

import pytest
from trainingdata import checkpoint, ledger, resume


def _record(**overrides) -> checkpoint.Checkpoint:
    """A checkpoint's metadata, with sensible defaults.

    Args:
        **overrides: Fields to replace.

    Returns:
        The record.
    """
    base = {
        "v": checkpoint.VERSION,
        "checkpoint_id": "ckpt-main-000007",
        "run_id": "run-test",
        "branch_id": "main",
        "attempt": 0,
        "step": 7,
        "cut": {0: 24, 1: 24},
        "segments": {0: 0, 1: 0},
        "weight_digest": "b2:" + "a" * 32,
        "plan_digest": "0123456789abcdef",
        "config_fingerprint": "a72bf6053187",
        "environment": {"device": "cpu"},
    }
    return checkpoint.Checkpoint(**{**base, **overrides})


def _segment(directory, rank: int, count: int, *, segment: int = 0, torn: bool = False):
    """A ledger segment holding `count` chained events.

    Args:
        directory: Where segments live.
        rank: Which worker.
        count: How many events.
        segment: Segment number.
        torn: Append an interrupted final line.

    Returns:
        The path written.
    """
    writer = ledger.LedgerWriter(
        directory=directory, run_id="run-test", branch_id="main", rank=rank, segment=segment
    )
    writer.open()
    for i in range(count):
        writer.append(
            attempt=0,
            global_step=i,
            accum=0,
            flat=i,
            checkpoint_id=None,
            samples=(ledger.PackedSample("s", 0, 8, "web", 7),),
            tokens=8,
            loss_tokens=7,
            pad_tokens=0,
            pack_util=1.0,
            stage="main",
            lane_mix={"web": 8},
            attention_policy="block-diagonal-causal",
            position_policy="restart-per-document-continue-across-window",
            pack_policy="concat-and-chop",
            opus_decision_id=None,
            microbatch_hash="b2:" + f"{i:032x}",
            loss_mask_hash="b2:" + "b" * 32,
            position_ids_hash="b2:" + "c" * 32,
            segment_ids_hash="b2:" + "d" * 32,
            tokenizer_sha256="sha256:" + "e" * 64,
            plan_digest="0123456789abcdef",
        )
    if torn:
        with writer.path.open("a", encoding="utf-8") as handle:
            handle.write('{"v": 1, "seq": 99, "pre')
    return writer.path


# --- the sidecar ---------------------------------------------------------------------------------


def test_the_sidecar_round_trips() -> None:
    """It is the auditor's only view of the checkpoint, so nothing may be lost in serialisation."""
    original = _record()
    assert checkpoint.Checkpoint.from_json(original.to_json()) == original


def test_the_cut_comes_back_as_integers_not_strings() -> None:
    """**The JSON trap.**

    Object keys are always strings, so a cut written as `{0: 24}` reads back as `{"0": 24}` unless
    converted. `cut[rank]` would then `KeyError` at exactly the moment a run is trying to recover,
    and the obvious "fix" — falling back to a default — would truncate to zero and discard the run.
    """
    restored = checkpoint.Checkpoint.from_json(_record().to_json())
    assert all(isinstance(rank, int) for rank in restored.cut)
    assert all(isinstance(rank, int) for rank in restored.segments)
    assert restored.cut[0] == 24


def test_a_future_schema_version_is_refused() -> None:
    """Reading v2 as v1 maps fields onto the wrong names and cuts the ledger in the wrong place."""
    payload = json.loads(_record().to_json())
    payload["v"] = checkpoint.VERSION + 1
    with pytest.raises(ValueError, match="schema"):
        checkpoint.Checkpoint.from_json(json.dumps(payload))


def test_the_id_is_derived_from_the_run_and_step_never_from_a_clock() -> None:
    """Re-running a branch must produce the same ids, or two artifacts about the same checkpoint
    cannot be compared, only counted."""
    first = checkpoint.checkpoint_id("run-a", "main", 7)
    assert first == checkpoint.checkpoint_id("run-a", "main", 7)
    assert first == "ckpt-main-000007"
    assert checkpoint.checkpoint_id("run-a", "main", 12) > first, "ids must sort by step"


def test_total_microbatches_sums_the_cut() -> None:
    """The single number a report quotes; it must be derived, never carried separately."""
    assert _record(cut={0: 24, 1: 25, 2: 26}).total_microbatches == 75


# --- writing -------------------------------------------------------------------------------------


def test_a_reader_holding_the_old_file_never_sees_a_half_written_one(tmp_path) -> None:
    """**What "atomic" actually means here, and the only way it is observable.**

    `os.replace` swaps a *new* file into the name; a reader that already opened the old one keeps
    reading the old one, whole. Rewriting the target in place instead — copy the staging file over
    it and delete the staging — leaves the same bytes at the end and passes every other test in this
    file, while a reader mid-read sees a torn checkpoint and a crash mid-copy destroys the only
    file that could have recovered the run.
    """
    target = tmp_path / "ckpt-main-000007.json"
    target.write_text("the old checkpoint")

    with target.open("rb") as still_open:
        checkpoint.write_atomically(target, lambda staging: staging.write_text("the new one"))
        assert still_open.read() == b"the old checkpoint", (
            "the open handle saw the new contents — the target was rewritten in place, not replaced"
        )
    assert target.read_text() == "the new one"


def test_a_write_leaves_no_partial_file_behind(tmp_path) -> None:
    """A stray `.partial` would be picked up by a glob and treated as a checkpoint."""
    target = tmp_path / "ckpt-main-000007.json"
    checkpoint.write_atomically(target, lambda staging: staging.write_text("payload"))
    assert target.read_text() == "payload"
    assert list(tmp_path.glob("*.partial")) == []


def test_a_failed_write_does_not_destroy_the_previous_checkpoint(tmp_path) -> None:
    """**The reason rename-into-place matters here specifically.**

    Writing straight to the final path and dying halfway corrupts the one file that could have
    recovered the run. The old checkpoint must survive a failed save intact.
    """
    target = tmp_path / "ckpt-main-000007.json"
    target.write_text("the good one")

    def explode(staging):
        staging.write_text("half a checkpoint")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        checkpoint.write_atomically(target, explode)
    assert target.read_text() == "the good one"


def test_a_checkpoint_without_its_sidecar_is_not_found(tmp_path) -> None:
    """The commit protocol. Tensors are renamed into place first, the sidecar last.

    So an interrupted save leaves tensors with no sidecar, and that must read as *absent* rather
    than as a checkpoint to restore from a truncated tensor file.
    """
    (tmp_path / "ckpt-main-000007.pt").write_bytes(b"tensors, possibly truncated")
    with pytest.raises(FileNotFoundError, match="interrupted mid-save"):
        checkpoint.load(tmp_path, "ckpt-main-000007")


def test_the_latest_checkpoint_is_chosen_by_step_not_by_id(tmp_path) -> None:
    """**Where lexical order over ids stops agreeing with step order.**

    Ids pad to six digits, so below a million steps the two orders coincide and a `max` over ids
    looks correct. At a million they diverge: `"ckpt-main-1000000"` sorts *before*
    `"ckpt-main-999999"`, and a resume would restore the earlier checkpoint and silently redo a
    thousand steps. A million steps is not a hypothetical scale for a pre-training run.
    """
    for step in (999_998, 999_999, 1_000_000):
        identifier = f"ckpt-main-{step:06d}"
        checkpoint.sidecar_path(tmp_path, identifier).write_text(
            _record(step=step, checkpoint_id=identifier).to_json()
        )
    found = checkpoint.latest(tmp_path, "main")
    assert found is not None
    assert found.step == 1_000_000, f"picked step {found.step}; ids sort the wrong way up here"


def test_the_latest_checkpoint_is_the_furthest_step(tmp_path) -> None:
    """The ordinary case, which any correct implementation passes."""
    for step in (3, 11, 7):
        path = checkpoint.sidecar_path(tmp_path, checkpoint.checkpoint_id("r", "main", step))
        path.write_text(_record(step=step, checkpoint_id=f"ckpt-main-{step:06d}").to_json())
    found = checkpoint.latest(tmp_path, "main")
    assert found is not None
    assert found.step == 11


def test_the_latest_checkpoint_can_be_asked_for_a_point_in_the_past(tmp_path) -> None:
    """What fork needs: a branch starts from a checkpoint, not from wherever the run ended up."""
    for step in (3, 7, 11):
        path = checkpoint.sidecar_path(tmp_path, checkpoint.checkpoint_id("r", "main", step))
        path.write_text(_record(step=step, checkpoint_id=f"ckpt-main-{step:06d}").to_json())
    found = checkpoint.latest(tmp_path, "main", at_or_before=8)
    assert found is not None and found.step == 7


def test_a_branch_with_no_checkpoints_returns_none(tmp_path) -> None:
    """A first run has none, and that is not an error."""
    assert checkpoint.latest(tmp_path, "main") is None
    assert checkpoint.latest(tmp_path / "never-created", "main") is None


def test_another_branch_s_checkpoints_are_not_picked_up(tmp_path) -> None:
    """A fork writes beside its parent; restoring the wrong one silently swaps the run."""
    for branch in ("main", "fork-a"):
        path = checkpoint.sidecar_path(tmp_path, f"ckpt-{branch}-000007")
        path.write_text(_record(branch_id=branch, checkpoint_id=f"ckpt-{branch}-000007").to_json())
    found = checkpoint.latest(tmp_path, "fork-a")
    assert found is not None and found.branch_id == "fork-a"


# --- the cut -------------------------------------------------------------------------------------


def test_each_rank_is_cut_to_its_own_number(tmp_path) -> None:
    """**The claim the vector exists for, tested where it can actually be non-uniform.**

    The end-to-end drill produces a *uniform* cut, because a synchronous checkpoint lands every rank
    on the same event count — so the drill alone would pass against an implementation that read one
    number and applied it to all four. This constructs the non-uniform case directly.
    """
    directory = tmp_path / "ledger"
    for rank, count in enumerate((10, 12, 9, 15)):
        _segment(directory, rank, count)

    record = _record(cut={0: 8, 1: 8, 2: 5, 3: 12}, segments=dict.fromkeys(range(4), 0))
    plan = resume.apply_cut(directory, record)

    assert plan.dropped == {0: 2, 1: 4, 2: 4, 3: 3}
    assert plan.reexecuted_microbatches == 13
    for rank, keep in record.cut.items():
        path = directory / f"main.rank{rank}.seg0.jsonl"
        assert len(ledger.read_segment(path)) == keep
        ok, message = ledger.verify_chain(ledger.read_segment(path))
        assert ok, f"rank {rank}: {message}"


def test_a_dry_run_changes_nothing(tmp_path) -> None:
    """It is what lets a mismatch be refused before any file has been modified."""
    directory = tmp_path / "ledger"
    before = {rank: _segment(directory, rank, 10).read_bytes() for rank in range(2)}
    resume.plan_resume(directory, _record(cut={0: 4, 1: 6}), dry_run=True)
    for rank, contents in before.items():
        assert (directory / f"main.rank{rank}.seg0.jsonl").read_bytes() == contents


def test_a_mismatched_checkpoint_is_refused_before_anything_is_cut(tmp_path) -> None:
    """**Half-applied is worse than either the crash or a clean refusal.**

    Rank 0's cut is fine and rank 1's is impossible. Applying them in order would truncate rank 0
    and then fail — leaving a run whose ranks disagree about which checkpoint they belong to.
    """
    directory = tmp_path / "ledger"
    before = {rank: _segment(directory, rank, 10).read_bytes() for rank in range(2)}

    with pytest.raises(ValueError, match="not from the same run"):
        resume.apply_cut(directory, _record(cut={0: 4, 1: 50}))

    for rank, contents in before.items():
        assert (directory / f"main.rank{rank}.seg0.jsonl").read_bytes() == contents, (
            f"rank {rank} was truncated before the mismatch on rank 1 was noticed"
        )


def test_a_missing_segment_is_refused(tmp_path) -> None:
    """A rank whose file is gone cannot be cut, and pretending otherwise loses its whole history."""
    directory = tmp_path / "ledger"
    _segment(directory, 0, 10)
    with pytest.raises(ValueError, match="does not exist"):
        resume.apply_cut(directory, _record(cut={0: 4, 1: 4}))


def test_a_torn_tail_is_repaired_before_the_cut_is_measured(tmp_path) -> None:
    """A torn line is not an event.

    Counting it would place the cut one event too far — and, worse, report one fewer microbatch as
    re-executed than really was.
    """
    directory = tmp_path / "ledger"
    _segment(directory, 0, 10, torn=True)

    plan = resume.apply_cut(directory, _record(cut={0: 6}, segments={0: 0}))
    assert plan.torn == (0,)
    assert plan.dropped == {0: 4}, "the torn line was counted as an event"
    assert len(ledger.read_segment(directory / "main.rank0.seg0.jsonl")) == 6


def test_the_resumed_run_continues_from_the_step_after_the_checkpoint(tmp_path) -> None:
    """Off-by-one here either repeats a completed step or skips one entirely."""
    directory = tmp_path / "ledger"
    _segment(directory, 0, 10)
    plan = resume.apply_cut(directory, _record(step=7, attempt=0, cut={0: 6}, segments={0: 0}))
    assert plan.next_step == 8
    assert plan.next_attempt == 1
