"""Forking a branch, and the lineage that makes it distinguishable from an unrelated run.

Two branches that share their first eighty steps and diverge after are the ordinary shape of any
real training programme. Without a recorded parent, the shared prefix looks like a coincidence —
two runs that happened to use the same shards — and the whole point of a fork is lost.
"""

from pathlib import Path

import pytest
from trainingdata import checkpoint, fork, ledger


def _checkpoint(directory: Path, branch: str, step: int, **extra) -> None:
    """Write a checkpoint sidecar.

    Args:
        directory: Where checkpoints live.
        branch: The branch.
        step: The step it follows.
        **extra: Fields to override.
    """
    identifier = checkpoint.checkpoint_id("r", branch, step)
    record = checkpoint.Checkpoint(
        v=checkpoint.VERSION,
        checkpoint_id=identifier,
        run_id="r",
        branch_id=branch,
        attempt=0,
        step=step,
        cut={0: step * 2},
        segments={0: 0},
        weight_digest="b2:" + "a" * 32,
        plan_digest="0123456789abcdef",
        config_fingerprint="a72bf6053187",
        **extra,
    )
    checkpoint.sidecar_path(directory, identifier).write_text(record.to_json())


def test_a_fork_starts_from_the_checkpoint_at_or_before_the_requested_step(tmp_path) -> None:
    """Checkpoints are periodic, so the requested step usually is not one."""
    for step in (3, 7, 11):
        _checkpoint(tmp_path, "main", step)
    plan = fork.plan_fork(tmp_path, "main", "fork-a", at_step=9)
    assert plan.at_step == 7
    assert plan.checkpoint_id == "ckpt-main-000007"
    assert plan.next_step == 8


def test_forking_from_a_point_that_was_never_saved_is_refused(tmp_path) -> None:
    """Silently starting from a different point is the failure this prevents.

    A fork "at step 9" that actually began at step 3 would produce a branch whose recorded lineage
    and whose weights disagree, and nothing downstream could tell.
    """
    _checkpoint(tmp_path, "main", 11)
    with pytest.raises(FileNotFoundError, match="never saved"):
        fork.plan_fork(tmp_path, "main", "fork-a", at_step=5)


def test_a_fork_may_not_share_its_parent_s_name(tmp_path) -> None:
    """Segment files are keyed by branch, so a same-named fork would write into its parent's
    ledger and the two could never be separated again."""
    _checkpoint(tmp_path, "main", 7)
    with pytest.raises(ValueError, match="named differently"):
        fork.plan_fork(tmp_path, "main", "main", at_step=7)


def test_the_lineage_is_recorded_on_the_child_s_checkpoint(tmp_path) -> None:
    """Recorded, so an audit can follow it — not inferred from a shared prefix."""
    _checkpoint(tmp_path, "main", 7)
    _checkpoint(tmp_path, "fork-a", 11, parent_branch_id="main", forked_at_step=7)
    assert fork.lineage(tmp_path, "fork-a") == ["fork-a", "main"]
    assert fork.lineage(tmp_path, "main") == ["main"]


def test_a_lineage_cycle_stops_rather_than_hanging(tmp_path) -> None:
    """`plan_fork` cannot create one, but a hand-edited sidecar can.

    An audit that hung on a malformed record would be worse than one that stopped.
    """
    _checkpoint(tmp_path, "a", 1, parent_branch_id="b", forked_at_step=0)
    _checkpoint(tmp_path, "b", 1, parent_branch_id="a", forked_at_step=0)
    assert fork.lineage(tmp_path, "a") == ["a", "b"]


def test_the_shared_prefix_is_computed_from_the_ledgers_not_from_the_claim(tmp_path) -> None:
    """**The fork record is the claim; this is the check.**

    A child that re-ran its parent's steps rather than inheriting them would produce a *shorter*
    shared prefix than it claims — the re-run events would carry different coordinates — and that
    is the difference between a fork and a second recording of the same run.
    """
    directory = tmp_path / "ledger"
    for branch, hashes in (("main", "abc"), ("fork-a", "abd")):
        writer = ledger.LedgerWriter(
            directory=directory, run_id="r", branch_id=branch, rank=0, segment=0
        )
        writer.open()
        for step, marker in enumerate(hashes):
            writer.append(
                attempt=0,
                global_step=step,
                accum=0,
                flat=step,
                checkpoint_id=None,
                samples=(ledger.PackedSample("s", 0, 8, "web", 7),),
                sequence_length=8,
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
                microbatch_hash=f"b2:{marker * 10}ab",
                loss_mask_hash="b2:" + "b" * 32,
                position_ids_hash="b2:" + "c" * 32,
                segment_ids_hash="b2:" + "d" * 32,
                tokenizer_sha256="sha256:" + "e" * 64,
                plan_digest="0123456789abcdef",
            )

    assert fork.common_prefix(directory, "main", "fork-a") == 2, (
        "the branches agree on their first two events and diverge on the third"
    )


def test_two_unrelated_branches_share_nothing(tmp_path) -> None:
    """The control. A prefix computed as "the shorter length" would report agreement between runs
    that have none."""
    assert fork.common_prefix(tmp_path / "empty", "main", "other") == 0
