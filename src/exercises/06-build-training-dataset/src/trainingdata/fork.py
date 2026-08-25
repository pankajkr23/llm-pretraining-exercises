"""Forking a branch from an earlier checkpoint — and making the lineage a fact.

**The problem.** "Restart from step 80 with a different mixture" is the most ordinary thing anyone
does to a training run, and it is the one that makes a folder of artifacts unreadable six weeks
later. Two branches share their first eighty steps and diverge after; without a record of that, a
forked run is indistinguishable from an unrelated one that happens to use the same shards.

**The strategy.** A fork is not a copy. The parent's ledger up to the fork point *is* the fork's
history — re-running those steps would produce a second recording of the same events, and the
"never recompute" rule exists precisely to stop that. So the child records **where it came from**
and starts writing at the step after, and anything reading the child follows the pointer back.

**What makes this checkable rather than asserted.** The child's first checkpoint names its parent
branch and the step it forked at. `lineage` walks that chain, and `common_prefix` says which events
the two branches genuinely share — computed from the ledgers, not from the intent.

Torch-free: a fork is a bookkeeping operation, and the training that follows it is somebody else's
problem.
"""

from dataclasses import dataclass
from pathlib import Path

from . import checkpoint as checkpoint_module
from . import ledger


@dataclass(frozen=True, slots=True)
class Fork:
    """A branch and where it came from.

    Attributes:
        branch_id: The new branch.
        parent_branch_id: The branch it forked from.
        at_step: The last step the two share. The child's first own step is `at_step + 1`.
        checkpoint_id: The parent checkpoint the child restores from.
        next_step: The first step the child executes.
        next_attempt: The attempt number the child records.
    """

    branch_id: str
    parent_branch_id: str
    at_step: int
    checkpoint_id: str
    next_step: int
    next_attempt: int


def plan_fork(checkpoint_dir: Path, parent_branch_id: str, branch_id: str, at_step: int) -> Fork:
    """Work out which parent checkpoint a fork starts from.

    Args:
        checkpoint_dir: Where checkpoints live.
        parent_branch_id: The branch being forked.
        branch_id: The new branch's name.
        at_step: Fork at or before this step.

    Returns:
        What the fork will do.

    Raises:
        ValueError: If the child would be named the same as its parent, which would make the two
            share a ledger and be unseparable afterwards.
        FileNotFoundError: If the parent has no checkpoint at or before that step. Forking from a
            point that was never saved would mean silently starting from a different one.
    """
    if branch_id == parent_branch_id:
        raise ValueError(
            f"a fork must be named differently from its parent; both are {branch_id!r}, so the two "
            f"would share segment files and could never be told apart"
        )

    record = checkpoint_module.latest(checkpoint_dir, parent_branch_id, at_or_before=at_step)
    if record is None:
        raise FileNotFoundError(
            f"{parent_branch_id} has no checkpoint at or before step {at_step}. Forking from a "
            f"point that was never saved would start from a different one without saying so."
        )
    return Fork(
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        at_step=record.step,
        checkpoint_id=record.checkpoint_id,
        next_step=record.step + 1,
        next_attempt=0,
    )


def common_prefix(ledger_dir: Path, first: str, second: str) -> int:
    """How many events two branches genuinely share, computed from their ledgers.

    Not from the fork record — that is the *claim*. This is the check: two branches that forked at
    step 80 must agree on every event up to it and diverge after, and a fork whose child re-ran the
    shared steps would show a shorter prefix than it claims.

    Args:
        ledger_dir: Where segment files live.
        first: One branch.
        second: The other.

    Returns:
        The number of leading events that are identical in both.
    """
    left = ledger.read_branch(ledger_dir, first)
    right = ledger.read_branch(ledger_dir, second)

    shared = 0
    for a, b in zip(left, right, strict=False):
        if (a.global_step, a.rank, a.accum, a.flat, a.microbatch_hash) != (
            b.global_step,
            b.rank,
            b.accum,
            b.flat,
            b.microbatch_hash,
        ):
            break
        shared += 1
    return shared


def lineage(checkpoint_dir: Path, branch_id: str) -> list[str]:
    """The chain of branches this one descends from, oldest last.

    Args:
        checkpoint_dir: Where checkpoints live.
        branch_id: The branch to trace.

    Returns:
        `[branch_id, parent, grandparent, ...]`. A branch with no parent returns just itself.
    """
    chain = [branch_id]
    seen = {branch_id}
    current = branch_id
    while True:
        record = checkpoint_module.latest(checkpoint_dir, current)
        parent = record.parent_branch_id if record else None
        if not parent or parent in seen:
            # A cycle cannot happen through `plan_fork`, which refuses a self-named fork — but a
            # hand-edited sidecar could produce one, and an audit that hung on it would be worse
            # than one that stopped.
            return chain
        chain.append(parent)
        seen.add(parent)
        current = parent
