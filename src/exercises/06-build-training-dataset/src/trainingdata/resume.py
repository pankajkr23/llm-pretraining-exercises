"""Bringing a ledger back into agreement with a checkpoint, after a crash.

**The problem.** The weights stop at the last completed checkpoint. The ledger does not — it kept
recording right up to the moment the process died, because an event is written when the model is
*fed*, not when an update completes. So after a crash the two disagree, and the disagreement is
exactly the set of microbatches the model saw but whose learning the checkpoint does not contain.

**The strategy — cut the ledger back to the checkpoint, and publish what was cut.** Each rank's
segment is truncated to that rank's own entry in the cut vector. What remains is a ledger that
describes precisely the run these weights came from.

**What is honest to claim, and what is not.** "No skipped or repeated batches" is true of the
**effective post-cut ledger**. It is not true of the **device**: those microbatches really were
computed once before the crash and are computed again after it. Both things are true at once, and
the second is published rather than quietly dropped — the re-executed events carry `replayed_from`,
naming the discarded event each one repeats.

**The torn tail is dropped before the cut, never after.** A cut is expressed in events, and a torn
line is not an event; counting it would put the cut one place too far. Repair first, then cut.

Torch-free, so the auditor can check the arithmetic without the producer's dependencies.
"""

from dataclasses import dataclass
from pathlib import Path

from . import checkpoint as checkpoint_module
from . import ledger


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """What a resume is about to do, computed before it does any of it.

    Attributes:
        record: The checkpoint being resumed from.
        dropped: Rank to the number of events discarded by the cut.
        torn: Ranks whose segment ended in an interrupted write.
        next_step: The first step the resumed run will execute.
        next_attempt: The attempt number the resumed run will record.
    """

    record: checkpoint_module.Checkpoint
    dropped: dict[int, int]
    torn: tuple[int, ...]
    next_step: int
    next_attempt: int

    @property
    def reexecuted_microbatches(self) -> int:
        """Microbatches the device will compute a second time.

        The number that makes the resume claim honest. A run that reports "no repeated batches"
        without this is reporting a property of its bookkeeping, not of its compute.

        Returns:
            The total across every rank.
        """
        return sum(self.dropped.values())


def plan_resume(
    ledger_dir: Path, record: checkpoint_module.Checkpoint, *, dry_run: bool = True
) -> ResumePlan:
    """Work out — and optionally perform — the cut a checkpoint implies.

    Args:
        ledger_dir: Where segment files live.
        record: The checkpoint to resume from.
        dry_run: When True, nothing is written; the counts are computed and returned. Resume calls
            it once this way first, so a mismatched checkpoint is refused **before** any ledger is
            modified rather than after two of four have been truncated.

    Returns:
        What the resume will do.

    Raises:
        ValueError: If a rank's segment holds fewer events than its cut records — which means this
            checkpoint and this ledger are not from the same run, and applying the cut would be
            silently wrong in both directions.
    """
    dropped: dict[int, int] = {}
    torn: list[int] = []

    for rank, keep in sorted(record.cut.items()):
        path = ledger_dir / f"{record.branch_id}.rank{rank}.seg{record.segments[rank]}.jsonl"
        if not path.is_file():
            raise ValueError(
                f"the checkpoint records a cut of {keep} events for rank {rank}, but "
                f"{path.name} does not exist"
            )
        # One scan answers both questions, so the dry run and the repair cannot disagree about
        # what the file holds. A torn line is not an event; counting it would place the cut one
        # event too far.
        present, torn_here = ledger.scan_segment(path)
        if torn_here:
            torn.append(rank)
        if keep > present:
            raise ValueError(
                f"rank {rank}: the checkpoint cuts at {keep} events but the segment holds "
                f"{present} — this checkpoint and this ledger are not from the same run"
            )
        dropped[rank] = present - keep
        if not dry_run:
            ledger.drop_torn_tail(path)
            ledger.truncate_to(path, keep)

    return ResumePlan(
        record=record,
        dropped=dropped,
        torn=tuple(torn),
        next_step=record.step + 1,
        next_attempt=record.attempt + 1,
    )


def apply_cut(ledger_dir: Path, record: checkpoint_module.Checkpoint) -> ResumePlan:
    """Truncate every rank's ledger to the checkpoint's cut.

    Checked first, then applied. A checkpoint that disagrees with the ledger is refused before a
    single file is modified — otherwise a run could be left with two ranks cut and two not, which
    is a worse state than either the crash or a clean refusal.

    Args:
        ledger_dir: Where segment files live.
        record: The checkpoint to resume from.

    Returns:
        What was done.
    """
    plan_resume(ledger_dir, record, dry_run=True)
    return plan_resume(ledger_dir, record, dry_run=False)
