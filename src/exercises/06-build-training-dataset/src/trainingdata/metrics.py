"""Throughput and packing efficiency, derived from the ledger rather than reported alongside it.

**The problem.** A packing or throughput figure that cannot be reconstructed earns no credit, and
rightly so. A number a run prints while it happens is a number nobody can check
afterwards. Every figure here is computed from artifacts that survive the run — the ledger and the
per-rank telemetry — so an auditor with the folder and none of the code can arrive at the same one.

**The number a throughput figure hides.** Tokens per second is easy to make large: pad more, grade
less, count padding as work. The honest figure is **loss-bearing tokens per second** — positions
that actually produced a gradient — and it is always the smaller of the two. Both are reported, so
the gap between them is visible rather than inferred.

**And one figure that is a constant, said plainly.** `pack_util` is **1.0 by construction**: the
plan cuts spans of exactly `sequence_length` and drops each shard's tail, so a window is always
full. Reporting it as though it were measured would be dressing a constant as a statistic. What
does vary is *loss* utilisation, because masking and document boundaries are ragged — 73.8% on the
context-masked reasoning lane against 99.6% on web.

Torch-free: these are sums over a record.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ledger, mixture


@dataclass(frozen=True, slots=True)
class Throughput:
    """What a run cost and what it bought.

    Attributes:
        steps: Optimizer steps completed.
        seconds: Wall clock across those steps, summed per rank.
        tokens: Token positions fed, padding included.
        loss_tokens: Positions that produced a gradient.
        tokens_per_second: The easy number.
        loss_tokens_per_second: The honest one — always the smaller.
        loss_utilization: Share of fed positions that earned gradient.
        pack_utilization: Share of positions holding real tokens. 1.0 by construction here.
        ranks: How many worker processes.
    """

    steps: int
    seconds: float
    tokens: int
    loss_tokens: int
    tokens_per_second: float
    loss_tokens_per_second: float
    loss_utilization: float
    pack_utilization: float
    ranks: int

    def as_json(self) -> dict:
        """The report as a JSON-serialisable dict.

        Returns:
            Field names to values.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LedgerTotals:
    """Everything a report needs, summed over a branch's ledger once.

    Attributes:
        events: Microbatches consumed.
        tokens: Token positions fed.
        loss_tokens: Positions that earned gradient.
        pad_tokens: Positions holding nothing.
        lane_tokens: Real tokens per lane.
        steps: Distinct optimizer steps.
        ranks: Distinct workers.
        reexecuted: Events re-executing a discarded one after a resume.
        context_masked: Events whose loss mask excluded a context span.
    """

    events: int = 0
    tokens: int = 0
    loss_tokens: int = 0
    pad_tokens: int = 0
    lane_tokens: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    ranks: int = 0
    reexecuted: int = 0
    context_masked: int = 0


def totals(events: list[ledger.ConsumeEvent]) -> LedgerTotals:
    """Sum a branch's ledger.

    Args:
        events: Every event the branch consumed.

    Returns:
        The totals.
    """
    lanes: dict[str, int] = {}
    steps: set[int] = set()
    ranks: set[int] = set()
    tokens = loss_tokens = pad_tokens = reexecuted = masked = 0

    for event in events:
        tokens += event.tokens
        loss_tokens += event.loss_tokens
        pad_tokens += event.pad_tokens
        steps.add(event.global_step)
        ranks.add(event.rank)
        reexecuted += event.replayed_from is not None
        masked += event.loss_policy == "context-masked"
        for lane, count in event.lane_mix.items():
            lanes[lane] = lanes.get(lane, 0) + count

    return LedgerTotals(
        events=len(events),
        tokens=tokens,
        loss_tokens=loss_tokens,
        pad_tokens=pad_tokens,
        lane_tokens=dict(sorted(lanes.items())),
        steps=len(steps),
        ranks=len(ranks),
        reexecuted=reexecuted,
        context_masked=masked,
    )


def read_telemetry(artifact_dir: Path, branch_id: str) -> list[dict]:
    """Every rank's telemetry file for a branch, across attempts.

    Args:
        artifact_dir: The run's artifact directory.
        branch_id: Which branch.

    Returns:
        The parsed reports, sorted by attempt then rank.
    """
    directory = artifact_dir / "telemetry"
    if not directory.is_dir():
        return []
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob(f"{branch_id}.rank*.attempt*.json"))
        if ".exit." not in path.name
    ]
    return sorted(reports, key=lambda r: (r.get("attempt", 0), r.get("rank", 0)))


def throughput(events: list[ledger.ConsumeEvent], telemetry: list[dict]) -> Throughput:
    """Compute the run's throughput from the ledger and the per-rank timings.

    **Wall clock is taken as the SLOWEST rank per step, not the sum.** The ranks run concurrently;
    adding their seconds together would report a four-rank run as four times slower than it was,
    and dividing tokens by that sum would understate throughput by the same factor. A step ends
    when its slowest rank ends, because the all-reduce makes them wait.

    Args:
        events: The branch's ledger.
        telemetry: Per-rank reports from `read_telemetry`.

    Returns:
        The report.
    """
    summed = totals(events)

    per_step: dict[int, float] = {}
    for report in telemetry:
        for entry in report.get("steps", []):
            step = int(entry["step"])
            per_step[step] = max(per_step.get(step, 0.0), float(entry["seconds"]))
    seconds = sum(per_step.values())

    return Throughput(
        steps=len(per_step) or summed.steps,
        seconds=seconds,
        tokens=summed.tokens,
        loss_tokens=summed.loss_tokens,
        tokens_per_second=(summed.tokens / seconds) if seconds else 0.0,
        loss_tokens_per_second=(summed.loss_tokens / seconds) if seconds else 0.0,
        loss_utilization=(summed.loss_tokens / summed.tokens) if summed.tokens else 0.0,
        pack_utilization=(
            (summed.tokens - summed.pad_tokens) / summed.tokens if summed.tokens else 0.0
        ),
        ranks=max((int(r.get("rank", 0)) for r in telemetry), default=summed.ranks - 1) + 1,
    )


def mixture_report(events: list[ledger.ConsumeEvent], *, tolerance: float = 0.01) -> dict:
    """Planned versus actual lane shares, from what was actually consumed.

    Args:
        events: The branch's ledger.
        tolerance: How far a lane may drift before it is out of compliance.

    Returns:
        `{lanes: {...}, compliant: bool, floors_held: bool, consumed: {...}}`.
    """
    consumed = totals(events).lane_tokens
    lanes = mixture.compliance(consumed, tolerance=tolerance)
    funded = [row for lane, row in lanes.items() if mixture.LANE_SHARES[lane] > 0]
    return {
        "consumed": consumed,
        "lanes": lanes,
        "compliant": all(row["within_tolerance"] for row in funded),
        "floors_held": all(row["floor_held"] for row in lanes.values()),
    }
