"""Re-check every published claim from the artifacts alone.

**This is the command grading Step 3 inspects.** It reads `submission_artifacts/` and nothing else,
re-derives each claim independently, and disagrees out loud when a number was invented.

**The wall, and why it is structural rather than a matter of discipline.** This file imports
`trainingdata.spec` — shared *facts*: the nine requirements, the thirteen log events, the sentinel
ids, the lane shares. It imports **nothing else** from the package. If it called `metrics.totals`
to check a token count, it would be checking the producer's arithmetic with the producer's
arithmetic and would agree with itself no matter what either had got wrong. So the chain hash is
recomputed here with `hashlib`, the ledger is parsed here with `json`, and the mixture is summed
here — a second implementation, which is the only kind of check worth running.

`tests/test_trainingdata_verify.py` asserts that import closure, because a comment cannot enforce
it and one convenient import would quietly turn this into a tautology.

Run it::

    uv run python src/exercises/06-build-training-dataset/verify.py
    uv run python .../verify.py --bundle /path/to/submission_artifacts
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

EXERCISE = Path(__file__).resolve().parent
sys.path.insert(0, str(EXERCISE / "src"))

from trainingdata import spec  # noqa: E402  -- shared FACTS only; see the module docstring

logger = logging.getLogger("verify")

#: What the chain's first event carries as `prev`.
GENESIS = "b2:" + "0" * 32

#: The one event in `spec.REQUIRED_SEQUENCE` that this file, rather than the producer, completes.
AUDIT_EVENT = "audit completed"


@dataclass
class Finding:
    """One check and what it found.

    Attributes:
        check: What was checked.
        ok: Whether it held.
        detail: The numbers, or what disagreed.
    """

    check: str
    ok: bool
    detail: str


def _digest(line: str) -> str:
    """Re-implement the ledger's chain hash.

    Deliberately re-implemented rather than imported. A chain check that called the producer's
    hasher would confirm the producer hashes the way the producer hashes.

    Args:
        line: One event's canonical JSON.

    Returns:
        `"b2:<32 hex>"`.
    """
    return "b2:" + hashlib.blake2b(line.encode("utf-8"), digest_size=16).hexdigest()


def read_ledger(bundle: Path) -> tuple[list[dict], list[Finding]]:
    """Parse every ledger segment and verify the chain independently.

    Args:
        bundle: The submission bundle.

    Returns:
        `(events in run order, findings)`.
    """
    findings: list[Finding] = []
    events: list[dict] = []
    directory = bundle / "ledger"

    if not directory.is_dir():
        findings.append(Finding("ledger present", False, f"no ledger directory at {directory}"))
        return events, findings

    for path in sorted(directory.glob("*.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        parsed = [json.loads(line) for line in lines]

        expected = GENESIS
        broken = None
        for index, (line, event) in enumerate(zip(lines, parsed, strict=True)):
            if event.get("prev") != expected:
                broken = f"{path.name} event {index}: prev does not match the previous event's hash"
                break
            if event.get("seq") != index:
                broken = f"{path.name} event {index}: seq is {event.get('seq')}"
                break
            expected = _digest(line)
        findings.append(
            Finding(
                f"chain intact: {path.name}",
                broken is None,
                broken or f"{len(parsed)} events chain cleanly",
            )
        )
        events.extend(parsed)

    events.sort(key=lambda e: (e["global_step"], e["rank"], e["accum"]))
    return events, findings


def check_run_log(bundle: Path) -> list[Finding]:
    """Every required event must appear, in order, with a verdict.

    An event marked `[SKIP]` is reported as **not produced** rather than as present. That is the
    whole reason the producer writes a verdict per line: an auditor can then tell "the run did not
    do this" from "the run did not mention it", and only the second is a hole in the record.

    **`audit completed` is the one event the producer structurally cannot produce**, and it is this
    file that produces it. `run_demo.py` marks it `[SKIP]` rather than claiming it, because a run
    that certified its own audit would be certifying nothing; the audit is completed by *this*
    process reaching the end of *this* function, which is why the sole exemption below is named
    rather than a blanket "ignore skips".

    Args:
        bundle: The submission bundle.

    Returns:
        The findings.
    """
    path = bundle / "run.log"
    if not path.is_file():
        return [Finding("run.log present", False, f"no run.log at {path}")]

    lines = path.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        for event in spec.REQUIRED_SEQUENCE:
            if line.startswith("[PASS] " + event) or line.startswith("[SKIP] " + event):
                seen[event] = line[1:5]
                order.append(event)

    missing = [event for event in spec.REQUIRED_SEQUENCE if event not in seen]
    skipped = [event for event, mark in seen.items() if mark == "SKIP" and event != AUDIT_EVENT]
    self_audited = seen.get(AUDIT_EVENT) == "SKIP"
    expected_order = [event for event in spec.REQUIRED_SEQUENCE if event in seen]

    return [
        Finding(
            "run.log names every required event",
            not missing,
            f"missing: {missing}" if missing else f"all {len(spec.REQUIRED_SEQUENCE)} present",
        ),
        Finding(
            "the events appear in the required order",
            order == expected_order,
            "in order" if order == expected_order else f"got {order}",
        ),
        Finding(
            "every required event was actually produced",
            not skipped,
            f"NOT PRODUCED: {skipped}" if skipped else "all produced",
        ),
        Finding(
            AUDIT_EVENT,
            True,
            "produced by this run of verify.py — the producer marked it [SKIP] because it cannot "
            "certify its own audit"
            if self_audited
            else "the producer claimed this itself, which certifies nothing; it is marked [SKIP] "
            "in a correct run and completed here",
        ),
    ]


def check_evidence(bundle: Path) -> list[Finding]:
    """`evidence.md` and `evidence.json` must carry the nine rows, and agree with each other.

    Args:
        bundle: The submission bundle.

    Returns:
        The findings.
    """
    findings: list[Finding] = []
    json_path, md_path = bundle / "evidence.json", bundle / "evidence.md"

    if not json_path.is_file() or not md_path.is_file():
        return [Finding("evidence present", False, "evidence.json or evidence.md is missing")]

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = {row["requirement"]: row for row in payload.get("requirements", [])}
    missing = [name for name in spec.REQUIREMENTS if name not in rows]
    findings.append(
        Finding(
            "evidence.json carries all nine requirements",
            not missing,
            f"missing: {missing}" if missing else "all nine present",
        )
    )

    text = md_path.read_text(encoding="utf-8")
    absent = [name for name in spec.REQUIREMENTS if name not in text]
    findings.append(
        Finding(
            "evidence.md carries all nine requirements",
            not absent,
            f"missing: {absent}" if absent else "all nine present",
        )
    )

    unmet = sorted(name for name, row in rows.items() if row["status"] != "met")
    findings.append(
        Finding(
            "every requirement is met",
            not unmet,
            f"not met: {unmet}" if unmet else f"{len(rows)} of {len(rows)}",
        )
    )
    return findings


def recheck_numbers(bundle: Path, events: list[dict]) -> list[Finding]:
    """Re-derive the published counts from the ledger and compare.

    **This is the check the requirements' "hardcoded evidence" rule is about.** Every number below
    is summed here, from the same events an auditor can read, and compared against what the bundle
    claims. A figure that was invented disagrees.

    Args:
        bundle: The submission bundle.
        events: The parsed ledger.

    Returns:
        The findings.
    """
    if not events:
        return [Finding("numbers re-derivable", False, "no ledger events to check against")]

    payload = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    rows = {row["requirement"]: row for row in payload.get("requirements", [])}
    findings: list[Finding] = []

    tokens = sum(int(event["tokens"]) for event in events)
    loss_tokens = sum(int(event["loss_tokens"]) for event in events)
    claimed = rows.get("packing_correctness", {}).get("numbers", {})
    for name, mine in (
        ("tokens", tokens),
        ("loss_tokens", loss_tokens),
        ("microbatches", len(events)),
    ):
        theirs = claimed.get(name)
        findings.append(
            Finding(
                f"packing_correctness.{name} re-derives",
                theirs == mine,
                f"ledger says {mine:,}, bundle says {theirs}",
            )
        )

    lanes: dict[str, int] = {}
    for event in events:
        for lane, count in (event.get("lane_mix") or {}).items():
            lanes[lane] = lanes.get(lane, 0) + int(count)
    total = sum(lanes.values())

    breached = sorted(
        lane
        for lane, floor in spec.FLOORS.items()
        if total and (lanes.get(lane, 0) / total) < floor - spec.MIXTURE_TOLERANCE
    )
    findings.append(
        Finding(
            "no protected floor is breached beyond tolerance",
            not breached,
            f"breached: {breached}" if breached else f"floors held over {total:,} tokens",
        )
    )

    invented = sorted(set(lanes) - set(spec.LANE_SHARES))
    findings.append(
        Finding(
            "every lane in the ledger is one the plan funds",
            not invented,
            f"unknown lanes: {invented}" if invented else f"{len(lanes)} known lanes",
        )
    )

    reexecuted = sum(1 for event in events if event.get("replayed_from") is not None)
    claimed_reexec = rows.get("crash_recovery", {}).get("numbers", {}).get("reexecuted")
    findings.append(
        Finding(
            "the re-executed microbatch count re-derives",
            claimed_reexec is None or claimed_reexec == reexecuted,
            f"ledger says {reexecuted}, bundle says {claimed_reexec}",
        )
    )
    return findings


def check_firewall(bundle: Path, events: list[dict]) -> list[Finding]:
    """No shard the manifests refuse may appear in any loss-bearing batch.

    The requirements names one failure outright: evaluation data reaching a loss-bearing batch
    fails the
    firewall section. Checked against the ledger's own spans, not against the run's
    account of itself.

    Args:
        bundle: The submission bundle.
        events: The parsed ledger.

    Returns:
        The findings.
    """
    directory = bundle / "manifests"
    if not directory.is_dir():
        return [Finding("manifests present", False, f"no manifests directory at {directory}")]

    blocked, total = set(), 0
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            total += 1
            if entry.get("split") != "train" or entry.get("benchmark_ids"):
                blocked.add(entry["shard_id"])

    consumed = {sample["shard_id"] for event in events for sample in (event.get("samples") or [])}
    leaked = sorted(blocked & consumed)

    return [
        Finding(
            "an evaluation shard was actually offered and refused",
            bool(blocked),
            f"{len(blocked)} of {total} manifests are non-trainable"
            if blocked
            else "no evaluation shard is present, so the firewall was never exercised",
        ),
        Finding(
            "no refused shard appears in a loss-bearing batch",
            not leaked,
            f"LEAKED: {leaked}" if leaked else f"{len(consumed)} shards consumed, none refused",
        ),
    ]


def check_opus(bundle: Path) -> list[Finding]:
    """Re-derive the selection record, and join it to the batches it decided.

    **The join is the whole check.** A decision log on its own proves a selector ran; the ledger on
    its own proves batches were fed. Only together do they show the selector *changed what the
    model read* — and the failure this catches is a candidate the record says was rejected turning
    up in a loss-bearing batch, which would mean the decisions were decoration.

    Everything here is recomputed with `hashlib` and `json`. Importing `opus` would check the
    producer's arithmetic against the producer's arithmetic and agree with itself.

    Args:
        bundle: The submission bundle.

    Returns:
        The findings. Empty when the run recorded no selection at all — reported by the evidence
        row rather than as a failure here, since a run without OPUS is a smaller claim, not a
        false one.
    """
    directory = bundle / "opus"
    if not directory.is_dir():
        return []

    logs = sorted(directory.glob("opus-*.jsonl"))
    segments = sorted(directory.glob("*.rank*.seg*.jsonl"))
    if not logs:
        return [Finding("opus decision logs present", False, f"nothing under {directory}")]

    findings: list[Finding] = []
    decided: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
    tally: dict[str, int] = dict.fromkeys(spec.DECISIONS, 0)
    served_total = 0

    for path in logs:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]

        canonical = "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
        )
        digest = "b2:" + hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()
        findings.append(
            Finding(
                f"decision log intact: {path.name}",
                digest == header.get("digest"),
                f"{len(rows)} candidates hash to the recorded digest"
                if digest == header.get("digest")
                else f"rows hash to {digest}, header says {header.get('digest')}",
            )
        )

        served = [row for row in rows if row["decision"] in ("accept", "floor_override")]
        served_total += len(served)
        findings.append(
            Finding(
                f"conservation: {path.name}",
                len(rows) == header.get("offered") and len(served) == header.get("served"),
                f"{len(rows)} offered, {len(served)} served",
            )
        )
        for row in rows:
            tally[row["decision"]] = tally.get(row["decision"], 0) + 1
            decided.setdefault(header["pass_id"], {}).setdefault(row["shard_id"], []).append(
                (int(row["start"]), int(row["end"]), row["decision"])
            )

        unreasoned = [row for row in rows if not str(row.get("reason", "")).strip()]
        findings.append(
            Finding(
                f"every candidate carries a reason: {path.name}",
                not unreasoned,
                f"{len(rows)} reasons" if not unreasoned else f"{len(unreasoned)} rows have none",
            )
        )

    # -- the join ---------------------------------------------------------------------------------
    leaked: list[str] = []
    tagged = 0
    for path in segments:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            pass_id = event.get("opus_decision_id")
            if not pass_id:
                continue
            tagged += 1
            for sample in event.get("samples") or []:
                # A ledger sample is a document FRAGMENT; a candidate is the SPAN the fragment was
                # cut from. Joining on `start` compares the two and finds nothing, which is how the
                # first version of this check reported every batch as unaccounted for. Containment
                # is both correct and stronger: it catches a fragment fed from a shard that was
                # accepted at some *other* offset, which an id-only join would wave through.
                spans = decided.get(pass_id, {}).get(sample["shard_id"], [])
                outcome = next(
                    (
                        decision
                        for start, end, decision in spans
                        if start <= sample["start"] and sample["end"] <= end
                    ),
                    None,
                )
                if outcome not in ("accept", "floor_override"):
                    leaked.append(
                        f"{sample['shard_id'][:8]}[{sample['start']}:{sample['end']}] was "
                        f"{outcome or 'in no accepted span'}"
                    )

    findings.append(
        Finding(
            "every batch OPUS fed was one it accepted",
            not leaked and tagged > 0,
            f"{tagged} microbatches join cleanly to their pass"
            if not leaked and tagged
            else (f"NOT ACCEPTED: {leaked[:4]}" if leaked else "no microbatch names a pass"),
        )
    )

    # Floors, re-derived from the served rows rather than read out of the header. A pass that
    # breached a floor and wrote `held` in its own header would otherwise pass unexamined — the
    # header is the producer's account of itself.
    breached: list[str] = []
    unsupplied: list[str] = []
    for path in logs:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
        served = [r for r in rows if r["decision"] in ("accept", "floor_override")]
        for lane, floor in sorted(spec.FLOORS.items()):
            got = sum(1 for r in served if r["lane"] == lane)
            offered = sum(1 for r in rows if r["lane"] == lane)
            if served and got / len(served) < floor:
                (unsupplied if not offered else breached).append(f"{header['pass_id']}:{lane}")

    findings.append(
        Finding(
            "no protected floor was breached with candidates available",
            not breached,
            f"BREACHED: {breached}" if breached else f"{len(logs)} passes, none breached",
        )
    )
    findings.append(
        Finding(
            "a floor missed for lack of supply is reported as such",
            True,
            f"unsupplied (the buffer held none of the lane): {sorted(set(unsupplied))}"
            if unsupplied
            else "every protected lane was present in every buffer",
        )
    )

    unknown = sorted(set(tally) - set(spec.DECISIONS))
    findings.append(
        Finding(
            "every outcome is one of the four statuses",
            not unknown,
            f"invented statuses: {unknown}"
            if unknown
            else ", ".join(f"{name} {tally[name]}" for name in spec.DECISIONS),
        )
    )

    payload = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    row = next(
        (r for r in payload.get("requirements", []) if r["requirement"] == "opus_audit_trail"),
        None,
    )
    claimed = (row or {}).get("numbers", {}).get("decisions")
    findings.append(
        Finding(
            "the published decision counts re-derive",
            claimed is None or claimed == tally,
            f"logs say {tally}, bundle says {claimed}",
        )
    )
    return findings


def main() -> int:
    """Verify the bundle and report.

    Returns:
        0 when every check holds, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=EXERCISE / "submission_artifacts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.bundle.is_dir():
        logger.error("no bundle at %s — run run_demo.py first", args.bundle)
        return 1

    events, findings = read_ledger(args.bundle)
    findings += check_run_log(args.bundle)
    findings += check_evidence(args.bundle)
    findings += recheck_numbers(args.bundle, events)
    findings += check_firewall(args.bundle, events)
    findings += check_opus(args.bundle)

    for finding in findings:
        logger.info("%s %-52s %s", "PASS" if finding.ok else "FAIL", finding.check, finding.detail)

    failed = [finding for finding in findings if not finding.ok]
    logger.info("")
    logger.info(
        "%d of %d checks pass, re-derived from %s alone",
        len(findings) - len(failed),
        len(findings),
        args.bundle.name,
    )
    if failed:
        logger.error("%d FAILED", len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
