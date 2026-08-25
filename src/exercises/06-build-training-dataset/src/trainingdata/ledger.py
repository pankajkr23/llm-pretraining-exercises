"""The consumption ledger — an append-only record of what training actually consumed.

**The problem.** A planned order is not what happened. Workers restart, ranks retry, checkpoints are
restored, a selector rejects a batch. The lecture's opening: fifty days into a run you want to know
what the model read on day forty, and all you have is a folder.

**The strategy — write it down as it happens, and make the record authoritative.** The session's own
answer to "how is this reproducible without a seed":

> *"I will not run the code… I'm going to run the ledger. I'm going to read and send. **I will not
> calculate it.**"*

That only works if the record carries everything replay needs. So an event names its span by
`(shard_id, start, end)` — addressable at any corpus size — and carries the hashes of what was
actually fed, so replay can re-materialise and **re-derive** them rather than echo them back.

**Three structural choices, each for a specific failure:**

*One file per `(branch, rank, segment)`.* Four ranks writing one file corrupt it. No shared writer
means no locking, and no locking means no lock to hold at the moment a process dies.

*A new segment on every process start, created with `O_EXCL`.* If a resumed run reopened the file
its predecessor was writing, a crash mid-line would leave a torn record that the new process would
append to. A fresh file per attempt makes the boundary explicit, and `O_EXCL` makes two processes
claiming the same segment an error rather than a silent interleave.

*Each event carries the hash of the previous one.* The file is a chain: alter any line and every
line after it fails to verify. An append-only file you can edit is not append-only.

Torch-free, like everything on the data path.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Schema version. Present in every event so a reader can refuse a file it does not understand
#: rather than silently misinterpreting fields.
VERSION = 1

#: The chain's starting value — what `prev` holds for the first event in a segment.
GENESIS = "b2:" + "0" * 32


class LedgerCorruptionError(Exception):
    """A ledger file could not be read as a ledger.

    Separate from `ValueError` so a caller can tell "this file is damaged" from "you passed me a
    bad argument" — resume needs to act on the first and abort on the second.
    """


def _digest(payload: str) -> str:
    """Chain-hash one serialised event.

    Args:
        payload: The event's canonical JSON.

    Returns:
        `"b2:<32 hex>"`.
    """
    return "b2:" + hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class PackedSample:
    """One document's contribution to a packed window.

    `pass_no` is what lets the learning ledger tell a re-read from a first exposure. Without it a
    second epoch is invisible, and the repeated-pass effect cannot be measured at all.
    """

    shard_id: str
    start: int
    end: int
    lane: str
    loss_tokens: int
    pass_no: int = 1


@dataclass(frozen=True, slots=True)
class ConsumeEvent:
    """One microbatch, as it was actually fed to the model.

    The hashes are the load-bearing part. Replay re-materialises the window from the spans and
    **re-derives** these; comparing a recomputed hash against a recorded one is a real check, while
    reading the recorded one back and printing it proves nothing.
    """

    # -- chain ---------------------------------------------------------------------------------
    v: int
    seq: int
    prev: str

    # -- which run, and which attempt at it ----------------------------------------------------
    run_id: str
    branch_id: str
    segment: int
    attempt: int

    # -- where in the run ----------------------------------------------------------------------
    global_step: int
    rank: int
    accum: int
    flat: int
    checkpoint_id: str | None

    # -- what was fed --------------------------------------------------------------------------
    samples: tuple[PackedSample, ...]
    tokens: int
    loss_tokens: int
    pad_tokens: int
    pack_util: float

    # -- how it was assembled ------------------------------------------------------------------
    stage: str
    lane_mix: dict[str, int]
    attention_policy: str
    position_policy: str
    pack_policy: str

    # -- why it was fed ------------------------------------------------------------------------
    opus_decision_id: str | None

    # -- what it hashed to ---------------------------------------------------------------------
    microbatch_hash: str
    loss_mask_hash: str
    position_ids_hash: str
    segment_ids_hash: str

    # -- provenance of the meaning of the ids --------------------------------------------------
    tokenizer_sha256: str
    plan_key_digest: str

    #: Set when this event re-executes a parent after a resume. `None` on the happy path. Published
    #: rather than hidden: "no skipped or repeated batches" is true of the effective post-cut
    #: ledger, never of the device, and the count of re-executions is the honest version of that.
    replayed_from: int | None = None

    def canonical(self) -> str:
        """The event as canonical JSON, with `prev` included and the chain hash excluded.

        Returns:
            Sorted-key JSON with no incidental whitespace, so the digest depends on the values and
            not on how they happened to be formatted.
        """
        payload = asdict(self)
        payload["samples"] = [asdict(s) for s in self.samples]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def chain_hash(self) -> str:
        """This event's contribution to the chain.

        Returns:
            The digest the next event will carry as `prev`.
        """
        return _digest(self.canonical())

    @classmethod
    def from_json(cls, payload: dict) -> "ConsumeEvent":
        """Rebuild an event from its serialised form.

        Args:
            payload: A decoded JSON object.

        Returns:
            The event.

        Raises:
            ValueError: If the schema version is not one this code understands.
        """
        if payload.get("v") != VERSION:
            raise ValueError(
                f"ledger event has schema v{payload.get('v')}, this code reads v{VERSION}. "
                f"Refusing rather than misinterpreting fields."
            )
        data = dict(payload)
        data["samples"] = tuple(PackedSample(**s) for s in data.get("samples", []))
        data["lane_mix"] = dict(data.get("lane_mix") or {})
        return cls(**data)


@dataclass
class LedgerWriter:
    """Appends events to one `(branch, rank, segment)` file, maintaining the chain.

    Not thread-safe and not meant to be: each rank owns its own writer and its own file, which is
    what removes the need for a lock at all.
    """

    directory: Path
    run_id: str
    branch_id: str
    rank: int
    segment: int
    _seq: int = field(default=0, init=False)
    _prev: str = field(default=GENESIS, init=False)
    _path: Path | None = field(default=None, init=False)

    @property
    def path(self) -> Path:
        """The file this writer appends to.

        Returns:
            `<dir>/<branch>.rank<N>.seg<M>.jsonl`.
        """
        return self.directory / f"{self.branch_id}.rank{self.rank}.seg{self.segment}.jsonl"

    def open(self) -> Path:
        """Create the segment file, refusing to reuse one.

        `O_EXCL` on purpose: if a resumed process reopened its predecessor's file, a crash
        mid-line would leave a torn record that the new process would append to. Two processes
        claiming one segment becomes an error rather than a silent interleave.

        Returns:
            The path created.

        Raises:
            FileExistsError: If the segment already exists.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        self._path = self.path
        return self.path

    def append(self, **fields) -> ConsumeEvent:
        """Write one consume event, linked to the previous.

        The write is followed by `flush` and `fsync`, so an event that has returned has reached the
        filesystem. On macOS that guarantees survival of *process death*, not of power loss — only
        `F_FULLFSYNC` buys the latter, and that distinction is stated rather than assumed.

        Args:
            **fields: Everything `ConsumeEvent` needs except the chain fields.

        Returns:
            The event as written.

        Raises:
            LedgerCorruptionError: If `open()` was never called, which would bypass the
                `O_EXCL` claim.
        """
        if self._path is None:
            raise LedgerCorruptionError(
                f"append() before open() on {self.path.name}. Appending straight to the file would "
                f"bypass the O_EXCL claim, so a second process could interleave into a segment "
                f"this one believes it owns."
            )
        event = ConsumeEvent(
            v=VERSION,
            seq=self._seq,
            prev=self._prev,
            run_id=self.run_id,
            branch_id=self.branch_id,
            segment=self.segment,
            rank=self.rank,
            **fields,
        )
        line = event.canonical()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            # No test covers this line and none can at this level: closing the handle already hands
            # the bytes to the OS, so process death is survived either way, and the difference
            # fsync buys -- surviving power loss -- is not observable without cutting power. It is
            # kept because it is correct, and named here so it is not mistaken for tested.
            os.fsync(handle.fileno())
        self._seq += 1
        self._prev = _digest(line)
        return event

    @property
    def length(self) -> int:
        """How many events this writer has appended.

        This is the value a checkpoint records as the rank's cut. A scalar cannot cut R files.

        Returns:
            The event count.
        """
        return self._seq


def read_segment(path: Path) -> list[ConsumeEvent]:
    """Every event in one segment file, in order.

    Args:
        path: The segment file.

    Returns:
        The events. Empty when the file does not exist or holds nothing.

    Raises:
        LedgerCorruptionError: If any line does not parse. The line number is named, because
            "the ledger is broken" is not actionable when a run has four ranks and eight segments.
    """
    if not path.is_file():
        return []
    events: list[ConsumeEvent] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            events.append(ConsumeEvent.from_json(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LedgerCorruptionError(f"{path.name} line {i + 1}: {exc}") from exc
    return events


def scan_segment(path: Path) -> tuple[int, bool]:
    """How many complete events a segment holds, and whether its last line is torn.

    Read-only. `drop_torn_tail` repairs using this, and a resume's dry run measures using it — one
    implementation of "what does this file actually contain", so a check and the repair that follows
    it cannot disagree.

    Args:
        path: The segment file.

    Returns:
        `(complete lines, last line is an interrupted write)`.

    Raises:
        LedgerCorruptionError: If a line other than the last fails to parse. That cannot be an
            interrupted write — only the last line can be — so it is corruption, and repairing it
            would hide real damage behind a routine crash-recovery path.
    """
    if not path.is_file():
        return 0, False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return 0, False

    for i, line in enumerate(lines[:-1]):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerCorruptionError(
                f"{path.name} line {i + 1} does not parse, and it is not the last line — this is "
                f"corruption, not an interrupted write, and repairing it would hide the damage: "
                f"{exc}"
            ) from exc

    try:
        json.loads(lines[-1])
    except json.JSONDecodeError:
        return len(lines) - 1, True
    return len(lines), False


def drop_torn_tail(path: Path) -> bool:
    """Remove a trailing line the writing process did not finish.

    **Why a torn tail is possible at all.** `append` flushes and `fsync`s every event, so a
    completed event has reached the filesystem. But the kill can land *during* the `write` — POSIX
    does not promise atomicity for a write larger than `PIPE_BUF` — so the last line can be a
    prefix of an event rather than an event.

    **Why parse-failure is a sound detector.** A JSON object ends in `}`. Truncating one always
    removes that brace, so a torn line never parses; and a line that *does* parse carries every
    field, so it is a complete event that merely lacks its newline. The check is therefore exact in
    both directions rather than a heuristic.

    Args:
        path: The segment file.

    Returns:
        Whether a torn line was dropped.

    Raises:
        LedgerCorruptionError: If a line other than the last fails to parse.
    """
    complete, torn = scan_segment(path)
    if not torn:
        return False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path.write_text("".join(line + "\n" for line in lines[:complete]), encoding="utf-8")
    return True


def next_segment(directory: Path, branch_id: str, rank: int) -> int:
    """The segment number a newly-started process should claim for this rank.

    One higher than the highest already present, so a resumed process never reopens the file its
    predecessor was writing.

    Args:
        directory: The ledger directory.
        branch_id: The branch.
        rank: The worker.

    Returns:
        `0` when the rank has written nothing yet.
    """
    existing = [
        int(p.name.split(".")[2].removeprefix("seg"))
        for p in directory.glob(f"{branch_id}.rank{rank}.seg*.jsonl")
    ]
    return max(existing) + 1 if existing else 0


def verify_chain(events: list[ConsumeEvent]) -> tuple[bool, str]:
    """Check that the chain has not been altered.

    Args:
        events: A segment's events, in file order.

    Returns:
        Whether the chain holds, and a message naming the first break.
    """
    expected = GENESIS
    for i, event in enumerate(events):
        if event.prev != expected:
            return False, (
                f"event {i} (seq {event.seq}) expected prev={expected[:14]}… but carries "
                f"{event.prev[:14]}… — the file was altered at or before this line"
            )
        if event.seq != i:
            return False, f"event at line {i} claims seq {event.seq}: the file is out of order"
        expected = event.chain_hash()
    return True, f"{len(events)} events chain cleanly"


def segments_for(directory: Path, branch_id: str) -> list[Path]:
    """Every segment file belonging to a branch, sorted by rank then segment.

    Args:
        directory: The ledger directory.
        branch_id: The branch.

    Returns:
        The paths.
    """
    if not directory.is_dir():
        return []

    def key(p: Path) -> tuple[int, int]:
        parts = p.name.split(".")
        return int(parts[1].removeprefix("rank")), int(parts[2].removeprefix("seg"))

    return sorted(directory.glob(f"{branch_id}.rank*.seg*.jsonl"), key=key)


def read_branch(directory: Path, branch_id: str) -> list[ConsumeEvent]:
    """Every event a branch consumed, ordered as the run consumed them.

    Sorted by `(global_step, rank, accum)` rather than by file, because the files are per-rank and
    the run interleaved them. This is the order an audit reads.

    Args:
        directory: The ledger directory.
        branch_id: The branch.

    Returns:
        The events.
    """
    events: list[ConsumeEvent] = []
    for path in segments_for(directory, branch_id):
        events.extend(read_segment(path))
    return sorted(events, key=lambda e: (e.global_step, e.rank, e.accum))


def truncate_to(path: Path, keep: int) -> int:
    """Cut a segment file back to its first `keep` events.

    This is what resume does with the checkpoint's recorded cut: anything after it was consumed by
    a process whose work the checkpoint does not include, so replaying it would double-count.

    Rewrites rather than truncating in place, because a byte offset is not a line boundary and
    cutting mid-line would leave a corrupt final record.

    Args:
        path: The segment file.
        keep: How many events to keep.

    Returns:
        How many were discarded.

    Raises:
        ValueError: If `keep` is negative or exceeds what the file holds.
    """
    if keep < 0:
        raise ValueError(f"keep must be non-negative, got {keep}")
    events = read_segment(path)
    if keep > len(events):
        raise ValueError(
            f"cut asks to keep {keep} events but the segment holds {len(events)} — the checkpoint "
            f"and this ledger do not belong to the same run"
        )
    dropped = len(events) - keep
    lines = [e.canonical() for e in events[:keep]]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return dropped
