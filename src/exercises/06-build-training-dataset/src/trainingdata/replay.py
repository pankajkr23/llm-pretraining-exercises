"""Replay — reconstructing what a run consumed by **reading the ledger**, never by recomputing it.

**The problem.** "Reproducible" usually means: keep the seed, run it again, get the same thing. That
breaks the moment anything in the data path depends on the model — once a selector scores candidates
against the current checkpoint, the plan stops being a pure function of position, and re-deriving it
can never be bit-identical. It also breaks quietly for duller reasons: a planner change, a different
shard order, a library upgrade.

**The strategy**, and it is deliberately not the obvious one: do not re-run the code, because
nondeterminism creeps in. **Run the ledger instead** — read what was recorded and replay it, rather
than recomputing it and hoping the second answer matches the first.

So this module never asks `plan.py` anything. It reads the recorded spans, slices the immutable
shards at exactly those offsets, rebuilds the masks and positions, and hashes the result.

**Re-derive, never echo.** The event's recorded hashes are the *comparison*, never the answer.
Reading `event.microbatch_hash` back and printing it proves nothing at all; recomputing it from the
shard bytes and finding it equal proves the shard still holds what the run was fed. That distinction
is the whole value of the exercise and it is one line of code apart from being worthless.

**No torch, and no import of the planner.** `tests/test_trainingdata_replay.py` asserts the import
closure — transitively, not just this file's own line — because the discipline is not something a
comment can enforce. A single `from . import plan` would make it possible to accidentally recompute,
and the failure would look exactly like success.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ledger, masks, pack, shards, spec


@dataclass(frozen=True, slots=True)
class Rebuilt:
    """One microbatch, re-materialised from the ledger and the shards.

    Attributes:
        tokens: `(microbatch, sequence_length)`.
        segments: Which fragment each position belongs to; `-1` for padding.
        positions: Position within each token's own document.
        loss: Which positions were graded.
    """

    tokens: np.ndarray
    segments: np.ndarray
    positions: np.ndarray
    loss: np.ndarray

    def hashes(self) -> dict[str, str]:
        """Hashes derived from the rebuilt arrays.

        Returns:
            The same four keys a `ConsumeEvent` carries.
        """
        return {
            "microbatch_hash": pack.hash_array(self.tokens),
            "loss_mask_hash": pack.hash_array(self.loss),
            "position_ids_hash": pack.hash_array(self.positions),
            "segment_ids_hash": pack.hash_array(self.segments),
        }


@dataclass
class ShardSource:
    """Somewhere to get shard tokens from, with each shard verified once.

    Takes an explicit `shard_id -> path` map rather than a directory, because the corpus is written
    one directory per lane and because the auditor should be told where the shards are by the
    evidence bundle rather than guessing from a layout convention.

    Verification happens on first use rather than up front: a replay of steps 80–120 touches a
    handful of shards, and re-hashing a whole corpus to check four of them is the difference between
    a check people run and one they skip.
    """

    paths: dict[str, Path]
    expected: dict[str, str] = field(default_factory=dict)
    _cache: dict[str, tuple[np.ndarray, pack.DocIndex]] = field(default_factory=dict, init=False)
    #: Shards whose bytes no longer match what was recorded, with what they hash to now. Collected
    #: rather than raised: one tampered shard should turn exactly the batches that used it red, and
    #: leave the rest of the replay as evidence that the damage was local.
    tampered: dict[str, str] = field(default_factory=dict, init=False)

    @classmethod
    def from_directories(cls, directories, expected=None) -> "ShardSource":
        """Find every `*.bin` under the given directories.

        Args:
            directories: Directories to scan, non-recursively.
            expected: Shard id to recorded content hash.

        Returns:
            The source.
        """
        paths = {}
        for directory in directories:
            for path in sorted(Path(directory).glob("*.bin")):
                paths[path.stem] = path
        return cls(paths=paths, expected=dict(expected or {}))

    def get(self, shard_id: str) -> tuple[np.ndarray, pack.DocIndex]:
        """Tokens and document index for a shard, verifying it the first time.

        Args:
            shard_id: Which shard.

        Returns:
            `(tokens, index)`.

        Raises:
            FileNotFoundError: If the shard is unknown or missing. A replay that silently skipped a
                missing shard would report on a subset of the run and call it the run.
        """
        if shard_id in self._cache:
            return self._cache[shard_id]

        path = self.paths.get(shard_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"shard {shard_id} is named by the ledger but was not found")
        tokens = np.asarray(shards.read(path))
        digest = shards.content_hash(tokens)
        if shard_id in self.expected and digest != self.expected[shard_id]:
            self.tampered[shard_id] = digest

        self._cache[shard_id] = (tokens, pack.DocIndex(tokens, shard_id=shard_id))
        return self._cache[shard_id]


def rebuild(event: ledger.ConsumeEvent, source: ShardSource) -> Rebuilt:
    """Re-materialise the microbatch an event describes.

    Every number comes from the event or from the shard bytes. Nothing is asked of the planner: the
    spans were written down at the time, and reading them is the point.

    The document offsets *are* re-derived, from the shard's own `EOS` positions. That is not
    recomputation of the plan — it is reading the data the event points at, which is exactly what a
    reader checking a citation does.

    Args:
        event: The ledger event.
        source: Where to read shards from.

    Returns:
        The rebuilt microbatch.

    Raises:
        ValueError: If the event records no samples, which would rebuild an empty batch and compare
            it favourably against nothing.
    """
    if not event.samples:
        raise ValueError(f"event seq {event.seq} records no samples; there is nothing to rebuild")

    # **Refuse a policy this code cannot rebuild, rather than rebuilding it the one way it knows.**
    #
    # Every reconstruction below is concat-and-chop with per-document positions and no context
    # mask. Fed an event produced under a different policy, it would happily rebuild the wrong
    # window, hash it, and report a mismatch — and a mismatch is the signal reserved for *a shard
    # whose bytes moved*. The report would blame the data for a difference in the reader.
    #
    # `replay_interval` catches this and records it as the verdict's `error`, so the report names
    # the policy instead of implying corruption.
    for name, value, implemented in (
        ("pack_policy", event.pack_policy, "concat-and-chop"),
        ("position_policy", event.position_policy, "restart-per-document-continue-across-window"),
        ("attention_policy", event.attention_policy, "block-diagonal-causal"),
    ):
        if value != implemented:
            raise ValueError(
                f"event seq {event.seq} was produced under {name}={value!r}; this replay only "
                f"rebuilds {implemented!r}. Refusing rather than reporting a hash mismatch that "
                f"would read as a tampered shard."
            )

    if event.loss_policy not in ("grade-all-but-document-final", "context-masked"):
        raise ValueError(
            f"event seq {event.seq} was produced under loss_policy={event.loss_policy!r}; this "
            f"replay cannot rebuild it. Refusing rather than reporting a hash mismatch."
        )
    if event.loss_policy == "context-masked" and not event.context_spans:
        raise ValueError(
            f"event seq {event.seq} claims loss_policy='context-masked' but records no context "
            f"spans. The mask it was graded under is unrecoverable, and rebuilding it as unmasked "
            f"would report a mismatch that reads as a tampered shard."
        )

    windows = sorted({sample.window for sample in event.samples})
    size = event.sequence_length
    tokens = np.full((len(windows), size), spec.PAD, dtype=np.int64)
    segments = np.full((len(windows), size), -1, dtype=np.int32)
    positions = np.zeros((len(windows), size), dtype=np.int32)
    loss = np.zeros((len(windows), size), dtype=bool)

    for row, window in enumerate(windows):
        fragments = [s for s in event.samples if s.window == window]
        lengths = [s.end - s.start for s in fragments]
        offsets = []
        at = 0
        for sample in fragments:
            stream, index = source.get(sample.shard_id)
            tokens[row, at : at + sample.end - sample.start] = stream[sample.start : sample.end]
            # The fragment's true offset inside its own document, read back from the shard.
            doc = index.document_at(sample.start)
            offsets.append(sample.start - index.document_start(doc))
            at += sample.end - sample.start

        segments[row] = masks.segment_ids(lengths, size)
        positions[row] = masks.position_ids(segments[row], offsets=offsets)
        # The spans come off the EVENT, not the manifest. Replay re-materialises from the shards
        # and the record alone; needing a second file to agree with would make it an audit of two
        # documents rather than of the run.
        excluded = [(start, end) for index, start, end in event.context_spans if index == window]
        loss[row] = masks.loss_mask(segments[row], tokens[row], context_spans=excluded or None)

    return Rebuilt(tokens=tokens, segments=segments, positions=positions, loss=loss)


@dataclass(frozen=True, slots=True)
class EventVerdict:
    """What replay found for one event.

    Attributes:
        step: Which optimizer step.
        rank: Which worker.
        accum: Which accumulation slot.
        flat: The run-wide address of the microbatch's first sequence.
        matched: Hash name to whether the re-derived value equals the recorded one.
        error: Set when the event could not be rebuilt at all.
    """

    step: int
    rank: int
    accum: int
    flat: int
    matched: dict[str, bool]
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether everything about this event checked out.

        Returns:
            True when there was no error and every hash matched.
        """
        return self.error is None and all(self.matched.values())


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """The result of replaying an interval.

    Attributes:
        branch_id: Which branch.
        interval: `[start, end)` in optimizer steps.
        verdicts: One per event, in run order.
        tampered: Shards whose bytes no longer match what was recorded.
    """

    branch_id: str
    interval: tuple[int, int]
    verdicts: tuple[EventVerdict, ...]
    tampered: dict[str, str]

    @property
    def checked(self) -> int:
        """How many events were replayed.

        Returns:
            The count.
        """
        return len(self.verdicts)

    @property
    def matched(self) -> int:
        """How many replayed cleanly.

        Returns:
            The count.
        """
        return sum(1 for v in self.verdicts if v.ok)

    @property
    def failures(self) -> tuple[EventVerdict, ...]:
        """The events that did not.

        Returns:
            The failing verdicts, in run order.
        """
        return tuple(v for v in self.verdicts if not v.ok)

    def summary(self) -> str:
        """One line, generated from the counts rather than written beside them.

        **It names a tampered shard even when every microbatch matched**, and that combination is
        not a contradiction: a shard whose bytes changed outside the spans this interval read
        produces a clean replay and a corrupt corpus. The report has always held both facts; the
        summary used to print only the first, so the line a reader quotes said `all match` while
        the object it came from knew a shard no longer hashed to its manifest.

        Returns:
            The summary.
        """
        start, end = self.interval
        state = "all match" if not self.failures else f"{len(self.failures)} MISMATCH"
        line = (
            f"{self.branch_id} steps [{start}, {end}): "
            f"{self.matched}/{self.checked} microbatches re-derived, {state}"
        )
        if self.tampered:
            names = ", ".join(sorted(self.tampered)[:3])
            where = "and they are why" if self.failures else "though none in this interval"
            line += f" — {len(self.tampered)} TAMPERED SHARD(S) [{names}], {where}"
        return line


def replay_interval(
    ledger_dir: Path,
    branch_id: str,
    start_step: int,
    end_step: int,
    source: ShardSource,
) -> ReplayReport:
    """Replay every microbatch in `[start_step, end_step)` and compare against the record.

    Args:
        ledger_dir: Where segment files live.
        branch_id: Which branch.
        start_step: First step, inclusive.
        end_step: One past the last.
        source: Where to read shards from.

    Returns:
        The report.

    Raises:
        ValueError: If the interval is empty, or the ledger's chain does not verify. A replay over a
            ledger that has been altered is a measurement of the alteration, not of the run.
    """
    if end_step <= start_step:
        raise ValueError(f"empty interval [{start_step}, {end_step})")

    for path in ledger.segments_for(ledger_dir, branch_id):
        ok, message = ledger.verify_chain(ledger.read_segment(path))
        if not ok:
            raise ValueError(f"{path.name}: {message}")

    verdicts: list[EventVerdict] = []
    for event in ledger.read_branch(ledger_dir, branch_id):
        if not start_step <= event.global_step < end_step:
            continue
        recorded = {
            "microbatch_hash": event.microbatch_hash,
            "loss_mask_hash": event.loss_mask_hash,
            "position_ids_hash": event.position_ids_hash,
            "segment_ids_hash": event.segment_ids_hash,
        }
        try:
            derived = rebuild(event, source).hashes()
        except (ValueError, FileNotFoundError, KeyError) as exc:
            verdicts.append(
                EventVerdict(
                    step=event.global_step,
                    rank=event.rank,
                    accum=event.accum,
                    flat=event.flat,
                    matched=dict.fromkeys(recorded, False),
                    error=str(exc),
                )
            )
            continue
        verdicts.append(
            EventVerdict(
                step=event.global_step,
                rank=event.rank,
                accum=event.accum,
                flat=event.flat,
                matched={name: derived[name] == value for name, value in recorded.items()},
            )
        )

    return ReplayReport(
        branch_id=branch_id,
        interval=(start_step, end_step),
        verdicts=tuple(verdicts),
        tampered=dict(source.tampered),
    )
