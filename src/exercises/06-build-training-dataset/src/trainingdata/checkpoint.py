"""Checkpoints — weights, optimizer state, and the ledger cut that goes with them.

**The problem.** A run dies. On restart you must not re-feed data the model already learned from,
nor skip data it did not. Weights alone cannot answer that: they say what the model *is*, never what
it *read*. And the kill lands where it lands — *"you give the kill command at 3,000 and it might get
killed at 3,005"* — so the two facts drift apart at exactly the moment you need them to agree.

**The strategy — a checkpoint records a position in the data, not only a position in the loss
curve.** On resume each rank's ledger is truncated to its own entry; anything past it belongs to
work these weights do not contain.

**The position is a vector, and it is worth being exact about why.** Four ranks wrote four separate
files, so the cut must be *applied* per rank whatever its values are. At a synchronous checkpoint
those four values coincide, and in this system today they do — every rank writes the same number of
events per step and the barrier lands them together. What is already ragged is how much each rank
wrote *after* the checkpoint before it died, which is what `resume.plan_resume` computes per rank
and why a resume cannot use one number either. A scalar cut would be correct only for as long as
every rank writes the same number of events per step; per-rank selection breaks that the moment a
rank rejects a candidate, and so does any rank-local retry. The structure is there because the
invariant it depends on is not one this system should be relying on.

**Two files, and the order they are written in is a commit protocol.**

`<id>.pt` holds the tensors and needs torch. `<id>.json` holds everything else — the cut, the plan
key, the environment — and is deliberately **torch-free readable**, because `verify.py` audits from
artifacts alone and must never need the producer's dependencies to do it.

The `.pt` is written and renamed into place *first*; the `.json` last. So the JSON's existence is
the commit: a checkpoint interrupted halfway leaves a `.pt` with no `.json` and is simply not found,
rather than being found and restored from a truncated tensor file.

**Every write is rename-into-place.** `torch.save` straight to the final path leaves a corrupt file
if the process dies during it — and the file it corrupts is the one thing that could have recovered
the run.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Schema version for the JSON sidecar.
VERSION = 1


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """What a checkpoint knows, minus the tensors.

    Attributes:
        v: Schema version.
        checkpoint_id: Stable id, derived from the run and step rather than from a clock.
        run_id: Which run.
        branch_id: Which branch.
        attempt: Which attempt wrote it.
        step: The last optimizer step whose update these weights include.
        cut: Rank to ledger length. **A vector, not a scalar.**
        segments: Rank to the segment number its cut refers to.
        weight_digest: What the weights hashed to when saved.
        plan_digest: The plan these weights were trained under.
        config_fingerprint: The settings they were trained under.
        environment: Device, threads and library versions.
    """

    v: int
    checkpoint_id: str
    run_id: str
    branch_id: str
    attempt: int
    step: int
    cut: dict[int, int]
    segments: dict[int, int]
    weight_digest: str
    plan_digest: str
    config_fingerprint: str
    environment: dict = field(default_factory=dict)

    @property
    def total_microbatches(self) -> int:
        """Microbatches consumed across every rank up to this checkpoint.

        Returns:
            The sum of the cut vector.
        """
        return sum(self.cut.values())

    def to_json(self) -> str:
        """Serialise the sidecar.

        Returns:
            Indented, sorted JSON.
        """
        payload = asdict(self)
        # JSON object keys are strings; the ranks come back as strings unless converted on read.
        payload["cut"] = {str(k): v for k, v in sorted(self.cut.items())}
        payload["segments"] = {str(k): v for k, v in sorted(self.segments.items())}
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Checkpoint":
        """Rebuild a checkpoint's metadata.

        Args:
            text: The sidecar's contents.

        Returns:
            The checkpoint.

        Raises:
            ValueError: If the schema version is not one this code understands.
        """
        payload = json.loads(text)
        if payload.get("v") != VERSION:
            raise ValueError(
                f"checkpoint sidecar has schema v{payload.get('v')}, this code reads v{VERSION}"
            )
        payload["cut"] = {int(k): int(v) for k, v in payload["cut"].items()}
        payload["segments"] = {int(k): int(v) for k, v in payload["segments"].items()}
        return cls(**payload)


def checkpoint_id(run_id: str, branch_id: str, step: int) -> str:
    """A stable id for a checkpoint.

    Derived from the run, branch and step — never from a clock, so re-running a branch produces the
    same id and two artifacts claiming the same checkpoint can be compared rather than merely
    counted.

    Args:
        run_id: Which run.
        branch_id: Which branch.
        step: The step it follows.

    Returns:
        `"ckpt-<branch>-<step:06d>"`.
    """
    del run_id  # part of the directory, not of the id; kept in the signature for call-site clarity
    return f"ckpt-{branch_id}-{step:06d}"


def sidecar_path(directory: Path, identifier: str) -> Path:
    """Where a checkpoint's torch-free metadata lives.

    Args:
        directory: The checkpoint directory.
        identifier: The checkpoint id.

    Returns:
        The path.
    """
    return directory / f"{identifier}.json"


def tensor_path(directory: Path, identifier: str) -> Path:
    """Where a checkpoint's tensors live.

    Args:
        directory: The checkpoint directory.
        identifier: The checkpoint id.

    Returns:
        The path.
    """
    return directory / f"{identifier}.pt"


def write_atomically(path: Path, write) -> Path:
    """Write via a temporary file and rename into place.

    A direct write leaves a corrupt file if the process dies during it — and here the file it would
    corrupt is the only thing that could have recovered the run. `os.replace` is atomic on both
    macOS and Linux, so a reader sees either the old file or the new one and never a half of either.

    Args:
        path: Final destination.
        write: Callable taking the temporary `Path` and writing to it.

    Returns:
        The final path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".partial")
    write(staging)
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, path)
    return path


def digest_file(path: Path) -> str:
    """Content hash of a file on disk.

    Args:
        path: The file.

    Returns:
        `"sha256:<64 hex>"`.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load(directory: Path, identifier: str) -> Checkpoint:
    """Read a checkpoint's metadata without torch.

    Args:
        directory: The checkpoint directory.
        identifier: The checkpoint id.

    Returns:
        The checkpoint.

    Raises:
        FileNotFoundError: If the sidecar is absent, which is how an interrupted save presents.
    """
    path = sidecar_path(directory, identifier)
    if not path.is_file():
        raise FileNotFoundError(
            f"no sidecar at {path}. A checkpoint interrupted mid-save leaves its tensors with no "
            f"sidecar and is deliberately not found, rather than found and half-restored."
        )
    return Checkpoint.from_json(path.read_text(encoding="utf-8"))


def latest(
    directory: Path, branch_id: str, *, at_or_before: int | None = None
) -> Checkpoint | None:
    """The most recent complete checkpoint of a branch.

    Args:
        directory: The checkpoint directory.
        branch_id: Which branch.
        at_or_before: Ignore checkpoints after this step. Used by fork, which starts from a point in
            the past rather than from the end.

    Returns:
        The checkpoint, or None when the branch has none.
    """
    if not directory.is_dir():
        return None
    found = [
        Checkpoint.from_json(path.read_text(encoding="utf-8"))
        for path in directory.glob(f"ckpt-{branch_id}-*.json")
    ]
    usable = [c for c in found if at_or_before is None or c.step <= at_or_before]
    return max(usable, key=lambda c: c.step, default=None)
