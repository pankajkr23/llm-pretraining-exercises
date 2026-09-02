"""Shard manifests, and the gate that decides what may enter training.

**The problem.** A sealed shard proves its bytes have not changed. It says nothing about whether
those bytes are *allowed* into a training batch — whether they were deduplicated, screened for
personal information, checked against the evaluation sets, or licensed for this use at all.

**The strategy.** Every shard carries a manifest, and admission is a function of the manifest
rather than a habit of the caller. The stated minimum is blunt: no shard is
trained on unless it carries the cleaning hashes — dedup, eval and PII.

So the gate refuses on a **missing** hash, not only on a failing one. An unanswered question is not
a pass — which is the same rule exercise 03 enforces on its dataset grades, arrived at
independently.

Manifests are stored as JSONL, appended, never rewritten: a shard that has been admitted is a fact
about the past, and editing the record of it is how a run stops being auditable.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

#: A shard may be trained on, held out for evaluation during the run, or reserved for the
#: benchmark suite and never read by training at all.
Split = Literal["train", "heldout", "eval"]

#: The three hashes the source names as the minimum for admission. Absent means refused.
REQUIRED_HASHES = ("dedup_hash", "pii_hash", "eval_overlap_hash")

#: Filename of the append-only manifest log inside a shard directory.
MANIFEST_FILE = "manifests.jsonl"


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """What a training system needs to know about a shard before it consumes one.

    The field list follows the source's, with the names it uses. Several are `str | None` on
    purpose: `None` means *nobody has answered this*, which the gate treats as a refusal rather
    than as a pass. A field that could only ever hold a truthy value would make the gate
    unfalsifiable.
    """

    # -- identity ---------------------------------------------------------------------------
    shard_id: str
    content_hash: str
    token_count: int
    dtype: str

    # -- provenance -------------------------------------------------------------------------
    source: str
    lane: str
    language: str
    licence: str
    provenance_tier: str

    # -- the tokenizer that gives the ids meaning -------------------------------------------
    tokenizer_id: str
    tokenizer_sha256: str

    # -- admission ---------------------------------------------------------------------------
    #: How the raw text became admitted training data. `None` = unknown lineage.
    cleaning_hash: str | None = None
    #: Deduplication status. `None` = nobody has deduplicated this.
    dedup_hash: str | None = None
    #: PII screening. `None` = nobody has screened it.
    pii_hash: str | None = None
    #: Checked against the evaluation registry. `None` = nobody has checked.
    eval_overlap_hash: str | None = None

    # -- placement in the run ----------------------------------------------------------------
    split: Split = "train"
    #: Held back for the annealing phase rather than spent during the main run.
    anneal_reserve: bool = False
    #: Benchmarks this shard is known to overlap. Non-empty means it can never be trained on.
    benchmark_ids: tuple[str, ...] = ()
    #: Shards this one was derived from. A shard cut from another keeps the lineage.
    parent_shard_ids: tuple[str, ...] = ()

    #: Half-open `[start, end)` ranges, SHARD-relative, that provide context but earn no loss — an
    #: instruction, a question, a tool observation.
    #:
    #: They live here because a shard is a flat token stream with no header and no side file: the
    #: manifest is the only place that already travels with it and is already mandatory. They are
    #: **token** offsets computed by tokenising the document's parts separately, never a character
    #: index mapped afterwards — 10.8% of separator sites are absorbed into a longer BPE token, so
    #: at those sites no token boundary exists to map to.
    context_spans: tuple[tuple[int, int], ...] = ()

    #: The run configuration under which it was built.
    config_fingerprint: str = ""
    #: Share of tokens that came back `[UNK]`. A count that is mostly unknown is not a count.
    unk_share: float = 0.0

    def as_json(self) -> dict[str, object]:
        """The manifest as a JSON-serialisable dict.

        Returns:
            Field names to values, with tuples flattened to lists.
        """
        out = asdict(self)
        out["benchmark_ids"] = list(self.benchmark_ids)
        out["parent_shard_ids"] = list(self.parent_shard_ids)
        out["context_spans"] = [list(span) for span in self.context_spans]
        return out

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ShardManifest":
        """Rebuild a manifest from its JSON form.

        Args:
            payload: A dict as produced by `as_json`.

        Returns:
            The manifest.
        """
        data = dict(payload)
        data["benchmark_ids"] = tuple(data.get("benchmark_ids") or ())
        data["context_spans"] = tuple(
            (int(start), int(end)) for start, end in data.get("context_spans") or ()
        )
        data["parent_shard_ids"] = tuple(data.get("parent_shard_ids") or ())
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why a shard was not admitted.

    Carries the reason as well as the verdict, because a firewall that only says "no" cannot be
    debugged and cannot be audited — the same rule exercise 03 applies to its gate verdicts.
    """

    shard_id: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        """A refusal is truthy when there is anything to refuse.

        Returns:
            True when at least one reason was recorded.
        """
        return bool(self.reasons)


def admit(manifest: ShardManifest, *, max_unk_share: float = 0.05) -> Refusal:
    """Decide whether a shard may enter a loss-bearing training batch.

    Every refusal is a separate reason rather than a first-failure short-circuit, so one call
    reports everything wrong with a shard instead of one thing at a time.

    Args:
        manifest: The shard's manifest.
        max_unk_share: The share of `[UNK]` tokens above which the token count is not trustworthy.
            Mirrors exercise 04's publication gate.

    Returns:
        A `Refusal`. Falsy when the shard is admitted.
    """
    reasons: list[str] = []

    if manifest.split != "train":
        reasons.append(f"split is {manifest.split!r}, which is never loss-bearing")

    if manifest.benchmark_ids:
        reasons.append(
            f"overlaps benchmarks {list(manifest.benchmark_ids)} — training on it would make "
            f"those scores meaningless"
        )

    for name in REQUIRED_HASHES:
        if getattr(manifest, name) is None:
            reasons.append(f"{name} is missing — an unanswered question is not a pass")

    if manifest.cleaning_hash is None:
        reasons.append("cleaning_hash is missing — the shard has unknown lineage")

    if not manifest.tokenizer_sha256:
        reasons.append("tokenizer_sha256 is empty — the token ids have no defined meaning")

    if manifest.unk_share > max_unk_share:
        reasons.append(
            f"unk_share {manifest.unk_share:.1%} exceeds {max_unk_share:.0%} — the token count is "
            f"not trustworthy enough to budget with"
        )

    if manifest.token_count <= 0:
        reasons.append("token_count is not positive")

    return Refusal(manifest.shard_id, tuple(reasons))


def append(manifest: ShardManifest, directory: Path) -> Path:
    """Append a manifest to the directory's log.

    Append-only. A shard that has been built is a fact about the past; rewriting the record of it
    is how a run stops being auditable.

    Args:
        manifest: The manifest to record.
        directory: The shard directory.

    Returns:
        The manifest log path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILE
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest.as_json(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_all(directory: Path) -> list[ShardManifest]:
    """Every manifest in the directory's log, in the order it was written.

    Args:
        directory: The shard directory.

    Returns:
        The manifests. Empty when the log does not exist.
    """
    path = directory / MANIFEST_FILE
    if not path.is_file():
        return []
    return [
        ShardManifest.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def trainable(manifests: list[ShardManifest]) -> list[ShardManifest]:
    """The shards that pass the gate.

    Args:
        manifests: Candidates.

    Returns:
        Those admitted, in input order.
    """
    return [m for m in manifests if not admit(m)]
