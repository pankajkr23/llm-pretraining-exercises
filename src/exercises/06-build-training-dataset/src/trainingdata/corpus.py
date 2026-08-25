"""Fetched text becomes sealed, admitted shards — with a lineage the gate can check.

**The problem.** `manifest.admit` refuses a shard whose `cleaning_hash`, `dedup_hash`, `pii_hash`
or `eval_overlap_hash` is `None`, because *an unanswered question is not a pass*. Something has to
answer them, and answering them with a constant would make the gate unfalsifiable — a guard that
cannot fail, which this repo treats as worse than no guard at all.

So this module runs the **real** exercise-04 stages — dedup, PII, decontamination — over the
fetched documents and hashes what each one actually produced. If deduplication removes nothing, the
hash still changes when the input does; if it removes something, the token count changes with it.
Neither number can be written down in advance.

**Three shapes the rest of the system requires, each of which is silently wrong if missed.**

*Documents are separated by `EOS` inside the flat stream.* There is no side file and no per-document
index — `DocIndex` finds boundaries with `np.flatnonzero(tokens == EOS)`. A stream written without
them is not an error: it indexes as **one document**, which reinstates exactly the cross-document
attention the block-diagonal mask exists to prevent.

*The held-out split is taken at a document boundary, before tokenising.* Splitting after would put
a token from one side into the other, and the two would no longer be disjoint in the only sense
that matters.

*A shard shorter than one sequence is dropped, loudly.* `build_span_table` discards such a shard in
silence, so its tokens would be counted in the manifest, counted in the mixture, and never trained
on — a lane could report its full budget while feeding less.

Torch-free, like everything else on the data path.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import manifest as manifest_module
from . import shards, spec
from .config import Config

logger = logging.getLogger(__name__)

#: Tokens per shard when building from a proxy-scale corpus.
#:
#: `Config.tokens_per_shard` is 5,000,000 — sized for a real run, and larger than any single lane
#: here. Using it would give every lane exactly one shard, which makes the shard set degenerate:
#: the plan's permutation would have nothing to interleave, `shard_set_hash` would barely vary, and
#: a tampered-shard drill would turn an entire lane red rather than a few batches.
PROXY_TOKENS_PER_SHARD: int = 200_000


@dataclass(frozen=True)
class LaneText:
    """One lane's fetched text and the provenance the manifest needs.

    Attributes:
        lane: Lane key.
        path: The fetched file, one document per line.
        licence: Verified at fetch time from the source, not from a catalogue.
        language: Primary language of the text.
        provenance_tier: How close this is to what the specification funds the lane from.
        dataset: Where it came from.
    """

    lane: str
    path: Path
    licence: str
    language: str
    provenance_tier: str
    dataset: str


@dataclass
class LaneBuild:
    """What building one lane produced.

    Attributes:
        lane: Lane key.
        documents_in: Documents read from the fetched file.
        documents_kept: Documents surviving the cleaning stages.
        train_tokens: Tokens written to training shards.
        heldout_tokens: Tokens withheld for evaluation.
        shard_ids: The shards written, in order.
        dropped_short: Pieces discarded for being shorter than one sequence.
        unk_share: Share of training ids that came back `[UNK]`.
        stage_removals: What each cleaning stage removed.
    """

    lane: str
    documents_in: int = 0
    documents_kept: int = 0
    train_tokens: int = 0
    heldout_tokens: int = 0
    shard_ids: list[str] = field(default_factory=list)
    dropped_short: int = 0
    unk_share: float = 0.0
    stage_removals: dict[str, int] = field(default_factory=dict)


def read_documents(path: Path) -> list[str]:
    """The fetched text — one JSON-encoded document per line.

    **JSONL, not newline-joined text, and the difference is not cosmetic.** Documents contain
    newlines: measured on a real fetch, 2,174 FineWeb articles read back as 47,456 "documents"
    under a plain `splitlines()`. Every paragraph would get its own `EOS`, and the block-diagonal
    mask would then wall off paragraphs of the same article from each other — which is precisely
    the boundary the mask exists to draw, drawn in the wrong place, with every count still looking
    plausible.

    Args:
        path: The lane's `.jsonl` file.

    Returns:
        Non-empty documents, in file order.

    Raises:
        ValueError: If a line is not a JSON string. A bare line would be a newline-joined file, and
            reading it as documents would silently reintroduce the bug above.
    """
    if not path.is_file():
        return []
    documents = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} line {number} is not JSON. This file must be JSONL — one encoded "
                f"document per line — because documents contain newlines and a plain text file "
                f"cannot say where one ends: {exc}"
            ) from exc
        if not isinstance(document, str):
            raise ValueError(
                f"{path.name} line {number} is {type(document).__name__}, not a string"
            )
        if document.strip():
            documents.append(document)
    return documents


def clean(documents: list[str], lane: str) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Run the real exercise-04 stages and hash what each actually produced.

    Not a placeholder, and that is the point: `admit` refuses a `None` lineage hash because an
    unanswered question is not a pass, and a constant would answer it without checking anything.

    Args:
        documents: The fetched text.
        lane: Lane key, used as the corpus label the stages group by.

    Returns:
        `(surviving documents, {cleaning,dedup,pii,eval_overlap}_hash, removals per stage)`.

    Raises:
        RuntimeError: If every document is removed, which means the corpus is unusable rather than
            merely clean.
    """
    from datacleaning import decontaminate, dedup, pii
    from datacleaning.config import Config as CleaningConfig
    from datacleaning.manifest import corpus_hash, script_hash
    from datacleaning.records import Document

    config = CleaningConfig()
    docs = [
        Document(
            doc_id=f"{lane}-{index:07d}",
            text=text,
            corpus=lane,
            shard=lane,
            claimed_lang="und",
        )
        for index, text in enumerate(documents)
    ]

    removals: dict[str, int] = {}
    hashes: dict[str, str] = {"cleaning_hash": script_hash()}

    for name, stage in (
        ("dedup_hash", dedup.dedup_stage),
        ("pii_hash", pii.pii_stage),
        ("eval_overlap_hash", decontaminate.decontaminate_stage),
    ):
        before = len(docs)
        docs, _ = stage(docs, config)
        removals[name.removesuffix("_hash")] = before - len(docs)
        # Hash the stage's OUTPUT, so the answer changes when the corpus does. A hash of the
        # stage's name would satisfy the gate and check nothing.
        hashes[name] = corpus_hash(docs)

    if not docs:
        raise RuntimeError(f"{lane}: every document was removed by cleaning; nothing to train on")
    return [doc.text for doc in docs], hashes, removals


def _encode(documents: list[str], tokenizer) -> np.ndarray:
    """Tokenise documents into one flat EOS-separated stream.

    Args:
        documents: The text.
        tokenizer: The frozen tokenizer.

    Returns:
        A flat `int64` array of ids, each document followed by `spec.EOS`.
    """
    ids: list[int] = []
    for document in documents:
        ids.extend(tokenizer.encode(document).ids)
        ids.append(spec.EOS)
    return np.asarray(ids, dtype=np.int64)


def build_lane(
    text: LaneText,
    out_dir: Path,
    config: Config,
    tokenizer,
    *,
    tokenizer_sha256: str,
    tokens_per_shard: int = PROXY_TOKENS_PER_SHARD,
) -> LaneBuild:
    """Clean, split, tokenise, shard and manifest one lane.

    Args:
        text: The lane's fetched text and provenance.
        out_dir: Where shard directories go; this lane gets `out_dir / lane`.
        config: The run shape, for `heldout_share` and the fingerprint.
        tokenizer: The frozen tokenizer.
        tokenizer_sha256: Digest of the tokenizer file, so the ids have a defined meaning.
        tokens_per_shard: Shard size.

    Returns:
        What was built.
    """
    documents = read_documents(text.path)
    result = LaneBuild(lane=text.lane, documents_in=len(documents))
    if not documents:
        logger.warning("%s: no text at %s", text.lane, text.path)
        return result

    kept, hashes, removals = clean(documents, text.lane)
    result.documents_kept = len(kept)
    result.stage_removals = removals

    # Split at a DOCUMENT boundary, before tokenising: a token from one side landing in the other
    # would make the two overlap in the only sense that matters.
    cut = max(1, int(round(len(kept) * (1.0 - config.heldout_share))))
    train_docs, heldout_docs = kept[:cut], kept[cut:]

    train = _encode(train_docs, tokenizer)
    heldout = _encode(heldout_docs, tokenizer)
    result.train_tokens = int(train.size)
    result.heldout_tokens = int(heldout.size)
    result.unk_share = float((train == 0).mean()) if train.size else 0.0

    lane_dir = out_dir / text.lane
    for piece in shards.split(train, tokens_per_shard):
        if piece.size < config.sequence_length:
            # `build_span_table` discards it in silence, so its tokens would be counted in the
            # manifest and in the mixture and never trained on.
            result.dropped_short += 1
            logger.warning(
                "%s: dropping a %d-token tail, shorter than one %d-token sequence",
                text.lane,
                piece.size,
                config.sequence_length,
            )
            continue

        shard_id, path = shards.write(piece, lane_dir)
        entry = manifest_module.ShardManifest(
            shard_id=shard_id,
            content_hash=shards.content_hash(piece),
            token_count=int(piece.size),
            dtype=shards.DTYPE.str,
            source=str(text.path.relative_to(text.path.parents[2]))
            if len(text.path.parents) > 2
            else str(text.path),
            lane=text.lane,
            language=text.language,
            licence=text.licence,
            provenance_tier=text.provenance_tier,
            tokenizer_id=config.tokenizer_id,
            tokenizer_sha256=tokenizer_sha256,
            split="train",
            config_fingerprint=config.fingerprint(),
            unk_share=float((piece == 0).mean()),
            **hashes,
        )
        manifest_module.append(entry, lane_dir)
        result.shard_ids.append(shard_id)
        logger.info("%s: shard %s, %d tokens -> %s", text.lane, shard_id, piece.size, path.name)

    return result


def lanes_from_fetch(corpus_dir: Path) -> list[LaneText]:
    """Read the fetcher's manifest and describe each lane it wrote.

    The provenance comes from the fetch, not from a second table here: the licence was verified
    against the dataset's own card at download time, and re-declaring it in this module would let
    the two drift — with the manifest confidently recording a licence nobody checked.

    Args:
        corpus_dir: Where the fetcher wrote.

    Returns:
        One entry per lane that has text on disk.

    Raises:
        FileNotFoundError: If the fetch manifest is absent. Building from loose files whose
            provenance nobody recorded is how an unlicensed corpus gets trained on.
    """
    path = corpus_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no fetch manifest at {path}. Run tools/fetch_corpus.py first — building from loose "
            f"files would produce shards whose licence and provenance nobody recorded."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))

    found: list[LaneText] = []
    for entry in payload["lanes"]:
        lane = entry["lane"]
        text_path = corpus_dir / f"{lane}.jsonl"
        if not text_path.is_file():
            logger.warning("%s: manifest lists the lane but %s is absent", lane, text_path.name)
            continue
        sources = entry.get("sources") or []
        if not sources:
            logger.warning("%s: no source recorded; skipping rather than guessing", lane)
            continue
        # A lane may be fed by several sources (indic is three Sangraha splits). They share a
        # licence by construction — the fetcher refuses anything outside PERMISSIVE — so the first
        # is representative, and the manifest keeps the full list either way.
        first = sources[0]
        found.append(
            LaneText(
                lane=lane,
                path=text_path,
                licence=first.get("licence") or first.get("license") or "",
                language=first.get("language", "und"),
                provenance_tier=first.get("provenance_tier", "C"),
                dataset=first.get("dataset", "unknown"),
            )
        )
    return found


def build(
    corpus_dir: Path,
    out_dir: Path,
    config: Config,
    tokenizer,
    *,
    tokenizer_sha256: str,
    tokens_per_shard: int = PROXY_TOKENS_PER_SHARD,
) -> dict[str, LaneBuild]:
    """Build every fetched lane into sealed, manifested shards.

    Args:
        corpus_dir: Where the fetcher wrote.
        out_dir: Where shard directories go.
        config: The run shape.
        tokenizer: The frozen tokenizer.
        tokenizer_sha256: Digest of the tokenizer file.
        tokens_per_shard: Shard size.

    Returns:
        Lane name to what was built.
    """
    return {
        text.lane: build_lane(
            text,
            out_dir,
            config,
            tokenizer,
            tokenizer_sha256=tokenizer_sha256,
            tokens_per_shard=tokens_per_shard,
        )
        for text in lanes_from_fetch(corpus_dir)
    }
