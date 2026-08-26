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

#: What the fetcher joins a document's parts with, and what the builder must rejoin them with so a
#: cleaned document can be compared byte-for-byte against what went in.
PART_SEPARATOR: str = "\n\n"


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
        heldout_tokens: Tokens withheld from training.
        heldout_shard_ids: The shards those tokens were written to. They exist on disk: a count
            without data is a claim nothing can check, and this one went unchecked for a while.
        shard_ids: The shards written, in order.
        dropped_short: Pieces discarded for being shorter than one sequence.
        unk_share: Share of training ids that came back `[UNK]`.
        context_documents: Documents carrying a context span — a prompt that conditions the model
            without earning loss.
        stage_removals: What each cleaning stage removed.
    """

    lane: str
    documents_in: int = 0
    documents_kept: int = 0
    train_tokens: int = 0
    heldout_tokens: int = 0
    #: Shards written for the held-out split. Empty when the lane withheld less than one window.
    heldout_shard_ids: list[str] = field(default_factory=list)
    shard_ids: list[str] = field(default_factory=list)
    dropped_short: int = 0
    unk_share: float = 0.0
    context_documents: int = 0
    stage_removals: dict[str, int] = field(default_factory=dict)


def read_documents(path: Path) -> list[list[str]]:
    """The fetched text — one JSON-encoded document per line, as its list of PARTS.

    A bare JSON string is a single-part document; a JSON array is a structured one, and by
    convention **everything but the last part is context**. `[problem, solution]` means the problem
    conditions the model and only the solution earns loss.

    **JSONL, not newline-joined text, and the difference is not cosmetic.** Documents contain
    newlines. Measured on real fetches: 2,174 FineWeb articles read back as 47,456 "documents"
    under a plain `splitlines()`, and the code lane is far worse — 775 Python files carry
    **155,778 newlines between them**, so every file would have shattered into its individual
    lines, a 200x inflation. Each fragment would then get its own `EOS`, and the block-diagonal
    mask would wall off consecutive lines of the *same function* from each other — precisely the
    boundary the mask exists to draw, drawn in the wrong place, with every count still plausible.

    Args:
        path: The lane's `.jsonl` file.

    Returns:
        Non-empty documents, in file order, each as its list of parts.

    Raises:
        ValueError: If a line is not a JSON string or array of strings. A bare line would be a
            newline-joined file, and reading it as documents would reintroduce the bug above.
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
        if isinstance(document, str):
            parts = [document]
        elif isinstance(document, list) and all(isinstance(part, str) for part in document):
            parts = [part for part in document if part.strip()]
        else:
            raise ValueError(
                f"{path.name} line {number} is {type(document).__name__}, not a string or a list "
                f"of strings"
            )
        if any(part.strip() for part in parts):
            documents.append(parts)
    return documents


def clean(
    documents: list[list[str]], lane: str
) -> tuple[list[tuple[int, str, bool]], dict[str, str], dict[str, int]]:
    """Run the real exercise-04 stages and hash what each actually produced.

    Not a placeholder, and that is the point: `admit` refuses a `None` lineage hash because an
    unanswered question is not a pass, and a constant would answer it without checking anything.

    Args:
        documents: The fetched text.
        lane: Lane key, used as the corpus label the stages group by.

    Returns:
        `(survivors as (original index, cleaned text, boundary intact) triples, the four lineage
        hashes, removals per stage)`.

    Raises:
        RuntimeError: If every document is removed, which means the corpus is unusable rather than
            merely clean.
    """
    from datacleaning import decontaminate, dedup, pii
    from datacleaning.config import Config as CleaningConfig
    from datacleaning.manifest import corpus_hash, script_hash
    from datacleaning.records import Document

    config = CleaningConfig()
    # The stages take whole documents: deduplication and decontamination are claims about a
    # document, not about half of one. So the parts are joined for cleaning and the boundary is
    # recovered afterwards.
    joined = {
        f"{lane}-{index:07d}": PART_SEPARATOR.join(parts) for index, parts in enumerate(documents)
    }
    docs = [
        Document(doc_id=doc_id, text=text, corpus=lane, shard=lane, claimed_lang="und")
        for doc_id, text in joined.items()
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

    # **Whether the boundary survived cleaning is a fact, not an assumption.**
    #
    # PII scrubbing rewrites text in place, so a document it touched no longer splits where its
    # parts said it did — the prompt's character length has moved and any span derived from it
    # would grade the wrong tokens. Rather than trust that scrubbing is rare (measured: 0 removals
    # on every lane, which says nothing about substitutions), each surviving document reports
    # whether its text is byte-identical to what went in. One that changed keeps no context span
    # and is graded in full, which is the safe direction: too much loss, never loss on the wrong
    # half.
    return (
        [
            (int(doc.doc_id.rsplit("-", 1)[1]), doc.text, doc.text == joined.get(doc.doc_id))
            for doc in docs
        ],
        hashes,
        removals,
    )


def _flatten(encoded: list[list[int]]) -> np.ndarray:
    """One flat EOS-separated stream from per-document id lists.

    Args:
        encoded: Token ids per document.

    Returns:
        A flat `int64` array, each document followed by `spec.EOS`.
    """
    ids: list[int] = []
    for document in encoded:
        ids.extend(document)
        ids.append(spec.EOS)
    return np.asarray(ids, dtype=np.int64)


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


def _write_heldout(
    heldout: np.ndarray,
    out_dir: Path,
    text: "LaneText",
    config: Config,
    tokenizer_sha256: str,
    hashes: dict[str, str],
) -> list[str]:
    """Materialise the held-out split, instead of counting it and throwing it away.

    **This existed as a number and not as data, which is the worst of both.** `heldout_tokens` was
    computed here, recorded in `LaneBuild`, summed into the build report and published — while the
    array itself went out of scope one line later. A tenth of the corpus was reported as withheld
    for evaluation and did not exist anywhere on disk, so nothing could have been evaluated on it
    and no test failed, because every test asked about the number.

    It surfaced when OPUS needed a proxy set. `g_proxy` is the direction the run would like to move
    in, and scoring against training data selects for whatever the model is already being pushed
    toward — so the reference has to be text the run never trains on. The selector asked for the
    held-out split and found an empty lane.

    **Written with `split="heldout"`, which `manifest.admit` refuses**, so the firewall keeps it out
    of every loss-bearing batch by the same rule that stops benchmark data. It carries no
    `benchmark_ids`: it is not a benchmark, it is a reference sample, and conflating the two would
    make the firewall's benchmark reason fire for something that overlaps no benchmark.

    Args:
        heldout: The withheld tokens for this lane.
        out_dir: Where lane directories go.
        text: The lane being built.
        config: The run shape.
        tokenizer_sha256: Provenance of what the token ids mean.
        hashes: The cleaning lineage this lane's documents carry, so a held-out shard is traceable
            to the same pipeline its training siblings came from.

    Returns:
        The shard ids written, which is empty when the lane withheld less than one sequence.
    """
    lane_dir = out_dir / "heldout"
    written: list[str] = []
    for piece in shards.split(heldout, config.tokens_per_shard):
        if piece.size < config.sequence_length:
            continue  # shorter than one window; it could never be read
        shard_id, _ = shards.write(piece, lane_dir)
        manifest_module.append(
            manifest_module.ShardManifest(
                context_spans=(),
                shard_id=shard_id,
                content_hash=shards.content_hash(piece),
                token_count=int(piece.size),
                dtype=shards.DTYPE.str,
                source=f"{text.lane}:heldout",
                lane="heldout",
                language=text.language,
                licence=text.licence,
                provenance_tier=text.provenance_tier,
                tokenizer_id=config.tokenizer_id,
                tokenizer_sha256=tokenizer_sha256,
                split="heldout",
                config_fingerprint=config.fingerprint(),
                unk_share=float((piece == 0).mean()),
                **hashes,
            ),
            lane_dir,
        )
        written.append(shard_id)
    return written


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

    # Split at a document boundary, by TOKENS rather than by document count.
    #
    # Counting documents looks equivalent and is not: document sizes are wildly skewed on some
    # lanes — the code lane's longest file is 282,355 characters — so the last 10% of documents can
    # be far more than 10% of the tokens. Measured on the first real build, a 10% document split
    # withheld **16.1%** of the code lane's tokens, which pushed that lane 1.59 points below its
    # planned share and put the whole mixture out of compliance. The boundary is still a document
    # boundary; only the thing being counted changed.
    encoded, context_lengths = [], []
    for index, cleaned, intact in kept:
        parts = documents[index]
        if intact and len(parts) > 1:
            # **Tokenise the parts SEPARATELY and concatenate.** That makes the boundary exact by
            # construction. Encoding the joined text and then splitting at a character index does
            # not: measured on the frozen BPE, 10.8% of separator sites are absorbed into a longer
            # token, so at those sites no token boundary exists to split at, and 4.7% of prompts do
            # not tokenise to a prefix of the whole document. Measured cost of splitting: -0.14%.
            head = tokenizer.encode(PART_SEPARATOR.join(parts[:-1]) + PART_SEPARATOR).ids
            tail = tokenizer.encode(parts[-1]).ids
            encoded.append(head + tail)
            context_lengths.append(len(head))
        else:
            encoded.append(tokenizer.encode(cleaned).ids)
            context_lengths.append(0)

    result.context_documents = sum(1 for length in context_lengths if length)
    total = sum(len(ids) + 1 for ids in encoded)  # +1 for the EOS each document is terminated with
    want_train = total * (1.0 - config.heldout_share)

    running, cut = 0, len(encoded)
    for index, ids in enumerate(encoded):
        running += len(ids) + 1
        if running >= want_train:
            cut = index + 1
            break
    cut = min(max(cut, 1), len(encoded))  # never withhold everything, never withhold nothing

    train = _flatten(encoded[:cut])
    heldout = _flatten(encoded[cut:])

    # Where each training document's context ends, in LANE-stream coordinates. Converted to
    # shard-relative below, because a shard is the unit anything downstream can address.
    lane_spans: list[tuple[int, int]] = []
    at = 0
    for ids, context in zip(encoded[:cut], context_lengths[:cut], strict=True):
        if context:
            lane_spans.append((at, at + context))
        at += len(ids) + 1
    result.train_tokens = int(train.size)
    result.heldout_tokens = int(heldout.size)
    result.heldout_shard_ids = _write_heldout(
        heldout, out_dir, text, config, tokenizer_sha256, hashes
    )
    result.unk_share = float((train == 0).mean()) if train.size else 0.0

    lane_dir = out_dir / text.lane
    written_tokens = 0
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
            written_tokens += int(piece.size)
            continue

        shard_id, path = shards.write(piece, lane_dir)
        offset = written_tokens
        within = tuple(
            (max(start, offset) - offset, min(end, offset + int(piece.size)) - offset)
            for start, end in lane_spans
            if start < offset + int(piece.size) and end > offset
        )
        written_tokens += int(piece.size)
        entry = manifest_module.ShardManifest(
            context_spans=within,
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
