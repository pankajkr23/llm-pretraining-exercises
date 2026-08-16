"""The eight stages, composed.

Every stage has the same shape — documents and a config in, documents and a `StageStat` out — so
the descent is a fold and the page renders `stages[]` without re-deriving anything.

**Right now seven of the eight stages are counting pass-throughs.** That is deliberate. Landing the
skeleton first means `python -m datacleaning` produces a valid bundle from the first commit, the
yield-chain invariant is live before there is anything to break it, and each later change replaces
exactly one no-op with the real thing. The alternative — building stages in isolation and wiring
them at the end — ships three changes that cannot produce the artifact anyone reviews.

A pass-through is honest about itself: `StageStat.real` is False and its note says so, so a stage
that has not been written yet cannot be mistaken for a stage that found nothing.

The stage list is the session's, and the numbering is the session's too — including `2b` for format
discipline, which the pipeline map never numbers even though §14 counts it. See `BRIEF.md` §D1.
"""

import logging
import time
from collections.abc import Callable

from datacleaning import corpus, manifest, tokens
from datacleaning.config import Config
from datacleaning.records import Document, PipelineResult, StageStat

logger = logging.getLogger(__name__)

StageFn = Callable[[list[Document], Config], tuple[list[Document], StageStat]]


STAGES: tuple[tuple[str, str, str, str], ...] = (
    ("1", "extract", "Extract", "Session 3's topic — Sangraha and the rest ship extracted text."),
    ("2", "normalize", "Normalize", "Unicode, invisibles, entities, whitespace — joiners kept."),
    ("2b", "formats", "Format discipline", "Ghost tags are created by rendering, not inherited."),
    ("3", "langid", "Language ID", "Detect the language; never trust the folder it came from."),
    ("4", "quality", "Quality filter", "Nine Gopher/C4 rules at the session's thresholds."),
    ("5", "dedup", "Deduplicate", "Exact hashes, then MinHash/LSH for near-duplicates."),
    ("6", "pii", "PII scrub", "Structured identifiers by regex; names by a declared stand-in."),
    ("7", "decontaminate", "Decontaminate", "Canaries and n-grams against held-out evaluation."),
    ("8", "manifest", "Manifest", "Provenance, hashes, and a run id that proves determinism."),
)
"""`(number, id, name, one-line description)` for each stage, in pipeline order.

Nine entries for eight strategies: `1 Extract` is inherited from Session 3 and `2b` is the
never-numbered ninth. Both are rendered, neither is double-counted — the count is argued in
`BRIEF.md` §D1.
"""


def _passthrough(stage_id: str) -> StageFn:
    """Build a stage that counts documents and changes nothing.

    Args:
        stage_id: Which stage this stands in for.

    Returns:
        A stage function that reports `real=False`.
    """

    def run(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
        counts = tokens.count_many([d.text for d in docs], cfg)
        number, _, name, _ = next(s for s in STAGES if s[1] == stage_id)
        return docs, StageStat(
            n=number,
            stage_id=stage_id,
            name=name,
            real=False,
            docs_in=len(docs),
            docs_out=len(docs),
            tokens_in=counts.as_figure(),
            tokens_out=counts.as_figure(),
            note="Not implemented yet — counts only, changes nothing.",
        )

    return run


# Stages replace their pass-through here as each lands. `extract` is permanently a pass-through:
# every corpus ships extracted text, so claiming a yield for it would be inventing one.
IMPLEMENTED: dict[str, StageFn] = {}


def stage_fn(stage_id: str) -> StageFn:
    """Return the real stage if it exists, otherwise its counting pass-through."""
    return IMPLEMENTED.get(stage_id, _passthrough(stage_id))


def run(cfg: Config | None = None) -> PipelineResult:
    """Load the corpora and fold every stage over them.

    Args:
        cfg: Configuration; defaults apply.

    Returns:
        The stage records, the surviving documents, the manifest, and per-stage findings.
    """
    cfg = cfg or Config()
    logger.info("profile %s — loading corpora", cfg.profile)

    started = time.perf_counter()
    loaded = corpus.load(cfg)
    docs = loaded.documents
    logger.info("loaded %d documents in %.1fs", len(docs), time.perf_counter() - started)

    stages: list[StageStat] = []
    extras: dict[str, object] = {}

    for _, stage_id, _, _ in STAGES:
        if stage_id == "manifest":
            continue
        t0 = time.perf_counter()
        docs, stat = stage_fn(stage_id)(docs, cfg)
        stat = StageStat(
            n=stat.n,
            stage_id=stat.stage_id,
            name=stat.name,
            real=stat.real,
            docs_in=stat.docs_in,
            docs_out=stat.docs_out,
            tokens_in=stat.tokens_in,
            tokens_out=stat.tokens_out,
            rejections=stat.rejections,
            detail=stat.detail,
            runtime_s=time.perf_counter() - t0,
            note=stat.note,
        )
        stages.append(stat)
        logger.info(
            "  stage %-3s %-18s %7d -> %7d docs  %s",
            stat.n,
            stat.stage_id,
            stat.docs_in,
            stat.docs_out,
            "" if stat.real else "(pass-through)",
        )

    selections = [s.as_json() for s in loaded.selections]
    record = manifest.build(cfg, docs, stages, selections)

    # Stage 8 is the manifest, and it reports as a real stage because it genuinely produced
    # something: hashes, a run id, and the determinism claim. It drops no documents.
    counts = tokens.count_many([d.text for d in docs], cfg)
    stages.append(
        StageStat(
            n="8",
            stage_id="manifest",
            name="Manifest",
            real=True,
            docs_in=len(docs),
            docs_out=len(docs),
            tokens_in=counts.as_figure(),
            tokens_out=counts.as_figure(),
            detail={
                "run_id": record["run_id"],
                "content_hash": record["content_hash"],
                "script_hash": record["script_hash"],
                "config_hash": record["config_hash"],
            },
            note="Provenance stamped; the run id is derived from content, not the clock.",
        )
    )

    extras["selections"] = selections
    extras["tokenizer_spread"] = tokens.spread_table()
    extras["unreadable"] = tokens.unreadable_languages(cfg)
    extras["strategies"] = [
        {"n": n, "id": sid, "name": name, "summary": summary} for n, sid, name, summary in STAGES
    ]

    return PipelineResult(
        run_id=str(record["run_id"]),
        profile=cfg.profile,
        stages=stages,
        docs=docs,
        manifest=record,
        extras=extras,
    )


def yield_descent(stages: list[StageStat]) -> dict[str, object]:
    """Build the yield descent the page renders, so it is never re-derived in JavaScript.

    Args:
        stages: Every stage record, in order.

    Returns:
        Labels and parallel document/token/share series, plus the session's illustrative descent
        for comparison.
    """
    labels = [s.name for s in stages]
    docs = [s.docs_out for s in stages]
    toks = [s.tokens_out.value for s in stages]
    first = next((t for t in toks if t), None)
    share = [round(t / first, 4) if (t and first) else None for t in toks]
    return {
        "labels": labels,
        "docs": docs,
        "tokens": toks,
        "share": share,
        "real": [s.real for s in stages],
        "session_illustrative": [100, 92, 88, 61, 44, 43, 42, 42],
        "note": (
            "The session's descent is illustrative and covers eight stages; ours is measured and "
            "covers nine rows, because format discipline is counted separately."
        ),
    }
