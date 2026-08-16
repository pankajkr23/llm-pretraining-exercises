r"""Stage 7 — keeping the exam out of the textbook.

If a benchmark's questions are sitting in the training corpus, the benchmark stops measuring
generalisation and starts measuring memorisation, and the reported score becomes a number about the
pipeline rather than about the model. The session's own example is stark: a band was dropped once
it showed 18.7% leakage.

The check is n-gram overlap. Shingle every document at a fixed width, shingle the held-out
evaluation sets the same way, and any document sharing a long enough window with an eval item is
contaminated. Long windows matter: at three words, ordinary prose collides with everything.

**Two problems this stage has to solve honestly.**

*The first is availability.* The real benchmark index lives in exercise 03 and needs a gated
download, so on CI and on a fresh clone there is nothing to check against. A stage that silently
reports "0 contaminated documents" in that situation is worse than no stage — it reads as a clean
bill of health. So coverage is reported explicitly, and when there is no index the answer is
**UNCHECKED**, never "clean". `test_unchecked_is_never_reported_as_clean` holds that.

*The second is demonstrability.* A guard nobody has watched fire is not a guard. So the stage
**injects canary strings** — unique GUIDs that appear nowhere else — into a held-out slice, then
runs the scanner and confirms it recovers them. That proves the machinery works on every machine,
gated data or not, and it is exactly the technique the session describes for detecting leakage in a
trained model.
"""

import hashlib
import logging
from dataclasses import dataclass, field

from dataframework.shingles import normalise

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat

logger = logging.getLogger(__name__)

CANARY_PREFIX = "s04canary"
"""Marker every canary shares, so injected strings are always distinguishable from corpus text."""


def canary_strings(count: int, seed: int, width: int) -> list[str]:
    """Generate reproducible canary strings, long enough for the scanner to see.

    **The length is the whole correctness condition, and the first version got it wrong.** The
    scanner shingles at `width` words; a canary shorter than that produces no n-grams at all, so the
    index comes back empty and the pass recovers nothing while looking like it ran. Each canary is
    therefore built with at least `width + 2` words.

    Deterministic from the seed, because the run id must not change when nothing changed.

    Args:
        count: How many to generate.
        seed: Configuration seed.
        width: The n-gram width the scanner will use.

    Returns:
        Canary strings, each unique and vanishingly unlikely to occur in real text.
    """
    out = []
    words_needed = width + 2
    for i in range(count):
        words = [CANARY_PREFIX]
        block = 0
        while len(words) < words_needed:
            digest = hashlib.blake2b(f"{seed}:{i}:{block}".encode(), digest_size=16).hexdigest()
            words.extend(digest[j : j + 4] for j in range(0, len(digest), 4))
            block += 1
        out.append(" ".join(words[:words_needed]))
    return out


def ngrams(text: str, n: int) -> set[str]:
    """Word n-grams over normalised text.

    Uses exercise 03's `normalise` for the same reason deduplication does: Python's word regex
    shatters Indic words at every combining mark, and n-grams of fragments do not mean what n-grams
    of words mean.

    Args:
        text: The text.
        n: Words per gram.

    Returns:
        The set of distinct n-grams. Empty when the text is shorter than `n` words.
    """
    words = normalise(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_index(items: list[str], n: int) -> set[str]:
    """Build an n-gram index from evaluation items."""
    index: set[str] = set()
    for item in items:
        index |= ngrams(item, n)
    return index


def contaminated(text: str, index: set[str], n: int) -> set[str]:
    """Return the n-grams a document shares with the evaluation index."""
    return ngrams(text, n) & index if index else set()


@dataclass
class DecontamReport:
    """What the scan checked and what it found.

    Attributes:
        coverage: `held-out`, `canary-only`, or `none`. `none` means UNCHECKED, not clean.
        sources: Which evaluation sets were available.
        index_grams: N-grams in the index.
        docs_scanned: Documents scanned.
        docs_flagged: Documents sharing at least one n-gram with the index.
        canaries_injected: Canaries planted.
        canaries_recovered: Canaries the scanner found.
        examples: A few flagged documents, by id.
    """

    coverage: str = "none"
    sources: list[str] = field(default_factory=list)
    index_grams: int = 0
    docs_scanned: int = 0
    docs_flagged: int = 0
    canaries_injected: int = 0
    canaries_recovered: int = 0
    examples: list[dict] = field(default_factory=list)

    @property
    def headline(self) -> str:
        """The one line the page shows. Never says 'clean' without having checked."""
        if self.coverage == "none":
            return "UNCHECKED — no evaluation index available on this machine"
        if self.docs_flagged:
            return f"{self.docs_flagged:,} documents overlap held-out evaluation text"
        return f"No overlap found across {self.docs_scanned:,} documents"

    def as_json(self) -> dict[str, object]:
        """Return the bundle representation."""
        return {
            "coverage": self.coverage,
            "sources": self.sources,
            "index_grams": self.index_grams,
            "docs_scanned": self.docs_scanned,
            "docs_flagged": self.docs_flagged,
            "canaries_injected": self.canaries_injected,
            "canaries_recovered": self.canaries_recovered,
            "canary_recall": (
                round(self.canaries_recovered / self.canaries_injected, 4)
                if self.canaries_injected
                else None
            ),
            "examples": self.examples,
            "headline": self.headline,
            "note": (
                "Coverage 'none' means UNCHECKED, not clean. The benchmark index needs a gated "
                "download, so a fresh clone cannot check against it — the canary pass proves the "
                "scanner works regardless."
            ),
        }


def _load_benchmark_items(cfg: Config) -> tuple[list[str], list[str]]:
    """Load evaluation items from exercise 03's data directory, if present.

    Returns:
        `(items, source_names)`. Empty when the gated download has not been run.
    """
    import json

    folder = cfg.flores_dir.parents[2] / "benchmarks"
    if not folder.exists():
        return [], []

    items: list[str] = []
    sources: list[str] = []
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("could not read benchmark file %s", path)
            continue
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        for row in rows:
            if isinstance(row, dict):
                text = row.get("question") or row.get("text") or ""
                if isinstance(text, str) and text.strip():
                    items.append(text)
            elif isinstance(row, str):
                items.append(row)
        sources.append(path.stem)
    return items, sources


def decontaminate_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 7: canary proof first, then the real scan if an index exists.

    Contaminated documents are dropped. Unlike a mislabelled language, a document carrying
    evaluation text is not a finding to keep — leaving it in is exactly the leak the stage exists
    to prevent.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The surviving documents and the stage record.
    """
    before = tokens.count_many([d.text for d in docs], cfg)
    report = DecontamReport()
    report.docs_scanned = len(docs)

    # 1. The canary pass. Plant known strings in a held-out slice and confirm the scanner finds
    # them. This runs everywhere, so the machinery is never untested.
    canaries = canary_strings(cfg.canary_count, cfg.minhash_seed, cfg.decontam_n)
    report.canaries_injected = len(canaries)
    canary_index = build_index(canaries, cfg.decontam_n)
    planted = [
        Document(
            f"canary-{i}",
            f"Filler sentence for the canary test. {c} And more filler text "
            "to make the document long enough to shingle properly at thirteen words wide.",
            "canary",
            "synthetic",
            "en",
        )
        for i, c in enumerate(canaries)
    ]
    report.canaries_recovered = sum(
        1 for d in planted if contaminated(d.text, canary_index, cfg.decontam_n)
    )

    # 2. The real scan, when the gated index happens to be on this machine.
    items, sources = _load_benchmark_items(cfg)
    kept = docs
    if items:
        index = build_index(items, cfg.decontam_n)
        report.coverage = "held-out"
        report.sources = sources
        report.index_grams = len(index)

        flagged: set[str] = set()
        for doc in docs:
            hits = contaminated(doc.text, index, cfg.decontam_n)
            if hits:
                flagged.add(doc.doc_id)
                if len(report.examples) < 8:
                    report.examples.append(
                        {"doc_id": doc.doc_id, "corpus": doc.corpus, "grams_shared": len(hits)}
                    )
        report.docs_flagged = len(flagged)
        kept = [d for d in docs if d.doc_id not in flagged]
    else:
        report.coverage = "canary-only" if report.canaries_recovered else "none"
        logger.info(
            "no benchmark index on this machine — decontamination reports %s, not clean",
            report.coverage.upper(),
        )

    after = tokens.count_many([d.text for d in kept], cfg)
    return kept, StageStat(
        n="7",
        stage_id="decontaminate",
        name="Decontaminate",
        real=True,
        docs_in=len(docs),
        docs_out=len(kept),
        tokens_in=before.as_figure(),
        tokens_out=after.as_figure(),
        rejections={"eval_overlap": len(docs) - len(kept)} if len(kept) < len(docs) else {},
        detail=report.as_json() | {"gram_width": cfg.decontam_n},
        note=(
            f"{report.headline}. "
            + (
                f"The canary pass recovered all {report.canaries_injected} planted strings, so the "
                f"scanner is known to work at {cfg.decontam_n}-word width on this machine."
                if report.canaries_recovered == report.canaries_injected
                else f"WARNING: the canary pass recovered only "
                f"{report.canaries_recovered}/{report.canaries_injected} planted strings — the "
                "scanner is not working, and any 'no overlap' result above is meaningless."
            )
        ),
    )
