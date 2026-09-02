r"""Stage 5 — deduplication, the stage the source material says this corpus never had.

Two passes, because they catch different things:

1. **Exact.** A sha256 over the *cleaned* text. Catches byte-identical reposts. Cheap, and it only
   works because stage 2 ran first — hashing raw text gives two documents differing by a zero-width
   space two different hashes.
2. **Near-duplicate.** Shingle each document into overlapping k-word windows, reduce each set to a
   fixed-length MinHash signature, and band the signature so that similar documents collide in at
   least one band. Catches the rewritten repost, the same article with a different header, the page
   that differs only in its navigation.

The parameters are FineWeb's: `k=5`, 112 permutations arranged as 14 bands of 8. The banding
approximation puts the similarity threshold at `(1/b)**(1/r)` = **0.719** — the source material
quotes this
preset as "target ~0.75", and we report what the arithmetic actually gives rather than the quoted
figure.

**Word tokenization comes from exercise 03, deliberately.** `dataframework.shingles.normalise`
carries correction X16: Python's `\\w+` splits an Indic word at every combining mark, so `भारत`
becomes several fragments. Shingles built from fragments compare fragment-overlap rather than
word-overlap, and the whole stage would be quietly, plausibly wrong. Reusing the function is
cheaper than re-finding the bug.

The MinHash is vectorised because it has to be: 112 permutations over ~50k documents of ~500
shingles each is on the order of 10^9 operations, which is minutes in numpy and hours in a Python
loop.
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from dataframework.shingles import normalise

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat

logger = logging.getLogger(__name__)

MERSENNE = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1


def _stable_hash(window: str) -> int:
    """Hash one shingle to a 64-bit int, reproducibly across processes.

    **Not** Python's built-in `hash()`. String hashing is randomised per interpreter unless
    `PYTHONHASHSEED` is pinned, so the same corpus would bucket differently on every run and the
    candidate pairs — and therefore the documents deleted — would drift. A pipeline whose
    deduplication is not reproducible cannot claim reproducibility at all.

    Args:
        window: The joined k-word window.

    Returns:
        A 64-bit integer.
    """
    return int.from_bytes(hashlib.blake2b(window.encode("utf-8"), digest_size=8).digest(), "big")


def shingles(text: str, k: int) -> set[int]:
    """Reduce a document to a set of hashed k-word windows.

    Args:
        text: Document text.
        k: Words per window.

    Returns:
        64-bit hashes of each distinct window. Hashes rather than strings because the MinHash only
        needs identity, and a set of ints is far smaller than a set of joined strings.
    """
    words = normalise(text)
    if not words:
        return set()
    if len(words) < k:
        # Too short to shingle at this width. Treated as one window rather than an empty set, which
        # would otherwise make every short document identical to every other short document.
        return {_stable_hash(" ".join(words))}
    return {_stable_hash(" ".join(words[i : i + k])) for i in range(len(words) - k + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    """True Jaccard similarity, `|A ∩ B| / |A ∪ B|`."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def lsh_threshold(bands: int, rows: int) -> float:
    """The similarity at which a pair becomes roughly even odds to be a candidate."""
    return (1.0 / bands) ** (1.0 / rows)


def p_candidate(similarity: float, bands: int, rows: int) -> float:
    """Probability that a pair at this similarity collides in at least one band.

    The S-curve `1 - (1 - s**r)**b`. Its steepness is why banding works at all: pairs above the
    threshold are almost certain to be caught and pairs below it almost certain to be missed.
    """
    return 1.0 - (1.0 - similarity**rows) ** bands


def _permutations(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return the `(a, b)` coefficients for the hash family, seeded for reproducibility."""
    rng = np.random.default_rng(cfg.minhash_seed)
    n = cfg.minhash_permutations
    a = rng.integers(1, MERSENNE - 1, size=n, dtype=np.uint64)
    b = rng.integers(0, MERSENNE - 1, size=n, dtype=np.uint64)
    return a, b


def signature(sh: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a MinHash signature for one shingle set.

    Each slot holds the minimum of `(a*x + b) mod p` over every shingle. The probability that two
    documents share a slot equals their Jaccard similarity, which is the whole trick: an expensive
    set comparison becomes a cheap vector comparison.

    Args:
        sh: The document's shingle hashes.
        a: Multiplier coefficients.
        b: Offset coefficients.

    Returns:
        A `uint64` array of length `len(a)`. An empty document yields all-maximum.
    """
    if not sh:
        return np.full(len(a), np.uint64(MERSENNE), dtype=np.uint64)
    x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
    # (len(shingles), n_perm) — the vectorisation that turns hours into seconds.
    hashed = (np.outer(x, a) + b) % np.uint64(MERSENNE)
    return hashed.min(axis=0)


def band_keys(sig: np.ndarray, bands: int, rows: int) -> list[bytes]:
    """Split a signature into `bands` chunks of `rows` and hash each into a bucket key."""
    return [
        hashlib.blake2b(sig[i * rows : (i + 1) * rows].tobytes(), digest_size=16).digest()
        for i in range(bands)
    ]


@dataclass
class DedupReport:
    """What the two passes removed.

    Attributes:
        exact_removed: Documents dropped as byte-identical after cleaning.
        exact_clusters: Distinct groups of identical documents.
        near_removed: Documents dropped as near-duplicates.
        near_clusters: Distinct near-duplicate groups.
        candidate_pairs: Pairs LSH proposed.
        verified_pairs: Pairs that survived the true-Jaccard check.
        cluster_sizes: Cluster size -> how many clusters of that size.
        largest_cluster: Documents in the biggest cluster.
    """

    exact_removed: int = 0
    exact_clusters: int = 0
    near_removed: int = 0
    near_clusters: int = 0
    candidate_pairs: int = 0
    verified_pairs: int = 0
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    largest_cluster: int = 0


def _exact_pass(docs: list[Document], cfg: Config) -> tuple[list[Document], DedupReport]:
    """Drop byte-identical documents, keeping the first occurrence of each."""
    report = DedupReport()
    seen: dict[str, int] = {}
    kept: list[Document] = []

    for doc in docs:
        digest = hashlib.new(cfg.hash_algo, doc.text.encode("utf-8")).hexdigest()
        if digest in seen:
            seen[digest] += 1
            report.exact_removed += 1
            continue
        seen[digest] = 1
        kept.append(doc)

    duplicated = [n for n in seen.values() if n > 1]
    report.exact_clusters = len(duplicated)
    report.largest_cluster = max(duplicated, default=0)
    return kept, report


def _near_pass(
    docs: list[Document], cfg: Config, report: DedupReport
) -> tuple[list[Document], list[dict]]:
    """Drop near-duplicates found by MinHash/LSH, keeping one document per cluster."""
    a, b = _permutations(cfg)
    sigs: dict[str, np.ndarray] = {}
    shs: dict[str, set[int]] = {}

    for doc in docs:
        sh = shingles(doc.text, cfg.shingle_k)
        shs[doc.doc_id] = sh
        sigs[doc.doc_id] = signature(sh, a, b)

    # Bucket by band. Any pair sharing a bucket in any band is a candidate.
    buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for doc_id, sig in sigs.items():
        for band, key in enumerate(band_keys(sig, cfg.bands, cfg.rows_per_band)):
            buckets[(band, key)].append(doc_id)

    candidates: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                candidates.add((left, right) if left < right else (right, left))

    report.candidate_pairs = len(candidates)
    threshold = lsh_threshold(cfg.bands, cfg.rows_per_band)

    # LSH proposes; the true Jaccard decides. Banding is a recall device, not a verdict — skipping
    # this check is how a dedup pass starts deleting documents that merely share boilerplate.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    merged: set[str] = set()
    examples: list[dict] = []
    for left, right in sorted(candidates):
        similarity = jaccard(shs[left], shs[right])
        if similarity < threshold:
            if len(examples) < 60 and similarity > 0.05:
                examples.append(
                    {"a": left, "b": right, "jaccard": round(similarity, 4), "verdict": "kept"}
                )
            continue
        report.verified_pairs += 1
        if len(examples) < 60:
            examples.append(
                {"a": left, "b": right, "jaccard": round(similarity, 4), "verdict": "duplicate"}
            )
        merged.update((left, right))
        ra, rb = find(left), find(right)
        if ra != rb:
            parent[ra] = rb

    # Cluster over every document that appeared in a confirmed pair. Checking `parent` membership
    # instead would miss documents that are only ever a union-find *root*, silently splitting
    # clusters and under-reporting how much duplication there is.
    clusters: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(merged):
        clusters[find(doc_id)].append(doc_id)

    drop: set[str] = set()
    sizes: dict[int, int] = defaultdict(int)
    for members in clusters.values():
        if len(members) < 2:
            continue
        sizes[len(members)] += 1
        report.near_clusters += 1
        report.largest_cluster = max(report.largest_cluster, len(members))
        drop.update(sorted(members)[1:])

    report.near_removed = len(drop)
    report.cluster_sizes = dict(sorted(sizes.items()))
    return [d for d in docs if d.doc_id not in drop], examples


def sweep(docs: list[Document], cfg: Config, settings: tuple[tuple[int, int], ...]) -> list[dict]:
    """Report what different `(bands, rows)` choices would remove.

    The threshold is not a setting, it is a decision about what you keep — so the page shows the
    curve rather than one point. A coarse sweep does not report roughly the optimum, it reports the
    wrong one, so the settings are contiguous in `rows`.

    Args:
        docs: Documents to sweep over.
        cfg: Configuration.
        settings: `(bands, rows)` pairs to try.

    Returns:
        One row per setting.
    """
    from dataclasses import replace

    rows = []
    for bands, per_band in settings:
        variant = replace(cfg, bands=bands, rows_per_band=per_band)
        report = DedupReport()
        _, _ = _near_pass(list(docs), variant, report)
        rows.append(
            {
                "bands": bands,
                "rows": per_band,
                "permutations": bands * per_band,
                "threshold": round(lsh_threshold(bands, per_band), 4),
                "docs_removed": report.near_removed,
                "candidate_pairs": report.candidate_pairs,
                "verified_pairs": report.verified_pairs,
            }
        )
    return rows


def dedup_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 5 over a corpus: exact pass, then near-duplicate pass.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The surviving documents and the stage record.
    """
    before = tokens.count_many([d.text for d in docs], cfg)

    after_exact, report = _exact_pass(docs, cfg)
    kept, examples = _near_pass(after_exact, cfg, report)

    by_corpus: dict[str, dict[str, int]] = {}
    surviving = {d.doc_id for d in kept}
    for doc in docs:
        row = by_corpus.setdefault(doc.corpus, {"in": 0, "out": 0})
        row["in"] += 1
        row["out"] += doc.doc_id in surviving

    duplicates = [e for e in examples if e["verdict"] == "duplicate"]
    near_misses = [e for e in examples if e["verdict"] == "kept"]

    after = tokens.count_many([d.text for d in kept], cfg)
    removed = len(docs) - len(kept)

    return kept, StageStat(
        n="5",
        stage_id="dedup",
        name="Deduplicate",
        real=True,
        docs_in=len(docs),
        docs_out=len(kept),
        tokens_in=before.as_figure(),
        tokens_out=after.as_figure(),
        rejections={"exact_duplicate": report.exact_removed, "near_duplicate": report.near_removed},
        detail={
            "params": {
                "shingle_k": cfg.shingle_k,
                "bands": cfg.bands,
                "rows_per_band": cfg.rows_per_band,
                "permutations": cfg.minhash_permutations,
                "threshold": round(lsh_threshold(cfg.bands, cfg.rows_per_band), 4),
                "preset": "FineWeb (k=5, 112 = 14x8)",
                "quoted_target": 0.75,
                "note": (
                    "The source quotes this preset as target ~0.75; the banding approximation "
                    "gives 0.719. We publish what the code computes."
                ),
            },
            "exact": {
                "docs_removed": report.exact_removed,
                "clusters": report.exact_clusters,
            },
            "near": {
                "docs_removed": report.near_removed,
                "clusters": report.near_clusters,
                "candidate_pairs": report.candidate_pairs,
                "verified_pairs": report.verified_pairs,
                "false_candidate_pairs": report.candidate_pairs - report.verified_pairs,
            },
            "cluster_sizes": report.cluster_sizes,
            "largest_cluster": report.largest_cluster,
            "by_corpus": by_corpus,
            "example_duplicates": duplicates[:12],
            "example_near_misses": near_misses[:12],
        },
        note=(
            f"Removed {removed:,} documents: {report.exact_removed:,} byte-identical and "
            f"{report.near_removed:,} near-duplicates at a similarity threshold of "
            f"{lsh_threshold(cfg.bands, cfg.rows_per_band):.3f}. LSH proposed "
            f"{report.candidate_pairs:,} candidate pairs and the true Jaccard confirmed "
            f"{report.verified_pairs:,} — banding is a recall device, not a verdict."
        ),
    )
