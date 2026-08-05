"""Hash benchmark text into 13-gram shingles for contamination detection (INV-1).

The gate this feeds is the most convincing artifact in the submission: plant a known MILU item in a
training shard, run the build, and watch CI fail and name the benchmark it collided with.

**Only hashes ever leave this module.** Shipping benchmark text — even to detect it later — would
itself be the contamination, and `web/` is a public bundle. Each shingle is a truncated blake2b
digest: enough to collide on identical 13-grams, useless for reconstructing the sentence.

Raw benchmark items live in git-ignored `data/benchmarks/` and are **not** in the repo (open item
B3). With no corpus present this module reports reduced coverage rather than silently producing an
empty index that would let every document through.

**Short items need a narrower window.** An eval item of seven words cannot be found by comparing
thirteen-grams: the item emits one seven-gram, a training shard emits only thirteen-grams, and the
two sets are structurally incapable of intersecting. Earlier versions of this module indexed short
items anyway and reported the resulting silence as "clean" — a false negative that appears exactly
when someone starts trusting the gate. `ShingleIndex` therefore records the window width each item
was indexed at, and `find_collisions` shingles the document once per width in play. When every item
is at least `SHINGLE_N` words — the common case — there is one width and the cost is unchanged.

Below `MIN_SHINGLE_N` words an item is not indexed at all, because a two-word window matches
ordinary prose everywhere and a gate that cries wolf gets switched off. Those items are counted in
`ShingleIndex.unindexable` and surfaced in the coverage report, so the gap is a number somebody can
read rather than a silence.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config

SHINGLE_N = 13
DIGEST_BYTES = 8

# Narrower than this and a window stops identifying anything: "what is the capital of" occurs in
# ordinary text. Items below the floor are reported as uncovered rather than indexed.
MIN_SHINGLE_N = 5

_WORD = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> list[str]:
    """Reduce text to comparable word tokens.

    Args:
        text: Raw text.

    Returns:
        Lowercased word tokens.
    """
    return _WORD.findall(text.lower())


def shingle(text: str, n: int = SHINGLE_N) -> set[str]:
    """Hash a text into n-gram shingles.

    Args:
        text: Raw text.
        n: Shingle length in words.

    Returns:
        Truncated blake2b digests, one per n-gram. Text shorter than `n` words yields a single
        shingle of the whole thing — which collides only if the other side is shingled at that
        same width. Comparing a short item against a long document at `n` therefore finds nothing;
        use `ShingleIndex` and `find_collisions`, which handle the widths for you.

    Raises:
        ValueError: If `n` is not positive.
    """
    if n <= 0:
        raise ValueError("shingle length must be positive")
    words = normalise(text)
    if not words:
        return set()
    grams = (
        [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(words) >= n
        else [" ".join(words)]
    )
    return {hashlib.blake2b(g.encode("utf-8"), digest_size=DIGEST_BYTES).hexdigest() for g in grams}


def overlap(document: str, index: set[str], n: int = SHINGLE_N) -> set[str]:
    """Find shingles a document shares with an eval index.

    Args:
        document: Candidate training text.
        index: Shingles from the eval registry.
        n: Shingle length.

    Returns:
        The colliding shingles; empty means clean.
    """
    return shingle(document, n) & index


def gram_width(text: str, n: int = SHINGLE_N) -> int:
    """The window width an item is indexed at.

    Args:
        text: The eval item.
        n: Preferred shingle length.

    Returns:
        `n`, or the item's own word count when it is shorter than `n`.
    """
    return min(len(normalise(text)), n)


@dataclass(frozen=True)
class ShingleIndex:
    """An eval registry that knows which window widths it was built at.

    Attributes:
        grams: Shingle digest to the benchmark that contributed it. On collision between
            benchmarks the first one wins, which is harmless — the point is to name a source, not
            all of them.
        widths: Every window width present in `grams`. A document must be shingled at each of
            them, or items indexed at the missing widths are invisible.
        unindexable: Benchmark to the count of items too short to index at all. Not a detail:
            these are eval items the gate cannot protect, and the number belongs in the coverage
            report.
    """

    grams: dict[str, str] = field(default_factory=dict)
    widths: frozenset[int] = frozenset()
    unindexable: dict[str, int] = field(default_factory=dict)


def is_contaminated(document: str, index: "ShingleIndex | set[str]", n: int = SHINGLE_N) -> bool:
    """Whether a document collides with the eval registry at all.

    One collision is enough: thirteen consecutive words matching by chance is vanishingly
    unlikely, so a single hit means the text was copied.

    Args:
        document: Candidate training text.
        index: A `ShingleIndex`, or a bare digest set to compare at a single width.
        n: Shingle length, used only for the bare-set form.

    Returns:
        True if any shingle collides.
    """
    if isinstance(index, ShingleIndex):
        return bool(find_collisions(document, index))
    return bool(overlap(document, index, n))


def build_attributed_index(
    items_by_benchmark: dict[str, list[str]], n: int = SHINGLE_N
) -> ShingleIndex:
    """Build a shingle index that remembers which benchmark each hash came from.

    A gate that says "contaminated" is a shrug; one that says "this collides with MILU" tells you
    which score to distrust and which shard to pull. Attribution costs one string per shingle.

    Items shorter than `n` are indexed at their own width so they remain findable inside a longer
    document; items shorter than `MIN_SHINGLE_N` are refused and counted instead.

    Args:
        items_by_benchmark: Benchmark name to its eval item texts.
        n: Preferred shingle length in words.

    Returns:
        A `ShingleIndex` carrying the digests, the widths in play, and the refused items.
    """
    grams: dict[str, str] = {}
    widths: set[int] = set()
    unindexable: dict[str, int] = {}
    for benchmark, items in items_by_benchmark.items():
        for item in items:
            width = gram_width(item, n)
            if width < MIN_SHINGLE_N:
                unindexable[benchmark] = unindexable.get(benchmark, 0) + 1
                continue
            widths.add(width)
            for digest in shingle(item, width):
                grams.setdefault(digest, benchmark)
    return ShingleIndex(grams=grams, widths=frozenset(widths), unindexable=unindexable)


def find_collisions(document: str, index: ShingleIndex) -> dict[str, int]:
    """Find which benchmarks a document collides with, and how hard.

    The document is shingled once per width the index holds, so a seven-word item is looked for
    with a seven-word window while a full-length item still uses thirteen.

    Args:
        document: Candidate training text.
        index: Output of `build_attributed_index`.

    Returns:
        Benchmark name to the number of colliding shingles; empty means clean.
    """
    hits: dict[str, int] = {}
    for width in sorted(index.widths):
        for digest in shingle(document, width):
            benchmark = index.grams.get(digest)
            if benchmark is not None:
                hits[benchmark] = hits.get(benchmark, 0) + 1
    return hits


def build_index(cfg: Config | None = None) -> dict[str, Any]:
    """Build the shingle index from the git-ignored benchmark corpus.

    Args:
        cfg: Paths to use; defaults to `Config()`.

    Returns:
        The index plus a coverage report. When no corpus is present, `coverage` is `"none"` and
        `benchmarks` is empty — a caller must treat that as "cannot certify", never as "clean".
    """
    cfg = cfg or Config()
    corpus_dir = cfg.data_dir / "benchmarks"

    if not corpus_dir.exists():
        return {
            "shingle_count": 0,
            "benchmarks": [],
            "coverage": "none",
            "gram_widths": [],
            "unindexable_items": 0,
            "note": (
                "No benchmark corpus in data/benchmarks/ (open item B3). Contamination is "
                "UNCHECKED — not clean. Supply raw items, or the MILU validation split as the "
                "documented fallback, before relying on the gate."
            ),
            "shingles": [],
        }

    items_by_benchmark: dict[str, list[str]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        texts = items if isinstance(items, list) else items.get("items", [])
        items_by_benchmark[path.stem] = [
            item if isinstance(item, str) else str(item.get("text", "")) for item in texts
        ]

    index = build_attributed_index(items_by_benchmark)
    covered = list(items_by_benchmark)
    refused = sum(index.unindexable.values())
    note = (
        f"Index built from {len(covered)} benchmark file(s) at window widths "
        f"{sorted(index.widths) or '[]'}. Coverage is limited to these; any benchmark absent here "
        "is undetectable."
    )
    if refused:
        note += (
            f" {refused} item(s) are shorter than {MIN_SHINGLE_N} words and are NOT indexed — "
            "too short to identify anything, so the gate cannot protect them."
        )

    return {
        "shingle_count": len(index.grams),
        "benchmarks": covered,
        "coverage": "partial" if covered else "none",
        "gram_widths": sorted(index.widths),
        "unindexable_items": refused,
        "note": note,
        "shingles": sorted(index.grams),
    }


def write_index(cfg: Config | None = None) -> dict[str, Any]:
    """Build the index and write the hashes-only bundle for the web build.

    Args:
        cfg: Paths to use; defaults to `Config()`.

    Returns:
        The index metadata (without the hashes).
    """
    cfg = cfg or Config()
    index = build_index(cfg)
    cfg.web_dir.mkdir(parents=True, exist_ok=True)

    # The digests are a build artifact, not a browser asset. MILU's validation split alone produces
    # 411,442 of them — 9.6 MB that no page ever fetches, because the pages only ever report the
    # count. They stay in the git-ignored data directory, where CI and a local run can use them.
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "shingle_index.json").write_text(
        json.dumps({"gram_widths": index["gram_widths"], "shingles": index["shingles"]}),
        encoding="utf-8",
    )

    payload = {
        "shingle_count": index["shingle_count"],
        "benchmarks": index["benchmarks"],
        "coverage": index["coverage"],
        "gram_widths": index["gram_widths"],
        "unindexable_items": index["unindexable_items"],
        "note": index["note"],
    }
    (cfg.web_dir / "shingles.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return {key: value for key, value in index.items() if key != "shingles"}
