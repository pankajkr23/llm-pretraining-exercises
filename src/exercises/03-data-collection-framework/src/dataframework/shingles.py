"""Hash benchmark text into 13-gram shingles for contamination detection (INV-1).

The gate this feeds is the most convincing artifact in the submission: plant a known MILU item in a
training shard, run the build, and watch CI fail and name the benchmark it collided with.

**Only hashes ever leave this module.** Shipping benchmark text — even to detect it later — would
itself be the contamination, and `web/` is a public bundle. Each shingle is a truncated blake2b
digest: enough to collide on identical 13-grams, useless for reconstructing the sentence.

Raw benchmark items live in git-ignored `data/benchmarks/` and are **not** in the repo (open item
B3). With no corpus present this module reports reduced coverage rather than silently producing an
empty index that would let every document through.
"""

import hashlib
import json
import re
from typing import Any

from .config import Config

SHINGLE_N = 13
DIGEST_BYTES = 8

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
        Truncated blake2b digests, one per n-gram. Text shorter than `n` words yields one shingle
        of the whole thing, so short benchmark items are still detectable.

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


def is_contaminated(document: str, index: set[str], n: int = SHINGLE_N) -> bool:
    """Whether a document collides with the eval registry at all.

    One 13-gram collision is enough: thirteen consecutive words matching by chance is vanishingly
    unlikely, so a single hit means the text was copied.

    Args:
        document: Candidate training text.
        index: Shingles from the eval registry.
        n: Shingle length.

    Returns:
        True if any shingle collides.
    """
    return bool(overlap(document, index, n))


def build_attributed_index(
    items_by_benchmark: dict[str, list[str]], n: int = SHINGLE_N
) -> dict[str, str]:
    """Build a shingle index that remembers which benchmark each hash came from.

    A gate that says "contaminated" is a shrug; one that says "this collides with MILU" tells you
    which score to distrust and which shard to pull. Attribution costs one string per shingle.

    Args:
        items_by_benchmark: Benchmark name to its eval item texts.
        n: Shingle length in words.

    Returns:
        Shingle digest to the benchmark that contributed it. On collision between benchmarks the
        first one wins, which is harmless — the point is to name a source, not all of them.
    """
    index: dict[str, str] = {}
    for benchmark, items in items_by_benchmark.items():
        for item in items:
            for digest in shingle(item, n):
                index.setdefault(digest, benchmark)
    return index


def find_collisions(
    document: str, attributed_index: dict[str, str], n: int = SHINGLE_N
) -> dict[str, int]:
    """Find which benchmarks a document collides with, and how hard.

    Args:
        document: Candidate training text.
        attributed_index: Output of `build_attributed_index`.
        n: Shingle length.

    Returns:
        Benchmark name to the number of colliding shingles; empty means clean.
    """
    hits: dict[str, int] = {}
    for digest in shingle(document, n):
        benchmark = attributed_index.get(digest)
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
            "note": (
                "No benchmark corpus in data/benchmarks/ (open item B3). Contamination is "
                "UNCHECKED — not clean. Supply raw items, or the MILU validation split as the "
                "documented fallback, before relying on the gate."
            ),
            "shingles": [],
        }

    index: set[str] = set()
    covered: list[str] = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        texts = items if isinstance(items, list) else items.get("items", [])
        for item in texts:
            index |= shingle(item if isinstance(item, str) else str(item.get("text", "")))
        covered.append(path.stem)

    return {
        "shingle_count": len(index),
        "benchmarks": covered,
        "coverage": "partial" if covered else "none",
        "note": (
            f"Index built from {len(covered)} benchmark file(s). Coverage is limited to these; "
            "any benchmark absent here is undetectable."
        ),
        "shingles": sorted(index),
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
    payload = {
        "shingle_count": index["shingle_count"],
        "benchmarks": index["benchmarks"],
        "coverage": index["coverage"],
        "note": index["note"],
        "shingles": index["shingles"],  # hashes only — never source text
    }
    (cfg.web_dir / "shingles.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return {key: value for key, value in index.items() if key != "shingles"}
