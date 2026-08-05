"""Map capabilities to the benchmarks that can measure them — and expose the holes.

A capability with no instrument is a goal you cannot tell whether you hit. The interesting output
of this module is therefore not the matrix but its **empty cells**: the Atlas's own observation is
that "India-context worldview" — arguably the whole point of the model — has almost no column.

Coverage is deliberately about *capabilities*, not datasets. Whether a tier of data is detectable
at all is a different question, answered by `orphans.py`.
"""

from typing import Any

# The capabilities this model is being built for, as (name, include, exclude).
#
# The exclusions are not fussiness — a false positive *hides a hole*, and the holes are the whole
# point of this matrix. Two traps in this corpus specifically: "code-mixed" means Hindi-English
# switching, not programming; and a bare "mt" matches inside HMMT, a maths contest.
CAPABILITIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("indic-language", ("indic", "indian", "bhasha"), ()),
    (
        "india-context",
        ("india-context", "india context", "cultural", "worldview", "sanskriti"),
        ("translated from",),
    ),
    ("code", ("code", "programming", "swe-bench", "humaneval"), ("code-mix", "code-switch")),
    ("agentic-coding", ("agentic", "swe-bench", "terminal-bench", "tool use"), ()),
    ("math-reasoning", ("math", "reasoning", "gsm", "stem", "aime"), ()),
    ("knowledge", ("knowledge", "mmlu", "question answering", " qa"), ()),
    ("translation", ("translation", "machine translation", "parallel", "bitext"), ()),
    ("speech", ("speech", "asr", "audio", "spoken"), ()),
    ("multimodal", ("multimodal", "vision", "image", "ocr", "document"), ()),
    ("safety", ("safety", "harm", "jailbreak", "toxic"), ()),
    ("long-context", ("long context", "long-context"), ()),
)


def capabilities_for(benchmark: dict[str, Any]) -> list[str]:
    """Decide which capabilities a benchmark measures.

    Args:
        benchmark: A benchmark record.

    Returns:
        Matching capability names, possibly empty.
    """
    haystack = " ".join(
        str(benchmark.get(field) or "").lower() for field in ("type", "coverage", "name", "notes")
    )
    matched = []
    for name, include, exclude in CAPABILITIES:
        if not any(k in haystack for k in include):
            continue
        # A disqualifying phrase wins: "code-mixed" is not programming.
        if any(k in haystack for k in exclude) and not any(
            k in haystack for k in include if k not in ("code",)
        ):
            continue
        matched.append(name)
    return matched


def build_matrix(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the capability x benchmark matrix.

    Args:
        benchmarks: Benchmark records.

    Returns:
        Per-capability instruments and trust bands, plus the explicit list of holes.
    """
    matrix: dict[str, dict[str, Any]] = {
        name: {"benchmarks": [], "trust_bands": set()} for name, _, _unused in CAPABILITIES
    }

    for benchmark in benchmarks:
        for capability in capabilities_for(benchmark):
            matrix[capability]["benchmarks"].append(benchmark.get("name"))
            matrix[capability]["trust_bands"].add(benchmark.get("trust_band"))

    rows = []
    for name, _, _unused in CAPABILITIES:
        entry = matrix[name]
        rows.append(
            {
                "capability": name,
                "benchmarks": entry["benchmarks"],
                "count": len(entry["benchmarks"]),
                "trust_bands": sorted(b for b in entry["trust_bands"] if b),
                # A single instrument is a coverage risk, not coverage.
                "is_hole": len(entry["benchmarks"]) == 0,
                "is_thin": len(entry["benchmarks"]) == 1,
            }
        )

    return {
        "capabilities": rows,
        "holes": [row["capability"] for row in rows if row["is_hole"]],
        "thin": [row["capability"] for row in rows if row["is_thin"]],
        "benchmark_count": len(benchmarks),
    }


def uninstrumented(matrix: dict[str, Any]) -> list[str]:
    """List capabilities with no instrument at all.

    Args:
        matrix: Output of `build_matrix`.

    Returns:
        Capability names with zero benchmarks.
    """
    return list(matrix["holes"])
