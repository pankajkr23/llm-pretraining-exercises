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
# Exclusion is two different jobs, and merging them into one field made one of them dead code.
#
# A **trap** is a phrase that makes an include term match for the wrong reason: "code-mixed" is
# Hindi-English switching, and it contains the word "code". The include term is fine; the substring
# is a coincidence, so the trap is removed from the text before the include terms are tested.
#
# A **disqualifier** rules the capability out however it matched: a benchmark translated from
# English does not measure an Indian worldview, whatever else its description says.
#
# The old single `exclude` field was tested as `any(exclude) and not any(include except "code")`.
# Because the first `any(include)` had already succeeded on the line above, the second was always
# true for every capability whose include list lacked the literal "code" — so the clause could only
# ever fire for `code`, and the `india-context` exclusion never ran once. None of this is fussiness:
# a false positive *hides a hole*, and the holes are the whole point of this matrix.
CAPABILITIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ("indic-language", ("indic", "indian", "bhasha"), (), ()),
    (
        "india-context",
        ("india-context", "india context", "cultural", "worldview", "sanskriti"),
        (),
        # Phrased as the register actually phrases it. "translated from" alone matched nothing:
        # IndicMMLU-Pro says "Translated via IndicTrans2 + back-translation" and was being counted
        # as an instrument for Indian worldview, which is the one thing a translated set cannot
        # measure — it measures how the model handles translated English.
        ("translated", "back-translation", "machine-translated"),
    ),
    ("code", ("code", "programming", "swe-bench", "humaneval"), ("code-mix", "code-switch"), ()),
    ("agentic-coding", ("agentic", "swe-bench", "terminal-bench", "tool use"), (), ()),
    ("math-reasoning", ("math", "reasoning", "gsm", "stem", "aime"), (), ()),
    ("knowledge", ("knowledge", "mmlu", "question answering", " qa"), (), ()),
    ("translation", ("translation", "machine translation", "parallel", "bitext"), (), ()),
    ("speech", ("speech", "asr", "audio", "spoken"), (), ()),
    ("multimodal", ("multimodal", "vision", "image", "ocr", "document"), (), ()),
    ("safety", ("safety", "harm", "jailbreak", "toxic"), (), ()),
    ("long-context", ("long context", "long-context"), (), ()),
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
    for name, include, traps, disqualifiers in CAPABILITIES:
        # Outright: however it matched, it does not measure this.
        if any(phrase in haystack for phrase in disqualifiers):
            continue
        # Coincidental: remove the trap, then ask whether anything still matches on its own merit.
        probe = haystack
        for trap in traps:
            probe = probe.replace(trap, " ")
        if not any(k in probe for k in include):
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
        name: {"benchmarks": [], "trust_bands": set()} for name, *_ in CAPABILITIES
    }

    for benchmark in benchmarks:
        for capability in capabilities_for(benchmark):
            matrix[capability]["benchmarks"].append(benchmark.get("name"))
            matrix[capability]["trust_bands"].add(benchmark.get("trust_band"))

    rows = []
    for name, *_ in CAPABILITIES:
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
