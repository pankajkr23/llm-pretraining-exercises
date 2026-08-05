"""Turn the five gate verdicts into one grade.

The grade is a summary, never a new judgment: it is derived from gates that already carry their own
reasoning, so a reviewer can always ask "why C?" and get five answers.

Two gates are load-bearing. A `FAIL` on **provenance** (you may not use it) or on **contamination**
(it would poison your evaluation) yields **X** regardless of everything else — those are not
deficiencies to trade off against a good score, they are exclusions. Grade X is what INV-2 keys on:
no X dataset may appear in a commercial mix.
"""

from typing import Any, Literal

Grade = Literal["A", "B", "C", "X"]

# Gates whose failure is disqualifying rather than merely bad.
BLOCKING_GATES: tuple[str, ...] = ("provenance", "contamination")

# What each verdict contributes to the 10-point score. UNKNOWN scores 0 — an unmeasured gate earns
# nothing, so ignorance costs the same as a poor result and cannot be mistaken for a pass.
_VERDICT_POINTS: dict[str, int] = {"PASS": 2, "CONDITIONAL": 1, "UNKNOWN": 0, "FAIL": 0}

# Score thresholds, calibrated against the worked example in `docs/DESIGN.md` §5: Sangraha, with
# provenance PASS · composition CONDITIONAL · contamination PASS · yield UNKNOWN · evidence PASS,
# scores 7 and is shown as grade B.
GRADE_A_MIN = 8
GRADE_B_MIN = 5


def score_gates(gates: dict[str, Any]) -> int:
    """Sum the gate verdicts into a 0–10 score.

    Args:
        gates: Gate name to serialised `Gate`.

    Returns:
        The score; 10 means every gate passed.
    """
    return sum(_VERDICT_POINTS.get((gate or {}).get("verdict", ""), 0) for gate in gates.values())


def grade_dataset(record: dict[str, Any]) -> tuple[Grade, str]:
    """Grade one dataset and explain the grade.

    Args:
        record: A Dataset Card record.

    Returns:
        A `(grade, reasoning)` pair; the reasoning names the gate that decided it.
    """
    gates: dict[str, Any] = record.get("gates") or {}
    if not gates:
        return "C", "No gates recorded, so nothing has been established either way."

    for name in BLOCKING_GATES:
        gate = gates.get(name) or {}
        if gate.get("verdict") == "FAIL":
            return "X", f"Blocked: the {name} gate failed — {gate.get('reasoning', '')}"

    # A blocking gotcha excludes a dataset even when the gates were derived leniently.
    for gotcha in record.get("gotchas") or []:
        if gotcha.get("severity") == "blocking":
            return "X", f"Blocked: {gotcha.get('type')} caveat — {gotcha.get('text', '')[:120]}"

    score = score_gates(gates)
    passed = [n for n, g in gates.items() if (g or {}).get("verdict") == "PASS"]
    unknown = [n for n, g in gates.items() if (g or {}).get("verdict") == "UNKNOWN"]
    detail = f"score {score}/10; passed {', '.join(passed) or 'nothing'}"
    if unknown:
        detail += f"; unmeasured: {', '.join(unknown)}"

    if score >= GRADE_A_MIN:
        return "A", detail
    if score >= GRADE_B_MIN:
        return "B", detail
    return "C", detail


def grade_all(records: list[dict[str, Any]]) -> dict[str, tuple[Grade, str]]:
    """Grade every record.

    Args:
        records: Dataset Card records.

    Returns:
        Record id to `(grade, reasoning)`.
    """
    return {record["id"]: grade_dataset(record) for record in records if record.get("id")}


def is_commercially_usable(grade: Grade) -> bool:
    """Whether a grade may appear in a mix marked commercial (INV-2).

    Args:
        grade: The dataset's grade.

    Returns:
        False only for grade X.
    """
    return grade != "X"
