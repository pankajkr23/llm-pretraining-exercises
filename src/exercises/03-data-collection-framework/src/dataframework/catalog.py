"""Load and validate every record in the spine.

The build fails loudly rather than shipping a quietly broken catalogue. What counts as broken is set
by the framework's own rules, not by taste:

* a **judgment without reasoning** — an unauditable verdict looks like evidence (INV-3);
* an **untyped number** — a bare float has no provenance and cannot be rendered honestly;
* a **duplicate or unknown id**, which would silently overwrite a record downstream;
* a **row whose Risk & Notes yielded nothing**, i.e. research content dropped on the floor (INV-5);
* a **record count that drifts** from the expected spine.

Run: ``uv run python -m dataframework.catalog --validate``
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .models import CONFIDENCES, GOTCHA_TYPES, PROVENANCES, SEVERITIES, VERDICTS

# The spine the Atlas actually produces. Drift here means content was gained or lost.
#
# Two counts differ from the estimates in `docs/TODO.md`, and the extraction was right to differ —
# padding to hit a planned number would be inventing rows (ground rule 7):
#
#   risks         20, not 21 — §18 holds 8 technical + 6 legal + 5 unknowns = 19, plus the R*_D
#                             unknown, which appears only in Addendum B. There is no 21st.
#   market.deals  17, not 16 — §15.2's table lists 15 (including the Anthropic settlement row the
#                             source prints there) and §15.3 names 2 more.
#
# The reference tier then totals 179 rows, which is what TODO's own per-file table sums to; the
# "~186" in its prose was approximate.
EXPECTED_COUNTS: dict[str, int] = {
    "catalog": 145,
    "benchmarks": 31,
    "priors": 17,
    "papers": 19,
    "risks": 20,
    "confidence": 21,
    "corrections": 14,
    "tools": 17,
    "acquisition": 8,
    "plan": 12,
    "languages": 22,
    "architectures": 5,
    "milestones": 4,
    # `market.json` holds two arrays; both are counted so neither can quietly empty.
    "market.deals": 17,
    "market.trends": 3,
}

TIERS: frozenset[str] = frozenset({"GREEN", "AMBER", "RED"})


class ValidationError(Exception):
    """Raised when the spine is not fit to build from."""


def _check_value(value: Any, where: str, errors: list[str]) -> None:
    """Check one serialised `Value`.

    Args:
        value: The candidate mapping.
        where: Location, for the error message.
        errors: Accumulator.
    """
    if not isinstance(value, dict):
        errors.append(f"{where}: expected a provenance-typed object, got {type(value).__name__}")
        return
    if value.get("provenance") not in PROVENANCES:
        errors.append(f"{where}: provenance {value.get('provenance')!r} is not recognised")
    if not (value.get("unit") or "").strip():
        errors.append(f"{where}: a number without a unit is a bare number")
    known = value.get("value") is not None
    if known and not (value.get("source") or "").strip():
        errors.append(f"{where}: stated value has no source — never invent figures")
    if not known and value.get("provenance") != "unknown":
        errors.append(f"{where}: null value must declare provenance 'unknown'")


def _check_gate(gate: Any, where: str, errors: list[str]) -> None:
    """Check one serialised `Gate` (INV-3).

    Args:
        gate: The candidate mapping.
        where: Location, for the error message.
        errors: Accumulator.
    """
    if not isinstance(gate, dict):
        errors.append(f"{where}: expected a gate object")
        return
    if gate.get("verdict") not in VERDICTS:
        errors.append(f"{where}: verdict {gate.get('verdict')!r} is not recognised")
    if not (gate.get("reasoning") or "").strip():
        errors.append(f"{where}: verdict without reasoning (INV-3)")
    if gate.get("confidence") not in CONFIDENCES:
        errors.append(f"{where}: confidence {gate.get('confidence')!r} is not recognised")


def validate_dataset(record: dict[str, Any], errors: list[str]) -> None:
    """Validate one Dataset Card record.

    Args:
        record: The loaded record.
        errors: Accumulator.
    """
    rid = record.get("id") or "<no id>"
    if not (record.get("name") or "").strip():
        errors.append(f"{rid}: missing name")
    tier = record.get("tier")
    if tier is not None and tier not in TIERS:
        errors.append(f"{rid}: tier {tier!r} is not one of {sorted(TIERS)}")

    _check_value(record.get("size", {}).get("tokens"), f"{rid}.size.tokens", errors)
    for part, value in (record.get("size", {}).get("naturalness") or {}).items():
        _check_value(value, f"{rid}.size.naturalness.{part}", errors)

    for name, gate in (record.get("gates") or {}).items():
        _check_gate(gate, f"{rid}.gates.{name}", errors)

    for i, gotcha in enumerate(record.get("gotchas") or []):
        if gotcha.get("type") not in GOTCHA_TYPES:
            errors.append(f"{rid}.gotchas[{i}]: type {gotcha.get('type')!r} is not recognised")
        if gotcha.get("severity") not in SEVERITIES:
            errors.append(
                f"{rid}.gotchas[{i}]: severity {gotcha.get('severity')!r} is not recognised"
            )
        if not (gotcha.get("text") or "").strip():
            errors.append(f"{rid}.gotchas[{i}]: empty text")

    # INV-5: a Risk & Notes field must survive as a caveat, an upside, or a kept note.
    if not (record.get("gotchas") or record.get("opportunity") or record.get("note")):
        errors.append(f"{rid}: Risk & Notes yielded nothing — research content dropped (INV-5)")


def validate_benchmark(record: dict[str, Any], errors: list[str]) -> None:
    """Validate one benchmark record.

    Args:
        record: The loaded record.
        errors: Accumulator.
    """
    name = record.get("name") or "<no name>"
    if not (record.get("split_policy") or "").strip():
        errors.append(f"{name}: no split policy — the train/validate/test discipline is the point")
    # INV-1 at the source: benchmark items must never enter the spine.
    for banned in ("items", "questions", "examples", "samples", "data"):
        if banned in record:
            errors.append(
                f"{name}: carries {banned!r} — eval text must never enter the repo (INV-1)"
            )


def _check_row_ids(rows: list[Any], where: str, errors: list[str]) -> None:
    """Check that reference rows carry unique ids.

    The reference tier is lower-ceremony than the catalogue, but a duplicate id still means one row
    silently shadows another once the arrays are indexed for export.

    Args:
        rows: The loaded rows.
        where: Record name, for the error message.
        errors: Accumulator.
    """
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{where}[{i}]: expected an object, got {type(row).__name__}")
            continue
        # `languages` is keyed by ISO code rather than a synthetic id.
        rid = row.get("id") or row.get("code")
        if not rid:
            errors.append(f"{where}[{i}]: row has no id")
        elif rid in seen:
            errors.append(f"{where}[{i}]: duplicate id {rid!r}")
        else:
            seen.add(str(rid))


def load_json(path: Path) -> Any:
    """Load one JSON file with a useful error message.

    Args:
        path: The file.

    Returns:
        The parsed content.

    Raises:
        ValidationError: If the file is missing or malformed.
    """
    if not path.exists():
        raise ValidationError(f"missing record file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON — {exc}") from exc


def validate(cfg: Config | None = None) -> tuple[dict[str, int], list[str]]:
    """Load the whole spine and collect every problem with it.

    Args:
        cfg: Paths to use; defaults to `Config()`.

    Returns:
        A `(counts, errors)` pair. `counts` maps record type to rows loaded.
    """
    cfg = cfg or Config()
    errors: list[str] = []
    counts: dict[str, int] = {}

    seen_ids: set[str] = set()
    catalog = load_json(cfg.catalog_file)
    counts["catalog"] = len(catalog)
    for i, record in enumerate(catalog):
        rid = record.get("id")
        if not rid:
            errors.append(f"catalog[{i}]: record has no id")
        elif rid in seen_ids:
            errors.append(f"catalog[{i}]: duplicate id {rid!r}")
        else:
            seen_ids.add(rid)
        validate_dataset(record, errors)

    benchmarks = load_json(cfg.benchmarks_file)
    counts["benchmarks"] = len(benchmarks)
    for record in benchmarks:
        validate_benchmark(record, errors)

    for name in EXPECTED_COUNTS:
        if name in ("catalog", "benchmarks"):
            continue
        # "market.deals" addresses one array inside a multi-array record file.
        file_stem, _, array_key = name.partition(".")
        path = cfg.records_dir / f"{file_stem}.json"
        if not path.exists():
            errors.append(f"missing records/{file_stem}.json")
            counts[name] = 0
            continue
        content = load_json(path)
        if array_key:
            if not isinstance(content, dict) or array_key not in content:
                errors.append(f"records/{file_stem}.json: expected an object with '{array_key}'")
                counts[name] = 0
                continue
            rows = content[array_key]
        else:
            rows = content
        if not isinstance(rows, list):
            errors.append(f"records/{name}.json: expected a list of rows")
            counts[name] = 0
            continue
        counts[name] = len(rows)
        _check_row_ids(rows, name, errors)

    for name, expected in EXPECTED_COUNTS.items():
        actual = counts.get(name, 0)
        if actual != expected:
            errors.append(f"record count drift: {name} has {actual}, expected {expected}")

    return counts, errors


def main() -> int:
    """Validate the spine from the command line.

    Returns:
        Process exit code: 0 when the spine is clean, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Load and validate the record spine.")
    parser.add_argument("--validate", action="store_true", help="validate and report (default)")
    parser.parse_args()

    try:
        counts, errors = validate()
    except ValidationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    total = sum(counts.values())
    for name in EXPECTED_COUNTS:
        actual, expected = counts.get(name, 0), EXPECTED_COUNTS[name]
        mark = "ok " if actual == expected else "DRIFT"
        print(f"  {mark} {name:14s} {actual:>4} / {expected}")
    print(f"  total records: {total}")

    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for error in errors[:40]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more", file=sys.stderr)
        return 1

    print("\nspine is clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
