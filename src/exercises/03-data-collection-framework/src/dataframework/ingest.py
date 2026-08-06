"""Expand the seed CSVs into one reviewable JSON file per record.

`data/seed/*.csv` is a convenient bootstrap format and a terrible review format: a licence
downgrade buried in a 40KB CSV diff is invisible. So the CSVs are expanded once into
`catalog/<ID>.<slug>.json` and `benchmarks/<slug>.json` — the *contested tier*, where every licence
and trust judgment arrives as its own diff hunk.

The expansion is derivation, not invention. Licence flags, size splits, and the five gates are all
read off the row, and each gate records the field it reasoned from, so a reviewer can check the
judgment against the source. Where the row is silent the gate says `UNKNOWN` rather than guessing
(ground rule 7).

Run: ``uv run python -m dataframework.ingest``
"""

import argparse
import csv
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from .config import Config
from .gotchas import parse as parse_notes
from .models import Gate, Value

# Gate names, in the order the Dataset Card renders them.
GATE_NAMES: tuple[str, ...] = ("provenance", "composition", "contamination", "yield", "evidence")

_TOKEN_SIZE = re.compile(r"([\d.]+)\s*([BMTK])\b", re.IGNORECASE)
_MULTIPLIER = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_SPLIT_SIZE = re.compile(r"(verified|unverified|synthetic)\s*([\d.]+)\s*([BMTK])", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|arXiv:[\d.]+", re.IGNORECASE)


def slugify(name: str) -> str:
    """Reduce a dataset name to a filename-safe slug.

    Args:
        name: The dataset or benchmark name.

    Returns:
        A lowercase, hyphenated slug.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "unnamed"


def parse_tokens(text: str) -> Value:
    """Read a token count out of a free-text size field.

    Args:
        text: The `Size_Scale` cell, e.g. ``"251B tokens total: Verified 64B / ..."``.

    Returns:
        A `Value` in tokens, or an explicitly unknown one when the cell states no token count.
    """
    if not text:
        return Value.unknown("tokens", source="size field empty")
    match = _TOKEN_SIZE.search(text)
    if not match or "token" not in text.lower():
        # The unparsed cell lives in catalog.json, which the atlas loads on demand; carrying it in
        # the index too cost 18KB of a 100KB budget to say the same thing 121 times.
        return Value.unknown("tokens", source="no token count stated")
    amount = float(match.group(1)) * _MULTIPLIER[match.group(2).upper()]
    return Value(value=amount, unit="tokens", provenance="estimated", source="seed:Size_Scale")


def parse_naturalness(text: str) -> dict[str, Any]:
    """Split a size field into verified / unverified / synthetic parts.

    The Atlas's headline sizes hide this: Sangraha's 251B is 64B verified, 24B unverified and 162B
    machine-translated, and only the first two are natural Indic text.

    Args:
        text: The `Size_Scale` cell.

    Returns:
        A mapping of part name to serialised `Value`, empty when the cell states no split.
    """
    parts: dict[str, Any] = {}
    for name, amount, suffix in _SPLIT_SIZE.findall(text or ""):
        value = float(amount) * _MULTIPLIER[suffix.upper()]
        parts[name.lower()] = dataclasses.asdict(
            Value(value=value, unit="tokens", provenance="estimated", source="seed:Size_Scale")
        )
    return parts


def parse_licence(text: str) -> dict[str, Any]:
    """Read the permission flags the Dataset Card shows off a licence string.

    Args:
        text: The `License` cell.

    Returns:
        The raw string plus `commercial`, `attribution` and `share_alike` flags, each `True`,
        `False`, or `None` when the cell does not say.
    """
    lowered = (text or "").lower()
    noncommercial = any(m in lowered for m in ("-nc", " nc", "noncommercial", "non-commercial"))
    permits_commercial = "permits commercial" in lowered or "commercial use" in lowered
    commercial: bool | None
    if noncommercial:
        commercial = False
    elif permits_commercial or any(
        m in lowered for m in ("cc-by-4", "cc by 4", "cc-0", "cc0", "mit", "apache")
    ):
        commercial = True
    else:
        commercial = None
    return {
        "raw": text or None,
        "commercial": commercial,
        "attribution": ("by" in lowered and "cc" in lowered) or "attribution" in lowered or None,
        "share_alike": ("-sa" in lowered or "share-alike" in lowered or "sharealike" in lowered)
        or None,
    }


def _gate(verdict: str, reasoning: str, confidence: str, field: str) -> dict[str, Any]:
    """Build a serialised gate that cites the seed field it reasoned from.

    Args:
        verdict: One of the four verdicts.
        reasoning: Why — never blank.
        confidence: How much weight it bears.
        field: The seed column the judgment derives from.

    Returns:
        The gate as a JSON-ready dict.
    """
    gate = Gate(
        verdict=verdict,  # type: ignore[arg-type]
        reasoning=reasoning,
        confidence=confidence,  # type: ignore[arg-type]
        citations=(f"seed:{field}",),
    )
    return {**dataclasses.asdict(gate), "citations": list(gate.citations)}


def derive_gates(row: dict[str, str], gotcha_types: set[str], blocking: bool) -> dict[str, Any]:
    """Derive the five framework gates from one catalogue row.

    Each verdict is read off the row and cites the field it came from; nothing is invented. A silent
    row yields `UNKNOWN`, which the UI renders honestly rather than as a pass.

    Args:
        row: The seed CSV row.
        gotcha_types: Gotcha types found in its Risk & Notes.
        blocking: Whether any of those gotchas is blocking.

    Returns:
        A mapping of the five gate names to serialised `Gate` records.
    """
    tier = (row.get("Tier") or "").strip().upper()
    licence = (row.get("License") or "").strip()

    if blocking:
        provenance = _gate(
            "FAIL",
            f"Blocking caveat in Risk & Notes; tier {tier or 'unset'}.",
            "high",
            "Risk_Notes",
        )
    elif tier == "GREEN" and licence:
        provenance = _gate(
            "PASS", f"Tier GREEN with a stated licence: {licence[:80]}", "high", "License"
        )
    elif tier == "AMBER":
        provenance = _gate(
            "CONDITIONAL", f"Tier AMBER: {licence[:80] or 'licence unclear'}", "medium", "License"
        )
    elif tier == "RED":
        provenance = _gate("FAIL", f"Tier RED: {licence[:80] or 'excluded'}", "high", "Tier")
    else:
        provenance = _gate("UNKNOWN", "Tier or licence not stated in the seed row.", "low", "Tier")

    composition = (
        _gate("CONDITIONAL", "Risk & Notes flags a composition caveat.", "medium", "Risk_Notes")
        if "COMPOSITION" in gotcha_types
        else _gate(
            "UNKNOWN",
            "No composition caveat recorded; not independently verified.",
            "low",
            "Risk_Notes",
        )
    )
    contamination = (
        _gate("FAIL", "Risk & Notes flags upstream contamination.", "high", "Risk_Notes")
        if "PROVENANCE" in gotcha_types and blocking
        else _gate(
            "UNKNOWN",
            "Decontamination is measured in phase 2, not asserted here.",
            "low",
            "Risk_Notes",
        )
    )
    dedup_note = "DEDUP" in gotcha_types
    yield_gate = (
        _gate(
            "CONDITIONAL",
            "Duplication caveat recorded; usable yield is below the headline size.",
            "medium",
            "Risk_Notes",
        )
        if dedup_note
        else _gate("UNKNOWN", "Yield after dedup not measured.", "low", "Size_Scale")
    )
    used_by = (row.get("Used_By") or "").strip()
    evidence = (
        _gate("PASS", f"Prior use recorded: {used_by[:80]}", "medium", "Used_By")
        if used_by
        else _gate("UNKNOWN", "No prior use recorded in the seed row.", "low", "Used_By")
    )
    return {
        "provenance": provenance,
        "composition": composition,
        "contamination": contamination,
        "yield": yield_gate,
        "evidence": evidence,
    }


# How catalogued datasets are related to each other.
#
# Adding two corpora only gives you their sum when they are independent, and most of the large ones
# here are not. This table records what is *published* about those relationships, in three kinds
# that must not be conflated:
#
#   contained_by   A publisher states that this dataset is a subset or an earlier version of
#                  another one in the catalogue. Exact, so the contained row is subtracted from any
#                  sum that also holds its parent.
#   shares_source  Two datasets are built from the same upstream corpus. The overlap is real and
#                  its size is not published by anybody. Recorded so a reader can see it, and
#                  discounted only through risk R01's range — never a per-pair coefficient, which
#                  would be a number with no source wearing the authority of a computation.
#   independent    Explicitly *not* crawl-derived. Recorded because "no overlap" is as much a
#                  finding as overlap is, and a reader should not have to assume it.
#
# `note` is the callout: the concrete, checkable thing that is known about the relationship, even
# where the magnitude is not. Modality and crawl dates are published; overlap fractions are not.
#
# Multi-parent is normal and the shape allows it. CulturaX is mC4 plus OSCAR; Sangraha's unverified
# portion is drawn from "existing multilingual corpora", plural and unnamed, which is why that one
# is `unknown` rather than modelled.
RELATIONSHIPS: dict[str, dict[str, Any]] = {
    "ENG-02": {
        "kind": "contained_by",
        "parents": ["ENG-01"],
        "note": "The educational slice of FineWeb, not an addition to it.",
        "source": (
            "HuggingFaceFW/fineweb-edu: '1.3T tokens of educational web pages filtered from the "
            "FineWeb dataset'"
        ),
    },
    "ENG-04": {
        "kind": "contained_by",
        "parents": ["ENG-03"],
        "note": "The educational slice of FinePDFs, not an addition to it.",
        "source": (
            "HuggingFaceFW/finepdfs-edu: '350B+ tokens of educational PDFs filtered from the "
            "FinePDFs"
            "dataset'"
        ),
    },
    "ENG-07": {
        "kind": "contained_by",
        "parents": ["ENG-08"],
        "note": (
            "v2 is this dataset plus eight further crawl snapshots, so holding v2 means holding "
            "this."
        ),
        "source": (
            "nvidia/Nemotron-CC-v2: 'based on Nemotron-CC with eight additional Common Crawl "
            "snapshots (2024-2025)'"
        ),
    },
    "ENG-09": {
        "kind": "additional_to",
        "parents": ["ENG-08"],
        "note": (
            "Genuinely additive, unlike the v1/v2 pair — its publisher says to use it alongside "
            "v2,"
            "not instead of it."
        ),
        "source": (
            "nvidia/Nemotron-CC-v2.1: '2.5T new tokens ... to be used in conjunction with the "
            "previously released 6.6T tokens of Nemotron-CC-v2'"
        ),
    },
    "ENG-01": {
        "kind": "shares_source",
        "parents": ["Common Crawl"],
        "note": (
            "96 Common Crawl dumps, summer 2013 to April 2024, HTML pages. Deduplicated per "
            "snapshot"
            "rather than globally, by the authors' own ablation, so it carries cross-snapshot "
            "duplicates of its own."
        ),
        "source": "HuggingFaceFW/fineweb and arXiv:2406.17557",
    },
    "ENG-03": {
        "kind": "shares_source",
        "parents": ["Common Crawl"],
        "note": (
            "106 Common Crawl dumps, 2013 to February 2025 — the same crawls as FineWeb, but the "
            "PDFs"
            "in them rather than the HTML. A PDF and a web page are different documents, so the "
            "overlap with the HTML corpora is far smaller than the shared source suggests."
        ),
        "source": "HuggingFaceFW/finepdfs",
    },
    "ENG-08": {
        "kind": "shares_source",
        "parents": ["Common Crawl"],
        "note": (
            "Common Crawl, with synthetic rephrasing applied to part of it — so some of this is "
            "not"
            "collected text at all."
        ),
        "source": (
            "nvidia/Nemotron-CC-v2: 'synthetic rephrasing using Qwen3-30B-A3B, filtered for "
            "English"
            "and globally deduplicated'"
        ),
    },
    "MUL-03": {
        "kind": "shares_source",
        "parents": ["Common Crawl", "Internet Archive"],
        "note": (
            "7.2 petabytes of raw crawl: 45% Common Crawl, 33% Internet Archive, 22% ArchiveBot, "
            "2012-2024. Only the Common Crawl portion overlaps the English corpora here, and "
            "English"
            "is one of three languages it deduplicates per-crawl rather than globally."
        ),
        "source": "arXiv:2511.01066, HPLT 3.0",
    },
    "MUL-04": {
        "kind": "shares_source",
        "parents": ["Common Crawl"],
        "note": (
            "Not built from the crawl directly but from two corpora that were: mC4 v3.1.0 and "
            "every"
            "OSCAR release to 23.01. Two parents, both downstream of the same crawl."
        ),
        "source": "uonlp/CulturaX",
    },
    "MUL-05": {
        "kind": "shares_source",
        "parents": ["Common Crawl"],
        "note": (
            "Common Crawl, cleaned. Named in risk R01 as one of the corpora whose mutual overlap "
            "drives the 60-80% estimate."
        ),
        "source": "MADLAD-400, and risk R01",
    },
    "MTH-01": {
        "kind": "shares_source",
        "parents": ["Common Crawl"],
        "note": "The mathematics slice of the same Nemotron Common Crawl pipeline.",
        "source": "nvidia/Nemotron-CC-Math-v1",
    },
    "COD-01": {
        "kind": "independent",
        "parents": [],
        "note": (
            "Source code from Software Heritage, not a web crawl. It does not overlap any of the "
            "web"
            "corpora here."
        ),
        "source": "BigCode, The Stack v2",
    },
    "COD-02": {
        "kind": "independent",
        "parents": [],
        "note": (
            "Curated from GitHub, not a web crawl. It does not overlap any of the web corpora here."
        ),
        "source": "nvidia/Nemotron-Pretraining-Code-v1",
    },
    "IND-01": {
        "kind": "independent",
        "parents": [],
        "note": (
            "The verified portion counted here is crawled from manually checked Indic websites, "
            "OCR'd from PDFs (Internet Archive, eGyanKosh, the Indian Parliament, AIR News, "
            "government magazines, school textbooks) and transcribed from audio (YouTube, "
            "OpenSubtitles, NPTEL, Mann Ki Baat). None of that is a general web crawl, so risk "
            "R01's Common Crawl discount does not apply to it. The unverified portion is a "
            "different matter and is excluded from the figure used here: the paper builds it by "
            "perplexity-filtering CulturaX and MADLAD-400, both catalogued separately and both "
            "Common Crawl. IndicCorp is named as an input to neither portion."
        ),
        "source": (
            "IndicLLMSuite, arXiv:2403.06350 - verified data from 'high-quality, manually "
            "verified Indic language websites' plus PDFs and video; unverified data from 'all "
            "the high-quality tagged documents from CulturaX and MADLAD-400'"
        ),
    },
    "IND-02": {
        "kind": "independent",
        "parents": [],
        "note": (
            "AI4Bharat's own crawl, released December 2022 and predating Sangraha by more than a "
            "year. The IndicLLMSuite paper compares the two and names IndicCorp nowhere as an "
            "input to Sangraha, so they are counted separately. What nobody publishes is a "
            "cross-deduplication between them, and both crawl Indian websites, so some overlap "
            "is plausible and unquantified - but neither contains the other."
        ),
        "source": "ai4bharat/IndicCorpV2; IndicLLMSuite, arXiv:2403.06350",
    },
}


# Figures the seed row gets wrong, corrected against the dataset's own card.
#
# The seed CSV is the machine-readable extract of the research and is never hand-edited, so a
# correction to one of its numbers lives here instead: keyed by id, carrying the replacement and the
# source that establishes it, and applied on ingest so the whole pipeline sees one value. An entry
# in this table is a claim about the outside world and is held to the same standard as any other —
# it names where the figure comes from.
SIZE_CORRECTIONS: dict[str, dict[str, Any]] = {
    "COD-02": {
        "tokens": 747_400_000_000,
        "source": (
            "nvidia/Nemotron-Pretraining-Code-v1 dataset card: 'metadata to reproduce a 747.4B "
            "token curated code dataset'. The seed row recorded 377M, which is NVIDIA's count of "
            "filtered GitHub *files*, not tokens — a units error of roughly 900x. Two caveats "
            "attach and neither is resolvable from published material: this is metadata to "
            "reproduce a corpus by re-fetching from GitHub rather than a corpus distributed "
            "directly, and the catalogue row covers v1/v2/v3 together while 747.4B is v1's figure "
            "alone, so the row now understates rather than overstates."
        ),
    },
}


def _corrected_tokens(row: dict[str, str], size_raw: str) -> dict[str, Any]:
    """The row's token count, with a cited correction applied where one exists.

    Args:
        row: The seed CSV row.
        size_raw: Its raw size string.

    Returns:
        A serialised `Value`.
    """
    fix = SIZE_CORRECTIONS.get((row.get("ID") or "").strip())
    if fix is None:
        return dataclasses.asdict(parse_tokens(size_raw))
    return dataclasses.asdict(
        Value(value=fix["tokens"], unit="tokens", provenance="estimated", source=fix["source"])
    )


def build_dataset_record(row: dict[str, str]) -> dict[str, Any]:
    """Expand one catalogue row into a Dataset Card record.

    Args:
        row: The seed CSV row.

    Returns:
        The record, JSON-ready.
    """
    parsed = parse_notes(row.get("Risk_Notes"))
    gotcha_types = {g.type for g in parsed.gotchas}
    blocking = any(g.is_blocking for g in parsed.gotchas)
    size_raw = row.get("Size_Scale") or ""
    # A "-" tier marks a dataset that does not exist yet — the Atlas records these deliberately,
    # because an unfilled slot is where the differentiation argument lives (see `#gaps`).
    tier_raw = (row.get("Tier") or "").strip().upper()
    tier = tier_raw if tier_raw in {"GREEN", "AMBER", "RED"} else None
    return {
        "id": row["ID"].strip(),
        "slug": slugify(row["Dataset"]),
        "name": row["Dataset"].strip(),
        "category": (row.get("Category") or "").strip(),
        "tier": tier,
        "is_gap": tier is None,
        "owner": (row.get("Owner_Steward") or "").strip() or None,
        "licence": parse_licence(row.get("License", "")),
        "size": {
            "raw": size_raw or None,
            "tokens": _corrected_tokens(row, size_raw),
            "naturalness": parse_naturalness(size_raw),
        },
        # How this dataset relates to others in the catalogue, where anybody has published it.
        "derivation": RELATIONSHIPS.get((row.get("ID") or "").strip()),
        "stage": [s.strip() for s in re.split(r"[/,]", row.get("Stage") or "") if s.strip()],
        "languages": (row.get("Languages") or "").strip() or None,
        "gotchas": [dataclasses.asdict(g) for g in parsed.gotchas],
        "opportunity": parsed.opportunity,
        "note": parsed.note,
        "gates": derive_gates(row, gotcha_types, blocking),
        "used_by": (row.get("Used_By") or "").strip() or None,
        "access": {
            "raw": (row.get("Access") or "").strip() or None,
            "links": _URL.findall(row.get("Access") or ""),
        },
        "confidence": "high" if (row.get("Access") or "").strip() else "low",
    }


def build_benchmark_record(row: dict[str, str]) -> dict[str, Any]:
    """Expand one benchmark row into a record.

    Carries the split policy and trust band, and **never** benchmark items — the eval registry must
    not reach `web/` (INV-1).

    Args:
        row: The seed CSV row.

    Returns:
        The record, JSON-ready.
    """
    notes = row.get("Notes") or ""
    policy = (row.get("Split_Policy") or "").strip()
    lowered = f"{notes} {row.get('Type', '')}".lower()
    if "translat" in lowered:
        trust = "translation-derived"
    elif "harness" in lowered or "harness" in policy.lower():
        trust = "harness-dependent"
    elif "contaminat" in lowered:
        trust = "contamination-prone"
    else:
        trust = "native-sourced"
    return {
        "name": row["Benchmark"].strip(),
        "slug": slugify(row["Benchmark"]),
        "owner": (row.get("Owner") or "").strip() or None,
        "type": (row.get("Type") or "").strip() or None,
        "coverage": (row.get("Coverage") or "").strip() or None,
        "size": (row.get("Size") or "").strip() or None,
        "split_policy": policy or None,
        "held_out": "held-out" in policy.lower() or "locked" in policy.lower(),
        "trust_band": trust,
        "access": {
            "raw": (row.get("Access") or "").strip() or None,
            "links": _URL.findall(row.get("Access") or ""),
        },
        "notes": notes.strip() or None,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a seed CSV.

    Args:
        path: The CSV path.

    Returns:
        The rows as dicts.

    Raises:
        FileNotFoundError: If the seed file is absent (it is git-ignored by design).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The seed CSVs are local working files, kept out of git "
            "(see docs/README.md) — restore them from your backup before running ingest."
        )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a register as pretty JSON, so diffs stay line-oriented and reviewable.

    Args:
        path: Destination file.
        records: The records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ingest(cfg: Config | None = None) -> dict[str, int]:
    """Expand both seed CSVs into per-record JSON files.

    Args:
        cfg: Paths to use; defaults to `Config()`.

    Returns:
        Counts of records written, keyed by tier name.
    """
    cfg = cfg or Config()
    datasets = _read_csv(cfg.seed_dir / "master_dataset_catalog.csv")
    benchmarks = _read_csv(cfg.seed_dir / "benchmarks.csv")

    _write(cfg.catalog_file, [build_dataset_record(row) for row in datasets])
    _write(cfg.benchmarks_file, [build_benchmark_record(row) for row in benchmarks])

    return {"catalog": len(datasets), "benchmarks": len(benchmarks)}


def main() -> None:
    """Run the expansion from the command line."""
    parser = argparse.ArgumentParser(description="Expand the seed CSVs into per-record JSON.")
    parser.parse_args()
    counts = ingest()
    print(f"wrote {counts['catalog']} catalog + {counts['benchmarks']} benchmark records")


if __name__ == "__main__":
    main()
