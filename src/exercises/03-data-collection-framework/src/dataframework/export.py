"""Assemble the precomputed bundle the static site renders.

Python computes; the widget renders. Only mix arithmetic runs live in the browser, because it
responds to input — everything else is decided here, once, and shipped as data.

Two constraints shape the output. `data.json` is an **index**, kept under 100KB so the first page
paints immediately; the full reasoning for a dataset lives in `catalog.json`, which the
Reasoning surface loads on demand. And **no bare numbers cross the boundary** — every figure is
`{value, unit, provenance, source}`, so the UI can render measured and estimated differently and a
reader can always see which is which.

Run: ``uv run python -m dataframework``
"""

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

from . import __version__
from .catalog import EXPECTED_COUNTS, load_json, validate
from .config import Config
from .coverage import build_matrix
from .fertility import D_MODEL_DEFAULT, PARITY_TARGET, unmeasured
from .grade import grade_dataset
from .milestones import TIER_SHAPE, build_all
from .mix import ALWAYS_ON_SHARE, MAX_EPOCHS_ADVISED, MAX_EPOCHS_HARD
from .models import Value
from .orphans import find_orphans
from .shingles import write_index
from .sourcing import build_lifecycle, build_plan
from .vocab_sweep import summarise, sweep

# Fertility has no measured anchor until task 2.2b runs, so the sweep's own reference point is
# declared estimated and the curve is labelled accordingly.
REFERENCE_VOCAB = 100_000
REFERENCE_FERTILITY = 2.4


def _value(value: float | int | None, unit: str, provenance: str, source: str) -> dict[str, Any]:
    """Serialise one provenance-typed number.

    Args:
        value: The magnitude, or None when unknown.
        unit: What it counts.
        provenance: measured / estimated / unknown.
        source: Where it came from.

    Returns:
        The serialised `Value`.
    """
    if value is None:
        return dataclasses.asdict(Value.unknown(unit, source=source))
    return dataclasses.asdict(
        Value(value=value, unit=unit, provenance=provenance, source=source)  # type: ignore[arg-type]
    )


def _strip_tier_prose(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the per-tier prose from every preset, since it is identical across all of them.

    Args:
        presets: Built milestone presets.

    Returns:
        The presets with the shape-level fields removed from each tier; they are exported once
        under `milestones.tier_info` instead.
    """
    for preset in presets:
        mix = preset["mix"]
        for tier in mix["tiers"]:
            for field in ("sources", "why", "capabilities"):
                tier.pop(field, None)
            # A token count carried to nine decimal places is noise: these are estimates, and the
            # trailing float precision is pure bytes.
            for field in ("unique_tokens", "seen_tokens"):
                if isinstance(tier.get(field), float):
                    tier[field] = round(tier[field])
            if isinstance(tier.get("share"), float):
                tier["share"] = round(tier["share"], 4)
        for field in ("total_seen_tokens", "total_unique_tokens"):
            if isinstance(mix.get(field), float):
                mix[field] = round(mix[field])
        for field in (
            "indic_share",
            "natural_indic_share",
            "synthetic_share_of_indic",
            "always_on_share",
        ):
            if isinstance(mix.get(field), float):
                mix[field] = round(mix[field], 4)
        if isinstance(preset.get("target_seen_tokens"), float):
            preset["target_seen_tokens"] = round(preset["target_seen_tokens"])
    return presets


def _load_records(cfg: Config) -> dict[str, Any]:
    """Load every reference record array.

    Args:
        cfg: Paths to use.

    Returns:
        Record name to loaded content.
    """
    records: dict[str, Any] = {}
    for name in EXPECTED_COUNTS:
        if name in ("catalog", "benchmarks") or "." in name:
            continue
        path = cfg.records_dir / f"{name}.json"
        if path.exists():
            records[name] = load_json(path)
    market_path = cfg.records_dir / "market.json"
    if market_path.exists():
        records["market"] = load_json(market_path)

    # Documents rather than record arrays: these were extracted from the prose in DECISIONS.md and
    # ATLAS.md, where the answers to the assignment's questions had been written and never
    # published. They are object-shaped, so they load by name instead of through EXPECTED_COUNTS,
    # whose check is a length.
    for name in (
        "posttraining",
        "cleaning_rules",
        "eval_policy",
        "vocab_blocks",
        "fertility_targets",
        "cost",
    ):
        path = cfg.records_dir / f"{name}.json"
        if path.exists():
            records[name] = load_json(path)
    return records


def _dataset_index_entry(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full Dataset Card to the fields the index needs.

    Args:
        record: The full record.

    Returns:
        The trimmed entry, carrying its grade.
    """
    grade, _ = grade_dataset(record)
    return {
        "id": record["id"],
        "slug": record.get("slug"),
        "name": record.get("name"),
        "category": record.get("category"),
        "tier": record.get("tier"),
        # Omitted when false: `blocking` and `is_gap` are absent-means-no, and 145 explicit
        # `false`s cost 5KB of the index budget to say nothing.
        **({"is_gap": True} if record.get("is_gap") else {}),
        "grade": grade,
        "stage": record.get("stage"),
        "languages": record.get("languages"),
        # Fully typed even in the index: ground rule 4 leaves the UI no way to render a bare
        # number, so trimming this to a plain float would be a false economy. The grade's
        # reasoning and every gate live in catalog.json, loaded on demand.
        "size_tokens": record.get("size", {}).get("tokens"),
        "gotcha_types": sorted({g["type"] for g in record.get("gotchas") or []}),
        **(
            {"blocking": True}
            if any(g.get("severity") == "blocking" for g in record.get("gotchas") or [])
            else {}
        ),
        "licence_commercial": record.get("licence", {}).get("commercial"),
    }


def _fertility_block(cfg: Config, languages: list[str]) -> dict[str, Any]:
    """Assemble the fertility block from a real measurement when one exists.

    Ground rule 8 forbids shipping fertility as `estimated`: it is measured or it is `unknown`.
    So this reads `records/fertility.json` if `measure_fertility` has been run and falls back to
    `unknown` otherwise — there is deliberately no middle path.

    Args:
        cfg: Paths to use.
        languages: Every scheduled language code.

    Returns:
        The fertility block, carrying whatever was actually measured plus what the run could not
        cover.
    """
    block: dict[str, Any] = {
        "parity_target": _value(
            PARITY_TARGET, "ratio", "estimated", "the project's fertility target"
        ),
    }
    path = cfg.records_dir / "fertility.json"
    if not path.exists():
        block["by_language"] = unmeasured(
            languages,
            "task 2.2b has not run: no tokenizer measurement exists yet (ground rule 8 forbids "
            "shipping fertility as estimated)",
        )
        return block

    run = load_json(path)
    measured = run["by_tokenizer"]
    baseline = "tiktoken/cl100k_base"

    # The headline column stays the published baseline, so our numbers sit beside the paper's.
    rows = measured.get(baseline, {})
    by_language: dict[str, Any] = {}
    for code in [*languages, "en"]:
        # The corpora key Kashmiri differently — IN22-Gen has one column, FLORES ships both
        # scripts — so take the plain code when it exists and fall back to Perso-Arabic, the
        # script the language is predominantly written in.
        key = code if code in rows else f"{code}-Arab"
        if key in rows:
            by_language[code] = rows[key]
        else:
            absent = run["languages_unavailable"].get(code, "not measured")
            by_language[code] = dataclasses.asdict(
                Value.unknown("tokens/word", source=f"absent from {run['corpus']}: {absent}")
            )
    block["by_language"] = by_language
    block["baseline_tokenizer"] = baseline

    # One mean per tokenizer: enough for the report to rank them without shipping the whole matrix.
    ranking = []
    for ref, values in measured.items():
        english = (values.get("en") or {}).get("value")
        indic = [v["value"] for c, v in values.items() if c != "en" and v.get("value")]
        if english and indic:
            ranking.append(
                {
                    "tokenizer": ref,
                    "mean_tax": _value(
                        round(sum(i / english for i in indic) / len(indic), 3),
                        "ratio",
                        "measured",
                        f"{ref}@{run['run_id']}",
                    ),
                    "worst_tax": _value(
                        round(max(indic) / english, 3),
                        "ratio",
                        "measured",
                        f"{ref}@{run['run_id']}",
                    ),
                }
            )
    block["by_tokenizer_mean"] = sorted(ranking, key=lambda r: r["mean_tax"]["value"])
    # The full per-tokenizer matrix is three times this size and only §3 reads it, so it rides in
    # records.json rather than the index — the 100KB budget is what keeps the first paint fast.
    # Expansion is by_language divided by English, so shipping it would be a second copy of the
    # same fact — and a copy made of bare floats, which the bundle forbids. The UI divides.
    block["run_id"] = run["run_id"]
    block["corpus"] = run["corpus"]
    block["corpus_trust_band"] = run["corpus_trust_band"]
    block["tokenizers_unavailable"] = run["tokenizers_unavailable"]
    block["languages_unavailable"] = run["languages_unavailable"]
    block["protocol_gaps"] = run["protocol_gaps"]
    return block


def build_bundle(cfg: Config | None = None) -> dict[str, Any]:
    """Compute the whole bundle.

    Args:
        cfg: Paths to use; defaults to `Config()`.

    Returns:
        A mapping with `data` (the index), `details` (per-dataset) and `shingles` metadata.

    Raises:
        RuntimeError: If the spine does not validate — a broken spine must not reach the site.
    """
    cfg = cfg or Config()

    counts, errors = validate(cfg)
    if errors:
        raise RuntimeError(
            f"refusing to export from an invalid spine ({len(errors)} problem(s)); "
            f"first: {errors[0]}"
        )

    datasets = load_json(cfg.catalog_file)
    benchmarks = load_json(cfg.benchmarks_file)
    records = _load_records(cfg)
    fertility_run = cfg.records_dir / "fertility.json"
    if fertility_run.exists():
        records["fertility"] = load_json(fertility_run)

    # Anchor the sweep on our own measurement when one exists. cl100k_base is a ~100K vocabulary,
    # which is the sweep's reference point, so its measured mean Indic fertility is the right anchor
    # — and it is four times the 2.4 that was assumed before anything was measured.
    fertility_run = cfg.records_dir / "fertility.json"
    anchor, anchor_provenance = REFERENCE_FERTILITY, "estimated"
    if fertility_run.exists():
        rows = load_json(fertility_run)["by_tokenizer"].get("tiktoken/cl100k_base", {})
        indic = [v["value"] for code, v in rows.items() if code != "en" and v.get("value")]
        if indic:
            anchor, anchor_provenance = round(sum(indic) / len(indic), 4), "measured"
    curve = summarise(sweep(REFERENCE_VOCAB, anchor))
    languages = [row.get("code") for row in records.get("languages", []) if row.get("code")]
    shingle_meta = write_index(cfg)

    presets = _strip_tier_prose(build_all(records.get("milestones", [])))

    grades: dict[str, int] = {}
    for record in datasets:
        grade, _ = grade_dataset(record)
        grades[grade] = grades.get(grade, 0) + 1

    data = {
        "generated_at": None,  # stamped by the caller; the pipeline itself must stay deterministic
        "pipeline_version": __version__,
        "registry_root": "catalog.json",
        # Series-level provenance: these blocks are wholly computed, and every number inside
        # inherits the declaration. Typing 33 curve points individually would add bytes, not
        # information — the whole series shares one origin.
        "record_counts": {
            **counts,
            "provenance": "measured",
            "source": "counted from the catalogue",
        },
        "datasets": [_dataset_index_entry(record) for record in datasets],
        "benchmarks": benchmarks,
        "grades": {**grades, "provenance": "measured", "source": "computed from the five gates"},
        "coverage": {
            **build_matrix(benchmarks),
            "provenance": "measured",
            "source": "counted from the benchmark register",
        },
        # The per-tier prose is a property of the mix shape, not of any one rung — carrying it in
        # all four presets cost ~10KB of duplication and pushed the index over budget.
        "milestones": {
            "presets": presets,
            "tier_info": {
                tier["name"]: {
                    "sources": tier.get("sources"),
                    "why": tier.get("why"),
                    "capabilities": tier.get("capabilities", []),
                    "always_on": bool(tier.get("always_on")),
                    "is_indic": bool(tier.get("is_indic")),
                    "is_synthetic": bool(tier.get("is_synthetic")),
                }
                for tier in TIER_SHAPE
            },
            "provenance": "estimated",
            "source": "computed from the proposed tier shape",
        },
        "vocab_sweep": {
            **curve,
            "provenance": "estimated",
            "note": (
                "The fertility model behind this curve has no measured anchor until task 2.2b "
                "runs, so the peak is illustrative of the trade-off, not the recommended "
                "vocabulary."
            ),
            "d_model": _value(
                D_MODEL_DEFAULT, "dimensions", "estimated", "the project's model plan"
            ),
            "anchor_fertility": _value(
                anchor,
                "tokens/word",
                anchor_provenance,
                "our own measurement, cl100k_base over IN22-Gen"
                if anchor_provenance == "measured"
                else "assumed before anything was measured",
            ),
        },
        "fertility": _fertility_block(cfg, languages),
        "mix_rules": {
            "always_on_share": _value(
                ALWAYS_ON_SHARE, "share", "estimated", "the framework's protected-lane rule"
            ),
            "max_epochs_advised": _value(
                MAX_EPOCHS_ADVISED, "epochs", "estimated", "published repetition studies"
            ),
            "max_epochs_hard": _value(
                MAX_EPOCHS_HARD, "epochs", "estimated", "published repetition studies"
            ),
        },
        "contamination": {
            "coverage": shingle_meta["coverage"],
            "shingle_count": _value(
                shingle_meta["shingle_count"],
                "shingles",
                "measured",
                "computed from the indexed eval items",
            ),
            # Eval items the gate structurally cannot protect. A reported number, because the
            # alternative is silence that reads as "clean".
            "unindexable_items": _value(
                shingle_meta["unindexable_items"],
                "items",
                "measured",
                "computed from the indexed eval items",
            ),
            # As text, not a list of ints: every number in this bundle must be provenance-typed,
            # and a window width is a parameter rather than a measurement.
            "gram_widths": ", ".join(str(w) for w in shingle_meta["gram_widths"]) or "none",
            "note": shingle_meta["note"],
        },
        # `priors` stays in the index because the Decision cites it inline — but only the three
        # fields both surfaces actually render. The full records live in records/priors.json.
        # The catalogue graded against the proposed mixture: what can actually be committed, and
        # what is blocking the rest. Computed at the recommended rung.
        "sourcing": build_plan(
            [_dataset_index_entry(record) for record in datasets],
            next(
                (p["mix"] for p in presets if p.get("recommended")),
                presets[len(presets) // 2]["mix"],
            ),
        ),
        # Every dataset grouped by the training stage it serves, so post-training and evaluation
        # stop being invisible. Nothing unmatched is dropped — it is listed with its tags.
        "lifecycle": build_lifecycle([_dataset_index_entry(record) for record in datasets]),
        # Tiers no benchmark can detect, priced. "Every tier must have an instrument" is one of the
        # four evaluation disciplines, and until now nothing checked it.
        "orphan_tiers": {
            "provenance": "measured",
            "source": "matched from the mixture against the benchmark register",
            # `_strip_tier_prose` removes `capabilities` from the shipped presets to save bytes,
            # and that is exactly the field the orphan check matches on — so this rebuilds the
            # recommended mix from TIER_SHAPE, which still carries it. Passing the stripped mix
            # reported all eight tiers as orphans, which is the answer you get when you ask a
            # question with the evidence deleted.
            "tiers": find_orphans(
                build_all(records.get("milestones", []))[
                    next(
                        (i for i, p in enumerate(presets) if p.get("recommended")),
                        len(presets) // 2,
                    )
                ]["mix"],
                benchmarks,
            ),
        },
        "priors": [
            {
                "id": row.get("id"),
                "claim": row.get("claim"),
                "source": row.get("source"),
                "effect_on_design": row.get("effect_on_design"),
            }
            for row in records.get("priors", [])
        ],
    }

    reference = {name: rows for name, rows in records.items() if name != "priors"}
    return {"data": data, "records": reference, "shingles": shingle_meta}


def write_bundle(cfg: Config | None = None) -> dict[str, Any]:
    """Compute and write the web bundle.

    Writes only what does not already exist: the index, the reference arrays, and the shingle
    hashes. **Per-dataset detail is not written** — it would duplicate `catalog.json`, which is
    already tracked as the reviewable source of truth. The deploy script serves that register
    alongside `web/`, and the Reasoning surface reads it directly.

    Args:
        cfg: Paths to use; defaults to `Config()`.

    Returns:
        A summary of what was written.
    """
    cfg = cfg or Config()
    bundle = build_bundle(cfg)

    # `build_bundle` is pure so its output can be diffed and tested; the timestamp is stamped here,
    # at the I/O boundary, because a page that says when it was built is worth more than a bundle
    # that is byte-identical between runs. Nothing downstream compares two bundles for equality.
    bundle["data"]["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    cfg.web_dir.mkdir(parents=True, exist_ok=True)
    index_path = cfg.web_dir / "data.json"
    index_path.write_text(
        json.dumps(bundle["data"], ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    records_path = cfg.web_dir / "records.json"
    records_path.write_text(
        json.dumps(bundle["records"], ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    return {
        "index_bytes": index_path.stat().st_size,
        "records_bytes": records_path.stat().st_size,
        "detail_source": "catalog.json (served directly, not duplicated)",
        "shingle_coverage": bundle["shingles"]["coverage"],
    }


def main() -> int:
    """Run the export from the command line.

    Returns:
        Process exit code.
    """
    try:
        summary = write_bundle()
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return 1
    kb = summary["index_bytes"] / 1024
    print(f"  web/data.json      {kb:,.1f} KB  (budget 100 KB)")
    print(f"  web/records.json   {summary['records_bytes'] / 1024:,.1f} KB  (lazy)")
    print(f"  per-dataset detail {summary['detail_source']}")
    print(f"  contamination      coverage={summary['shingle_coverage']}")
    if kb > 100:
        print(f"  WARNING: index is {kb:,.1f} KB, over the 100KB budget")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
