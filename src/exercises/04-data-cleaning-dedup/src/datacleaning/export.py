r"""Writing what the page reads — and refusing to write what it must not.

Two outputs:

- `artifacts/run.json` — the complete record, git-ignored, everything a run produced.
- `web/data.json` — tracked, derived, under a 100 KB budget. **This is what the page renders and
  what the tests assert on**, because `artifacts/` does not exist in CI.

The budget is not tidiness. A reader on a phone downloads this file before seeing anything, and
exercise 03 found that prose in the bundle is what blows it. So prose lives in `chapters.js`, which
has no budget, and `data.json` carries numbers.

Two rules govern what corpus text may leave this module, and they are separate rules for separate
reasons (`DECISIONS.md` §D6):

1. **No real PII, ever.** Interactive demos use hand-written synthetic documents. From the real
   corpus only aggregates are published, and any text shown is post-scrub.
2. **Corpus excerpts are bounded** — at most `Config.max_excerpts` of `Config.max_excerpt_chars` —
   and exist only because the deduplication chapter is unconvincing without two near-identical
   documents on screen.

`ensure_ascii=False` throughout: escaping Devanagari to `\\uXXXX` triples the byte cost of exactly
the fields that carry Indic text, for no benefit on a UTF-8 page.
"""

import json
import logging
from pathlib import Path

from datacleaning import pipeline
from datacleaning.config import Config
from datacleaning.records import PipelineResult
from datacleaning.sources import ALL_SPECS

logger = logging.getLogger(__name__)


def bundle(result: PipelineResult, cfg: Config) -> dict[str, object]:
    """Build the tracked bundle the page reads.

    Args:
        result: What the run produced.
        cfg: Configuration.

    Returns:
        The bundle, ready to serialise.
    """
    return {
        "run": {
            "run_id": result.run_id,
            "profile": result.profile,
            "generated": result.manifest.get("generated"),
            "config_hash": result.manifest.get("config_hash"),
            "script_hash": result.manifest.get("script_hash"),
            "content_hash": result.manifest.get("content_hash"),
            "tokenizer": result.manifest.get("tokenizer"),
        },
        "strategies": result.extras.get("strategies", []),
        "corpora": [
            {
                "key": spec.key,
                "title": spec.title,
                "repo_id": spec.repo_id,
                "licence": spec.licence,
                "licence_note": spec.licence_note,
                "attribution": spec.attribution,
                "why": spec.why,
                "counts_toward_budget": spec.counts_toward_budget,
            }
            for spec in ALL_SPECS
        ],
        "selections": result.extras.get("selections", []),
        "stages": [s.as_json() for s in result.stages],
        "yield": pipeline.yield_descent(result.stages),
        "tokenizer_spread": result.extras.get("tokenizer_spread", {}),
        "unreadable": result.extras.get("unreadable", {}),
        "manifest": result.manifest,
    }


def write(result: PipelineResult, cfg: Config | None = None) -> dict[str, object]:
    """Write `artifacts/run.json` and `web/data.json`.

    Args:
        result: What the run produced.
        cfg: Configuration; defaults apply.

    Returns:
        A summary: the paths written and the bundle's size against its budget.
    """
    cfg = cfg or Config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.web_dir.mkdir(parents=True, exist_ok=True)

    payload = bundle(result, cfg)

    full = dict(payload)
    full["documents"] = len(result.docs)
    _dump(cfg.artifacts_dir / "run.json", full)

    data_json = cfg.web_dir / "data.json"
    size_kb = _dump(data_json, payload)

    if size_kb > cfg.data_json_budget_kb:
        logger.warning(
            "data.json is %.1f KB, over the %.0f KB budget. Prose belongs in chapters.js.",
            size_kb,
            cfg.data_json_budget_kb,
        )
    else:
        logger.info("data.json %.1f KB of %.0f KB budget", size_kb, cfg.data_json_budget_kb)

    return {
        "run_json": str(cfg.artifacts_dir / "run.json"),
        "data_json": str(data_json),
        "data_json_kb": round(size_kb, 2),
        "budget_kb": cfg.data_json_budget_kb,
        "within_budget": size_kb <= cfg.data_json_budget_kb,
    }


def _dump(path: Path, payload: dict[str, object]) -> float:
    """Write JSON and return its size in KB."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    return len(text.encode("utf-8")) / 1024
