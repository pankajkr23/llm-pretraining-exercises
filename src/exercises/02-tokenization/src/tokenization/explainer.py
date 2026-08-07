"""Measured data for the page's two explainers.

The explainers let a reader move a dial and watch the consequences. Every position of that dial is
a **real training run**, not an interpolation: the reader is stepping through measurements, so the
curve they discover is the one the tokenizer actually produces. Interpolating between two real
points would be quicker and would quietly invent the shape of the thing the widget exists to show.

Writes ``web/explainer.json``. Regenerate with::

    uv run python -m tokenization.explainer

Three datasets:

* **corpus** — how much text each language actually contributes, before and after weighting. This
  is what a weight *means*: "Maithili ×3" moves it from 1.1% to 1.6% of the training mix.
* **budget** — Maithili's weight swept across ten real settings. The score peaks at ×3 and falls
  away on both sides, while total tokens climb the whole way: evenness bought with compression.
* **holdout** — the same recipes scored on text they trained on and on text they never saw, plus
  the five-split spread that shows why held-out cannot be used to choose between them.
"""

import json

from .ablate import OVERTUNED, REFERENCE, SUBMISSION, Spec, measure, run, train_spec
from .config import V2, Config
from .corpus import load_all
from .holdout import split_lines, stability
from .metrics import count_units, mean_ratio, score

# Maithili's weight, swept. Every one of these is trained and measured; the widget snaps between
# them rather than interpolating. 3 is the submission; 2 is where the reference recipe sits.
MAI_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16)


def _spec(mai: float) -> Spec:
    """The submission recipe with Maithili's share dialled to ``mai``."""
    return Spec(
        algo="bpe",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label=f"mai ×{mai:g}",
        weights=(("en", 3), ("hi", 4), ("te", 4), ("mai", mai)),
        train_unit="documents",
    )


def budget_sweep(cfg: Config) -> list[dict]:
    """Two real training runs per position of the dial — one full, one held out.

    The full-corpus score is the stable one: it peaks at Maithili ×3 and falls away on both sides.
    The held-out column is reported alongside it and is deliberately *not* used to pick a winner —
    ``holdout.stability`` shows its variance between splits exceeds the gap between recipes. It is
    here so a reader can see that noise rather than be told about it.
    """
    corpora = load_all(V2, cfg.corpus_dir)
    units = {c: count_units(t) for c, t in corpora.items()}

    parts = {code: split_lines(text) for code, text in corpora.items()}
    train = {code: part[0] for code, part in parts.items()}
    held = {code: part[1] for code, part in parts.items()}
    held_units = {c: count_units(t) for c, t in held.items()}
    out = []
    for mai in MAI_WEIGHTS:
        spec = _spec(mai)
        result = run(spec, corpora, units)
        unseen = measure(train_spec(spec, train), held, held_units)
        out.append(
            {
                "mai": mai,
                "ratios": result.ratios,
                "spread": result.spread,
                "score": result.adjusted,
                "total_tokens": result.total_tokens,
                "mean_ratio": result.mean_ratio,
                "unseen_score": round(score(unseen), 2),
                "unseen_mean_ratio": round(mean_ratio(unseen), 6),
                "is_submission": mai == 3,
            }
        )
    return out


def corpus_mix(cfg: Config) -> list[dict]:
    """What each language contributes, before and after weighting.

    This is what a weight *means*, and it is not obvious: "Maithili ×3" does not give Maithili a
    third of anything. Its article is 5,808 units — 1.8% of the corpus — so at the reference's ×2
    it is 1.1% of the text the trainer actually reads, and at ×3 it is 1.6%. The whole argument
    turns on a language that never rises above two percent of the mix.
    """
    corpora = load_all(V2, cfg.corpus_dir)
    units = {c: count_units(t) for c, t in corpora.items()}
    total_units = sum(units.values())
    rows = []
    for lang in V2.languages:
        code = lang.code
        rows.append(
            {
                "code": code,
                "name": lang.name,
                "units": units[code],
                "share": round(units[code] / total_units, 6),
                "chars": len(corpora[code]),
            }
        )
    # The share of training text each language occupies, at every stop on the dial.
    for row in rows:
        row["weighted_share"] = {}
    for mai in MAI_WEIGHTS:
        weights = {"en": 3, "hi": 4, "te": 4, "mai": mai}
        weighted = {c: units[c] * weights[c] for c in units}
        total = sum(weighted.values())
        for row in rows:
            row["weighted_share"][str(mai)] = round(weighted[row["code"]] / total, 6)
    return rows


def holdout_splits(cfg: Config) -> list[dict]:
    """Each headline recipe scored on all five possible held-out slices.

    One split per recipe would let a reader read a ranking off the numbers. Five splits show that
    the ranking is not there to be read: it changes depending on which fifth of the corpus you
    hold back, because the variance between splits is larger than the gap between recipes.
    """
    corpora = load_all(V2, cfg.corpus_dir)
    roles = ((REFERENCE, "benchmark"), (SUBMISSION, "submitted"), (OVERTUNED, "rejected"))
    out = []
    for spec, role in roles:
        stats = stability(spec, corpora)
        out.append(
            {
                "label": spec.label,
                "role": role,
                "splits": [round(v, 2) for v in stats["holdout_scores"]],
                "mean": stats["mean"],
                "stdev": stats["stdev"],
                "low": stats["low"],
                "high": stats["high"],
            }
        )
    return out


def main() -> None:
    """Generate both datasets and write ``web/explainer.json``."""
    from .widget import WEB_DIR  # noqa: PLC0415 — one definition of where the bundle lives

    cfg = Config()
    payload = {
        "corpus": corpus_mix(cfg),
        "budget": budget_sweep(cfg),
        "holdout": holdout_splits(cfg),
        "note": (
            "Every point is a real training run on the committed corpus, not an interpolation."
        ),
    }
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    out = WEB_DIR / "explainer.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    for row in payload["budget"]:
        mark = "  <- submitted" if row["is_submission"] else ""
        print(
            f"  mai ×{row['mai']:<3} score {row['score']:>9.2f}  tokens {row['total_tokens']}{mark}"
        )


if __name__ == "__main__":
    main()
