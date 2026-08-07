"""Held-out scoring, and why it cannot pick a winner here.

This module was built to settle which weighting is best by scoring on text the trainer never saw.
It does not settle it, and that is the finding: **the split matters more than the recipe.**

Hold out every 5th line and one configuration scores 3,687. Hold out a different fifth of the same
corpus and the same configuration scores 10,956. Across five splits the standard deviation within
a single recipe (1,000–2,800 points) is larger than the gap between recipes (~1,000), so any
ranking read off one split is noise wearing a decimal point.

That is a property of the data, not of the method. Four Wikipedia articles is a few hundred
thousand units; a fifth of that is too little for the max-minus-min of four fertilities to settle
down, because the score depends on the two extreme languages and the smallest of them contributes
about 1,100 units to a held-out slice.

So the submission is chosen on measurements that do not move: total tokens over the whole corpus,
and score on the whole corpus. Held-out is reported for what it can actually support — that the
gaudy 35,603 configuration is not reliably better than anything, and neither is anyone else.

Run it with::

    uv run python -m tokenization.holdout
"""

import json
import statistics

from .ablate import OVERTUNED, REFERENCE, SUBMISSION, Spec, measure, train_spec
from .config import V2, Config
from .corpus import load_all
from .metrics import count_units, mean_ratio, score, spread

HOLDOUT_EVERY = 5


def split_lines(text: str, every: int = HOLDOUT_EVERY, offset: int = 0) -> tuple[str, str]:
    """Split ``text`` into (train, held-out), holding out every ``every``-th line from ``offset``.

    ``offset`` selects *which* fifth is held out. Running all five is what exposes how much of a
    held-out score is the recipe and how much is the luck of the slice.
    """
    lines = text.split("\n")
    train = "\n".join(line for i, line in enumerate(lines) if i % every != offset)
    held = "\n".join(line for i, line in enumerate(lines) if i % every == offset)
    return train, held


def evaluate(spec: Spec, corpora: dict[str, str], offset: int = 0) -> dict:
    """Train ``spec`` on one 80% split and report its in-sample and held-out numbers."""
    parts = {code: split_lines(text, offset=offset) for code, text in corpora.items()}
    train = {code: part[0] for code, part in parts.items()}
    held = {code: part[1] for code, part in parts.items()}
    tok = train_spec(spec, train)

    out = {"label": spec.label, "offset": offset, "weights": dict(spec.weights)}
    for name, texts in (("train", train), ("holdout", held)):
        scores = measure(tok, texts, {c: count_units(t) for c, t in texts.items()})
        out[name] = {
            "spread": round(spread(scores), 6),
            "score": round(score(scores), 2),
            "mean_ratio": round(mean_ratio(scores), 6),
        }
    return out


def stability(spec: Spec, corpora: dict[str, str], splits: int = HOLDOUT_EVERY) -> dict:
    """Score ``spec`` on every one of the ``splits`` possible held-out slices."""
    runs = [evaluate(spec, corpora, offset) for offset in range(splits)]
    held = [r["holdout"]["score"] for r in runs]
    return {
        "label": spec.label,
        "weights": dict(spec.weights),
        "holdout_scores": held,
        "mean": round(statistics.mean(held), 2),
        "stdev": round(statistics.stdev(held), 2) if len(held) > 1 else 0.0,
        "low": round(min(held), 2),
        "high": round(max(held), 2),
    }


def main() -> None:
    """Show that held-out score cannot rank these recipes, and write the evidence."""
    cfg = Config()
    corpora = load_all(V2, cfg.corpus_dir)
    specs = [REFERENCE, SUBMISSION, OVERTUNED]
    results = [stability(spec, corpora) for spec in specs]

    print(f"{'recipe':34} {'low':>8} {'high':>8} {'mean':>9} {'stdev':>8}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:34} {r['low']:>8.0f} {r['high']:>8.0f} "
            f"{r['mean']:>9.0f} {r['stdev']:>8.0f}"
        )

    widest = max(r["high"] - r["low"] for r in results)
    apart = max(r["mean"] for r in results) - min(r["mean"] for r in results)
    print(
        f"\nWidest range within one recipe: {widest:.0f} points. "
        f"Furthest apart any two recipes' means: {apart:.0f} points."
    )
    print(
        "The first number being larger than the second is the finding: held-out score on a corpus "
        "this small cannot rank these recipes, so the submission is not chosen on it."
    )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "holdout.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
