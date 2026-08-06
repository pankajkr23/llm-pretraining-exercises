"""Held-out validation: how much of a weighting gain is real, and how much is in-sample fit?

The reference setup trains and evaluates on the same four files. That is fine for reproducing a
published number, but it makes corpus weighting a knob tuned directly against the test set: sweep
the weights far enough and the spread collapses toward zero without the tokenizer getting any
better at tokenizing.

This module quantifies that. It splits every article by line — every 5th line held out — trains
on the remaining 80%, and scores on the 20% the trainer never saw. Run it with::

    uv run python -m tokenization.holdout

Splitting by line rather than by character keeps each side a plausible sample of the same
document: both halves contain prose, tables, reference blocks and category links in roughly the
proportions the whole article has.
"""

import json

from .ablate import SUBMISSION, Spec, measure, train_spec
from .config import Config
from .corpus import load_faithful
from .metrics import adjusted_score, count_units, mean_ratio, score, spread

HOLDOUT_EVERY = 5


def split_lines(text: str, every: int = HOLDOUT_EVERY) -> tuple[str, str]:
    """Split ``text`` into (train, held-out) by taking every ``every``-th line as held out."""
    lines = text.split("\n")
    train = "\n".join(line for i, line in enumerate(lines) if i % every)
    held = "\n".join(line for i, line in enumerate(lines) if not i % every)
    return train, held


def evaluate(spec: Spec, corpora: dict[str, str]) -> dict:
    """Train ``spec`` on the 80% split and report its in-sample and held-out numbers."""
    parts = {code: split_lines(text) for code, text in corpora.items()}
    train = {code: part[0] for code, part in parts.items()}
    held = {code: part[1] for code, part in parts.items()}
    tok = train_spec(spec, train)

    out = {"label": spec.label, "weights": dict(spec.weights)}
    for name, texts in (("train", train), ("holdout", held)):
        scores = measure(tok, texts, {c: count_units(t) for c, t in texts.items()})
        out[name] = {
            "spread": round(spread(scores), 6),
            "score": round(score(scores), 2),
            "adjusted": round(adjusted_score(scores), 2),
            "mean_ratio": round(mean_ratio(scores), 6),
            "ratios": {s.code: round(s.ratio, 6) for s in scores},
        }
    return out


def main() -> None:
    """Compare the reference recipe, the submission, and an over-tuned config out of sample."""
    from .ablate import REFERENCE, _reweighted  # noqa: PLC0415 — keeps the suite in one place

    cfg = Config()
    corpora = {lang.code: load_faithful(lang.code, cfg.corpus_dir) for lang in cfg.languages}
    specs = [
        REFERENCE,
        SUBMISSION,
        _reweighted("mai ×6 alone (no documents)", en=3, hi=4, te=4, mai=6),
        _reweighted("over-tuned · te ×6 · mai ×7", en=3, hi=4, te=6, mai=7),
    ]
    results = [evaluate(spec, corpora) for spec in specs]

    print(f"{'config':34} {'train adj':>10} {'HELD adj':>10} {'HELD mean X':>12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['label']:34} {r['train']['adjusted']:>10.2f} "
            f"{r['holdout']['adjusted']:>10.2f} {r['holdout']['mean_ratio']:>12.4f}"
        )

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "holdout.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
