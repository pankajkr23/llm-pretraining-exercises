"""Which fourth language, and what does the choice actually change?

English, Hindi and Telugu are fixed by the assignment; the fourth is ours. The reference solution
chose Maithili, we chose Tamil, and both snapshots sit in ``corpus/`` — so the swap can be
measured rather than argued about. Run it with::

    uv run python -m tokenization.fourth_language

The two are not small variations of each other. Maithili's article is 5,808 faithful units — 1.8%
of that corpus — and shares Devanagari with Hindi, so it rides merges it never paid for. Tamil's
is 188,367 units, larger than English, in a script nothing else in the mix uses. Swapping them
moves which language is *starved*, and therefore which weight is worth raising: with Maithili the
binding constraint is Maithili itself, and with Tamil it moves to Telugu, now the smallest corpus
in the set by a factor of five.

That is the finding worth reporting. The score difference between the two sets is not, because
they are different corpora — a fourth language with a large article and its own script is simply
a different problem from a tiny one sharing a script with Hindi.
"""

import json

from .ablate import Spec, run
from .config import REFERENCE_LANGUAGES, TAMIL, V2, Config, Language
from .corpus import load
from .metrics import count_units

MAITHILI_SET: tuple[Language, ...] = REFERENCE_LANGUAGES
TAMIL_SET: tuple[Language, ...] = (
    *(lang for lang in REFERENCE_LANGUAGES if lang.code != "mai"),
    TAMIL,
)


def _spec(label: str, **weights: float) -> Spec:
    """The submission recipe (documents, NFKC, Metaspace) at the given weights."""
    return Spec(
        algo="bpe",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label=label,
        weights=tuple(weights.items()),
        train_unit="documents",
    )


def compare(cfg: Config | None = None) -> list[dict]:
    """Score the submission recipe on both fourth-language sets, plus each set's tuned weights."""
    cfg = cfg or Config()
    trials = [
        (MAITHILI_SET, _spec("mai · reference weights", en=3, hi=4, te=4, mai=2)),
        (MAITHILI_SET, _spec("mai · tuned (mai ×6)", en=3, hi=4, te=4, mai=6)),
        (TAMIL_SET, _spec("ta · reference weights", en=3, hi=4, te=4, ta=2)),
        (TAMIL_SET, _spec("ta · tuned (te ×6)", en=3, hi=4, te=6, ta=2)),
    ]
    out = []
    for langs, spec in trials:
        corpora = {lang.code: load(V2, lang.code, cfg.corpus_dir) for lang in langs}
        units = {c: count_units(t) for c, t in corpora.items()}
        result = run(spec, corpora, units)
        out.append(
            {
                "label": spec.label,
                "fourth": next(lang.code for lang in langs if lang.code in {"mai", "ta"}),
                "weights": dict(spec.weights),
                "units": result.units,
                "ratios": result.ratios,
                "spread": result.spread,
                "adjusted": result.adjusted,
                "total_tokens": result.total_tokens,
                "mean_ratio": result.mean_ratio,
                "worst_language": max(result.ratios, key=lambda c: result.ratios[c]),
            }
        )
    return out


def main() -> None:
    """Print the fourth-language comparison and write ``artifacts/fourth_language.json``."""
    cfg = Config()
    results = compare(cfg)
    print(f"{'config':28} {'spread':>8} {'adj':>10} {'tokens':>9} {'mean X':>7}  worst")
    print("-" * 78)
    for r in results:
        print(
            f"{r['label']:28} {r['spread']:>8.4f} {r['adjusted']:>10.2f} "
            f"{r['total_tokens']:>9} {r['mean_ratio']:>7.4f}  {r['worst_language']}"
        )
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "fourth_language.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
