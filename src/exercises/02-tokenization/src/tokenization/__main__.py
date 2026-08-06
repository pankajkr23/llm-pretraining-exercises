"""Build the submission tokenizer and score it across all languages.

Trains the submission recipe on the committed wiki-faithful corpus and writes to ``artifacts/``:

* ``tokenizer.json`` — the tokenizer itself, and
* ``report.json`` — per-language units, tokens, fertility, spread, raw score, Hindi penalty and
  adjusted score.

Everything is read from ``corpus/`` and nothing is fetched, so this runs offline and a fresh
clone reproduces the published numbers exactly.

Run with:  uv run python -m tokenization
"""

import json

from .ablate import SUBMISSION, measure, train_spec
from .config import Config
from .corpus import load_faithful
from .metrics import (
    adjusted_score,
    count_units,
    count_words,
    hindi_penalty,
    score,
    spread,
)


def main() -> None:
    """Train the submission tokenizer on the committed corpus and write its report."""
    cfg = Config()
    names = {lang.code: lang.name for lang in cfg.languages}
    corpora = {lang.code: load_faithful(lang.code, cfg.corpus_dir) for lang in cfg.languages}
    units = {c: count_units(t) for c, t in corpora.items()}

    tok = train_spec(SUBMISSION, corpora)
    scores = measure(tok, corpora, units)

    report = {
        "recipe": SUBMISSION.label,
        "vocab_size": tok.get_vocab_size(),
        "languages": [
            {
                "code": s.code,
                "name": names.get(s.code, s.code),
                "units": s.units,
                "tokens": s.tokens,
                "ratio": round(s.ratio, 6),
                # Whitespace words are reported for contrast only — nothing is scored on them.
                "words": count_words(corpora[s.code]),
            }
            for s in scores
        ],
        "spread": round(spread(scores), 6),
        "score": round(score(scores), 2),
        "hindi_penalty": round(hindi_penalty(scores), 6),
        "adjusted_score": round(adjusted_score(scores), 2),
    }

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(cfg.artifacts_dir / "tokenizer.json"))
    (cfg.artifacts_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
