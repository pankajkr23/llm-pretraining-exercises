"""Build the shared BPE vocabulary and score it across all languages.

Run with:  uv run python -m tokenization
"""

import json

from .config import Config
from .corpus import fetch_article
from .metrics import LangScore, count_words, score, spread
from .tokenizer import count_tokens, save, train_bpe


def main() -> None:
    """Fetch corpora, train one shared BPE, and report the per-language ratios + score."""
    cfg = Config()
    corpora = {lang.code: fetch_article(lang, cfg.data_dir) for lang in cfg.languages}
    weights = {lang.code: lang.weight for lang in cfg.languages}

    tok = train_bpe(corpora, cfg.vocab_size, weights)
    save(tok, cfg.artifacts_dir / "tokenizer.json")

    scores = [
        LangScore(
            code=lang.code,
            words=count_words(corpora[lang.code]),
            tokens=count_tokens(tok, corpora[lang.code]),
        )
        for lang in cfg.languages
    ]
    report = {
        "vocab_size": tok.get_vocab_size(),
        "languages": [
            {"code": s.code, "words": s.words, "tokens": s.tokens, "ratio": round(s.ratio, 4)}
            for s in scores
        ],
        "spread": round(spread(scores), 4),
        "score": round(score(scores), 2),
    }
    (cfg.artifacts_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
