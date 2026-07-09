"""Build the shared BPE vocabulary and score it across all languages.

Trains two tokenizers on the same corpora and writes both to ``artifacts/``:

* the HuggingFace byte-level BPE baseline (``tokenizer.json``), and
* the hand-written char-level BPE (``tokenizer_scratch.json``, see :mod:`.bpe_scratch`).

Run with:  uv run python -m tokenization
"""

import json
from typing import Protocol

from .bpe_scratch import ScratchBPE
from .config import Config
from .corpus import fetch_article
from .metrics import LangScore, count_words, score, spread
from .tokenizer import save, train_bpe


class _Tokenizer(Protocol):
    """The slice of the tokenizer API both engines expose (HuggingFace and :class:`ScratchBPE`)."""

    def encode(self, text: str) -> object: ...
    def get_vocab_size(self) -> int: ...


def _report(tok: _Tokenizer, corpora: dict[str, str], words: dict[str, int]) -> dict:
    """Score ``tok`` on every corpus and assemble the per-language ratios + spread + score."""
    scores = [LangScore(c, words[c], len(tok.encode(t).ids)) for c, t in corpora.items()]
    return {
        "vocab_size": tok.get_vocab_size(),
        "languages": [
            {"code": s.code, "words": s.words, "tokens": s.tokens, "ratio": round(s.ratio, 4)}
            for s in scores
        ],
        "spread": round(spread(scores), 4),
        "score": round(score(scores), 2),
    }


def main() -> None:
    """Fetch corpora, train the baseline + from-scratch BPE, and report their scores."""
    cfg = Config()
    corpora = {lang.code: fetch_article(lang, cfg.data_dir) for lang in cfg.languages}
    words = {lang.code: count_words(corpora[lang.code]) for lang in cfg.languages}
    weights = {lang.code: lang.weight for lang in cfg.languages}

    # HuggingFace byte-level BPE baseline.
    baseline = train_bpe(corpora, cfg.vocab_size, weights)
    save(baseline, cfg.artifacts_dir / "tokenizer.json")
    report = _report(baseline, corpora, words)
    (cfg.artifacts_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Hand-written char-level BPE (the winning char + NFKC recipe), persisted for review.
    scratch = ScratchBPE(normalization="NFKC")
    scratch.train(corpora, cfg.vocab_size, dict.fromkeys(corpora, 1.0))
    scratch.save(cfg.artifacts_dir / "tokenizer_scratch.json")
    print(
        f"\nsaved from-scratch BPE → artifacts/tokenizer_scratch.json "
        f"(score {_report(scratch, corpora, words)['score']})"
    )


if __name__ == "__main__":
    main()
