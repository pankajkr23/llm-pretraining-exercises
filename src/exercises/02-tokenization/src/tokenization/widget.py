"""Export the data the Netlify widget renders: ratios, score, and the full token list.

Trains a handful of featured tokenizers and writes ``web/data.json`` — the widget
(``web/index.html``) is a static page that loads this file. Regenerate with::

    uv run python -m tokenization.widget
"""

import json
from pathlib import Path

from .ablate import Spec, train_spec
from .config import Config
from .corpus import fetch_article
from .metrics import LangScore, count_words, score, spread

# Configs the reviewer can flip between in the widget (best score first).
FEATURED: list[Spec] = [
    Spec("unigram", "char", "NFKC", 10_000, "flat", "Unigram · char · NFKC"),
    Spec("bpe-scratch", "char", "NFKC", 10_000, "flat", "BPE from scratch · char · NFKC"),
    Spec("bpe", "char", "NFKC", 10_000, "flat", "BPE · char · NFKC"),
    Spec("bpe", "byte", None, 10_000, "flat", "BPE · byte (baseline)"),
]

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def build_payload(cfg: Config, corpora: dict[str, str], words: dict[str, int]) -> dict:
    """Train every featured config and assemble the widget's JSON payload."""
    names = {lang.code: lang.name for lang in cfg.languages}
    configs = []
    for spec in FEATURED:
        tok = train_spec(spec, corpora)
        scores = sorted(
            (LangScore(c, words[c], len(tok.encode(t).ids)) for c, t in corpora.items()),
            key=lambda s: s.ratio,
        )
        vocab = tok.get_vocab()
        tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        configs.append(
            {
                "label": spec.label,
                "algo": spec.algo,
                "level": spec.level,
                "normalization": spec.normalization,
                "vocab_actual": tok.get_vocab_size(),
                "languages": [
                    {
                        "code": s.code,
                        "name": names.get(s.code, s.code),
                        "words": s.words,
                        "tokens": s.tokens,
                        "ratio": round(s.ratio, 4),
                    }
                    for s in scores
                ],
                "x_min": round(scores[0].ratio, 4),
                "x_max": round(scores[-1].ratio, 4),
                "spread": round(spread(scores), 4),
                "score": round(score(scores), 2),
                "tokens": tokens,
            }
        )
    return {"target_vocab": cfg.vocab_size, "configs": configs}


def main() -> None:
    """Fetch corpora, train the featured configs, and write ``web/data.json``."""
    cfg = Config()
    corpora = {lang.code: fetch_article(lang, cfg.data_dir) for lang in cfg.languages}
    words = {c: count_words(t) for c, t in corpora.items()}
    payload = build_payload(cfg, corpora, words)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    out = WEB_DIR / "data.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    sizes = " · ".join(f"{c['label']}: score {c['score']}" for c in payload["configs"])
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) — {sizes}")


if __name__ == "__main__":
    main()
