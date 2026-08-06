"""Export the data the web widget renders: scores, the vocabulary, and the merge list.

Trains the featured tokenizers and writes ``web/data.json`` — the widget (``web/index.html``) is
a static page that loads this file. Regenerate with::

    uv run python -m tokenization.widget

The payload carries **ordered merges**, not just the vocabulary. A vocabulary alone cannot
reproduce a score: without the merge order there is no way to encode text, so the widget could
show a token list but never tokenize anything. The merges are what let the page's JavaScript
encoder — and anyone who downloads the JSON — reproduce our token counts exactly.
"""

import json
from pathlib import Path

from .ablate import REFERENCE_WEIGHTS, SUBMISSION, Spec, measure, spec_weights, train_spec
from .bpe_scratch import UNK_TOKEN, WORD_PREFIX, ScratchBPE
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

# Configs the reviewer can flip between in the widget. The submission leads; the rest are the
# comparisons that justify it.
FEATURED: list[Spec] = [
    SUBMISSION,
    Spec(
        algo="bpe-scratch",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label="BPE from scratch, no library",
        weights=REFERENCE_WEIGHTS,
    ),
    Spec(
        algo="unigram",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label="Unigram (ablation)",
        weights=REFERENCE_WEIGHTS,
    ),
    Spec(
        algo="bpe",
        level="byte",
        normalization=None,
        vocab_size=10_000,
        weighting="manual",
        label="byte-level BPE (what we rejected)",
        weights=REFERENCE_WEIGHTS,
    ),
]

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def extract_merges(tok: object) -> list[list[str]]:
    """Ordered merge pairs for a trained tokenizer; empty for models that have none (Unigram)."""
    if isinstance(tok, ScratchBPE):
        return [list(pair) for pair in tok.merges]
    model = json.loads(tok.to_str())["model"]
    merges = model.get("merges") or []
    return [list(m) if isinstance(m, list | tuple) else list(m.split(" ", 1)) for m in merges]


def encoder_spec(spec: Spec, tok: object) -> dict:
    """Everything the page's JavaScript needs to reproduce this tokenizer's encoding.

    ``kind`` selects the pre-tokenizer the JS must replicate. The two BPE kinds differ in exactly
    one place — how raw text is cut into pre-tokens before the merge loop runs — and getting that
    wrong is the classic way a "same" tokenizer in two languages produces different counts.
    """
    if isinstance(tok, ScratchBPE):
        return {
            "kind": "scratch-bpe",
            "normalization": spec.normalization,
            # ``str.split()`` — split on any whitespace run and discard it, so newlines vanish.
            "split": "whitespace",
            "prefix": WORD_PREFIX,
            "unk": UNK_TOKEN,
        }
    if spec.algo == "bpe" and spec.level == "char":
        return {
            "kind": "metaspace-bpe",
            "normalization": spec.normalization,
            # Metaspace replaces U+0020 only; newlines and tabs stay inside a pre-token.
            "split": "space",
            "prefix": WORD_PREFIX,
            "prepend_scheme": spec.prepend_scheme,
            "unk": spec.unk_token,
        }
    return {"kind": "unsupported", "reason": f"{spec.algo}/{spec.level}"}


def build_payload(cfg: Config, corpora: dict[str, str], units: dict[str, int]) -> dict:
    """Train every featured config and assemble the widget's JSON payload."""
    names = {lang.code: lang.name for lang in cfg.languages}
    words = {c: count_words(t) for c, t in corpora.items()}
    configs = []
    for spec in FEATURED:
        tok = train_spec(spec, corpora)
        scores = sorted(measure(tok, corpora, units), key=lambda s: s.ratio)
        vocab = tok.get_vocab()
        configs.append(
            {
                "label": spec.label,
                "algo": spec.algo,
                "level": spec.level,
                "normalization": spec.normalization,
                "weights": spec_weights(spec, corpora),
                "is_submission": spec == SUBMISSION,
                "vocab_actual": tok.get_vocab_size(),
                "languages": [
                    {
                        "code": s.code,
                        "name": names.get(s.code, s.code),
                        "units": s.units,
                        "tokens": s.tokens,
                        "words": words[s.code],
                        "ratio": round(s.ratio, 6),
                    }
                    for s in scores
                ],
                "x_min": round(scores[0].ratio, 6),
                "x_max": round(scores[-1].ratio, 6),
                "spread": round(spread(scores), 6),
                "score": round(score(scores), 2),
                "penalty": round(hindi_penalty(scores), 6),
                "adjusted": round(adjusted_score(scores), 2),
                "encoder": encoder_spec(spec, tok),
                "tokens": [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
                "merges": extract_merges(tok),
            }
        )
    return {"target_vocab": cfg.vocab_size, "configs": configs}


def main() -> None:
    """Load the committed corpus, train the featured configs, and write the shipped bundle."""
    cfg = Config()
    corpora = {lang.code: load_faithful(lang.code, cfg.corpus_dir) for lang in cfg.languages}
    units = {c: count_units(t) for c, t in corpora.items()}
    payload = build_payload(cfg, corpora, units)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    out = WEB_DIR / "data.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # The submission tokenizer in HuggingFace's own format, tracked and served alongside the page,
    # so a grader can `Tokenizer.from_file(...)` it directly rather than reassembling one from the
    # widget's vocab + merges. `artifacts/` is gitignored by design; this is the deliverable.
    submission = train_spec(SUBMISSION, corpora)
    submission.save(str(WEB_DIR / "tokenizer.json"))

    sizes = " · ".join(f"{c['label']}: {c['adjusted']}" for c in payload["configs"])
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) + tokenizer.json — {sizes}")


if __name__ == "__main__":
    main()
