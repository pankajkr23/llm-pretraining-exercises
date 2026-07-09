"""Ablation harness — sweep tokenizer variants and compare their cross-language spread.

Each experiment is a :class:`Spec` (algorithm × representation × normalization × vocab size ×
corpus weighting). Run the curated suite with::

    uv run python -m tokenization.ablate

Results (per-language fertility, spread, score) are printed as a table sorted by score and
written to ``artifacts/ablations.json``. To add an experiment, append a :class:`Spec` to
``SUITE`` — the two findings so far (10k byte BPE saturates → weighting is inert; a scarce
vocab makes weighting matter) are already encoded as the first four rows.
"""

import json
from dataclasses import asdict, dataclass

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

from .bpe_scratch import ScratchBPE
from .config import Config
from .corpus import fetch_article
from .metrics import LangScore, count_words, score, spread


@dataclass(frozen=True)
class Spec:
    """One ablation cell.

    Attributes:
        algo: ``"bpe"`` | ``"unigram"`` | ``"wordpiece"`` | ``"bpe-scratch"`` (our hand-written BPE;
            always char-level + word-boundary, so ``level`` is ignored for it).
        level: ``"byte"`` (UTF-8 bytes) | ``"char"`` (Unicode codepoints).
        normalization: ``None`` | ``"NFC"`` | ``"NFKC"``.
        vocab_size: shared vocabulary budget.
        weighting: ``"flat"`` | ``"balance"`` (equalize corpus chars) | ``"sqrt"`` (milder).
        label: human-readable name for the results table.
    """

    algo: str = "bpe"
    level: str = "byte"
    normalization: str | None = None
    vocab_size: int = 10_000
    weighting: str = "flat"
    label: str = ""


@dataclass
class Result:
    """Outcome of running one :class:`Spec`."""

    label: str
    spec: dict
    vocab_actual: int
    ratios: dict[str, float]
    spread: float
    score: float
    error: str | None = None


def _build(spec: Spec) -> tuple[Tokenizer, object]:
    """Construct an (untrained tokenizer, trainer) pair for ``spec``."""
    if spec.algo == "bpe":
        tok = Tokenizer(BPE(unk_token="<unk>"))
        trainer = BpeTrainer(
            vocab_size=spec.vocab_size, special_tokens=["<unk>"], show_progress=False
        )
    elif spec.algo == "unigram":
        tok = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=spec.vocab_size,
            special_tokens=["<unk>"],
            unk_token="<unk>",
            show_progress=False,
        )
    elif spec.algo == "wordpiece":
        tok = Tokenizer(WordPiece(unk_token="<unk>"))
        trainer = WordPieceTrainer(
            vocab_size=spec.vocab_size, special_tokens=["<unk>"], show_progress=False
        )
    else:
        msg = f"unknown algo {spec.algo!r}"
        raise ValueError(msg)

    if spec.normalization == "NFC":
        tok.normalizer = normalizers.NFC()
    elif spec.normalization == "NFKC":
        tok.normalizer = normalizers.NFKC()

    if spec.level == "byte":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
    elif spec.level == "char":
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
    else:
        msg = f"unknown level {spec.level!r}"
        raise ValueError(msg)
    return tok, trainer


def compute_weights(corpora: dict[str, str], strategy: str) -> dict[str, float]:
    """Corpus upsampling weights (min 1.0) for a weighting ``strategy``."""
    if strategy == "flat":
        return dict.fromkeys(corpora, 1.0)
    sizes = {c: max(1, len(t)) for c, t in corpora.items()}
    biggest = max(sizes.values())
    if strategy == "balance":
        raw = {c: biggest / s for c, s in sizes.items()}
    elif strategy == "sqrt":
        raw = {c: (biggest / s) ** 0.5 for c, s in sizes.items()}
    else:
        msg = f"unknown weighting {strategy!r}"
        raise ValueError(msg)
    low = min(raw.values())
    return {c: v / low for c, v in raw.items()}


def _weighted_lines(corpora: dict[str, str], weights: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for code, text in corpora.items():
        lines.extend([text] * max(1, round(weights.get(code, 1.0))))
    return lines


def train_spec(spec: Spec, corpora: dict[str, str]) -> Tokenizer | ScratchBPE:
    """Build and train a tokenizer for ``spec`` over the weighted corpora.

    The hand-written :class:`~tokenization.bpe_scratch.ScratchBPE` duck-types the slice of the
    HuggingFace API used downstream (``encode().ids``, ``get_vocab``, ``get_vocab_size``), so
    callers stay identical regardless of which engine trained the tokenizer.
    """
    weights = compute_weights(corpora, spec.weighting)
    if spec.algo == "bpe-scratch":
        tok = ScratchBPE(normalization=spec.normalization)
        tok.train(corpora, spec.vocab_size, weights)
        return tok
    hf_tok, trainer = _build(spec)
    hf_tok.train_from_iterator(_weighted_lines(corpora, weights), trainer)
    return hf_tok


def run(spec: Spec, corpora: dict[str, str], words: dict[str, int]) -> Result:
    """Train one tokenizer per ``spec`` and measure per-language fertility, spread, and score."""
    try:
        tok = train_spec(spec, corpora)
        scores = [LangScore(c, words[c], len(tok.encode(t).ids)) for c, t in corpora.items()]
        return Result(
            label=spec.label or f"{spec.algo}/{spec.level}",
            spec=asdict(spec),
            vocab_actual=tok.get_vocab_size(),
            ratios={s.code: round(s.ratio, 3) for s in scores},
            spread=round(spread(scores), 3),
            score=round(score(scores), 2),
        )
    except Exception as exc:  # noqa: BLE001 — a bad spec shouldn't abort the whole sweep
        return Result(
            spec.label, asdict(spec), 0, {}, 0.0, 0.0, error=f"{type(exc).__name__}: {exc}"
        )


# The curated sweep. Rows 1-4 encode the findings so far; the rest probe the alternatives.
SUITE: list[Spec] = [
    Spec("bpe", "byte", None, 10_000, "flat", "byte BPE · 10k · flat  (baseline)"),
    Spec("bpe", "byte", None, 10_000, "balance", "byte BPE · 10k · balance  (saturates → inert)"),
    Spec("bpe", "byte", None, 2_000, "flat", "byte BPE · 2k · flat  (scarce)"),
    Spec("bpe", "byte", None, 2_000, "balance", "byte BPE · 2k · balance  (weighting bites)"),
    Spec("bpe", "char", "NFC", 10_000, "flat", "char BPE · 10k · NFC · flat"),
    Spec("bpe", "char", "NFC", 10_000, "balance", "char BPE · 10k · NFC · balance"),
    Spec("bpe", "char", "NFKC", 10_000, "flat", "char BPE · 10k · NFKC · flat"),
    Spec("unigram", "char", "NFKC", 10_000, "flat", "Unigram char · 10k · NFKC · flat"),
    Spec("unigram", "char", "NFKC", 10_000, "balance", "Unigram char · 10k · NFKC · balance"),
    Spec("unigram", "byte", None, 10_000, "flat", "Unigram byte · 10k · flat"),
    # Our from-scratch BPE (no HuggingFace), pinned to the winning char + NFKC recipe.
    Spec(
        "bpe-scratch", "char", "NFKC", 10_000, "flat", "BPE from scratch · char · 10k · NFKC · flat"
    ),
    Spec(
        "bpe-scratch",
        "char",
        "NFKC",
        10_000,
        "balance",
        "BPE from scratch · char · 10k · NFKC · balance",
    ),
]


def sweep(specs: list[Spec], corpora: dict[str, str], words: dict[str, int]) -> list[Result]:
    """Run every spec and return results sorted by score (best first, failures last)."""
    results = [run(s, corpora, words) for s in specs]
    return sorted(results, key=lambda r: (r.error is not None, -r.score))


def _print_table(results: list[Result]) -> None:
    print(f"{'experiment':38} {'vocab':>6} {'spread':>7} {'score':>8}   ratios")
    print("-" * 96)
    for r in results:
        if r.error:
            print(f"{r.label:38} {'—':>6} {'—':>7} {'FAILED':>8}   {r.error}")
            continue
        ratios = " ".join(f"{k}:{v:.2f}" for k, v in r.ratios.items())
        print(f"{r.label:38} {r.vocab_actual:>6} {r.spread:>7.2f} {r.score:>8.1f}   {ratios}")


def main() -> None:
    """Run the curated SUITE over the cached corpora, print a table, and dump JSON."""
    cfg = Config()
    corpora = {lang.code: fetch_article(lang, cfg.data_dir) for lang in cfg.languages}
    words = {c: count_words(t) for c, t in corpora.items()}
    results = sweep(SUITE, corpora, words)
    _print_table(results)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "ablations.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
