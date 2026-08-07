"""Ablation harness — sweep tokenizer variants and compare their cross-language spread.

Each experiment is a :class:`Spec` (algorithm × representation × normalization × vocab size ×
corpus weighting). Run the curated suite with::

    uv run python -m tokenization.ablate

Every spec trains and is scored on the same committed wiki-faithful corpus, so the only thing
varying across rows is the recipe. Results (per-language fertility, spread, raw score, Hindi
penalty, adjusted score) are printed as a table sorted by adjusted score and written to
``artifacts/ablations.json``.

``SUITE`` opens with :data:`REFERENCE` — the published reference recipe, reproduced exactly. It
is the correctness gate, not a result: if that row does not print 6502.56 the harness is wrong
and no other row means anything. Everything after it is ours. To add an experiment, append a
:class:`Spec`.
"""

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

from .bpe_scratch import ScratchBPE
from .config import PROFILES, V1, V2, Config, EvalProfile
from .corpus import load_all
from .metrics import (
    LangScore,
    count_denominator,
    hindi_penalty,
    mean_ratio,
    score,
    spread,
)


@dataclass(frozen=True)
class Spec:
    """One ablation cell.

    Attributes:
        algo: ``"bpe"`` | ``"unigram"`` | ``"wordpiece"`` | ``"bpe-scratch"`` (our hand-written BPE;
            always char-level + word-boundary, so ``level`` is ignored for it).
        level: ``"byte"`` (UTF-8 bytes) | ``"char"`` (Unicode codepoints, via Metaspace).
        normalization: ``None`` | ``"NFC"`` | ``"NFKC"``.
        vocab_size: shared vocabulary budget.
        weighting: ``"flat"`` | ``"manual"`` (use ``weights``) | ``"balance"`` (equalize corpus
            chars) | ``"sqrt"`` (milder).
        label: human-readable name for the results table.
        profile: which measurement this row belongs to — ``"v1"`` or ``"v2"``. Selects the corpus,
            the denominator and whether the Hindi penalty applies. Rows from different profiles
            are never ranked against each other.
        weights: language code -> upsampling weight, as ordered pairs; used by ``"manual"``.
        min_frequency: minimum pair frequency a merge must reach (HuggingFace trainers only).
        unk_token: spelling of the unknown-token symbol.
        prepend_scheme: Metaspace ``prepend_scheme`` — ``"never"`` | ``"always"`` | ``"first"``.
        train_unit: ``"lines"`` (train from files, so no merge may span a newline — the reference
            recipe) or ``"documents"`` (train from whole texts, allowing cross-line merges).
    """

    algo: str = "bpe"
    level: str = "char"
    normalization: str | None = "NFKC"
    vocab_size: int = 10_000
    weighting: str = "flat"
    label: str = ""
    weights: tuple[tuple[str, float], ...] = ()
    min_frequency: int = 1
    unk_token: str = "[UNK]"
    prepend_scheme: str = "never"
    train_unit: str = "lines"
    profile: str = "v2"


@dataclass
class Result:
    """Outcome of running one :class:`Spec`."""

    label: str
    profile: str
    spec: dict
    vocab_actual: int
    ratios: dict[str, float]
    tokens: dict[str, int] = field(default_factory=dict)
    units: dict[str, int] = field(default_factory=dict)
    spread: float = 0.0
    score: float = 0.0
    penalty: float = 1.0
    adjusted: float = 0.0
    total_tokens: int = 0
    mean_ratio: float = 0.0
    error: str | None = None


def _build(spec: Spec) -> tuple[Tokenizer, object]:
    """Construct an (untrained tokenizer, trainer) pair for ``spec``."""
    if spec.algo == "bpe":
        tok = Tokenizer(BPE(unk_token=spec.unk_token))
        trainer = BpeTrainer(
            vocab_size=spec.vocab_size,
            min_frequency=spec.min_frequency,
            special_tokens=[spec.unk_token],
            show_progress=False,
        )
    elif spec.algo == "unigram":
        tok = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=spec.vocab_size,
            special_tokens=[spec.unk_token],
            unk_token=spec.unk_token,
            show_progress=False,
        )
    elif spec.algo == "wordpiece":
        tok = Tokenizer(WordPiece(unk_token=spec.unk_token))
        trainer = WordPieceTrainer(
            vocab_size=spec.vocab_size,
            min_frequency=spec.min_frequency,
            special_tokens=[spec.unk_token],
            show_progress=False,
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
        # Metaspace keeps every visible character — including punctuation, brackets and URL
        # machinery — which the faithfulness rule requires and ByteLevel pays for in Indic bytes.
        tok.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=spec.prepend_scheme
        )
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=spec.prepend_scheme)
    else:
        msg = f"unknown level {spec.level!r}"
        raise ValueError(msg)
    return tok, trainer


def compute_weights(
    corpora: dict[str, str], strategy: str, manual: dict[str, float] | None = None
) -> dict[str, float]:
    """Corpus upsampling weights (min 1.0) for a weighting ``strategy``.

    Args:
        corpora: language code -> raw text.
        strategy: ``"flat"``, ``"manual"``, ``"balance"`` or ``"sqrt"``.
        manual: explicit per-language weights, required by the ``"manual"`` strategy.
    """
    if strategy == "flat":
        return dict.fromkeys(corpora, 1.0)
    if strategy == "manual":
        if not manual:
            msg = "weighting='manual' needs explicit weights"
            raise ValueError(msg)
        return {c: float(manual.get(c, 1.0)) for c in corpora}
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


def _repeat(corpora: dict[str, str], weights: dict[str, float]) -> list[str]:
    """Language codes repeated ``round(weight)`` times — the recipe's file-repetition upsampling."""
    order: list[str] = []
    for code in corpora:
        order.extend([code] * max(1, round(weights.get(code, 1.0))))
    return order


def _train_hf(
    tok: Tokenizer,
    trainer: object,
    corpora: dict[str, str],
    weights: dict[str, float],
    train_unit: str,
) -> None:
    """Train a HuggingFace tokenizer, upsampling by repetition.

    ``train_unit`` is load-bearing, not a detail. HuggingFace splits *files* into lines, so
    training from files means no merge may span a newline. Handing the trainer whole *documents*
    instead lets it learn cross-line pairs — worth ~0.6% of every token count on this corpus.
    The reference recipe trains from files; ``"documents"`` is our variant, measured separately.
    """
    if train_unit == "documents":
        tok.train_from_iterator([corpora[c] for c in _repeat(corpora, weights)], trainer)
        return
    if train_unit != "lines":
        msg = f"unknown train_unit {train_unit!r}"
        raise ValueError(msg)
    with tempfile.TemporaryDirectory() as tmp:
        paths: dict[str, str] = {}
        for code, text in corpora.items():
            path = Path(tmp) / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            paths[code] = str(path)
        tok.train([paths[c] for c in _repeat(corpora, weights)], trainer)


def spec_weights(spec: Spec, corpora: dict[str, str]) -> dict[str, float]:
    """Resolve ``spec``'s weighting strategy against ``corpora``."""
    return compute_weights(corpora, spec.weighting, dict(spec.weights))


def train_spec(spec: Spec, corpora: dict[str, str]) -> Tokenizer | ScratchBPE:
    """Build and train a tokenizer for ``spec`` over the weighted corpora.

    The hand-written :class:`~tokenization.bpe_scratch.ScratchBPE` duck-types the slice of the
    HuggingFace API used downstream (``encode().ids``, ``get_vocab``, ``get_vocab_size``), so
    callers stay identical regardless of which engine trained the tokenizer.
    """
    weights = spec_weights(spec, corpora)
    if spec.algo == "bpe-scratch":
        tok = ScratchBPE(normalization=spec.normalization)
        tok.train(corpora, spec.vocab_size, weights)
        return tok
    hf_tok, trainer = _build(spec)
    _train_hf(hf_tok, trainer, corpora, weights, spec.train_unit)
    return hf_tok


def measure(
    tok: Tokenizer | ScratchBPE, corpora: dict[str, str], units: dict[str, int]
) -> list[LangScore]:
    """Per-language :class:`~tokenization.metrics.LangScore` for a trained tokenizer."""
    return [LangScore(c, units[c], len(tok.encode(t).ids)) for c, t in corpora.items()]


def run(spec: Spec, corpora: dict[str, str], units: dict[str, int]) -> Result:
    """Train one tokenizer per ``spec`` and measure fertility, spread, score and penalty.

    The Hindi penalty is applied only when the spec's profile defines one. v1 was designed and
    reported without it, and retro-fitting it would silently restate numbers that were published
    without it — so v1 rows carry ``penalty = 1.0`` and ``adjusted == score`` by construction.
    """
    profile = PROFILES[spec.profile]
    try:
        tok = train_spec(spec, corpora)
        scores = measure(tok, corpora, units)
        penalty = hindi_penalty(scores) if profile.penalty else 1.0
        return Result(
            label=spec.label or f"{spec.algo}/{spec.level}",
            profile=spec.profile,
            spec=asdict(spec),
            vocab_actual=tok.get_vocab_size(),
            ratios={s.code: round(s.ratio, 6) for s in scores},
            tokens={s.code: s.tokens for s in scores},
            units={s.code: s.units for s in scores},
            spread=round(spread(scores), 6),
            score=round(score(scores), 2),
            penalty=round(penalty, 6),
            adjusted=round(score(scores) / penalty, 2),
            total_tokens=sum(s.tokens for s in scores),
            mean_ratio=round(mean_ratio(scores), 6),
        )
    except Exception as exc:  # noqa: BLE001 — a bad spec shouldn't abort the whole sweep
        return Result(
            spec.label, spec.profile, asdict(spec), 0, {}, error=f"{type(exc).__name__}: {exc}"
        )


# The published reference recipe, reproduced exactly: HF BPE · 10k · min_frequency=1 · NFKC only ·
# Metaspace("▁", prepend_scheme="never") for both pre-tokenizer and decoder · unk "[UNK]" ·
# weights en3/hi4/te4/mai2 applied by repetition. This row is the correctness gate.
REFERENCE_WEIGHTS: tuple[tuple[str, float], ...] = (("en", 3), ("hi", 4), ("te", 4), ("mai", 2))

REFERENCE = Spec(
    algo="bpe",
    level="char",
    normalization="NFKC",
    vocab_size=10_000,
    weighting="manual",
    label="reference recipe (gate)",
    weights=REFERENCE_WEIGHTS,
)


def _reweighted(label: str, **weights: float) -> Spec:
    """The reference recipe with different corpus weights — our one deliberate variable."""
    return Spec(
        algo="bpe",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label=label,
        weights=tuple(weights.items()),
    )


def _documents(label: str, **weights: float) -> Spec:
    """The reference recipe trained on whole documents, so merges may span a newline."""
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


def _v1(algo: str, level: str, norm: str | None, vocab: int, weighting: str, label: str) -> Spec:
    """A v1 experiment, pinned to the settings its published numbers were produced with.

    These are not stylistic defaults — they are what the original code did, and each one moves the
    result: it trained from an in-memory iterator of whole **documents**, spelled the unknown token
    ``<unk>``, left ``min_frequency`` at the trainer's own default of 0, and used HuggingFace's
    default Metaspace, whose ``prepend_scheme`` is ``"always"`` rather than ``"never"``. Inheriting
    v2's defaults here would quietly restate v1's history.
    """
    return Spec(
        algo=algo,
        level=level,
        normalization=norm,
        vocab_size=vocab,
        weighting=weighting,
        label=label,
        min_frequency=0,
        unk_token="<unk>",
        prepend_scheme="always",
        train_unit="documents",
        profile="v1",
    )


# v1 — the original experiments, retained and still runnable. Clipped prose, whitespace words, no
# Hindi penalty. Their finding stands on its own: **representation is the dominant lever**, not
# corpus weighting. Byte-level BPE spends its budget rebuilding 3-byte UTF-8 Indic characters;
# char-level + NFKC collapses the spread. Weighting only bites while the vocabulary is scarce
# (compare the 2k rows to the 10k ones) and can over-correct at char level.
V1_SUITE: list[Spec] = [
    _v1("bpe", "byte", None, 10_000, "flat", "byte BPE · 10k · flat  (baseline)"),
    _v1("bpe", "byte", None, 10_000, "balance", "byte BPE · 10k · balance  (saturates → inert)"),
    _v1("bpe", "byte", None, 2_000, "flat", "byte BPE · 2k · flat  (scarce)"),
    _v1("bpe", "byte", None, 2_000, "balance", "byte BPE · 2k · balance  (weighting bites)"),
    _v1("bpe", "char", "NFC", 10_000, "flat", "char BPE · 10k · NFC · flat"),
    _v1("bpe", "char", "NFC", 10_000, "balance", "char BPE · 10k · NFC · balance"),
    _v1("bpe", "char", "NFKC", 10_000, "flat", "char BPE · 10k · NFKC · flat"),
    _v1("unigram", "char", "NFKC", 10_000, "flat", "Unigram char · 10k · NFKC · flat"),
    _v1("unigram", "char", "NFKC", 10_000, "balance", "Unigram char · 10k · NFKC · balance"),
    _v1("unigram", "byte", None, 10_000, "flat", "Unigram byte · 10k · flat"),
    _v1("bpe-scratch", "char", "NFKC", 10_000, "flat", "BPE from scratch · char · NFKC · flat"),
    _v1("bpe-scratch", "char", "NFKC", 10_000, "balance", "BPE from scratch · char · balance"),
]

# v2 — the graded measurement: the reference recipe as the gate, then our permutations on its
# corpus. Maithili sits at the *maximum* fertility and is only ~1.8% of the corpus, so it wins
# almost no merges of its own and rides Hindi's Devanagari. Spread is max − min, so the honest
# lever is pulling that maximum down; pushing the minimum (Hindi) up would also shrink the spread
# but is exactly the exploit the penalty exists to block — see ``metrics.degrades_best``.
V2_SUITE: list[Spec] = [
    REFERENCE,
    # E0 — the same recipe trained on whole documents instead of lines. Markdown repeats a lot of
    # cross-line structure (list scaffolding, table rows, reference blocks) that a line-split
    # trainer can never merge; letting merges span newlines is a genuine compression win, not a
    # denominator trick, and it lowers all four fertilities rather than trading one against another.
    _documents("E0 · train on documents, not lines", en=3, hi=4, te=4, mai=2),
    # E1 — Maithili sits at the maximum fertility and is ~1.1% of the weighted mix, so it wins
    # almost no merges of its own. Pulling the maximum down is the honest way to shrink a spread.
    # The sweep overshoots on purpose: past ×6 Maithili becomes the new *minimum* and the spread
    # widens again from the other end, which is the whole shape of the lever in three rows.
    _reweighted("E1a · mai ×6", en=3, hi=4, te=4, mai=6),
    _reweighted("E1b · mai ×10", en=3, hi=4, te=4, mai=10),
    _reweighted("E1c · mai ×16", en=3, hi=4, te=4, mai=16),
    # E2 — with Maithili fixed, Telugu becomes the ceiling. Lift both, mildly. These score far
    # higher in sample and are the trap this suite is built to expose: see ``holdout``.
    _reweighted("E2a · te ×5 · mai ×6", en=3, hi=4, te=5, mai=6),
    _reweighted("E2b · te ×6 · mai ×7", en=3, hi=4, te=6, mai=7),
    # E5 — the two independent wins composed: documents (E0) and Maithili ×6 (E1a). This is the
    # submission. It is the only configuration that beats the reference on *both* axes at once,
    # in sample and out of it: smaller spread and fewer total tokens.
    _documents("E5 · documents · mai ×6  (submission)", en=3, hi=4, te=4, mai=6),
    _documents("E5b · documents · mai ×5", en=3, hi=4, te=4, mai=5),
    # E3/E4 — algorithm ablations. The brief asks for BPE, so neither is the submission.
    Spec(
        algo="unigram",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label="E3 · Unigram (ablation)",
        weights=REFERENCE_WEIGHTS,
    ),
    Spec(
        algo="bpe-scratch",
        level="char",
        normalization="NFKC",
        vocab_size=10_000,
        weighting="manual",
        label="E4 · BPE from scratch, no library",
        weights=REFERENCE_WEIGHTS,
    ),
]


# The recipe we submit: two independent changes to the reference, each justified on its own.
#
#   documents  — train on whole articles rather than lines, so a merge may span a newline. Pure
#                compression: fewer tokens for the same text, no denominator involved.
#   mai ×6     — Maithili is 1.1% of the weighted mix and sat at the worst fertility, so it won
#                almost no merges of its own. Raising its weight pulls the *maximum* down, which
#                is the honest direction to shrink a spread.
#
# It is not the highest in-sample scorer — E2b reaches 35603 against this one's 10934. That gap
# is overfitting, and ``tokenization.holdout`` measures it: on text the trainer never saw, E2b's
# 3.3× in-sample lead is worth nothing (4103 vs 4213 adjusted, i.e. it is behind) and it
# compresses worse. This configuration is the one that beats the reference on every axis both in
# sample and out of it.
SUBMISSION = _documents("submission · documents · mai ×6", en=3, hi=4, te=4, mai=6)


def sweep(specs: list[Spec], corpora: dict[str, str], units: dict[str, int]) -> list[Result]:
    """Run every spec and return results sorted by adjusted score (best first, failures last).

    Only ever call this with specs from a **single** profile. Sorting rows measured in different
    denominators would produce a ranked list whose order means nothing.
    """
    profiles = {s.profile for s in specs}
    if len(profiles) > 1:
        msg = f"cannot rank across profiles {sorted(profiles)} — they are different measurements"
        raise ValueError(msg)
    results = [run(s, corpora, units) for s in specs]
    return sorted(results, key=lambda r: (r.error is not None, -r.adjusted))


def run_profile(profile: EvalProfile, specs: list[Spec], cfg: Config) -> list[Result]:
    """Load a profile's corpus and sweep its specs against the denominator it is scored in."""
    corpora = load_all(profile, cfg.corpus_dir)
    counts = {c: count_denominator(t, profile.denominator) for c, t in corpora.items()}
    return sweep(specs, corpora, counts)


def _print_table(profile: EvalProfile, results: list[Result]) -> None:
    denom = profile.denominator
    print(f"\n{profile.title}")
    print(f"{'experiment':42} {'spread':>8} {'score':>10} {'tokens':>9} {'mean X':>7}  ratios")
    print("-" * 122)
    for r in results:
        if r.error:
            print(f"{r.label:42} {'—':>8} {'FAILED':>10} {'—':>9} {'—':>7}  {r.error}")
            continue
        ratios = " ".join(f"{k}:{v:.3f}" for k, v in r.ratios.items())
        print(
            f"{r.label:42} {r.spread:>8.4f} {r.adjusted:>10.2f} {r.total_tokens:>9} "
            f"{r.mean_ratio:>7.4f}  {ratios}"
        )
    print(f"({len(results)} rows · X = tokens per {denom})")


def main() -> None:
    """Run both profiles' suites and dump each to its own JSON, never into one ranked list."""
    cfg = Config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    everything: dict[str, list[dict]] = {}
    for profile, specs in ((V1, V1_SUITE), (V2, V2_SUITE)):
        results = run_profile(profile, specs, cfg)
        _print_table(profile, results)
        everything[profile.name] = [asdict(r) for r in results]
    print(
        "\nScores are not comparable across the two tables: different corpus, different "
        "denominator. The same tokenizer reads ~2.1 under v1 and ~0.6 under v2."
    )
    (cfg.artifacts_dir / "ablations.json").write_text(
        json.dumps(everything, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
