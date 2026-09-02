"""Export the data the web widget renders: scores, the vocabulary, and the merge list.

Trains the featured tokenizers and writes ``web/data.json`` — the widget (``web/index.html``) is
a static page that loads this file. Regenerate with::

    uv run python -m tokenization.widget

Two things this module is careful about:

* **The payload carries ordered merges**, not just the vocabulary. A vocabulary alone cannot
  reproduce a score — without the merge order there is no way to encode text, so the widget could
  show a token list but never tokenize anything.
* **Every config is tagged with its profile**, and the page renders one section per profile. v1
  and v2 measure different corpora with different denominators, so a single ranked list across
  them would be meaningless however it were sorted.
"""

import json
from pathlib import Path

from .ablate import (
    OVERTUNED,
    REFERENCE,
    REFERENCE_WEIGHTS,
    SUBMISSION,
    Spec,
    _v1,
    measure,
    train_spec,
)
from .bpe_scratch import UNK_TOKEN, WORD_PREFIX, ScratchBPE
from .config import PROFILES, V1, V2, Config, EvalProfile
from .corpus import load_all
from .metrics import count_denominator, hindi_penalty, mean_ratio, score, spread

# The configs a reviewer can flip between, per profile.
#
# v1 keeps the four the original widget shipped — the byte-level baseline it rejected, the
# char-level BPE that fixed it, the hand-written BPE, and the Unigram that scored highest.
FEATURED_V1: list[Spec] = [
    _v1("unigram", "char", "NFKC", 10_000, "flat", "Unigram · char · NFKC"),
    _v1("bpe-scratch", "char", "NFKC", 10_000, "flat", "BPE from scratch · char · NFKC"),
    _v1("bpe", "char", "NFKC", 10_000, "flat", "BPE · char · NFKC"),
    _v1("bpe", "byte", None, 10_000, "flat", "BPE · byte (baseline)"),
]

# v2 leads with the reference solution exactly as published, then our permutations on it —
# including the one that scores highest and was rejected anyway, because a reader who only sees
# the submitted number has no way to know a bigger one was found and turned down.
FEATURED_V2: list[Spec] = [
    REFERENCE,
    SUBMISSION,
    OVERTUNED,
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
]

FEATURED: list[Spec] = [*FEATURED_V2, *FEATURED_V1]

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

# What each section tells a reader it is. Kept next to the data so the copy cannot drift from it.
SECTION_NOTES = {
    "v1": (
        "Our first pass at the problem, kept exactly as it was run: clipped article prose, scored "
        "in tokens per whitespace word. This is where the finding came from — representation is "
        "the dominant lever. Byte-level BPE rebuilds every Indic character out of three UTF-8 "
        "bytes; moving to character level collapses the gap between languages."
    ),
    "v2": (
        "The measurement the requirements grades: wiki-faithful Markdown — links, URLs, tables and "
        "all — scored in tokens per faithful unit. The reference solution is shown exactly as "
        "published; everything after it is ours."
    ),
}


# One short explanation per experiment: what was changed, why it was worth trying, and what came
# out of it. Every featured config must have one — `tests/test_widget_render.py` fails if a
# tokenizer reaches the page unexplained, because a row of numbers with no story is not a finding.
BLURBS = {
    "the reference solution (benchmark)": (
        "The published reference solution, rebuilt from its own recipe and reproduced to the last "
        "digit. It is the yardstick, not a result of ours — if this row ever stops reading 6,503, "
        "our measuring apparatus is broken and no other number here can be trusted."
    ),
    "submission · documents · mai ×3": (
        "Two changes. Train on whole articles instead of line by line, so a merge may span a line "
        "break; and feed Maithili in three times instead of twice, taking it from 1.1% to 1.6% of "
        "the training mix. Best score of the honest family <em>and</em> its best compression."
    ),
    "more Telugu + Maithili (rejected)": (
        "Pushes Telugu and Maithili harder still and posts by far the biggest number here — by "
        "making English and Hindi <em>worse</em> until all four are equally mediocre. It needs "
        "192,713 tokens for the same corpus against the submission's 189,785: evenness, bought."
    ),
    "BPE from scratch, no library": (
        "The same algorithm written out by hand — the merge loop, the tie-breaking, the encoder — "
        "with no tokenizer library involved. It compresses better than anything else here (188,091 "
        "tokens) yet scores worst, because it throws newlines away and never pays for one."
    ),
    "Unigram (ablation)": (
        "A different algorithm: rather than merging pairs upward, it starts with a large candidate "
        "vocabulary and prunes downward. It scores well, but BPE was asked for, so it stays "
        "an ablation — useful evidence, not a candidate for submission."
    ),
    "Unigram · char · NFKC": (
        "v1's best score. Same pruning algorithm, run on clipped prose against the word "
        "denominator. It beat every BPE variant of that round, which is what made the algorithm "
        "look important — until v2 showed the corpus and the denominator mattered far more."
    ),
    "BPE from scratch · char · NFKC": (
        "Our hand-written BPE under v1's measurement, where it edges out HuggingFace's own "
        "char-level BPE (1,300 against 1,228) on an identical recipe. Part of that margin is the "
        "discarded newlines rather than a better merge loop — worth stating plainly."
    ),
    "BPE · char · NFKC": (
        "The fix for the byte-level baseline: merge whole characters rather than UTF-8 bytes, "
        "after NFKC normalisation. Every Indic language improves sharply and the spread collapses "
        "from 5.27 to 0.81 — this single change is v1's headline finding."
    ),
    "BPE · byte (baseline)": (
        "Where v1 started, and the row that taught us the most by failing. Working on raw bytes "
        "turns each Devanagari or Telugu character into three, so Indic text costs 3.5–6.5 tokens "
        "a word against English's 1.2 — a spread of 5.27, and the worst score on this page."
    ),
}


def extract_merges(tok: object) -> list[list[str]]:
    """Ordered merge pairs for a trained tokenizer; empty for models that have none (Unigram)."""
    if isinstance(tok, ScratchBPE):
        return [list(pair) for pair in tok.merges]
    model = json.loads(tok.to_str())["model"]
    merges = model.get("merges") or []
    return [list(m) if isinstance(m, list | tuple) else list(m.split(" ", 1)) for m in merges]


def encoder_spec(spec: Spec, tok: object) -> dict:
    """Everything the page's JavaScript needs to reproduce this tokenizer's encoding.

    ``kind`` selects the pre-tokenizer the JS must replicate. The kinds differ in exactly one
    place — how raw text is cut into pre-tokens before the merge loop runs — and getting that
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


def build_config(
    spec: Spec,
    profile: EvalProfile,
    corpora: dict[str, str],
    baseline: dict[str, float] | None = None,
) -> dict:
    """Train one featured spec and assemble its entry in the payload.

    Languages are emitted in the profile's own fixed order — **never sorted by fertility**. Sorting
    reorders the rows from one tokenizer to the next, so two tabs showing the identical four
    languages look like they are showing different ones, and a reader cannot compare a row against
    the same row on another tab. Best and worst are flagged instead, which is what the sort was
    really for.

    ``baseline`` is the benchmark's fertility per language, so every other config can show what it
    moved. That is the apples-to-apples comparison the score alone cannot give you.
    """
    names = {lang.code: lang.name for lang in profile.languages}
    counts = {c: count_denominator(t, profile.denominator) for c, t in corpora.items()}
    tok = train_spec(spec, corpora)
    by_code = {s.code: s for s in measure(tok, corpora, counts)}
    scores = [by_code[lang.code] for lang in profile.languages]
    best = min(scores, key=lambda s: s.ratio).code
    worst = max(scores, key=lambda s: s.ratio).code
    penalty = hindi_penalty(scores) if profile.penalty else 1.0
    vocab = tok.get_vocab()
    return {
        "label": spec.label,
        "profile": profile.name,
        "algo": spec.algo,
        "level": spec.level,
        "normalization": spec.normalization,
        "denominator": profile.denominator,
        "weights": dict(spec.weights) or dict.fromkeys(corpora, 1.0),
        "is_submission": spec == SUBMISSION,
        "is_reference": spec == REFERENCE,
        "is_rejected": spec == OVERTUNED,
        "blurb": BLURBS[spec.label],
        "vocab_actual": tok.get_vocab_size(),
        "languages": [
            {
                "code": s.code,
                "name": names.get(s.code, s.code),
                "units": s.units,
                "tokens": s.tokens,
                "ratio": round(s.ratio, 6),
                "is_best": s.code == best,
                "is_worst": s.code == worst,
                # How this language moved against the benchmark. None on the benchmark itself.
                "delta": (None if baseline is None else round(s.ratio - baseline[s.code], 6)),
            }
            for s in scores
        ],
        "x_min": round(min(s.ratio for s in scores), 6),
        "x_max": round(max(s.ratio for s in scores), 6),
        "spread": round(spread(scores), 6),
        "score": round(score(scores), 2),
        "penalty": round(penalty, 6),
        "adjusted": round(score(scores) / penalty, 2),
        "total_tokens": sum(s.tokens for s in scores),
        "mean_ratio": round(mean_ratio(scores), 6),
        "encoder": encoder_spec(spec, tok),
        "tokens": [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
        "merges": extract_merges(tok),
    }


def build_payload(cfg: Config) -> dict:
    """Train every featured config in both profiles and assemble the widget's JSON payload."""
    configs = []
    for profile, specs in ((V2, FEATURED_V2), (V1, FEATURED_V1)):
        corpora = load_all(profile, cfg.corpus_dir)
        # The benchmark is built first so every other row can be shown as a delta against it.
        # Both suites lead with their baseline: v2's is the reference solution, v1's is the
        # byte-level tokenizer everything else was an attempt to improve on.
        baseline_spec = specs[0] if profile is V2 else specs[-1]
        baseline_cfg = build_config(baseline_spec, profile, corpora)
        baseline = {lang["code"]: lang["ratio"] for lang in baseline_cfg["languages"]}
        for spec in specs:
            if spec == baseline_spec:
                configs.append(baseline_cfg)
            else:
                configs.append(build_config(spec, profile, corpora, baseline))
    return {
        "target_vocab": cfg.vocab_size,
        "profiles": [
            {
                "name": p.name,
                "title": p.title,
                "denominator": p.denominator,
                "languages": [lang.code for lang in p.languages],
                "note": SECTION_NOTES[p.name],
            }
            for p in (PROFILES["v2"], PROFILES["v1"])
        ],
        "configs": configs,
    }


def main() -> None:
    """Train the featured configs in both profiles and write the shipped bundle."""
    cfg = Config()
    payload = build_payload(cfg)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    out = WEB_DIR / "data.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # The submission tokenizer in HuggingFace's own format, tracked and served alongside the page,
    # so a grader can `Tokenizer.from_file(...)` it directly rather than reassembling one from the
    # widget's vocab + merges. `artifacts/` is gitignored by design; this is the deliverable.
    submission = train_spec(SUBMISSION, load_all(V2, cfg.corpus_dir))
    submission.save(str(WEB_DIR / "tokenizer.json"))

    print(f"wrote {out} ({out.stat().st_size // 1024} KB) + tokenizer.json")
    for c in payload["configs"]:
        print(f"  [{c['profile']}] {c['label']:38} {c['adjusted']:>10.2f}")


if __name__ == "__main__":
    main()
