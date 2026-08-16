"""Stage 3 — detecting the language, because the folder lies.

The taught defect: a pipeline trusted its directory layout, a `LANG_3_TO_2` table had a malformed
`tel` key, `.get()` fell through to a default that happened to be right, and a Bengali file sat in
the Assamese folder for an entire training run. Nothing failed. The lookup "worked" by accident.

Our corpus reproduces the shape of that bug in public data: `sangraha/verified/doi/data-0.parquet`
is labelled Dogri and its first row is **plain English**.

**Script detection is not enough here, and that is deliberate.** Ten of the languages in this
corpus — Hindi, Maithili, Bhojpuri, Awadhi, Magahi, Chhattisgarhi, Marathi, Nepali, Sanskrit and
Kashmiri — are all written in *one script*. A script detector scores them all "Devanagari" and
tells you nothing. Six of them are Hindi-belt neighbours that share most of their vocabulary. So
the discriminator is a character n-gram model, and it is **graded on held-out data**: profiles are
trained on FLORES-200 `dev` and accuracy is measured on `devtest`, which the model never sees.

Two refusals are built in, because both were the actual failure in the taught bug:

- **`undecided` is a real answer.** Below `Config.langid_min_chars` we decline rather than guess. A
  detector that always answers cannot be graded, because its confident wrong answers look exactly
  like its confident right ones.
- **The claimed language is never an input.** Feeding the folder's label into the detector would
  reproduce the original defect in a more sophisticated form.
"""

import logging
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from datacleaning import tokens
from datacleaning.config import Config
from datacleaning.records import Document, StageStat

logger = logging.getLogger(__name__)

# Unicode blocks, by first codepoint of each range. Enough to name a script, which is all the first
# pass needs — telling two languages *within* a script apart is the n-gram model's job.
SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x024F, "Latin"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0600, 0x06FF, "Arabic"),
    (0xABC0, 0xABFF, "MeeteiMayek"),
)

# FLORES-200 stems for every language our corpora claim, plus the neighbours that make the
# Devanagari problem hard. `doi` is absent from FLORES, which matters — see `grade`.
FLORES_STEMS: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mai": "mai_Deva",
    "mar": "mar_Deva",
    "ne": "npi_Deva",
    "sa": "san_Deva",
    "bho": "bho_Deva",
    "awa": "awa_Deva",
    "mag": "mag_Deva",
    "hne": "hne_Deva",
    "te": "tel_Telu",
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "mni": "mni_Beng",
}

NGRAM = 3
TOP_NGRAMS = 2_000
"""N-grams kept per language profile.

Two thousand covers the frequent character sequences that separate Hindi-belt neighbours without
turning the profile into a memorised copy of the training sentences.
"""


def detect_script(text: str) -> tuple[str, dict[str, int], float]:
    """Identify the dominant script by counting letters.

    Args:
        text: The text to inspect.

    Returns:
        `(script, counts_per_script, dominant_share)`. Script is `"Unknown"` for text with no
        letters at all.
    """
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= code <= hi:
                counts[name] += 1
                break
        else:
            counts[unicodedata.name(ch, "Unknown").split()[0].title()] += 1

    total = sum(counts.values())
    if not total:
        return "Unknown", {}, 0.0
    script, top = counts.most_common(1)[0]
    return script, dict(counts), top / total


def _ngrams(text: str, n: int = NGRAM) -> Counter:
    """Character n-grams over a space-padded string."""
    padded = f" {' '.join(text.split())} "
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


@dataclass(frozen=True, slots=True)
class Profile:
    """A language's character n-gram fingerprint.

    Attributes:
        lang: Language code.
        script: Dominant script of the training text.
        weights: N-gram -> log probability.
    """

    lang: str
    script: str
    weights: dict[str, float]


def build_profile(lang: str, text: str) -> Profile:
    """Build one language profile from training text.

    Log probabilities rather than raw counts, so scoring is a sum and long documents do not
    overflow. Frequencies are smoothed by the vocabulary size, which keeps an unseen n-gram from
    zeroing an otherwise good match.

    Args:
        lang: Language code.
        text: Training text for this language.

    Returns:
        The profile.
    """
    counts = _ngrams(text)
    kept = dict(counts.most_common(TOP_NGRAMS))
    total = sum(kept.values()) + len(kept)
    script, _, _ = detect_script(text)
    return Profile(
        lang=lang,
        script=script,
        weights={g: math.log((c + 1) / total) for g, c in kept.items()},
    )


@lru_cache(maxsize=2)
def load_profiles(flores_dir: str, split: str = "dev") -> dict[str, Profile]:
    """Train one profile per language from a FLORES-200 split.

    Args:
        flores_dir: Directory holding the split's files.
        split: File extension, `dev` or `devtest`.

    Returns:
        `{lang: Profile}`, empty if FLORES is not on disk.
    """
    folder = Path(flores_dir)
    if not folder.exists():
        logger.info("FLORES-200 %s not found at %s; language ID will be script-only", split, folder)
        return {}

    profiles: dict[str, Profile] = {}
    for lang, stem in FLORES_STEMS.items():
        path = folder / f"{stem}.{split}"
        if path.exists():
            profiles[lang] = build_profile(lang, path.read_text(encoding="utf-8"))
    return profiles


@dataclass(frozen=True, slots=True)
class LangVerdict:
    """What the detector concluded, and how confident it is entitled to be.

    Attributes:
        detected: Language code, or `None` for undecided. `None` is a real answer.
        script: Dominant script.
        script_share: Share of letters in that script.
        confidence: Margin between the best and second-best language, normalised. 0.0 when
            undecided or when only the script is known.
        scores: Language -> score, so a verdict can be checked rather than trusted.
        code_switched: Whether a second script holds a meaningful share of the letters.
        reason: Why, when the answer is `undecided`.
    """

    detected: str | None
    script: str
    script_share: float
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    code_switched: bool = False
    reason: str = ""


def detect(text: str, cfg: Config, profiles: dict[str, Profile] | None = None) -> LangVerdict:
    """Identify the language of one document.

    The claimed language is deliberately not a parameter — feeding the folder's label to the
    detector would reproduce the very bug this stage exists to catch.

    Args:
        text: Cleaned document text.
        cfg: Configuration.
        profiles: Trained profiles. Loaded from FLORES `dev` when omitted.

    Returns:
        The verdict.
    """
    if profiles is None:
        profiles = load_profiles(str(cfg.flores_dir))

    script, counts, share = detect_script(text)
    letters = sum(counts.values())

    second = sorted(counts.values(), reverse=True)[1] if len(counts) > 1 else 0
    code_switched = bool(letters) and (second / letters) >= cfg.code_switch_threshold

    if len(text) < cfg.langid_min_chars or not letters:
        return LangVerdict(
            detected=None,
            script=script,
            script_share=share,
            confidence=0.0,
            code_switched=code_switched,
            reason=f"under {cfg.langid_min_chars} characters — too short to judge",
        )

    candidates = {k: p for k, p in profiles.items() if p.script == script}
    if not candidates:
        return LangVerdict(
            detected=None,
            script=script,
            script_share=share,
            confidence=0.0,
            code_switched=code_switched,
            reason=f"no trained profile for script {script}",
        )

    grams = _ngrams(text)
    total = sum(grams.values()) or 1
    scores = {
        lang: sum(count * profile.weights.get(gram, -20.0) for gram, count in grams.items()) / total
        for lang, profile in candidates.items()
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    if len(ranked) == 1:
        return LangVerdict(best, script, share, 1.0, scores, code_switched)

    runner_up = ranked[1][1]
    margin = abs(best_score - runner_up) / max(abs(best_score), 1e-9)
    return LangVerdict(
        detected=best,
        script=script,
        script_share=share,
        confidence=min(1.0, margin * 10),
        scores={k: round(v, 4) for k, v in scores.items()},
        code_switched=code_switched,
    )


def _grade_at(cfg: Config, train: dict[str, Profile], devtest: Path, sentences: int) -> dict:
    """Grade the detector on documents of a given length. Helper for `grade`."""
    per_lang: dict[str, dict[str, float]] = {}
    correct = seen = 0
    script_only = 0.0
    devanagari = [lang for lang, p in train.items() if p.script == "Devanagari"]

    for lang, stem in FLORES_STEMS.items():
        path = devtest / f"{stem}.devtest"
        if not path.exists() or lang not in train:
            continue
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        docs = [" ".join(lines[i : i + sentences]) for i in range(0, len(lines), sentences)]
        hits = sum(1 for d in docs if detect(d, cfg, train).detected == lang)
        per_lang[lang] = {"documents": len(docs), "accuracy": round(hits / len(docs), 4)}
        correct += hits
        seen += len(docs)
        # A script-only detector must pick uniformly among the languages sharing that script.
        peers = devanagari if train[lang].script == "Devanagari" else [lang]
        script_only += len(docs) / max(len(peers), 1)

    return {
        "sentences_per_document": sentences,
        "languages": per_lang,
        "documents": seen,
        "accuracy": round(correct / seen, 4) if seen else 0.0,
        "script_only_accuracy": round(script_only / seen, 4) if seen else 0.0,
    }


def grade(cfg: Config, doc_sentences: int = 5) -> dict[str, object]:
    """Measure the detector on held-out data, at three document lengths.

    Profiles are trained on FLORES `dev`; accuracy is measured on `devtest`, which training never
    touched.

    **Three lengths, not one, because a single number here would mislead.** At five sentences the
    detector scores perfectly, and quoting only that would claim a solved problem. Telling Hindi
    from Bhojpuri from Awadhi is not solved; five sentences of professionally-translated prose is
    simply a great deal of evidence. Grading at one, two and five sentences shows where the accuracy
    comes from, and the single-sentence figure is the one to trust when reasoning about short web
    documents.

    Two limits this measurement cannot cover, stated because the headline looks better than the
    detector is:

    - FLORES is clean, edited, single-language prose from one domain. Web crawl is none of those,
      so this is an upper bound rather than an estimate of field accuracy.
    - Only languages present in FLORES can be graded *or detected at all*. Bodo (`brx`) is in our
      corpus and absent from FLORES-200 — its documents are **unadjudicable**, not mislabelled.

    Args:
        cfg: Configuration.
        doc_sentences: The headline document length.

    Returns:
        Per-length accuracy, the script-only baseline, and what the grading cannot cover.
    """
    train = load_profiles(str(cfg.flores_dir))
    if not train:
        return {"coverage": "none", "note": "FLORES-200 not on disk; the detector is UNGRADED"}

    devtest = cfg.flores_dir.parent / "devtest"
    if not devtest.exists():
        return {"coverage": "none", "note": "FLORES-200 devtest split not on disk; UNGRADED"}

    by_length = {n: _grade_at(cfg, train, devtest, n) for n in sorted({1, 2, doc_sentences})}
    headline = by_length[doc_sentences]
    devanagari = sorted(lang for lang, p in train.items() if p.script == "Devanagari")

    return {
        "coverage": "held-out",
        "protocol": "trained on FLORES-200 dev, graded on devtest — disjoint sentence sets",
        "headline_sentences": doc_sentences,
        "accuracy": headline["accuracy"],
        "script_only_accuracy": headline["script_only_accuracy"],
        "languages": headline["languages"],
        "documents": headline["documents"],
        "by_document_length": {
            str(n): {"accuracy": g["accuracy"], "documents": g["documents"]}
            for n, g in by_length.items()
        },
        "devanagari_languages": devanagari,
        "profiled_languages": sorted(train),
        "limits": [
            "FLORES is clean, edited, single-language prose from one domain; web crawl is none of "
            "those, so this is an upper bound, not an estimate of field accuracy.",
            "Only languages in FLORES-200 can be detected at all. Bodo (brx) is in our corpus and "
            "absent from FLORES, so its documents are unadjudicable rather than mislabelled.",
            "Accuracy is reported per document length because the five-sentence figure flatters "
            "the detector; one sentence is the honest number for short web text.",
        ],
        "note": (
            f"{len(devanagari)} of these languages share the Devanagari script, so a script "
            "detector cannot separate them at all — its accuracy is the chance baseline, not a "
            "weak result."
        ),
    }


def langid_stage(docs: list[Document], cfg: Config) -> tuple[list[Document], StageStat]:
    """Run stage 3 over a corpus.

    Documents are **not** dropped for disagreeing with their folder. A mismatch is a finding about
    the corpus, and deleting the evidence would be the wrong response to discovering that a source
    mislabels its data. The count is what matters.

    Args:
        docs: Documents entering the stage.
        cfg: Configuration.

    Returns:
        The same documents and the stage record.
    """
    counts = tokens.count_many([d.text for d in docs], cfg)
    profiles = load_profiles(str(cfg.flores_dir))

    detected: Counter = Counter()
    scripts: Counter = Counter()
    mismatches: Counter = Counter()
    unadjudicable: Counter = Counter()
    undecided = code_switched = 0
    examples: list[dict[str, object]] = []

    # A claimed language with no trained profile cannot be confirmed *or* contradicted. Bodo is the
    # case here: it is in the corpus and absent from FLORES-200, so the detector inevitably assigns
    # its documents to the nearest Devanagari neighbour. Counting those as "mismatches" would
    # publish a limitation of our detector as a defect in the corpus — roughly 1,900 fabricated
    # findings in the lite profile alone. They are counted separately and named for what they are.
    adjudicable = set(profiles)

    for doc in docs:
        verdict = detect(doc.text, cfg, profiles)
        scripts[verdict.script] += 1
        if verdict.code_switched:
            code_switched += 1
        if verdict.detected is None:
            undecided += 1
            detected["undecided"] += 1
            continue
        detected[verdict.detected] += 1

        if not doc.claimed_lang or verdict.detected == doc.claimed_lang:
            continue

        if doc.claimed_lang not in adjudicable:
            unadjudicable[f"{doc.claimed_lang} (no FLORES profile)"] += 1
            continue

        key = f"{doc.claimed_lang}->{verdict.detected}"
        mismatches[key] += 1
        if len(examples) < 8 and verdict.confidence > 0.3:
            examples.append(
                {
                    "doc_id": doc.doc_id,
                    "shard": doc.shard,
                    "claimed": doc.claimed_lang,
                    "detected": verdict.detected,
                    "script": verdict.script,
                    "confidence": round(verdict.confidence, 3),
                }
            )

    grading = grade(cfg)
    total_mismatched = sum(mismatches.values())
    total_unadjudicable = sum(unadjudicable.values())

    return docs, StageStat(
        n="3",
        stage_id="langid",
        name="Language ID",
        real=True,
        docs_in=len(docs),
        docs_out=len(docs),
        tokens_in=counts.as_figure(),
        tokens_out=counts.as_figure(),
        detail={
            "detected": dict(detected.most_common()),
            "scripts": dict(scripts.most_common()),
            "mismatches": dict(mismatches.most_common(20)),
            "mismatch_examples": examples,
            "unadjudicable": dict(unadjudicable.most_common()),
            "unadjudicable_total": total_unadjudicable,
            "undecided": undecided,
            "code_switched": code_switched,
            "grading": grading,
        },
        note=(
            f"{total_mismatched:,} documents disagree with the language their folder claims. "
            f"{undecided:,} were left undecided rather than guessed, and {total_unadjudicable:,} "
            "are unadjudicable because their claimed language has no FLORES profile — a limit of "
            "the detector, not a defect in the corpus. Nothing is dropped here: a mislabelled "
            "document is a finding about the source, not a document to delete."
        ),
    )
