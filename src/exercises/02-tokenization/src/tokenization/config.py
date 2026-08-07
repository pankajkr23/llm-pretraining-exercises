"""Configuration for the multilingual BPE tokenizer exercise.

Two evaluation profiles live here, and they are **not** two versions of one thing — they are two
different measurements, retained side by side:

* **v1** — our original experiments. Clipped article prose from Wikipedia's ``explaintext`` API,
  scored in tokens per whitespace **word**. This is where the representation findings came from
  (byte-level vs char-level, NFKC, Unigram, the hand-written BPE), and those findings still hold.
* **v2** — the reference solution's measurement. Wiki-faithful Markdown, scored in tokens per
  **faithful unit**, with the Hindi penalty. This is what the assignment grades.

A score from one profile can never be ranked against a score from the other: different corpus,
different denominator. The same tokenizer reads ≈ 2.13 under v1 and ≈ 0.60 under v2. Everything
downstream — the ablation table, the report, the widget — keeps them in separate labelled
sections for exactly that reason.
"""

from dataclasses import dataclass, field
from pathlib import Path

# exercise root (…/02-tokenization), two levels up from this file's package dir
EXERCISE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Language:
    """A Wikipedia language edition to include in the shared vocabulary.

    Attributes:
        code: Wikipedia subdomain, e.g. ``"en"``.
        name: Human-readable name, e.g. ``"English"``.
        title: Article title in that language, e.g. ``"India"``.
        weight: Corpus upsampling weight — the language's text is repeated this many times during
            training, so a higher weight wins more of the shared merges.
    """

    code: str
    name: str
    title: str
    weight: float = 1.0


# v1's four languages. Flat weights: the original experiments varied weighting by *strategy*
# (flat/balance/sqrt) rather than per language.
V1_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "India"),
    Language("hi", "Hindi", "भारत"),
    Language("te", "Telugu", "భారత దేశం"),
    Language("ta", "Tamil", "இந்தியா"),
)

# The reference recipe's four languages and its 3/4/4/2 weights. Titles are the exact article
# names behind the committed v2 snapshots — its Telugu is ``భారతదేశం`` (no space), unlike v1's.
REFERENCE_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "India", weight=3),
    Language("hi", "Hindi", "भारत", weight=4),
    Language("te", "Telugu", "భారతదేశం", weight=4),
    Language("mai", "Maithili", "भारत", weight=2),
)

# Our own fourth-language choice, fetched with the same wiki-faithful pipeline.
TAMIL = Language("ta", "Tamil", "இந்தியா", weight=2)


@dataclass(frozen=True)
class EvalProfile:
    """One complete way of measuring: which corpus, which denominator, which score.

    Attributes:
        name: ``"v1"`` or ``"v2"``.
        title: how the profile is labelled for a reader.
        subdir: directory under ``corpus/`` holding this profile's snapshots.
        suffix: filename suffix for a language's snapshot.
        denominator: ``"words"`` (whitespace split) or ``"units"`` (faithful units).
        penalty: whether the Hindi penalty applies to the score.
        languages: the language set this profile is measured over.
    """

    name: str
    title: str
    subdir: str
    suffix: str
    denominator: str
    penalty: bool
    languages: tuple[Language, ...]


V1 = EvalProfile(
    name="v1",
    title="v1 — our original experiments (clipped prose · words)",
    subdir="v1",
    suffix=".txt",
    denominator="words",
    penalty=False,
    languages=V1_LANGUAGES,
)

V2 = EvalProfile(
    name="v2",
    title="v2 — the graded measurement (wiki-faithful Markdown · faithful units)",
    subdir="v2",
    suffix=".faithful.txt",
    denominator="units",
    penalty=True,
    languages=REFERENCE_LANGUAGES,
)

PROFILES: dict[str, EvalProfile] = {V1.name: V1, V2.name: V2}


def all_languages() -> tuple[Language, ...]:
    """Every language we have a v2 snapshot for — the four reference ones plus Tamil."""
    return (*REFERENCE_LANGUAGES, TAMIL)


@dataclass
class Config:
    """Paths and the shared vocabulary budget."""

    vocab_size: int = 10_000
    corpus_dir: Path = field(default=EXERCISE_ROOT / "corpus")
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
