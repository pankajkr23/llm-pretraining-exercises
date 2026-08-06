"""Configuration for the multilingual BPE tokenizer exercise."""

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


# The reference recipe's four languages and its 3/4/4/2 weights. Titles are the exact article
# names behind the committed snapshots in ``corpus/`` — Telugu is ``భారతదేశం`` (no space).
REFERENCE_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "India", weight=3),
    Language("hi", "Hindi", "भारत", weight=4),
    Language("te", "Telugu", "భారతదేశం", weight=4),
    Language("mai", "Maithili", "भारत", weight=2),
)

# Our own fourth-language choice, fetched with the same wiki-faithful pipeline.
TAMIL = Language("ta", "Tamil", "இந்தியா", weight=2)


def all_languages() -> tuple[Language, ...]:
    """Every language we have a snapshot for — the four reference ones plus Tamil."""
    return (*REFERENCE_LANGUAGES, TAMIL)


@dataclass
class Config:
    """Knobs for building and scoring the tokenizer."""

    vocab_size: int = 10_000
    languages: tuple[Language, ...] = REFERENCE_LANGUAGES
    corpus_dir: Path = field(default=EXERCISE_ROOT / "corpus")
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")

    def weights(self) -> dict[str, float]:
        """Language code -> upsampling weight, as configured."""
        return {lang.code: lang.weight for lang in self.languages}
