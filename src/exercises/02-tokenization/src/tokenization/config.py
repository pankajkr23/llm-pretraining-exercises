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
        weight: Corpus upsampling weight — raise it to give a language more merges.
    """

    code: str
    name: str
    title: str
    weight: float = 1.0


@dataclass
class Config:
    """Knobs for building and scoring the tokenizer."""

    vocab_size: int = 10_000
    target_ratio: float = 1.2
    languages: tuple[Language, ...] = (
        Language("en", "English", "India"),
        Language("hi", "Hindi", "भारत"),
        Language("te", "Telugu", "భారత దేశం"),
        Language("ta", "Tamil", "இந்தியா"),  # your 4th language — change freely
    )
    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
