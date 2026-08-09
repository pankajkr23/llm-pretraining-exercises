"""Configuration for the cleaning/dedup pipeline.

One `@dataclass` holds the knobs (per the repo convention). Right now it carries only the
directory layout — the thresholds (n-gram size, similarity cutoff, quality filters) get added
once the exercise's brief lands.
"""

from dataclasses import dataclass, field
from pathlib import Path

# exercise root (…/04-data-cleaning-dedup), two levels up from this file's package dir
EXERCISE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Knobs for the cleaning/dedup pipeline.

    Attributes:
        data_dir: Where fetched/raw corpora are cached (git-ignored).
        artifacts_dir: Generated outputs — reports, plots, intermediates (git-ignored).
    """

    data_dir: Path = field(default=EXERCISE_ROOT / "data")
    artifacts_dir: Path = field(default=EXERCISE_ROOT / "artifacts")
